"""ハザード分析に足りない L1 の付随情報だけを追加で落とす。

`fetch_l1.py` は板の再構成に要る 10 列しか取っていない。ハザードを**口座別**に
見るには発注元(`user`)が要り、**欠測時間帯**を右打ち切りとして正しく扱うには
パイプラインが残したサイドカー(`_stats.json` の `missing_hours`)が要る。
この 2 つだけを追加で取る。

★費用: `user` は辞書化が効いて 1 行あたり約 0.65 バイト、`status` は約 0.17
バイト。99 日で約 0.9GB = GLACIER_IR 取り出し $0.03/GB + egress $0.09/GB で
**$0.11 程度**の見込み。`_stats.json` は 1 日 200 バイト前後で無視できる。

【位置合わせの検算】
`oid` は 1 行 2.9 バイトあり、取り直すと $0.35 増える。手元の l1 と同じ
オブジェクトを同じ読み方で読むので行順は一致するはずだが、それを仮定せずに
`status` 列(1 行 0.17 バイト)も一緒に取り、手元の l1 の `status` 列と
**全行一致するか**を検算してから `oid` を手元側から借りる。一致しなければ
その日は捨てる。

口座は文字列のまま持つとメモリが足りない(1 日 2,400 万行 × 42 バイト)ので、
辞書の添字として読み、通し番号 `wid` に付け替えて保存する。

x が確定する時刻 / y の期間: 該当なし(データ取得)。

    uv run python scripts/fetch_l1_extra.py --coin xyz:MU
出力: data/l1user_<coin>/dt=*.parquet  (oid, wid) … open 行だけ
      data/l1user_<coin>/_wallets.parquet  (wid, user)
      data/l1_meta_<coin>.csv  (dt, rows, missing_hours)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
import polars as pl
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")
    tag = a.coin.replace(":", "_")
    localdir = ROOT / "data" / f"l1_{tag}"
    outdir = ROOT / "data" / f"l1user_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    sel = (pl.col("coin") == a.coin) & (pl.col("layer") == "l1")
    part = dict(inv.filter(sel & (pl.col("leaf") == "part-000.parquet"))
                .select("dt", "key").rows())
    stat = dict(inv.filter(sel & (pl.col("leaf") == "_stats.json"))
                .select("dt", "key").rows())
    days = sorted(d for d in part if (localdir / f"dt={d}.parquet").exists())
    if a.days:
        days = days[: a.days]
    print(f"[fetch] 手元に l1 のある {len(days)} 日", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(access_key=cr.access_key, secret_key=cr.secret_key,
                           session_token=cr.token, region="us-east-1")
    s3 = sess.client("s3")

    wmap: dict[str, int] = {}
    wpath = outdir / "_wallets.parquet"
    if wpath.exists():
        w = pl.read_parquet(wpath)
        wmap = dict(zip(w["user"].to_list(), w["wid"].to_list()))

    meta, nbytes, bad = [], 0, []
    for i, dt in enumerate(days, 1):
        if dt in stat:
            o = s3.get_object(Bucket=bucket, Key=stat[dt])
            j = json.loads(o["Body"].read())
            meta.append({"dt": dt, "rows": j.get("rows"),
                         "missing_hours": json.dumps(j.get("missing_hours"))})
        out = outdir / f"dt={dt}.parquet"
        if out.exists():
            continue
        loc = pl.read_parquet(localdir / f"dt={dt}.parquet", columns=["oid", "status"])
        pf = pq.ParquetFile(fs.open_input_file(f"{bucket}/{part[dt]}"))
        md = pf.metadata          # 実際に課金される「圧縮後の取り出しバイト数」
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for c in range(g.num_columns):
                cc = g.column(c)
                if cc.path_in_schema in ("user", "status"):
                    nbytes += cc.total_compressed_size
        t = pf.read(columns=["user", "status"], use_threads=True)
        rem = pl.from_arrow(t.column("status").combine_chunks())
        if rem.len() != loc.height or not rem.equals(loc["status"].rename("")):
            bad.append(dt)
            print(f"  {dt} ★status 列が手元と一致しない -> 捨てる", file=sys.stderr)
            del t, rem, loc
            continue
        users = pl.from_arrow(t.column("user").combine_chunks()).rename("user")
        del t, rem
        keep = (loc["status"] == "open").to_numpy()
        u = users.filter(pl.Series(keep))
        for s in u.unique().to_list():
            if s not in wmap:
                wmap[s] = len(wmap)
        pl.DataFrame({"oid": loc["oid"].filter(pl.Series(keep)),
                      "wid": u.replace_strict(wmap, return_dtype=pl.Int32)}
                     ).write_parquet(out, compression="zstd")
        del loc, users, u
        if i % 10 == 0 or i == len(days):
            print(f"  {i}/{len(days)} 取り出し {nbytes/1e9:.3f} GB "
                  f"(≈${nbytes/1e9*0.12:.2f}) 口座 {len(wmap):,}",
                  file=sys.stderr)

    pl.DataFrame({"user": list(wmap), "wid": list(wmap.values())}).write_parquet(wpath)
    if meta:
        pl.DataFrame(meta).write_csv(ROOT / "data" / f"l1_meta_{tag}.csv")
    print(f"[fetch] 口座 {len(wmap):,} 件 / 取り出し {nbytes/1e9:.3f} GB "
          f"≈ ${nbytes/1e9*0.12:.2f}"
          + (f" / 不一致 {bad}" if bad else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
