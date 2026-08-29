"""l1 から `user` 列だけを追加取得する。

ウォレット別の分析には注文 1 本ごとの発注者が要るが、最初の取得
(fetch_l1.py)では容量のため `hash` / `cloid` / `user` を落としていた。
実測すると `user` は圧縮後で全体の **1.7%**(99 日で約 0.71GB)しかないので、
列プルーニングして取り直す。

★行の対応づけ: 元ファイルは ts 昇順で、fetch_l1.py は並べ替えずに書いている。
したがって**行番号がそのまま対応する**。ただしそれを仮定するだけでは危ういので、
検証用に `status` も一緒に取り、ローカルの l1 と**行単位で完全一致するか**を
確かめてから user を貼る。一致しない日は書かずに止める。

★費用: GLACIER_IR の取り出し $0.03/GB + egress $0.09/GB。
user + status で全体の約 2.2% = 0.92GB ≈ $0.11 の見込み。

    uv run python scripts/fetch_l1_user.py --coin xyz:MU
出力: data/l1u_<coin>/dt=*.parquet(user 列のみ。行順は l1 と同じ)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import polars as pl
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COLS = ["status", "user"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--days", type=int, default=0, help="先頭 N 日だけ")
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")
    tag = a.coin.replace(":", "_")
    l1dir = ROOT / "data" / f"l1_{tag}"
    outdir = ROOT / "data" / f"l1u_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    sel = (pl.col("coin") == a.coin) & (pl.col("layer") == "l1")
    keys = dict(inv.filter(sel & (pl.col("leaf") == "part-000.parquet"))
                   .select("dt", "key").rows())
    have = sorted(p.stem.split("=")[1] for p in l1dir.glob("dt=*.parquet"))
    todo = [d for d in have if d in keys and not (outdir / f"dt={d}.parquet").exists()]
    if a.days:
        todo = todo[: a.days]
    print(f"[fetch] 未取得 {len(todo)} 日 / ローカル l1 {len(have)} 日", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(access_key=cr.access_key, secret_key=cr.secret_key,
                           session_token=cr.token, region="us-east-1")

    def one(day: str):
        t = pq.ParquetFile(fs.open_input_file(f"{bucket}/{keys[day]}")).read(columns=COLS)
        u = pl.from_arrow(t)
        loc = pl.read_parquet(l1dir / f"dt={day}.parquet", columns=["status"])
        if u.height != loc.height:
            return day, f"行数が違う {u.height:,} vs {loc.height:,}", 0
        if not (u["status"].to_numpy() == loc["status"].to_numpy()).all():
            return day, "status の並びが一致しない", 0
        # ★一時ファイルへ書いてから rename する。直接書くと、
        #   並行して読む側が書きかけの parquet を掴む(実際に踏んだ)
        tmp = outdir / f".dt={day}.parquet.tmp"
        u.select("user").write_parquet(tmp, compression="zstd")
        tmp.replace(outdir / f"dt={day}.parquet")
        return day, None, t.nbytes

    nb = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (day, err, n) in enumerate(ex.map(one, todo), 1):
            if err:
                sys.exit(f"★{day}: {err} — 行の対応づけが崩れているので中止")
            nb += n
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  展開後 {nb / 1e9:.2f} GB", file=sys.stderr)
    print(f"[fetch] -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
