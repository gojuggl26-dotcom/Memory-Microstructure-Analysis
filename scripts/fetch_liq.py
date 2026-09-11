"""WORK_BUCKET の fills から**清算(liquidation)行だけ**を落として 1 本の parquet にする。

    uv run python scripts/fetch_liq.py --coin xyz:MU

出力: data/liq_<coin>.parquet(清算行のみ)と data/liqcol_<coin>.csv(列の有無の監査)

なぜ fetch_fills.py と別に要るか
--------------------------------
`fetch_fills.py` が落としている列には `liquidation`(真偽)しか無く、
**どちらの口座が清算されたのか**が判らない。1 つの約定は必ずメイカーと
テイカーの 2 行になるので、真偽だけで数えると清算ノーショナルを
2 倍に数えてしまう。生の fills には

    liquidatedUser      清算された口座(2 行とも同じ値が入る)
    liquidationMethod   market / backstop
    liquidationMarkPx   清算が起きた時点のマーク価格

がある。**清算された建玉 = `user == liquidatedUser` の行の sz** で確定する。

列の有無が時間ファイルごとに揺れる
----------------------------------
Artemis の時間ファイルは、その 1 時間に清算が 1 件も無いと `liquidation` 列
ごと落ちる(全行 null になる)。したがって `liquidation == True` と
`liquidatedUser is not null` の両方で拾い、両者が一致するかを監査に残す。

x が確定する時刻 / y の期間: 該当なし(データ取得)。
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COLS = ["ts", "user", "px", "sz", "side", "startPosition", "dir", "crossed", "tid",
        "oid", "closedPnl", "fee", "liquidation", "liquidatedUser",
        "liquidationMethod", "liquidationMarkPx"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    keys = (inv.filter((pl.col("coin") == a.coin) & (pl.col("layer") == "fills")
                       & (pl.col("leaf") == "part-000.parquet"))
            .sort("dt").select("dt", "key").rows())
    done = set(inv.filter((pl.col("coin") == a.coin) & (pl.col("layer") == "fills")
                          & (pl.col("leaf") == "_SUCCESS"))["dt"].to_list())
    keys = [(d, k) for d, k in keys if d in done]
    print(f"[fetch] {len(keys)} 日分", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(access_key=cr.access_key, secret_key=cr.secret_key,
                           session_token=cr.token, region="us-east-1")

    def one(item):
        dt, key = item
        pf = pq.ParquetFile(f"{bucket}/{key}", filesystem=fs)
        have = set(pf.schema_arrow.names)
        cols = [c for c in COLS if c in have]
        t = pf.read(columns=cols)
        d = pl.from_arrow(t)
        for c in COLS:                       # 欠けた列は null で埋めて形をそろえる
            if c not in d.columns:
                d = d.with_columns(pl.lit(None).alias(c))
        n_all = d.height
        flag = pl.col("liquidation").fill_null(False) | pl.col("liquidatedUser").is_not_null()
        liq = d.filter(flag)
        audit = {"dt": dt, "n_rows": n_all,
                 "has_liq_col": "liquidation" in have,
                 "has_user_col": "liquidatedUser" in have,
                 "n_true": int(d.select(pl.col("liquidation").fill_null(False).sum()).item()),
                 "n_luser": int(d.select(pl.col("liquidatedUser").is_not_null().sum()).item()),
                 "n_kept": liq.height}
        return liq.with_columns(dt=pl.lit(dt)), audit

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(one, keys))

    liq = pl.concat([r[0] for r in res], how="diagonal_relaxed").sort("ts")
    aud = pl.DataFrame([r[1] for r in res]).sort("dt")
    tag = a.coin.replace(":", "_")
    liq.write_parquet(ROOT / "data" / f"liq_{tag}.parquet", compression="zstd")
    aud.write_csv(ROOT / "data" / f"liqcol_{tag}.csv")
    print(f"[fetch] 清算行 {liq.height:,} / 全 {aud['n_rows'].sum():,} 行 "
          f"({liq.height/aud['n_rows'].sum()*100:.4f}%)", file=sys.stderr)
    print(f"[fetch] 列が欠けた日: liquidation={int((~aud['has_liq_col']).sum())} / "
          f"liquidatedUser={int((~aud['has_user_col']).sum())}", file=sys.stderr)
    print(f"[fetch] true と liquidatedUser の不一致: "
          f"{int((aud['n_true'] != aud['n_luser']).sum())} 日", file=sys.stderr)


if __name__ == "__main__":
    main()
