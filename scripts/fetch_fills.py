"""WORK_BUCKET の fills から必要列だけを落としてローカルに 1 本の parquet を作る。

列プルーニングするので egress は全列読みの約 1/3。

x が確定する時刻 / y の期間: 該当なし(データ取得)。

    uv run python scripts/fetch_fills.py --coin xyz:MU
出力: data/fills_<coin>.parquet
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
COLS = ["ts", "user", "px", "sz", "side", "startPosition", "dir", "crossed", "tid", "liquidation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    keys = (
        inv.filter(
            (pl.col("coin") == a.coin)
            & (pl.col("layer") == "fills")
            & (pl.col("leaf") == "part-000.parquet")
        )
        .sort("dt")
        .select("dt", "key")
        .rows()
    )
    # _SUCCESS のある日だけを対象にする(.inprogress や未完成日を拾わないため)
    done = set(
        inv.filter(
            (pl.col("coin") == a.coin) & (pl.col("layer") == "fills") & (pl.col("leaf") == "_SUCCESS")
        )["dt"].to_list()
    )
    keys = [(d, k) for d, k in keys if d in done]
    print(f"[fetch] {len(keys)} 日分", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(
        access_key=cr.access_key, secret_key=cr.secret_key, session_token=cr.token, region="us-east-1"
    )

    def one(item):
        dt, key = item
        t = pq.read_table(f"{bucket}/{key}", columns=COLS, filesystem=fs)
        return t.append_column("dt", pa.array([dt] * t.num_rows, pa.string()))

    with ThreadPoolExecutor(a.threads) as ex:
        tables = list(ex.map(one, keys))

    tbl = pa.concat_tables(tables, promote_options="permissive")
    df = pl.from_arrow(tbl).sort("ts")
    out = ROOT / "data" / f"fills_{a.coin.replace(':', '_')}.parquet"
    df.write_parquet(out, compression="zstd")
    print(f"[fetch] {df.height:,} rows -> {out} ({out.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
    print(df.head(3))


if __name__ == "__main__":
    main()
