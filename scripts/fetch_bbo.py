"""WORK_BUCKET の l2/bbo から必要列だけを落としてローカルに 1 本の parquet を作る。

bbo は 1 イベント 1 行で、最良気配の価格と数量が入っている。MicroPrice には
**気配の数量**が要るので、fills だけでは作れずこの表が要る。

列プルーニングして落とす。`coin` は全行同じ値の文字列で嵩むため落とし、
`mid` と `spread_bp` は best_bid / best_ask から復元できるので落とす。

x が確定する時刻 / y の期間: 該当なし(データ取得)。

    uv run python scripts/fetch_bbo.py --coin xyz:MU
出力: data/bbo_<coin>.parquet
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
COLS = ["ts", "best_bid", "best_ask", "bid_sz", "ask_sz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")
    tag = a.coin.replace(":", "_")

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    sel = (pl.col("coin") == a.coin) & (pl.col("layer") == "l2") & (pl.col("table") == "bbo")
    keys = inv.filter(sel & (pl.col("leaf") == "part-000.parquet")).sort("dt").select("dt", "key").rows()
    # _SUCCESS のある日だけ(.inprogress や未完成日を拾わないため)
    done = set(inv.filter(sel & (pl.col("leaf") == "_SUCCESS"))["dt"].to_list())
    keys = [(d, k) for d, k in keys if d in done]
    print(f"[fetch] {len(keys)} 日分", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(
        access_key=cr.access_key, secret_key=cr.secret_key, session_token=cr.token, region="us-east-1"
    )

    def one(item):
        dt, key = item
        t = pq.ParquetFile(fs.open_input_file(f"{bucket}/{key}")).read(columns=COLS)
        return dt, t

    parts, nbytes = [], 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (dt, t) in enumerate(ex.map(one, keys), 1):
            parts.append(t.append_column("dt", pa.array([dt] * t.num_rows, pa.string())))
            nbytes += t.nbytes
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}  {nbytes/1e9:.2f} GB", file=sys.stderr)

    d = pl.from_arrow(pa.concat_tables(parts)).sort("ts")
    out = ROOT / "data" / f"bbo_{tag}.parquet"
    d.write_parquet(out, compression="zstd")
    print(f"[fetch] {d.height:,} 行 -> {out}", file=sys.stderr)
    print(f"  日数 {d['dt'].n_unique()} / 読み取り {nbytes/1e9:.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
