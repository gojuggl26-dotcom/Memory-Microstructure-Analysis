"""銘柄ごとの日次行数を、parquet のフッタ(メタデータ)だけを range GET して集計する。

本体を落とさないので egress はファイルあたり数 KB(L2 全読みは数十 GB になるので厳禁)。
L1 / fills は書き出し時の `_stats.json` に行数があるのでそちらを読む。

x が確定する時刻 / y の期間: 該当なし(在庫の棚卸し)。

    uv run python scripts/inventory_rows.py --coin xyz:MU
出力: data/rows_<coin>.parquet(列: layer, table, dt, rows, size, row_groups)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import polars as pl
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()

    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")
    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet").filter(pl.col("coin") == a.coin)
    if inv.is_empty():
        sys.exit(f"{a.coin} のオブジェクトが在庫キャッシュに無い(--refresh したか?)")

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    s3 = sess.client("s3")
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(
        access_key=cr.access_key, secret_key=cr.secret_key, session_token=cr.token, region="us-east-1"
    )

    # L2 / L3: parquet フッタから行数
    parts = inv.filter(
        (pl.col("leaf") == "part-000.parquet") & pl.col("layer").is_in(["l2", "l3"])
    ).select("layer", "table", "dt", "key", "size").rows()

    def footer(r):
        layer, table, dt, key, size = r
        md = pq.ParquetFile(fs.open_input_file(f"{bucket}/{key}")).metadata
        return {"layer": layer, "table": table, "dt": dt, "rows": md.num_rows,
                "size": size, "row_groups": md.num_row_groups}

    # L1 / fills: _stats.json の rows
    stats = inv.filter(
        (pl.col("leaf") == "_stats.json") & pl.col("layer").is_in(["l1", "fills"])
    ).select("layer", "table", "dt", "key", "size").rows()

    def stat(r):
        layer, table, dt, key, _ = r
        s = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        return {"layer": layer, "table": table, "dt": dt, "rows": int(s["rows"]),
                "size": None, "row_groups": None}

    with ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(footer, parts)) + list(ex.map(stat, stats))

    df = pl.DataFrame(rows)
    # L1/fills の size は part-000.parquet 側から補う
    sizes = inv.filter(pl.col("leaf") == "part-000.parquet").select(
        "layer", "table", "dt", pl.col("size").alias("size_part")
    )
    df = (
        df.join(sizes, on=["layer", "table", "dt"], how="left")
        .with_columns(size=pl.coalesce("size", "size_part"))
        .drop("size_part")
        .sort(["layer", "table", "dt"])
    )
    out = ROOT / "data" / f"rows_{a.coin.replace(':', '_')}.parquet"
    df.write_parquet(out)
    print(f"[rows] {df.height} partitions -> {out}", file=sys.stderr)
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(
            df.group_by(["layer", "table"]).agg(
                n_days=pl.len(), rows=pl.col("rows").sum(),
                rows_per_day_med=pl.col("rows").median().cast(pl.Int64),
                gib=(pl.col("size").sum() / 1024**3).round(3),
            ).sort(["layer", "table"])
        )


if __name__ == "__main__":
    main()
