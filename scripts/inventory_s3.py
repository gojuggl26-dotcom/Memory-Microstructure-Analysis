"""WORK_BUCKET の全オブジェクトを一度だけ列挙してローカルに保存し、保有状況を集計する。

x が確定する時刻 / y の期間: 該当なし(予測モデルではなく在庫の棚卸し)。

使い方:
    uv run python scripts/inventory_s3.py --refresh      # S3 を再列挙(LIST ≈ $0.0002)
    uv run python scripts/inventory_s3.py --coin xyz:MU  # キャッシュから銘柄別に集計

キャッシュ: data/s3_inventory.parquet(列: key, size, last_modified, layer, table, coin, dt, leaf)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import boto3
import polars as pl

CACHE = Path(__file__).resolve().parents[1] / "data" / "s3_inventory.parquet"
KV = re.compile(r"^([a-z_]+)=(.+)$")


def crawl(bucket: str) -> pl.DataFrame:
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro")).client("s3")
    rows, pages = [], 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        pages += 1
        for o in page.get("Contents", []):
            key = o["Key"]
            parts = key.split("/")
            layer = parts[0]
            coin = dt = None
            table_parts = []
            for p in parts[1:-1]:
                m = KV.match(p)
                if not m:
                    table_parts.append(p)
                elif m.group(1) == "coin":
                    coin = unquote(m.group(2))
                elif m.group(1) == "dt":
                    dt = m.group(2)
            leaf = parts[-1]
            # state/ は dt=YYYY-MM-DD.json のようにファイル名側に日付が入る
            if dt is None:
                m = re.match(r"^dt=(\d{4}-\d{2}-\d{2})\.\w+$", leaf)
                if m:
                    dt = m.group(1)
            rows.append(
                {
                    "key": key,
                    "size": o["Size"],
                    "last_modified": o["LastModified"].replace(tzinfo=None),
                    "layer": layer,
                    "table": "/".join(table_parts) or None,
                    "coin": coin,
                    "dt": dt,
                    "leaf": leaf,
                }
            )
    print(f"[crawl] {len(rows):,} objects / {pages} LIST pages", file=sys.stderr)
    return pl.DataFrame(rows)


def load(refresh: bool, bucket: str) -> pl.DataFrame:
    if refresh or not CACHE.exists():
        df = crawl(bucket)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(CACHE)
    return pl.read_parquet(CACHE)


def summarize(df: pl.DataFrame, coin: str | None) -> None:
    if coin:
        df = df.filter(pl.col("coin") == coin)
        print(f"\n=== coin = {coin} ===")
    g = (
        df.group_by(["layer", "table"])
        .agg(
            n_files=pl.len(),
            gib=(pl.col("size").sum() / 1024**3).round(3),
            n_days=pl.col("dt").n_unique(),
            dt_min=pl.col("dt").min(),
            dt_max=pl.col("dt").max(),
            last_write=pl.col("last_modified").max(),
        )
        .sort(["layer", "table"])
    )
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=40, tbl_width_chars=200):
        print(g)
    print(f"合計 {df.height:,} files / {df['size'].sum()/1024**3:.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--coin")
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET", "")
    if a.refresh and not bucket:
        sys.exit("WORK_BUCKET が未設定(~/.hlpipe.env を source すること)")
    summarize(load(a.refresh, bucket), a.coin)
