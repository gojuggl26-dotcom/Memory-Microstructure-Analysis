"""E:/hlpipe のローカルミラーから data/bbo_<coin>.parquet と
data/fills_<coin>.parquet を作る(S3 egress ゼロ)。

fetch_bbo.py / fetch_fills.py は WORK_BUCKET から落とすが、既に
E:/hlpipe に L2(bbo)と fills のミラーがある銘柄は、そこから同じ形の
単一ファイルを組めば費用がかからない。列は fetch_*.py の出力に合わせる。

    uv run python scripts/mirror_to_local.py --coin xyz:AMD [--what bbo,fills]

x が確定する時刻 / y の期間: 該当なし(データ整形)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIRROR = Path("E:/hlpipe")
BBO_COLS = ["ts", "best_bid", "best_ask", "bid_sz", "ask_sz"]
FILL_COLS = ["ts", "user", "px", "sz", "side", "startPosition", "dir",
             "crossed", "tid", "liquidation"]


def enc(coin: str) -> str:
    return "coin=" + quote(coin, safe="")


def build_bbo(coin: str) -> None:
    tag = coin.replace(":", "_")
    base = MIRROR / "l2" / enc(coin) / "bbo"
    days = sorted(p.name[3:] for p in base.glob("dt=*") if (p / "part-000.parquet").exists())
    if not days:
        sys.exit(f"{base} に bbo が無い")
    fr = []
    for d in days:
        t = pl.read_parquet(base / f"dt={d}" / "part-000.parquet", columns=BBO_COLS)
        fr.append(t.with_columns(dt=pl.lit(d)))
    out = pl.concat(fr).sort("ts")
    fp = DATA / f"bbo_{tag}.parquet"
    out.write_parquet(fp, compression="zstd")
    print(f"[bbo] {out.height:,} 行 {len(days)} 日 -> {fp.name}", flush=True)


def build_fills(coin: str) -> None:
    tag = coin.replace(":", "_")
    base = MIRROR / "fills"
    days = sorted(p.name[3:] for p in base.glob("dt=*")
                  if (p / enc(coin) / "part-000.parquet").exists())
    if not days:
        sys.exit(f"{base} に {coin} の fills が無い")
    fr = []
    for d in days:
        t = pl.read_parquet(base / f"dt={d}" / enc(coin) / "part-000.parquet",
                            columns=FILL_COLS)
        fr.append(t.with_columns(dt=pl.lit(d)))
    out = pl.concat(fr).sort("ts")
    fp = DATA / f"fills_{tag}.parquet"
    out.write_parquet(fp, compression="zstd")
    print(f"[fills] {out.height:,} 行 {len(days)} 日 -> {fp.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--what", default="bbo,fills")
    a = ap.parse_args()
    what = set(a.what.split(","))
    if "bbo" in what:
        build_bbo(a.coin)
    if "fills" in what:
        build_fills(a.coin)


if __name__ == "__main__":
    main()
