r"""Binance ライブ記録の bookTicker → BBO parquet(リードラグ用)。

入力: E:/Binance-perp-data/data/raw/dt=*/symbol={SYM}USDT/stream=bookTicker/*.parquet
      (列 local_ns / event_ms / payload。payload は JSON 文字列)
出力: E:/Memory-lighter/bnb_{SYM}.parquet
      (event_ms=取引所イベント時刻, local_ns=このマシンの受信時刻, bid/ask/サイズ)

★突合の作法(binance-perp-orderbook-project の教訓):
  depth と bookTicker を時刻で突合してはならない(速度が 17 倍違う)。
  ここでは bookTicker 単独しか使わないので該当しない。
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl

RAW = "E:/Binance-perp-data/data/raw"
OUT = Path("E:/Memory-lighter")
PAIR = {"DRAM": "DRAMUSDT", "MU": "MUUSDT", "SNDK": "SNDKUSDT",
        "SKHYNIXUSD": "SKHYNIXUSDT", "SAMSUNGUSD": "SAMSUNGUSDT",
        "AAPL": "AAPLUSDT", "AMZN": "AMZNUSDT", "MSFT": "MSFTUSDT",
        "NVDA": "NVDAUSDT", "TSLA": "TSLAUSDT", "XAG": "XAGUSDT",
        "XAU": "XAUUSDT"}
DT = pl.Struct({"data": pl.Struct({"b": pl.String, "B": pl.String,
                                   "a": pl.String, "A": pl.String})})


def run(sym: str, d0: str, d1: str) -> None:
    bsym = PAIR[sym]
    fs = []
    import os
    for p in sorted(glob.glob(f"{RAW}/dt=*/symbol={bsym}/stream=bookTicker/*.parquet")):
        dt = p.replace("\\", "/").split("dt=")[1][:10]
        # ★録画途中・クラッシュ残骸の 0 バイト parquet が実在する(読むと落ちる)
        if d0 <= dt <= d1 and os.path.getsize(p) > 100:
            fs.append(p)
    if not fs:
        print(f"{sym}: ファイル無し")
        return
    d = (pl.scan_parquet(fs)
         .select(["local_ns", "event_ms", "payload"])
         .with_columns(pl.col("payload").str.json_decode(DT).alias("j"))
         .select([
             "local_ns", "event_ms",
             pl.col("j").struct.field("data").struct.field("b")
               .cast(pl.Float64).alias("bid"),
             pl.col("j").struct.field("data").struct.field("a")
               .cast(pl.Float64).alias("ask"),
             pl.col("j").struct.field("data").struct.field("B")
               .cast(pl.Float32).alias("bid_sz"),
             pl.col("j").struct.field("data").struct.field("A")
               .cast(pl.Float32).alias("ask_sz"),
         ])
         .sort("event_ms")
         .collect(engine="streaming"))
    d.write_parquet(OUT / f"bnb_{sym}.parquet")
    print(f"{sym}: {d.height:,} 行 ({len(fs)} ファイル)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(PAIR))
    ap.add_argument("--start", default="2026-08-18")
    ap.add_argument("--end", default="2026-09-05")
    a = ap.parse_args()
    for sym in a.symbols.split(","):
        run(sym, a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
