"""Binance DRAMUSDT(TRADIFI_PERPETUAL)の aggTrades を取得して parquet 化する。

【なぜ aggTrades か】
  data.binance.vision の DRAMUSDT には **bookTicker が存在しない**(prefix は
  あるがファイルが 0 件。2026-08-20 に確認)。bookDepth は 1 分粒度で粗すぎる。
  したがってミリ秒精度で入手できるのは約定列のみ。
  **Binance 側は「mid」ではなく「約定価格」である**ことを結論に明記すること。

【mid の代理】
  aggTrades の `is_buyer_maker` から攻撃側が判る。
    is_buyer_maker=true  → 買い手がメイカー = 売り攻撃 = **bid で約定**
    is_buyer_maker=false → 売り手がメイカー = 買い攻撃 = **ask で約定**
  スプレッドを 1 ティック(0.01)と仮定して mid ≈ price ± tick/2 を作る
  (気配跳ねを半分にする代理。仮定が外れても**符号の向きは変わらない**)。

【出力】 data/binance/dt=YYYY-MM-DD.parquet
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SYM = "DRAMUSDT"
TICK = 0.01
BASE = ("https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{SYM}/{SYM}-aggTrades-{{d}}.zip")
OUT = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/binance")


def fetch_day(d: str) -> int:
    out = OUT / f"dt={d}.parquet"
    if out.exists():
        return -1
    try:
        with urllib.request.urlopen(BASE.format(d=d), timeout=120) as r:
            raw = r.read()
    except Exception as e:                      # 未上場日・欠測日は飛ばす
        print(f"  {d}: 取得できず ({type(e).__name__})", flush=True)
        return 0
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pl.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df = df.select(
        ts_ms=pl.col("transact_time").cast(pl.Int64),
        px=pl.col("price").cast(pl.Float64),
        qty=pl.col("quantity").cast(pl.Float64),
        buyer_maker=pl.col("is_buyer_maker").cast(pl.Boolean),
    ).sort("ts_ms")
    # mid 代理: bid 約定なら +tick/2、ask 約定なら −tick/2
    df = df.with_columns(
        mid_proxy=pl.col("px") + pl.when(pl.col("buyer_maker"))
        .then(TICK / 2).otherwise(-TICK / 2))
    df.write_parquet(out)
    return len(df)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d0 = date(2026, 5, 18)                      # Binance 上場日
    d1 = date(2026, 8, 10)                      # HL アーカイブの最終日
    days = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    if len(sys.argv) > 1:
        days = days[::int(sys.argv[1])]
    tot = 0
    for i, d in enumerate(days):
        n = fetch_day(d)
        if n > 0:
            tot += n
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(days)} 累計 {tot:,} 行", flush=True)
    print(f"完了: {len(days)} 日 / {tot:,} 行")


if __name__ == "__main__":
    main()
