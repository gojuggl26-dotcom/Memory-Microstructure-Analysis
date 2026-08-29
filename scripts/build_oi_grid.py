"""Order Imbalance を「イベントごと」ではなく固定時間窓ごとに算出する。

    order_imbalance = (Volume_buy − Volume_sell) / (Volume_buy + Volume_sell)

窓の下限はデータで決まる: node_fills の時刻はミリ秒精度なので **1ms が最小単位**。
それより細かい窓は情報を増やさない。比較のため 1ms / 10ms / 100ms / 1s を出す。

出力は約定のあった窓だけ(空の窓は行にしない)。`ts` は窓の開始時刻(UTC ns)。
方向はテイカー(crossed=True)の side で決める。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

FILLS = Path("data/fills_v99")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
STEPS = {"1ms": 1_000_000, "10ms": 10_000_000, "100ms": 100_000_000,
         "1s": 1_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000}


def taker(dt: str):
    p = FILLS / f"dt={dt}"
    files = sorted(p.rglob("*.parquet")) if p.exists() else []
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f, columns=["ts", "sz", "side", "crossed", "tid"])
                    for f in files], how="diagonal_relaxed")
    return (df.filter(pl.col("crossed")).unique(subset=["tid"], keep="first")
              .with_columns(pl.col("ts").cast(pl.Int64).alias("ts_ns")).sort("ts_ns"))


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in FILLS.iterdir() if p.is_dir())
    stats = []
    for i, dt in enumerate(days, 1):
        d = taker(dt)
        if d is None or d.height == 0:
            continue
        row = {"dt": dt, "n_trades": d.height}
        for name, step in STEPS.items():
            g = (d.with_columns((pl.col("ts_ns") // step * step).alias("ts"))
                   .group_by("ts")
                   .agg(vol_buy=pl.col("sz").filter(pl.col("side") == "B").sum(),
                        vol_sell=pl.col("sz").filter(pl.col("side") == "A").sum(),
                        n_trades=pl.len())
                   .sort("ts")
                   .with_columns(
                       ((pl.col("vol_buy") - pl.col("vol_sell"))
                        / (pl.col("vol_buy") + pl.col("vol_sell"))).alias("order_imbalance"))
                   .with_columns(pl.col("order_imbalance").fill_nan(None)))
            out = OUT_DIR / "data" / "oi_grid" / f"step={name}" / f"dt={dt}"
            out.mkdir(parents=True, exist_ok=True)
            g.write_parquet(out / "part-000.parquet", compression="zstd")
            oi = g["order_imbalance"].to_numpy()
            row[f"{name}_windows"] = g.height
            row[f"{name}_trades_per_window"] = d.height / g.height
            row[f"{name}_saturated"] = float(np.mean(np.abs(oi) > 0.999))
            row[f"{name}_mean"] = float(np.nanmean(oi))
        stats.append(row)
        if i % 20 == 0:
            print(f"{i}/{len(days)}", flush=True)
    s = pl.DataFrame(stats)
    s.write_csv(OUT_DIR / "data" / "oi_grid_daily.csv")
    tot = {"days": s.height, "trades": int(s["n_trades"].sum())}
    for name in STEPS:
        w = int(s[f"{name}_windows"].sum())
        tot[name] = {
            "windows": w,
            "trades_per_window": float(s["n_trades"].sum() / w),
            "saturated_share": float((s[f"{name}_saturated"] * s[f"{name}_windows"]).sum() / w),
        }
    (OUT_DIR / "data" / "oi_grid_summary.json").write_text(
        json.dumps(tot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
