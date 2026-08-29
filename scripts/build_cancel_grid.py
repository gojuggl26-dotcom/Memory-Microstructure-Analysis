"""注文キャンセルを Bid/Ask 別・6 通りの窓幅で集計する。

    n_cancel_{bid,ask}    件数
    vol_cancel_{bid,ask}  数量(取消時点の残量 = orig_sz − filled_sz)
    lambda_cancel_{bid,ask} = vol_cancel / Δt        [数量/秒]

    cancel_diff_n   = n_cancel_ask   − n_cancel_bid        ← 指定の「Bid/Ask キャンセル率」
    cancel_diff_vol = vol_cancel_ask − vol_cancel_bid
    cancel_diff_lambda = λ_cancel_ask − λ_cancel_bid

指定の式は **ask − bid** なので、bid 側の取消が多いと負になる。
(`flow_report.md` の cancel_intensity_diff は bid − ask で符号が逆。両方あるので注意)

対象は板に載った注文のみ(棄却・非滞留 TIF・トリガーは除外)。
出力は「キャンセルが 1 件以上あった窓」だけ(無い窓は 0 と読む)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

SRC = Path("data/l2_v99/lifecycle")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
STEPS = {"1ms": 1_000_000, "10ms": 10_000_000, "100ms": 100_000_000,
         "1s": 1_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000}
NON_RESTING_TIF = {"Ioc", "FrontendMarket", "LiquidationMarket"}


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    stats = []
    for i, dt in enumerate(days, 1):
        can = (pl.read_parquet(
                SRC / f"dt={dt}" / "part-000.parquet",
                columns=["side", "orig_sz", "filled_sz", "ts_close",
                         "terminal_status", "tif", "is_rejected", "is_trigger"])
               .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                       & (~pl.col("tif").is_in(list(NON_RESTING_TIF)).fill_null(False))
                       & pl.col("terminal_status").str.contains("ancel"))
               .select(ts=pl.col("ts_close"), side=pl.col("side"),
                       sz=(pl.col("orig_sz") - pl.col("filled_sz")).clip(0.0, None)))
        row = {"dt": dt, "cancels": can.height,
               "vol_bid": float(can.filter(pl.col("side") == "B")["sz"].sum()),
               "vol_ask": float(can.filter(pl.col("side") == "A")["sz"].sum())}
        for name, step in STEPS.items():
            sec = step / 1e9
            g = (can.lazy()
                 .with_columns((pl.col("ts") // step * step).alias("ts_w"))
                 .group_by("ts_w")
                 .agg(
                     n_cancel_bid=(pl.col("side") == "B").sum().cast(pl.Int32),
                     n_cancel_ask=(pl.col("side") == "A").sum().cast(pl.Int32),
                     vol_cancel_bid=pl.col("sz").filter(pl.col("side") == "B").sum(),
                     vol_cancel_ask=pl.col("sz").filter(pl.col("side") == "A").sum(),
                 )
                 .collect()
                 .rename({"ts_w": "ts"})
                 .fill_null(0.0)
                 .sort("ts")
                 .with_columns(
                     (pl.col("vol_cancel_bid") / sec).alias("lambda_cancel_bid"),
                     (pl.col("vol_cancel_ask") / sec).alias("lambda_cancel_ask"),
                     (pl.col("n_cancel_ask") - pl.col("n_cancel_bid")).alias("cancel_diff_n"),
                     (pl.col("vol_cancel_ask") - pl.col("vol_cancel_bid"))
                     .alias("cancel_diff_vol"),
                 )
                 .with_columns(
                     ((pl.col("vol_cancel_ask") - pl.col("vol_cancel_bid")) / sec)
                     .alias("cancel_diff_lambda")))
            out = OUT_DIR / "data" / "cancel_grid" / f"step={name}" / f"dt={dt}"
            out.mkdir(parents=True, exist_ok=True)
            g.write_parquet(out / "part-000.parquet", compression="zstd")
            row[f"{name}_windows"] = g.height
            row[f"{name}_cancels_per_window"] = can.height / max(g.height, 1)
            # 検証: 窓に落とした後も件数・数量が保存されているか
            assert int(g["n_cancel_bid"].sum() + g["n_cancel_ask"].sum()) == can.height
        stats.append(row)
        print(f"[{i:3d}/{len(days)}] {dt} cancels={can.height:>10,} "
              f"1ms窓={row['1ms_windows']:>9,} 60s窓={row['60s_windows']:>6,}", flush=True)

    s = pl.DataFrame(stats)
    s.write_csv(OUT_DIR / "data" / "cancel_grid_daily.csv")
    tot = {"days": s.height, "cancels": int(s["cancels"].sum()),
           "vol_bid": float(s["vol_bid"].sum()), "vol_ask": float(s["vol_ask"].sum())}
    for name in STEPS:
        w = int(s[f"{name}_windows"].sum())
        tot[name] = {"windows": w, "cancels_per_window": tot["cancels"] / w}
    (OUT_DIR / "data" / "cancel_grid_summary.json").write_text(
        json.dumps(tot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
