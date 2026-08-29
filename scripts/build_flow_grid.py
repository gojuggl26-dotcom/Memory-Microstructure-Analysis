"""注文フローの強度を Bid/Ask 別・固定時間窓ごとに算出する。

    λ_cancel^side = (その窓で取り消された数量) / Δt          [数量/秒]
    λ_new^side    = (その窓で板に載った新規注文の数量) / Δt   [数量/秒]

    cancel_ratio        = λ_cancel^bid / (λ_cancel^bid + λ_cancel^ask)
    new_intensity_diff  = λ_new^buy − λ_new^sell

情報源は L2 の注文ライフサイクル表(注文 1 件 = 1 行)。
  新規 = ts_open、数量 = orig_sz
  取消 = ts_close かつ terminal_status が取消系、数量 = orig_sz − filled_sz(取消時の残量)

板に載らなかった注文は除く:
  - is_rejected(perpMarginRejected / badAloPxRejected / iocCancelRejected など)
  - 非滞留 TIF(Ioc / FrontendMarket / LiquidationMarket)
比較用に「全注文(棄却も含む)」版の件数も併せて出す。

出力は活動のあった窓のみ(空の窓は λ = 0 と読む)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

SRC = Path("data/l2_v99/lifecycle")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
STEPS = {"100ms": 100_000_000, "1s": 1_000_000_000,
         "10s": 10_000_000_000, "60s": 60_000_000_000}
CANCEL = {"canceled", "marginCanceled", "scheduledCancel", "liquidatedCanceled",
          "vaultWithdrawalCanceled", "openInterestCapCanceled", "selfTradeCanceled",
          "reduceOnlyCanceled", "siblingFilledCanceled", "delistedCanceled"}
NON_RESTING_TIF = {"Ioc", "FrontendMarket", "LiquidationMarket"}


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    stats = []
    for i, dt in enumerate(days, 1):
        raw = pl.read_parquet(
            SRC / f"dt={dt}" / "part-000.parquet",
            columns=["side", "orig_sz", "filled_sz", "ts_open", "ts_close",
                     "terminal_status", "tif", "is_rejected", "is_trigger"])
        n_all = raw.height
        # 板に載った注文だけを残す(トリガー注文は発火前は板に無いので除く)
        rest = raw.filter(
            (~pl.col("is_rejected"))
            & (~pl.col("is_trigger"))
            & (~pl.col("tif").is_in(list(NON_RESTING_TIF)).fill_null(False))
        )
        del raw
        new = rest.select(ts=pl.col("ts_open"), side=pl.col("side"), sz=pl.col("orig_sz"))
        # 取消系は "...Canceled" / "...Cancel" を含む終端。棄却は既に除いてある
        can = (rest.filter(pl.col("terminal_status").str.contains("ancel"))
                   .select(ts=pl.col("ts_close"), side=pl.col("side"),
                           sz=(pl.col("orig_sz") - pl.col("filled_sz")).clip(0.0, None)))
        row = {"dt": dt, "orders_all": n_all, "orders_resting": rest.height,
               "orders_canceled": can.height}
        del rest

        for name, step in STEPS.items():
            dt_sec = step / 1e9
            def agg(src: pl.DataFrame, tag: str) -> pl.DataFrame:
                return (src.lazy().with_columns((pl.col("ts") // step * step).alias("w"))
                        .group_by("w")
                        .agg(**{
                            f"vol_{tag}_bid": pl.col("sz").filter(pl.col("side") == "B").sum(),
                            f"vol_{tag}_ask": pl.col("sz").filter(pl.col("side") == "A").sum(),
                            f"n_{tag}_bid": (pl.col("side") == "B").sum(),
                            f"n_{tag}_ask": (pl.col("side") == "A").sum(),
                        }).collect())
            a = agg(new, "new")
            b = agg(can, "cancel")
            g = (a.join(b, on="w", how="full", coalesce=True)
                   .fill_null(0)
                   .sort("w")
                   .rename({"w": "ts"})
                   .with_columns(
                       (pl.col("vol_new_bid") / dt_sec).alias("lambda_new_buy"),
                       (pl.col("vol_new_ask") / dt_sec).alias("lambda_new_sell"),
                       (pl.col("vol_cancel_bid") / dt_sec).alias("lambda_cancel_bid"),
                       (pl.col("vol_cancel_ask") / dt_sec).alias("lambda_cancel_ask"),
                   )
                   .with_columns(
                       (pl.col("lambda_new_buy") - pl.col("lambda_new_sell"))
                       .alias("new_intensity_diff"),
                       pl.when(pl.col("lambda_cancel_bid") + pl.col("lambda_cancel_ask") > 0)
                       .then(pl.col("lambda_cancel_bid")
                             / (pl.col("lambda_cancel_bid") + pl.col("lambda_cancel_ask")))
                       .otherwise(None).alias("cancel_ratio"),
                   ))
            out = OUT_DIR / "data" / "flow_grid" / f"step={name}" / f"dt={dt}"
            out.mkdir(parents=True, exist_ok=True)
            g.write_parquet(out / "part-000.parquet", compression="zstd")
            row[f"{name}_windows"] = g.height
            row[f"{name}_cancel_ratio_mean"] = float(g["cancel_ratio"].mean() or 0.0)
            row[f"{name}_lambda_cancel_bid"] = float(g["lambda_cancel_bid"].mean())
            row[f"{name}_lambda_cancel_ask"] = float(g["lambda_cancel_ask"].mean())
            row[f"{name}_new_diff_mean"] = float(g["new_intensity_diff"].mean())
        stats.append(row)
        print(f"[{i:3d}/{len(days)}] {dt} orders={n_all:>10,} "
              f"windows(1s)={row['1s_windows']:>7,}", flush=True)

    s = pl.DataFrame(stats)
    s.write_csv(OUT_DIR / "data" / "flow_grid_daily.csv")
    print(json.dumps({"days": s.height, "orders_all": int(s["orders_all"].sum())}, indent=1))


if __name__ == "__main__":
    main()
