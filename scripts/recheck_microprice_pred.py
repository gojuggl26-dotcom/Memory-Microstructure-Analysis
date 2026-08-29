"""microprice_report.md §3-2 のイベント時間の回帰を、クロス状態を除いて再計算する。

クロス(best_ask < best_bid)は全ティックの 0.045% しかないが、mid が意味を失って
|MP − mid| が最大 7,349bp に飛ぶため、除かないと二乗和が数日に支配される。
除外前後を並べて出し、レポートの数値を訂正する根拠にする。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/microprice")
H = {"100ms": 100_000_000, "1s": 1_000_000_000, "10s": 10_000_000_000}


def stats(days: list[str], drop_crossed: bool) -> dict:
    acc = {k: np.zeros(6) for k in H}
    for dt in days:
        df = pl.read_parquet(SRC / f"dt={dt}" / "part-000.parquet",
                             columns=["ts", "mid", "microprice", "best_bid", "best_ask",
                                      "micro_dev_bp"]).sort("ts")
        if drop_crossed:
            df = df.filter(pl.col("best_ask") > pl.col("best_bid"))
        if df.height < 100:
            continue
        ts = df["ts"].to_numpy()
        mid = df["mid"].to_numpy()
        x_all = df["micro_dev_bp"].to_numpy()
        for k, h in H.items():
            tgt = ts + h
            idx = np.nonzero(tgt <= ts[-1])[0]
            if idx.size < 100:
                continue
            j = np.searchsorted(ts, tgt[idx], side="right") - 1
            y = (mid[j] - mid[idx]) / mid[idx] * 1e4
            x = x_all[idx]
            acc[k] += (idx.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())
    out = {}
    for k, a in acc.items():
        n, sx, sy, sxx, syy, sxy = a
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        cxy = sxy / n - (sx / n) * (sy / n)
        beta = cxy / vx
        out[k] = {
            "n": int(n), "beta": beta, "corr": cxy / (vx * vy) ** 0.5,
            "r2": cxy ** 2 / (vx * vy), "std_x_bp": vx ** 0.5, "std_y_bp": vy ** 0.5,
            "rmse_mid_bp": (syy / n) ** 0.5,
            "rmse_micro_calibrated_bp": ((syy - 2 * beta * sxy + beta * beta * sxx) / n) ** 0.5,
        }
    return out


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    res = {"with_crossed": stats(days, False), "without_crossed": stats(days, True)}
    Path("C:/Users/ii562/Downloads/Memory/data/DRAM/microprice_pred_recheck.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in H:
        a, b = res["with_crossed"][k], res["without_crossed"][k]
        print(f"{k:>6} 含む: beta={a['beta']:+.4f} corr={a['corr']:+.4f} std_x={a['std_x_bp']:8.3f} "
              f"| 除く: beta={b['beta']:+.4f} corr={b['corr']:+.4f} std_x={b['std_x_bp']:.3f} "
              f"rmse {b['rmse_mid_bp']:.3f}->{b['rmse_micro_calibrated_bp']:.3f}")


if __name__ == "__main__":
    main()
