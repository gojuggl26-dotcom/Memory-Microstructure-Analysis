"""enforce_uncrossed の resync ティックが 1 秒グリッドに与える影響を測り、除去する。

見つけた現象:
  クロス状態を解消する resync は 1 秒スナップショット時刻に走り、
  ts = 秒境界 − 1ns のティックとして出る(ts mod 1s == 999999999)。
  1 秒グリッドはちょうどその点を拾うため、**片側が退去した直後の異常な mid** が
  グリッドに載る。2026-08-09 では mid が 51.19 → 25.99 → 51.19 と 2 秒で往復し、
  |y| = 6,777bp の窓が 2 つできた。

判定: グリッド点 T の紐づくティック ts = T − stale_ns。
      T の側と T+Δ の側(= 次の行の紐づくティック)のどちらかが resync なら汚染。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); PANEL = D / "panel_1s"
days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
tot = {"n": 0, "sy2": 0.0, "n_bad": 0, "sy2_bad": 0, "n_big": 0, "n_big_bad": 0}
rows = []
for k, dt in enumerate(days):
    d = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet",
                        columns=["ts", "y", "stale_ns"]).sort("ts")
    att = (d["ts"].to_numpy() - d["stale_ns"].to_numpy().astype(np.int64))
    is_res = (att % 1_000_000_000) == 999_999_999
    bad = is_res.copy()
    bad[:-1] |= is_res[1:]                       # T+Δ 側が resync の行も汚染
    y = d["y"].to_numpy()
    big = np.abs(y) > 50
    tot["n"] += y.size; tot["sy2"] += float((y ** 2).sum())
    tot["n_bad"] += int(bad.sum()); tot["sy2_bad"] += float((y[bad] ** 2).sum())
    tot["n_big"] += int(big.sum()); tot["n_big_bad"] += int((big & bad).sum())
    rows.append({"dt": dt, "n": int(y.size), "n_bad": int(bad.sum()),
                 "sd_all": float(y.std()),
                 "sd_clean": float(y[~bad].std()) if (~bad).sum() > 100 else None})
    # 汚染フラグを保存(再利用のため)
    pl.DataFrame({"ts": d["ts"], "is_resync_contaminated": bad}).write_parquet(
        PANEL / f"dt={dt}" / "resync_flag.parquet", compression="zstd")

s = pl.DataFrame(rows)
s.write_csv(D / "resync_artifact_daily.csv")
out = {
    "panel_rows": tot["n"], "contaminated_rows": tot["n_bad"],
    "contaminated_share": tot["n_bad"] / tot["n"],
    "sum_y2_share_of_contaminated": tot["sy2_bad"] / tot["sy2"],
    "rows_abs_y_gt_50": tot["n_big"],
    "of_which_contaminated": tot["n_big_bad"],
    "worst_days": s.sort("n_bad", descending=True).head(5).to_dicts(),
    "sd_reduction_median": float((1 - s["sd_clean"] / s["sd_all"]).median()),
}
(D / "resync_artifact.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
