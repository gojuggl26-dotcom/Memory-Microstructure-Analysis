"""実現ボラティリティの測定(99 日 × 複数の標本化間隔)。

【なぜ複数間隔か】
  SDE の σ は「在庫を持っている時間スケール」の価格変動でなければならない。
  高頻度側はマイクロストラクチャー雑音で RV が膨らみ、低頻度側は標本が減る。
  signature plot(間隔 vs RV)を描いて、雑音の効く領域を目で確認してから選ぶ。

【時間契約】各日の RV はその日の中値のみから作る。将来の情報は入らない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
GRID_S = [1, 5, 10, 30, 60, 300]          # 標本化間隔 [秒]
NS = 1_000_000_000

rows = []
files = sorted((D / "microprice").glob("dt=*/part-000.parquet"))
for i, f in enumerate(files):
    dt = f.parent.name.split("=")[1]
    mp = (pl.read_parquet(f, columns=["ts", "mid", "is_crossed"])
          .filter(~pl.col("is_crossed") & pl.col("mid").is_not_null()).sort("ts"))
    if mp.height < 1000:
        continue
    ts = mp["ts"].to_numpy().astype(np.int64)
    mid = mp["mid"].to_numpy().astype(np.float64)
    t0, t1 = ts[0], ts[-1]
    cov = (t1 - t0) / NS / 86400.0                       # その日の被覆率
    r = {"dt": dt, "n_ev": mp.height, "cover": float(cov),
         "px_med": float(np.median(mid)),
         "range_pct": float((mid.max() - mid.min()) / np.median(mid) * 100)}
    for g in GRID_S:
        grid = np.arange(t0, t1, g * NS)
        if grid.size < 20:
            r[f"rv{g}"] = np.nan
            continue
        idx = np.searchsorted(ts, grid, side="right") - 1
        p = mid[np.clip(idx, 0, len(mid) - 1)]
        lr = np.diff(np.log(p))
        # 1 日(86,400 秒)に換算した実現ボラティリティ
        r[f"rv{g}"] = float(np.sqrt(np.sum(lr**2) / (grid.size - 1) * (86400 / g)))
    # 終値ベースの日次リターン(比較用)
    r["ret_cc"] = float(np.log(mid[-1] / mid[0]))
    rows.append(r)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

df = pl.DataFrame(rows).sort("dt")
df.write_csv(D / "sde_realized_vol.csv")

print(f"\n=== 実現ボラティリティ signature plot(99 日の中央値)===")
print(f"{'間隔':>6}{'RV 中央値':>12}{'年率':>10}{'p10':>10}{'p90':>10}{'最大':>10}")
for g in GRID_S:
    v = df[f"rv{g}"].drop_nulls().drop_nans().to_numpy()
    print(f"{g:>5}s{np.median(v)*100:>11.3f}%{np.median(v)*np.sqrt(365)*100:>9.0f}%"
          f"{np.percentile(v,10)*100:>9.3f}%{np.percentile(v,90)*100:>9.3f}%{v.max()*100:>9.3f}%")

ev = df.filter((pl.col("dt") >= "2026-07-10") & (pl.col("dt") <= "2026-08-08"))
print(f"\n=== 期間別(30s 標本化)===")
for a, b, nm in [("2026-05-04", "2026-08-10", "全 99 日"),
                 ("2026-07-10", "2026-08-08", "評価窓 30 日"),
                 ("2026-08-09", "2026-08-09", "最悪日 08-09")]:
    s = df.filter((pl.col("dt") >= a) & (pl.col("dt") <= b))["rv30"].drop_nulls().drop_nans().to_numpy()
    if s.size:
        print(f"  {nm:<14} n={s.size:>3}  中央値 {np.median(s)*100:>6.3f}%/日"
              f"  平均 {s.mean()*100:>6.3f}%  最大 {s.max()*100:>6.3f}%"
              f"  年率(中央値) {np.median(s)*np.sqrt(365)*100:>4.0f}%")

print(f"\n終値ベースの日次リターン sd(99 日): {df['ret_cc'].std()*100:.3f}%  "
      f"評価窓: {ev['ret_cc'].std()*100:.3f}%")
print(f"被覆率の最小: {df['cover'].min():.4f}  (1.0 が丸 1 日)")
