"""分内当てはめの R² の帰無分布(1 分の中で y をシャッフル)。

n≒60 行に 9 変数を当てると、関係が皆無でも R² は 0 にならない。
どこまでが「当てはめの自由度」でどこからが信号かを分けるため、
同じ手順で y だけを分内シャッフルして R² を出す。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl
from sklearn.linear_model import Lasso

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
r = json.loads((D/"minute_lasso.json").read_text(encoding="utf-8"))
FEATS = r["features"]; ALPHA = 0.0161; MIN_N = 40
PANEL = D/"panel_1s"; MIN_NS = 60_000_000_000
rng = np.random.default_rng(20260814)

days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
xs, ys, ms = [], [], []
for dt in days:
    d = pl.read_parquet(PANEL/f"dt={dt}"/"part-000.parquet", columns=["ts","y",*FEATS]).sort("ts")
    fl = PANEL/f"dt={dt}"/"resync_flag.parquet"
    if fl.exists():
        d = (d.join(pl.read_parquet(fl), on="ts", how="left")
              .with_columns(pl.col("is_resync_contaminated").fill_null(False))
              .filter(~pl.col("is_resync_contaminated")).drop("is_resync_contaminated"))
    xs.append(d.select(FEATS).to_numpy()); ys.append(d["y"].to_numpy()); ms.append(d["ts"].to_numpy()//MIN_NS)
X = np.vstack(xs); y = np.concatenate(ys); mi = np.concatenate(ms)
o = np.argsort(mi, kind="stable"); X, y, mi = X[o], y[o], mi[o]
uniq, start = np.unique(mi, return_index=True); end = np.append(start[1:], mi.size)

mdl = Lasso(alpha=ALPHA, max_iter=2000, tol=1e-4)
r2n = np.full(uniq.size, np.nan)
for i, (a, b) in enumerate(zip(start, end)):
    if b - a < MIN_N:
        continue
    xb = X[a:b]; yb = rng.permutation(y[a:b])          # ★分内でシャッフル = 関係を壊す
    sx, sy = xb.std(0), yb.std()
    if sy <= 0 or np.any(sx <= 0):
        continue
    z = (xb - xb.mean(0))/sx; t = (yb - yb.mean())/sy
    mdl.fit(z, t)
    p = z @ mdl.coef_ + mdl.intercept_
    r2n[i] = 1.0 - float(np.sum((t-p)**2)/np.sum((t-t.mean())**2))
ok = np.isfinite(r2n)
out = {"n": int(ok.sum()), "null_r2_median": float(np.median(r2n[ok])),
       "null_r2_q90": float(np.quantile(r2n[ok], .9)),
       "actual_r2_median": r[f"in_minute_alpha_{ALPHA}"]["r2_median"],
       "actual_r2_q90": r[f"in_minute_alpha_{ALPHA}"]["r2_q90"]}
out["excess_median"] = out["actual_r2_median"] - out["null_r2_median"]
(D/"minute_lasso_null.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
