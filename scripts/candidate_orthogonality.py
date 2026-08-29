"""候補特徴量の「既存 25 変数への線形従属性」を測る。

  R²(z | X) = 候補 z を既存 25 変数に回帰したときの決定係数
  固有分散  = 1 − R²   … これが大きいほど「既存では説明できない新しい情報」

併せて、y との日次相関(中央値)も出して「新しくて効くか」を判定する。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
CAND = D / "candidate_features"
days = sorted(p.stem for p in CAND.glob("*.parquet"))
rows_x, rows_z, rows_y, dayidx = [], [], [], []
zcols = None
for k, dt in enumerate(days):
    p = pl.read_parquet(D / f"panel_1s/dt={dt}/part-000.parquet")
    fl = D / f"panel_1s/dt={dt}/resync_flag.parquet"
    if fl.exists():
        p = (p.join(pl.read_parquet(fl), on="ts", how="left")
              .with_columns(pl.col("is_resync_contaminated").fill_null(False))
              .filter(~pl.col("is_resync_contaminated")).drop("is_resync_contaminated"))
    c = pl.read_parquet(CAND / f"{dt}.parquet")
    j = p.join(c, on="ts", how="inner")
    ex = [x for x in p.columns if x not in ("ts", "y")]
    zc = [x for x in c.columns if x != "ts"]
    zcols = zc
    rows_x.append(j.select(ex).to_numpy()); rows_z.append(j.select(zc).to_numpy())
    rows_y.append(j["y"].to_numpy()); dayidx.append(np.full(j.height, k))
X = np.vstack(rows_x); Z = np.vstack(rows_z); y = np.concatenate(rows_y); dd = np.concatenate(dayidx)
print(f"rows={len(y):,} 既存={X.shape[1]} 候補={Z.shape[1]} 日数={len(days)}")

# ウィンザライズしてから標準化(外れ値で R² が壊れないように)
def prep(A):
    lo, hi = np.quantile(A, 0.001, axis=0), np.quantile(A, 0.999, axis=0)
    A = np.clip(A, lo, hi)
    s = A.std(0); s[s == 0] = 1.0
    return (A - A.mean(0)) / s
Xs, Zs = prep(X), prep(Z)
Xd = np.column_stack([np.ones(len(Xs)), Xs])
XtX = Xd.T @ Xd + 1e-8 * np.eye(Xd.shape[1])
B = np.linalg.solve(XtX, Xd.T @ Zs)
resid = Zs - Xd @ B
r2 = 1 - resid.var(0) / np.maximum(Zs.var(0), 1e-12)

# y との日次相関(中央値)
cor = {}
for j, c in enumerate(zcols):
    v = []
    for k in range(len(days)):
        m = dd == k
        if m.sum() < 1000 or Z[m, j].std() == 0 or y[m].std() == 0:
            continue
        v.append(float(np.corrcoef(Z[m, j], y[m])[0, 1]))
    cor[c] = (float(np.median(v)) if v else float("nan"),
              int(sum(1 for x in v if x < 0)), len(v))
# 残差(既存で説明できない部分)と y の相関 = 増分情報
rcor = {}
for j, c in enumerate(zcols):
    v = []
    for k in range(len(days)):
        m = dd == k
        if m.sum() < 1000 or resid[m, j].std() == 0 or y[m].std() == 0:
            continue
        v.append(float(np.corrcoef(resid[m, j], y[m])[0, 1]))
    rcor[c] = float(np.median(v)) if v else float("nan")

out = {"days": days, "n_rows": int(len(y)),
       "features": {c: {"r2_on_existing": float(r2[j]), "unique_var": float(1 - r2[j]),
                        "corr_y_median": cor[c][0], "corr_y_days_negative": cor[c][1],
                        "corr_resid_y_median": rcor[c]}
                    for j, c in enumerate(zcols)}}
(D / "candidate_orthogonality.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
print(f"\n{'候補特徴量':>22} {'既存へのR²':>10} {'固有分散':>9} {'yとの相関':>10} {'残差とyの相関':>13} {'符号逆':>7}")
for c, v in sorted(out["features"].items(), key=lambda kv: kv[1]["r2_on_existing"]):
    print(f"{c:>22} {v['r2_on_existing']:>10.4f} {v['unique_var']:>9.3f} "
          f"{v['corr_y_median']:>+10.4f} {v['corr_resid_y_median']:>+13.4f} "
          f"{v['corr_y_days_negative']:>3}/{cor[c][2]}")
