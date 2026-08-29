"""各特徴量を単独で使った場合の標本外(後 28 日)日次相関。Lasso との比較用。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); PANEL = D / "panel_1s"
TRAIN_DAYS = 70
days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
res = json.loads((D / "lasso_results.json").read_text(encoding="utf-8"))
feats = res["features"]
acc = {f: [] for f in feats}
for k, dt in enumerate(days):
    if k < TRAIN_DAYS or k == 0:
        continue
    df = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet")
    y = df["y"].to_numpy()
    if y.size < 500 or y.std() == 0:
        continue
    for f in feats:
        x = df[f].to_numpy().astype(float)
        if x.std() == 0:
            continue
        acc[f].append(float(np.corrcoef(x, y)[0, 1]))
out = {f: {"days": len(v), "median": float(np.median(v)) if v else None,
           "n_days_negative": int(sum(1 for z in v if z < 0))} for f, v in acc.items()}
res["univariate_oos_daily"] = out
(D / "lasso_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
for f, v in sorted(out.items(), key=lambda kv: -abs(kv[1]["median"] or 0)):
    print(f"{f:>22} {v['median']:+.4f}  負={v['n_days_negative']}/{v['days']}")
