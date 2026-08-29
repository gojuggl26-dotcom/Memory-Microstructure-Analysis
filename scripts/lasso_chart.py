"""Lasso の図(22)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, CM, INK, GRID = "#3b6fd4", "#c2410c", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D/"lasso_results.json").read_text(encoding="utf-8"))
best = [v for v in r["variants"] if v["name"] == r["best_variant"]][0]
coef = best["coef"]; uni = r["univariate_corr_train"]
order = sorted(coef, key=lambda f: -abs(coef[f]))
sel = [f for f in order if coef[f] != 0]
drop = [f for f in order if coef[f] == 0]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                         gridspec_kw={"width_ratios": [1.35, 1]})
names = sel + drop
vals = [coef[f] for f in names]
cols = [C1 if coef[f] != 0 else GRID for f in names]
axes[0].barh(range(len(names))[::-1], vals, color=cols, height=0.7)
axes[0].set_yticks(range(len(names))[::-1], names, fontsize=7.5)
axes[0].axvline(0, color=INK, linewidth=0.9)
axes[0].set_xlabel("Lasso 係数(標準化・目的変数は日次ボラ正規化)")
axes[0].set_title(f"{len(sel)} / {len(names)} 変数が残った(α={best['alpha']:.4g})")
for i, f in enumerate(names):
    if coef[f] != 0:
        axes[0].text(coef[f] + 0.0015, len(names) - 1 - i, f"{coef[f]:+.3f}",
                     va="center", fontsize=7, color=INK)

cmp_ = [("モデル\n(9 変数)", best["oos_daily_corr"]["median"], C1),
        ("プラセボ\n(x を日内でずらす)", r["placebo_shifted_features"]["median"], C2),
        ("最良の単一特徴量\n(depth_one_level)", max(uni.values()), CM)]
axes[1].bar([c[0] for c in cmp_], [c[1] for c in cmp_], color=[c[2] for c in cmp_], width=0.6)
axes[1].axhline(0, color=INK, linewidth=0.9)
axes[1].set_ylabel("標本外・日次相関の中央値")
axes[1].set_title("標本外(後 28 日・学習に一切使っていない)")
for i, c in enumerate(cmp_):
    axes[1].text(i, c[1] + 0.004, f"{c[1]:.3f}", ha="center", fontsize=9, color=INK)
fig.savefig(CH / "xyz_DRAM_22_lasso.png")
print("ok", len(sel), "selected")
