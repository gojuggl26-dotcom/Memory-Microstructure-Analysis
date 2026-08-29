"""1 分フォールド Lasso の図(26)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D/"minute_lasso.json").read_text(encoding="utf-8"))
nl = json.loads((D/"minute_lasso_null.json").read_text(encoding="utf-8"))
c = pl.read_parquet(D/"minute_lasso_coefs.parquet")
FE = r["features"]; A = r["in_minute_alpha_0.0161"]["per_feature"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (1) 係数の符号の内訳
pos = [A[f]["share_positive"] for f in FE]
neg = [A[f]["share_negative"] for f in FE]
zer = [1 - p - n for p, n in zip(pos, neg)]
yy = np.arange(len(FE))[::-1]
axes[0].barh(yy, pos, color=C1, height=0.68, label="係数 > 0")
axes[0].barh(yy, zer, left=pos, color=GRID, height=0.68, label="0(除外)")
axes[0].barh(yy, neg, left=[p+z for p, z in zip(pos, zer)], color=C2, height=0.68, label="係数 < 0")
axes[0].axvline(0.5, color=INK, linewidth=1, linestyle="--")
axes[0].set_yticks(yy, FE, fontsize=8)
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("125,499 分に占める割合")
axes[0].set_title("符号がほぼ半々 = 分単位では係数が定まらない")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")

# (2) R² と帰無分布
r2 = c["r2"].to_numpy(); r2 = r2[np.isfinite(r2)]
axes[1].hist(r2, bins=60, range=(0, 0.8), color=C1, alpha=0.85, label="実測")
axes[1].axvline(nl["null_r2_median"], color=C2, linewidth=2,
                label=f"帰無の中央値 {nl['null_r2_median']:.3f}")
axes[1].axvline(nl["actual_r2_median"], color=C1, linewidth=2, linestyle="--",
                label=f"実測の中央値 {nl['actual_r2_median']:.3f}")
axes[1].set_xlabel("分内当てはめの R²")
axes[1].set_ylabel("分の数")
axes[1].set_title("60 行に 9 変数 → 無関係でも R²=0.12 出る")
axes[1].legend(frameon=False, fontsize=7.5)

# (3) 学習量ごとの標本外成績
names = ["直前 1 分\n(≒60 行)", "3 日ブロック\n(既存)", "直前 30 分\n(≒1,800 行)"]
vals = [r["walkforward_prev1"]["relmse_median"],
        r["block_trained_baseline"]["relmse_median"],
        r["walkforward_prev30"]["relmse_median"]]
shr = [r["walkforward_prev1"]["share_lt_1"],
       r["block_trained_baseline"]["share_lt_1"],
       r["walkforward_prev30"]["share_lt_1"]]
cols = [C2, CM, C1]
axes[2].bar(names, vals, color=cols, width=0.6)
axes[2].axhline(1, color=INK, linewidth=1, linestyle="--")
axes[2].set_ylim(0.9, 1.25)
axes[2].set_ylabel("次の 1 分の相対 MSE(中央値)")
axes[2].set_title("学習量が足りないと逆に害になる")
for i, (v, s) in enumerate(zip(vals, shr)):
    axes[2].text(i, v + 0.006, f"{v:.3f}\n(<1 が {s:.0%})", ha="center", fontsize=8, color=INK)
fig.savefig(CH / "xyz_DRAM_26_minute_lasso.png")
print("ok")
