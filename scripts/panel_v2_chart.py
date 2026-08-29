"""新特徴量の効果(図 27)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D/"panel_v2_results.json").read_text(encoding="utf-8"))
cor = r["daily_corr_full_period"]; add = r["added"]; old = r["old_features"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

# (1) 全特徴量の日次相関(新規を強調)
allf = sorted(cor, key=lambda c: -abs(cor[c]["median"]))[:16]
v = [cor[c]["median"] for c in allf]
col = [C2 if c in add else CM for c in allf]
yy = np.arange(len(allf))[::-1]
axes[0].barh(yy, v, color=col, height=0.7)
axes[0].axvline(0, color=INK, linewidth=0.9)
axes[0].set_yticks(yy, allf, fontsize=7.5)
axes[0].set_xlabel("y との日次相関(中央値・全 99 日)")
axes[0].set_title("赤 = 新規。book_slope_diff が最強")

# (2) 新特徴量の固有性
r2 = r["r2_on_existing"]
names = [c for c in add]
uni = [1 - r2[c] for c in names]
yy = np.arange(len(names))[::-1]
axes[1].barh(yy, uni, color=C1, height=0.7)
axes[1].axvline(0.7, color=C2, linewidth=1.2, linestyle="--")
axes[1].set_yticks(yy, names, fontsize=7.5)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("固有分散 = 1 − R²(既存 25 変数への回帰)")
axes[1].set_title("既存で説明できない割合(赤線 = 0.7)")

# (3) Lasso の標本外比較
a, b = r["lasso_old"], r["lasso_new"]
axes[2].bar(["既存 25", "既存 25 + 新規 11"],
            [a["oos_daily_corr_median"], b["oos_daily_corr_median"]],
            color=[CM, C1], width=0.55)
axes[2].set_ylabel("標本外・日次相関の中央値(28 日)")
axes[2].set_title(f"+{(b['oos_daily_corr_median']/a['oos_daily_corr_median']-1)*100:.0f}% 改善")
for i, (m, s) in enumerate(((a["oos_daily_corr_median"], a["n_selected"]),
                            (b["oos_daily_corr_median"], b["n_selected"]))):
    axes[2].text(i, m + 0.004, f"{m:.4f}\n({s} 変数選択)", ha="center", fontsize=9, color=INK)
axes[2].set_ylim(0, max(a["oos_daily_corr_median"], b["oos_daily_corr_median"]) * 1.25)
fig.savefig(CH / "xyz_DRAM_27_panel_v2.png")
print("ok")
