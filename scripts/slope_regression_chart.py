"""S^Ask / S^Bid の推定と検定(図 29)。"""
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

r = json.loads((D/"slope_regression.json").read_text(encoding="utf-8"))
e = json.loads((D/"slope_regression_extra.json").read_text(encoding="utf-8"))
full = r["model_full"]["coef"]; simp = r["model_simple"]["coef"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (1) β と 3 種の 95% 区間
names = ["S_ask", "S_bid"]
x = np.arange(2)
for k, (tag, key, col, off) in enumerate((("Newey-West", "se_nw", CM, -0.22),
                                          ("日次クラスター", "se_cluster", C1, 0.0),
                                          ("Fama-MacBeth", "se_fm", C2, 0.22))):
    b = [full[n]["beta"] for n in names]
    s = [full[n][key] * 1.96 for n in names]
    axes[0].errorbar(x + off, b, yerr=s, fmt="o", color=col, capsize=4,
                     markersize=7, linewidth=2, label=tag)
axes[0].axhline(0, color=INK, linewidth=1)
axes[0].set_xticks(x, ["β_A (S^Ask)", "β_B (S^Bid)"])
axes[0].set_ylabel("係数(bp リターン / bp 距離)")
axes[0].set_title("S + X モデルの β と 95% 区間")
axes[0].legend(frameon=False, fontsize=8)

# (2) t 統計量の比較
labs = ["z\n(NW)", "t₉₈\n(クラスター)", "t₉₈\n(FM)"]
w = 0.36
for i, n in enumerate(names):
    v = [abs(full[n]["z_newey_west"]), abs(full[n]["t98_cluster"]), abs(full[n]["fm_t98"])]
    axes[1].bar(np.arange(3) + (i - 0.5) * w, v, width=w,
                color=[C1, C2][i], label=n)
axes[1].axhline(1.98, color=INK, linewidth=1.2, linestyle="--")
axes[1].text(2.2, 3, "5% 有意水準 (t₉₈=1.98)", fontsize=7.5, color=INK)
axes[1].set_yscale("log")
axes[1].set_xticks(range(3), labs)
axes[1].set_ylabel("|統計量|(対数)")
axes[1].set_title("n=845 万の z は使わない。t₉₈ で判断")
axes[1].legend(frameon=False, fontsize=8)

# (3) 経済的大きさ
vals = [abs(full["S_ask"]["beta"]) * e["sd_S_ask"],
        abs(full["S_bid"]["beta"]) * e["sd_S_bid"]]
axes[2].bar(["S^Ask 1SD", "S^Bid 1SD"], vals, color=C1, width=0.5)
axes[2].axhline(e["half_spread_median_bp"], color=C2, linewidth=2, linestyle="--")
axes[2].text(0.1, e["half_spread_median_bp"] * 1.03,
             f"ハーフスプレッド {e['half_spread_median_bp']:.3f}bp", color=C2, fontsize=8)
axes[2].set_ylabel("予測される値動き (bp)")
axes[2].set_title("効果は執行コストの 4% 程度")
for i, v in enumerate(vals):
    axes[2].text(i, v + 0.02, f"{v:.4f}bp", ha="center", fontsize=9, color=INK)
fig.savefig(CH / "xyz_DRAM_29_slope_regression.png")
print("ok")
