"""非定常性の分解の図(23)。"""
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

d = pl.read_csv(D/"nonstationarity_daily.csv").with_columns(pl.col("dt").str.slice(0,7).alias("m"))
j = json.loads((D/"nonstationarity.json").read_text(encoding="utf-8"))
months = sorted(set(d["m"].to_list()))
FE = ["depth_one_level", "obi_2", "voi_sum", "ofi_sum"]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for f, col in zip(FE, [C1, C2, C3, CM]):
    v = [d.filter(pl.col("m") == m)["corr_" + f].median() for m in months]
    axes[0].plot(months, v, marker="o", color=col, linewidth=2, label=f)
axes[0].set_title("相関は月ごとに 5 倍以上変わる(全窓)")
axes[0].set_ylabel("日次相関の中央値")
axes[0].legend(frameon=False, fontsize=7.5)
axes[0].set_ylim(bottom=0)

for f, col in zip(FE, [C1, C2, C3, CM]):
    v = [d.filter(pl.col("m") == m)["corrnz_" + f].median() for m in months]
    axes[1].plot(months, v, marker="o", color=col, linewidth=2, label=f)
z = [d.filter(pl.col("m") == m)["zero_share"].median() for m in months]
axes[1].bar(months, z, color=GRID, width=0.5, zorder=0, label="y=0 の窓の割合")
axes[1].set_title("y=0 の窓を除いても関係の強まりは残る")
axes[1].legend(frameon=False, fontsize=7.5)
axes[1].set_ylim(bottom=0)

fe2 = ["depth_one_level", "obi_2", "voi_sum", "ofi_sum", "oi", "r_prev", "m_bp"]
x = np.arange(len(fe2))
axes[2].bar(x - 0.22, [j[f]["share_corr"] for f in fe2], width=0.42, color=C1, label="相関の変動")
axes[2].bar(x + 0.22, [j[f]["share_sigma_y"] for f in fe2], width=0.42, color=CM,
            label="σ_y(分散)の変動")
axes[2].axhline(0, color=INK, linewidth=0.9)
axes[2].axhline(1, color=C2, linewidth=1, linestyle="--")
axes[2].set_xticks(x, fe2, rotation=35, ha="right", fontsize=7.5)
axes[2].set_ylabel("log|β| の分散への寄与")
axes[2].set_title("β の揺れは分散ではなく関係が動いている")
axes[2].legend(frameon=False, fontsize=7.5)
fig.savefig(CH / "xyz_DRAM_23_nonstationarity.png")
print("ok")
