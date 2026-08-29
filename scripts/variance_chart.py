"""分散とブロック相対 MSE の関係(図 24)。"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, CM, INK, GRID = "#3b6fd4", "#c2410c", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

a = pl.read_csv(D/"variance_blocks.csv"); b = pl.read_csv(D/"variance_blocks_clean.csv")
dv = pl.read_csv(D/"variance_daily.csv")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

# (1) 解像度別の分散(日次中央値)— 時間スケーリング
res = ["var_1ms","var_10ms","var_100ms","var_1s","var_10s","var_60s"]
xs = [1, 10, 100, 1000, 10000, 60000]
med = [dv[c].median() for c in res]
axes[0].plot(xs, med, marker="o", color=C1, linewidth=2, label="実測(日次中央値)")
axes[0].plot(xs, [med[0]*x/xs[0] for x in xs], color=CM, linestyle="--", linewidth=1.5,
             label="ランダムウォーク(Δt に比例)")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xticks(xs, ["1ms","10ms","100ms","1s","10s","60s"])
axes[0].set_xlabel("サンプリング間隔"); axes[0].set_ylabel("リターン分散 (bp²)")
axes[0].set_title("分散はほぼ Δt に比例(60s で +20% 超過)")
axes[0].legend(frameon=False, fontsize=8)

# (2) ブロック分散の時系列
axes[1].plot(a["block"], a["var_1s"], marker="o", markersize=4, color=C2, linewidth=1.6,
             label="ブロック分散(1s)")
axes[1].set_yscale("log")
axes[1].set_xlabel("ブロック(3 日ずつ・33 分割)"); axes[1].set_ylabel("var_1s (bp²)")
axes[1].set_title("最後のブロックだけ 100 倍超 = アーティファクト")
axes[1].annotate("2 行が Σy² の 71%", (32, a["var_1s"][-1]), xytext=(20, 60),
                 color=C2, fontsize=8, arrowprops=dict(arrowstyle="->", color=C2))
axes[1].legend(frameon=False, fontsize=8)

# (3) 分散 vs ブロック相関(除去前後)
for d, col, lab in ((a, CM, "除去前"), (b, C1, "除去後")):
    axes[2].scatter(d["var_1s"], d["corr_raw"], s=32, color=col, label=lab, alpha=0.85)
axes[2].set_xscale("log")
axes[2].axhline(0, color=INK, linewidth=0.9)
axes[2].set_xlabel("ブロックの分散 var_1s (bp²)"); axes[2].set_ylabel("ブロック相関")
axes[2].set_title("汚染を除くと「分散が大きい方が当たる」に反転")
axes[2].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_24_variance_blocks.png")
print("ok")
