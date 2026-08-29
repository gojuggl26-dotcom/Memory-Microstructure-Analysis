"""リターン自己相関の図(17〜19)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C = {"mid": "#3b6fd4", "micro": "#c2410c", "trade": "#0f766e"}
LAB = {"mid": "中値", "micro": "マイクロプライス", "trade": "約定価格"}
INK, GRID, CM = "#1f2733", "#e3e7ee", "#9aa3b2"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D / "return_acf.json").read_text(encoding="utf-8"))
STEPS = ["1ms", "10ms", "100ms", "1s", "10s", "60s"]
X = [1, 10, 100, 1000, 10000, 60000]   # ms

# 17) rho1 vs サンプリング間隔
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for s in ("mid", "micro", "trade"):
    med = [r[f"{s}|{g}"]["rho_daily_median"][0] for g in STEPS]
    q25 = [r[f"{s}|{g}"]["rho_daily_q25"][0] for g in STEPS]
    q75 = [r[f"{s}|{g}"]["rho_daily_q75"][0] for g in STEPS]
    ax.fill_between(X, q25, q75, color=C[s], alpha=0.13, linewidth=0)
    ax.plot(X, med, color=C[s], linewidth=2, marker="o", markersize=5, label=LAB[s])
ax.axhline(0, color=INK, linewidth=0.9)
ax.set_xscale("log")
ax.set_xticks(X, STEPS)
ax.set_xlabel("サンプリング間隔")
ax.set_ylabel("ρ₁(1 次の自己相関)")
ax.set_title("リターンの 1 次自己相関 — 日次推定の中央値と四分位範囲")
ax.legend(frameon=False)
fig.savefig(CH / "xyz_DRAM_17_acf_lag1_by_frequency.png"); plt.close(fig)

# 18) ラグ 1..20 の ACF
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, g in zip(axes, ("100ms", "1s", "10s")):
    for s in ("mid", "micro", "trade"):
        ax.plot(range(1, 21), r[f"{s}|{g}"]["rho_daily_median"], color=C[s],
                linewidth=1.8, marker="o", markersize=3, label=LAB[s])
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("ラグ")
    ax.set_title(f"{g} サンプリング(ゼロ率 {r[f'mid|{g}']['zero_share']:.0%})")
axes[0].set_ylabel("自己相関(日次中央値)")
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_18_acf_by_lag.png"); plt.close(fig)

# 19) 月次推移とゼロ率
d = pl.read_csv(D / "return_acf_daily.csv").with_columns(pl.col("dt").str.slice(0, 7).alias("m"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
t = d.filter(pl.col("step") == "1s").group_by(["series", "m"]).agg(pl.col("rho1").median()).sort("m")
for s in ("mid", "micro", "trade"):
    v = t.filter(pl.col("series") == s)
    axes[0].plot(v["m"], v["rho1"], color=C[s], linewidth=2, marker="o", markersize=5, label=LAB[s])
axes[0].axhline(0, color=INK, linewidth=0.9)
axes[0].set_title("1 秒サンプリングの ρ₁ — 月次中央値")
axes[0].set_ylabel("ρ₁")
axes[0].legend(frameon=False, fontsize=8)
z = [r[f"mid|{g}"]["zero_share"] for g in STEPS]
axes[1].plot(X, [100 * x for x in z], color=CM, linewidth=2, marker="o", markersize=5)
axes[1].set_xscale("log")
axes[1].set_xticks(X, STEPS)
axes[1].set_ylabel("リターンが厳密に 0 の割合(%)")
axes[1].set_xlabel("サンプリング間隔")
axes[1].set_title("細かい窓ほど「価格が動いていない」点が占める")
fig.savefig(CH / "xyz_DRAM_19_acf_monthly_and_zero.png"); plt.close(fig)
print("ok")
