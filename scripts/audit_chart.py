"""監査結果の図: SE の取り方で検定統計量がどれだけ変わるか / 日次 β の分布と予測区間。

統計量は t₉₈(自由度 98 の t 分布)。時刻添字の t とは別物なので添字で区別する。
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl

OUT = Path("C:/Users/ii562/Downloads/Memory"); CH = OUT / "charts"
C1, C2, CM, INK, GRID = "#3b6fd4", "#c2410c", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

def nw(v, lag):
    d = v - v.mean(); s = float(d @ d) / v.size
    for j in range(1, lag + 1):
        s += 2 * (1 - j / (lag + 1)) * float(d[j:] @ d[:-j]) / v.size
    return math.sqrt(max(s, 0) / v.size)

d = pl.read_csv(OUT / "data" / "audit_daily.csv")
b = d["beta"].to_numpy(); se = d["se_hac"].to_numpy(); n = b.size
w = 1 / se**2; mu = (w*b).sum()/w.sum(); Q = (w*(b-mu)**2).sum()
C = w.sum() - (w**2).sum()/w.sum(); tau2 = max(0.0, (Q-(n-1))/C)
w2 = 1/(se**2+tau2); mu2 = (w2*b).sum()/w2.sum(); se2 = math.sqrt(1/w2.sum())
pi = 1.96*math.sqrt(tau2+se2**2)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
labels = ["FM(iid)", "NW lag5", "NW lag10", "NW lag20", "NW lag30", "ランダム効果"]
ts = [b.mean()/(b.std(ddof=1)/math.sqrt(n))] + [b.mean()/nw(b, L) for L in (5,10,20,30)] + [mu2/se2]
cols = [C2] + [C1]*4 + [C1]
axes[0].bar(labels, [abs(t) for t in ts], color=cols, width=0.6)
axes[0].axhline(1.96, color=INK, linewidth=1, linestyle="--")
axes[0].text(0.02, 0.90, "5% 有意水準", transform=axes[0].transAxes, color=INK, fontsize=8)
axes[0].set_ylabel("|t₉₈|")
axes[0].set_title("標準誤差の取り方で |t₉₈| は 8.5 → 2.1 まで動く")
axes[0].tick_params(axis="x", rotation=20)

axes[1].hist(b, bins=40, color=C1, alpha=0.8)
axes[1].axvline(0, color=INK, linewidth=1)
axes[1].axvline(mu2, color=C2, linewidth=2, label=f"ランダム効果平均 {mu2:.3f}")
axes[1].axvspan(mu2-pi, mu2+pi, color=C2, alpha=0.12, label="次の 1 日の予測区間")
axes[1].set_xlabel("日次 β(1 秒地平)")
axes[1].set_ylabel("日数")
axes[1].set_title(f"日次 β の分布(τ={math.sqrt(tau2):.3f}・I²=99.7%)")
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_16_audit_inference.png")
print("ok")
