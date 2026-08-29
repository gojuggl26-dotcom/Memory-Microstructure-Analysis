"""L10 book_slope スプラインの図(43)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, C4 = "#3b6fd4", "#c2410c", "#0f766e", "#7c3aed"
CM, INK, GRID = "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False})

r = json.loads((D / "slope_spline.json").read_text(encoding="utf-8"))
HZ = ["1ms", "100ms", "1s", "3s", "5s", "10s", "15s"]
SEC = [0.001, 0.1, 1, 3, 5, 10, 15]
lin = [r[h]["linear"] for h in HZ]
spl = [r[h]["spline"] for h in HZ]
sig = [r[h]["p"] < 0.05 for h in HZ]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ① 地平ごとの R² — 1 秒に山
ax = axes[0]
ax.plot(SEC, lin, "s--", color=C2, lw=1.8, ms=6, label="線形")
ax.plot(SEC, spl, "o-", color=C1, lw=2.4, ms=7, label="スプライン")
for x, y_, s in zip(SEC, spl, sig):
    if s:
        ax.annotate("★", (x, y_), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=11, color=C1)
i = int(np.argmax(spl))
ax.annotate(f"山は 1 秒\n{spl[i]:.4f}", (SEC[i], spl[i]), textcoords="offset points",
            xytext=(26, -6), fontsize=9, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.set_xscale("log"); ax.set_xticks(SEC)
ax.set_xticklabels(HZ, fontsize=8)
ax.set_xlabel("予測地平"); ax.set_ylabel("標本外 R²(日次中央値)")
ax.set_title("① 予測力は 1 秒で最大\n1ms で 1/15、15s で 1/12", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ② 「地平を伸ばせば解決する」かの検証
ax = axes[1]
eff = [np.sqrt(max(s, 0)) * np.sqrt(t) for s, t in zip(spl, SEC)]
base = eff[HZ.index("1s")]
ax.plot(SEC, [e / base for e in eff], "o-", color=C3, lw=2.4, ms=7)
ax.axhline(1, color=C2, lw=1.6, ls="--", label="1 秒 = 1")
for x, e in zip(SEC, eff):
    ax.annotate(f"{e/base:.2f}", (x, e / base), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_xticks(SEC); ax.set_xticklabels(HZ, fontsize=8)
ax.set_ylim(0, 1.45)
ax.set_xlabel("予測地平"); ax.set_ylabel("√R² × √h(1 秒 = 1)")
ax.set_title("② 地平を伸ばしても実質は横ばい\n予測力の低下がリターン増を相殺", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ③ 推定された形状(1 秒地平)
ax = axes[2]
sh = r["shape"]
ax.plot(sh["grid_bid"], sh["f_bid"], "-", color=C1, lw=2.6, label="f_B(S^Bid)")
ax.plot(sh["grid_ask"], sh["f_ask"], "-", color=C2, lw=2.6, label="f_A(S^Ask)")
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("板の重心の距離 S [bp]"); ax.set_ylabel("f(S)  中心化 bp")
ax.set_title("③ 符号が逆でほぼ対称\n買い板が遠いと下落、売り板が遠いと上昇", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_43_slope_spline.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_43_slope_spline.png")
