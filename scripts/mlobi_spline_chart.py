"""加法スプラインの図(42)。"""
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

r = json.loads((D / "mlobi_spline.json").read_text(encoding="utf-8"))
g = np.array(r["shape_grid"]); SH = r["shapes"]
ks = [k for k in r if k.startswith("スプライン")]
kn = [int(k.split("=")[1]) for k in ks]
o = np.argsort(kn); ks = [ks[i] for i in o]; kn = [kn[i] for i in o]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ① 節点数と標本外 R²(飽和曲線)
ax = axes[0]
rel = [r[k]["rel"] for k in ks]
pos = [r[k]["pos"] for k in ks]; nn = r[ks[0]]["n"]
ax.plot(kn, rel, "o-", color=C1, lw=2, ms=7)
ax.axhline(1, color=C2, lw=1.6, ls="--", label="線形(10 変数)= 1")
for x, y_, p_ in zip(kn, rel, pos):
    ax.annotate(f"{y_:.3f}\n{p_}/{nn}", (x, y_), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5, color=INK)
ax.set_xscale("log"); ax.set_xticks(kn); ax.set_xticklabels(kn)
ax.set_ylim(0.97, 1.21)
ax.set_xlabel("内部節点の数 K(対数軸)"); ax.set_ylabel("標本外 R²(線形 = 1)")
ax.set_title("① K=5〜8 で飽和\nそれ以上は母数が増えるだけ", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ② 推定された形状 f_L
ax = axes[1]
for L, c, lw in ((1, C1, 2.4), (2, C3, 2.0), (3, C4, 1.6), (5, C2, 1.6), (10, CM, 1.6)):
    ax.plot(g, SH[f"L{L}"], "-", color=c, lw=lw, label=f"L{L}")
ax.axhline(0, color=INK, lw=1); ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("そのレベルの不均衡 I_L"); ax.set_ylabel("f_L(I_L)  中心化 bp")
ax.set_title(f"② 推定された曲線(K={r['knots_best']})\n深い段ほど平ら", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2)

# ③ L1 の非線形性 — 線形との差
ax = axes[2]
f1 = np.array(SH["L1"])
sl = np.polyfit(g, f1, 1)
lin = np.polyval(sl, g)
ax.plot(g, f1, "-", color=C1, lw=2.6, label="スプライン f₁")
ax.plot(g, lin, "--", color=C2, lw=1.8, label="線形当てはめ")
ax.fill_between(g, lin, f1, where=(f1 >= lin), color=C1, alpha=.14, lw=0)
ax.fill_between(g, lin, f1, where=(f1 < lin), color=C2, alpha=.14, lw=0)
ax.axhline(0, color=INK, lw=1); ax.axvline(0, color=INK, lw=1)
ax.annotate("両端で傾きが急\n= 極端な不均衡ほど効く", (0.62, f1[16]),
            textcoords="offset points", xytext=(-100, -34), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.set_xlabel("最良気配の不均衡 I₁"); ax.set_ylabel("f₁(I₁)  中心化 bp")
ax.set_title("③ 1 段目は凸 — 裾で加速する\n線形では取り切れない部分", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_42_mlobi_spline.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_42_mlobi_spline.png")
