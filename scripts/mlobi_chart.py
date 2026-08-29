"""MLOBI の図(41)。"""
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

r = json.loads((D / "mlobi.json").read_text(encoding="utf-8"))
W = r["weights_full_sample"]
ORD = ["OBI(1) 単独", "累積 OBI(5)", "累積 OBI(10)", "w1 均等",
       "w5 指数減衰", "w2 単変量β", "w3 重回帰OLS", "w4 Ridge"]
x = np.arange(1, 11)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ① 標本外 R² の比
ax = axes[0]
vals = [r[f"A|{c}"]["rel"] for c in ORD]
sig = [r[f"A|{c}"]["p"] < 0.05 for c in ORD]
ys = np.arange(len(ORD))
ax.barh(ys, vals, color=[C1 if s else (C2 if v < 1 else CM) for v, s in zip(vals, sig)],
        height=.66)
ax.axvline(1, color=INK, lw=1.6, ls="--", label="OBI(1) 単独 = 1")
for y_, c, v in zip(ys, ORD, vals):
    e = r[f"A|{c}"]
    ax.text(v + .02, y_, f"{v:.3f}  {e['pos']}/{e['n']}", va="center", fontsize=8, color=INK)
ax.set_yticks(ys); ax.set_yticklabels(ORD, fontsize=8.5)
ax.set_xlim(0, 1.62)
ax.set_xlabel("標本外 R²(OBI(1) を 1 とした比)")
ax.set_title("① 重み付けで 1.31 倍\n均等重みは OBI(1) より悪い", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")

# ② 推定された重み
ax = axes[1]
for nm, c, mk in (("w3 重回帰OLS", C1, "o"), ("w4 Ridge", C3, "s"),
                  ("w2 単変量β", C4, "^"), ("w5 指数減衰", C2, "D")):
    w = np.array(W[nm]); w = w / np.abs(w).sum()
    ax.plot(x, w, mk + "-", color=c, lw=2, ms=6, label=nm)
ax.axhline(0.1, color=CM, lw=1.2, ls=":", label="均等(0.1)")
ax.set_xticks(x)
ax.set_xlabel("最良気配からの段数"); ax.set_ylabel("重み(Σ|w|=1 に正規化)")
ax.set_title("② 1 段目に 31〜35% を配分\n指数減衰は立ち上がりが緩すぎる", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ③ 2 つの定義の比較
ax = axes[2]
nm = ["w1 均等", "w5 指数減衰", "w2 単変量β", "w3 重回帰OLS", "w4 Ridge"]
xs = np.arange(len(nm)); w_ = .38
a = [r[f"A|{c}"]["rel"] for c in nm]
b = [r[f"B|{c}"]["rel"] for c in nm]
ax.bar(xs - w_/2, a, w_, color=C1, label="A 不均衡加重 Σw·I_L")
ax.bar(xs + w_/2, b, w_, color=C2, label="B 数量加重(標準形)")
ax.axhline(1, color=INK, lw=1.4, ls="--")
for i, c in enumerate(nm):
    if r[f"A|{c}"]["p"] < .05:
        ax.text(i - w_/2, a[i] + .03, "★", ha="center", fontsize=12, color=C1)
    if r[f"B|{c}"]["p"] < .05:
        ax.text(i + w_/2, b[i] + .03, "★", ha="center", fontsize=12, color=C2)
ax.set_xticks(xs); ax.set_xticklabels(nm, fontsize=8, rotation=20, ha="right")
ax.set_ylabel("標本外 R²(OBI(1) = 1)")
ax.set_title("③ 不均衡加重が数量加重を上回る\n薄い深段も 1 票を持つほうが良い", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_41_mlobi.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_41_mlobi.png")
