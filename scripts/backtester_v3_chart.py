"""バックテスタ v3 の図(44)。"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, C4 = "#3b6fd4", "#c2410c", "#0f766e", "#7c3aed"
CM, INK, GRID = "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False})

t = pl.read_csv(D / "backtest_v3.csv")
ORD = ["A+B 基準", "A+B+C 前方フロー", "A+B+D ゲート-1.0", "A+B+D ゲート-0.3",
       "A+B+D ゲート-0.1", "A+B+D ゲート0", "A+B+C+D ゲート-0.3"]
ORD = [p for p in ORD if p in t["policy"].unique().to_list()]
SH = [p.replace("A+B", "").replace(" 基準", "基準").replace("+C 前方フロー", "+C")
       .replace("+C+D ゲート", "+C+D ") .replace("+D ゲート", "+D ") for p in ORD]
piv = {p: t.filter(pl.col("policy") == p).sort("dt")["pnl_bp"].to_numpy() for p in ORD}
b = piv[ORD[0]]; n = len(b)

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

# ① 基準との対応差
ax = axes[0]
ys = np.arange(len(ORD))
d = [np.median(piv[p] - b) for p in ORD]
pos = [int(((piv[p] - b) > 0).sum()) for p in ORD]
ax.barh(ys, d, color=[CM if p == ORD[0] else (C1 if v > 0 else C2) for p, v in zip(ORD, d)],
        height=.66)
ax.axvline(0, color=INK, lw=1.6, ls="--")
for y_, v, q in zip(ys, d, pos):
    ax.text(v + (25 if v >= 0 else -25), y_, f"{q}/{n}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8, color=INK)
ax.set_yticks(ys); ax.set_yticklabels(SH, fontsize=8.5)
ax.set_xlabel("基準との差 bp/日(日次中央値)")
ax.set_title("① どの追加施策も基準を超えない\n(右の数字は改善した日数)", fontsize=10)

# ② ゲートの強さ: 単価 vs 総額
ax = axes[1]
G = ["A+B 基準", "A+B+D ゲート-1.0", "A+B+D ゲート-0.3", "A+B+D ゲート-0.1", "A+B+D ゲート0"]
G = [p for p in G if p in ORD]
lab = ["なし", "−1.0", "−0.3", "−0.1", "0"][:len(G)]
xs = np.arange(len(G))
per = [t.filter(pl.col("policy") == p)["per_fill"].median() for p in G]
tot = [np.median(piv[p]) for p in G]
ax.bar(xs - .2, np.array(per) / max(per) * 100, .38, color=C3, label="単価(最大=100)")
ax.bar(xs + .2, np.array(tot) / max(tot) * 100, .38, color=C1, label="合計(最大=100)")
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8.5)
ax.set_xlabel("予測ゲートの閾値(左ほど緩い)"); ax.set_ylabel("相対値 %")
ax.set_title("② 締めるほど単価は上がり総額は落ちる\n両側の往復を失うため", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ③ 損益の分解
ax = axes[2]
P3 = [p for p in ["A+B 基準", "A+B+C 前方フロー", "A+B+D ゲート0"] if p in ORD]
xs = np.arange(len(P3)); w = .26
gr = [t.filter(pl.col("policy") == p)["gross_bp"].median() for p in P3]
ad = [t.filter(pl.col("policy") == p)["adv_bp"].median() for p in P3]
fe = [-0.088] * len(P3)
rl = [(t.filter(pl.col("policy") == p)["pnl_bp"]
       / t.filter(pl.col("policy") == p)["n_fill"]).median() for p in P3]
ax.bar(xs - 1.5 * w, gr, w, color=C3, label="粗利")
ax.bar(xs - .5 * w, ad, w, color=C2, label="逆選択(1s)")
ax.bar(xs + .5 * w, fe, w, color=C4, label="手数料")
ax.bar(xs + 1.5 * w, rl, w, color=C1, label="実現(往復)")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs)
ax.set_xticklabels([p.replace("A+B", "").strip() or "基準" for p in P3], fontsize=8.5)
ax.set_ylabel("bp / 約定(日次中央値)")
ax.set_title("③ 分解値と実現値の差 = 手仕舞いコスト\n0.35〜0.37 bp(実測)", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ④ 日次の損益分布
ax = axes[3]
P4 = [p for p in ["A+B 基準", "A+B+C 前方フロー", "A+B+D ゲート0"] if p in ORD]
bp = ax.boxplot([piv[p] for p in P4], vert=True, widths=.55, patch_artist=True,
                medianprops=dict(color=INK, lw=1.8),
                flierprops=dict(marker="o", ms=3, mfc=CM, mec="none"))
for patch, c in zip(bp["boxes"], (C1, C3, C2)):
    patch.set_facecolor(c); patch.set_alpha(.35); patch.set_edgecolor(c)
ax.axhline(0, color=C2, lw=1.6, ls="--", label="損益ゼロ")
ax.set_xticklabels([p.replace("A+B", "").strip() or "基準" for p in P4], fontsize=8.5)
ax.set_ylabel("1 日の合計損益 bp")
ax.set_title(f"④ 日次のばらつき({n} 日)\n基準は 19/26 日で黒字・日次SR 0.70", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_44_backtester_v3.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_44_backtester_v3.png")
