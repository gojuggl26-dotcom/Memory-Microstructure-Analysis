"""項目 2・3 の図(32)。"""
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

fp = json.loads((D/"fill_prob_model.json").read_text(encoding="utf-8"))
bt = json.loads((D/"backtest_results.json").read_text(encoding="utf-8"))
t = pl.read_csv(D/"backtest_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (1) 実測の約定率(キュー位置 × 置き場所)
tb = fp["empirical_table"]
dists = ["内側","最良気配","1bp 外","1-5bp 外"]
qs = ["0(先頭)","小","中","大"]
M = np.full((len(qs), len(dists)), np.nan)
for r in tb:
    M[qs.index(r["q_ahead"]), dists.index(r["dist"])] = r["fill_rate"]
im = axes[0].imshow(M, cmap="Blues", aspect="auto")
axes[0].set_xticks(range(len(dists)), dists, rotation=20)
axes[0].set_yticks(range(len(qs)), qs)
axes[0].set_xlabel("置き場所"); axes[0].set_ylabel("自分の前の数量")
axes[0].grid(False)
for i in range(len(qs)):
    for j in range(len(dists)):
        if np.isfinite(M[i,j]):
            axes[0].text(j, i, f"{M[i,j]*100:.1f}%", ha="center", va="center", fontsize=8,
                         color="white" if M[i,j] > 0.12 else INK)
axes[0].set_title("実測の約定率(標本外)")

# (2) 較正
cal = fp["calibration_oos"]
axes[1].plot([c["pred"] for c in cal], [c["actual"] for c in cal], marker="o",
             color=C1, linewidth=2, label="モデル")
lim = max(max(c["pred"] for c in cal), max(c["actual"] for c in cal)) * 1.05
axes[1].plot([0, lim], [0, lim], color=CM, linestyle="--", linewidth=1.5, label="完全較正")
axes[1].set_xlabel("予測約定率"); axes[1].set_ylabel("実測約定率")
axes[1].set_title(f"標本外 AUC {fp['auc_oos_daily']['median']:.3f}・較正は要調整")
axes[1].legend(frameon=False, fontsize=8)

# (3) バックテスト: 1 約定あたり vs 1 日あたり
x = np.arange(2)
axes[2].bar(x - 0.2, [bt["always"]["mean_of_daily"], bt["gated"]["mean_of_daily"]],
            width=0.38, color=C1, label="1 約定あたり (bp)")
ax2 = axes[2].twinx()
ax2.bar(x + 0.2, [bt["always"]["total_bp_per_day"], bt["gated"]["total_bp_per_day"]],
        width=0.38, color=C2, label="1 日あたり合計 (bp)")
ax2.grid(False)
axes[2].set_xticks(x, ["常に出す", f"選別\n(約定 {bt['fill_retention']:.0%} 維持)"])
axes[2].set_ylabel("1 約定あたり (bp)", color=C1)
ax2.set_ylabel("1 日あたり合計 (bp)", color=C2)
axes[2].set_title("単価は +68% だが総額は −8.5%")
for i, v in enumerate([bt["always"]["mean_of_daily"], bt["gated"]["mean_of_daily"]]):
    axes[2].text(i - 0.2, v + 0.004, f"{v:+.4f}", ha="center", fontsize=8, color=INK)
for i, v in enumerate([bt["always"]["total_bp_per_day"], bt["gated"]["total_bp_per_day"]]):
    ax2.text(i + 0.2, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=INK)
fig.savefig(CH / "xyz_DRAM_32_fill_prob_backtest.png")
print("ok")
