"""キャンセル指標の図(21)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

p = json.loads((D/"cancel_grid_predictive.json").read_text(encoding="utf-8"))["predictive_next_window"]
g = json.loads((D/"cancel_grid_summary.json").read_text(encoding="utf-8"))
STEPS = ["1ms","10ms","100ms","1s","10s","60s"]; X = [1,10,100,1000,10000,60000]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for c, col, lab in (("cancel_diff_n", C1, "件数差 ask−bid"),
                    ("cancel_diff_vol", C2, "数量差 ask−bid"),
                    ("n_cancel_ask", C3, "ask 件数(単独)"),
                    ("n_cancel_bid", CM, "bid 件数(単独)")):
    y = [p.get(f"{c}|{s}", {}).get("median") for s in STEPS]
    axes[0].plot(X, y, color=col, linewidth=2, marker="o", markersize=5, label=lab)
axes[0].axhline(0, color=INK, linewidth=0.9)
axes[0].set_xscale("log"); axes[0].set_xticks(X, STEPS)
axes[0].set_xlabel("窓幅"); axes[0].set_ylabel("次窓リターンとの相関")
axes[0].set_title("100ms が最良 — 最狭の 1ms/10ms は情報ゼロ")
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(X, [g[s]["cancels_per_window"] for s in STEPS], color=C1,
             linewidth=2, marker="o", markersize=5)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xticks(X, STEPS)
axes[1].set_xlabel("窓幅"); axes[1].set_ylabel("1 窓あたりのキャンセル件数(対数)")
axes[1].set_title("1ms と 10ms が同値 = キャンセルは束で入る")
fig.savefig(CH / "xyz_DRAM_21_cancel_grid.png")
print("ok")
