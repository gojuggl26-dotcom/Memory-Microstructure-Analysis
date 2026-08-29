"""book_slope の粒度スイープ(図 28)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, CM, INK, GRID = "#3b6fd4", "#c2410c", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D/"book_slope_grain.json").read_text(encoding="utf-8"))
BANDS = ["5bp","10bp","25bp","100bp","all"]; HS = ["100ms","1s","5s","10s","60s"]
M = np.array([[abs(r[f"{b}|{h}"]["median"]) for h in HS] for b in BANDS])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
im = axes[0].imshow(M, cmap="Blues", aspect="auto")
axes[0].set_xticks(range(len(HS)), HS); axes[0].set_yticks(range(len(BANDS)), BANDS)
axes[0].set_xlabel("予測地平"); axes[0].set_ylabel("深さ帯(最良気配から)")
axes[0].grid(False)
for i in range(len(BANDS)):
    for j in range(len(HS)):
        axes[0].text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8,
                     color="white" if M[i,j] > 0.18 else INK)
axes[0].set_title("|相関| — 濃いほど強い")
fig.colorbar(im, ax=axes[0], shrink=0.85)

for b, col in zip(BANDS, [CM, "#94a3b8", C1, C2, "#0f766e"]):
    axes[1].plot(range(len(HS)), [abs(r[f"{b}|{h}"]["median"]) for h in HS],
                 marker="o", linewidth=2, color=col, label=b)
axes[1].set_xticks(range(len(HS)), HS)
axes[1].set_xlabel("予測地平"); axes[1].set_ylabel("|相関|(日次中央値)")
axes[1].set_title("地平は 1 秒が最良(どの深さ帯でも)")
axes[1].legend(frameon=False, fontsize=8, title="深さ帯")

v = [abs(r[f"{b}|1s"]["median"]) for b in BANDS]
axes[2].bar(BANDS, v, color=[CM, CM, C2, C1, C1], width=0.62)
axes[2].set_xlabel("含める深さ帯"); axes[2].set_ylabel("|相関|(1 秒地平)")
axes[2].set_title("25bp で頭打ち。5bp では足りない")
for i, x in enumerate(v):
    axes[2].text(i, x + 0.004, f"{x:.3f}", ha="center", fontsize=9, color=INK)
fig.savefig(CH / "xyz_DRAM_28_book_slope_grain.png")
print("ok")
