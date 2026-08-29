"""キュー内位置と QI の図(20)。"""
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

r = json.loads((D / "queue_qi_summary.json").read_text(encoding="utf-8"))
rows = r["by_queue_position"]
lbl = [str(x["b"]) for x in rows]
fr = [100 * x["fill_rate"] for x in rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
axes[0].bar(range(len(lbl)), fr, color=C1, width=0.65)
axes[0].set_xticks(range(len(lbl)), lbl, rotation=35, ha="right")
axes[0].set_ylabel("約定率(%)")
axes[0].set_xlabel("発注時に自分の前にあった数量 Q_ahead")
axes[0].set_title("キュー位置が後ろほど約定しにくい(先頭 1.58% → 1k–2.5k 0.30%)")
axes[0].annotate("深部は板が厚い局面に\n偏るため反転する", (8.5, fr[8]), xytext=(6.2, 1.25),
                 color=C2, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=C2, lw=1))

h = [x for x in r["qi_hist"] if x["b"] is not None]
b = np.array([x["b"] for x in h]); n = np.array([x["n"] for x in h], dtype=float)
f = np.array([x["filled"] for x in h], dtype=float)
axes[1].bar(b, (n - f) / n.sum() * 100, width=0.045, color=CM, label="取消で消滅")
axes[1].bar(b, f / n.sum() * 100, width=0.045, bottom=(n - f) / n.sum() * 100,
            color=C2, label="約定で消滅")
axes[1].set_xlabel("消滅時点の QI")
axes[1].set_ylabel("全注文に占める割合(%)")
axes[1].set_title("QI は両端に張り付く(±0.95 外側で 80%)")
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_20_queue_position_and_qi.png")
print("ok")
