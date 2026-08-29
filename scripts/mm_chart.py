"""マーケットメイク損益の分解と選別効果(図 30)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

d = json.loads((D/"mm_pnl_decomp.json").read_text(encoding="utf-8"))
s = json.loads((D/"mm_signal_selection.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (1) 損益分解(1s)
vals = [d["gross_capture_bp"]["mean"], -d["1s"]["adverse_bp"], -d["maker_fee_bp"]]
labs = ["粗いスプレッド\n獲得", "逆選択", "手数料"]
cum = np.cumsum([0] + vals)
for i, (v, l) in enumerate(zip(vals, labs)):
    axes[0].bar(i, v, bottom=cum[i], color=C1 if v > 0 else C2, width=0.6)
    axes[0].text(i, cum[i] + v / 2, f"{v:+.3f}", ha="center", va="center",
                 fontsize=9, color="white")
axes[0].bar(3, cum[-1], color=CM, width=0.6)
axes[0].text(3, cum[-1] + 0.02, f"{cum[-1]:+.4f}", ha="center", fontsize=10, color=INK)
axes[0].axhline(0, color=INK, linewidth=1)
axes[0].set_xticks(range(4), labs + ["正味"])
axes[0].set_ylabel("bp / 約定")
axes[0].set_title("素朴なメイクは実質ゼロ(1 秒地平)")

# (2) 整合度十分位ごとの損益
for h, col in (("1s", C1), ("10s", C3)):
    dec = s[h]["deciles"]
    axes[1].plot([x["decile"] for x in dec], [x["net"] for x in dec],
                 marker="o", linewidth=2, color=col, label=f"{h} 後")
axes[1].axhline(0, color=INK, linewidth=1)
axes[1].set_xticks(range(1, 11))
axes[1].set_xlabel("シグナルとの整合度(十分位)")
axes[1].set_ylabel("手数料後の損益 (bp/約定)")
axes[1].set_title("不利な約定を避けられれば黒字化する")
axes[1].legend(frameon=False, fontsize=8)

# (3) 選別の強さと損益
ks = ["all", "top50pct", "top30pct", "top10pct"]
lab = ["全約定", "上位 50%", "上位 30%", "上位 10%"]
for h, col, off in (("1s", C1, -0.2), ("10s", C3, 0.2)):
    v = [s[h][k]["net"] for k in ks]
    axes[2].bar(np.arange(4) + off, v, width=0.38, color=col, label=f"{h} 後")
axes[2].axhline(0, color=INK, linewidth=1)
axes[2].set_xticks(range(4), lab)
axes[2].set_ylabel("手数料後の損益 (bp/約定)")
axes[2].set_title("選別を強めるほど利幅は増えるが約定数は減る")
axes[2].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_30_market_making.png")
print("ok")
