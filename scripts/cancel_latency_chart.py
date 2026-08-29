"""キャンセル猶予の検証(図 31)。"""
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

r = json.loads((D/"cancel_latency.json").read_text(encoding="utf-8"))
e = json.loads((D/"cancel_latency_extra.json").read_text(encoding="utf-8"))
LAGS = ["0ms","100ms","200ms","500ms","1s","2s","5s"]
X = [0.05, 100, 200, 500, 1000, 2000, 5000]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (1) δ 別の上乗せと 95% CI + プラセボ
up = [r[k]["uplift_median"] for k in LAGS]
lo = [r[k]["boot_ci95"][0] for k in LAGS]; hi = [r[k]["boot_ci95"][1] for k in LAGS]
axes[0].fill_between(X, lo, hi, color=C1, alpha=0.18, linewidth=0)
axes[0].plot(X, up, marker="o", color=C1, linewidth=2, label="選別の上乗せ")
p = r["placebo"]
axes[0].errorbar([7000], [p["uplift_median"]],
                 yerr=[[p["uplift_median"]-p["boot_ci95"][0]], [p["boot_ci95"][1]-p["uplift_median"]]],
                 fmt="s", color=C2, capsize=5, markersize=8, label="プラセボ(1 時間ずらし)")
axes[0].axhline(0, color=INK, linewidth=1)
axes[0].axvline(70, color=C3, linewidth=1.5, linestyle=":")
axes[0].text(78, max(up)*0.85, "ブロック間隔\n約 70ms", color=C3, fontsize=7.5)
axes[0].set_xscale("log")
axes[0].set_xticks(X + [7000], LAGS + ["プラセボ"], rotation=30, fontsize=7.5)
axes[0].set_ylabel("選別による損益の上乗せ (bp/約定)")
axes[0].set_xlabel("反応時間 δ")
axes[0].set_title("200ms までは明確に残る(帯は 95% CI)")
axes[0].legend(frameon=False, fontsize=8)

# (2) 警告時間の分布
w = r["warning_time_ms"]
qs = [10,25,50,75,90]; vs = [w[f"p{q}"] for q in qs]
axes[1].plot(vs, qs, marker="o", color=C1, linewidth=2)
axes[1].axvline(70, color=C3, linewidth=1.5, linestyle=":")
axes[1].text(80, 20, "ブロック間隔 70ms", color=C3, fontsize=8)
axes[1].set_xscale("log")
axes[1].set_xlabel("警告時間 (ms・対数)"); axes[1].set_ylabel("累積割合 (%)")
axes[1].set_title(f"警告時間の中央値 {w['p50']:.0f}ms(最短でも p10={w['p10']:.0f}ms)")
for q, v in zip(qs, vs):
    axes[1].annotate(f"{v:.0f}ms", (v, q), textcoords="offset points", xytext=(6,-3), fontsize=7.5)

# (3) 警告できた割合と回避できる損失
ks = ["0ms","100ms","200ms","500ms","1s"]
x = np.arange(len(ks))
axes[2].bar(x-0.2, [e[k]["warned_share"] for k in ks], width=0.38, color=C1, label="警告のあった不利約定")
axes[2].bar(x+0.2, [e[k]["loss_avoided_share"] for k in ks], width=0.38, color=C3, label="回避できる損失")
axes[2].set_xticks(x, ks); axes[2].set_ylim(0, 1)
axes[2].set_ylabel("割合"); axes[2].set_xlabel("反応時間 δ")
axes[2].set_title("100ms なら不利約定の 76%・損失の 82% を回避")
axes[2].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_31_cancel_latency.png")
print("ok")
