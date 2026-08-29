"""レイテンシの図(47)。報告 = latency_report.md に対応。

出所:
  ①② data/block_latency.json(99 日・更新間隔 5,355 万本、1/3 抽出で 1,882 万本)
  ③  latency_report.md §2 の表(成熟期 12 日・BBO 更新 429 万件)。
      この表は報告本文にのみ数値があるため転記している。値の出所は同報告 §2。
再生成: uv run python scripts/latency_chart.py
"""
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

b = json.loads((D / "block_latency.json").read_text(encoding="utf-8"))
q = {float(k): v for k, v in b["quantiles_ms"].items()}

# latency_report.md §2 の表(転記。ms)
BLIND = {
    "全更新":            dict(share=100.0, med=131.0, p90=400.2, p99=1813.9, p999=8538.6),
    "≥1 ティック":       dict(share=43.0,  med=118.3, p90=278.1, p99=1207.9, p999=5992.6),
    "≥2 ティック":       dict(share=31.2,  med=116.5, p90=272.9, p99=1199.8, p999=5605.0),
    "≥5 ティック":       dict(share=11.5,  med=110.4, p90=265.8, p99=1007.0, p999=4363.8),
}

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# ① 更新間隔の分位 — 中央値と裾の乖離
ax = axes[0, 0]
ks = sorted(q)
ax.plot(ks, [q[k] for k in ks], "o-", color=C1, lw=2, ms=5)
ax.set_yscale("log")
ax.axhline(65, color=C3, lw=1.2, ls="--")
ax.annotate("ブロック間隔の床 65〜70ms", (0, 65), xytext=(4, -14),
            textcoords="offset points", fontsize=8, color=C3)
for k in (50, 99, 99.9):
    ax.annotate(f"p{k:g} = {q[k]:,.0f}ms", (k, q[k]), xytext=(-10, 8),
                textcoords="offset points", ha="right", fontsize=9, fontweight="bold")
ax.set_xlabel("分位 %"); ax.set_ylabel("更新間隔 ms(対数軸)")
ax.set_title(f"① 更新間隔は中央値 {q[50]:.0f}ms でも p99 は {q[99]:,.0f}ms\n"
             f"平均 {b['mean_ms']:.0f}ms は裾に引きずられていて代表値にならない", fontsize=10)

# ② 空白の長さ別 — どれだけの時間がその状態か
ax = axes[0, 1]
gk = sorted(int(k) for k in b["gaps"])
share = [b["gaps"][str(k)]["share_of_time_pct"] for k in gk]
perday = [b["gaps"][str(k)]["per_day"] for k in gk]
ax.plot(gk, share, "o-", color=C2, lw=2, ms=6, label="その閾値以上が占める時間の割合 %")
ax.set_xscale("log"); ax.set_xlabel("空白の長さの閾値 ms(対数軸)")
ax.set_ylabel("時間シェア %", color=C2); ax.tick_params(axis="y", labelcolor=C2)
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(gk, perday, "s--", color=C1, lw=1.6, ms=5)
ax2.set_yscale("log"); ax2.set_ylabel("発生回数 / 日(対数軸)", color=C1)
ax2.tick_params(axis="y", labelcolor=C1); ax2.spines["right"].set_visible(True)
ax.set_title(f"② 200ms 以上の空白が時間の {share[0]:.0f}% を占める\n"
             f"1 秒以上でも {b['gaps']['1000']['share_of_time_pct']:.0f}%"
             f"({b['gaps']['1000']['per_day']:,.0f} 回/日)", fontsize=10)

# ③ ★本題 — 価格が動く直前に反応できなかった時間
ax = axes[1, 0]
names = list(BLIND)
xs = np.arange(len(names))
w = 0.2
for i, (key, lab, c) in enumerate((("med", "中央値", C3), ("p90", "p90", C1),
                                   ("p99", "p99", C2), ("p999", "p99.9", C4))):
    ax.bar(xs + (i - 1.5) * w, [BLIND[n][key] for n in names], width=w, color=c, label=lab)
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("反応できなかった時間 ms(対数軸)")
ax.legend(frameon=False, fontsize=8, ncol=4)
ax.set_title("③ ★効くのは裾 — 値動き直前の空白時間\n"
             f"1 ティック動く直前、中央値 {BLIND['≥1 ティック']['med']:.0f}ms に対し "
             f"p99 は {BLIND['≥1 ティック']['p99']:,.0f}ms", fontsize=10)

# ④ 中央値だけ見ると何を見落とすか
ax = axes[1, 1]
k1 = BLIND["≥1 ティック"]
vals = [k1["med"], k1["p90"], k1["p99"], k1["p999"]]
labs = ["中央値", "p90", "p99", "p99.9"]
ax.barh(np.arange(4), vals, color=[C3, C1, C2, C4], height=0.55)
for i, v in enumerate(vals):
    ax.annotate(f"{v:,.0f} ms", (v, i), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="bold")
ax.set_yticks(np.arange(4)); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlim(50, 20000)
ax.set_xlabel("反応できなかった時間 ms(対数軸)")
ax.axvline(500, color=CM, lw=1.6, ls="--")
ax.annotate("報告44 の信号地平 500ms", (500, -0.45), xytext=(6, 0),
            textcoords="offset points", fontsize=8, color=CM)
ax.set_title(f"④ 100 回に 1 回は {k1['p99']/1000:.1f} 秒動けない\n"
             f"中央値なら信号地平 500ms に間に合うが、p99 では超える", fontsize=10)

fig.suptitle(f"Hyperliquid のレイテンシ — 更新間隔 {b['n_days']} 日 "
             f"{b['n_intervals_total']:,} 本 / 空白時間は成熟期 12 日 429 万件",
             fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = CH / "xyz_DRAM_47_latency_tail.png"
fig.savefig(out, facecolor="white")
print("保存:", out)
