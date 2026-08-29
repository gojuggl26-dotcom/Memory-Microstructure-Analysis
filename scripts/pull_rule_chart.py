"""前方取消フローによる引き規則の図(37)。"""
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

r = json.loads((D / "pull_rule_eval.json").read_text(encoding="utf-8"))
cells = r["cells"]
SIG = {"cancel": (C1, "o", "取消フロー"), "exec": (C2, "^", "約定フロー"),
       "total": (C3, "s", "合計フロー")}
nd = r["n_days"]

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3))

# ① lift 対 引いた率 — 格子 180 通りを全部出す
ax = axes[0]
for sn, (c, mk, lab) in SIG.items():
    sub = [x for x in cells.values() if x["signal"] == sn and x["delta_ms"] == 100]
    if not sub:
        continue
    ax.scatter([x["pull_rate"] * 100 for x in sub], [x["lift_median"] for x in sub],
               c=c, marker=mk, s=34, alpha=.75, label=lab, edgecolors="none")
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("引いた率 %"); ax.set_ylabel("1 約定あたりの改善 bp(日次中央値)")
ax.set_title("① 格子 180 通りを全部提示\n(δ=100ms の 60 点)", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ② ★総額での比較 — 「そもそも出さない」が基準線
ax = axes[1]
names, vals, cols = ["出し続ける"], [r["baseline_total_median"]], [C2]
for sn, (c, mk, lab) in SIG.items():
    v = r.get("vs_no_quote", {}).get(sn)
    if v:
        names.append(f"引く\n({lab})"); vals.append(v["remaining_total_median"]); cols.append(c)
xs = np.arange(len(names))
ax.bar(xs, vals, color=cols, width=.62)
ax.axhline(0, color=INK, lw=1.6, ls="--")
ax.text(len(names) - .5, 0, " 出さない = 0", va="bottom", ha="right", fontsize=9, color=INK)
for x, v in zip(xs, vals):
    ax.text(x, v + (30 if v >= 0 else -30), f"{v:+.0f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("1 日の合計損益 bp(中央値)")
ax.set_title("② ★引いても『出さない』に届かない\n損失の 99% は消せるが 0 は超えない", fontsize=10)

# ③ 反応時間 δ の影響
ax = axes[2]
for sn, (c, mk, lab) in SIG.items():
    xs2, ys2 = [], []
    for dl in (0, 100, 200):
        sub = [x for x in cells.values() if x["signal"] == sn and x["delta_ms"] == dl]
        if sub:
            xs2.append(dl); ys2.append(max(x["lift_median"] for x in sub))
    if xs2:
        ax.plot(xs2, ys2, mk + "-", color=c, lw=2, ms=7, label=lab)
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("反応時間 δ(ms)"); ax.set_ylabel("最良セルの改善 bp")
ax.set_title("③ 200ms でも劣化しない\n判別は水準でなく発火の早さ", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ④ ランダム対照を上回った日数
ax = axes[3]
labs, v1, v2 = [], [], []
for sn, (c, mk, lab) in SIG.items():
    sub = [x for x in cells.values() if x["signal"] == sn and x["delta_ms"] == 100]
    if not sub:
        continue
    b = max(sub, key=lambda x: x["lift_median"])
    labs.append(lab); v1.append(b["rand_beat_days"]); v2.append(b["n_days"])
xs = np.arange(len(labs)); w = .55
ax.bar(xs, v1, w, color=[SIG[s][0] for s in SIG if any(
    x["signal"] == s and x["delta_ms"] == 100 for x in cells.values())])
ax.axhline(nd * 0.05, color=C2, lw=1.6, ls="--", label=f"偶然の期待値 {nd*0.05:.1f} 日")
for x, a, b in zip(xs, v1, v2):
    ax.text(x, a + .4, f"{a}/{b}", ha="center", fontsize=9, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("ランダム対照の 95% 点を超えた日数")
ax.set_title("④ 選別は情報を持つ(取消・合計)\n約定フロー単独はほぼ無力", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_37_pull_rule.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_37_pull_rule.png")
