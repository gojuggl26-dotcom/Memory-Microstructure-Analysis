"""層別評価の図(38)。"""
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

r = json.loads((D / "pull_rule_strata.json").read_text(encoding="utf-8"))
rows = r["strata"]
best_q = [x for x in rows if x["label"].startswith("最良気配") and x["label"].endswith("両側")]
QL = ["q<=20", "20<q<=100", "100<q<=400", "q>400"]
SHQ = ["≤20", "21-100", "101-400", ">400"]

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

# ① 深さ別: 出し続ける vs 引く vs 出さない(最良気配・両側)
ax = axes[0]
xs = np.arange(len(best_q)); w = .38
b0 = [x["base_total_median"] for x in best_q]
b1 = [x["best"]["remaining_total_median"] for x in best_q]
ax.bar(xs - w/2, b0, w, color=C2, label="出し続ける")
ax.bar(xs + w/2, b1, w, color=C1, label="引く(最良セル)")
ax.axhline(0, color=INK, lw=1.6, ls="--")
ax.text(len(xs)-.4, 0, " 出さない", va="bottom", ha="right", fontsize=8.5, color=INK)
for x, v, e in zip(xs, b1, best_q):
    if e["best"]["sign_p_vs_zero"] < 0.05:
        ax.text(x + w/2, v + 25, "★", ha="center", fontsize=13, color=C1)
ax.set_xticks(xs)
ax.set_xticklabels([SHQ[QL.index(x["label"].split("|")[1])] for x in best_q], fontsize=8.5)
ax.set_xlabel("自分の前にある数量(発注時)"); ax.set_ylabel("1 日の合計損益 bp(中央値)")
ax.set_title("① 深さ別(最良気配・両側)\n★ = 出さないを有意に上回る", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ② 引いた率 — 浅いと撃ちすぎ、深いと撃たない
ax = axes[1]
pr = [x["best"]["pull_rate"] * 100 for x in best_q]
cols = [C1 if 20 <= p <= 70 else CM for p in pr]
ax.bar(xs, pr, color=cols, width=.6)
for x, p in zip(xs, pr):
    ax.text(x, p + 2, f"{p:.0f}%", ha="center", fontsize=8.5, color=INK)
ax.axhspan(20, 70, color=C3, alpha=.10, lw=0)
ax.text(len(xs)-.5, 45, "実務的な帯", ha="right", fontsize=8, color=C3)
ax.set_xticks(xs)
ax.set_xticklabels([SHQ[QL.index(x["label"].split("|")[1])] for x in best_q], fontsize=8.5)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("引いた率 %")
ax.set_ylim(0, 105)
ax.set_title("② 浅いと撃ちすぎ、深いと撃たない\n分母 q_ahead で正規化した副作用", fontsize=10)

# ③ Bid / Ask(最良気配)
ax = axes[2]
sides = ["両側", "Bid", "Ask"]
cols3 = [CM, C1, C2]
xs3 = np.arange(len(QL)); w3 = .26
for i, (sl, c) in enumerate(zip(sides, cols3)):
    v = []
    for q in QL:
        e = next((x for x in rows if x["label"] == f"最良気配|{q}|{sl}"), None)
        v.append(e["best"]["remaining_total_median"] if e else np.nan)
    ax.bar(xs3 + (i - 1) * w3, v, w3, color=c, label=sl)
    for x, y, q in zip(xs3, v, QL):
        e = next((z for z in rows if z["label"] == f"最良気配|{q}|{sl}"), None)
        if e and e["best"]["sign_p_vs_zero"] < 0.05 and np.isfinite(y):
            ax.text(x + (i - 1) * w3, y + 8, "★", ha="center", fontsize=11, color=c)
ax.axhline(0, color=INK, lw=1.6, ls="--")
ax.set_xticks(xs3); ax.set_xticklabels(SHQ, fontsize=8.5)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("引いた後の合計損益 bp/日")
ax.set_title("③ Bid / Ask の非対称\nAsk が一貫して強い", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ④ ランダム対照を突破した日数 — p 値より強い証拠
ax = axes[3]
lab = [x["label"].replace("|両側", "").replace("最良気配", "最良").replace("100<q<=400", "101-400")
       .replace("20<q<=100", "21-100").replace("q<=20", "≤20").replace("q>400", ">400")
       for x in rows]
bd = [x["best"]["rand_beat_days"] for x in rows]
nd = [x["best"]["n_days"] for x in rows]
o = np.argsort(bd)
ys = np.arange(len(o))
ax.barh(ys, [bd[i] for i in o],
        color=[C1 if rows[i]["best"]["sign_p_vs_zero"] < .05 else CM for i in o], height=.66)
exp = np.mean([n * 0.05 for n in nd])
ax.axvline(exp, color=C2, lw=1.6, ls="--", label=f"偶然の期待値 ≈{exp:.1f} 日")
for y, i in zip(ys, o):
    ax.text(bd[i] + .3, y, f"/{nd[i]}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([lab[i] for i in o], fontsize=7.5)
ax.set_xlabel("ランダム対照の 95% 点を超えた日数")
ax.set_title("④ 選別が情報を持つ最強の証拠\n(多重比較で p 値は使えないため)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_38_pull_rule_strata.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_38_pull_rule_strata.png")
