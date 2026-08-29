"""99 日・絶対量閾値での層別評価の図(39)。"""
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

J = {r: json.loads((D / f"pull_rule_strata_{r}.json").read_text(encoding="utf-8"))
     for r in ("all", "launch", "mature")}
QL = ["q<=20", "20<q<=100", "100<q<=400", "q>400"]
SHQ = ["≤20", "21-100", "101-400", ">400"]

def get(reg, lab):
    return next((x for x in J[reg]["strata"] if x["label"] == lab), None)

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

# ① 発火率: 割合 vs 絶対量(今回の変更の効果)
ax = axes[0]
rows = [get("all", f"最良気配|{q}|両側") for q in QL]
rows = [r for r in rows if r]
xs = np.arange(len(rows))
pr = [r["best"]["pull_rate"] * 100 for r in rows]
isabs = [r["best"]["is_abs"] for r in rows]
ax.bar(xs, pr, color=[C1 if a else C2 for a in isabs], width=.6)
for x, p, a in zip(xs, pr, isabs):
    ax.text(x, p + 2, f"{p:.0f}%\n{'絶対' if a else '割合'}", ha="center", fontsize=8, color=INK)
ax.axhspan(10, 60, color=C3, alpha=.10, lw=0)
ax.text(len(xs) - .5, 35, "実務的な帯", ha="right", fontsize=8, color=C3)
ax.set_xticks(xs); ax.set_xticklabels([SHQ[QL.index(r["label"].split("|")[1])] for r in rows],
                                      fontsize=8.5)
ax.set_ylim(0, 100)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("引いた率 %")
ax.set_title("① 絶対量閾値で発火率が実務帯に入った\n(前版は 55〜91% に張り付き)", fontsize=10)

# ② 出す価値(a)と 引く価値(c)を分けて見る
ax = axes[1]
w = .38
b0 = [r["base_total_median"] for r in rows]
b1 = [r["best"]["gain_vs_base_median"] for r in rows]
ax.bar(xs - w/2, b0, w, color=C3, label="(a) 出す価値")
ax.bar(xs + w/2, b1, w, color=C1, label="(c) 引く価値")
ax.axhline(0, color=INK, lw=1.2)
for x, r in zip(xs, rows):
    if r["base_sign_p"] < .05:
        ax.text(x - w/2, r["base_total_median"] + 5, "●", ha="center", fontsize=11, color=C3)
    if r["best"]["gain_sign_p"] < .05:
        ax.text(x + w/2, r["best"]["gain_vs_base_median"] + 5, "★", ha="center",
                fontsize=12, color=C1)
ax.set_xticks(xs); ax.set_xticklabels([SHQ[QL.index(r["label"].split("|")[1])] for r in rows],
                                      fontsize=8.5)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("bp/日(日次中央値)")
ax.set_title("② 出す価値と引く価値は別物\n●=出す価値あり ★=引く価値あり", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ③ 体制別 — q>400 だけが両体制で成立
ax = axes[2]
regs = ["launch", "mature"]
sides = ["両側", "Bid", "Ask"]
xs3 = np.arange(len(sides)); w3 = .38
for i, (reg, c) in enumerate(zip(regs, (C4, C1))):
    v, p_ = [], []
    for sl in sides:
        r = get(reg, f"最良気配|q>400|{sl}")
        v.append(r["best"]["gain_vs_base_median"] if r else np.nan)
        p_.append(r["best"]["gain_sign_p"] if r else 1.0)
    ax.bar(xs3 + (i - .5) * w3, v, w3, color=c,
           label="立ち上げ期" if reg == "launch" else "成熟期")
    for x, y, pv in zip(xs3, v, p_):
        if np.isfinite(y) and pv < .05:
            ax.text(x + (i - .5) * w3, y + .8, "★", ha="center", fontsize=12, color=c)
ax.axhline(0, color=INK, lw=1.2)
ax.set_xticks(xs3); ax.set_xticklabels(sides, fontsize=9)
ax.set_xlabel("最良気配 × 前に 400 単位超"); ax.set_ylabel("引く価値 bp/日")
ax.set_title("③ q>400 は両体制・両サイドで成立\n(事前基準を満たす唯一の層)", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ④ 全 16 層: 引く価値の符号一貫性
ax = axes[3]
st = J["all"]["strata"]
lab = [x["label"].replace("|両側", "").replace("最良気配", "最良")
       .replace("100<q<=400", "101-400").replace("20<q<=100", "21-100")
       .replace("q<=20", "≤20").replace("q>400", ">400") for x in st]
frac = [x["best"]["gain_pos_days"] / x["best"]["n_days"] * 100 for x in st]
o = np.argsort(frac)
ys = np.arange(len(o))
ax.barh(ys, [frac[i] for i in o],
        color=[C1 if st[i]["best"]["gain_sign_p"] < .05 else CM for i in o], height=.66)
ax.axvline(50, color=C2, lw=1.6, ls="--", label="コイン投げ 50%")
for y, i in zip(ys, o):
    ax.text(frac[i] + 1, y, f"{st[i]['best']['gain_pos_days']}/{st[i]['best']['n_days']}",
            va="center", fontsize=7, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([lab[i] for i in o], fontsize=7.5)
ax.set_xlim(0, 100)
ax.set_xlabel("引く価値が正だった日の割合 %")
ax.set_title("④ 引く価値があるのは深い行列のみ\n他は 50% を下回る(引くと損)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_39_pull_rule_99.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_39_pull_rule_99.png")
