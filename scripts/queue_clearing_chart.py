"""行列の消え方と逆選択の図(36)。"""
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

s = json.loads((D / "queue_clearing_summary.json").read_text(encoding="utf-8"))
x = json.loads((D / "queue_clearing_cross.json").read_text(encoding="utf-8"))
CB = ["-1.0〜-0.8 (ほぼ全部取消)", "-0.8〜-0.4", "-0.4〜0.0",
      "0.0〜0.4", "0.4〜0.8", "0.8〜1.0 (ほぼ全部約定)"]
SH = ["全部\n取消", "-0.8〜\n-0.4", "-0.4〜\n0.0", "0.0〜\n0.4", "0.4〜\n0.8", "全部\n約定"]
HZ = ["100ms", "1s", "10s", "60s"]
QB = ["q<=20", "20<q<=100", "100<q<=400", "q>400"]
CBK = ["全部取消", "取消優勢", "約定優勢", "全部約定"]
ag = s["buckets"]
pres = [b for b in CB if f"最良気配のみ|{b}" in ag]
xs = np.arange(len(pres)); lab = [SH[CB.index(b)] for b in pres]

fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8))

# ① 逆選択 × 地平
ax = axes[0, 0]
for h, c, mk in zip(HZ, (C4, C1, C3, C2), ("D", "o", "s", "^")):
    v = [ag[f"最良気配のみ|{b}"][f"adv_{h}"].get("median", np.nan) for b in pres]
    ax.plot(xs, v, mk + "-", color=c, lw=2, ms=6, label=h)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7.5)
ax.set_xlabel("clear_ratio(前の行列がどう消えたか)")
ax.set_ylabel("約定後の中値変動 bp(下が不利)")
ax.set_title("① 取消で消えたときが最も毒性が高い\n(事前の予測は逆だった)", fontsize=10)
ax.legend(frameon=False, fontsize=8, title="経過時間", title_fontsize=8)

# ② 件数の分布
ax = axes[0, 1]
n = np.array([ag[f"最良気配のみ|{b}"]["n"] for b in pres], dtype=float)
cols = [C2 if i == 0 else (C3 if i == len(pres) - 1 else CM) for i in range(len(pres))]
ax.bar(xs, n / n.sum() * 100, color=cols, width=.66)
for i, v in enumerate(n / n.sum() * 100):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7.5)
ax.set_xlabel("clear_ratio"); ax.set_ylabel("約定に占める割合 %")
ax.set_title(f"② 最も毒性が高い形が最も多い\n前の行列の {100-s['exec_share_overall']*100:.0f}% は取消で消える",
             fontsize=10)

# ③ 正味損益
ax = axes[0, 2]
g = np.array([ag[f"最良気配のみ|{b}"]["gross"]["median"] for b in pres])
a = np.array([ag[f"最良気配のみ|{b}"]["adv_1s"]["median"] for b in pres])
nt = np.array([ag[f"最良気配のみ|{b}"]["net_1s"]["median"] for b in pres])
w = .27
ax.bar(xs - w, g, w, color=C3, label="粗利")
ax.bar(xs, a, w, color=C2, label="逆選択(1s)")
ax.bar(xs + w, nt, w, color=C1, label="正味")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7.5)
ax.set_xlabel("clear_ratio"); ax.set_ylabel("bp / 約定")
ax.set_title("③ 粗利も最低 — 約定時点で\n既に中値が寄ってきている", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ④ 2 元表(ヒートマップ)
ax = axes[1, 0]
M = np.full((len(QB), len(CBK)), np.nan)
for i, q in enumerate(QB):
    for j, c in enumerate(CBK):
        e = x["cells"].get(f"{q}|{c}")
        if e:
            M[i, j] = e["median"]
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
for i in range(len(QB)):
    for j in range(len(CBK)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M[i, j]) > vmax * .55 else INK)
ax.set_xticks(range(len(CBK))); ax.set_xticklabels(CBK, fontsize=8)
ax.set_yticks(range(len(QB))); ax.set_yticklabels(QB, fontsize=8)
ax.set_xlabel("clear_ratio"); ax.set_ylabel("自分の前の数量")
ax.set_title("④ 層内でも取消側が不利\n(交絡ではない)", fontsize=10)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=.046, label="逆選択 1s bp")

# ⑤ 偏回帰の係数
ax = axes[1, 1]
pr = x["partial_regression"]
ks = ["clear_ratio", "log_q_ahead"]
nm = ["clear_ratio\n(消え方)", "log(1+q_ahead)\n(行列の深さ)"]
vals = [pr[k]["median"] for k in ks]
cols2 = [C1 if pr[k]["sign_p"] < 0.05 else CM for k in ks]
ax.barh(np.arange(2), vals, color=cols2, height=.5)
for i, k in enumerate(ks):
    e = pr[k]
    ax.text(vals[i] + 0.004, i, f"{e['pos_days']}/{e['n_days']} 日  p={e['sign_p']:.4f}",
            va="center", fontsize=8.5, color=INK)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(np.arange(2)); ax.set_yticklabels(nm, fontsize=8.5)
ax.set_xlabel("偏回帰係数(日次中央値)")
ax.set_xlim(0, max(vals) * 1.85)
ax.set_title("⑤ 統制すると深さの効果は消える\nadv = α + β₁·clear + β₂·log q", fontsize=10)

# ⑥ 正味損益の 2 元表
ax = axes[1, 2]
N = np.full((len(QB), len(CBK)), np.nan)
for i, q in enumerate(QB):
    for j, c in enumerate(CBK):
        e = x["cells"].get(f"{q}|{c}")
        if e and np.isfinite(e.get("net", np.nan)):
            N[i, j] = e["net"]
vm = np.nanmax(np.abs(N))
im2 = ax.imshow(N, cmap="RdBu", vmin=-vm, vmax=vm, aspect="auto")
for i in range(len(QB)):
    for j in range(len(CBK)):
        if np.isfinite(N[i, j]):
            ax.text(j, i, f"{N[i,j]:+.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(N[i, j]) > vm * .55 else INK)
ax.set_xticks(range(len(CBK))); ax.set_xticklabels(CBK, fontsize=8)
ax.set_yticks(range(len(QB))); ax.set_yticklabels(QB, fontsize=8)
ax.set_xlabel("clear_ratio"); ax.set_ylabel("自分の前の数量")
ax.set_title("⑥ 正味損益(粗利 + 逆選択)\n深い行列は粗利で救われる", fontsize=10)
ax.grid(False)
fig.colorbar(im2, ax=ax, fraction=.046, label="正味 bp / 約定")

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_36_queue_clearing.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_36_queue_clearing.png")
