"""キュー順位分析の図(35)。"""
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

r = json.loads((D / "queue_adverse_summary.json").read_text(encoding="utf-8"))
cf = json.loads((D / "queue_confound.json").read_text(encoding="utf-8"))
QO = ["0 (先頭)", "0<q<=20", "20<q<=100", "100<q<=400", "400<q<=1000", "q>1000"]
SH = ["0", "1-20", "21-100", "101-400", "401-1k", ">1k"]
DO = ["内側", "最良気配", "0-1bp", "1-5bp"]
HZ = ["100ms", "1s", "10s", "60s"]
fp, pf, st = r["fill_prob"], r["post_fill"], cf["stratified"]
pres = [q for q in QO if f"最良気配のみ|{q}" in pf]
xs = np.arange(len(pres)); lab = [SH[QO.index(q)] for q in pres]

fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))

# ① 約定確率
ax = axes[0, 0]
for db, c, mk in zip(DO, (C4, C1, C3, C2), ("D", "o", "s", "^")):
    v = [fp[f"{q}|{db}"]["median"] * 100 if f"{q}|{db}" in fp else np.nan for q in pres]
    ax.plot(xs, v, mk + "-", color=c, lw=2, ms=6, label=db)
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
ax.set_xlabel("自分の前にある数量(発注時点)"); ax.set_ylabel("約定確率 %(対数軸)")
ax.set_title("① 約定確率は素直に単調減少\n最良気配で 7.45% → 2.32%(3.2 倍)", fontsize=10)
ax.legend(frameon=False, fontsize=8, title="発注位置", title_fontsize=8)

# ② 約定後の平均価格変動
ax = axes[0, 1]
for h, c, mk in zip(HZ, (C4, C1, C3, C2), ("D", "o", "s", "^")):
    v = [pf[f"最良気配のみ|{q}"][f"adv_{h}"].get("median", np.nan) for q in pres]
    ax.plot(xs, v, mk + "-", color=c, lw=2, ms=6, label=h)
ax.axhline(0, color=INK, lw=1)
ax.axhspan(-0.6, 0, color=C2, alpha=.05, lw=0)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("約定後の中値変動 bp(下が不利)")
ax.set_title("② 逆選択は非単調\n中間層(101-1k)だけ有利", fontsize=10)
ax.legend(frameon=False, fontsize=8, title="経過時間", title_fontsize=8)

# ③ 分布(1 秒)
ax = axes[0, 2]
qs = ["5", "10", "25", "50", "75", "90", "95"]
bd = {p: np.array([pf[f"最良気配のみ|{q}"]["quantiles_1s"].get(p, np.nan) for q in pres])
      for p in qs}
ax.fill_between(xs, bd["5"], bd["95"], color=C1, alpha=.10, lw=0, label="p5-p95")
ax.fill_between(xs, bd["10"], bd["90"], color=C1, alpha=.17, lw=0, label="p10-p90")
ax.fill_between(xs, bd["25"], bd["75"], color=C1, alpha=.30, lw=0, label="p25-p75")
ax.plot(xs, bd["50"], "o-", color=C2, lw=2, ms=6, label="中央値")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("約定 1 秒後の中値変動 bp")
ax.set_title("③ 中央値はどこも 0 — 差は裾から\n深いほど分散が大きい(1.89→2.98)", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2)

# ④ 粗利 / 逆選択 / 正味 + 発注 1 件あたり期待値
ax = axes[1, 0]
g = np.array([pf[f"最良気配のみ|{q}"]["gross"]["median"] for q in pres])
a = np.array([pf[f"最良気配のみ|{q}"]["adv_1s"]["median"] for q in pres])
n = np.array([pf[f"最良気配のみ|{q}"]["net_1s"]["median"] for q in pres])
w = .27
ax.bar(xs - w, g, w, color=C3, label="粗利(半スプレッド)")
ax.bar(xs, a, w, color=C2, label="逆選択(1s)")
ax.bar(xs + w, n, w, color=C1, label="正味")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("bp / 約定(日次中央値)")
ax.set_title("④ 粗利 + 逆選択 = 正味", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ⑤⑥ 交絡の切り分け
for k, (by, tl, ttl) in enumerate((("spread", ("狭", "中", "広"), "発注時スプレッド"),
                                   ("depth", ("薄", "中", "厚"), "発注時の板の厚み"))):
    ax = axes[1, 1 + k]
    for i, (t, c, mk) in enumerate(zip(tl, (C4, C1, C2), ("D", "o", "^"))):
        v = [st[f"{by}|{q}|{i}"]["median"] if f"{by}|{q}|{i}" in st else np.nan for q in pres]
        ax.plot(xs, v, mk + "-", color=c, lw=2, ms=6, label=f"{t}い")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlabel("自分の前にある数量"); ax.set_ylabel("逆選択 1s bp")
    ax.set_title(f"{'⑤' if k == 0 else '⑥'} {ttl}を層内で固定しても\n"
                 f"非単調性は消えない(交絡ではない)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title=ttl, title_fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_35_queue_adverse.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_35_queue_adverse.png")
