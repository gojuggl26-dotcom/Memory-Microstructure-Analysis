"""SDE シミュレーションの図(45)。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, C4 = "#3b6fd4", "#c2410c", "#0f766e", "#7c3aed"
CM, INK, GRID = "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})

res = json.loads((D / "sde_results.json").read_text(encoding="utf-8"))
pz = np.load(D / "sde_paths.npz")
az = np.load(D / "sde_annual.npz")
dz = np.load(D / "sde_day_ensemble.npz")
CAP = 100.0

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

# ① 1,000 本の日中 equity 経路(ファンチャート)
ax = axes[0]
P = pz["path"]                       # (1000, 1441) 1 分刻み
h = np.arange(P.shape[1]) / 60.0
for lo, hi, a in [(1, 99, .12), (5, 95, .16), (25, 75, .22)]:
    ax.fill_between(h, np.percentile(P, lo, axis=0), np.percentile(P, hi, axis=0),
                    color=C1, alpha=a, lw=0)
ax.plot(h, np.median(P, axis=0), color=C1, lw=2.0, label="中央値")
for i in range(0, 1000, 140):
    ax.plot(h, P[i], color=INK, lw=.45, alpha=.35)
ax.axhline(0, color=INK, lw=1.2, ls="--")
ax.set_xlabel("時刻 [時]"); ax.set_ylabel("損益 [USD]")
ax.set_title("① 1,000 本の日中経路(SDE)", fontsize=10, loc="left")
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
ax.legend(frameon=False, fontsize=8, loc="upper left")

# ② 日次損益の分布: SDE vs 実測
ax = axes[1]
sde_fix = pz["pnl"]
sde_mix = dz["pnl"].ravel()
meas = dz["pnl_meas"]
bins = np.linspace(-16, 22, 46)
ax.hist(sde_mix, bins=bins, density=True, color=C1, alpha=.34,
        label=f"SDE 母数変動込み\n(sd {sde_mix.std(ddof=1):.2f})")
ax.hist(sde_fix, bins=bins, density=True, histtype="step", color=C3, lw=1.8,
        label=f"SDE 母数固定\n(sd {sde_fix.std(ddof=1):.2f})")
for v in meas:
    ax.plot([v, v], [0, 0.012], color=C2, lw=1.3, alpha=.85)
ax.plot([], [], color=C2, lw=1.3, label=f"実測 30 日\n(sd {meas.std(ddof=1):.2f})")
ax.axvline(0, color=INK, lw=1.2, ls="--")
ax.set_xlabel("日次損益 [USD]"); ax.set_ylabel("密度")
ax.set_title("② 分布の検証", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7.4, loc="upper right")

# ③ σ 応答曲線 — λ 固定(他条件同一)と λ∝σ(実測の結合)を並べる
ax = axes[2]
sw = res["sigma_sweep"]
s = np.array([w["sigma"] for w in sw]) * 100
ax.plot(s, [w["pnl_mean"] for w in sw], "o--", color=CM, lw=1.6, ms=3.4,
        label="E[PnL](λ 固定)")
jf = D / "sde_joint_sweep.json"
if jf.exists():
    jj = json.loads(jf.read_text(encoding="utf-8"))
    sj = np.array([w["sigma"] for w in jj["rows"]]) * 100
    pj = np.array([w["pnl_mean"] for w in jj["rows"]])
    ax.plot(sj, [w["gross"] for w in jj["rows"]], "o-", color=C3, lw=1.5, ms=3.0,
            label="スプレッド獲得")
    ax.plot(sj, [w["inv"] for w in jj["rows"]], "o-", color=C2, lw=1.5, ms=3.0,
            label="在庫コスト")
    ax.plot(sj, pj, "o-", color=C1, lw=2.6, ms=4.6, label="E[PnL](λ∝σ)")
    i = int(np.argmax(pj))
    ax.plot([sj[i]], [pj[i]], "*", color=C1, ms=15, zorder=6)
    ax.text(sj[i], pj[i] * 1.14, f"最適 σ≈{sj[i]:.1f}%", fontsize=7.6,
            ha="center", color=C1)
    k = np.where(np.diff(np.sign(pj)))[0]
    if k.size:
        j2 = k[-1]
        sb = sj[j2] + (sj[j2 + 1] - sj[j2]) * (0 - pj[j2]) / (pj[j2 + 1] - pj[j2])
        ax.plot([sb], [0], "v", color=INK, ms=8, zorder=5)
        ax.text(sb, -1.6, f"損益 0\nσ={sb:.1f}%", fontsize=7.6, ha="center", color=INK)
ax.axhline(0, color=INK, lw=1.2, ls="--")
sig0 = res["calib"]["sigma_day_med"] * 100
ax.axvline(sig0, color=C4, lw=1.4, ls=":")
ax.text(sig0 * 1.05, ax.get_ylim()[1] * .80, f"実測 σ\n{sig0:.1f}%/日",
        fontsize=7.6, color=C4)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.set_xlabel("日次ボラティリティ σ [%]"); ax.set_ylabel("USD/日")
ax.set_title("③ ボラティリティ応答", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7.4, loc="lower left")

# ④ 年次 1,000 本(★母数の不確実性を入れた二重ブートストラップ)
ax = axes[3]
fx = D / "sde_annual_fixed.npz"
ann_old = az["ann"]
ann = np.load(fx)["ann"] if fx.exists() else ann_old
ax.hist(ann_old / CAP * 100, bins=44, color=CM, alpha=.30, lw=0,
        label=f"母数誤差なし(誤)\nsd {ann_old.std(ddof=1):,.0f}")
ax.hist(ann / CAP * 100, bins=44, color=C1, alpha=.46, lw=0,
        label=f"★二重ブートストラップ\nsd {ann.std(ddof=1):,.0f}")
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.30)
for q, c, ls, dy in [(5, C2, "--", .52), (50, INK, "-", .62), (95, C3, "--", .52)]:
    v = np.percentile(ann / CAP * 100, q)
    ax.axvline(v, color=c, lw=1.6, ls=ls)
    ax.text(v, ymax * dy, f" p{q}\n {v:,.0f}%", fontsize=7.4, color=c,
            ha="left" if q != 5 else "right")
ax.set_xlabel("年次収益率 [%](資金 100 USD・単利)"); ax.set_ylabel("本数")
ax.set_title("④ 年次 1,000 本", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7.0, loc="upper left", handlelength=1.2,
          labelspacing=.35)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_45_sde.png", bbox_inches="tight", facecolor="white")
print("保存: charts/xyz_DRAM_45_sde.png")
