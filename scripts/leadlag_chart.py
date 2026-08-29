"""Binance → Hyperliquid リードラグの図(48)。報告 = leadlag_binance_report.md に対応。

出所: data/leadlag_hl_binance.json(85 日重複、CCF 4 解像度)
      data/leadlag_econ.json(経済的有意性。50ms 刻み)
再生成: uv run python scripts/leadlag_chart.py
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

d = json.loads((D / "leadlag_hl_binance.json").read_text(encoding="utf-8"))
e = json.loads((D / "leadlag_econ.json").read_text(encoding="utf-8"))
DELTA_STAR = 0.363    # 報告 38 の費用の壁 bp

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# ① CCF — 山は負のラグ側 = Binance 先行
ax = axes[0, 0]
g = d["grids"]["250000000"]
lags = np.array(g["lags_s"]); ccf = np.array(g["ccf_mean"]); pl = np.array(g["ccf_placebo"])
m = np.abs(lags) <= 3.0
ax.plot(lags[m], ccf[m], "-", color=C1, lw=2.2, label="実データ")
ax.plot(lags[m], pl[m], "-", color=CM, lw=1.4, label="プラセボ(日をずらす)")
ax.axvline(0, color=CM, lw=1.0, ls=":")
pk = lags[np.argmax(ccf)]
ax.axvline(pk, color=C2, lw=1.6, ls="--")
ax.annotate(f"山 {pk:.2f} 秒  ρ={ccf.max():.3f}", (pk, ccf.max()), xytext=(-10, -14),
            textcoords="offset points", ha="right", fontsize=9,
            color=C2, fontweight="bold")
ax.set_xlabel("ラグ 秒(負 = Binance が先)"); ax.set_ylabel("相互相関 ρ")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"① Binance が先行する\n山は {pk:.2f} 秒側・プラセボは平坦"
             f"(最大 {pl.max():.4f})", fontsize=10)

# ② 日ごとの山の位置 — 一貫しているか
ax = axes[0, 1]
pld = np.array(g["peak_lag_per_day"])
neg = int((pld < 0).sum()); n = len(pld)
bins = np.arange(pld.min() - 0.125, pld.max() + 0.25, 0.25)
ax.hist(pld, bins=bins, color=[C1], edgecolor="white", linewidth=0.5)
ax.axvline(0, color=CM, lw=1.4, ls="--")
ax.set_ylim(0, ax.get_ylim()[1]*1.22)
ax.annotate(f"{neg}/{n} 日が負 = Binance 先行", (0.98, 0.94), xycoords="axes fraction",
            ha="right", fontsize=10, color=C1, fontweight="bold")
ax.set_xlabel("その日の CCF の山の位置 秒"); ax.set_ylabel("日数")
ax.set_title(f"② 85 日中 {neg} 日で同じ向き\n少数日の外れ値が作った形ではない", fontsize=10)

# ③ 解像度を変えても山は残るか
ax = axes[1, 0]
# 5000ms 刻みは ±10 秒窓でも点が 5 個しかなく曲線として読めないので外す
for key, c, mk in zip(("100000000", "250000000", "1000000000"),
                      (C4, C1, C3), ("D", "o", "s")):
    gg = d["grids"][key]
    L = np.array(gg["lags_s"]); Y = np.array(gg["ccf_mean"])
    mm = np.abs(L) <= 10.0
    ax.plot(L[mm], Y[mm], mk + "-", color=c, lw=1.6, ms=3,
            label=f"{gg['step_ms']:.0f}ms 刻み")
ax.axvline(0, color=CM, lw=1.0, ls=":")
ax.set_xlabel("ラグ 秒(負 = Binance が先)"); ax.set_ylabel("相互相関 ρ")
ax.legend(frameon=False, fontsize=8)
ax.set_title("③ 3 つの解像度すべてで負のラグ側に山\n"
             "刻み幅の作り物ではない(粗い刻みほど ρ は大きく出る)", fontsize=10)

# ④ 経済的有意性 — 費用の壁を越えるか
ax = axes[1, 1]
hz = [2, 5, 10, 20]
step = e["step_ms"]
real = [e[f"real_dx5_hy{h}"]["corr"] for h in hz]
plac = [e[f"placebo_dx5_hy{h}"]["corr"] for h in hz]
xs = np.array(hz) * step
ax.plot(xs, real, "o-", color=C1, lw=2.2, ms=6, label="実データ")
ax.plot(xs, plac, "s--", color=CM, lw=1.4, ms=5, label="プラセボ")
for x, v in zip(xs, real):
    ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel(f"HL 側の先読み地平 ms(Binance の直前 {5*step:.0f}ms の動きに対して)")
ax.set_ylabel("相関 ρ")
ax.legend(frameon=False, fontsize=8)
best = hz[int(np.argmax(real))]
ax.set_title(f"④ 相関の山は {best*step:.0f}ms 先(ρ={max(real):.3f})\n"
             f"プラセボは {max(plac):.4f} で平坦 — 偶然ではない", fontsize=10)

fig.suptitle(f"Binance DRAMUSDT → Hyperliquid xyz:DRAM のリードラグ — "
             f"{g['n_days']} 日重複(2026-05-18〜08-10)", fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = CH / "xyz_DRAM_48_leadlag_binance.png"
fig.savefig(out, facecolor="white")
print("保存:", out)
