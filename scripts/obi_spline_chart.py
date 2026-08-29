"""期間体制 × OBI スプラインの図(49)。報告 = obi_spline_regime_report.md に対応。

出所: data/obi_spline_regimes.json(98 日 / 845 万行、1 秒グリッド)
再生成: uv run python scripts/obi_spline_chart.py
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

d = json.loads((D / "obi_spline_regimes.json").read_text(encoding="utf-8"))
regs = d["regimes"]
KEY = "depth_one_level"           # 報告の主表と同じ OBI(L=1)
COLS = [CM, C4, C3, C1, C2]       # 体制 1→5(古い→新しい)

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# ① 体制ごとのスプライン曲線 — 非線形性が「現れる」
ax = axes[0, 0]
for r, c in zip(regs, COLS):
    o = r["obi"][KEY]
    x, y = o.get("curve_x"), o.get("curve_y")
    if not x or not y:
        continue
    ax.plot(x, y, "-", color=c, lw=2.0,
            label=f"体制{r['regime']} ({r['days'].replace('2026-','')})")
ax.axhline(0, color=CM, lw=1.0, ls=":"); ax.axvline(0, color=CM, lw=1.0, ls=":")
ax.set_xlabel("OBI(L=1)"); ax.set_ylabel("予測される次の動き bp")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("① 体制が進むほど曲線が立ち、裾が反り返る\n"
             "初期(体制1)は直線 — 非線形性は最初から在ったのではない", fontsize=10)

# ② 線形 R² の推移
ax = axes[0, 1]
xs = [r["regime"] for r in regs]
r2 = [r["obi"][KEY]["r2_linear"] for r in regs]
# placebo_r2 は knot 候補ごとの辞書。選ばれた k_best のものを取る
def _placebo(o):
    pr = o.get("placebo_r2")
    if isinstance(pr, dict):
        return pr.get(str(o.get("k_best")), next(iter(pr.values()), np.nan))
    return pr if pr is not None else np.nan
pb = [_placebo(r["obi"][KEY]) for r in regs]
ax.plot(xs, r2, "o-", color=C1, lw=2.2, ms=7, label="実データ")
ax.plot(xs, pb, "s--", color=CM, lw=1.4, ms=5, label="プラセボ")
for x, v in zip(xs, r2):
    ax.annotate(f"{v:.5f}", (x, v), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(xs); ax.set_xlabel("期間体制(1=最古 → 5=最新)"); ax.set_ylabel("線形 R²")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"② 予測力は体制 1 → 5 で {r2[-1]/r2[0]:.0f} 倍\n"
             f"{r2[0]:.5f} → {r2[-1]:.5f}(市場が成熟して初めて効く)", fontsize=10)

# ③ スプラインが線形をどれだけ上回るか
ax = axes[1, 0]
gain = [r["obi"][KEY]["gain"] for r in regs]
# k_best は knot の本数(len(knots) と一致することを確認済み)
kb = [len(r["obi"][KEY].get("knots") or []) for r in regs]
bars = ax.bar(xs, gain, color=COLS, width=0.55)
ax.axhline(1.0, color=CM, lw=1.4, ls="--")
for x, v, k in zip(xs, gain, kb):
    ax.annotate(f"{v:.3f}\nknot={k}", (x, v), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(xs); ax.set_xlabel("期間体制"); ax.set_ylabel("スプライン R² ÷ 線形 R²")
ax.set_ylim(0.99, max(gain) * 1.05)
ax.set_title("③ 曲げて得られる上積みは小さい\n"
             "knot は交差検証で機械選択(体制1 は knot=0 = 直線が選ばれた)", fontsize=10)

# ④ 半スプレッドの縮小と予測力 — 同時に起きている
ax = axes[1, 1]
SPREAD = [8.140, 2.072, 1.343, 1.145, 0.287]     # 報告 §0 の表(bp)
ax.plot(SPREAD, r2, "o-", color=C4, lw=2.0, ms=8)
for sp, v, rg in zip(SPREAD, r2, xs):
    ax.annotate(f"体制{rg}", (sp, v), xytext=(6, 6), textcoords="offset points",
                fontsize=9, fontweight="bold")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("半スプレッド bp(対数軸・右へ行くほど狭い)")
ax.set_ylabel("線形 R²")
ax.set_title("④ スプレッドが縮むほど OBI が効く\n"
             f"8.14bp → 0.29bp と縮む間に R² は {r2[-1]/r2[0]:.0f} 倍", fontsize=10)

fig.suptitle(f"期間体制 × OBI(L=1)のスプライン回帰 — "
             f"{d['n_days']} 日 / {d['n_rows']:,} 行(1 秒グリッド)", fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = CH / "xyz_DRAM_49_obi_spline_regime.png"
fig.savefig(out, facecolor="white")
print("保存:", out)
