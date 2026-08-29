"""OBI レベル別の情報量の図(40)。"""
from __future__ import annotations
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3 = "#3b6fd4", "#c2410c", "#0f766e"
CM, INK, GRID = "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False})

NL, BURN = 10, 7
days = [json.loads(Path(f).read_text(encoding="utf-8"))
        for f in sorted(glob.glob(str(D / "obi_levels" / "*.json")))][BURN:]
n = len(days)
lv = np.array([[d["level"][L]["r2"] for L in range(NL)] for d in days])
lb = np.array([[d["level"][L]["beta"] for L in range(NL)] for d in days])
cu = np.array([[d["cumulative"][L]["r2"] for L in range(NL)] for d in days])
inc = np.array([[d["incremental"][L]["delta"] for L in range(NL)] for d in days])
cf = np.array([[d["incremental"][L]["r2_cumfit"] for L in range(NL)] for d in days])
rat = lambda a: np.nanmedian(np.where(a[:, [0]] > 1e-12, a / np.maximum(a[:, [0]], 1e-12), np.nan), 0)
x = np.arange(1, NL + 1)

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3))

# ① 3 つの定義を重ねる
ax = axes[0]
for a, c, mk, lab in ((lv, C1, "o", "① レベル別 I_L"), (cu, C3, "s", "② 累積 OBI(L)"),
                      (inc, C2, "^", "③ 増分 R²")):
    ax.plot(x, rat(a), mk + "-", color=c, lw=2, ms=6, label=lab)
ax.axhline(1, color=CM, lw=1, ls="--")
ax.set_yscale("log"); ax.set_xticks(x)
ax.set_xlabel("最良気配からの段数"); ax.set_ylabel("情報量(1 段目 = 1、対数軸)")
ax.set_title("① 定義で見え方が変わる\n累積が緩やかなのは L1 を含むから", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ② レベル別 — 日次のばらつき込み
ax = axes[1]
r = np.where(lv[:, [0]] > 1e-12, lv / np.maximum(lv[:, [0]], 1e-12), np.nan)
q1, q2, q3 = [np.nanpercentile(r, p, axis=0) for p in (25, 50, 75)]
ax.fill_between(x, q1, q3, color=C1, alpha=.18, lw=0, label="日次の四分位")
ax.plot(x, q2, "o-", color=C1, lw=2, ms=6, label="中央値")
ax.axhline(1, color=CM, lw=1, ls="--")
for i in (1, 2, 4, 9):
    ax.annotate(f"{q2[i]:.3f}", (x[i], q2[i]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_ylim(0, 1.15)
ax.set_xlabel("最良気配からの段数"); ax.set_ylabel("R²_L / R²_1")
ax.set_title("② レベル別は 2 段目で 1/4\n5 段目で 1/22", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# ③ 符号一貫性 — R² は符号を持たないので必須
ax = axes[2]
s1 = np.sign(np.median(lb[:, 0]))
same = [(np.sign(lb[:, L]) == s1).sum() for L in range(NL)]
frac = np.array(same) / n * 100
ax.bar(x, frac, color=[C1 if f >= 70 else (C3 if f >= 60 else CM) for f in frac], width=.66)
ax.axhline(50, color=C2, lw=1.6, ls="--", label="コイン投げ 50%")
for xi, f, s in zip(x, frac, same):
    ax.text(xi, f + 1.5, f"{s}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_ylim(0, 108)
ax.set_xlabel("最良気配からの段数"); ax.set_ylabel("β が 1 段目と同符号だった日の割合 %")
ax.set_title(f"③ 符号一貫性({n} 日)\nR² は符号を持たないので必須の確認", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left")

# ④ 段を足していったときの累積 R²
ax = axes[3]
m = np.nanmedian(cf, 0)
ax.plot(x, m, "o-", color=C3, lw=2, ms=6)
ax.fill_between(x, np.nanpercentile(cf, 25, 0), np.nanpercentile(cf, 75, 0),
                color=C3, alpha=.15, lw=0)
ax.axhline(m[0], color=C1, lw=1.6, ls="--", label=f"1 段目だけ {m[0]:.5f}")
ax.annotate(f"10 段で {m[-1]/m[0]:.2f} 倍", (x[-1], m[-1]), textcoords="offset points",
            xytext=(-8, -16), ha="right", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xlabel("使った段数(1..L の重回帰)"); ax.set_ylabel("R²(日次中央値)")
ax.set_title("④ 全 10 段を使っても 1.29 倍\n深さを足す価値は小さい", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(CH / "xyz_DRAM_40_obi_levels.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_40_obi_levels.png")
