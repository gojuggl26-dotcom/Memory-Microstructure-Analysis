"""book_slope モデルの図(34)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

r = json.loads((D / "slope_model_results.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))

# (1) Lasso 経路 — 検証 MSE が最小の点で止める
ax = axes[0]
a = np.array(r["lasso"]["path_alphas"]); v = np.array(r["lasso"]["path_val_mse"])
m = a > 0
ax.plot(a[m], v[m], "-", color=C1, lw=2)
ax.axhline(r["lasso"]["val_mse_null"], color=CM, ls="--", lw=1.2, label="定数モデル")
ax.axvline(r["lasso"]["alpha"], color=C2, ls=":", lw=1.6,
           label=f"選択 α={r['lasso']['alpha']:.2e}\n({r['lasso']['n_selected']} 特徴量)")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("α(正則化の強さ・右ほど強い)"); ax.set_ylabel("検証期間の MSE(正規化リターン)")
ax.set_title("① 段階 8: Lasso 選択\n選択は検証期間で行う", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# (2) 係数 — 日次 t と NW z の両方、FDR 通過を印で
ax = axes[1]
inf = sorted(r["inference"], key=lambda x: abs(x["t_day"]))
ys = np.arange(len(inf))
cols = [C1 if x["FDR_pass_day"] else CM for x in inf]
ax.barh(ys, [x["t_day"] for x in inf], color=cols, height=.66)
for i, x in enumerate(inf):
    ax.text(x["t_day"] + (0.4 if x["t_day"] >= 0 else -0.4), i,
            f"{x['sign_days']}/{x['n_days']}", va="center",
            ha="left" if x["t_day"] >= 0 else "right", fontsize=7, color=INK)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(ys); ax.set_yticklabels([x["feature"] for x in inf], fontsize=8)
ax.set_xlabel("日次係数の t 統計量")
ax.set_title("② 段階 10・14: 推論と FDR\n(濃い = FDR q=0.05 通過・右は符号一致日数)", fontsize=10)

# (3) ウォークフォワードの日次 OOS R²
ax = axes[2]
wf = r["walk_forward"]
vals = np.array([w["R2"] for w in wf])
ax.bar(np.arange(len(vals)), vals, color=[C1 if x > 0 else C2 for x in vals], width=.8)
ax.axhline(0, color=INK, lw=1)
b = r["bootstrap_R2"]
ax.axhline(b["mean"], color=C3, lw=1.6, ls="--", label=f"平均 {b['mean']:+.5f}")
ax.fill_between([-.5, len(vals) - .5], b["ci"][0], b["ci"][1], color=C3, alpha=.13, lw=0)
ax.set_xlim(-.5, len(vals) - .5)
ax.set_xticks(np.arange(0, len(wf), 5))
ax.set_xticklabels([wf[i]["dt"][5:] for i in range(0, len(wf), 5)], fontsize=7, rotation=45)
ax.set_xlabel("OOS の日"); ax.set_ylabel("その日の OOS R²")
ax.set_title(f"③ 段階 11〜13: ウォークフォワード\n正 {b['pos_days']}/{b['n_days']} 日"
             f"(符号検定 p={b['sign_test_p']:.3f})", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# (4) 取引コストと経済的有意性
ax = axes[3]
ec = r["economics"]
qs = sorted({e["閾値分位"] for e in ec})
for cname, c, mk in (("メイカー(手数料のみ)", C1, "o"), ("テイカー(半スプレッド+手数料)", C2, "s")):
    vv = [next(e["純益_bp_日"] for e in ec if e["閾値分位"] == q and e["コスト前提"] == cname)
          for q in qs]
    ax.plot(qs, vv, mk + "-", color=c, lw=2, ms=6, label=cname)
gross = [next(e["粗利_bp_日"] for e in ec if e["閾値分位"] == q) for q in qs]
ax.plot(qs, gross, "^--", color=C3, lw=1.6, ms=6, label="コスト前(粗利)")
ax.axhline(0, color=INK, lw=1)
ax.set_xlabel("建てる閾値(|予測| の分位)"); ax.set_ylabel("OOS 損益 bp/日")
be = ec[0]["損益分岐コスト_bp"]
ax.set_title(f"④ 段階 15・16: 経済的有意性\n損益分岐コスト(片道) {be:.4f} bp", fontsize=10)
ax.legend(frameon=False, fontsize=8)

fig.savefig(CH / "xyz_DRAM_34_slope_model.png", bbox_inches="tight", facecolor="white")
print("charts/xyz_DRAM_34_slope_model.png")
