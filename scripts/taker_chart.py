"""テイカー分類の図(46)。報告 47 = taker_classification_report.md に対応。

出所: data/taker_summary.json(99 日集計)/ data/taker_daily.csv(日次)
再生成: uv run python scripts/taker_chart.py
"""
from __future__ import annotations
import csv
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

s = json.loads((D / "taker_summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((D / "taker_daily.csv").open(encoding="utf-8")))

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# ① 件数と数量で符号が逆転する
ax = axes[0, 0]
metrics = ["件数", "出来高", "想定元本"]
buy = [s["n_buy"], s["vol_buy"], s["ntl_buy"]]
sell = [s["n_sell"], s["vol_sell"], s["ntl_sell"]]
share = [b / (b + v) * 100 for b, v in zip(buy, sell)]
x = np.arange(len(metrics))
bars = ax.bar(x, share, color=[C1 if v > 50 else C2 for v in share], width=0.55)
ax.axhline(50, color=CM, lw=1.2, ls="--")
for xi, v in zip(x, share):
    ax.annotate(f"{v:.2f}%", (xi, v), xytext=(0, 6 if v > 50 else -16),
                textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("積極的な買いの比率 %"); ax.set_ylim(48.5, 51.2)
ax.set_title("① 件数は買い越し・数量は売り越し\n"
             f"件数 {share[0]:.2f}% vs 想定元本 {share[2]:.2f}%(50% を跨ぐ)", fontsize=10)

# ② 原因は約定サイズの非対称
ax = axes[0, 1]
avg_b = s["vol_buy"] / s["n_buy"]; avg_s = s["vol_sell"] / s["n_sell"]
d_ratio = []
for r in rows:
    nb, ns = float(r["n_buy"]), float(r["n_sell"])
    vb, vs = float(r["vol_buy"]), float(r["vol_sell"])
    if nb > 0 and ns > 0 and vb > 0:
        d_ratio.append((vs / ns) / (vb / nb))
d_ratio = np.array(d_ratio)
ax.hist(d_ratio, bins=40, color=C3, alpha=0.75, edgecolor="white", linewidth=0.4)
ax.axvline(1.0, color=CM, lw=1.4, ls="--")
ax.axvline(float(np.median(d_ratio)), color=C2, lw=2.0)
ax.annotate(f"中央値 {np.median(d_ratio):.3f}\n{int((d_ratio>1).sum())}/{len(d_ratio)} 日で 1 超",
            (float(np.median(d_ratio)), ax.get_ylim()[1] * 0.82), xytext=(10, 0),
            textcoords="offset points", fontsize=9, color=C2, fontweight="bold")
ax.set_xlabel("売りの平均約定サイズ ÷ 買いの平均約定サイズ(日次)")
ax.set_ylabel("日数")
# 報告書 §0 の「6.9% 大きい」は **日次比の中央値**(1.069)に基づく。
# 全期間プールの平均比(1.067)とは僅かに違うので、報告書と同じ日次中央値を出す。
ax.set_title(f"② 原因は約定サイズの非対称\n売りが {(np.median(d_ratio)-1)*100:.1f}% 大きい"
             f"(日次比の中央値。全期間平均は {avg_s:.2f} 対 {avg_b:.2f} 単位)", fontsize=10)

# ③ 符号の自己相関 — 長期記憶
ax = axes[1, 0]
lags = np.array(s["acf_lags"]); acf = np.array(s["acf_mean"])
ax.bar(lags, acf, color=C1, width=0.7)
ax.plot(lags, acf, "o-", color=C4, lw=1.4, ms=4)
ax.set_xlabel("ラグ(約定)"); ax.set_ylabel("符号の自己相関")
ax.set_xticks(lags[::2])
ax.set_title(f"③ 約定符号は強い長期記憶を持つ\nラグ1 = {acf[0]:.3f}、ラグ20 でも {acf[-1]:.3f}"
             f"({s['acf_days']} 日平均)", fontsize=10)

# ④ L4 の値段 — 推定則の正答率
ax = axes[1, 1]
tick = s.get("tick_rule_accuracy"); lr = s.get("quote_rule_accuracy") or s.get("lee_ready_accuracy")
names, vals, cols = [], [], []
if tick is not None: names.append("tick rule"); vals.append(tick * 100); cols.append(C2)
if lr is not None:   names.append("quote rule\n(Lee-Ready)"); vals.append(lr * 100); cols.append(C1)
names.append("L4 の真値"); vals.append(100.0); cols.append(C3)
y = np.arange(len(names))
ax.barh(y, vals, color=cols, height=0.55)
for yi, v in zip(y, vals):
    ax.annotate(f"{v:.2f}%", (v, yi), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlim(0, 112); ax.set_xlabel("約定符号の正答率 %")
miss = (1 - tick) if tick else float("nan")
ax.set_title(f"④ L4 を持つ価値\n約定プリントだけの tick rule は {miss*100:.0f}% を誤分類"
             f"(3 件に 1 件)", fontsize=10)

fig.suptitle(f"テイカー分類 — 2026-05-04〜08-10 の {s['n_days']} 日・{s['n_trades']:,} 取引",
             fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = CH / "xyz_DRAM_46_taker_classification.png"
fig.savefig(out, facecolor="white")
print("保存:", out)
