"""発注・取消フローの予測力(54)。報告 = flow_report.md。

出所: data/flow_grid_summary.json(99 日 x 4 窓 x 5 指標)
再生成: uv run python scripts/flow_chart.py

比較のため data/features_summary.json の OI も並べる。
タイトルの数値はすべてデータから計算しており、報告本文からの転記はしていない。
"""
from __future__ import annotations
import json
from _chartstyle import C1, C2, C3, C4, CM, D, plt, np, save

d = json.loads((D / "flow_grid_summary.json").read_text(encoding="utf-8"))
pred = d["predictive_next_window"]
feat = json.loads((D / "features_summary.json").read_text(encoding="utf-8"))
OI = feat["predictive_daily_corr"]["order_imbalance|1s"]["median"]

WINS = ["100ms", "1s", "10s", "60s"]
METS = ["cancel_intensity_diff", "new_intensity_diff", "cancel_ratio",
        "lambda_cancel_bid", "lambda_cancel_ask"]
SHORT = {"cancel_intensity_diff": "取消強度の差", "new_intensity_diff": "新規強度の差",
         "cancel_ratio": "取消比率", "lambda_cancel_bid": "取消 λ(bid)",
         "lambda_cancel_ask": "取消 λ(ask)"}


def g(m, w, key="median"):
    v = pred.get(f"{m}|{w}")
    return v.get(key, np.nan) if v else np.nan


fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))

# (1) 窓ごとの予測力
ax = axes[0, 0]
xs = np.arange(len(WINS))
for m, c, mk in zip(METS, (C2, C1, C3, C4, CM), ("o", "s", "^", "D", "v")):
    ax.plot(xs, [g(m, w) for w in WINS], mk + "-", color=c, lw=1.8, ms=6,
            label=SHORT[m])
ax.axhline(0, color=CM, lw=1.2, ls="--")
ax.set_xticks(xs); ax.set_xticklabels(WINS)
ax.set_xlabel("集計窓"); ax.set_ylabel("次窓との相関(日次中央値)")
ax.legend(frameon=False, fontsize=7.5, ncol=2)
allv = [g(m, w) for m in METS for w in WINS]
ax.set_title(f"(1) どの指標・どの窓でも |相関| は {np.nanmax(np.abs(allv)):.3f} 以下\n"
             f"符号は指標によって逆 — 取消は負、新規は正", fontsize=10)

# (2) 特徴量との比較 — 桁が違う
ax = axes[0, 1]
best_flow = max(((m, w) for m in METS for w in WINS),
                key=lambda t: abs(g(*t)) if not np.isnan(g(*t)) else -1)
names = ["OI(約定)\n1 秒", f"{SHORT[best_flow[0]]}\n{best_flow[1]}"]
vals = [abs(OI), abs(g(*best_flow))]
ax.bar([0, 1], vals, color=[C3, C2], width=0.5)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.4f}", (i, v), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("|日次相関の中央値|")
ax.set_title(f"(2) フロー系は最強でも OI の 1/{vals[0]/vals[1]:.0f}\n"
             f"板や約定から作る指標と比べて桁が違う", fontsize=10)

# (3) 符号の一貫性 — 弱くても向きは揃うか
ax = axes[1, 0]
items = [(m, w) for m in METS for w in WINS]
items = sorted(items, key=lambda t: -abs(g(*t)) if not np.isnan(g(*t)) else 0)[:10]
ys = np.arange(len(items))
frac = []
for m, w in items:
    v = pred[f"{m}|{w}"]
    neg, dd = v.get("n_days_negative", 0), v.get("days", 1)
    # 中央値の符号と一致した日の割合
    frac.append((dd - neg) / dd * 100 if g(m, w) > 0 else neg / dd * 100)
ax.barh(ys, frac, color=[C3 if f >= 70 else (C1 if f >= 60 else CM) for f in frac],
        height=0.6)
ax.axvline(50, color=CM, lw=1.4, ls="--")
for y, f_, (m, w) in zip(ys, frac, items):
    ax.annotate(f"{f_:.0f}%", (f_, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8.5)
ax.set_yticks(ys)
ax.set_yticklabels([f"{SHORT[m]} {w}" for m, w in items], fontsize=7.5)
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("中央値と同じ符号だった日の割合 %")
ax.set_title(f"(3) 相関は小さいが符号は揃う — 最大 {max(frac):.0f}%\n"
             f"50%(破線)が偶然の水準。上回るなら向きは本物", fontsize=10)

# (4) bid と ask の非対称
ax = axes[1, 1]
b = [g("lambda_cancel_bid", w) for w in WINS]
a = [g("lambda_cancel_ask", w) for w in WINS]
ax.bar(xs - 0.18, b, width=0.33, color=C1, label="取消 λ(bid)")
ax.bar(xs + 0.18, a, width=0.33, color=C2, label="取消 λ(ask)")
ax.axhline(0, color=CM, lw=1.2)
ax.set_xticks(xs); ax.set_xticklabels(WINS)
ax.set_xlabel("集計窓"); ax.set_ylabel("次窓との相関(日次中央値)")
ax.legend(frameon=False, fontsize=8)
opp = sum(1 for i in range(len(WINS)) if b[i] * a[i] < 0)
ax.set_title(f"(4) bid と ask で符号が逆になる窓が {opp}/{len(WINS)}\n"
             f"片側の取消は反対側への圧力として効く", fontsize=10)

save(fig, "xyz_DRAM_54_flow_predictive.png",
     f"発注・取消フローの予測力 — 99 日 x {len(WINS)} 窓(flow_report.md)")
