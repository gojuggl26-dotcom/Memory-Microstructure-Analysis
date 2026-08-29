"""特徴量の予測力と相互相関(53)。報告 = features_report.md。

出所: data/features_summary.json(98 日、1 秒グリッド / イベント粒度)
再生成: uv run python scripts/features_chart.py

タイトルの数値はすべて下のデータから計算しており、報告本文からの転記はしていない。
"""
from __future__ import annotations
import json
from _chartstyle import C1, C2, C3, C4, CM, D, plt, np, save

f = json.loads((D / "features_summary.json").read_text(encoding="utf-8"))
pred = f["predictive_daily_corr"]
cc = f.get("cross_corr_median", {})

feats = ["order_imbalance", "depth_one_level", "voi", "ofi_cont", "voi_raw_diff"]
short = {"order_imbalance": "OI(約定)", "depth_one_level": "板 L=1",
         "voi": "VOI", "ofi_cont": "OFI", "voi_raw_diff": "VOI 生差分"}

fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))

# (1) 予測力 — 粒度別
ax = axes[0, 0]
xs = np.arange(len(feats))
ev = [pred[f"{k}|ev1"]["median"] for k in feats]
s1 = [pred[f"{k}|1s"]["median"] for k in feats]
ax.bar(xs - 0.2, ev, width=0.35, color=C1, label="イベント粒度")
ax.bar(xs + 0.2, s1, width=0.35, color=C2, label="1 秒グリッド")
for x, a, b in zip(xs, ev, s1):
    ax.annotate(f"{a:.3f}", (x - 0.2, a), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8)
    ax.annotate(f"{b:.3f}", (x + 0.2, b), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([short[k] for k in feats], fontsize=8.5)
ax.set_ylabel("日次相関の中央値"); ax.legend(frameon=False, fontsize=8)
best = feats[int(np.argmax(s1))]
ax.set_title(f"(1) 最も効くのは {short[best]}(1 秒で {max(s1):.3f})\n"
             f"どの特徴量も 1 秒グリッドの方が高い — 粒度を粗くすると雑音が減る",
             fontsize=10)

# (2) 符号の一貫性 — 何日が負だったか
ax = axes[0, 1]
neg = [pred[f"{k}|1s"]["n_days_negative"] for k in feats]
days = [pred[f"{k}|1s"]["days"] for k in feats]
pos = [d - n for d, n in zip(days, neg)]
ax.barh(xs, [p / d * 100 for p, d in zip(pos, days)],
        color=[C3 if n == 0 else (C1 if n < 10 else C2) for n in neg], height=0.55)
for x, p, d, n in zip(xs, pos, days, neg):
    ax.annotate(f"{p}/{d} 日 (負 {n})", (p / d * 100, x), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8.5)
ax.set_yticks(xs); ax.set_yticklabels([short[k] for k in feats], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 128); ax.set_xlabel("正の相関だった日の割合 %")
ax.set_title(f"(2) OI は {days[0]} 日すべてで正\n"
             f"VOI 生差分は {neg[-1]}/{days[-1]} 日が負 — 符号が安定しない", fontsize=10)

# (3) 特徴量どうしの相関 — 独立ではない
ax = axes[1, 0]
pairs = sorted(cc.items(), key=lambda kv: -abs(kv[1]))[:8]
ys = np.arange(len(pairs))
vals = [v for _, v in pairs]
ax.barh(ys, vals, color=[C2 if v > 0 else C1 for v in vals], height=0.6)
for y, (k, v) in zip(ys, pairs):
    ax.annotate(f"{v:+.3f}", (v, y), xytext=(6 if v > 0 else 6, 0),
                textcoords="offset points", va="center", ha="left", fontsize=8.5)
ax.set_yticks(ys)
ax.set_yticklabels([k.replace("|", " と ") for k, _ in pairs], fontsize=7.5)
ax.invert_yaxis(); ax.axvline(0, color=CM, lw=1.0)
ax.set_xlim(min(vals) * 1.25, max(vals) * 1.18)
ax.set_xlabel("相互相関(日次中央値)")
mx = pairs[0]
ax.set_title(f"(3) 特徴量は独立でない — 最大 {mx[1]:+.3f}\n"
             f"({mx[0].replace('|', ' と ')})。足し合わせても情報は加算されない",
             fontsize=10)

# (4) 予測力とばらつき
ax = axes[1, 1]
for k, c in zip(feats, (C1, C2, C3, C4, CM)):
    v = pred[f"{k}|1s"]
    lo, hi = v.get("q25"), v.get("q75")
    if lo is None:
        continue
    ax.plot([lo, hi], [feats.index(k)] * 2, "-", color=c, lw=6, alpha=0.45)
    ax.plot(v["median"], feats.index(k), "o", color=c, ms=10)
    ax.annotate(f"{v['median']:.3f}", (v["median"], feats.index(k)),
                xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=8.5, fontweight="bold")
ax.axvline(0, color=CM, lw=1.2, ls="--")
ax.set_yticks(range(len(feats))); ax.set_yticklabels([short[k] for k in feats], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlabel("日次相関(点は中央値、帯は 25〜75 パーセンタイル)")
# 実データではどの帯も 0 を跨いでいない。図に無いことを書かない。
q25 = {k: pred[f"{k}|1s"]["q25"] for k in feats}
worst = min(q25, key=q25.get)
ax.set_title("(4) どの特徴量も四分位帯が 0 より上\n"
             f"最も 0 に近いのは {short[worst]}(25% 点 {q25[worst]:.3f})"
             f" — 帯で見ても符号は一貫している", fontsize=10)

save(fig, "xyz_DRAM_53_features_predictive.png",
     f"特徴量の予測力と相互相関 — {pred['depth_one_level|1s']['days']} 日"
     f"(features_report.md)")
