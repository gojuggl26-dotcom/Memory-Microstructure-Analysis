"""資金 $100 のレバレッジ別収益分布(50)。報告 = capacity_report.md。

出所: data/capacity_100usd.json
再生成: uv run python scripts/capacity_chart.py
"""
from __future__ import annotations
import json
from _chartstyle import C1, C2, C3, C4, CM, D, CH, plt, np, save

d = json.loads((D / "capacity_100usd.json").read_text(encoding="utf-8"))
levs = sorted(d, key=lambda k: int(k))
cols = [C3, C1, C4, C2]

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# ① レバレッジ別の年率(平均と分位)
ax = axes[0, 0]
xs = np.arange(len(levs))
apy = [d[k]["apy_mean"] for k in levs]
p10 = [d[k]["pct"]["p10"] for k in levs]
p90 = [d[k]["pct"]["p90"] for k in levs]
ax.errorbar(xs, apy, yerr=[np.array(apy) - p10, np.array(p90) - np.array(apy)],
            fmt="o", color=C1, ecolor=CM, capsize=5, ms=9, lw=2)
for x, v in zip(xs, apy):
    ax.annotate(f"{v:,.0f}%", (x, v), xytext=(10, 0), textcoords="offset points",
                fontsize=9, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([f"{k}x" for k in levs])
ax.set_xlabel("レバレッジ"); ax.set_ylabel("年率 %(棒は p10〜p90)")
ax.set_title(f"① 年率はレバレッジにほぼ比例\n{levs[0]}x で {apy[0]:,.0f}% → "
             f"{levs[-1]}x で {apy[-1]:,.0f}%", fontsize=10)

# ② 損失確率
ax = axes[0, 1]
pl = [d[k]["prob_loss"] * 100 for k in levs]
ax.bar(xs, pl, color=cols, width=0.55)
for x, v in zip(xs, pl):
    ax.annotate(f"{v:.1f}%", (x, v), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([f"{k}x" for k in levs])
ax.set_xlabel("レバレッジ"); ax.set_ylabel("損失確率 %")
ax.set_title("② 損失確率はレバレッジで変わらない\n"
             "倍率は損益を定数倍するだけで、勝ち負けの向きは変えない", fontsize=10)

# ③ 分位の広がり
ax = axes[1, 0]
qs = ["p1", "p10", "p50", "p70", "p80", "p90", "p99"]
for k, c in zip(levs, cols):
    ax.plot([d[k]["pct"][q] for q in qs], "o-", color=c, lw=1.8, ms=5, label=f"{k}x")
ax.set_xticks(range(len(qs))); ax.set_xticklabels(qs)
ax.set_xlabel("分位"); ax.set_ylabel("年率 %")
ax.legend(frameon=False, fontsize=8, title="レバレッジ", title_fontsize=8)
ax.set_title("③ 分布は倍率で相似に広がる\n形は同じで、縦にだけ伸びている", fontsize=10)

# ④ 日次の平均とばらつき — 期待/σ はレバレッジで不変
ax = axes[1, 1]
dm = [d[k]["day_mean"] for k in levs]
ds = [d[k]["day_sd"] for k in levs]
ratio = [m / s for m, s in zip(dm, ds)]
ax.bar(xs - 0.2, dm, width=0.35, color=C1, label="日次平均 $")
ax.bar(xs + 0.2, ds, width=0.35, color=C2, label="日次σ $")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(xs, ratio, "D--", color=C3, lw=1.8, ms=7)
ax2.set_ylabel("期待 ÷ σ", color=C3); ax2.tick_params(axis="y", labelcolor=C3)
ax2.spines["right"].set_visible(True)
ax.set_xticks(xs); ax.set_xticklabels([f"{k}x" for k in levs])
ax.set_xlabel("レバレッジ"); ax.set_ylabel("$ / 日")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"④ 期待/σ は {min(ratio):.3f}〜{max(ratio):.3f} でほぼ一定\n"
             "レバレッジは効率を上げない — 分散も同じだけ増える", fontsize=10)

save(fig, "xyz_DRAM_50_capacity_leverage.png",
     "資金 $100 のレバレッジ別収益分布(capacity_report.md)")
