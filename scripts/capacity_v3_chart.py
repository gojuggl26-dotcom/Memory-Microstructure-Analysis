"""数量スイープの容量(51)。報告 = capacity_v3_report.md。

出所: data/capacity_v3.csv(日次 x 数量 Q)
再生成: uv run python scripts/capacity_v3_chart.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from _chartstyle import C1, C2, C3, C4, CM, D, plt, np, save

rows = list(csv.DictReader((D / "capacity_v3.csv").open(encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[float(r["qty"])].append(r)
qs = sorted(by)
cols = [C3, C1, C4, C2, CM][: len(qs)]


def med(q, key):
    v = [float(r[key]) for r in by[q] if r.get(key) not in (None, "")]
    return float(np.median(v)) if v else float("nan")


fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# (1) 単位あたり損益は数量とともに落ちる
ax = axes[0, 0]
unit = [med(q, "pnl_bp_per_unit") for q in qs]
ax.plot(qs, unit, "o-", color=C1, lw=2.2, ms=8)
for q, v in zip(qs, unit):
    ax.annotate(f"{v:.3f}", (q, v), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
ax.set_xscale("log"); ax.set_xticks(qs); ax.set_xticklabels([f"{q:g}" for q in qs])
ax.set_xlabel("片側数量 Q(対数軸)"); ax.set_ylabel("単位あたり損益 bp(日次中央値)")
ax.set_title(f"(1) 単位あたりは Q とともに減衰\nQ={qs[0]:g} の {unit[0]:.3f} bp が "
             f"Q={qs[-1]:g} で {unit[-1]:.3f} bp", fontsize=10)

# (2) 総額は増える — 単価と回数のトレードオフ
ax = axes[0, 1]
tot = [med(q, "pnl_usd") for q in qs]
ax.bar(range(len(qs)), tot, color=cols, width=0.55)
for i, v in enumerate(tot):
    ax.annotate(f"${v:,.0f}", (i, v), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"Q={q:g}" for q in qs])
ax.set_ylabel("日次損益 $(中央値)")
ax.set_title(f"(2) 総額は Q とともに増える\n単価だけ見ると Q を絞る方策が必ず勝つので、"
             f"総額も併せて見る", fontsize=10)

# (3) 約定比率 — どれだけ捌けているか
ax = axes[1, 0]
fr = [med(q, "fill_ratio") * 100 for q in qs]
nf = [med(q, "n_fill") for q in qs]
ax.plot(qs, fr, "o-", color=C2, lw=2.2, ms=8, label="約定比率 %")
ax.set_xscale("log"); ax.set_xticks(qs); ax.set_xticklabels([f"{q:g}" for q in qs])
ax.set_xlabel("片側数量 Q(対数軸)"); ax.set_ylabel("約定比率 %", color=C2)
ax.tick_params(axis="y", labelcolor=C2)
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(qs, nf, "s--", color=C1, lw=1.6, ms=6)
ax2.set_ylabel("約定件数 / 日", color=C1); ax2.tick_params(axis="y", labelcolor=C1)
ax2.spines["right"].set_visible(True)
ax.set_title(f"(3) Q を増やしても約定比率は {min(fr):.2f}〜{max(fr):.2f}%\n"
             f"件数はほぼ横ばい — 板が吸収できる量に上限がある", fontsize=10)

# (4) 期末在庫 — 方向性リスクの残り方
ax = axes[1, 1]
inv = [med(q, "end_inv") for q in qs]
inv_abs = [float(np.median([abs(float(r["end_inv"])) for r in by[q]])) for q in qs]
ax.bar(np.arange(len(qs)) - 0.2, inv, width=0.35, color=C1, label="期末在庫(中央値)")
ax.bar(np.arange(len(qs)) + 0.2, inv_abs, width=0.35, color=C2, label="期末在庫の絶対値")
ax.axhline(0, color=CM, lw=1.0)
ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"Q={q:g}" for q in qs])
ax.set_ylabel("在庫(単位)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"(4) 在庫は Q に比例して膨らむ\n絶対値の中央値は Q={qs[0]:g} で "
             f"{inv_abs[0]:.1f} → Q={qs[-1]:g} で {inv_abs[-1]:.1f}", fontsize=10)

save(fig, "xyz_DRAM_51_capacity_qty_sweep.png",
     f"数量スイープの容量 — {len(rows)} 日x数量(capacity_v3_report.md)")
