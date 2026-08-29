"""信号の組み合わせ(52)。報告 = combine_signals_report.md。

出所: data/combine_signals.json(40 日、17 通り)
      data/combine_robust.json(期間 x 気配側の頑健性)
再生成: uv run python scripts/combine_chart.py

タイトルの数値は下のデータから計算しており、報告本文からの転記はしていない。
"""
from __future__ import annotations
import json
from _chartstyle import C1, C2, C3, C4, CM, D, plt, np, save

rows = json.loads((D / "combine_signals.json").read_text(encoding="utf-8"))
rob = json.loads((D / "combine_robust.json").read_text(encoding="utf-8"))

base = rows[0]
cand = [r for r in rows if not r["label"].startswith("[参考]")]
hind = [r for r in rows if r["label"].startswith("[参考]")]
BONF = 0.05 / len(rows)          # 17 通り試しているので閾値を割る

fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))

# (1) 総額 — 何もしない との比較
ax = axes[0, 0]
lab = [r["label"].replace("        ", " ") for r in cand]
tot = [r["total"] for r in cand]
sig = [r["per_p"] < BONF for r in cand]
y = np.arange(len(cand))
ax.barh(y, tot, color=[C2 if s else C1 for s in sig], height=0.62)
ax.axvline(base["total"], color=CM, lw=1.6, ls="--")
ax.annotate(f"何もしない {base['total']:.1f}", (base["total"], len(cand) - 0.4),
            xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=CM)
best = int(np.argmax(tot))
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("総額(40 日合計)")
ax.set_title(f"(1) 最良は {cand[best]['label'].strip()} の {tot[best]:.1f}\n"
             f"何もしない({base['total']:.1f})の {tot[best]/base['total']:.1f} 倍。"
             f"色付きは Bonferroni 後も有意", fontsize=10)

# (2) 単価と総額は別の頂点を持つ
ax = axes[0, 1]
qs = [0.2, 0.4, 0.6]
for name, c, mk in (("A+B2", C2, "o"), ("A+B1", C1, "s"), ("B2 退出ゲート", C3, "^")):
    pf = [next(r["per_fill"] for r in rows if r["label"].startswith(name) and f"q={q}" in r["label"])
          for q in qs]
    ax.plot(qs, pf, mk + "-", color=c, lw=2, ms=7, label=f"{name} 単価")
ax.set_xlabel("ゲートの強さ q"); ax.set_ylabel("約定単価 bp", color=C2)
ax2 = ax.twinx(); ax2.grid(False)
tt = [next(r["total"] for r in rows if r["label"].startswith("A+B2") and f"q={q}" in r["label"])
      for q in qs]
ax2.plot(qs, tt, "D--", color=C4, lw=2, ms=8)
ax2.set_ylabel("A+B2 の総額", color=C4); ax2.tick_params(axis="y", labelcolor=C4)
ax2.spines["right"].set_visible(True)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xticks(qs)
ax.set_title(f"(2) 単価は q とともに上がるが、総額は q=0.4 が頂点\n"
             f"A+B2 の総額 {tt[0]:.1f} → {tt[1]:.1f} → {tt[2]:.1f}"
             f"(単価だけ見ると q を上げ続けたくなる)", fontsize=10)

# (3) 保持率 — 何回発火するか
ax = axes[1, 0]
kr = [r["keep_rate"] * 100 for r in cand]
ax.scatter(kr, tot, s=90, c=[C2 if s else C1 for s in sig], zorder=3)
for k, t, r in zip(kr, tot, cand):
    ax.annotate(r["label"].split()[0] + r["label"].split()[-1].replace("ゲート", ""),
                (k, t), xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.axhline(base["total"], color=CM, lw=1.4, ls="--")
ax.set_xlabel("保持率 %(高いほど発火が少ない = 元の約定を残す)")
ax.set_ylabel("総額")
ax.set_title(f"(3) 引きすぎても引かなすぎても総額は伸びない\n"
             f"保持率 {kr[int(np.argmax(tot))]:.0f}% 付近が頂点", fontsize=10)

# (4) 期間 x 気配側の頑健性
ax = axes[1, 1]
keys = [k for k in rob if "両側" in k]
per = [rob[k]["per_fill"] for k in keys]
pv = [rob[k].get("per_p", np.nan) for k in keys]
xs = np.arange(len(keys))
nd = [rob[k]["n_days"] for k in keys]
ax.bar(xs, per, color=[C2 if p < 0.05 else CM for p in pv], width=0.5)
ax.set_ylim(0, max(per) * 1.4)
for x, v, p, n in zip(xs, per, pv, nd):
    ax.annotate(f"{v:.3f}\np={p:.3f}\n{n} 日", (x, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels([k.replace("|両側", "") for k in keys])
ax.set_ylabel("約定単価 bp")
# 立ち上げ期は点推定が全期間より高い。有意でないのは日数が半分だからで、
# 「有意でない」を「効果がない」と書いてはいけない。
i_up = keys.index("立ち上げ|両側")
ax.set_title(f"(4) 立ち上げ期は点推定 {per[i_up]:.3f} と全期間 {per[0]:.3f} より高い\n"
             f"有意でないのは {nd[i_up]} 日しかないため。効果が無いという意味ではない",
             fontsize=10)

n_sig = sum(sig)
save(fig, "xyz_DRAM_52_combine_signals.png",
     f"信号の組み合わせ — {base['n_days']} 日 x {len(rows)} 通り"
     f"(Bonferroni 閾値 p<{BONF:.4f} を満たすのは {n_sig} 通り)")
