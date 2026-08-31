"""最良気配の入れ替わり(BBO turnover)9 指標を図示する。

    ① 置換回数(bid / ask)   ② BBO lifetime の分布
    ③ size / order / wallet の回転率   ④ 持続曲線(価格 と 中身)
    ⑤ touch ownership duration の分布  ⑥ 所有時間の日次推移
    ⑦ 立会日と閉場日                   ⑧ 検算(被覆率)

    uv run python scripts/plot_bbo_turnover.py --coin xyz:MU
出力: charts/<coin>_bbo_turnover.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
UP, DOWN, WARN, GREEN, PURPLE = "#e34948", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
DLAB = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s", "2s", "5s"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def logfmt(ax, axis="x"):
    f = mpl.ticker.FuncFormatter(
        lambda v, _: (f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax_ = ax.xaxis if axis == "x" else ax.yaxis
    ax_.set_major_formatter(f)
    ax_.set_minor_formatter(mpl.ticker.NullFormatter())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_parquet(ROOT / "data" / f"bbo_turnover_daily_{tag}.parquet").sort("dt")
    C = pl.read_parquet(ROOT / "data" / f"bbo_turnover_curves_{tag}.parquet")
    days = D["dt"].to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    op = np.array([d in sess for d in days])
    x = np.arange(len(days))
    xt = [0, len(days) // 2, len(days) - 1]
    xl = [days[i][5:] for i in xt]

    def md(k, sub=None):
        v = D[k].to_numpy().astype(float)
        if sub is not None:
            v = v[sub]
        return float(np.nanmedian(v))

    def curve(kind, agg="median"):
        g = (C.filter(pl.col("kind") == kind).group_by("x", "xv")
             .agg(v=(pl.col("v").sum() if agg == "sum" else pl.col("v").median()))
             .sort("xv"))
        return g["xv"].to_numpy(), g["v"].to_numpy()

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 9.8), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.50, wspace=0.30,
                          left=0.05, right=0.985, top=0.790, bottom=0.085)

    # ---- ① 置換回数 ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, D["repl_bid"].to_numpy(), "-", color=GREEN, lw=1.8, label="買い(bid)")
    ax.plot(x, D["repl_ask"].to_numpy(), "-", color=WARN, lw=1.8, label="売り(ask)")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("最良価格が変わった回数[回/秒]", color=INK2, fontsize=9.5)
    ax.set_title(f"① BBO replacement count\n"
                 f"買い {md('repl_bid'):.2f} / 売り {md('repl_ask'):.2f} 回/秒"
                 f"(1 日あたり {md('repl_bid')*86400/1000:.0f} 千回)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ② BBO lifetime の分布 ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    # ★買いと売りはほぼ完全に重なる。売りを破線にして両方見えるようにする
    for nm, cl, ls, lw2, lab in [("life_bid", GREEN, "-", 2.6, "買い"),
                                 ("life_ask", WARN, "--", 1.5, "売り")]:
        xv, v = curve(nm, "sum")
        ms = xv / 1e6
        ax.step(ms[1:], (v / v.sum() * 100)[1:], where="post", color=cl, lw=lw2,
                ls=ls, label=lab)
    ax.set_xscale("log")
    logfmt(ax)
    ax.axvline(md("life_bid_ms_p50"), color=UP, lw=1.3, ls="--")
    ax.text(md("life_bid_ms_p50") * 1.15, ax.get_ylim()[1] * 0.85,
            f"中央 {md('life_bid_ms_p50'):.0f}ms", color=UP, fontsize=8.5,
            weight="bold")
    ax.set_xlabel("最良価格が保たれた時間[ms](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"② BBO lifetime\n"
                 f"中央 {md('life_bid_ms_p50'):.0f}ms・平均 "
                 f"{md('life_bid_ms_mean'):.0f}ms・p90 {md('life_bid_ms_p90'):.0f}ms",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ③ 3 つの回転率 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for k, cl, lab in [("order_turn_bid", DOWN, "order turnover[本/秒]"),
                       ("wallet_turn_bid", PURPLE, "wallet turnover[者/秒]"),
                       ("size_turn_bid", GREEN, "size turnover[回転/秒]")]:
        ax.plot(x, D[k].to_numpy(), "-", color=cl, lw=1.8, label=lab)
    ax.set_yscale("log")
    logfmt(ax, "y")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("1 秒あたり(対数)", color=INK2, fontsize=9.5)
    ax.set_title(f"③ order / wallet / size turnover\n"
                 f"注文 {md('order_turn_bid'):.2f} 本/秒・口座 "
                 f"{md('wallet_turn_bid'):.2f} 者/秒・数量 "
                 f"{md('size_turn_bid'):.2f} 回転/秒(買い)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ④ 持続曲線 ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    xv, v1 = curve("bestpx_bid")
    _, v2 = curve("touch_bid")
    ax.plot(np.arange(len(v1)), v1, "o-", color=DOWN, lw=2.2, ms=5,
            label="best-price persistence", markeredgecolor=SURFACE,
            markeredgewidth=1.0)
    ax.plot(np.arange(len(v2)), v2, "o-", color=UP, lw=2.2, ms=5,
            label="touch persistence", markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.fill_between(np.arange(len(v1)), v2, v1, color=NEUTRAL, alpha=0.7, lw=0)
    ax.set_xticks(np.arange(len(DLAB)))
    ax.set_xticklabels(DLAB, fontsize=8, rotation=45)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("時間差 Δ", color=INK2, fontsize=9.5)
    ax.set_ylabel("Δ 続く割合", color=INK2, fontsize=9.5)
    i1 = DLAB.index("500ms")
    ax.set_title(f"④ best-price と touch の持続\n"
                 f"500ms で価格 {v1[i1]:.2f} 対 中身 {v2[i1]:.2f}。"
                 f"灰色 = 価格は同じでも中身が入れ替わった分",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
    style(ax)

    # ---- ⑤ 所有時間の分布 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for nm, cl, ls, lw2, lab in [("own_bid", GREEN, "-", 2.6, "買い"),
                                 ("own_ask", WARN, "--", 1.5, "売り")]:
        xv, v = curve(nm, "sum")
        ms = xv / 1e6
        ax.step(ms[1:], (v / v.sum() * 100)[1:], where="post", color=cl, lw=lw2,
                ls=ls, label=lab)
    ax.set_xscale("log")
    logfmt(ax)
    ax.set_xlabel("1 口座が最良に居続けた時間[ms](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"⑤ touch ownership duration\n"
                 f"中央 {md('own_bid_ms_p50'):.0f}ms・平均 "
                 f"{md('own_bid_ms_mean'):.0f}ms・p90 {md('own_bid_ms_p90'):.0f}ms",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑥ 所有時間と寿命の日次推移 -----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, D["life_bid_ms_p50"].to_numpy(), "-", color=DOWN, lw=1.8,
            label="BBO lifetime(中央)")
    ax.plot(x, D["own_bid_ms_p50"].to_numpy(), "-", color=PURPLE, lw=1.8,
            label="所有時間(中央)")
    ax.plot(x, D["touch_dwell_bid_ms_p50"].to_numpy(), "-", color=GREEN, lw=1.4,
            label="1 本の注文が最良に居た時間(中央)")
    ax.set_yscale("log")
    logfmt(ax, "y")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("ミリ秒(対数)", color=INK2, fontsize=9.5)
    ax.set_title(f"⑥ 3 つの時間の日次推移(買い)\n"
                 f"寿命 {md('life_bid_ms_p50'):.0f}ms / 所有 "
                 f"{md('own_bid_ms_p50'):.0f}ms / 1 本 "
                 f"{md('touch_dwell_bid_ms_p50'):.0f}ms",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left")
    style(ax)

    # ---- ⑦ 立会日と閉場日 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    keys = ["repl_bid", "order_turn_bid", "wallet_turn_bid", "size_turn_bid"]
    lab = ["置換[回/秒]", "注文回転", "口座回転", "数量回転"]
    o = [md(k, op) for k in keys]
    c = [md(k, ~op) for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.19, o, width=0.36, color=DOWN, label=f"立会日({int(op.sum())} 日)")
    ax.bar(xs + 0.19, c, width=0.36, color=WARN, label=f"閉場日({int((~op).sum())} 日)")
    for i, (v1_, v2_) in enumerate(zip(o, c)):
        ax.annotate(f"{v1_:.2f}", (i - 0.19, v1_), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color=DOWN)
        ax.annotate(f"{v2_:.2f}", (i + 0.19, v2_), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color=WARN)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=8.5)
    ax.set_ylabel("1 秒あたり", color=INK2, fontsize=9.5)
    ax.set_title("⑦ 立会日と閉場日\n"
                 "原資産市場が開くと最良気配の入れ替わりが速くなる",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑧ 検算 -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(x, D["touch_cover_bid"].to_numpy() * 100, "-", color=GREEN, lw=1.8,
            label="買い")
    ax.plot(x, D["touch_cover_ask"].to_numpy() * 100, "-", color=WARN, lw=1.4,
            label="売り")
    ax.axhline(100, color=INK, lw=1.0, ls="--")
    ax.set_ylim(90, 101)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("被覆率[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"⑧ 検算 — 再構成した注文が最良を覆えているか\n"
                 f"買い {md('touch_cover_bid')*100:.2f}% / 売り "
                 f"{md('touch_cover_ask')*100:.2f}%(100% なら常に誰かが居る)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    fig.suptitle(f"{a.coin} 最良気配の入れ替わり(BBO turnover)"
                 f"({len(days)} 日・格子を使わず区間演算で ns 精度)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.928,
             "最良価格が保たれる時間は中央 200〜460ms で、1 秒格子では測れない。"
             "そこで格子を使わず、bbo から作った run(ある価格が最良だった区間)と "
             "l1 から作った線分(ある注文がその数量で板にあった区間)の**交差**だけで測る。\n"
             "持続はどちらも Σ max(0, L−Δ) / Σ L で定義した。価格が同じでも中身が"
             "入れ替わることがあるので、best-price persistence ≧ touch persistence になる"
             "(この包含関係が検算)。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_bbo_turnover.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
