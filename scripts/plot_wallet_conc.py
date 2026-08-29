"""板の数量が何者の口座に集まっているか(wallet concentration)を図示する。

要求された 13 指標を 1 枚に並べる。

    ① 口座数(active / bid / ask / BBO)   ② HHI と実効口座数
    ③ エントロピー                          ④ ジニ係数
    ⑤ 上位 k のシェア                       ⑥ 集中の偏り(買い − 売り)
    ⑦ touch と deep の集中                  ⑧ 日内の推移

    uv run python scripts/plot_wallet_conc.py --coin xyz:MU
出力: charts/<coin>_wallet_conc.png
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


def band(ax, x, D, key, color, label, lw=1.9):
    """平均線と 10〜90% の帯。日ごとの分布を潰さずに出す。"""
    ax.fill_between(x, D[f"{key}_p10"].to_numpy(), D[f"{key}_p90"].to_numpy(),
                    color=color, alpha=0.16, lw=0)
    ax.plot(x, D[f"{key}_mean"].to_numpy(), "-", color=color, lw=lw, label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_parquet(ROOT / "data" / f"wallet_conc_daily_{tag}.parquet").sort("dt")
    H = pl.read_parquet(ROOT / "data" / f"wallet_conc_hourly_{tag}.parquet")
    days = D["dt"].to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    op = np.array([d in sess for d in days])
    x = np.arange(len(days))
    xt = [0, len(days) // 2, len(days) - 1]
    xl = [days[i][5:] for i in xt]

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 9.8), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.50, wspace=0.30,
                          left=0.05, right=0.985, top=0.790, bottom=0.085)

    def mean_of(k):
        return float(np.average(D[f"{k}_mean"].to_numpy(),
                                weights=D["n_grid"].to_numpy()))

    # ---- ① 口座数 --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for k, cl, lab in [("n_active", DOWN, "active(買いか売り)"),
                       ("n_bid", GREEN, "bid"), ("n_ask", WARN, "ask")]:
        band(ax, x, D, k, cl, lab)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("口座数", color=INK2, fontsize=9.5)
    ax.set_title(f"① 板に数量を置いている口座数\n"
                 f"平均 active {mean_of('n_active'):.0f} 者 / bid {mean_of('n_bid'):.0f}"
                 f" / ask {mean_of('n_ask'):.0f}。帯は日内の 10〜90%",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ② BBO 口座数 ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    band(ax, x, D, "n_bbo", PURPLE, "BBO 口座数")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("口座数", color=INK2, fontsize=9.5)
    ax.set_ylim(0, max(4.0, float(D["n_bbo_p90"].max()) * 1.1))
    ax2 = ax.twinx()
    ax2.plot(x, D["touch_share_mean"].to_numpy() * 100, "--", color=MUTED, lw=1.2)
    ax2.set_ylabel("最良気配が板全体に占める数量[%]", color=MUTED, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"② ★最良気配を持つのは平均 {mean_of('n_bbo'):.2f} 者だけ\n"
                 f"点線 = 最良気配の数量が板に占める割合(平均 "
                 f"{mean_of('touch_share')*100:.2f}%)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ③ HHI と実効口座数 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    band(ax, x, D, "hhi", DOWN, "HHI")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("wallet HHI", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(x, D["eff_n_mean"].to_numpy(), "-", color=GREEN, lw=1.6)
    ax2.set_ylabel("実効口座数 exp(H)", color=GREEN, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"③ wallet HHI と実効口座数\n"
                 f"HHI 平均 {mean_of('hhi'):.3f}(= 均等なら {1/mean_of('hhi'):.0f} 者相当)"
                 f"、緑 = exp(エントロピー) {mean_of('eff_n'):.1f} 者",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ④ ジニ ----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    band(ax, x, D, "gini", PURPLE, "Gini")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("Gini 係数", color=INK2, fontsize=9.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"④ Gini concentration\n"
                 f"平均 {mean_of('gini'):.3f}。母集団はその瞬間に残高が正の口座だけ",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑤ 上位 k のシェア -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ks = [1, 3, 5, 10]
    v = [mean_of(f"top{k}") * 100 for k in ks]
    lo = [float(np.average(D[f"top{k}_p10"].to_numpy(),
                           weights=D["n_grid"].to_numpy())) * 100 for k in ks]
    hi = [float(np.average(D[f"top{k}_p90"].to_numpy(),
                           weights=D["n_grid"].to_numpy())) * 100 for k in ks]
    xs = np.arange(len(ks))
    ax.bar(xs, v, color=DOWN, width=0.62)
    ax.errorbar(xs, v, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for i, s in enumerate(v):
        ax.annotate(f"{s:.1f}%", (i, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("板の数量に占めるシェア[%]", color=INK2, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_title("⑤ 上位 k 口座のシェア\n"
                 "棒 = 期間平均、ひげ = 日内の 10〜90% を日で平均したもの",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑥ 集中の偏り -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    band(ax, x, D, "conc_imb", WARN, "(HHI_bid − HHI_ask)/(和)")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("concentration imbalance", color=INK2, fontsize=9.5)
    mi = mean_of("conc_imb")
    ax.set_title(f"⑥ concentration imbalance\n"
                 f"平均 {mi:+.3f} — {'売り' if mi < 0 else '買い'}側のほうが集中している",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑦ touch と deep -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    band(ax, x, D, "hhi_touch", UP, "touch(最良気配)")
    band(ax, x, D, "hhi_deep", DOWN, "deep(最良以外)")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("HHI", color=INK2, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"⑦ ★最良気配だけ桁違いに集中している\n"
                 f"touch {mean_of('hhi_touch'):.3f} 対 deep {mean_of('hhi_deep'):.3f}"
                 f"(deep は全体とほぼ同じ)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right")
    style(ax)

    # ---- ⑧ 日内の推移 -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    Hh = H.with_columns(op=pl.col("dt").is_in(sorted(sess)))
    for flag, cl, lab in [(True, DOWN, "立会日"), (False, WARN, "閉場日")]:
        g = (Hh.filter(pl.col("op") == flag).group_by("hour")
             .agg(hhi=pl.col("hhi").mean(), n=pl.col("n_active").mean()).sort("hour"))
        ax.plot(g["hour"].to_numpy(), g["hhi"].to_numpy(), "o-", color=cl, lw=1.9,
                ms=3.5, label=lab, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.axvline(13.5, color=MUTED, lw=1.0, ls=":")
    ax.text(13.8, ax.get_ylim()[1] * 0.97, "米国市場の寄付き", color=MUTED,
            fontsize=8, va="top")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("時(UTC)", color=INK2, fontsize=9.5)
    ax.set_ylabel("wallet HHI", color=INK2, fontsize=9.5)
    ho = float(Hh.filter(pl.col("op"))["hhi"].mean())
    hc = float(Hh.filter(~pl.col("op"))["hhi"].mean())
    ax.set_title(f"⑧ 日内の集中度\n"
                 f"立会日 {ho:.4f} 対 閉場日 {hc:.4f} — "
                 f"{'立会日' if ho > hc else '閉場日'}のほうが集中している",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    fig.suptitle(f"{a.coin} 板の数量は何者の口座に集まっているか"
                 f"({len(days)} 日・1 秒ごとに {int(D['n_grid'].sum()):,} 回の板を復元)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.928,
             "各時点で板に数量を置いている口座 i の残高 Q_i からシェア w_i = Q_i / ΣQ_j を作り、"
             "HHI = Σ w_i^2 ほかを算出。分母はその瞬間に板にある数量であって、出された注文の総量ではない。\n"
             "母集団はトリガー注文とテイカー(Ioc など)を除いた Alo / Gtc の指値。reduce_only は板に載るので含める。"
             "指値の寿命は中央値 565ms なので、1 秒格子に写るのは滞留している注文だけである。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_wallet_conc.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
