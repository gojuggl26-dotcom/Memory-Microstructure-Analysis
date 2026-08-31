"""メッセージの流量(quote stuffing / message activity)14 指標を図示する。

    ① updates per second          ② 種別内訳(NEW / REMOVE / UPDATE / REJECTED)
    ③ messages / orders per block  ④ inter-event duration(分散・CV も)
    ⑤ burstiness と event clustering(B-M 平面)
    ⑥ Fano factor(窓長依存)      ⑦ activity entropy
    ⑧ activity autocorrelation

    uv run python scripts/plot_msg_activity.py --coin xyz:MU
出力: charts/<coin>_msg_activity.png
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
CLS_COL = {"NEW": DOWN, "REMOVE": UP, "UPDATE": GREEN, "REJECTED": MUTED}
FANO_LAB = ["10ms", "50ms", "100ms", "500ms", "1s", "5s", "10s", "60s"]


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


def logfmt(ax, axis="y"):
    """text.parse_math=False だと対数軸の既定目盛が mathtext のまま出る。"""
    f = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)
    (ax.yaxis if axis == "y" else ax.xaxis).set_minor_formatter(
        mpl.ticker.NullFormatter())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_parquet(ROOT / "data" / f"msg_activity_daily_{tag}.parquet").sort("dt")
    C = pl.read_parquet(ROOT / "data" / f"msg_activity_curves_{tag}.parquet")
    days = D["dt"].to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    op = np.array([d in sess for d in days])
    x = np.arange(len(days))
    xt = [0, len(days) // 2, len(days) - 1]
    xl = [days[i][5:] for i in xt]
    wt = D["n_msg"].to_numpy().astype(float)

    def wm(k):
        v = D[k].to_numpy().astype(float)
        m = np.isfinite(v)
        return float(np.average(v[m], weights=wt[m]))

    def md(k):
        # ★形の統計(CV・B・Fano など)は裾が重く、加重平均だと 1 日に引っぱられる。
        #   日ごとの中央値で要約する。
        v = D[k].to_numpy().astype(float)
        return float(np.nanmedian(v))

    def curve(kind, agg="mean"):
        g = (C.filter(pl.col("kind") == kind).group_by("x", "xv")
             .agg(v=pl.col("v").sum() if agg == "sum" else pl.col("v").mean())
             .sort("xv"))
        return g["xv"].to_numpy(), g["v"].to_numpy(), g["x"].to_list()

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 9.8), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.50, wspace=0.30,
                          left=0.05, right=0.985, top=0.790, bottom=0.085)

    # ---- ① updates per second ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(x, D["msg_per_sec_p50"].to_numpy(), D["msg_per_sec_p99"].to_numpy(),
                    color=DOWN, alpha=0.18, lw=0, label="日内 p50〜p99")
    ax.plot(x, D["msg_per_sec"].to_numpy(), "-", color=DOWN, lw=2.0, label="平均")
    ax.plot(x, D["msg_per_sec_max"].to_numpy(), "-", color=UP, lw=1.0, alpha=0.8,
            label="その日の最大")
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("メッセージ / 秒(対数)", color=INK2, fontsize=9.5)
    ax.set_title(f"① updates per second\n"
                 f"平均 {wm('msg_per_sec'):,.0f} 通/秒、日内 p99 "
                 f"{wm('msg_per_sec_p99'):,.0f}、最大 {D['msg_per_sec_max'].max():,.0f}",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    # ---- ② 種別内訳 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for c in ["NEW", "REMOVE", "REJECTED", "UPDATE"]:
        ax.plot(x, D[f"{c}_per_sec"].to_numpy(), "-", color=CLS_COL[c], lw=1.8,
                label=f"{c} {wm(f'{c}_share')*100:.1f}%")
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("メッセージ / 秒(対数)", color=INK2, fontsize=9.5)
    ax.set_title(f"② NEW / REMOVE / UPDATE / REJECTED\n"
                 f"約定 1 件あたり {md('msg_to_trade'):,.0f} 通、"
                 f"新規発注は {md('order_to_trade'):,.0f} 本",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=2)
    style(ax)

    # ---- ③ messages / orders per block ------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    xv, v, lab = curve("mpb_hist", "sum")
    tot = v.sum()
    ax.bar(np.arange(len(v)), v / tot * 100, color=DOWN, width=0.72)
    ax.set_xticks(np.arange(len(v)))
    ax.set_xticklabels([f"{int(z)}" for z in xv], fontsize=7.5, rotation=45)
    ax.set_xlabel("1 ブロックのメッセージ数(以上)", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"③ messages / orders per block\n"
                 f"平均 {md('mpb_mean'):.1f} 通・最大 {int(D['mpb_max'].max())}・"
                 f"CV {md('mpb_cv'):.2f}(動いた注文 {md('opb_mean'):.1f} 本)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ④ inter-event duration -------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    xv, v, lab = curve("gap_hist", "sum")
    ms = np.array(xv) / 1e6
    ax.step(ms[1:], (v / v.sum() * 100)[1:], where="post", color=PURPLE, lw=1.8)
    ax.set_xscale("log")
    logfmt(ax, "x")
    ax.axvline(md("gap_p50_ms"), color=UP, lw=1.4, ls="--")
    ax.text(md("gap_p50_ms") * 1.2, ax.get_ylim()[1] * 0.86,
            f"中央 {md('gap_p50_ms'):.1f}ms", color=UP, fontsize=8.5, weight="bold")
    ax.axvline(md("block_base_ms"), color=GREEN, lw=1.4, ls=":")
    ax.text(md("block_base_ms") * 0.93, ax.get_ylim()[1] * 0.5,
            f"ブロック周期\n{md('block_base_ms'):.1f}ms", color=GREEN, fontsize=8.5,
            ha="right", weight="bold")
    ax.set_xlabel("ブロック間隔[ms](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"④ inter-event duration — チェーンの刻みで量子化\n"
                 f"中央 {md('gap_p50_ms'):.1f}ms・σ {md('gap_sd_ms'):.0f}ms・"
                 f"分散 {md('gap_var_ms2'):,.0f}ms²・CV {md('cv'):.2f}",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑤ burstiness と memory(B-M 平面)---------------------------------
    ax = fig.add_subplot(gs[1, 0])
    B, M = D["burstiness"].to_numpy(), D["memory"].to_numpy()
    ax.scatter(B[op], M[op], s=22, color=DOWN, alpha=0.8, label="立会日",
               edgecolors=SURFACE, linewidths=0.6)
    ax.scatter(B[~op], M[~op], s=22, color=WARN, alpha=0.8, label="閉場日",
               edgecolors=SURFACE, linewidths=0.6)
    ax.axhline(0, color=INK, lw=1.0)
    ax.axvline(0, color=INK, lw=1.0)
    ax.scatter([0], [0], s=90, marker="*", color=INK, zorder=5)
    ax.annotate("ポアソン過程", (0, 0), textcoords="offset points", xytext=(8, -14),
                fontsize=8.5, color=INK, weight="bold")
    ax.set_xlabel("burstiness B", color=INK2, fontsize=9.5)
    ax.set_ylabel("memory M(連続する間隔の相関)", color=INK2, fontsize=9.5)
    ax.set_title(f"⑤ update burstiness と event clustering\n"
                 f"B {md('burstiness'):+.3f} / M {md('memory'):+.3f}。"
                 f"ブロック周期が下限を作り B は 0 付近に張り付く",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑥ Fano factor -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for kind, cl, lab2 in [("fano", DOWN, "1 日全体"),
                           ("fano_busy", GREEN, "その日の最繁 1 時間")]:
        xv, v, _ = curve(kind)
        ax.plot(np.arange(len(v)), v, "o-", color=cl, lw=2.0, ms=4.5, label=lab2,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.axhline(1.0, color=UP, lw=1.6, ls="--")
    ax.text(len(FANO_LAB) - 0.4, 1.25, "ポアソンなら 1", ha="right", color=UP,
            fontsize=8.5, weight="bold")
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_xticks(np.arange(len(FANO_LAB)))
    ax.set_xticklabels(FANO_LAB, fontsize=8, rotation=45)
    ax.set_xlabel("窓長", color=INK2, fontsize=9.5)
    ax.set_ylabel("Fano factor(分散/平均、対数)", color=INK2, fontsize=9.5)
    ax.set_title("⑥ Fano factor — 窓長で桁が変わる\n"
                 "ブロック間隔(中央 72.6ms)より短い窓は「ブロックが入ったか」を測る",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑦ activity entropy ------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, D["entropy_norm"].to_numpy(), "-", color=DOWN, lw=2.0,
            label="正規化エントロピー")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("H / ln(86400)", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(x, D["top1pct_share"].to_numpy() * 100, "-", color=WARN, lw=1.4)
    ax2.set_ylabel("最も忙しい 1% の秒が占める割合[%]", color=WARN, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"⑦ activity entropy\n"
                 f"中央 {md('entropy_norm'):.3f}(1 なら完全に平ら)。"
                 f"橙 = 上位 1% の秒に {md('top1pct_share')*100:.1f}%",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑧ activity autocorrelation ---------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    xv, v, _ = curve("acf")
    ax.plot(xv, v, "o-", color=PURPLE, lw=2.0, ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xscale("log")
    logfmt(ax, "x")
    ax.set_xlabel("ラグ[秒](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("1 秒ごとの計数の自己相関", color=INK2, fontsize=9.5)
    ax.set_title(f"⑧ activity autocorrelation — 1 時間先まで尾を引く\n"
                 f"ラグ 1 秒 {md('acf1'):.3f} → 60 秒 {md('acf60'):.3f} → "
                 f"3600 秒でも {v[-1]:.2f}",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    fig.suptitle(f"{a.coin} メッセージの流量(quote stuffing / message activity)"
                 f"({len(days)} 日・{int(D['n_msg'].sum()):,} 通)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.928,
             "板を作らないので除外は一切しない。トリガー注文もテイカーも Rejected も、"
             "流れてきた以上はメッセージである(板の再構成を伴う他のレポートとはここが違う)。\n"
             "ts はブロックの時刻で、同じブロックの事象は全て同じ ns を持つ。"
             "したがってメッセージ単位の間隔は同一ブロック内で 0 になり意味を持たないため、"
             "間隔の統計はすべてブロック間隔で測っている。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses / 取引数は node_fills",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_msg_activity.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
