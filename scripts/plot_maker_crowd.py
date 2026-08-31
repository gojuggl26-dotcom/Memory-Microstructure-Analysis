"""最良気配近傍のメイカーの混み具合と競争(maker crowd / competition)を図示する。

    ① 近傍のメイカー数(買い / 売り)   ② HHI と実効メイカー数 N_eff
    ③ maker competition index          ④ 帯 K を広げたときの変わり方
    ⑤ 入替・参入・退出 vs 時間差 Δ     ⑥ 入替の日次推移
    ⑦ quote overlap と clustering      ⑧ 立会日と閉場日

    uv run python scripts/plot_maker_crowd.py --coin xyz:MU
出力: charts/<coin>_maker_crowd.png
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
BANDS = [0, 5, 10, 25]
KMAIN = 10
DELTAS = [1, 5, 10, 60, 300]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_parquet(ROOT / "data" / f"maker_crowd_daily_{tag}.parquet").sort("dt")
    C = pl.read_parquet(ROOT / "data" / f"maker_crowd_curves_{tag}.parquet")
    days = D["dt"].to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    op = np.array([d in sess for d in days])
    x = np.arange(len(days))
    xt = [0, len(days) // 2, len(days) - 1]
    xl = [days[i][5:] for i in xt]

    def md(k, sub=None):
        # 日次の代表値は中央値。列名は指標そのまま(turnover60 等)か "_mean" 付き
        col = k if k in D.columns else f"{k}_mean"
        v = D[col].to_numpy().astype(float)
        if sub is not None:
            v = v[sub]
        return float(np.nanmedian(v))

    def band(metric):
        g = (C.filter((pl.col("kind") == "band") & (pl.col("metric") == metric))
             .group_by("K").agg(v=pl.col("mean").median()).sort("K"))
        return g["K"].to_numpy(), g["v"].to_numpy()

    def delta(metric):
        g = (C.filter((pl.col("kind") == "delta") & (pl.col("metric") == metric))
             .group_by("p50").agg(v=pl.col("mean").median()).sort("p50"))
        return g["p50"].to_numpy(), g["v"].to_numpy()

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 9.8), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.50, wspace=0.30,
                          left=0.05, right=0.985, top=0.790, bottom=0.085)

    def series(ax, key, color, label, lw=1.9):
        ax.fill_between(x, D[f"{key}_p10"].to_numpy(), D[f"{key}_p90"].to_numpy(),
                        color=color, alpha=0.15, lw=0)
        ax.plot(x, D[f"{key}_mean"].to_numpy(), "-", color=color, lw=lw, label=label)

    # ---- ① 近傍のメイカー数 -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    series(ax, "n_maker", DOWN, "買い + 売り")
    ax.plot(x, D["n_maker_bid_mean"].to_numpy(), "-", color=GREEN, lw=1.4, label="買い")
    ax.plot(x, D["n_maker_ask_mean"].to_numpy(), "-", color=WARN, lw=1.4, label="売り")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("口座数", color=INK2, fontsize=9.5)
    ax.set_title(f"① number of makers near touch\n"
                 f"最良から {KMAIN} ティック以内に中央 {md('n_maker'):.1f} 者"
                 f"(買い {md('n_maker_bid'):.1f} / 売り {md('n_maker_ask'):.1f})",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ② HHI と N_eff -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    series(ax, "hhi", UP, "maker concentration (HHI)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("HHI", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(x, D["eff_n_mean"].to_numpy(), "-", color=DOWN, lw=1.6)
    ax2.set_ylabel("実効メイカー数 N_eff = 1/HHI", color=DOWN, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"② maker concentration と effective maker count\n"
                 f"HHI {md('hhi'):.3f} → N_eff {md('eff_n'):.2f} 者。"
                 f"上位 1 者が {md('top1')*100:.0f}%",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ③ maker competition index -----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    series(ax, "mci", PURPLE, "MCI = N_eff / N")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax.text(len(days) * 0.02, 1.02, "1 = 全員が同じ数量(互角)", color=INK2,
            fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("maker competition index", color=INK2, fontsize=9.5)
    ax.set_title(f"③ maker competition index\n"
                 f"中央 {md('mci'):.3f} — 近傍に {md('n_maker'):.1f} 者居ても"
                 f"実質 {md('eff_n'):.2f} 者ぶんの力しかない",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ④ 帯を広げると ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    kk, nm = band("n_maker")
    ax.plot(np.arange(len(kk)), nm, "o-", color=DOWN, lw=2.0, ms=5, label="メイカー数",
            markeredgecolor=SURFACE, markeredgewidth=1.0)
    kk2, ne = band("eff_n")
    ax.plot(np.arange(len(kk2)), ne, "o-", color=GREEN, lw=2.0, ms=5,
            label="実効メイカー数 N_eff", markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xticks(np.arange(len(kk)))
    ax.set_xticklabels([f"{int(k)}" for k in kk])
    ax.set_xlabel("最良からの距離の上限[ティック]", color=INK2, fontsize=9.5)
    ax.set_ylabel("口座数", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    kk3, mc = band("mci")
    ax2.plot(np.arange(len(kk3)), mc, "s--", color=PURPLE, lw=1.6, ms=4.5)
    ax2.set_ylabel("MCI(紫・破線)", color=PURPLE, fontsize=8.5)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"④ ★帯を広げても実効メイカー数は増えない\n"
                 f"0 → 25 ティックで人数 {nm[0]:.1f} → {nm[-1]:.1f} 者、"
                 f"N_eff は {ne[0]:.2f} → {ne[-1]:.2f} 止まり",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)

    # ---- ⑤ 入替・参入・退出 vs Δ --------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    # ★参入率と退出率は定常過程なら一致する。実際ぴたりと重なるので、
    #   退出のほうを破線にして両方見えるようにする(この一致自体が検算になる)。
    for met, cl, ls, lw2, lab in [("turnover", DOWN, "-", 2.0, "maker turnover"),
                                  ("entry", GREEN, "-", 2.6, "maker entry rate"),
                                  ("exit", WARN, "--", 1.6, "maker exit rate")]:
        dd, vv = delta(met)
        ax.plot(dd, vv, ls, color=cl, lw=lw2, marker="o", ms=4.5, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks(DELTAS)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}s"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("時間差 Δ", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合", color=INK2, fontsize=9.5)
    dd, tv = delta("turnover")
    ax.set_title(f"⑤ maker turnover / entry / exit\n"
                 f"1 秒で {tv[0]:.2f}、60 秒で {tv[list(dd).index(60)]:.2f} 入れ替わる。"
                 f"参入と退出は一致(定常の検算)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    # ---- ⑥ 入替の日次推移 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, D["turnover60"].to_numpy(), "-", color=DOWN, lw=1.9, label="入替(60 秒)")
    ax.plot(x, D["entry60"].to_numpy(), "-", color=GREEN, lw=1.4, label="参入")
    ax.plot(x, D["exit60"].to_numpy(), "-", color=WARN, lw=1.4, label="退出")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("割合(60 秒あたり)", color=INK2, fontsize=9.5)
    ax.set_title(f"⑥ 顔ぶれの入れ替わり(60 秒)\n"
                 f"入替 {md('turnover60'):.3f} / 参入 {md('entry60'):.3f} / "
                 f"退出 {md('exit60'):.3f}",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑦ overlap と clustering --------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    series(ax, "overlap", DOWN, "quote overlap(1 価格あたりの口座数)")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("1 価格あたりの口座数", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(x, D["clustering_mean"].to_numpy(), "-", color=PURPLE, lw=1.6)
    ax2.set_ylabel("maker clustering(紫)", color=PURPLE, fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title(f"⑦ quote overlap と maker clustering\n"
                 f"1 価格に {md('overlap'):.2f} 者(破線 1 = 独占)。"
                 f"1 者の価格集中度は {md('clustering'):.2f}",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)

    # ---- ⑧ 立会日と閉場日 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    keys = ["n_maker", "eff_n", "mci", "hhi", "overlap", "clustering", "turnover60"]
    lab = ["メイカー数", "N_eff", "MCI", "HHI", "重なり", "集中度", "入替60s"]
    o = [md(k, op) for k in keys]
    c = [md(k, ~op) for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.19, o, width=0.36, color=DOWN, label=f"立会日({int(op.sum())} 日)")
    ax.bar(xs + 0.19, c, width=0.36, color=WARN, label=f"閉場日({int((~op).sum())} 日)")
    for i, (v1, v2) in enumerate(zip(o, c)):
        ax.annotate(f"{v1:.2f}", (i - 0.19, v1), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.5, color=DOWN)
        ax.annotate(f"{v2:.2f}", (i + 0.19, v2), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=6.5, color=WARN)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=8, rotation=30, ha="right")
    ax.set_title("⑧ 立会日と閉場日\n"
                 "原資産市場が開くとメイカーは増えるが、力の偏りは変わらない",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    fig.suptitle(f"{a.coin} 最良気配の近くにいるメイカーの混み具合と競争"
                 f"({len(days)} 日・1 秒ごとに板を復元)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.928,
             f"「近傍」= 同じ側の最良気配から {KMAIN} ティック以内(④ だけ 0 / 5 / 10 / 25 を比べる)。"
             "HHI と N_eff は近傍の数量シェアで測る。MCI = N_eff / N は均等度で、HHI とは別物である"
             "(2 者しか居なくても均等なら 1)。\n"
             "quote overlap は「同じ価格に何者居るか」、maker clustering は「1 者が自分の数量を"
             "何価格に散らしているか」で、向きが逆の指標である。母集団は Alo / Gtc の指値"
             "(トリガーとテイカーを除き、reduce_only は含む)。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_maker_crowd.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
