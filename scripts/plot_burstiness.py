"""イベント到着のかたまり具合(burstiness)を図示する。

    A 生 B と畳んだ B … 同時刻を畳むと何が残るか。ポアソンの帰無も置く
    B 日ごとの推移   … 主要 4 系列の B_uniq
    C 有限標本の効き … B と補正版 A_n をイベント数に対して並べる
    D ウォレット別   … B_uniq の分布を行動クラスタで分ける

★生の B の 7〜8 割は「取引所が同じナノ秒を打っている」ことの反映で、
参加者の到着過程ではない。**B_uniq(同時刻を畳んだ版)で読むこと。**

    uv run python scripts/plot_burstiness.py --coin xyz:MU
出力: charts/<coin>_burstiness.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ORDER = ["UPDATE", "NEW", "REMOVE", "bid", "ask", "BBO"]
JP = {"UPDATE": "UPDATE(全イベント)", "NEW": "NEW(新規)", "REMOVE": "REMOVE(終端)",
      "bid": "bid(買い側)", "ask": "ask(売り側)", "BBO": "BBO(最良気配の変化)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def spread_labels(ax, ends, gap=0.06):
    lo, hi = ax.get_ylim()
    sp = hi - lo
    it = sorted(ends, key=lambda e: e[0])
    fr = [(e[0] - lo) / sp for e in it]
    for i in range(1, len(fr)):
        fr[i] = max(fr[i], fr[i - 1] + gap)
    for (y, lb, c, x), f in zip(it, fr):
        ax.annotate(lb, (x, lo + f * sp), textcoords="offset points", xytext=(7, 0),
                    fontsize=8.2, color=c, weight="bold", va="center",
                    annotation_clip=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_csv(ROOT / "data" / f"burst_daily_{tag}.csv")
    W = pl.read_csv(ROOT / "data" / f"burst_wallet_{tag}.csv")
    cl = None
    cp = ROOT / "data" / f"wallet_clusters_{tag}.csv"
    if cp.exists():
        cl = pl.read_csv(cp).select("user", "cluster")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.0, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.24, left=0.10, right=0.955,
                          top=0.845, bottom=0.075)

    # ---- A 生 B と畳んだ B ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    y = np.arange(len(ORDER))
    for lab, col, c, off in (("生の B(同時刻を含む)", "B", SLOT[1], -0.17),
                             ("B_uniq(同時刻を畳む)", "B_uniq", SLOT[0], 0.17)):
        med, lo, hi = [], [], []
        for s in ORDER:
            v = D.filter(pl.col("series") == s)[col].drop_nans().drop_nulls().to_numpy()
            q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax.hlines(y + off, lo, hi, color=c, lw=5.5, zorder=4)
        ax.plot(med, y + off, "o", color=SURFACE, ms=5, markeredgecolor=c,
                markeredgewidth=1.8, zorder=5)
        ax.annotate(lab, (hi[0], y[0] + off), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.4, color=c, weight="bold", va="center")
    nb = float(D["B_null"].median())
    ax.axvline(nb, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=3)
    ax.annotate(f"ポアソン {nb:+.3f}", (nb, len(ORDER) - 0.5),
                textcoords="offset points", xytext=(5, 0), fontsize=8.2, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([JP[s] for s in ORDER], fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("burstiness B / 帯 = 25〜75%", color=INK2, fontsize=9.5)
    ax.set_title("A  同時刻を畳むと何が残るか", loc="left", color=INK, fontsize=11.5,
                 pad=20, weight="bold")
    zs = float(D.filter(pl.col("series") == "UPDATE")["zero_share"].median())
    ax.text(0, 1.035, f"★全イベントの {zs:.0%} は直前と同じナノ秒。生の B はその反映",
            transform=ax.transAxes, color=INK2, fontsize=8.8)

    # ---- B 日ごとの推移 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    days = sorted(D["dt"].unique().to_list())
    xi = {d: i for i, d in enumerate(days)}
    ends = []
    for i, s in enumerate(["UPDATE", "NEW", "ask", "bid", "BBO"]):
        q = D.filter(pl.col("series") == s).sort("dt")
        x = np.array([xi[d] for d in q["dt"].to_list()])
        v = q["B_uniq"].to_numpy()
        ax.plot(x, v, "-o", color=SLOT[i], lw=1.5, ms=2.4, zorder=4)
        ends.append((v[-1], JP[s].split("(")[0], SLOT[i], x[-1]))
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
    spread_labels(ax, ends, gap=0.07)
    st = max(len(days) // 4, 1)
    ax.set_xticks(list(range(0, len(days), st)))
    ax.set_xticklabels([days[i][5:] for i in range(0, len(days), st)], fontsize=8.5,
                       color=MUTED)
    ax.set_xlabel("日", color=INK2, fontsize=9.5)
    ax.set_ylabel("B_uniq", color=INK2, fontsize=9.5)
    ax.set_title("B  日ごとの推移(同時刻を畳んだ版)", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, "0 = ポアソン。上ほどかたまる", transform=ax.transAxes,
            color=INK2, fontsize=8.8)

    # ---- C 有限標本の効き ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    n = W["n"].to_numpy().astype(float)
    ax.scatter(n, W["B"].to_numpy(), s=14, color=SLOT[1], alpha=0.55,
               edgecolors="none", zorder=4, label="生の B")
    ax.scatter(n, W["A_n"].to_numpy(), s=14, color=SLOT[0], alpha=0.55,
               edgecolors="none", zorder=5, label="補正版 A_n")
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("そのウォレットのイベント数(対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("burstiness", color=INK2, fontsize=9.5)
    ax.set_title("C  イベント数が少ないと B は 0 へ引き寄せられる", loc="left",
                 color=INK, fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, f"1 点 = 1 ウォレット × {W.height:,} 者。"
                      "数の違う相手を生の B で比べてはいけない",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)

    # ---- D ウォレット別 ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    if cl is not None:
        Wc = W.join(cl, on="user", how="inner")
        ks = sorted(Wc["cluster"].unique().to_list())
        bins = np.linspace(-0.2, 1.0, 40)
        for i, k in enumerate(ks):
            v = Wc.filter(pl.col("cluster") == k)["A_n"].drop_nans().to_numpy()
            h, e = np.histogram(v, bins=bins, density=True)
            c = (e[:-1] + e[1:]) / 2
            ax.plot(c, h, "-", color=SLOT[i], lw=2.2, zorder=4 + i)
            ax.annotate(f"クラスタ {k}({len(v)} 者)", (c[np.argmax(h)], h.max()),
                        textcoords="offset points", xytext=(6, 6), fontsize=8.4,
                        color=SLOT[i], weight="bold")
        sub = f"行動クラスタ別({Wc.height} 者が両方に載る)"
    else:
        v = W["A_n"].drop_nans().to_numpy()
        ax.hist(v, bins=40, color=SLOT[0], alpha=0.8)
        sub = "クラスタ情報なし"
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=3)
    ax.annotate("ポアソン", (0, ax.get_ylim()[1] * 0.95), textcoords="offset points",
                xytext=(5, 0), fontsize=8.2, color=MUTED)
    ax.set_xlabel("ウォレットの burstiness(補正版 A_n)", color=INK2, fontsize=9.5)
    ax.set_ylabel("密度", color=INK2, fontsize=9.5)
    ax.set_title("D  ウォレットごとのかたまり具合", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.8)

    fig.suptitle(f"{a.coin} イベント到着のかたまり具合(burstiness)",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.10, 0.932,
             "到着間隔 τ の平均 μ と標準偏差 σ から B = (σ−μ)/(σ+μ)。"
             "−1 = 等間隔、0 = ポアソン、+1 = 極端にかたまる。\n"
             "★イベントの 7〜8 割が直前と同じナノ秒(取引所がブロック単位で時刻を打つ)なので、"
             "参加者の到着過程を見るなら同時刻を畳んだ B_uniq を読むこと。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) l1 注文イベント + l2/bbo / 2026-05-04〜08-10",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_burstiness.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
