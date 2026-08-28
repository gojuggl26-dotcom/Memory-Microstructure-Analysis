"""100ms 窓の分散が将来 log リターンを説明するかを図示する。

    A 絶対リターンへの相関 … 分散が説明しうるのはこちら
    B 符号つきリターンへの相関 … 分散は符号を持たないので構造的に 0 のはず
    C t 値のホライズン依存 … 重なる窓(全標本)と重ならない部分標本の差
    D 動いた窓に絞ると何が変わるか … 100ms 窓の 8 割は最良気配が動かない

色は特徴量の識別。dataviz の参照パレット slot 1〜4 で、palette_check 済み
(隣接ペアの CVD ΔE は最小 9.1、通常視は最小 22.9)。ただし aqua と yellow は
背景との対比が 3:1 未満なので、**relief rule に従い全系列に直接ラベル**を置く。

    uv run python scripts/plot_var100.py --coin xyz:MU [--src l1]
出力: charts/<coin>_var100.png / <coin>_var100_l1.png
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
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LAB = {"var_bid_depth": "買い最良数量", "var_ask_depth": "売り最良数量",
       "var_obi": "OBI", "var_ofi": "OFI",
       "var_bid_exp10": "買い 10 レベル指数減衰", "var_ask_exp10": "売り 10 レベル指数減衰",
       "var_bid_sum10": "買い 10 レベル単純和", "var_ask_sum10": "売り 10 レベル単純和"}
NCELL = 2304          # 多重比較の規模(feat 4 × lay 2 × xform 2 × ykind 3 × 日区分 2 × ホライズン 12 × 標本 2)
BONF_T = 4.24         # 両側 0.05 / 2304 に対応する正規分位


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


def xaxis(ax):
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["10ms", "100ms", "1s", "10s", "100s"], fontsize=8.5,
                       color=MUTED)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("予測ホライズン(対数)", color=INK2, fontsize=9.5)


def series(R, feats, **f):
    q = R
    for k, v in f.items():
        q = q.filter(pl.col(k) == v)
    for i, ft in enumerate(feats):
        s = q.filter(pl.col("feat") == ft).sort("hor_ms")
        if s.height:
            yield i, ft, s


def place_labels(ax, ends):
    """終端の直接ラベルが重ならないように縦へ散らす。"""
    lo, hi = ax.get_ylim()
    span = hi - lo
    items = sorted(ends, key=lambda e: e[0])
    fr = [(e[0] - lo) / span for e in items]
    for i in range(1, len(fr)):
        if fr[i] - fr[i - 1] < 0.055:
            fr[i] = fr[i - 1] + 0.055
    for (y, lb, c, x), f in zip(items, fr):
        ax.annotate(lb, (x, lo + f * span), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.3, color=c, weight="bold",
                    va="center", annotation_clip=False)


def line_panel(ax, R, feats, col, title, sub, ylab, *, zero=True, **f):
    style(ax)
    if zero:
        ax.axhline(0, color=BASELINE, lw=1.1, zorder=2)
    ends = []
    for i, ft, s in series(R, feats, **f):
        x, y = s["hor_ms"].to_numpy(), s[col].to_numpy()
        ax.plot(x, y, "-o", color=SLOT[i], lw=2.0, ms=4.5, zorder=4 + i,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        ends.append((y[-1], LAB[ft], SLOT[i], x[-1]))
    xaxis(ax)
    ax.set_xlim(8, 8e5)
    place_labels(ax, ends)                         # relief rule: 直接ラベル
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.8)


def pair_panel(ax, R, feats, col, title, sub, ylab, split, opts, opt_lab,
               *, logy=False, **f):
    """1 つの軸に「実線 = opts[0] / 破線 = opts[1]」で 2 条件を重ねる。"""
    style(ax)
    if not logy:
        ax.axhline(0, color=BASELINE, lw=1.1, zorder=2)
    ends = []
    for j, (val, ls) in enumerate(zip(opts, ("-", (0, (4, 2))))):
        for i, ft, s in series(R, feats, **{**f, split: val}):
            x, y = s["hor_ms"].to_numpy(), s[col].to_numpy()
            ax.plot(x, y, linestyle=ls, color=SLOT[i], lw=2.0 if j == 0 else 1.4,
                    alpha=1.0 if j == 0 else 0.75, zorder=4 + i)
            if j == 0:
                ends.append((y[-1], LAB[ft], SLOT[i], x[-1]))
    xaxis(ax)
    ax.set_xlim(8, 8e5)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    place_labels(ax, ends)
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.8)
    h = [plt.Line2D([], [], color=MUTED, ls=s, lw=2.0) for s in ("-", (0, (4, 2)))]
    ax.legend(h, list(opt_lab), loc="upper left", frameon=False, fontsize=8.5,
              labelcolor=INK2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--src", default="bbo", choices=["bbo", "l1"])
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    fn = f"var100_ols_{tag}" if a.src == "bbo" else f"var100_l1_ols_{tag}"
    R = pl.read_parquet(ROOT / "data" / f"{fn}.parquet")
    feats = [f for f in LAB if f in set(R["feat"].to_list())]
    base = dict(day_type="立会日", x_form="log1p")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.0, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.24,
                          left=0.062, right=0.93, top=0.845, bottom=0.07)

    line_panel(fig.add_subplot(gs[0, 0]), R, feats, "r",
               "A  100ms 窓の分散 → 将来の絶対リターン",
               "動いた窓のみ / log(1+x) / 立会日 / 相関(= 標準化した傾き。二乗が決定係数)",
               "相関", y_kind="絶対値", lay="動いた窓", sample="full", **base)
    line_panel(fig.add_subplot(gs[0, 1]), R, feats, "r",
               "B  同じ説明変数 → 符号つきリターン",
               "買い側と売り側がほぼ鏡像になる。OBI と OFI は左右対称な量なので 0 のまま",
               "相関", y_kind="符号つき", lay="動いた窓", sample="full", **base)
    pair_panel(fig.add_subplot(gs[1, 0]), R, feats, "t",
               "C  重なる窓は t 値を水増しする",
               f"絶対リターン / 破線 = 全標本(窓が重なる)、実線 = 重ならない部分標本",
               "t 値(対数)", "sample", ("sub", "full"),
               ("重ならない部分標本", "全標本"), logy=True,
               y_kind="絶対値", lay="動いた窓", **base)
    pair_panel(fig.add_subplot(gs[1, 1]), R, feats, "r",
               "D  生の分散では関係が見えない(裾の重さ)",
               "絶対リターン / 破線 = 生の分散、実線 = log(1+x)",
               "相関", "x_form", ("log1p", "生"), ("log(1+x)", "生の分散"),
               y_kind="絶対値", lay="動いた窓", sample="full", day_type="立会日")

    ax = fig.axes[2]
    ax.axhline(BONF_T, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=3)
    ax.annotate(f"Bonferroni |t| = {BONF_T:.2f}({NCELL:,} セル)。全 48 本がこの上",
                (12, BONF_T), textcoords="offset points", xytext=(0, 5),
                fontsize=8.0, color=MUTED)

    n_tot = int(R.filter((pl.col("sample") == "full") & (pl.col("lay") == "動いた窓")
                         & (pl.col("hor") == "10ms") & (pl.col("y_kind") == "絶対値")
                         & (pl.col("x_form") == "log1p"))["n"].max() or 0)
    src = "最良気配の数量・OBI・OFI(l2/bbo)" if a.src == "bbo" \
        else "上位 10 レベルの depth(l1 から板を再構成)"
    fig.suptitle(f"{a.coin} 100ms 窓の分散は将来のリターンを説明するか — {src}",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.062, 0.932,
             "10ms 格子の 10 点の標本分散を説明変数にし、窓の終端 T から先の log リターンへ OLS で当てた。"
             f"立会日の「動いた窓」は 1 ホライズンあたり最大 {n_tot:,} 窓。\n"
             "x は T で確定し y の期間は (T, T+h] なので先読みは無い。"
             "標本が 2,500 万窓あるため t 値は意味を持たない(r = 0.01 でも有意になる)。効果の大きさで読むこと。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) / 2026-05-04〜08-09 の 98 日",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / (f"{tag}_var100.png" if a.src == "bbo"
                             else f"{tag}_var100_l1.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
