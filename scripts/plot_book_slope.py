"""Book Slope と将来 log リターンの関係、および OLS / GLS の比較を描く。

4 枚:
  左上 帯ごとの平均 log リターン。**関係が線形でないこと**が一目で判る
  右上 傾き β をホライズン別に。OLS / GLS(AR(1)) / 重ならない部分標本
  左下 t 値。古典的な標準誤差と HAC(Newey–West)の差
  右下 決定係数。BookSlope と OBI の比較

配色は scripts/palette_check.py で検証済み:
  左上は k が順序を持つので **順序ランプ**(青 4 段、明度 0.764→0.338 と単調、
  最も明るい段の対比 2.06 ≥ 2.0)。順序ランプにカテゴリの明度帯は当てない。
  他は系列 1〜3(青/橙/aqua)。CVD ΔE は最小 9.2、通常視は最小 24.0。
  aqua は対比 2.74 < 3 なので **全系列に直接ラベル**を付ける(relief rule)。

    uv run python scripts/plot_book_slope.py --coin xyz:MU
出力: charts/<coin>_book_slope.png
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
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # 順序ランプ(k 用)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
HOR = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300, 500]
SHOW_K = [1, 10, 100, 500]
EDGES = [-3, -2, -1.5, -1, -0.5, -0.1, 0.1, 0.5, 1, 1.5, 2, 3]
LABELS = ["< −3"] + [f"{EDGES[i]}〜{EDGES[i+1]}" for i in range(len(EDGES) - 1)] + ["> +3"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    F = pl.read_parquet(ROOT / "data" / f"book_slope_fits_{tag}.parquet")
    B = pl.read_parquet(ROOT / "data" / f"book_slope_bins_{tag}.parquet")
    meta = pl.read_parquet(ROOT / "data" / f"book_slope_meta_{tag}.parquet")
    half = float(meta["half_spread_bp_median"][0])

    Fs = F.filter((pl.col("x") == "BookSlope_z") & (pl.col("day_type") == "立会日")).sort("k")
    Fo = F.filter((pl.col("x") == "OBI") & (pl.col("day_type") == "立会日")).sort("k")
    Bs = B.filter((pl.col("x") == "BookSlope_z") & (pl.col("day_type") == "立会日"))

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.4), dpi=160)
    fig.subplots_adjust(left=0.065, right=0.955, top=0.805, bottom=0.075,
                        hspace=0.46, wspace=0.24)
    for ax in axes.ravel():
        style(ax)

    # ---- 左上: 帯ごとの平均 log リターン --------------------------------------
    ax = axes[0, 0]
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    for c, k in zip(RAMP, SHOW_K):
        r = Bs.filter(pl.col("k") == k).sort("bin_i")
        xs, ys = r["bin_i"].to_numpy(), r["mean_bp"].to_numpy()
        ax.plot(xs, ys, "-o", color=c, lw=2.1, ms=5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"k={k}", xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=c, fontsize=10, weight="bold", va="center")
    big = Bs.filter((pl.col("k") == 10) & (pl.col("bin_i") == 6))
    nbig = int(big["n"][0]); ntot = int(Bs.filter(pl.col("k") == 10)["n"].sum())
    ax.annotate(f"この 1 帯に {nbig/1e6:.1f}M 件\n= 全体の {nbig/ntot*100:.0f}%",
                xy=(6, 0), xytext=(6, -0.62), ha="center", color=INK, fontsize=9,
                weight="bold", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlim(-0.5, len(LABELS) + 0.6)
    ax.set_xlabel("BookSlope_z の帯(σ 単位)", color=INK2, fontsize=10)
    ax.set_ylabel("k イベント後までの平均 log リターン[bp]", color=INK2, fontsize=10)
    ax.set_title("★関係は線形ではない —— 中央で急、両端で寝る\n"
                 "直線を当てると中央の大きな塊と両端のどちらにも合わない",
                 loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")

    # ---- 右上: β のホライズン依存 ---------------------------------------------
    ax = axes[0, 1]
    ax.axhspan(-half, half, color=NEUTRAL, lw=0, zorder=1)
    ax.annotate(f"片道の半スプレッド {half:.2f} bp", xy=(1, half), xytext=(2, 2),
                textcoords="offset points", color=MUTED, fontsize=8.5, va="bottom")
    # OLS と非重複はほぼ重なるので、ラベルを上下にずらして衝突を避ける
    for col, key, nm, dy in ((S1, "beta_ols", "OLS", 9), (S2, "beta_gls", "GLS(AR(1))", 0),
                             (S3, "beta_nonoverlap", "重ならない部分標本", -10)):
        ys = Fs[key].to_numpy()
        ax.plot(HOR, ys, "-o", color=col, lw=2.2, ms=5.5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.3)
        ax.annotate(nm, xy=(HOR[-1], ys[-1]), xytext=(7, dy), textcoords="offset points",
                    color=col, fontsize=10, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_xticks(HOR)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(0.85, 1500)
    ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
    ax.set_ylabel("β(BookSlope_z が 1σ 増えたときの bp)", color=INK2, fontsize=10)
    ax.set_title("★GLS の AR(1) だけが外れる\n"
                 "重ならない部分標本(重なりが無いので古典的な推定が正しい)は OLS と一致",
                 loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")

    # ---- 左下: t 値 -----------------------------------------------------------
    ax = axes[1, 0]
    for col, key, nm in ((S1, "t_ols", "古典的な標準誤差"), (S2, "t_hac", "HAC(Newey–West)")):
        ys = np.abs(Fs[key].to_numpy())
        ax.plot(HOR, ys, "-o", color=col, lw=2.2, ms=5.5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.3)
        ax.annotate(nm, xy=(HOR[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    color=col, fontsize=10, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(HOR); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(0.85, 1800)
    # text.parse_math=False だと LogFormatter の mathtext がそのまま文字列で出るので、
    # 目盛を明示して素の数字で書く
    ax.set_yticks([10, 20, 50, 100, 200, 500])
    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.get_yaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
    ax.set_ylabel("|t| (対数目盛)", color=INK2, fontsize=10)
    rat = float(np.abs(Fs["t_ols"][-1]) / np.abs(Fs["t_hac"][-1]))
    ax.set_title(f"重なる窓は t 値を水増しする\n"
                 f"k=500 では古典的な t が HAC の {rat:.1f} 倍",
                 loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")

    # ---- 右下: 決定係数 -------------------------------------------------------
    ax = axes[1, 1]
    # k=500 では 2 本ともほぼ 0 に収束するので、ラベルを上下に離す
    for col, src, nm, dy in ((S1, Fs, "BookSlope_z", -9), (S2, Fo, "OBI", 9)):
        ys = src["r2"].to_numpy() * 100
        ax.plot(HOR, ys, "-o", color=col, lw=2.2, ms=5.5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.3)
        ax.annotate(nm, xy=(HOR[-1], ys[-1]), xytext=(7, dy), textcoords="offset points",
                    color=col, fontsize=10, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_xticks(HOR)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(0.85, 1800)
    ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
    ax.set_ylabel("決定係数[%]", color=INK2, fontsize=10)
    ax.set_title("★スプレッドで割ると情報が減る\n"
                 "無次元の OBI のほうがどのホライズンでも 3〜4 倍説明する",
                 loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")

    n = int(Fs["n"][0])
    fig.suptitle(f"{a.coin} Book Slope と将来の log リターン(立会日 67 日・{n:,} 観測)",
                 fontsize=13.5, y=0.978, color=INK, weight="bold")
    fig.text(0.065, 0.945,
             "BookSlope = (bid数量 − ask数量) / 半スプレッド[bp]。最良気配のみ(多階層の板は "
             "Deep Archive にあり per-event では読めない)。直前 5,000 イベントの σ で正規化。\n"
             "x は時刻 t で確定し、y = ln(mid_(t+k)/mid_t) の期間は (t, t+k]。先読みは無い。"
             "重なる窓のため誤差は MA(k−1) で、AR(1) を仮定する GLS は誤設定である。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) l2/bbo を再構成 / 窓 2026-05-04〜08-09 (98 日)",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_book_slope.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
