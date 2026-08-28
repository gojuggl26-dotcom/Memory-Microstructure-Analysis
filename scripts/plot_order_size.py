"""4 つの時間帯のオーダーサイズ分布を描く。

サイズは 4 桁以上にわたって広がるので、横軸は必ず対数にする。
左が密度(対数階級のヒストグラム)、右が「その大きさ以上が占める割合」
(補相補累積分布、両対数)。裾の重さは右の図で読む。

    uv run python scripts/plot_order_size.py --coin xyz:MU
出力: charts/<coin>_order_size_dist.png
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
# 検証済みパレットの 1〜4 番(隣接ペアで色覚特性下の分離を満たす並び)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
WINDOWS = ["開場後 2 時間", "昼 11 時から 2 時間", "閉場前 2 時間", "深夜 11 時から 2 時間"]
UTC = {"開場後 2 時間": "13:30–15:30", "昼 11 時から 2 時間": "11:00–13:00",
       "閉場前 2 時間": "18:00–20:00", "深夜 11 時から 2 時間": "23:00–01:00"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    H = pl.read_csv(ROOT / "data" / f"order_size_hist_{tag}.csv")
    S = pl.read_csv(ROOT / "data" / f"order_size_stats_{tag}.csv").filter(
        pl.col("day_type") == "全曜日")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.0, 6.4), dpi=170)
    for w, col in zip(WINDOWS, COLORS):
        g = H.filter(pl.col("window") == w).sort("lo")
        lo = g["lo"].to_numpy(); hi = g["hi"].to_numpy(); c = g["count"].to_numpy().astype(float)
        n = c.sum()
        mid = np.sqrt(lo * hi)
        # 対数階級なので、幅で割って密度にしないと階級の広さの差が形に化ける
        dens = c / n / (np.log10(hi) - np.log10(lo))
        lab = f"{w}({UTC[w]} UTC)"
        a1.plot(mid, dens, color=col, lw=2.0, zorder=3, label=lab, solid_capstyle="round")
        # 補相補累積分布: そのサイズ以上が全体に占める割合
        ccdf = 1.0 - np.cumsum(c) / n
        a2.plot(hi, np.clip(ccdf, 1e-7, None), color=col, lw=2.0, zorder=3, label=lab)

    # text.parse_math を切っているので、対数軸の既定の目盛りは数式のまま出てしまう。
    # 明示的に素の文字列で書く。
    def plain(v, _):
        return f"{v:g}"

    def pct(v, _):
        return f"{v * 100:g}%"

    for ax in (a1, a2):
        style(ax)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e4)
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("成行注文 1 本のサイズ(枚)", color=INK2, fontsize=10.5)
    a1.set_ylabel("密度(対数階級あたりの割合)", color=INK2, fontsize=10.5)
    a2.set_yscale("log")
    a2.set_ylabel("そのサイズ以上が占める割合", color=INK2, fontsize=10.5)
    a2.set_ylim(1e-6, 1.2)
    a2.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    a2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(pct))
    a2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    a1.set_title("分布のかたち", loc="left", color=INK, fontsize=11.5, pad=8, weight="bold")
    a2.set_title("裾の重さ(両対数)", loc="left", color=INK, fontsize=11.5, pad=8, weight="bold")
    a1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)

    # 中央値を下端に印として置く
    for w, col in zip(WINDOWS, COLORS):
        med = float(S.filter(pl.col("window") == w)["median"][0])
        a1.axvline(med, color=col, lw=1.0, ls=(0, (3, 3)), alpha=0.7, zorder=2)

    fig.suptitle(f"{a.coin} 時間帯ごとのオーダーサイズ分布(全曜日 99 日)",
                 x=0.005, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.005, 0.925,
             "成行注文 1 本のサイズ(同一 ns・同一ユーザーの約定を合算)。破線は各時間帯の中央値。"
             "時刻はすべて UTC で、標本期間は全日が米国東部夏時間(寄り付き 13:30、引け 20:00 UTC)。",
             color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) node_fills / 窓 2026-05-04〜08-10 / "
             "成行注文 6,372,171 本のうち 4 つの時間帯に入る 2,724,681 本",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.855, bottom=0.115, wspace=0.22)
    out = ROOT / "charts" / f"{tag}_order_size_dist.png"
    fig.savefig(out)
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
