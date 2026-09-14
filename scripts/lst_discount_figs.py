r"""LST の乖離(市場価格 / 償還価値 - 1)を図にする。

    uv run python scripts/lst_discount_figs.py

入力: data/staking/lst_discount.parquet  (lst_discount_analysis.py)
出力: reports/staking/fig/discount_ts.png     … 乖離の推移
      reports/staking/fig/discount_dist.png   … 乖離の分布

=============================================================================
パレットは目視で選んでいない
=============================================================================
Okabe-Ito の 4 色。OKLab 距離と Vienot(1999) の 3 型色覚シミュレーションで
検証済み(2026-09-14 実測):

    通常視の最小 ΔE = 16.4  (下限 15)
    P/D/T 型の最小 ΔE = 8.2 (目標 8)
    明度 L = 0.532〜0.679、彩度 C = 0.118〜0.170

色は**銘柄に固定**する。系列を絞っても残った系列の色は変わらない。

y 軸は 1 本だけ(乖離率)。2 軸は使わない。
"""
from __future__ import annotations

import datetime as dt
import pathlib

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.dates as mdates                  # noqa: E402
from matplotlib import font_manager                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "staking" / "lst_discount.parquet"
FIG = ROOT / "reports" / "staking" / "fig"

# 日本語が豆腐にならないようにする。無ければ英字のまま(文字化けよりまし)。
_have = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP"):
    if _f in _have:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# 色は銘柄に固定する(順位ではなく実体に付ける)
COLORS = {
    "stETH/ETH": "#0072B2",
    "rETH/ETH":  "#D55E00",
    "cbETH/ETH": "#009E73",
    "weETH/ETH": "#CC79A7",
}
ORDER = list(COLORS)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main() -> None:
    d = pl.read_parquet(SRC)
    FIG.mkdir(parents=True, exist_ok=True)
    feeds = [f for f in ORDER if f in set(d["feed"])]

    # ---- 図 1: 推移 -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=140)
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)
    for f in feeds:
        s = d.filter(pl.col("feed") == f).sort("ts")
        x = [dt.datetime.fromtimestamp(t, dt.timezone.utc) for t in s["ts"]]
        y = [v * 100 for v in s["disc"]]
        ax.plot(x, y, color=COLORS[f], linewidth=1.4, label=f.replace("/ETH", ""),
                zorder=3, solid_joinstyle="round")
    ax.set_ylabel("乖離  市場価格 / 償還価値 − 1  (%)", color=INK, fontsize=10)
    ax.set_title("LST は償還価値に対していくら安く取引されてきたか",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=4)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.annotate("0 = 償還価値どおり", xy=(0.004, 0.5), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "discount_ts.png", facecolor="white")
    plt.close(fig)
    print(f"[write] {FIG / 'discount_ts.png'}")

    # ---- 図 2: 分布 -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=140)
    style(ax)
    ax.axvline(0, color=MUTED, linewidth=1.2, zorder=2)
    data = [[v * 100 for v in d.filter(pl.col("feed") == f)["disc"]] for f in feeds]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3,
                                                     markerfacecolor=MUTED,
                                                     markeredgecolor="none", alpha=0.35),
                    medianprops=dict(color="white", linewidth=1.6),
                    whiskerprops=dict(color=MUTED, linewidth=1.0),
                    capprops=dict(color=MUTED, linewidth=1.0))
    for patch, f in zip(bp["boxes"], feeds):
        patch.set_facecolor(COLORS[f])
        patch.set_edgecolor("white")
        patch.set_linewidth(1.4)
    ax.set_yticks(range(1, len(feeds) + 1))
    ax.set_yticklabels([f.replace("/ETH", "") for f in feeds], color=INK, fontsize=10)
    ax.set_xlabel("乖離 (%)", color=INK, fontsize=10)
    ax.set_title("分布（箱＝四分位、ひげ＝1.5IQR、点＝外れ値）",
                 color=INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "discount_dist.png", facecolor="white")
    plt.close(fig)
    print(f"[write] {FIG / 'discount_dist.png'}")


if __name__ == "__main__":
    main()
