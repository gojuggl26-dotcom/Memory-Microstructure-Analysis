"""標本期間の日足(OHLC ローソク足)を fills から作って描く。

日足は UTC 日で区切る(OI・出来高のレポートと同じ区切り)。原資産は米国株だが
perp は 24 時間動いているため、立会時間で切らず 1 日全体を 1 本にする。
背景の白は米国市場(NYSE)の立会日、薄いオレンジは休場日。

x が確定する時刻 / y の期間: 該当なし(実測価格の集計であって予測ではない)。

    uv run python scripts/plot_price.py --coin xyz:MU
出力: charts/<coin>_price_daily.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]

# dataviz スキルの参照パレット(light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
UP, DOWN = "#2a78d6", "#e34948"   # 発散ペア(青↔赤)。上昇=青、下落=赤
CLOSED_BG = "#fadfc9"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    f = pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet", columns=["ts", "px", "crossed"]).filter(
        pl.col("crossed")
    )
    d = (
        f.sort("ts")
        .group_by(pl.col("ts").dt.date().alias("d"))
        .agg(open=pl.col("px").first(), high=pl.col("px").max(),
             low=pl.col("px").min(), close=pl.col("px").last())
        .sort("d")
    )
    days = d["d"].to_list()

    cal = xc.get_calendar("XNYS")
    sessions = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}
    closed = [x for x in days if x not in sessions]

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False,
        "text.parse_math": False,   # "$545 → $861" が数式扱いされるのを防ぐ
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=170)
    ax.set_facecolor(SURFACE)

    half = 0.5  # matplotlib の日付数値(1.0 = 1 日)
    for c in closed:
        x = mdates.date2num(c)
        ax.axvspan(x - half, x + half, color=CLOSED_BG, lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)

    body_w = 0.62
    for row in d.iter_rows(named=True):
        x = mdates.date2num(row["d"])
        col = UP if row["close"] >= row["open"] else DOWN
        ax.add_line(Line2D([x, x], [row["low"], row["high"]], color=col, lw=1.0, zorder=3,
                           solid_capstyle="butt"))
        lo, hi = sorted((row["open"], row["close"]))
        ax.add_patch(Rectangle((x - body_w / 2, lo), body_w, max(hi - lo, 1e-9),
                               facecolor=col, edgecolor="none", zorder=4))

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.set_xlim(mdates.date2num(days[0]) - 1, mdates.date2num(days[-1]) + 1)
    lo, hi = float(d["low"].min()), float(d["high"].max())
    ax.set_ylim(lo - (hi - lo) * 0.07, hi + (hi - lo) * 0.13)

    chg = d["close"][-1] / d["open"][0] - 1
    ax.set_title(f"{a.coin} 日足(標本期間 {days[0]} 〜 {days[-1]}、{len(days)} 日)",
                 loc="left", color=INK, fontsize=13, pad=30, weight="bold")
    ax.text(0, 1.055,
            f"UTC 日区切りの OHLC・約定履歴から作成。始値 ${d['open'][0]:,.2f} → 終値 "
            f"${d['close'][-1]:,.2f}({chg:+.1%})、期間高値 ${hi:,.2f} / 安値 ${lo:,.2f}",
            transform=ax.transAxes, color=INK2, fontsize=9.5)

    i_hi = int(d["high"].arg_max())
    ax.annotate(f"期間高値 ${hi:,.0f}", xy=(days[i_hi], hi), xytext=(0, 9),
                textcoords="offset points", ha="center", color=INK, fontsize=9, weight="bold")
    ax.annotate(f"${d['close'][-1]:,.0f}", xy=(days[-1], float(d["close"][-1])), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=10, weight="bold")

    ax.legend(
        handles=[
            Patch(facecolor=UP, edgecolor="none", label="上昇(終値 ≥ 始値)"),
            Patch(facecolor=DOWN, edgecolor="none", label="下落"),
            Patch(facecolor=SURFACE, edgecolor=BASELINE, lw=0.8, label="米国市場 立会日"),
            Patch(facecolor=CLOSED_BG, edgecolor="none", label="米国市場 休場日(週末・祝日)"),
        ],
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, ncol=4,
        handlelength=1.6, handleheight=1.0, borderpad=0.2,
    )

    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) node_fills / 窓 2026-05-04〜08-10 (99 日)",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.87, bottom=0.09)
    out = ROOT / "charts" / f"{tag}_price_daily.png"
    fig.savefig(out)
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
