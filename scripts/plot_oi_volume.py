"""日次平均 OI と出来高の時系列を描く。米国市場の立会日は白、休場日は薄いオレンジ。

    uv run python scripts/plot_oi_volume.py --coin xyz:MU --unit usd
出力: charts/<coin>_oi_volume_<unit>.png

休場日は NYSE(XNYS)のカレンダーから取る。原資産は米国株なので、
Hyperliquid の perp が 24/7 動いていても原市場が閉じている日は性質が違う。
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

# dataviz スキルの参照パレット(light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_FILL = "#2a78d6", "#2a78d6"
CLOSED_BG = "#fadfc9"   # 薄いオレンジ(slot2 オレンジの淡いステップ)


def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--unit", choices=["usd", "contracts"], default="usd")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = pl.read_parquet(ROOT / "data" / f"daily_oi_volume_{tag}.parquet").sort("d")
    days = d["d"].to_list()

    if a.unit == "usd":
        oi, band, vol = d["oi_mean_usd"], d["oi_halfgap_usd"], d["volume_usd"]
        oi_lbl, vol_lbl = "平均建玉(名目 USD)", "出来高(名目 USD)"
        tick = lambda v, _: fmt_money(v)  # noqa: E731
    else:
        oi, band, vol = d["oi_mean"], d["oi_halfgap"], d["volume"]
        oi_lbl, vol_lbl = "平均建玉(枚)", "出来高(枚)"
        tick = lambda v, _: f"{v/1000:,.0f}k"  # noqa: E731

    cal = xc.get_calendar("XNYS")
    sessions = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}
    closed = [x for x in days if x not in sessions]
    holidays = {
        dt.date(2026, 5, 25): "メモリアルデー",
        dt.date(2026, 6, 19): "ジューンティーンス",
        dt.date(2026, 7, 3): "独立記念日(振替)",
    }

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.6), sharex=True, dpi=170,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.22},
    )

    half = 0.5  # matplotlib の日付数値(1.0 = 1 日)。date に timedelta(days=0.5) を足しても切り捨てられる
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for c in closed:
            x = mdates.date2num(c)
            ax.axvspan(x - half, x + half, color=CLOSED_BG, lw=0, zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(tick))

    # --- 上段: 平均建玉 -------------------------------------------------------
    ax1.fill_between(days, (oi - band).to_list(), (oi + band).to_list(),
                     color=BLUE_FILL, alpha=0.16, lw=0, zorder=2)
    ax1.plot(days, oi.to_list(), color=BLUE, lw=2.0, zorder=3, solid_capstyle="round")
    ax1.set_ylim(0, float((oi + band).max()) * 1.18)
    ax1.set_title(f"{a.coin} 日次平均建玉(OI)", loc="left", color=INK, fontsize=13, pad=26, weight="bold")
    ax1.text(0, 1.045, f"{oi_lbl}・時間加重平均。帯は復元の不確かさ(ロング側とショート側の食い違いの半分)",
             transform=ax1.transAxes, color=INK2, fontsize=9.5)
    ax1.annotate(f"{fmt_money(float(oi[-1])) if a.unit=='usd' else f'{oi[-1]:,.0f} 枚'}",
                 xy=(days[-1], float(oi[-1])), xytext=(-4, 14), textcoords="offset points",
                 color=INK, fontsize=10, weight="bold", ha="right")

    for hd, name in holidays.items():
        if hd in closed:
            ax1.annotate(name, xy=(hd, 0), xytext=(0, 8), textcoords="offset points",
                         rotation=90, ha="center", va="bottom", color=MUTED, fontsize=8)

    ax1.legend(
        handles=[
            Patch(facecolor=SURFACE, edgecolor=BASELINE, lw=0.8, label="米国市場 立会日"),
            Patch(facecolor=CLOSED_BG, edgecolor="none", label="米国市場 休場日(週末・祝日)"),
        ],
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
        handlelength=1.6, handleheight=1.0, borderpad=0.2,
    )

    # --- 下段: 出来高 ---------------------------------------------------------
    ax2.bar(days, vol.to_list(), width=0.78, color=BLUE, zorder=3, linewidth=0)
    ax2.set_ylim(0, float(vol.max()) * 1.18)
    ax2.set_title(f"{a.coin} 日次出来高", loc="left", color=INK, fontsize=13, pad=26, weight="bold")
    ax2.text(0, 1.045, f"{vol_lbl}・テイカー側で計上(1 取引 1 回)",
             transform=ax2.transAxes, color=INK2, fontsize=9.5)
    imax = int(vol.arg_max())
    ax2.annotate(f"最大 {fmt_money(float(vol[imax])) if a.unit=='usd' else f'{vol[imax]:,.0f}枚'}\n{days[imax]}",
                 xy=(days[imax], float(vol[imax])), xytext=(0, 8), textcoords="offset points",
                 ha="center", color=INK, fontsize=9, weight="bold")

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.set_xlim(mdates.date2num(days[0]) - half, mdates.date2num(days[-1]) + half)

    fig.text(0.005, 0.005,
             "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10 (99 日) / "
             "OI は約定履歴からの復元値であり、期間中一度も約定しなかった建玉は含まない",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.075)
    out = ROOT / "charts" / f"{tag}_oi_volume_{a.unit}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
