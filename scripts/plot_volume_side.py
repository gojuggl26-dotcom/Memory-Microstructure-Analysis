"""日次出来高を買い(テイカー買い)と売り(テイカー売り)に分けて描く。

上段は積み上げ棒。2 つを足すと元の日次出来高に一致する。
下段は売買差(買い − 売り)。買い比率の中央値が 49.6% なので上段だけでは
偏りがほとんど見えない。偏りは下段でしか読めない。

灰色の帯は帰無対照 ±1.96σ。取引の向きを独立なコイン投げに置き換えたときに
売買差がどこまで散らばるかで、自己相関を考慮した HAC 標準偏差を使っている。
帯の外に出た日は「たまたまの偏り」では説明がつかない。

配色は dataviz の diverging pair(青 ↔ 赤)。scripts/palette_check.py で検証済み:
CVD ΔE 21.6(目標 8)、通常視 ΔE 32.3(下限 15)、対比は白地 4.30 / 3.85、
休場日の帯の上で 3.46 / 3.10(いずれも下限 3.0 以上)。

    uv run python scripts/plot_volume_side.py --coin xyz:MU
出力: charts/<coin>_volume_side.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

# dataviz の参照パレット(light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BUY, SELL = "#2a78d6", "#e34948"      # diverging pair: 青(買い) ↔ 赤(売り)
# 帰無対照の帯。白地の上でも休場日のオレンジの上でも読める必要があるので、
# diverging の中立 #f0efec ではなく chrome の baseline トーンを使う
# (対比は白地 1.75 / 休場帯 1.41。#f0efec は 1.12 / 1.11、#e1e0d9 は 1.29 / 1.04 で
#  オレンジの上に乗ると消える)。
NEUTRAL = "#c3c2b7"
CLOSED_BG = "#fadfc9"                  # 休場日


def fmt_money(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = pl.read_parquet(ROOT / "data" / f"daily_volume_side_{tag}.parquet").sort("d")
    days = d["d"].to_list()
    buy, sell = d["buy_usd"].to_list(), d["sell_usd"].to_list()
    total, net = d["volume_usd"].to_list(), d["net_usd"].to_list()
    band = (d["sd_null_hac"] * 1.96).to_list()

    cal = xc.get_calendar("XNYS")
    sessions = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}
    closed = [x for x in days if x not in sessions]

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False,
        "text.parse_math": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.8), sharex=True, dpi=170,
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.32},
    )

    half = 0.5
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
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: fmt_money(v)))

    # --- 上段: 積み上げ(買い + 売り = 元の出来高)-------------------------------
    ymax = max(total) * 1.18
    gap = ymax * 0.004      # 積み上げの境目に地色の隙間を入れる(約 2px 相当)
    ax1.bar(days, buy, width=0.78, color=BUY, zorder=3, linewidth=0)
    ax1.bar(days, [s - gap for s in sell], width=0.78, bottom=[b + gap for b in buy],
            color=SELL, zorder=3, linewidth=0)
    ax1.set_ylim(0, ymax)
    ax1.set_title(f"{a.coin} 日次出来高の買い・売り内訳", loc="left", color=INK,
                  fontsize=13, pad=46, weight="bold")
    ax1.text(0, 1.012,
             "名目 USD・テイカー(板を取りに行った側)の向きで分解。2 つを足すと日次出来高に一致する。"
             "\n取引には必ず買い手と売り手がいるので、意味を持つ分け方はアグレッサーの向きだけである。",
             transform=ax1.transAxes, color=INK2, fontsize=9.5, va="bottom", linespacing=1.5)

    imax = total.index(max(total))
    ax1.annotate(f"最大 {fmt_money(total[imax])}\n{days[imax]}",
                 xy=(days[imax], total[imax]), xytext=(0, 8), textcoords="offset points",
                 ha="center", color=INK, fontsize=9, weight="bold")

    ax1.legend(
        handles=[
            Patch(facecolor=BUY, edgecolor="none", label="買い(テイカーが買った)"),
            Patch(facecolor=SELL, edgecolor="none", label="売り(テイカーが売った)"),
            Patch(facecolor=CLOSED_BG, edgecolor="none", label="米国市場 休場日"),
        ],
        loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
        handlelength=1.6, handleheight=1.0, borderpad=0.2,
    )

    # --- 下段: 売買差 -----------------------------------------------------------
    ax2.fill_between(days, [-b for b in band], band, color=NEUTRAL, lw=0, zorder=2)
    ax2.bar(days, net, width=0.78, zorder=3, linewidth=0,
            color=[BUY if v >= 0 else SELL for v in net])
    ax2.axhline(0, color=BASELINE, lw=0.8, zorder=4)
    lim = max(max(net), -min(net), max(band)) * 1.28
    ax2.set_ylim(-lim, lim)
    ax2.set_title("売買差(買い − 売り)", loc="left", color=INK, fontsize=13,
                  pad=46, weight="bold")
    ax2.text(0, 1.015,
             "上が買い越し、下が売り越し。灰色の帯は帰無対照 ±1.96σ ——"
             " 取引の向きが独立なコイン投げなら、この幅に収まる。"
             "\n帯の外に出た日は 28 / 99 日(偶然なら 5 日程度)。σ は向きの自己相関を織り込んだ HAC。",
             transform=ax2.transAxes, color=INK2, fontsize=9.5, va="bottom", linespacing=1.5)

    for i in (net.index(max(net)), net.index(min(net))):
        up = net[i] > 0
        ax2.annotate(f"{net[i]/1e6:+.0f}M\n{days[i]}",
                     xy=(days[i], net[i]), xytext=(0, 7 if up else -7),
                     textcoords="offset points", ha="center",
                     va="bottom" if up else "top", color=INK, fontsize=9, weight="bold")

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.set_xlim(mdates.date2num(days[0]) - half, mdates.date2num(days[-1]) + half)

    fig.text(0.005, 0.005,
             "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10 (99 日) / "
             "全期間では買い 49.53%・売り 50.47%(数値は data/daily_volume_side_xyz_MU.csv)",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.075)
    out = ROOT / "charts" / f"{tag}_volume_side.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
