"""特定の 1 日について、時間ごとの価格と出来高(買い・売り内訳)を描く。

ニュースが板に着弾する時刻を確かめるための図。原資産が米国株なので、
決算のような重要な発表は **米国市場の引け後**(16:00 ET = 20:00 UTC、夏時間)に
出ることが多い。株式はその値を翌営業日の寄りまで反映できないが、
perp は 24 時間動いているのでその場で反映する。

    uv run python scripts/plot_intraday_day.py --coin xyz:MU --date 2026-06-24 \
        --event 20:00 --event-label "FQ3 決算発表(米国引け後)"
出力: charts/<coin>_intraday_<date>.png
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BUY, SELL = "#2a78d6", "#e34948"
SESSION_BG = "#fadfc9"      # 米国株の立会時間(13:30-20:00 UTC、夏時間)

US_OPEN, US_CLOSE = 13.5, 20.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--event", default=None, help="UTC の HH:MM。縦線を引く")
    ap.add_argument("--event-label", default="")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    day = dt.date.fromisoformat(a.date)

    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet")
        .filter(pl.col("crossed") & (pl.col("ts").dt.date() == day))
        .with_columns(h=pl.col("ts").dt.hour(), notional=pl.col("px") * pl.col("sz"),
                      buy=pl.col("side") == "B")
    )
    g = (
        f.group_by("h")
        .agg(px=pl.col("px").sort_by("ts").last(),
             buy_usd=pl.col("notional").filter(pl.col("buy")).sum(),
             sell_usd=pl.col("notional").filter(~pl.col("buy")).sum())
        .sort("h")
    )
    h = g["h"].to_list()
    px = g["px"].to_list()
    bu = (g["buy_usd"] / 1e6).to_list()
    se = (g["sell_usd"] / 1e6).to_list()

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7.6), sharex=True, dpi=170,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.30},
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.axvspan(US_OPEN, US_CLOSE, color=SESSION_BG, lw=0, zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
        if a.event:
            hh, mm = (int(v) for v in a.event.split(":"))
            ax.axvline(hh + mm / 60, color=INK, lw=1.4, ls=(0, (4, 2.5)), zorder=5)

    # --- 上段: 価格 -------------------------------------------------------------
    ax1.plot(h, px, color=BUY, lw=2.0, zorder=3, solid_capstyle="round",
             marker="o", markersize=4.5, markerfacecolor=BUY,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax1.set_ylabel("価格(USD)", color=INK2, fontsize=10)
    ax1.set_title(f"{a.coin} {day} の 1 時間ごとの値動きと出来高", loc="left",
                  color=INK, fontsize=13, pad=46, weight="bold")
    lo, hi = min(px), max(px)
    ax1.text(0, 1.012,
             f"時刻は UTC。薄いオレンジは原資産(米国株)の立会時間 13:30-20:00 UTC。"
             f"00 時台 {px[0]:,.0f} → 23 時台 {px[-1]:,.0f}"
             f"({(px[-1]/px[0]-1)*100:+.1f}%)、安値 {lo:,.0f} / 高値 {hi:,.0f}。"
             "\n点は各時間の最後の約定価格。perp は 24 時間動くので、"
             "引け後に出た材料もその場で価格に入る。",
             transform=ax1.transAxes, color=INK2, fontsize=9.5, va="bottom", linespacing=1.5)
    if a.event and a.event_label:
        hh, mm = (int(v) for v in a.event.split(":"))
        # 図の右端に近い時刻だと文字がはみ出るので、線の左側へ寄せる
        left = (hh + mm / 60) > 14
        ax1.annotate(a.event_label, xy=(hh + mm / 60, hi), xytext=(-8 if left else 8, -2),
                     textcoords="offset points", color=INK, fontsize=10,
                     weight="bold", ha="right" if left else "left", va="top")

    # --- 下段: 出来高の内訳 -----------------------------------------------------
    ymax = max(b + s for b, s in zip(bu, se)) * 1.18
    gap = ymax * 0.005
    ax2.bar(h, bu, width=0.8, color=BUY, zorder=3, linewidth=0)
    ax2.bar(h, [s - gap for s in se], bottom=[b + gap for b in bu], width=0.8,
            color=SELL, zorder=3, linewidth=0)
    ax2.set_ylim(0, ymax)
    ax2.set_ylabel("出来高(百万 USD)", color=INK2, fontsize=10)
    ax2.set_xlabel("時刻(UTC)", color=INK2, fontsize=10)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{v:02d}" for v in range(0, 24, 2)])
    ax2.set_xlim(-0.6, 23.6)
    ax2.set_title("同じ日の出来高(テイカーの向きで分解)", loc="left", color=INK,
                  fontsize=12, pad=30, weight="bold")
    tot = [b + s for b, s in zip(bu, se)]
    imax = tot.index(max(tot))
    ax2.text(0, 1.02,
             f"最も濃いのは {h[imax]:02d}:00 台の {tot[imax]:.0f}M USD で、"
             f"その日の他のどの時間の {max(tot)/sorted(tot)[-2]:.1f} 倍。",
             transform=ax2.transAxes, color=INK2, fontsize=9.5, va="bottom")
    ax2.legend(handles=[Patch(facecolor=BUY, edgecolor="none", label="買い(テイカー)"),
                        Patch(facecolor=SELL, edgecolor="none", label="売り(テイカー)")],
               loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
               handlelength=1.6, handleheight=1.0, borderpad=0.2)

    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) node_fills を再構成",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.085)
    out = ROOT / "charts" / f"{tag}_intraday_{day}.png"
    fig.savefig(out)
    print(f"[chart] {out}")
    for i, hh in enumerate(h):
        print(f"  {hh:02d}:00  {px[i]:9,.2f}  買い {bu[i]:6.1f}M  売り {se[i]:6.1f}M")


if __name__ == "__main__":
    main()
