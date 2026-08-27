"""実現分散と、同じ窓で測った回転率を 3 つの時間尺度で並べる。

## 分散の定義

1 分ごとの最終約定値で価格系列を作り、対数差をリターンとする。約定の無い分は
直前の値を持ち越す(標本期間の 1.0% にあたる)。窓 W の実現分散は

    RV(W) = Σ_{i∈W} r_i^2 ,   r_i = ln(p_i / p_{i-1})

窓の長さが違うものを並べるので、**単位時間あたりに揃えて** 比較する。

| 呼び名 | 算出期間 | 単位 | 図の横軸 |
|---|---|---|---|
| 5 日窓の日次分散 | 直前 5 日 | 1 日あたり(RV を 5 で割る) | 99 日 |
| 1 日窓の日次分散 | その日 | 1 日あたり | 1 週間 |
| 1 時間窓の 1 時間次分散 | その 1 時間 | 1 時間あたり | 1 日 |

数値は (%)^2 で表示する(対数リターンを百分率に直した分散)。

## 回転率

分散と同じ窓で測る。分子はその窓の出来高(枚)、分母はその窓の時間加重平均建玉(枚)。

    Turnover(W) = Volume(W) / meanOI(W)

5 日窓は日あたりに揃えるため 5 で割る。1 日窓・1 時間窓はそのまま。

## 窓の選び方

結果を見てから都合の良い期間を選ぶのは生存バイアスなので、機械的な規則で決める。
1 週間の図は **標本の最後の 7 日**、1 日の図は **標本の最終日** を使う。

x が確定する時刻 / y の期間: 該当なし(実測量の集計であって予測ではない)。
5 日窓は後ろ向き(その日を含む直前 5 日)なので、未来の情報は入らない。

    uv run python scripts/build_variance.py --coin xyz:MU
出力: data/variance_{daily,hourly}_<coin>.csv と charts/<coin>_variance_*.png
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
VAR_C, TURN_C = "#2a78d6", "#eb6834"     # 分散 / 回転率(検証済みパレットの 1 番と 2 番)
CLOSED_BG = "#fadfc9"
SESSION_UTC = (13.5, 20.0)               # ニューヨーク証券取引所の立会時間(夏時間)
WIN_DAYS = 5


def minute_returns(tag: str) -> pl.DataFrame:
    """1 分足の対数リターンと、その分の出来高。"""
    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet", columns=["ts", "px", "sz", "crossed"])
        .filter(pl.col("crossed"))
        .sort("ts")
    )
    m = (
        f.with_columns(t=pl.col("ts").dt.truncate("1m"))
        .group_by("t").agg(px=pl.col("px").last(), v=pl.col("sz").sum())
        .sort("t")
    )
    grid = pl.datetime_range(m["t"][0], m["t"][-1], interval="1m", eager=True, time_unit="ns")
    g = (
        pl.DataFrame({"t": grid})
        .join(m, on="t", how="left")
        .with_columns(px=pl.col("px").fill_null(strategy="forward"), v=pl.col("v").fill_null(0.0))
        .with_columns(r=pl.col("px").log() - pl.col("px").log().shift(1))
        .drop_nulls("r")
        .with_columns(r2=pl.col("r") ** 2)
    )
    print(f"[分足] {g.height:,} 本 / 約定のあった分 {m.height:,} 本"
          f"(持ち越した分 {1 - m.height / g.height:.1%})")
    return g


def mean_oi(tag: str, freq: str) -> pl.DataFrame:
    """建玉の階段関数から、区間ごとの時間加重平均を出す。freq は "1d" か "1h"。"""
    oi = pl.read_parquet(ROOT / "data" / f"oi_series_{tag}.parquet").select("ts", "oi").sort("ts")
    step = dt.timedelta(days=1) if freq == "1d" else dt.timedelta(hours=1)
    t0 = oi["ts"][0].replace(hour=0, minute=0, second=0, microsecond=0)
    t1 = oi["ts"][-1] + step
    edges = []
    t = t0
    while t <= t1:
        edges.append(t)
        t += step
    e = (
        pl.DataFrame({"ts": edges})
        .with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        .sort("ts")
        .join_asof(oi, on="ts", strategy="backward")
    )
    ev = (
        pl.concat([oi, e.select("ts", "oi")])
        .drop_nulls("oi")
        .sort("ts")
        .with_columns(dur=(pl.col("ts").shift(-1) - pl.col("ts")).dt.total_nanoseconds().cast(pl.Float64))
        .drop_nulls("dur")
        .with_columns(k=pl.col("ts").dt.truncate(freq))
    )
    return ev.group_by("k").agg(
        oi_mean=(pl.col("oi") * pl.col("dur")).sum() / pl.col("dur").sum()
    ).sort("k")


def style(ax, closed_spans, xlim=None) -> None:
    ax.set_facecolor(SURFACE)
    for lo, hi in closed_spans:
        ax.axvspan(lo, hi, color=CLOSED_BG, lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    if xlim:
        ax.set_xlim(*xlim)


def figure(coin, tag, name, title, sub, x, var, turn, var_lab, turn_lab,
           closed_spans, kind, xfmt, xloc, xlabel, xlim):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, dpi=170,
                                 gridspec_kw={"hspace": 0.2})
    for ax, vals, col, lab in ((a1, var, VAR_C, var_lab), (a2, turn, TURN_C, turn_lab)):
        style(ax, closed_spans, xlim)
        if kind == "line":
            ax.plot(x, vals, color=col, lw=2.0, zorder=3, solid_capstyle="round")
            ax.fill_between(x, 0, vals, color=col, alpha=0.13, lw=0, zorder=2)
        else:
            ax.bar(x, vals, width=kind, color=col, zorder=3, linewidth=0)
        ax.set_ylabel(lab, color=INK2, fontsize=10)
        ax.set_ylim(0, float(np.nanmax(vals)) * 1.18)
    a1.set_title(f"{coin} {title}", loc="left", color=INK, fontsize=13, pad=30, weight="bold")
    a1.text(0, 1.045, sub, transform=a1.transAxes, color=INK2, fontsize=9.5)
    a2.set_title("同じ窓で測った回転率", loc="left", color=INK, fontsize=11.5, pad=8, weight="bold")
    if xfmt:
        a2.xaxis.set_major_formatter(xfmt)
    if xloc:
        a2.xaxis.set_major_locator(xloc)
    a2.set_xlabel(xlabel, color=INK2, fontsize=10)
    fig.text(0.005, 0.008,
             "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10 / "
             "分散は 1 分足対数リターンの二乗和",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.885, bottom=0.125)
    out = ROOT / "charts" / f"{tag}_variance_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[chart] {out}")


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

    g = minute_returns(tag)

    # --- 日次 ---------------------------------------------------------------
    day = (
        g.group_by(pl.col("t").dt.truncate("1d").alias("k"))
        .agg(rv=pl.col("r2").sum(), vol=pl.col("v").sum())
        .sort("k")
        .join(mean_oi(tag, "1d"), on="k", how="left")
    )
    day = day.with_columns(
        var_1d=pl.col("rv") * 1e4,                                   # (%)^2
        turn_1d=pl.col("vol") / pl.col("oi_mean"),
    ).with_columns(
        var_5d=pl.col("rv").rolling_sum(WIN_DAYS) / WIN_DAYS * 1e4,
        turn_5d=pl.col("vol").rolling_sum(WIN_DAYS)
        / pl.col("oi_mean").rolling_mean(WIN_DAYS) / WIN_DAYS,
    )
    assert day.height == 99, day.height

    # --- 1 時間 -------------------------------------------------------------
    hour = (
        g.group_by(pl.col("t").dt.truncate("1h").alias("k"))
        .agg(rv=pl.col("r2").sum(), vol=pl.col("v").sum())
        .sort("k")
        .join(mean_oi(tag, "1h"), on="k", how="left")
    )
    hour = hour.with_columns(var_1h=pl.col("rv") * 1e4,
                             turn_1h=pl.col("vol") / pl.col("oi_mean"))

    day.select("k", "var_1d", "var_5d", "turn_1d", "turn_5d").write_csv(
        ROOT / "data" / f"variance_daily_{tag}.csv")
    hour.select("k", "var_1h", "turn_1h").write_csv(
        ROOT / "data" / f"variance_hourly_{tag}.csv")

    cal = xc.get_calendar("XNYS")
    days = [d.date() for d in day["k"].to_list()]
    sess = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}
    closed_day_spans = [(mdates.date2num(d) - 0.5, mdates.date2num(d) + 0.5)
                        for d in days if d not in sess]

    print(f"\n=== 5 日窓の日次分散((%)^2)===")
    v = day["var_5d"].drop_nulls()
    print(f"  中央値 {v.median():.2f} / 最小 {v.min():.2f} / 最大 {v.max():.2f}"
          f"  → 日次ボラ 中央値 {np.sqrt(v.median()):.2f}%")
    print(f"  同窓の回転率: 中央値 {day['turn_5d'].drop_nulls().median():.2f} 回/日")
    print(f"=== 1 日窓の日次分散((%)^2)===")
    print(f"  中央値 {day['var_1d'].median():.2f} / 最小 {day['var_1d'].min():.2f} / "
          f"最大 {day['var_1d'].max():.2f}({days[int(day['var_1d'].arg_max())]})")
    print(f"=== 1 時間窓の分散((%)^2)===")
    print(f"  中央値 {hour['var_1h'].median():.3f} / 最大 {hour['var_1h'].max():.2f}")
    lv = np.log(day["var_1d"].to_numpy()); lt = np.log(day["turn_1d"].to_numpy())
    print(f"\n[同時点の相関] 日次: log 分散 と log 回転率 {np.corrcoef(lv, lt)[0, 1]:+.3f}(n=99)")
    hv, ht = hour["var_1h"].to_numpy(), hour["turn_1h"].to_numpy()
    ok = (hv > 0) & (ht > 0)
    print(f"[同時点の相関] 1 時間: {np.corrcoef(np.log(hv[ok]), np.log(ht[ok]))[0, 1]:+.3f}"
          f"(n={int(ok.sum())})")
    print("  ※ どちらも同時点の関係であって予測力ではない。")

    # --- 図 1: 5 日窓 / 99 日 -------------------------------------------------
    figure(a.coin, tag, "5d_over_99d", "5 日窓の日次分散(99 日間)",
           f"直前 {WIN_DAYS} 日の 1 分足リターンから求め、1 日あたりに換算した分散。"
           f"最初の {WIN_DAYS - 1} 日は窓が埋まらないため空白。",
           days, day["var_5d"].to_numpy(), day["turn_5d"].to_numpy(),
           "日次分散((%)^2)", "回転率(回/日)", [], "line",
           mdates.DateFormatter("%m/%d"), mdates.WeekdayLocator(byweekday=mdates.MO, interval=2),
           "日付(UTC)", (mdates.date2num(days[0]) - 0.5, mdates.date2num(days[-1]) + 0.5))

    # --- 図 2: 1 日窓 / 1 週間 ------------------------------------------------
    wk = day.tail(7)
    wkd = [d.date() for d in wk["k"].to_list()]
    spans = [(mdates.date2num(d) - 0.5, mdates.date2num(d) + 0.5) for d in wkd if d not in sess]
    figure(a.coin, tag, "1d_over_1w", f"1 日窓の日次分散({wkd[0]} 〜 {wkd[-1]} の 7 日間)",
           "その日の 1 分足リターンだけから求めた分散。薄いオレンジは米国市場の休場日。",
           wkd, wk["var_1d"].to_numpy(), wk["turn_1d"].to_numpy(),
           "日次分散((%)^2)", "回転率(回/日)", spans, 0.7,
           mdates.DateFormatter("%m/%d\n(%a)"), mdates.DayLocator(),
           "日付(UTC)", (mdates.date2num(wkd[0]) - 0.6, mdates.date2num(wkd[-1]) + 0.6))

    # --- 図 3: 1 時間窓 / 1 日 ------------------------------------------------
    last = days[-1]
    hd = hour.filter(pl.col("k").dt.date() == last)
    hx = np.array([h.hour + 0.5 for h in hd["k"].to_list()])
    spans3 = ([(0, SESSION_UTC[0]), (SESSION_UTC[1], 24)] if last in sess else [(0, 24)])
    note = ("薄いオレンジは原資産(米国株)の取引時間の外。"
            if last in sess else "この日は米国市場が休場のため全面がオレンジ。")
    figure(a.coin, tag, "1h_over_1d", f"1 時間窓の分散({last} の 1 日)",
           "その 1 時間の 1 分足リターンだけから求めた分散。" + note,
           hx, hd["var_1h"].to_numpy(), hd["turn_1h"].to_numpy(),
           "1 時間次分散((%)^2)", "回転率(回/時)", spans3, 0.7,
           None, mpl.ticker.MultipleLocator(3), "時刻(UTC)", (0, 24))


if __name__ == "__main__":
    main()
