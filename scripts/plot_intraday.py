"""日内(1 日の中)の出来高と建玉の変化を、立会日と休場日に分けて描く。

出来高は 30 分ごとの約定数量の合計、建玉の変化は 30 分の区間の始点と終点の差を取り、
それぞれ日をまたいで平均する。時刻はすべて UTC。

建玉の値は約定ごとに変わる階段関数なので、区間の境界時刻における値は
**後ろ向きの asof 結合**(その時刻以前で最後の値)で取る。前向きや最近傍を使うと
未来の情報が混じるため使わない。

背景の薄いオレンジは、原資産である米国株の取引時間の外を表す。
米国東部夏時間の 9:30-16:00 は UTC の 13:30-20:00 にあたる。
休場日は 1 日を通して原市場が閉じているので、全面がオレンジになる。

    uv run python scripts/plot_intraday.py --coin xyz:MU
出力: charts/<coin>_intraday_{volume,oichange}_{open,closed}.png と 2x2 のまとめ図
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"
CLOSED_BG = "#fadfc9"
BUCKET_MIN = 30
SESSION_UTC = (13.5, 20.0)   # ニューヨーク証券取引所の立会時間(夏時間)


def build(coin: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    tag = coin.replace(":", "_")
    f = pl.read_parquet(
        ROOT / "data" / f"fills_{tag}.parquet", columns=["ts", "sz", "crossed"]
    ).filter(pl.col("crossed"))
    days = f["ts"].dt.date().unique().sort().to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}

    nb = 24 * 60 // BUCKET_MIN
    vol = (
        f.with_columns(
            d=pl.col("ts").dt.date(),
            # dt.hour() は Int8 なので 60 倍すると桁あふれする。必ず広げてから計算する
            b=(pl.col("ts").dt.hour().cast(pl.Int32) * 60
               + pl.col("ts").dt.minute().cast(pl.Int32)) // BUCKET_MIN,
        )
        .group_by("d", "b")
        .agg(v=pl.col("sz").sum())
    )
    # 約定の無いバケットは 0 として埋める
    grid = pl.DataFrame(
        {"d": [d for d in days for _ in range(nb)], "b": list(range(nb)) * len(days)}
    ).with_columns(b=pl.col("b").cast(pl.Int64))
    vol = grid.join(vol, on=["d", "b"], how="left").with_columns(v=pl.col("v").fill_null(0.0))

    # 建玉は境界時刻の値を後ろ向き asof で拾い、区間差分を取る
    oi = pl.read_parquet(ROOT / "data" / f"oi_series_{tag}.parquet").select("ts", "oi").sort("ts")
    # 境界時刻には日とバケット番号を明示的に持たせる。時刻から復元すると、
    # ある日の末尾(24:00)と翌日の先頭(0:00)が同じ時刻になり重複するため。
    e = pl.DataFrame({
        "d": [d for d in days for _ in range(nb + 1)],
        "b": list(range(nb + 1)) * len(days),
        "ts": [dt.datetime.combine(d, dt.time()) + dt.timedelta(minutes=BUCKET_MIN * k)
               for d in days for k in range(nb + 1)],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")), pl.col("b").cast(pl.Int64)).sort("ts")
    e = e.join_asof(oi, on="ts", strategy="backward").with_columns(
        oi=pl.col("oi").fill_null(strategy="forward")
    )
    e = e.sort("d", "b").with_columns(nxt=pl.col("oi").shift(-1).over("d"))
    doi = (
        e.filter(pl.col("b") < nb)
        .with_columns(doi=pl.col("nxt") - pl.col("oi"))
        .select("d", "b", "doi")
        .drop_nulls()
    )

    for name, df in (("出来高", vol), ("建玉", doi)):
        assert df["b"].min() == 0 and df["b"].max() == nb - 1, f"{name}のバケット番号が 0..{nb-1} を覆っていない"
    both = vol.join(doi, on=["d", "b"], how="inner").with_columns(
        open_day=pl.col("d").is_in(list(sess))
    )
    assert both.height == len(days) * nb, f"日 x バケットの数が合わない: {both.height} != {len(days) * nb}"
    prof = (
        both.group_by("open_day", "b")
        .agg(
            v_mean=pl.col("v").mean(),
            v_q1=pl.col("v").quantile(0.25),
            v_q3=pl.col("v").quantile(0.75),
            o_mean=pl.col("doi").mean(),
            o_q1=pl.col("doi").quantile(0.25),
            o_q3=pl.col("doi").quantile(0.75),
            n=pl.len(),
        )
        .sort("open_day", "b")
    )
    return prof, both


def style(ax, open_day: bool) -> None:
    ax.set_facecolor(SURFACE)
    if open_day:
        ax.axvspan(0, SESSION_UTC[0], color=CLOSED_BG, lw=0, zorder=0)
        ax.axvspan(SESSION_UTC[1], 24, color=CLOSED_BG, lw=0, zorder=0)
    else:
        ax.axvspan(0, 24, color=CLOSED_BG, lw=0, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("時刻(UTC)", color=INK2, fontsize=10)


def draw_volume(ax, g: pl.DataFrame, open_day: bool) -> None:
    xs = (g["b"].to_numpy() + 0.5) * BUCKET_MIN / 60
    ax.fill_between(
        xs, g["v_q1"].to_numpy(), g["v_q3"].to_numpy(), color=BLUE, alpha=0.16, lw=0, zorder=2
    )
    ax.bar(xs, g["v_mean"].to_numpy(), width=BUCKET_MIN / 60 * 0.86, color=BLUE, zorder=3, linewidth=0)
    style(ax, open_day)
    ax.set_ylabel("30 分あたりの出来高(枚)", color=INK2, fontsize=10)


def draw_doi(ax, g: pl.DataFrame, open_day: bool) -> None:
    xs = (g["b"].to_numpy() + 0.5) * BUCKET_MIN / 60
    m = g["o_mean"].to_numpy()
    ax.fill_between(
        xs, g["o_q1"].to_numpy(), g["o_q3"].to_numpy(), color=MUTED, alpha=0.18, lw=0, zorder=2
    )
    ax.bar(xs, m, width=BUCKET_MIN / 60 * 0.86, color=np.where(m >= 0, BLUE, RED), zorder=3, linewidth=0)
    ax.axhline(0, color=BASELINE, lw=0.9, zorder=2)
    style(ax, open_day)
    ax.set_ylabel("30 分あたりの建玉の変化(枚)", color=INK2, fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False,
        "text.parse_math": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    prof, both = build(a.coin)
    gs = {True: prof.filter(pl.col("open_day")), False: prof.filter(~pl.col("open_day"))}
    ndays = {k: int(both.filter(pl.col("open_day") == k)["d"].n_unique()) for k in (True, False)}
    print(f"立会日 {ndays[True]} 日 / 休場日 {ndays[False]} 日、各日 {24 * 60 // BUCKET_MIN} バケット")

    peak = {k: float(gs[k]["v_mean"].max()) for k in (True, False)}
    note = ("背景の薄いオレンジは原資産(米国株)の取引時間の外。"
            "帯は日ごとのばらつき(第 1 四分位から第 3 四分位)。")

    specs = [
        ("volume", True, f"立会日 {ndays[True]} 日の日内出来高", draw_volume),
        ("oichange", True, f"立会日 {ndays[True]} 日の日内 建玉の変化", draw_doi),
        ("volume", False, f"休場日 {ndays[False]} 日の日内出来高", draw_volume),
        ("oichange", False, f"休場日 {ndays[False]} 日の日内 建玉の変化", draw_doi),
    ]
    for kind, od, title, fn in specs:
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=170)
        fn(ax, gs[od], od)
        ax.set_title(f"{a.coin} {title}", loc="left", color=INK, fontsize=12.5, pad=26, weight="bold")
        sub = note
        if kind == "volume" and not od:
            sub += f" 縦軸は休場日に合わせてある(立会日のピークは {peak[True] / peak[False]:.0f} 倍)。"
        ax.text(0, 1.045, sub, transform=ax.transAxes, color=INK2, fontsize=9)
        fig.text(
            0.005, 0.012,
            "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10",
            color=MUTED, fontsize=8,
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.855, bottom=0.13)
        suffix = "open" if od else "closed"
        out = ROOT / "charts" / f"{tag}_intraday_{kind}_{suffix}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"[chart] {out}")

    # 比較用の 2x2。縦軸は指標ごとに左右で揃える
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), dpi=170)
    for j, od in enumerate((True, False)):
        draw_volume(axes[0][j], gs[od], od)
        draw_doi(axes[1][j], gs[od], od)
        lab = f"立会日 {ndays[True]} 日" if od else f"休場日 {ndays[False]} 日"
        axes[0][j].set_title(f"{lab} の日内出来高", loc="left", color=INK, fontsize=12, pad=10, weight="bold")
        axes[1][j].set_title(f"{lab} の日内 建玉の変化", loc="left", color=INK, fontsize=12, pad=10, weight="bold")
    for row in axes:
        lo = min(ax.get_ylim()[0] for ax in row)
        hi = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    fig.suptitle(f"{a.coin} 日内プロファイル(縦軸は左右で共通)", x=0.006, ha="left",
                 color=INK, fontsize=14, weight="bold")
    fig.text(0.006, 0.945, note, color=INK2, fontsize=9.5)
    fig.text(0.005, 0.008, "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.07, hspace=0.42, wspace=0.16)
    out = ROOT / "charts" / f"{tag}_intraday_2x2.png"
    fig.savefig(out)
    print(f"[chart] {out}")

    prof.write_csv(ROOT / "data" / f"intraday_profile_{tag}.csv")
    print()
    print("=== 要約 ===")
    for od in (True, False):
        g = gs[od]
        i = int(g["v_mean"].arg_max())
        h = (int(g["b"][i]) * BUCKET_MIN) // 60
        mm = (int(g["b"][i]) * BUCKET_MIN) % 60
        lab = "立会日" if od else "休場日"
        print(f"  {lab}: 出来高のピークは {h:02d}:{mm:02d} UTC / 1 日合計 {g['v_mean'].sum():,.0f} 枚 / "
              f"建玉の変化の合計 {g['o_mean'].sum():+,.0f} 枚")


if __name__ == "__main__":
    main()
