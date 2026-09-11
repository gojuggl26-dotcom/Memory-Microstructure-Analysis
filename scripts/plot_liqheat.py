"""実際に清算された建玉のヒートマップ(時刻 × 価格)。

    uv run python scripts/plot_liqheat.py --coin xyz:MU

出力: charts/<tag>_liqheat.png      生のヒートマップ(ロング清算 / ショート清算)
      charts/<tag>_liqheat_cum.png  価格帯ごとの累積

縦軸は **`liquidationMarkPx`**(清算が起きた瞬間のマーク価格)である。
これはその口座の建玉が実際に維持証拠金を割った価格そのもので、
推定した清算ラダーではなく**起きた事実**を置いている。

色は 1 セルに入った清算ノーショナル(USD)。**対数**で塗る
(最大の 1 時間が全体の 8.5% を占めるため、線形だと他が全部つぶれる)。
価格の折れ線はセルの**下**に薄く敷く(上に重ねるとセルを隠すため)。

x が確定する時刻 / y の期間: 該当なし(記述)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

SIDES = [("long", "ロングの清算(強制の売り)", "YlOrRd", C2),
         ("short", "ショートの清算(強制の買い)", "YlGnBu", C1)]


def build(ev, px, tbin, pbin):
    t = (ev["ts"].dt.epoch("s").to_numpy() // tbin) * tbin
    p = np.floor(ev["mark_px"].to_numpy() / pbin) * pbin
    usd, lside = ev["notional"].to_numpy(), ev["lside"].to_numpy()
    t0 = (int(px["tb"].min()) // tbin) * tbin
    t1 = (int(px["tb"].max()) // tbin) * tbin
    tax = np.arange(t0, t1 + tbin, tbin)
    plo = np.floor(float(px["lo"].min()) / pbin) * pbin
    phi = np.ceil(float(px["hi"].max()) / pbin) * pbin
    pax = np.arange(plo, phi + pbin, pbin)
    ti = {v: i for i, v in enumerate(tax)}
    pi = {v: i for i, v in enumerate(pax)}
    M = {}
    for s, *_ in SIDES:
        m = np.zeros((pax.size, tax.size))
        k = lside == s
        np.add.at(m, ([pi[v] for v in p[k]], [ti[v] for v in t[k]]), usd[k])
        M[s] = m
        print(f"[heat] {s}: 空でないセル {int((m > 0).sum()):,} / {m.size:,} "
              f"({(m > 0).mean()*100:.3f}%) 最大セル ${m.max()/1e6:.2f}M", file=sys.stderr)
    return M, tax, pax, plo, phi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--tbin", type=int, default=21600)
    ap.add_argument("--pbin", type=float, default=10.0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet")
    px = pl.read_parquet(D / f"liqpx_{tag}.parquet").sort("tb")
    M, tax, pax, plo, phi = build(ev, px, a.tbin, a.pbin)
    M["vmax"] = max(M["long"].max(), M["short"].max())

    pxd = (px.with_columns(tb2=(pl.col("tb") // a.tbin) * a.tbin)
           .group_by("tb2").agg(pl.col("c").last()).sort("tb2"))
    xt = mdates.date2num(pxd["tb2"].to_numpy().astype("datetime64[s]"))
    pc = pxd["c"].to_numpy()
    xe = mdates.date2num((np.r_[tax, tax[-1] + a.tbin]).astype("datetime64[s]"))
    ye = np.r_[pax, pax[-1] + a.pbin]

    for kind in ("raw", "cum"):
        fig = plt.figure(figsize=(14.8, 8.8))
        gs = fig.add_gridspec(2, 3, width_ratios=[6.4, 1.05, 0.10],
                              hspace=0.22, wspace=0.045)
        for r, (s, name, cmap, col) in enumerate(SIDES):
            m0 = M[s]
            grid = np.cumsum(m0, axis=1) if kind == "cum" else m0
            ax = fig.add_subplot(gs[r, 0])
            ax.plot(xt, pc, color=CM, lw=0.7, alpha=0.55, zorder=1)
            im = ax.pcolormesh(xe, ye, np.ma.masked_less_equal(grid, 0), cmap=cmap,
                               norm=LogNorm(vmin=1e3, vmax=M["vmax"] if kind == "raw"
                                            else grid.max()),
                               shading="flat", rasterized=True, zorder=2)
            ax.set_ylim(plo, phi)
            ax.set_ylabel("価格(USD)")
            suf = "" if kind == "raw" else " — 累計"
            ax.set_title(f"{'AB'[r]}. {name}{suf}(合計 ${m0.sum()/1e6:.1f}M / "
                         f"{int((ev['lside'] == s).sum()):,} 件)", loc="left")
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 8, 15, 22)))
            if r == 1:
                ax.set_xlabel("日付(UTC)")

            axr = fig.add_subplot(gs[r, 1], sharey=ax)
            axr.barh(pax, m0.sum(1) / 1e6, height=a.pbin, color=col, align="edge")
            axr.tick_params(labelleft=False)
            axr.set_xlabel("価格帯の合計(百万 USD)", fontsize=7.5)
            axr.set_title("価格帯ごとの合計", fontsize=8.5, loc="left")

            cax = fig.add_subplot(gs[r, 2])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("1 セルの清算額(USD・対数)" if kind == "raw"
                         else "累計の清算額(USD・対数)", fontsize=7.5)
        if kind == "raw":
            save(fig, f"{tag}_liqheat.png",
                 f"{a.coin}: 実際に清算された建玉のヒートマップ"
                 f"(縦 = 清算時のマーク価格 {a.pbin:.0f} USD 刻み / "
                 f"横 = {a.tbin//3600} 時間刻み / {ev['dt'].min()} 〜 {ev['dt'].max()}・"
                 f"{ev.height:,} 件 ${ev['notional'].sum()/1e6:.1f}M)")
        else:
            save(fig, f"{tag}_liqheat_cum.png",
                 f"{a.coin}: 実際に清算された建玉の累積ヒートマップ"
                 f"(同じ格子で、その価格帯にそれまで積み上がった額)")


if __name__ == "__main__":
    main()
