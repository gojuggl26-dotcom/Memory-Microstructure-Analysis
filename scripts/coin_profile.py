"""約定(fills)だけから銘柄の素性を出す。板を組まないので l1 が要らない。

    uv run python scripts/coin_profile.py --coin xyz:INTC [--ref xyz:MU]

出力: data/profile_<tag>.csv(日次)と charts/<tag>_profile.png

fills は 1 取引につき **メイカーとテイカーの 2 行**入っているので、
数量と件数は `crossed`(= テイカー側)だけで数える。両方数えると 2 倍になる。

x が確定する時刻 / y の期間: 該当なし(記述統計)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, D, plt, save  # noqa: E402


def daily(tag: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    f = pl.read_parquet(D / f"fills_{tag}.parquet",
                        columns=["ts", "user", "px", "sz", "side", "crossed",
                                 "tid", "liquidation", "dt"])
    tk = f.filter(pl.col("crossed"))
    d = (tk.group_by("dt").agg(
        n_trade=pl.col("tid").n_unique(),
        n_row=pl.len(),
        vol=pl.col("sz").sum(),
        notional=(pl.col("sz") * pl.col("px")).sum(),
        px_open=pl.col("px").first(), px_close=pl.col("px").last(),
        px_min=pl.col("px").min(), px_max=pl.col("px").max(),
        px_med=pl.col("px").median(),
        sz_med=pl.col("sz").median(), sz_p99=pl.col("sz").quantile(0.99),
        n_taker=pl.col("user").n_unique(),
        buy_vol=pl.col("sz").filter(pl.col("side") == "B").sum(),
        sell_vol=pl.col("sz").filter(pl.col("side") == "A").sum(),
        n_liq=pl.col("liquidation").sum(),
    ).sort("dt"))
    mk = (f.filter(~pl.col("crossed")).group_by("dt")
          .agg(n_maker=pl.col("user").n_unique()).sort("dt"))
    d = d.join(mk, on="dt", how="left")
    hr = (tk.with_columns(h=pl.col("ts").dt.hour())
          .group_by("h").agg(vol=pl.col("sz").sum(), n=pl.len()).sort("h"))
    return d, hr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d, hr = daily(tag)
    d.write_csv(D / f"profile_{tag}.csv")
    nd = d.height
    print(f"{a.coin}: {nd} 日 ({d['dt'][0]} 〜 {d['dt'][-1]})")
    print(f"  取引 {d['n_trade'].sum():,} 件 / 出来高 {d['vol'].sum():,.0f} 枚 / "
          f"想定元本 {d['notional'].sum()/1e9:.3f} B USD")
    print(f"  1 日あたり 取引 {d['n_trade'].median():,.0f} 件 "
          f"({d['n_trade'].median()/86400:.2f} 件/秒) / "
          f"想定元本 {d['notional'].median()/1e6:.2f} M USD")
    print(f"  1 取引 中央 {d['sz_med'].median():.3f} 枚 "
          f"(= {d['sz_med'].median()*d['px_med'].median():,.0f} USD) / "
          f"99% 点 {d['sz_p99'].median():.3f} 枚")
    print(f"  価格 {d['px_min'].min():.2f} 〜 {d['px_max'].max():.2f} "
          f"(中央 {d['px_med'].median():.2f})")
    print(f"  テイカー口座 中央 {d['n_taker'].median():.0f} 者/日 / "
          f"メイカー口座 中央 {d['n_maker'].median():.0f} 者/日")
    print(f"  強制決済を含む行 {d['n_liq'].sum():,}")
    bs = d["buy_vol"].sum() / (d["buy_vol"].sum() + d["sell_vol"].sum())
    print(f"  テイカー買いの数量シェア {bs*100:.2f}%")

    ref = None
    if a.ref:
        rt = a.ref.replace(":", "_")
        if (D / f"fills_{rt}.parquet").exists():
            ref, _ = daily(rt)

    x = np.arange(nd)
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 6.2))
    a0 = ax[0, 0]
    a0.fill_between(x, d["px_min"], d["px_max"], color=C1, alpha=0.22, lw=0)
    a0.plot(x, d["px_close"], color=C1, lw=1.2)
    a0.set_title("A. 価格(日の高安と終値)")
    a0.set_ylabel("USD")

    a1 = ax[0, 1]
    a1.plot(x, d["notional"] / 1e6, color=C2, lw=1.1, label=a.coin)
    if ref is not None:
        m = min(ref.height, nd)
        a1.plot(np.arange(m), ref["notional"][:m] / 1e6, color=CM, lw=1.0,
                label=a.ref)
    a1.set_yscale("log")
    a1.set_title("B. 日次の想定元本(百万 USD・対数軸)")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[1, 0]
    a2.plot(x, d["n_trade"] / 86400, color=C3, lw=1.1, label="取引 (件/秒)")
    a2.plot(x, d["n_taker"] / 100, color=C1, lw=1.0, ls="--",
            label="テイカー口座 ÷100")
    a2.plot(x, d["n_maker"] / 100, color=C2, lw=1.0, ls=":", label="メイカー口座 ÷100")
    a2.set_xlabel("日 (標本の通し番号)")
    a2.set_title("C. 取引の頻度と参加者")
    a2.legend(fontsize=7, frameon=False)

    a3 = ax[1, 1]
    a3.bar(hr["h"], hr["vol"] / hr["vol"].sum() * 100, color=C1, width=0.8)
    a3.axvspan(13.5, 20, color=C2, alpha=0.12, lw=0)
    a3.text(16.7, a3.get_ylim()[1] * 0.92, "米国立会 13:30–20:00 UTC",
            ha="center", fontsize=7, color=C2)
    a3.set_xlabel("時 (UTC)")
    a3.set_ylabel("出来高シェア (%)")
    a3.set_title("D. 時刻ごとの出来高")
    save(fig, f"{tag}_profile.png",
         f"{a.coin}: 約定から見た銘柄の素性({nd} 日)")


if __name__ == "__main__":
    main()
