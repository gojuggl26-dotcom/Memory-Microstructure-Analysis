"""板に指値を出している口座ごとの癖と、その集中度を測る(L4 でしか作れない層)。

    uv run python scripts/analyze_wallet_orders.py --coin xyz:INTC

入力: data/orderlife_<tag>/dt=*.parquet(`user` 列つき。build_orderlife.py)
出力: data/wallet_orders_<tag>.csv(口座 × 日)
      data/wallet_conc_daily_<tag>.csv(日次の集中度)
      charts/<tag>_wallet_orders.png

測るもの
--------
1. **集中度** — その日に出された指値の数量が何者に集まっているか。
   シェア $`w_i`$ から HHI $`\\sum_i w_i^2`$、実効口座数 $`1/\\mathrm{HHI}`$、
   上位 1 / 5 / 10 者のシェア。
2. **口座ごとの癖** — 発注時に判る量(数量・最良からの距離)と、
   結末(寿命・取消率・約定率)。マーケットメイカーらしさの素点になる。

★分母は「その日に出された注文の数量」であって「その瞬間に板にある数量」ではない
--------------------------------------------------------------------------------
同じ「集中度」でも分母が変わると別の量になる。板に在る量で測ると、
長く置きっぱなしの口座が重くなる。ここは**発注の流量**で測っている。
`mu_wallet_conc_report.md` は板に在る量で測っており、直接は比較できない。

★`reduce_only` は数量の集計から外す
-----------------------------------
残量の自動縮小を約定と数えないため(`build_orderlife.py` の注記)。

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
from build_obi_levels import SZ_LOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((D / f"orderlife_{tag}").glob("dt=*.parquet"))
    if not files:
        sys.exit("orderlife が無い")

    parts = []
    for fp in files:
        d = pl.read_parquet(fp, columns=["user", "dt_open", "is_bid", "orig_lots",
                                         "dist_tick", "life_ns", "filled_lots",
                                         "reduce_only"])
        d = d.filter(pl.col("user").is_not_null() & (~pl.col("reduce_only")))
        if d.height == 0:
            continue
        parts.append(d.group_by("dt_open", "user").agg(
            n=pl.len(),
            vol=pl.col("orig_lots").sum(),
            n_bid=pl.col("is_bid").sum(),
            n_touch=(pl.col("dist_tick") <= 0).sum(),
            n_inside=(pl.col("dist_tick") < 0).sum(),
            sum_dist=pl.col("dist_tick").filter(pl.col("dist_tick") >= 0).sum(),
            n_dist=(pl.col("dist_tick") >= 0).sum(),
            sum_life=pl.col("life_ns").sum(),
            n_fast=(pl.col("life_ns") <= 10 ** 9).sum(),
            n_fill=(pl.col("filled_lots") > 0).sum(),
            fill_vol=pl.col("filled_lots").sum(),
            sz_med=pl.col("orig_lots").median(),
            life_med=pl.col("life_ns").median()))
    if not parts:
        sys.exit("user 列が空。先に fetch_l1_user.py を回すこと")
    W = (pl.concat(parts).group_by("dt_open", "user").agg(
        pl.col(["n", "vol", "n_bid", "n_touch", "n_inside", "sum_dist", "n_dist",
                "sum_life", "n_fast", "n_fill", "fill_vol"]).sum(),
        pl.col("sz_med").median(), pl.col("life_med").median())
        .sort("dt_open", "vol", descending=[False, True]))
    W = W.with_columns(vol_c=pl.col("vol") * SZ_LOT,
                       fill_c=pl.col("fill_vol") * SZ_LOT,
                       bid_share=pl.col("n_bid") / pl.col("n"),
                       touch_share=pl.col("n_touch") / pl.col("n"),
                       fast_share=pl.col("n_fast") / pl.col("n"),
                       fill_rate=pl.col("n_fill") / pl.col("n"),
                       life_s=pl.col("life_med") / 1e9,
                       mean_dist=pl.col("sum_dist") / pl.col("n_dist").clip(1))
    W.write_csv(D / f"wallet_orders_{tag}.csv")

    rows = []
    for dt_, g in W.partition_by("dt_open", as_dict=True).items():
        v = np.sort(g["vol_c"].to_numpy())[::-1]
        w = v / v.sum()
        rows.append({"dt": dt_[0], "n_wallet": v.size,
                     "hhi": float((w * w).sum()),
                     "neff": float(1.0 / (w * w).sum()),
                     "top1": float(w[0]),
                     "top5": float(w[:5].sum()), "top10": float(w[:10].sum()),
                     "vol": float(v.sum()),
                     "n_orders": int(g["n"].sum())})
    C = pl.DataFrame(rows).sort("dt")
    C.write_csv(D / f"wallet_conc_daily_{tag}.csv")

    print(f"{a.coin}: 口座 × 日 {W.height:,} 行 / 実口座 {W['user'].n_unique():,} 者")
    print(f"  1 日あたり 指値を出した口座 中央 {C['n_wallet'].median():.0f} 者 "
          f"(最小 {C['n_wallet'].min()} / 最大 {C['n_wallet'].max()})")
    print(f"  数量の HHI 中央 {C['hhi'].median():.4f} → 実効 {C['neff'].median():.2f} 者")
    print(f"  上位 1 者 {C['top1'].median()*100:.1f}% / 上位 5 者 "
          f"{C['top5'].median()*100:.1f}% / 上位 10 者 {C['top10'].median()*100:.1f}%")

    # 窓全体での口座の癖(数量の多い順に上位 20 者)
    G = (W.group_by("user").agg(
        n=pl.col("n").sum(), vol=pl.col("vol_c").sum(),
        days=pl.len(),
        bid_share=(pl.col("n_bid").sum() / pl.col("n").sum()),
        touch=(pl.col("n_touch").sum() / pl.col("n").sum()),
        inside=(pl.col("n_inside").sum() / pl.col("n").sum()),
        fast=(pl.col("n_fast").sum() / pl.col("n").sum()),
        fill=(pl.col("n_fill").sum() / pl.col("n").sum()),
        life=pl.col("life_s").median(),
        sz=(pl.col("sz_med").median() * SZ_LOT))
        .sort("vol", descending=True))
    G = G.with_columns(wid=pl.int_range(1, G.height + 1))
    G.select("wid", "n", "vol", "days", "bid_share", "touch", "inside", "fast",
             "fill", "life", "sz").write_csv(D / f"wallet_profile_{tag}.csv")
    print("\n  数量上位 12 者(口座は通し番号のみ・アドレスは載せない)")
    print("    wid       本数      枚数  日数  買い比  最良以内  1秒以内  約定率  寿命s   中央枚")
    for r in G.head(12).iter_rows(named=True):
        print(f"    {r['wid']:>3d} {r['n']:>10,} {r['vol']:>9,.0f} {r['days']:>5d} "
              f"{r['bid_share']:>7.3f} {r['touch']:>9.3f} {r['fast']:>8.3f} "
              f"{r['fill']:>7.4f} {r['life']:>6.2f} {r['sz']:>8.3f}")

    x = np.arange(C.height)
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 6.6))
    a0 = ax[0, 0]
    a0.plot(x, C["n_wallet"], color=C1, lw=1.1, label="指値を出した口座")
    a0.plot(x, C["neff"], color=C2, lw=1.1, label="実効口座数 1/HHI")
    a0.set_yscale("log")
    a0.set_ylabel("者")
    a0.set_title("A. 何者が板を作っているか")
    a0.legend(fontsize=7, frameon=False)

    a1 = ax[0, 1]
    a1.plot(x, C["top1"] * 100, color=C1, lw=1.1, label="上位 1 者")
    a1.plot(x, C["top5"] * 100, color=C2, lw=1.1, label="上位 5 者")
    a1.plot(x, C["top10"] * 100, color=C3, lw=1.1, label="上位 10 者")
    a1.set_ylabel("発注数量に占めるシェア (%)")
    a1.set_ylim(0, 102)
    a1.set_title("B. 集中度")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[1, 0]
    g = G.head(60)
    v = g["vol"].to_numpy().astype(float)
    sz = 10.0 + 150.0 * np.sqrt(v / v.max())      # 面積を最大値で正規化する
    a2.scatter(g["life"], g["touch"] * 100, s=sz,
               color=C1, alpha=0.55, linewidths=0)
    a2.set_xscale("log")
    a2.set_xlabel("注文の寿命の中央値 (秒・対数軸)")
    a2.set_ylabel("最良以内に置いた割合 (%)")
    a2.set_title("C. 上位 60 者の置き方(点の大きさ = 発注数量)")
    for r in g.head(6).iter_rows(named=True):
        a2.annotate(f"#{r['wid']}", (r["life"], r["touch"] * 100), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", color=C2)

    a3 = ax[1, 1]
    a3.scatter(g["fast"] * 100, g["fill"] * 100, s=sz,
               color=C3, alpha=0.55, linewidths=0)
    a3.set_xlabel("1 秒以内に消した割合 (%)")
    a3.set_ylabel("一度でも約定した割合 (%)")
    a3.set_yscale("log")
    a3.set_title("D. 引きの速さと約定率")
    save(fig, f"{tag}_wallet_orders.png",
         f"{a.coin}: 指値を出した口座 {W['user'].n_unique():,} 者の集中度と癖"
         f"({C.height} 日)")


if __name__ == "__main__":
    main()
