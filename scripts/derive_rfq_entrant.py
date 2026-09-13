r"""Derive — RFQ メイカー 84 者を「規模」と「参入年」で切り、実現損益で見る。

[3] の「11 位以下の edge が +15.86bp」は mark 基準の見かけの値でしかない。
実現損益(反対売買 + 満期決済 + perp)で裏を取る。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "rfq"


def main() -> int:
    M = pl.read_parquet(OUT / "maker_edge.parquet")
    W = pl.read_parquet(SRC / "mm" / "wallet_pnl.parquet")
    R = M.filter(pl.col("rfq"))

    g = (R.group_by("wallet").agg(
        pl.len().alias("rfq_n"),
        pl.col("notional").sum().alias("rfq_nom"),
        ((pl.col("edge_bp") * pl.col("notional")).sum()
         / pl.col("notional").sum()).alias("edge_w"),
        pl.col("edge_bp").median().alias("edge_med"),
        pl.col("ts").min().alias("t1"),
        pl.col("day").min().alias("d1"),
        pl.col("day").max().alias("dN"),
        pl.col("day").n_unique().alias("n_day"),
    ).with_columns(pl.col("d1").str.slice(0, 4).alias("yr1")))

    A = g.join(W.select(["wallet", "pnl_realized", "pnl_opt_only", "pnl_perp",
                         "pnl_settle", "notional_both", "n_rows", "maker_share"]),
               on="wallet", how="left").sort("rfq_nom", descending=True)

    print("=" * 86)
    print("[A] RFQ メイカー 84 者 — 規模階層ごとの実現損益")
    print("=" * 86)
    print(f"{'階層':10s} {'者数':>4s} {'RFQ名目':>9s} {'edge_w':>8s} "
          f"{'実現損益':>13s} {'黒字':>7s} {'中央値':>11s}")
    for lab, sl in (("上位3", slice(0, 3)), ("4-10位", slice(3, 10)),
                    ("11-30位", slice(10, 30)), ("31位以下", slice(30, None))):
        D = A[sl]
        p = D["pnl_realized"].to_numpy()
        print(f"{lab:10s} {D.height:4d} ${D['rfq_nom'].sum()/1e9:7.3f}B "
              f"{D['edge_w'].mean():+8.2f} ${p.sum():12,.0f} "
              f"{(p>0).mean()*100:6.0f}% ${np.median(p):10,.0f}")

    print("\n" + "=" * 86)
    print("[B] 初めて RFQ を取った年ごと(= 参入コホート)")
    print("=" * 86)
    print(f"{'年':6s} {'者数':>4s} {'RFQ名目':>9s} {'RFQ日数(中央)':>13s} "
          f"{'実現損益':>13s} {'黒字':>7s} {'中央値':>11s}")
    for yr in ("2024", "2025", "2026"):
        D = A.filter(pl.col("yr1") == yr)
        if D.height == 0:
            continue
        p = D["pnl_realized"].to_numpy()
        print(f"{yr:6s} {D.height:4d} ${D['rfq_nom'].sum()/1e9:7.3f}B "
              f"{D['n_day'].median():13.0f} ${p.sum():12,.0f} "
              f"{(p>0).mean()*100:6.0f}% ${np.median(p):10,.0f}")

    print("\n" + "=" * 86)
    print("[C] 上位 12 者(RFQ 名目の降順)")
    print("=" * 86)
    print(f"{'ウォレット':14s} {'初RFQ':>10s} {'RFQ約定':>8s} {'RFQ日数':>7s} "
          f"{'RFQ名目':>9s} {'edge_w':>8s} {'実現損益':>13s}")
    for r in A.head(12).iter_rows(named=True):
        print(f"{r['wallet'][:12]:14s} {r['d1']:>10s} {r['rfq_n']:8,d} "
              f"{r['n_day']:7d} ${r['rfq_nom']/1e9:7.3f}B {r['edge_w']:+8.2f} "
              f"${r['pnl_realized']:12,.0f}")

    print("\n" + "=" * 86)
    print("[D] 11 位以下(74 者)の edge は本物か — 分布を見る")
    print("=" * 86)
    tail = A[10:]
    e = tail["edge_w"].to_numpy()
    n = tail["rfq_n"].to_numpy()
    print(f"者数 {len(e)} / RFQ 約定の中央値 {np.median(n):.0f} 件 / "
          f"10 件未満の者 {(n<10).sum()} 者")
    print(f"  加重 edge: 中央 {np.median(e):+.2f} / 25% {np.percentile(e,25):+.2f} / "
          f"75% {np.percentile(e,75):+.2f} / 正の者 {(e>0).mean()*100:.0f}%")
    tail10 = tail.filter(pl.col("rfq_n") >= 10)
    e2 = tail10["edge_w"].to_numpy()
    nm = tail10["rfq_nom"].sum()
    ew = float((tail10["edge_w"] * tail10["rfq_nom"]).sum() / nm)
    print(f"  10 件以上に限る({tail10.height} 者): 名目加重 edge {ew:+.2f} bp / "
          f"中央 {np.median(e2):+.2f} / 正の者 {(e2>0).mean()*100:.0f}%")
    p = tail10["pnl_realized"].to_numpy()
    print(f"  同じ {tail10.height} 者の実現損益: 合計 ${p.sum():,.0f} / "
          f"黒字 {(p>0).mean()*100:.0f}% / 中央 ${np.median(p):,.0f}")

    print("\n" + "=" * 86)
    print("[E] RFQ を一度も取っていないメイカーとの比較(maker 比 70% 以上・200 約定以上)")
    print("=" * 86)
    rfq_set = R["wallet"].unique().to_list()
    B = W.filter((pl.col("maker_share") >= 0.7) & (pl.col("n_rows") >= 200))
    for lab, D in (("RFQ 経験あり", B.filter(pl.col("wallet").is_in(rfq_set))),
                   ("RFQ 経験なし", B.filter(~pl.col("wallet").is_in(rfq_set)))):
        p = D["pnl_realized"].to_numpy()
        nm2 = D["notional_both"].sum()
        print(f"{lab:14s} {D.height:4d} 者 / 名目 ${nm2/1e9:6.2f}B / "
              f"損益 ${p.sum():12,.0f} = {p.sum()/nm2*1e4:+7.2f} bp / "
              f"黒字 {(p>0).mean()*100:3.0f}% / 中央 ${np.median(p):>10,.0f}")

    A.write_parquet(OUT / "rfq_makers.parquet")
    print(f"\n-> {OUT/'rfq_makers.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
