r"""Derive — RFQ でマーケットメイクが可能かを実測で判定する。

問い: RFQ 経由の +9.48bp の edge は「新規参入者が取りに行ける」ものか、
それとも既存の少数に閉じているのか。

★観測できるのは約定した RFQ だけである。提示して負けた見積り、
  引き合いを受けなかった事実は記録に残らない(選択が入る)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "rfq"


def hhi(v: np.ndarray) -> float:
    s = v.sum()
    if s <= 0:
        return float("nan")
    w = v / s
    return float((w ** 2).sum())


def main() -> int:
    T = pl.read_parquet(SRC / "ledger_trades.parquet")
    M = pl.read_parquet(OUT / "maker_edge.parquet")

    print("=" * 78)
    print("[1] RFQ を取っているメイカーは何者か")
    print("=" * 78)

    R = M.filter(pl.col("rfq"))
    C = M.filter(~pl.col("rfq"))
    for lab, D in (("RFQ", R), ("CLOB", C)):
        w = D.group_by("wallet").agg(pl.col("notional").sum().alias("n")).sort("n", descending=True)
        v = w["n"].to_numpy()
        print(f"{lab:5s} メイカー {len(v):5d} 者 / 名目 ${v.sum()/1e9:6.2f}B / "
              f"HHI {hhi(v):.3f} / 上位3 {v[:3].sum()/v.sum()*100:5.1f}% / "
              f"上位10 {v[:10].sum()/v.sum()*100:5.1f}%")

    print("\n年別 RFQ メイカー(名目基準)")
    print(f"{'年':6s} {'メイカー数':>8s} {'HHI':>7s} {'上位1':>7s} {'上位3':>7s} {'約定/日':>8s}")
    for yr in ("2024", "2025", "2026"):
        D = R.filter(pl.col("day").str.starts_with(yr))
        if D.height == 0:
            continue
        w = D.group_by("wallet").agg(pl.col("notional").sum().alias("n")).sort("n", descending=True)
        v = w["n"].to_numpy()
        nd = D["day"].n_unique()
        print(f"{yr:6s} {len(v):8d} {hhi(v):7.3f} {v[:1].sum()/v.sum()*100:6.1f}% "
              f"{v[:3].sum()/v.sum()*100:6.1f}% {D.height/nd:8.0f}")

    print("\n" + "=" * 78)
    print("[2] 新規参入者は RFQ を取れるのか — 初約定から初 RFQ までの距離")
    print("=" * 78)
    first_any = T.group_by("wallet").agg(pl.col("ts").min().alias("t0"))
    first_rfq = (R.group_by("wallet")
                 .agg(pl.col("ts").min().alias("t1"),
                      pl.col("notional").sum().alias("rfq_nom"),
                      pl.len().alias("rfq_n")))
    J = first_rfq.join(first_any, on="wallet", how="left").with_columns(
        ((pl.col("t1") - pl.col("t0")) / 86400000.0).alias("days_to_rfq"))
    d = J["days_to_rfq"].to_numpy()
    print(f"RFQ メイカーになった経験のあるウォレット: {len(d)} 者")
    print(f"  初約定から初 RFQ メイカー約定まで [日]: "
          f"中央 {np.median(d):.1f} / 25% {np.percentile(d,25):.1f} / "
          f"75% {np.percentile(d,75):.1f} / 最大 {d.max():.0f}")
    print(f"  初日のうちに RFQ を取った者: {(d < 1).sum()} 者 ({(d<1).mean()*100:.0f}%)")

    J = J.with_columns(pl.from_epoch(pl.col("t1"), time_unit="ms").dt.strftime("%Y").alias("yr1"))
    print("\n  初めて RFQ を取った年別(その後の RFQ 名目)")
    for r in (J.group_by("yr1").agg(pl.len().alias("n"),
                                    pl.col("rfq_nom").sum().alias("nom"),
                                    pl.col("rfq_nom").median().alias("med"))
              .sort("yr1").iter_rows(named=True)):
        print(f"    {r['yr1']}: {r['n']:4d} 者 / 合計 ${r['nom']/1e9:5.2f}B / 中央 ${r['med']:,.0f}")

    print("\n" + "=" * 78)
    print("[3] RFQ の edge は上位以外にも届いているか")
    print("=" * 78)
    w = R.group_by("wallet").agg(pl.col("notional").sum().alias("nom")).sort("nom", descending=True)
    top3 = w["wallet"].to_list()[:3]
    top10 = w["wallet"].to_list()[:10]

    def stat(D, lab):
        if D.height == 0:
            return
        n = D["notional"].sum()
        e = float((D["edge_bp"] * D["notional"]).sum() / n)
        mo = D.filter(pl.col("markout_bp").is_not_nan())
        m = (float((mo["markout_bp"] * mo["notional"]).sum() / mo["notional"].sum())
             if mo.height else float("nan"))
        print(f"  {lab:10s} {D.height:7,d} 約定 / {D['wallet'].n_unique():4d} 者 / "
              f"${n/1e9:6.3f}B / edge {e:+7.3f} bp / markout {m:+7.3f} bp / "
              f"負の edge {(D['edge_bp']<0).mean()*100:4.1f}%")

    print("RFQ:")
    stat(R.filter(pl.col("wallet").is_in(top3)), "上位3")
    stat(R.filter(pl.col("wallet").is_in(top10) & ~pl.col("wallet").is_in(top3)), "4-10位")
    stat(R.filter(~pl.col("wallet").is_in(top10)), "11位以下")
    print("CLOB:")
    wc = C.group_by("wallet").agg(pl.col("notional").sum().alias("nom")).sort("nom", descending=True)
    c3 = wc["wallet"].to_list()[:3]
    c10 = wc["wallet"].to_list()[:10]
    stat(C.filter(pl.col("wallet").is_in(c3)), "上位3")
    stat(C.filter(pl.col("wallet").is_in(c10) & ~pl.col("wallet").is_in(c3)), "4-10位")
    stat(C.filter(~pl.col("wallet").is_in(c10)), "11位以下")

    print("\n" + "=" * 78)
    print("[4] 引き合いの規模と頻度 — 何をどれだけ捌く必要があるか")
    print("=" * 78)
    for lab, D in (("RFQ", R), ("CLOB", C)):
        q = np.percentile(D["notional"].to_numpy(), [50, 75, 90, 99, 99.9])
        print(f"{lab:5s} 名目 [$]: 中央 {q[0]:>9,.0f} / 75% {q[1]:>9,.0f} / "
              f"90% {q[2]:>10,.0f} / 99% {q[3]:>11,.0f} / 99.9% {q[4]:>12,.0f} / "
              f"最大 {D['notional'].max():>13,.0f}")
    nd = R["day"].n_unique()
    print(f"\nRFQ 約定(メイカー行)は {nd} 日で {R.height:,} 件 = 1 日 {R.height/nd:.0f} 件")
    recent = R.filter(pl.col("day") >= "2026-06-01")
    rd = recent["day"].n_unique()
    print(f"直近(2026-06 以降 {rd} 日): 1 日 {recent.height/rd:.0f} 件 / "
          f"名目 ${recent['notional'].sum()/rd/1e6:.1f}M")

    print("\n" + "=" * 78)
    print("[5] 1 回の引き合いは複数銘柄か(多脚を建値できる必要があるか)")
    print("=" * 78)
    grp = R.group_by(["ts", "wallet"]).agg(pl.col("ins").n_unique().alias("legs"),
                                           pl.col("notional").sum().alias("nom"))
    lg = grp["legs"].to_numpy()
    print(f"RFQ 取引 {len(lg):,} 回 / 脚数: 1 脚 {(lg==1).mean()*100:5.1f}% / "
          f"2 脚 {(lg==2).mean()*100:4.1f}% / 3 脚以上 {(lg>=3).mean()*100:4.1f}% / 最大 {lg.max()} 脚")
    print(f"  多脚(2 脚以上)が名目に占める割合: "
          f"{grp.filter(pl.col('legs')>=2)['nom'].sum()/grp['nom'].sum()*100:.1f}%")
    gc = C.group_by(["ts", "wallet"]).agg(pl.col("ins").n_unique().alias("legs"),
                                          pl.col("notional").sum().alias("nom"))
    lc = gc["legs"].to_numpy()
    print(f"CLOB 取引 {len(lc):,} 回 / 1 脚 {(lc==1).mean()*100:5.1f}% / "
          f"多脚が名目に占める割合 {gc.filter(pl.col('legs')>=2)['nom'].sum()/gc['nom'].sum()*100:.1f}%")

    print("\n" + "=" * 78)
    print("[6] 引き合いを出している側(テイカー)は誰か")
    print("=" * 78)
    TK = T.filter(pl.col("rfq") & (pl.col("role") == "taker")).with_columns(
        (pl.col("amt").abs() * pl.col("index")).alias("notional"))
    w2 = TK.group_by("wallet").agg(pl.col("notional").sum().alias("n")).sort("n", descending=True)
    v = w2["n"].to_numpy()
    print(f"RFQ テイカー {len(v):,} 者 / HHI {hhi(v):.3f} / 上位3 {v[:3].sum()/v.sum()*100:.1f}% / "
          f"上位10 {v[:10].sum()/v.sum()*100:.1f}%")
    TKC = T.filter((~pl.col("rfq")) & (pl.col("role") == "taker")).with_columns(
        (pl.col("amt").abs() * pl.col("index")).alias("notional"))
    wc2 = TKC.group_by("wallet").agg(pl.col("notional").sum().alias("n")).sort("n", descending=True)
    vc = wc2["n"].to_numpy()
    print(f"CLOB テイカー {len(vc):,} 者 / HHI {hhi(vc):.3f} / 上位3 {vc[:3].sum()/vc.sum()*100:.1f}% / "
          f"上位10 {vc[:10].sum()/vc.sum()*100:.1f}%")
    mk = R["wallet"].unique().to_list()
    both = w2.filter(pl.col("wallet").is_in(mk))
    print(f"RFQ テイカーのうち RFQ メイカーでもある者: {both.height} 者 / "
          f"テイカー名目の {both['n'].sum()/v.sum()*100:.1f}%")

    res = {"rfq_makers": int(R["wallet"].n_unique()),
           "clob_makers": int(C["wallet"].n_unique()),
           "rfq_takers": int(len(v)),
           "rfq_per_day_recent": float(recent.height / rd)}
    (OUT / "feasibility.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {OUT/'feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
