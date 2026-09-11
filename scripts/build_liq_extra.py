"""清算の徹底分析 — 価格帯の滞在時間・カスケード・集中・季節性・OI 比。

    uv run python scripts/build_liq_extra.py --coin xyz:MU

入力: data/liqev_<coin>.parquet / data/fills_<coin>.parquet / data/oi_series_<coin>.parquet
出力(すべて data/):
    liqprof_<coin>.csv    価格帯ごとの清算額・滞在時間・出来高・強度
    liqcasc_<coin>.csv    清算どうしの間隔と、時間内一様の帰無との比
    liqburst_<coin>.csv   60 秒以内で繋がる塊(バースト)の規模
    liqacct_<coin>.csv    口座ごとの清算額・回数(住所は出さない)
    liqhour_<coin>.csv    UTC 時刻帯ごとの清算額・出来高・強度
    liqday_<coin>.csv     日次の清算額・出来高・値幅・OI

★分母を必ず添える
------------------
「この価格帯で清算が多い」は、その価格帯に**価格が長く居ただけ**かもしれない。
価格帯ごとに滞在時間(最終約定価格で補間)と出来高を出し、
**単位時間あたり・単位出来高あたり**の強度も並べる。

★カスケードの帰無対照
--------------------
清算の到着率は 1 時間の中でも激しく変わる。素の指数分布と比べると
「固まっている」のは当たり前になる。そこで**同じ 1 時間の中で一様に**
置き直した系列を 200 回作り、その分布と比べる。これで「時間帯の濃淡」を
除いた、真の自己励起だけが残る。

x が確定する時刻 / y の期間: 該当なし(記述統計)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
NS = 10 ** 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--pbin", type=float, default=10.0)
    ap.add_argument("--nnull", type=int, default=200)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(0)
    ev = pl.read_parquet(D / f"liqev_{tag}.parquet").sort("ts")
    ts = ev["ts"].dt.epoch("ns").to_numpy()
    usd = ev["notional"].to_numpy()
    sz = ev["sz"].to_numpy()
    side = ev["lside"].to_numpy()

    # ---------- A. 価格帯ごとの滞在時間と出来高 --------------------------
    f = (pl.scan_parquet(D / f"fills_{tag}.parquet").filter(pl.col("crossed"))
         .select("ts", "px", "sz").sort("ts").collect())
    ft = f["ts"].dt.epoch("ns").to_numpy()
    fp = f["px"].to_numpy()
    fv = (f["px"] * f["sz"]).to_numpy()
    dwell = np.diff(ft, append=ft[-1]) / NS          # 次の約定までの秒数
    dwell = np.clip(dwell, 0, 3600)                  # 板が止まった長い空白は 1 時間で頭打ち
    pb = np.floor(fp / a.pbin) * a.pbin
    lo = float(pb.min())
    hi = float(np.floor(ev["mark_px"].max() / a.pbin) * a.pbin)
    hi = max(hi, float(pb.max()))
    edges = np.arange(lo, hi + a.pbin, a.pbin)
    idx = np.clip(((pb - lo) / a.pbin).astype(int), 0, edges.size - 1)
    t_at = np.bincount(idx, weights=dwell, minlength=edges.size)
    v_at = np.bincount(idx, weights=fv, minlength=edges.size)
    n_at = np.bincount(idx, minlength=edges.size)

    eb = np.clip(((np.floor(ev["mark_px"].to_numpy() / a.pbin) * a.pbin - lo)
                  / a.pbin).astype(int), 0, edges.size - 1)
    liq_l = np.bincount(eb[side == "long"], weights=usd[side == "long"],
                        minlength=edges.size)
    liq_s = np.bincount(eb[side == "short"], weights=usd[side == "short"],
                        minlength=edges.size)
    prof = pl.DataFrame({
        "px_bucket": edges, "liq_long_usd": liq_l, "liq_short_usd": liq_s,
        "liq_usd": liq_l + liq_s, "dwell_h": t_at / 3600, "vol_usd": v_at,
        "n_trades": n_at,
    }).with_columns(
        liq_per_hour=pl.when(pl.col("dwell_h") > 0.5)
                      .then(pl.col("liq_usd") / pl.col("dwell_h")).otherwise(None),
        liq_per_vol_bp=pl.when(pl.col("vol_usd") > 1e6)
                        .then(pl.col("liq_usd") / pl.col("vol_usd") * 1e4).otherwise(None))
    prof.write_csv(D / f"liqprof_{tag}.csv")
    ok = prof.filter(pl.col("liq_per_hour").is_not_null())
    print(f"[A] 価格帯 {prof.height} / 清算が起きた帯 {int((prof['liq_usd']>0).sum())}")
    print("    単位時間あたり清算額の上位:")
    print(ok.sort("liq_per_hour", descending=True)
          .select("px_bucket", "liq_usd", "dwell_h", "liq_per_hour").head(5))
    r = np.corrcoef(np.log1p(ok["liq_usd"]), np.log1p(ok["dwell_h"]))[0, 1]
    print(f"    log(清算額) と log(滞在時間) の相関 {r:.3f}")

    # ---------- B. 清算エンジンの刻みと連鎖 -------------------------------
    # 清算は「同じミリ秒にまとめて起きる」= ラウンド。まずラウンドに畳む。
    rt, rinv = np.unique(ts, return_inverse=True)
    rnd = (pl.DataFrame({"r": rinv, "usd": usd, "sz": sz,
                         "long": (side == "long").astype(float)})
           .group_by("r").agg(pl.len().alias("n"), pl.col("usd").sum(),
                              pl.col("sz").sum(), pl.col("long").mean().alias("long_share"))
           .sort("r"))
    gu = np.diff(rt) / NS                       # ラウンド間の秒
    print(f"\n[B] ラウンド {rt.size:,} 個 / 1 ラウンドあたり中央 {rnd['n'].median():.0f} 件・"
          f"最大 {rnd['n'].max()} 件")
    print(f"    同じミリ秒に起きる清算の割合 {(1 - rt.size/ts.size)*100:.1f}%")
    print(f"    ラウンド間隔の最小 {gu.min():.3f} 秒 / 2.9〜3.2 秒に入る割合 "
          f"{((gu >= 2.9) & (gu <= 3.2)).mean()*100:.1f}%")
    hist, edg = np.histogram(gu[gu < 12], bins=120, range=(0, 12))
    pl.DataFrame({"gap_s_lo": edg[:-1], "gap_s_hi": edg[1:], "n": hist}) \
        .write_csv(D / f"liqcasc_{tag}.csv")

    # 連鎖 = 1 刻み(3.5 秒)以内で繋がるラウンドの列
    TICK = 3.5
    cid = np.r_[0, np.cumsum(gu > TICK)]
    ch = (pl.DataFrame({"c": cid, "n": rnd["n"], "usd": rnd["usd"], "sz": rnd["sz"],
                        "ls": rnd["long_share"] * rnd["n"], "t": rt})
          .group_by("c").agg(pl.len().alias("rounds"), pl.col("n").sum(),
                             pl.col("usd").sum(), pl.col("sz").sum(),
                             (pl.col("ls").sum() / pl.col("n").sum()).alias("long_share"),
                             ((pl.col("t").max() - pl.col("t").min()) / NS).alias("dur_s"),
                             pl.col("t").min().alias("t0"))
          .sort("usd", descending=True))
    ch.write_csv(D / f"liqburst_{tag}.csv")
    tot = usd.sum()
    print(f"    連鎖 {ch.height} 本(1 ラウンドで終わる {int((ch['rounds']==1).sum())} 本)")
    for k in (1, 5, 10, 50):
        print(f"      上位 {k:>2} 連鎖で全体の {ch['usd'].head(k).sum()/tot*100:5.1f}%")
    big = ch.filter(pl.col("rounds") >= 2)
    print(f"    最長の連鎖: {int(ch.sort('rounds', descending=True)['rounds'][0])} ラウンド")
    print(f"    最大の連鎖: {ch['rounds'][0]} ラウンド {ch['n'][0]} 件 "
          f"${ch['usd'][0]/1e6:.2f}M {ch['dur_s'][0]:.0f} 秒 "
          f"ロング比率 {ch['long_share'][0]:.2f}")
    pure = float(((big["long_share"] > 0.9) | (big["long_share"] < 0.1)).mean())
    print(f"    2 ラウンド以上の連鎖 {big.height} 本のうち、9 割以上が同じ向き {pure*100:.1f}%")

    # 自己励起: 次の刻みが続く確率(ラウンドの大きさ別)と、時間内一様の帰無
    cont = (gu <= TICK).astype(float)
    nn = rnd["n"].to_numpy()[:-1]
    CLS = [(1, 1, "1 件"), (2, 4, "2〜4 件"), (5, 19, "5〜19 件"), (20, 10 ** 9, "20 件以上")]
    print("    ラウンドの大きさ別「次の刻み(3.5 秒以内)も清算が続く」確率:")
    self_ex = []
    for lo_, hi_, nm in CLS:
        m = (nn >= lo_) & (nn <= hi_)
        if m.sum() > 20:
            print(f"      {nm:<10}: {cont[m].mean()*100:5.1f}%  n={m.sum():,}")
            self_ex.append({"class": nm, "n": int(m.sum()), "p_cont": float(cont[m].mean())})
    pl.DataFrame(self_ex).write_csv(D / f"liqself_{tag}.csv")
    hb = (rt // (3600 * NS)).astype(np.int64)
    uq, inv = np.unique(hb, return_inverse=True)
    base = uq * 3600.0
    nulc = np.array([(np.diff(np.sort(base[inv] + rng.random(rt.size) * 3600.0)) <= TICK).mean()
                     for _ in range(a.nnull)])
    print(f"    続く確率 実測 {cont.mean()*100:.1f}% / "
          f"時間内一様の帰無 {nulc.mean()*100:.2f}% "
          f"[{np.percentile(nulc,5)*100:.2f}, {np.percentile(nulc,95)*100:.2f}] "
          f"→ 比 {cont.mean()/nulc.mean():.0f} 倍")

    # ---------- D. 口座の集中 -------------------------------------------
    acc = (ev.group_by("luid").agg(pl.col("notional").sum().alias("usd"),
                                   pl.len().alias("n"),
                                   pl.col("closedPnl").sum().alias("pnl"),
                                   pl.col("dt").n_unique().alias("days"),
                                   (pl.col("lside") == "long").mean().alias("long_share"))
           .sort("usd", descending=True))
    acc.write_csv(D / f"liqacct_{tag}.csv")
    v = np.sort(acc["usd"].to_numpy())[::-1]
    p = v / v.sum()
    gini = 1 - 2 * np.trapezoid(np.cumsum(np.sort(v)) / v.sum(),
                                np.arange(1, v.size + 1) / v.size)
    print(f"\n[D] 清算された口座 {acc.height:,}  上位 1 {p[:1].sum()*100:.1f}% / "
          f"上位 10 {p[:10].sum()*100:.1f}% / 上位 100 {p[:100].sum()*100:.1f}%")
    print(f"    HHI {(p**2).sum():.4f}(実効 {1/(p**2).sum():.0f} 口座) / ジニ {gini:.3f}")
    print(f"    1 回だけ清算された口座 {int((acc['n']==1).sum()):,} "
          f"({(acc['n']==1).mean()*100:.1f}%) / 10 回以上 {int((acc['n']>=10).sum()):,}")
    cp = (ev.group_by("cuid").agg(pl.col("notional").sum().alias("usd"), pl.len().alias("n"))
          .sort("usd", descending=True))
    w = cp["usd"].to_numpy() / cp["usd"].sum()
    print(f"    相手方 {cp.height:,}  上位 1 {w[:1].sum()*100:.1f}% / "
          f"上位 10 {w[:10].sum()*100:.1f}%  HHI {(w**2).sum():.4f}")

    # ---------- E. 時刻帯と日次 ------------------------------------------
    fh = (f.with_columns(h=pl.col("ts").dt.hour(), wd=pl.col("ts").dt.weekday())
          .group_by("h").agg((pl.col("px") * pl.col("sz")).sum().alias("vol_usd"),
                             pl.len().alias("n_tr")).sort("h"))
    eh = (ev.with_columns(h=pl.col("ts").dt.hour())
          .group_by("h").agg(pl.col("notional").sum().alias("liq_usd"),
                             pl.len().alias("n")).sort("h"))
    hour = (fh.join(eh, on="h", how="left")
            .with_columns(pl.col("liq_usd").fill_null(0), pl.col("n").fill_null(0))
            .with_columns(liq_per_vol_bp=pl.col("liq_usd") / pl.col("vol_usd") * 1e4))
    hour.write_csv(D / f"liqhour_{tag}.csv")
    print("\n[E] 時刻帯(UTC)の清算強度 上位")
    print(hour.sort("liq_per_vol_bp", descending=True).head(4))
    print(hour.sort("liq_per_vol_bp").head(3))

    oi = pl.read_parquet(D / f"oi_series_{tag}.parquet").sort("ts")
    oid = (oi.with_columns(dt=pl.col("ts").dt.strftime("%Y-%m-%d"))
           .group_by("dt").agg(pl.col("oi").mean().alias("oi_mean"),
                               pl.col("oi").last().alias("oi_close")).sort("dt"))
    fd = (f.with_columns(dt=pl.col("ts").dt.strftime("%Y-%m-%d"))
          .group_by("dt").agg((pl.col("px") * pl.col("sz")).sum().alias("vol_usd"),
                              pl.col("px").max().alias("hi"), pl.col("px").min().alias("lo"),
                              pl.col("px").first().alias("o"), pl.col("px").last().alias("c"))
          .sort("dt"))
    ed = (ev.group_by("dt").agg(pl.col("notional").sum().alias("liq_usd"),
                                pl.col("sz").sum().alias("liq_sz"),
                                pl.col("oi_cut").sum().alias("oi_cut_sz"),
                                pl.len().alias("n"),
                                pl.col("luid").n_unique().alias("n_acct"),
                                (pl.col("lside") == "long").sum().alias("n_long")))
    day = (fd.join(ed, on="dt", how="left").join(oid, on="dt", how="left")
           .with_columns(pl.col(["liq_usd", "liq_sz", "oi_cut_sz", "n", "n_acct",
                                 "n_long"]).fill_null(0))
           .with_columns(ret_pct=(pl.col("c") / pl.col("o") - 1) * 100,
                         range_pct=(pl.col("hi") / pl.col("lo") - 1) * 100,
                         liq_over_vol_bp=pl.col("liq_usd") / pl.col("vol_usd") * 1e4,
                         liq_over_oi_pct=pl.col("liq_sz") / pl.col("oi_mean") * 100,
                         oicut_over_oi_pct=pl.col("oi_cut_sz") / pl.col("oi_mean") * 100))
    day.write_csv(D / f"liqday_{tag}.csv")
    d = day.filter(pl.col("liq_usd") > 0)
    print(f"\n[F] 清算のあった日 {d.height} / {day.height}")
    print(f"    日次 清算/出来高 中央値 {float(d['liq_over_vol_bp'].median()):.1f} bp / "
          f"最大 {float(day['liq_over_vol_bp'].max()):.1f} bp")
    print(f"    日次 清算枚数/平均 OI 中央値 {float(d['liq_over_oi_pct'].median()):.2f}% / "
          f"最大 {float(day['liq_over_oi_pct'].max()):.2f}%")
    print(f"    OI を実際に減らした分 中央値 {float(d['oicut_over_oi_pct'].median()):.2f}% / "
          f"最大 {float(day['oicut_over_oi_pct'].max()):.2f}%")
    x = np.log10(day["liq_usd"].to_numpy() + 1)
    print(f"    log(日次清算額) と 日中値幅 の相関 "
          f"{np.corrcoef(x, day['range_pct'].to_numpy())[0,1]:.3f} / "
          f"日次リターン との相関 {np.corrcoef(x, day['ret_pct'].to_numpy())[0,1]:.3f}")

    # ---------- G. 価格帯は「初回の訪問」で片付くのか ---------------------
    # 1 つの価格帯に価格が入るたびを「訪問」と数え(1 時間以上離れたら別の訪問)、
    # 何回目の訪問かで清算の強さが変わるかを見る。溜まっていた建玉が最初の
    # 訪問で一掃されるなら、初回だけ強度が高いはず。
    vis_rows = []
    et = ts
    ebk = eb
    for b in range(edges.size):
        m = idx == b
        if m.sum() < 50:
            continue
        tb_ = ft[m]
        dw = dwell[m]
        vv = fv[m]
        vb = np.r_[0, np.cumsum(np.diff(tb_) > 3600 * NS)]
        for v in range(vb.max() + 1):
            k = vb == v
            lo_t, hi_t = tb_[k][0], tb_[k][-1]
            lm = (ebk == b) & (et >= lo_t) & (et <= hi_t)
            vis_rows.append({"bucket": float(edges[b]), "visit": v + 1,
                             "dwell_h": float(dw[k].sum() / 3600),
                             "vol_usd": float(vv[k].sum()),
                             "liq_usd": float(usd[lm].sum()), "n_liq": int(lm.sum())})
    vis = pl.DataFrame(vis_rows)
    vis.write_csv(D / f"liqvisit_{tag}.csv")
    CLSV = [("1 回目", 1, 1), ("2 回目", 2, 2), ("3〜5 回目", 3, 5),
            ("6 回目以降", 6, 10 ** 9)]
    vs = vis.to_dict(as_series=False)
    vnum = np.array(vs["visit"])
    vbk = np.array(vs["bucket"])
    vdw = np.array(vs["dwell_h"])
    vvol = np.array(vs["vol_usd"])
    vliq = np.array(vs["liq_usd"])
    ub = np.unique(vbk)
    print("\n[G] 価格帯への「何回目の訪問」か別の清算強度"
          "(括弧は価格帯を入れ替えた 1000 回の bootstrap 95% 区間)")
    for nm, lo_, hi_ in CLSV:
        k = (vnum >= lo_) & (vnum <= hi_)
        ph = vliq[k].sum() / vdw[k].sum()
        pv = vliq[k].sum() / vvol[k].sum() * 1e4
        bs = np.empty((1000, 2))
        for i in range(1000):
            pick = rng.choice(ub, ub.size, replace=True)
            sel = np.concatenate([np.flatnonzero(k & (vbk == b)) for b in pick])
            bs[i] = (vliq[sel].sum() / max(vdw[sel].sum(), 1e-9),
                     vliq[sel].sum() / max(vvol[sel].sum(), 1e-9) * 1e4)
        print(f"    {nm:<10} 訪問 {int(k.sum()):>5} / 滞在 {vdw[k].sum():8.1f} h / "
              f"清算 ${vliq[k].sum()/1e6:6.2f}M  "
              f"1 時間あたり ${ph/1e3:6.1f}k "
              f"[{np.percentile(bs[:,0],2.5)/1e3:.1f}, {np.percentile(bs[:,0],97.5)/1e3:.1f}] / "
              f"出来高比 {pv:5.1f} bp "
              f"[{np.percentile(bs[:,1],2.5):.1f}, {np.percentile(bs[:,1],97.5):.1f}]")
    print(f"    訪問 {vis.height:,} 回 / 価格帯 {vis['bucket'].n_unique()}")


if __name__ == "__main__":
    main()
