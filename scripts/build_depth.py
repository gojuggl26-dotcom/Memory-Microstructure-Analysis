"""BBO から 10 ティック奥までの各水準で P(Fill) x PnL_fill を測る。

    uv run python scripts/build_depth.py --coin xyz:MU

置き方
------
時刻 T に、買い指値を **最良買い気配 − k ティック** (k = 0..10) に置く。
売り指値は **最良売り気配 + k ティック**。ティックは価格で変わる
(< 1000 なら 0.01、>= 1000 なら 0.1)。

約定の判定 — 待ち行列の先頭にいる前提(上限側)
---------------------------------------------
最良気配より奥の水準については板の厚みを持っていないので、
**自分がその水準の待ち行列の先頭にいる**とみなす。すなわち

    自分の値段以下で約定したテイカー売りが 1 件でも来たら約定

とする。これは k >= 1 では**それほど乱暴ではない**。価格がまだ来ていない
水準に先回りして置くのだから、後から来る人より前に並ぶのが自然である。
一方 k = 0 (最良気配) では明らかに甘い。そこで k=0 については
[メイカーの反実仮想](../reports/) で使った実際の待ち行列
(Q_ahead = bid_sz) の版と並べ、この仮定がどれだけ甘いかを較正する。

損益
----
    edge   = mid(tau - 2ms) - 約定値段          (奥へ置くほど k ティック増える)
    AS     = r_{tau -> tau+h}                   (約定時刻から測る)
    PnL_fill = edge - Fee + (買いなら +AS / 売りなら -AS)
    EV     = P(Fill) x PnL_fill

mid と スプレッドを **約定 2ms 前**で引くのは、約定時刻ちょうどの板が
もう約定後になっているため (ev の報告 1 節)。

時間契約
--------
- 発注値段も十分位も T 時点の情報だけで決まる
- 約定判定は (T + 5ms, T+h] のテイカー約定のみ
- tau は 60 秒以内の最初の到達を 1 度だけ求め、各ホライズンは
  tau <= T+h で導出する (判定がホライズン間で矛盾しない)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import PX_UNIT, clean_bbo  # noqa: E402
from build_maker import MAKER_FEE_BP  # noqa: E402
from build_pred import FILL_GUARD_NS, GRID_N, day_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HORIZONS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ("obi", "ofi_10s", "ai_net_10s")
SIDES = (1, -1)
KMAX = 10
NS_ = 10                    # 信号の十分位
MAXW = 60                   # tau を探す最大秒数
PRE_NS = 2_000_000
DAY_NS = 86_400_000_000_000
STATS = ("n", "n_fill", "sum_edge", "sum_as", "sum_pnl", "sum_wait")


def first_reach(tg, bts, ft, fp, msk, thr, sgn, d0):
    """各格子点・各水準について、値段に到達した最初の約定の時刻を返す。

    sgn=+1 (買い指値) なら「約定値段 <= 閾値」、-1 (売り指値) なら「>=」。
    戻り値 tau[k] は ns、到達しなければ -1。
    """
    n = tg.size
    inf = np.inf if sgn > 0 else -np.inf
    sec = np.clip((ft - d0) // 10 ** 9, 0, GRID_N).astype(np.int64)
    pex = np.where(msk, fp.astype(np.float64), inf)
    pmin = np.full(GRID_N + 1, inf)
    if sgn > 0:
        np.minimum.at(pmin, sec, pex)
    else:
        pmin = np.full(GRID_N + 1, -np.inf)
        np.maximum.at(pmin, sec, np.where(msk, fp.astype(np.float64), -np.inf))

    # 最初の 1 秒だけは 5ms の余裕を入れて 1 件ずつ見る
    lo0 = np.searchsorted(ft, tg + FILL_GUARD_NS, side="right")
    hi0 = np.searchsorted(ft, tg + 10 ** 9, side="right")
    c0 = np.maximum(hi0 - lo0, 0)
    t0v = int(c0.sum())
    p0 = np.full(n, inf)
    if t0v:
        g = np.repeat(np.arange(n), c0)
        off = np.arange(t0v) - np.repeat(np.cumsum(c0) - c0, c0)
        fi = np.repeat(lo0, c0) + off
        v = np.where(msk[fi], fp[fi].astype(np.float64), inf)
        if sgn > 0:
            np.minimum.at(p0, g, v)
        else:
            p0 = np.full(n, -np.inf)
            np.maximum.at(p0, g, np.where(msk[fi], fp[fi].astype(np.float64),
                                          -np.inf))
    idx = np.arange(n)
    dstar = np.full((KMAX + 1, n), -1, dtype=np.int64)
    run = p0.copy()
    tick = np.where(thr * PX_UNIT >= 1000.0, 10.0, 1.0)
    for d in range(0, MAXW):
        if d > 0:
            j = np.minimum(idx + d, GRID_N)
            run = np.minimum(run, pmin[j]) if sgn > 0 else np.maximum(run, pmin[j])
        with np.errstate(invalid="ignore"):
            reach = np.floor((thr - run) / tick) if sgn > 0 \
                else np.floor((run - thr) / tick)
        reach = np.where(np.isfinite(reach), reach, -1)
        for k in range(KMAX + 1):
            m = (dstar[k] < 0) & (reach >= k)
            dstar[k, m] = d
    # 交差した秒の中を 1 件ずつたどって正確な時刻を取る
    tau = np.full((KMAX + 1, n), -1, dtype=np.int64)
    for k in range(KMAX + 1):
        kk = np.flatnonzero(dstar[k] >= 0)
        if kk.size == 0:
            continue
        st = np.where(dstar[k][kk] == 0, tg[kk] + FILL_GUARD_NS,
                      d0 + (idx[kk] + dstar[k][kk]) * 10 ** 9)
        en = np.where(dstar[k][kk] == 0, tg[kk] + 10 ** 9,
                      d0 + (idx[kk] + dstar[k][kk] + 1) * 10 ** 9)
        lo = np.searchsorted(ft, st, side="right")
        hi = np.searchsorted(ft, en, side="right")
        cnt = np.maximum(hi - lo, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        g = np.repeat(np.arange(kk.size), cnt)
        off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        fi = np.repeat(lo, cnt) + off
        lim = thr[kk][g] - sgn * k * tick[kk][g]
        ok = msk[fi] & ((fp[fi] <= lim + 1e-9) if sgn > 0
                        else (fp[fi] >= lim - 1e-9))
        oi = np.flatnonzero(ok)
        if oi.size == 0:
            continue
        gg = g[oi]
        sel = np.ones(oi.size, dtype=bool)
        sel[1:] = gg[1:] != gg[:-1]
        tau[k, kk[gg[sel]]] = ft[fi[oi[sel]]]
    return tau, tick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 行を除外)", flush=True)
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "px", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    days = sorted(bb["dt"].unique().to_list())

    nk, nh, nsd = KMAX + 1, len(HORIZONS), len(SIDES)
    acc = {k: np.zeros((nk, nsd, nh)) for k in STATS}
    acs = {k: np.zeros((len(FEATS), nk, nsd, NS_, nh)) for k in STATS}
    drows = []
    prev_s: dict[str, np.ndarray] = {}
    for kd, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        fd = FID.get(dt)
        D = day_features(d, fd)
        if D is None:
            continue
        F = D["F"]
        bid, ask = D["bid"], D["ask"]
        mid = 0.5 * (bid + ask)
        tg = D["tg"]
        d0 = int(tg[0])
        day_end = d0 + DAY_NS
        bts = d["ts"].cast(pl.Int64).to_numpy()
        bmid = 0.5 * (d["best_bid"].to_numpy() + d["best_ask"].to_numpy())
        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fp = np.round(fd["px"].to_numpy() / PX_UNIT)
        isb = fd["side"].to_numpy() == "B"
        bt = np.round(bid / PX_UNIT)
        at = np.round(ask / PX_UNIT)

        TAU = {}
        TICK = {}
        TAU[1], TICK[1] = first_reach(tg, bts, ft, fp, ~isb, bt, 1, d0)
        TAU[-1], TICK[-1] = first_reach(tg, bts, ft, fp, isb, at, -1, d0)

        ok_s = {f: (prev_s.get(f) is not None
                    and bool(np.all(np.diff(prev_s[f]) > 0))) for f in FEATS}
        for si, sd in enumerate(SIDES):
            thr = bt if sd == 1 else at
            tick = TICK[sd]
            for k in range(KMAX + 1):
                tau = TAU[sd][k]
                has = tau >= 0
                pq = (thr - sd * k * tick) * PX_UNIT
                safe = np.where(has, tau, bts[0])
                ip = np.clip(np.searchsorted(bts, safe - PRE_NS, side="right") - 1,
                             0, bts.size - 1)
                m_pre = np.where(has, bmid[ip], np.nan)
                edge = (m_pre - pq) * sd / m_pre * 1e4
                for hi, h in enumerate(HORIZONS):
                    te = tau + int(h * 1e9)
                    ih = np.clip(np.searchsorted(bts, np.where(has, te, bts[0]),
                                                 side="right") - 1,
                                 0, bts.size - 1)
                    AS = np.where(has & (te <= day_end),
                                  np.log(bmid[ih] / m_pre) * 1e4, np.nan)
                    fill = has & (tau <= tg + int(h * 1e9)) & np.isfinite(AS) \
                        & np.isfinite(mid)
                    pnl = edge - MAKER_FEE_BP + sd * AS
                    base = np.isfinite(mid)
                    vals = {"n": (base, np.ones(GRID_N + 1)),
                            "n_fill": (fill, np.ones(GRID_N + 1)),
                            "sum_edge": (fill, edge),
                            "sum_as": (fill, AS),
                            "sum_pnl": (fill, pnl),
                            "sum_wait": (fill, (tau - tg) / 1e9)}
                    for key, (m, v) in vals.items():
                        acc[key][k, si, hi] += float(np.nansum(v[m]))
                    for fj, f in enumerate(FEATS):
                        if not ok_s[f]:
                            continue
                        sdec = np.searchsorted(prev_s[f], F[f], side="right")
                        fin = np.isfinite(F[f])
                        for key, (m, v) in vals.items():
                            t = np.bincount(sdec[m & fin],
                                            weights=np.nan_to_num(v[m & fin]),
                                            minlength=NS_)[:NS_]
                            acs[key][fj, k, si, :, hi] += t
                    if h in (0.1, 0.5, 1.0, 10.0):
                        drows.append({"dt": dt, "k": k, "side": sd, "h": h,
                                      **{key: float(np.nansum(v[m]))
                                         for key, (m, v) in vals.items()}})
        prev_s = {f: np.nanquantile(F[f][np.isfinite(F[f])],
                                    np.arange(1, NS_) / NS_)
                  for f in FEATS if np.isfinite(F[f]).any()}
        if (kd + 1) % 20 == 0 or kd == 0:
            print(f"  [{kd+1}/{len(days)}] {dt}", flush=True)

    rows = []
    for k in range(nk):
        for si, sd in enumerate(SIDES):
            for hi, h in enumerate(HORIZONS):
                rows.append({"k": k, "side": sd, "h": h,
                             **{key: float(acc[key][k, si, hi]) for key in STATS}})
    pl.DataFrame(rows).write_csv(DATA / f"depth_pooled_{tag}.csv")
    rows = []
    for fj, f in enumerate(FEATS):
        for k in range(nk):
            for si, sd in enumerate(SIDES):
                for s in range(NS_):
                    for hi, h in enumerate(HORIZONS):
                        if acs["n"][fj, k, si, s, hi] == 0:
                            continue
                        rows.append({"feat": f, "k": k, "side": sd, "s_dec": s,
                                     "h": h,
                                     **{key: float(acs[key][fj, k, si, s, hi])
                                        for key in STATS}})
    pl.DataFrame(rows).write_csv(DATA / f"depth_signal_{tag}.csv")
    pl.DataFrame(drows).write_parquet(DATA / f"depth_daily_{tag}.parquet")
    print(f"書き出し depth_pooled / depth_signal({len(rows):,}) / "
          f"depth_daily({len(drows):,})")


if __name__ == "__main__":
    main()
