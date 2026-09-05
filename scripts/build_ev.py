"""約定時刻 tau から測った逆選択 AS(S,D) と、クオートの期待値 EV(S,D)。

    uv run python scripts/build_ev.py --coin xyz:MU

なぜ作り直したか
----------------
これまでの損益は **発注時刻 T から T+h** までの mid の動きで測っていた。
だが T から約定時刻 tau までの区間は**まだ何も持っていない**ので損益に入らない。
正しい逆選択は

    AS(S,D) = E[ r_{tau -> tau+h} | Fill, S, D ]

である。これを使って、クオートを出すべきセルを直接判定する:

    EV_bid(S,D) = P(Fill_bid | S,D) * [ Spread/2 - Fee + AS_bid(S,D) ]
    EV_ask(S,D) = P(Fill_ask | S,D) * [ Spread/2 - Fee - AS_ask(S,D) ]

AS は**符号をつけない** mid のリターン。買い指値なら価格が上がれば得なので +、
売り指値なら逆なので -。AS_bid と AS_ask は条件づける事象が違うので別の量である。

Spread/2 を 2 通りで出す
------------------------
1. `hs`   = 約定時刻の半スプレッド spread(tau)/2。上の式のとおりの解釈
2. `edge` = mid(tau) - 約定値段。**実際に取れた瞬間の優位**
両者の差は「発注してから約定するまでに mid が動いた分」である。
自分の値段が置き去りになっていれば edge は hs より小さくなる。
どちらが実態かは 2 つ並べて初めて判る。

時間契約
--------
- 発注の判断に使う特徴量も十分位の境目も、すべて T 以前の情報だけで決まる
  (境目は前日の分布)
- 約定判定は (T + 5ms, T+h] のテイカー約定のみ
- AS は tau から tau+h までで、tau+h が当日内に収まるものだけを使う
- mid(tau) と mid(tau+h) はそれぞれの時刻**以前**の最後の気配 (後ろ向き asof)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_maker import MAKER_FEE_BP, day_fill_times  # noqa: E402
from build_pred import GRID_N, day_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HORIZONS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ("obi", "ofi_10s", "ai_net_10s")
SIDES = (1, -1)
NB = 10
RATE_W = 10
DAY_NS = 86_400_000_000_000
DAILY_H = (0.1, 1.0, 10.0)
# ★ 約定時刻ちょうどの板は「もう約定した後」である。node_fills の時刻は
# ミリ秒精度で、同じブロックの板更新が tau 以前に入るため、tau で引くと
# 自分の指値が消えた後の板を見てしまう。実測 (2026-06-24・1 秒以内の約定):
#   tau-1ms : mid-指値 +0.465bp / スプレッド 0.969bp / 自分が最良 76.6%
#   tau     : mid-指値 -0.000bp / スプレッド 2.974bp / 自分が最良 12.1%
# -1ms から -50ms までは値が動かない平坦域なので、2ms 手前を約定直前とみなす。
PRE_NS = 2_000_000
STATS = ("n", "n_fill", "sum_as", "sum_as2", "sum_hs", "sum_hs_post",
         "sum_imp", "sum_edge", "sum_sprT", "sum_wait")


def edges_ok(e) -> bool:
    return e is not None and bool(np.all(np.diff(e) > 0))


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

    nf, ns, nh = len(FEATS), len(SIDES), len(HORIZONS)
    acc = {k: np.zeros((nf, ns, NB, NB, nh)) for k in STATS}
    drows = []
    prev_s: dict[str, np.ndarray] = {}
    prev_t: dict[int, np.ndarray] = {}
    n_drop = 0
    n_tot = 0
    for kk, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        fd = FID.get(dt)
        D = day_features(d, fd)
        if D is None:
            continue
        F = D["F"]
        bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
        mid = 0.5 * (bid + ask)
        sprT = (ask - bid) / mid * 1e4
        tg = D["tg"]
        d0 = int(tg[0])
        day_end = d0 + DAY_NS
        idx = np.arange(GRID_N + 1)

        # 生の気配 (tau は格子点に乗らないので、格子ではなく気配そのものを引く)
        bts = d["ts"].cast(pl.Int64).to_numpy()
        bmid = 0.5 * (d["best_bid"].to_numpy() + d["best_ask"].to_numpy())
        bspr = ((d["best_ask"].to_numpy() - d["best_bid"].to_numpy())
                / bmid * 1e4)

        TAU = day_fill_times(D, fd)

        # 待ち行列がはける推定時間 (十分位の軸)
        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fs = fd["sz"].to_numpy()
        isb = fd["side"].to_numpy() == "B"
        sec = np.clip((ft - d0) // 10 ** 9, 0, GRID_N).astype(np.int64)
        vsell = np.zeros(GRID_N + 1)
        vbuy = np.zeros(GRID_N + 1)
        np.add.at(vsell, sec[~isb], fs[~isb])
        np.add.at(vbuy, sec[isb], fs[isb])
        cs_ = np.concatenate([[0.0], np.cumsum(vsell)])
        cb_ = np.concatenate([[0.0], np.cumsum(vbuy)])
        lo10 = np.maximum(idx - RATE_W, 0)
        rate = {1: (cs_[idx] - cs_[lo10]) / RATE_W,
                -1: (cb_[idx] - cb_[lo10]) / RATE_W}
        QA = {1: qb, -1: qa}
        PXQ = {1: bid, -1: ask}
        TAUQ = {sd: np.where(rate[sd] > 0, QA[sd] / np.maximum(rate[sd], 1e-12),
                             np.inf) for sd in SIDES}

        ok_s = {f: edges_ok(prev_s.get(f)) for f in FEATS}
        ok_t = {sd: edges_ok(prev_t.get(sd)) for sd in SIDES}

        # 約定時刻まわりの量を側ごとに 1 度だけ作る
        PRE = {}
        for sd in SIDES:
            tau = TAU[sd]
            has = tau >= 0
            safe = np.where(has, tau, bts[0])
            # 約定直前の板 (これが「取れた優位」と Spread/2 の基準)
            ip = np.clip(np.searchsorted(bts, safe - PRE_NS, side="right") - 1,
                         0, bts.size - 1)
            m_pre = np.where(has, bmid[ip], np.nan)
            s_pre = np.where(has, bspr[ip], np.nan)
            edge = (m_pre - PXQ[sd]) * sd / m_pre * 1e4
            # 約定直後の板 (診断用。ここを基準にすると即時の値動きが AS から抜ける)
            it = np.clip(np.searchsorted(bts, safe, side="right") - 1,
                         0, bts.size - 1)
            s_post = np.where(has, bspr[it], np.nan)
            imp = np.where(has, np.log(bmid[it] / m_pre) * 1e4, np.nan)
            AS = {}
            for h in HORIZONS:
                te = tau + int(h * 1e9)
                ih = np.clip(np.searchsorted(bts, np.where(has, te, bts[0]),
                                             side="right") - 1, 0, bts.size - 1)
                inday = has & (te <= day_end)
                AS[h] = np.where(inday, np.log(bmid[ih] / m_pre) * 1e4, np.nan)
            PRE[sd] = (has, tau, m_pre, s_pre, edge, AS, s_post, imp)
            n_tot += int(has.sum())
            n_drop += int((has & (tau + int(60 * 1e9) > day_end)).sum())

        for fi_, f in enumerate(FEATS):
            if not ok_s[f]:
                continue
            sdec = np.searchsorted(prev_s[f], F[f], side="right")
            finf = np.isfinite(F[f])
            for si, sd in enumerate(SIDES):
                if not ok_t[sd]:
                    continue
                has, tau, m_pre, s_pre, edge, AS, s_post, imp = PRE[sd]
                tdec = np.searchsorted(prev_t[sd], TAUQ[sd], side="right")
                base = finf & np.isfinite(mid) & (QA[sd] > 0)
                cell = sdec * NB + tdec
                for hi, h in enumerate(HORIZONS):
                    fill = base & has & (tau <= tg + int(h * 1e9)) \
                        & np.isfinite(AS[h])
                    add = {"n": (base, np.ones(GRID_N + 1)),
                           "n_fill": (fill, np.ones(GRID_N + 1)),
                           "sum_as": (fill, AS[h]),
                           "sum_as2": (fill, AS[h] ** 2),
                           "sum_hs": (fill, s_pre / 2.0),
                           "sum_hs_post": (fill, s_post / 2.0),
                           "sum_imp": (fill, imp),
                           "sum_edge": (fill, edge),
                           "sum_sprT": (base, sprT / 2.0),
                           "sum_wait": (fill, (tau - tg) / 1e9)}
                    flat = {}
                    for key, (m, val) in add.items():
                        t = np.bincount(cell[m], weights=np.nan_to_num(val[m]),
                                        minlength=NB * NB)[: NB * NB]
                        acc[key][fi_, si, :, :, hi] += t.reshape(NB, NB)
                        flat[key] = t
                    if h in DAILY_H:
                        for c in np.flatnonzero(flat["n"] > 0):
                            drows.append({
                                "dt": dt, "feat": f, "side": sd, "h": h,
                                "s_dec": int(c // NB), "q_dec": int(c % NB),
                                **{key: float(flat[key][c]) for key in STATS}})
        prev_s = {f: np.nanquantile(F[f][np.isfinite(F[f])],
                                    np.arange(1, NB) / NB)
                  for f in FEATS if np.isfinite(F[f]).any()}
        prev_t = {}
        for sd in SIDES:
            v = TAUQ[sd][np.isfinite(TAUQ[sd])]
            if v.size:
                prev_t[sd] = np.quantile(v, np.arange(1, NB) / NB)
        if (kk + 1) % 20 == 0 or kk == 0:
            print(f"  [{kk+1}/{len(days)}] {dt}", flush=True)

    rows = []
    for fi_, f in enumerate(FEATS):
        for si, sd in enumerate(SIDES):
            for s in range(NB):
                for dd in range(NB):
                    for hi, h in enumerate(HORIZONS):
                        if acc["n"][fi_, si, s, dd, hi] == 0:
                            continue
                        rows.append({"feat": f, "side": sd, "h": h,
                                     "s_dec": s, "q_dec": dd,
                                     **{key: float(acc[key][fi_, si, s, dd, hi])
                                        for key in STATS}})
    pl.DataFrame(rows).write_csv(DATA / f"ev_pooled_{tag}.csv")
    pl.DataFrame(drows).write_parquet(DATA / f"ev_daily_{tag}.parquet")
    print(f"書き出し ev_pooled({len(rows):,} セル) / ev_daily({len(drows):,} 行)")
    print(f"約定 {n_tot:,} 件のうち tau+60s が翌日に食い込むもの {n_drop:,} 件 "
          f"({100*n_drop/max(n_tot,1):.3f}%) は該当ホライズンで除外")


if __name__ == "__main__":
    main()
