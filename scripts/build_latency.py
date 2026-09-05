"""約定の 500/250/100/50/20/10 ms 前の特徴量を貯め、標本外 AUC を測れるようにする。

    uv run python scripts/build_latency.py --coin xyz:MU

目的
----
「約定の delta ミリ秒前に、その約定が toxic になると判るか」を delta を
詰めながら測る。判るようになる delta が、実際に取り消せる遅延より短ければ
その情報は使えない。だから AUC の横に**実際の取り消し往復遅延**を置く。

貯めるもの
----------
約定 1 件につき、delta in {500, 250, 100, 50, 20, 10} ms 前の

    obi        板の不均衡 (自分側が厚いと +)
    depth      自分側の最良気配の数量
    spread     スプレッド (bp)
    ofi_1s     直前 1 秒の注文流不均衡 (自分側に有利なら +)
    ofi_100ms  直前 100 ミリ秒の同じもの
    aggr_1s    直前 1 秒の符号つきテイカー数量 (自分側に有利なら +)
    aggr_100ms 直前 100 ミリ秒の同じもの
    qdepl      発注時の待ち行列のうち食われた割合
    drift      発注時 T からその時点までの mid の動き (自分側に有利なら +)
    age        発注してからの経過秒

目的変数は「約定後 10 秒の markout(損益の向き)が**その日の下位 20%**」。

時間契約
--------
- 説明変数はすべて tau - delta 以前の情報だけで作る
- 目的変数は tau 以降の値動きで、説明変数と重ならない
- 標本外は**日で前後に切る**(前 60% で学習、後 40% で評価)。
  標準化も回帰係数もすべて学習期間だけで推定する
- 1 日あたりの標本数を揃えるため無作為に間引く(種を固定)。
  間引きは目的変数と無関係に行う
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_maker import day_fill_times  # noqa: E402
from build_pred import GRID_N, day_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H_FILL = 10.0
H_HOLD = 10.0
DELTAS_MS = [500, 250, 100, 50, 20, 10]
SIDES = (1, -1)
NSAMP = 3000             # 1 日 1 側あたりの標本上限
TOX_Q = 0.2              # 下位 20% を toxic とする
PRE_NS = 2_000_000
DAY_NS = 86_400_000_000_000
FEATS = ("obi", "depth", "spread", "ofi_1s", "ofi_100ms", "aggr_1s",
         "aggr_100ms", "qdepl", "drift", "age")


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

    out = []
    for kd, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        fd = FID.get(dt)
        D = day_features(d, fd)
        if D is None:
            continue
        bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
        mid = 0.5 * (bid + ask)
        tg = D["tg"]
        d0 = int(tg[0])
        day_end = d0 + DAY_NS
        TAU = day_fill_times(D, fd)

        bts = d["ts"].cast(pl.Int64).to_numpy()
        pbb = d["best_bid"].to_numpy()
        paa = d["best_ask"].to_numpy()
        qbb = d["bid_sz"].to_numpy()
        qaa = d["ask_sz"].to_numpy()
        bmid = 0.5 * (pbb + paa)
        bspr = (paa - pbb) / bmid * 1e4
        tt = qbb + qaa
        bobi = np.where(tt > 0, (qbb - qaa) / np.where(tt > 0, tt, 1.0), 0.0)
        e = np.concatenate([[0.0], (
            (pbb[1:] >= pbb[:-1]) * qbb[1:] - (pbb[1:] <= pbb[:-1]) * qbb[:-1]
            - ((paa[1:] <= paa[:-1]) * qaa[1:] - (paa[1:] >= paa[:-1]) * qaa[:-1]))])
        cofi = np.cumsum(e)

        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fs = fd["sz"].to_numpy()
        isb = fd["side"].to_numpy() == "B"
        cnet = np.concatenate([[0.0], np.cumsum(fs * np.where(isb, 1.0, -1.0))])
        csell = np.concatenate([[0.0], np.cumsum(np.where(~isb, fs, 0.0))])
        cbuy = np.concatenate([[0.0], np.cumsum(np.where(isb, fs, 0.0))])

        def at(t):
            return np.clip(np.searchsorted(bts, t, side="right") - 1,
                           0, bts.size - 1)

        rng = np.random.default_rng(abs(hash(dt)) % (2 ** 31))
        for sd in SIDES:
            tau = TAU[sd]
            Q0 = qb if sd == 1 else qa
            ok = (tau >= 0) & (tau <= tg + int(H_FILL * 1e9)) & np.isfinite(mid) \
                & (Q0 > 0) & (tau + int(H_HOLD * 1e9) <= day_end)
            idx = np.flatnonzero(ok)
            if idx.size < 500:
                continue
            tv = tau[idx]
            m_T = mid[idx]
            ip = at(tv - PRE_NS)
            m_pre = bmid[ip]
            ih = at(tv + int(H_HOLD * 1e9))
            mk = sd * np.log(bmid[ih] / m_pre) * 1e4
            thr = np.quantile(mk, TOX_Q)
            y = (mk <= thr).astype(np.int8)
            # 目的変数と無関係に間引く
            if idx.size > NSAMP:
                pick = rng.choice(idx.size, NSAMP, replace=False)
            else:
                pick = np.arange(idx.size)
            idx, tv, m_T, y, mk = idx[pick], tv[pick], m_T[pick], y[pick], mk[pick]
            k0 = np.searchsorted(ft, tg[idx], side="right")
            cc = csell if sd == 1 else cbuy
            for dms in DELTAS_MS:
                tp = tv - dms * 1_000_000
                ib = at(tp)
                i1s = at(tp - 10 ** 9)
                i100 = at(tp - 100_000_000)
                j1 = np.searchsorted(ft, tp, side="right")
                j1s = np.searchsorted(ft, tp - 10 ** 9, side="right")
                j100 = np.searchsorted(ft, tp - 100_000_000, side="right")
                rows = {
                    "dt": dt, "side": sd, "delta_ms": dms,
                    "y": y, "markout": mk,
                    "obi": sd * bobi[ib],
                    "depth": (qbb if sd == 1 else qaa)[ib],
                    "spread": bspr[ib],
                    "ofi_1s": sd * (cofi[ib] - cofi[i1s]),
                    "ofi_100ms": sd * (cofi[ib] - cofi[i100]),
                    "aggr_1s": sd * (cnet[j1] - cnet[j1s]),
                    "aggr_100ms": sd * (cnet[j1] - cnet[j100]),
                    "qdepl": np.clip((cc[j1] - cc[k0]) / Q0[idx], 0, 3),
                    "drift": sd * np.log(bmid[ib] / m_T) * 1e4,
                    "age": (tp - tg[idx]) / 1e9,
                }
                out.append(pl.DataFrame(rows))
        if (kd + 1) % 20 == 0 or kd == 0:
            print(f"  [{kd+1}/{len(days)}] {dt}", flush=True)

    R = pl.concat(out)
    R.write_parquet(DATA / f"latency_features_{tag}.parquet")
    print(f"書き出し latency_features({R.height:,} 行 = "
          f"{R.height // len(DELTAS_MS):,} 約定 × {len(DELTAS_MS)} 時点)")


if __name__ == "__main__":
    main()
