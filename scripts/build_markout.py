"""約定の前後を 3 つに分解し、toxic な約定の事前兆候を測る。

    uv run python scripts/build_markout.py --coin xyz:MU

3 分解
------
約定した注文について、発注時刻 T・約定時刻 tau・保有終了 tau+h を使って

    A. r_{T -> tau}      発注してから約定するまでに、もう何 bp 動いていたか
    B. r_{tau -> tau+h}  約定してから何 bp 動くか (= これまでの AS)
    P. SelectionPenalty  E[r_{T->T+h} | Fill] - E[r_{T->T+h}]

を**別々に**持つ。A と B は窓が違うので足しても P にはならない。
恒等式として成り立つのは

    r_{T -> T+h} = r_{T -> tau} + r_{tau -> T+h}

のほうで、これも出して検算する (B は tau+h まで伸びるので別物)。

A は「置いたまま待っている間に食らった損」、B は「約定した後に食らう損」で、
**取り消しの設計に直結する**のは A のほうである。A が大きいなら
「約定する前に引く」余地があり、B が大きいなら「そもそも置かない」しかない。

toxic / non-toxic の比較
------------------------
約定後の markout (B を損益の向きに符号を付けたもの) の**日ごとの五分位**で
約定を分ける。G1 = 最も toxic、G5 = 最も無害。そのうえで
**約定の 500ms / 1s / 2s 前**の状態を比べる:

    OBI          板の不均衡
    OFI          直前 1 秒の注文流不均衡
    攻撃的流量    直前 1 秒の符号つきテイカー数量
    自分側の厚み  最良気配の数量
    待ち行列の消化 発注時の Q_ahead のうち何割が食われたか
    値動き        発注時 T からその時点までの mid の動き

★注意: toxic かどうかは**約定後**の情報で決めている。したがってこれは
「toxic な約定の前には何が起きているか」という**記述**であって、
そのまま取り消し規則になるわけではない。規則にするには、事前の指標だけで
toxic を当てられるかを別途確かめる必要がある (本スクリプトは
各指標の判別力 AUC も出す)。
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
H_FILL = 10.0            # toxic 比較に使うクオートの寿命
# A (置いてから約定するまでの劣化) はクオートの寿命そのもので決まるので、
# 寿命を振って曲線にする。これが取り消し設計に直結する。
H_QUOTE = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
H_HOLD = [1.0, 10.0]     # 約定後の保有時間
DELTAS = [0.0, 0.5, 1.0, 2.0]     # 約定の何秒前を見るか
SIDES = (1, -1)
NG = 5                   # toxic の五分位
PRE_NS = 2_000_000
DAY_NS = 86_400_000_000_000
VARS = ("obi", "ofi", "aggr", "depth", "qdepl", "drift")


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

    dec, tox, auc = [], [], []
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
        tot = qbb + qaa
        bobi = np.where(tot > 0, (qbb - qaa) / np.where(tot > 0, tot, 1.0), 0.0)
        # 気配の更新ごとの OFI を累積して、任意の時刻で窓の差が取れるようにする
        e = np.concatenate([[0.0], (
            (pbb[1:] >= pbb[:-1]) * qbb[1:] - (pbb[1:] <= pbb[:-1]) * qbb[:-1]
            - ((paa[1:] <= paa[:-1]) * qaa[1:] - (paa[1:] >= paa[:-1]) * qaa[:-1]))])
        cofi = np.cumsum(e)

        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fs = fd["sz"].to_numpy()
        fp = fd["px"].to_numpy()
        isb = fd["side"].to_numpy() == "B"
        cnet = np.concatenate([[0.0], np.cumsum(fs * np.where(isb, 1.0, -1.0))])
        csell = np.concatenate([[0.0], np.cumsum(np.where(~isb, fs, 0.0))])
        cbuy = np.concatenate([[0.0], np.cumsum(np.where(isb, fs, 0.0))])

        def at(t):
            return np.clip(np.searchsorted(bts, t, side="right") - 1,
                           0, bts.size - 1)

        for sd in SIDES:
            tau = TAU[sd]
            Q0 = qb if sd == 1 else qa
            ok = (tau >= 0) & (tau <= tg + int(H_FILL * 1e9)) & np.isfinite(mid) \
                & (Q0 > 0)
            idx = np.flatnonzero(ok)
            if idx.size < 200:
                continue
            tv = tau[idx]
            m_T = mid[idx]
            ip = at(tv - PRE_NS)
            m_pre = bmid[ip]
            rA = np.log(m_pre / m_T) * 1e4                       # A: T -> tau
            row = {"dt": dt, "side": sd, "n": int(idx.size),
                   "sum_A": float(rA.sum())}
            # クオートの寿命ごとの A と B (約定集合が寿命で変わる)
            for hq in H_QUOTE:
                m = tv <= tg[idx] + int(hq * 1e9)
                row[f"nq_{hq:g}"] = int(m.sum())
                row[f"A_{hq:g}"] = float(rA[m].sum())
                row[f"spr_{hq:g}"] = float(
                    ((ask[idx] - bid[idx]) / m_T * 1e4 / 2.0)[m].sum())
                te2 = tv[m] + int(10 * 1e9)
                gd = te2 <= day_end
                ih2 = at(np.where(gd, te2, tv[m]))
                row[f"B_{hq:g}"] = float(np.nansum(
                    np.where(gd, np.log(bmid[ih2] / m_pre[m]) * 1e4, np.nan)))
                row[f"nB_{hq:g}"] = int(gd.sum())
            for h in H_HOLD:
                te = tv + int(h * 1e9)
                good = te <= day_end
                ih = at(np.where(good, te, tv))
                rB = np.where(good, np.log(bmid[ih] / m_pre) * 1e4, np.nan)
                # 元の窓の残り T+H_FILL まで
                tw = tg[idx] + int(H_FILL * 1e9)
                iw = at(np.minimum(tw, day_end - 1))
                rRest = np.log(bmid[iw] / m_pre) * 1e4
                rTot = np.log(bmid[iw] / m_T) * 1e4
                row[f"sum_B_{h:g}"] = float(np.nansum(rB))
                row[f"n_B_{h:g}"] = int(np.isfinite(rB).sum())
                row[f"sum_rest_{h:g}"] = float(np.nansum(rRest))
                row[f"sum_tot_{h:g}"] = float(np.nansum(rTot))
                if h == H_HOLD[-1]:
                    mk = sd * rB                        # 損益の向きの markout
                    fin = np.isfinite(mk)
                    if fin.sum() < 100:
                        continue
                    q = np.nanquantile(mk[fin], np.arange(1, NG) / NG)
                    grp = np.searchsorted(q, mk)
                    # 事前の状態
                    for dl in DELTAS:
                        tp = tv - int(dl * 1e9)
                        ib = at(tp)
                        v = {"obi": sd * bobi[ib],
                             "depth": (qbb if sd == 1 else qaa)[ib],
                             "drift": sd * np.log(bmid[ib] / m_T) * 1e4}
                        i1 = at(tp - 10 ** 9)
                        v["ofi"] = sd * (cofi[ib] - cofi[i1])
                        j0 = np.searchsorted(ft, tp - 10 ** 9, side="right")
                        j1 = np.searchsorted(ft, tp, side="right")
                        v["aggr"] = sd * (cnet[j1] - cnet[j0])
                        k0 = np.searchsorted(ft, tg[idx], side="right")
                        cc = csell if sd == 1 else cbuy
                        v["qdepl"] = np.clip((cc[j1] - cc[k0]) / Q0[idx], 0, 3)
                        for g in range(NG):
                            m = fin & (grp == g)
                            if not m.any():
                                continue
                            tox.append({"dt": dt, "side": sd, "delta": dl,
                                        "grp": g, "n": int(m.sum()),
                                        "markout": float(mk[m].mean()),
                                        **{f"{k}": float(np.nanmean(v[k][m]))
                                           for k in VARS}})
                    # 判別力: 最も toxic な 20% を事前指標だけで当てられるか
                    y = (grp == 0)
                    for dl in (0.5, 1.0, 2.0):
                        tp = tv - int(dl * 1e9)
                        ib = at(tp)
                        i1 = at(tp - 10 ** 9)
                        j0 = np.searchsorted(ft, tp - 10 ** 9, side="right")
                        j1 = np.searchsorted(ft, tp, side="right")
                        k0 = np.searchsorted(ft, tg[idx], side="right")
                        cc = csell if sd == 1 else cbuy
                        vv = {"obi": sd * bobi[ib],
                              "ofi": sd * (cofi[ib] - cofi[i1]),
                              "aggr": sd * (cnet[j1] - cnet[j0]),
                              "depth": (qbb if sd == 1 else qaa)[ib],
                              "qdepl": np.clip((cc[j1] - cc[k0]) / Q0[idx], 0, 3),
                              "drift": sd * np.log(bmid[ib] / m_T) * 1e4}
                        for k, x in vv.items():
                            f2 = fin & np.isfinite(x)
                            if f2.sum() < 100 or y[f2].sum() < 10:
                                continue
                            r = np.argsort(np.argsort(x[f2])) + 1.0
                            n1 = float(y[f2].sum())
                            n0 = float(f2.sum() - n1)
                            a_ = (r[y[f2]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                            auc.append({"dt": dt, "side": sd, "delta": dl,
                                        "var": k, "auc": float(a_),
                                        "n": int(f2.sum())})
            dec.append(row)
        if (kd + 1) % 20 == 0 or kd == 0:
            print(f"  [{kd+1}/{len(days)}] {dt}", flush=True)

    pl.DataFrame(dec).write_csv(DATA / f"markout_decomp_{tag}.csv")
    pl.DataFrame(tox).write_csv(DATA / f"markout_toxic_{tag}.csv")
    pl.DataFrame(auc).write_csv(DATA / f"markout_auc_{tag}.csv")
    print(f"書き出し markout_decomp({len(dec)}) / markout_toxic({len(tox):,}) / "
          f"markout_auc({len(auc):,})")


if __name__ == "__main__":
    main()
