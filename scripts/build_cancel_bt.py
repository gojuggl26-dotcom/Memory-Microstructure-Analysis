"""遅延を織り込んだ取り消しの backtest。

    uv run python scripts/build_cancel_bt.py --coin xyz:MU

何をするか
----------
1 秒ごとに最良気配へ指値を置き(寿命 10 秒)、**板が更新されるたびに**
toxic 度のスコアを計算する。スコアが閾値を超えたら取り消しを出す。
取り消しは **L ミリ秒後**に有効になる。約定 tau がそれより早ければ
**取り消しは間に合わず約定する**。

    有効時刻 t_eff = (最初にスコアが閾値を超えた板更新の時刻) + L
    約定する条件: tau <= min(t_eff, T + 寿命)

L は「通信の往復 + 次のブロックまでの待ち」をまとめた 1 つの数として振る。
実測ではブロックは約 65 ms 周期なので、L = 65 / 130 / 200 / 300 ms は
それぞれ 1 / 2 / 3 / 5 ブロック程度に対応する。L = 0 は「遅延ゼロ」の
理想で、上限を見るための対照である。

スコア
------
[遅延の報告](../reports/) で使った 10 変数のうち、**発注に依存しない 7 変数**
(OBI / 自分側の厚み / スプレッド / OFI 1秒・100ms / 攻撃的流量 1秒・100ms)
だけを使う。任意の時刻で計算できる必要があるためで、標本外 AUC は
全 10 変数の 99.7%(250ms 前で 0.5751 対 0.5770)なので損失はほぼ無い。

係数と標準化は**学習期間(前 60% の日)だけ**で推定し、閾値も学習期間の
スコア分布から決める。評価は後 40% の日で行う。

対照
----
同じ発火率で**無作為に**取り消す版も回す。スコアが本当に効いているのか、
それとも「取り消す回数を増やせば当たり前に良くなる」だけなのかを分ける
(CLAUDE.md checklist B5/B12)。
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
from plot_latency import TRAIN_FRAC, fit_logit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H_QUOTE = 10.0                 # クオートの寿命 (秒)
H_HOLD = 10.0                  # 約定後の保有 (秒)
LATENCY_MS = [0, 65, 130, 200, 300]
TRIG_RATE = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
IND = ("obi", "depth", "spread", "ofi_1s", "ofi_100ms", "aggr_1s", "aggr_100ms")
SIDES = (1, -1)
PRE_NS = 2_000_000
DAY_NS = 86_400_000_000_000


def book_features(d, sd):
    """板が更新されるたびの、発注に依存しない特徴量。任意の時刻で計算できる。"""
    bts = d["ts"].cast(pl.Int64).to_numpy()
    pb = d["best_bid"].to_numpy()
    pa = d["best_ask"].to_numpy()
    qb = d["bid_sz"].to_numpy()
    qa = d["ask_sz"].to_numpy()
    mid = 0.5 * (pb + pa)
    tot = qb + qa
    obi = np.where(tot > 0, (qb - qa) / np.where(tot > 0, tot, 1.0), 0.0)
    e = np.concatenate([[0.0], (
        (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
        - ((pa[1:] <= pa[:-1]) * qa[1:] - (pa[1:] >= pa[:-1]) * qa[:-1]))])
    cofi = np.cumsum(e)
    i1s = np.clip(np.searchsorted(bts, bts - 10 ** 9, side="right") - 1, 0, None)
    i100 = np.clip(np.searchsorted(bts, bts - 100_000_000, side="right") - 1,
                   0, None)
    return bts, {
        "obi": sd * obi,
        "depth": qb if sd == 1 else qa,
        "spread": (pa - pb) / mid * 1e4,
        "ofi_1s": sd * (cofi - cofi[i1s]),
        "ofi_100ms": sd * (cofi - cofi[i100]),
    }


def aggr_at(bts, ft, fs, isb, sd, win_ns):
    """板更新時刻ごとの、直前 win_ns の符号つきテイカー数量。"""
    cnet = np.concatenate([[0.0], np.cumsum(fs * np.where(isb, 1.0, -1.0))])
    j1 = np.searchsorted(ft, bts, side="right")
    j0 = np.searchsorted(ft, bts - win_ns, side="right")
    return sd * (cnet[j1] - cnet[j0])


def first_true_at_or_after(mask):
    """nxt[j] = j 以上で最初に True になる添字。無ければ len(mask)。"""
    n = mask.size
    idx = np.where(mask, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    return np.append(nxt, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--hq", type=float, default=H_QUOTE,
                    help="クオートの寿命 (秒)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    hq = a.hq
    suf = "" if hq == H_QUOTE else f"_hq{hq:g}"

    # --- スコアの係数と標準化を学習期間から作る -----------------------------
    R = pl.read_parquet(DATA / f"latency_features_{tag}.parquet").filter(
        pl.col("delta_ms") == 250)
    alld = sorted(R["dt"].unique().to_list())
    ntr = int(round(len(alld) * TRAIN_FRAC))
    tr_days = set(alld[:ntr])
    istr = R["dt"].is_in(list(tr_days)).to_numpy()
    y = R["y"].to_numpy().astype(float)
    X = np.column_stack([R[f].to_numpy().astype(float) for f in IND])
    med = np.nanmedian(X[istr], axis=0)
    X = np.where(np.isfinite(X), X, med)
    mu = X[istr].mean(axis=0)
    sg = np.where(X[istr].std(axis=0) > 0, X[istr].std(axis=0), 1.0)
    W = fit_logit((X[istr] - mu) / sg, y[istr])
    print(f"学習 {ntr} 日で係数を推定。標準化後の重み: "
          + " ".join(f"{f}={w:+.3f}" for f, w in zip(IND, W[1:])), flush=True)

    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "px", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    days = sorted(bb["dt"].unique().to_list())

    # --- 1 巡目: 学習期間のスコア分布から閾値を決める -----------------------
    samp = []
    for dt in [d for d in days if d in tr_days][::4]:
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        for sd in SIDES:
            bts, F = book_features(d, sd)
            fd = FID.get(dt)
            if fd is None:
                continue
            ft = fd["ts"].cast(pl.Int64).to_numpy()
            fs = fd["sz"].to_numpy()
            isb = fd["side"].to_numpy() == "B"
            F["aggr_1s"] = aggr_at(bts, ft, fs, isb, sd, 10 ** 9)
            F["aggr_100ms"] = aggr_at(bts, ft, fs, isb, sd, 100_000_000)
            Z = (np.column_stack([F[f] for f in IND]) - mu) / sg
            samp.append((Z @ W[1:])[::20])
    sc = np.concatenate(samp)
    THR = {r: (np.quantile(sc, 1 - r) if r > 0 else np.inf) for r in TRIG_RATE}
    print(f"閾値 (学習期間 {sc.size:,} 標本): "
          + " ".join(f"{int(r*100)}%={THR[r]:.3f}" for r in TRIG_RATE if r > 0),
          flush=True)

    rows = []
    rng = np.random.default_rng(20260905)
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
        bmid = 0.5 * (d["best_bid"].to_numpy() + d["best_ask"].to_numpy())
        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fs = fd["sz"].to_numpy()
        isb = fd["side"].to_numpy() == "B"
        split = "学習" if dt in tr_days else "評価"

        for sd in SIDES:
            bts, F = book_features(d, sd)
            F["aggr_1s"] = aggr_at(bts, ft, fs, isb, sd, 10 ** 9)
            F["aggr_100ms"] = aggr_at(bts, ft, fs, isb, sd, 100_000_000)
            Z = (np.column_stack([F[f] for f in IND]) - mu) / sg
            s = Z @ W[1:]
            rnd = rng.random(s.size)

            tau = TAU[sd]
            Q0 = qb if sd == 1 else qa
            base = np.isfinite(mid) & (Q0 > 0)
            fill_base = base & (tau >= 0) & (tau <= tg + int(hq * 1e9)) \
                & (tau + int(H_HOLD * 1e9) <= day_end)
            k = np.flatnonzero(base)
            tv = tau[k]
            has = fill_base[k]
            ip = np.clip(np.searchsorted(bts, np.where(has, tv - PRE_NS, bts[0]),
                                         side="right") - 1, 0, bts.size - 1)
            ih = np.clip(np.searchsorted(bts, np.where(has, tv + int(H_HOLD * 1e9),
                                                       bts[0]),
                                         side="right") - 1, 0, bts.size - 1)
            m_pre = bmid[ip]
            pq = (bid if sd == 1 else ask)[k]
            edge = (m_pre - pq) * sd / m_pre * 1e4
            mk = sd * np.log(bmid[ih] / m_pre) * 1e4
            pnl = edge - MAKER_FEE_BP + mk
            j_start = np.searchsorted(bts, tg[k], side="right")

            for r in TRIG_RATE:
                for name, sig in (("スコア", s), ("無作為", rnd)):
                    if r == 0.0:
                        if name == "無作為":
                            continue
                        t_trig = np.full(k.size, np.inf)
                    else:
                        th = THR[r] if name == "スコア" else 1.0 - r
                        nxt = first_true_at_or_after(sig > th)
                        jj = nxt[np.minimum(j_start, bts.size)]
                        t_trig = np.where(jj < bts.size,
                                          bts[np.minimum(jj, bts.size - 1)]
                                          .astype(np.float64), np.inf)
                    for L in LATENCY_MS:
                        t_eff = t_trig + L * 1e6
                        kept = has & (tv <= t_eff)
                        cancelled = base[k] & (t_trig < np.where(has, tv, np.inf))
                        rows.append({
                            "dt": dt, "split": split, "side": sd, "rule": name,
                            "trig_rate": r, "lat_ms": L,
                            "n": int(k.size),
                            "n_cancel": int(cancelled.sum()),
                            "n_fill": int(kept.sum()),
                            "sum_mk": float(np.nansum(mk[kept])),
                            "sum_edge": float(np.nansum(edge[kept])),
                            "sum_pnl": float(np.nansum(pnl[kept]))})
        if (kd + 1) % 20 == 0 or kd == 0:
            print(f"  [{kd+1}/{len(days)}] {dt}", flush=True)

    pl.DataFrame(rows).write_parquet(DATA / f"cancel_bt_{tag}{suf}.parquet")
    print(f"書き出し cancel_bt{suf}({len(rows):,} 行・寿命 {hq:g} 秒)")


if __name__ == "__main__":
    main()
