"""信号の十分位 × 待ち行列がはける時間の十分位 — 2 次元の集計。

    uv run python scripts/build_heat.py --coin xyz:MU

軸
--
  信号 (S1..S10)   特徴量そのものの十分位。S1 = 最も売り側、S10 = 最も買い側
  待ち行列 (D1..D10) tau = 自分の側の板の厚み ÷ 直近 10 秒の同方向テイカー流量。
                   D1 = すぐはける (約定しやすい)、D10 = はけない

どちらも**前日の分布**で境目を作る。当日の実現値を知らないと決まらない境目は
各時点で計算できず、実装不能な規則になる (CLAUDE.md checklist C-16)。

セルに入れる量
--------------
  n         その状態が現れた回数 (1 秒格子)
  n_fill    BBO に置いたメイカー注文が h までに約定した回数
  sum_pnl   約定したときの損益 (bp)。買い指値なら (mid(T+h) - bid(T))
  sum_ret   **符号をつけない**前向きリターン (bp)。信号の向きが見えるように
  sum_retc  同じものを microprice で測ったもの
  sum_mm    T+h 時点の microprice - mid (約定した分だけ)

sum_ret に信号方向の符号を掛けないのは、ヒートマップ上で
「S1 では下がり S10 では上がる」という勾配をそのまま見せるためである。

時間契約は build_maker.py と同じ。約定判定 (day_fills) と特徴量
(day_features) は同じ関数を共用しており、定義がずれない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_maker import MAKER_FEE_BP, day_fills  # noqa: E402
from build_pred import GRID_N, day_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HORIZONS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ("obi", "ofi_10s", "ai_net_10s")
SIDES = (1, -1)
NB = 10                      # 両軸とも十分位
DAILY_H = (0.1, 1.0, 10.0)   # 日ごとにも残すホライズン (誤差評価用)
STATS = ("n", "n_fill", "sum_pnl", "sum_ret", "sum_retc",
         "sum_ret_fill", "sum_retc_fill", "sum_mm")


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
    nday = 0
    for k, dt in enumerate(days):
        D = day_features(bb.filter(pl.col("dt") == dt).sort("ts"), FID.get(dt))
        if D is None:
            continue
        F, R, RC = D["F"], D["R"], D["RC"]
        bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
        mid = 0.5 * (bid + ask)
        mmg = D["mmg"]
        FILL, TAU = day_fills(D, FID[dt])
        SZ = {1: qb, -1: qa}
        PX = {1: bid, -1: ask}

        ok_s = {f: edges_ok(prev_s.get(f)) for f in FEATS}
        ok_t = {sd: edges_ok(prev_t.get(sd)) for sd in SIDES}
        if all(ok_s.values()) and all(ok_t.values()):
            nday += 1
        for fi, f in enumerate(FEATS):
            if not ok_s[f]:
                continue
            sdec = np.searchsorted(prev_s[f], F[f], side="right")
            finf = np.isfinite(F[f])
            for si, sd in enumerate(SIDES):
                if not ok_t[sd]:
                    continue
                tdec = np.searchsorted(prev_t[sd], TAU[sd], side="right")
                base = finf & np.isfinite(mid) & (SZ[sd] > 0)
                cell = sdec * NB + tdec          # 0..99 の平坦な添字
                for hi, h in enumerate(HORIZONS):
                    vb, va = FILL[h]
                    got = (vb if sd == 1 else va) >= SZ[sd]
                    m_end = mid * np.exp(R[h] / 1e4)
                    sel = base & np.isfinite(m_end) & np.isfinite(RC[h])
                    pnl = (m_end - PX[sd]) * sd / mid * 1e4
                    mm_end = mmg + RC[h] - R[h]
                    g = sel & got
                    add = {
                        "n": (sel, np.ones(GRID_N + 1)),
                        "n_fill": (g, np.ones(GRID_N + 1)),
                        "sum_pnl": (g, pnl),
                        "sum_ret": (sel, R[h]),
                        "sum_retc": (sel, RC[h]),
                        # 約定した部分集合だけの、**同じ窓 T→T+h** のリターン。
                        # SelectionPenalty = これ − 全体 を作るために要る。
                        # tau 起点だと約定しなかった側が定義できないので、
                        # 比較は必ず同じ窓 (T 起点) で行う。
                        "sum_ret_fill": (g, R[h]),
                        "sum_retc_fill": (g, RC[h]),
                        "sum_mm": (g, mm_end)}
                    flat = {}
                    for key, (m, val) in add.items():
                        t = np.bincount(cell[m], weights=val[m],
                                        minlength=NB * NB)[: NB * NB]
                        acc[key][fi, si, :, :, hi] += t.reshape(NB, NB)
                        flat[key] = t
                    if h in DAILY_H:
                        nz = np.flatnonzero(flat["n"] > 0)
                        for c in nz:
                            drows.append({
                                "dt": dt, "feat": f, "side": sd, "h": h,
                                "s_dec": int(c // NB), "q_dec": int(c % NB),
                                **{key: float(flat[key][c]) for key in STATS}})
        prev_s = {f: np.nanquantile(F[f][np.isfinite(F[f])],
                                    np.arange(1, NB) / NB)
                  for f in FEATS if np.isfinite(F[f]).any()}
        prev_t = {}
        for sd in SIDES:
            v = TAU[sd][np.isfinite(TAU[sd])]
            if v.size:
                prev_t[sd] = np.quantile(v, np.arange(1, NB) / NB)
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True)

    rows = []
    for fi, f in enumerate(FEATS):
        for si, sd in enumerate(SIDES):
            for s in range(NB):
                for d in range(NB):
                    for hi, h in enumerate(HORIZONS):
                        if acc["n"][fi, si, s, d, hi] == 0:
                            continue
                        rows.append({"feat": f, "side": sd, "h": h,
                                     "s_dec": s, "q_dec": d,
                                     **{key: float(acc[key][fi, si, s, d, hi])
                                        for key in STATS}})
    pl.DataFrame(rows).write_csv(DATA / f"heat_pooled_{tag}.csv")
    pl.DataFrame(drows).write_parquet(DATA / f"heat_daily_{tag}.parquet")
    print(f"書き出し heat_pooled({len(rows):,} セル) / "
          f"heat_daily({len(drows):,} 行) / {nday} 日")


if __name__ == "__main__":
    main()
