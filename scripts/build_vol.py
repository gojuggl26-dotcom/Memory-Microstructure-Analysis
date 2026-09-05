"""L4 由来の 5 本の価格・板系列から実現ボラティリティ族を測る (item 47)。

測る量
------
価格系列 (mid / microprice / spread は対数、book pressure は水準、OFI は累積) を
刻み幅 delta の暦時間格子に落とし、その差分 r_i について 1 日ごとに

  RV   = sum r_i^2                     realized variance
  RVol = sqrt(RV)                      realized volatility
  BV   = (pi/2)(n/(n-1)) sum |r_i||r_{i-1}|   bipower variation (跳びに頑健)
  RS+  = sum r_i^2 1{r_i>0}            upside semivariance
  RS-  = sum r_i^2 1{r_i<0}            downside semivariance   (RS+ + RS- = RV)
  RQ   = (n/3) sum r_i^4               realized quarticity
  TQ   = tripower quarticity           (跳び検定の分母に使う)
  z    = Huang-Tauchen の跳び統計量     (帰無 = 跳びなし で N(0,1))

を計算する。SE(RV) = sqrt(2 RQ / n) が RV そのものの測定誤差である。

時間契約 (CLAUDE.md の厳禁事項)
------------------------------
- 出力は日次の実現量で、日 D の値は日 D 内の観測だけから作る。日をまたぐ
  リターン (オーバーナイト) は取らない。
- 説明変数として使うなら日 D の終了後、すなわち D+1 以降のリターンにのみ
  当てること。同じ日のリターンを同じ日の RV で説明してはならない。
- 格子への割り当ては後ろ向き (searchsorted の side='right' - 1) で、
  格子時刻 t の値は t 以前の最後の気配である。

雑音について
------------
RV は刻みを細かくするほどマイクロストラクチャー雑音で上振れする。i.i.d. 雑音
(分散 omega^2) のもとで E[RV_n] = IV + 2 n omega^2 なので、細かい格子での
RV を n に回帰した傾きの半分が omega^2 の推定になる。signature plot を出す
目的はこれを見せることである。
"""
from __future__ import annotations

import argparse
import sys
from math import gamma, pi
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_hazard import ofi_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DAY_NS = 86_400_000_000_000
# 暦時間の刻み (秒)。0 は「気配が動くたび」= delta -> 0 の極限。
GRIDS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0]
# 雑音 omega^2 の推定に使う細かい格子
NOISE_GRIDS = [0.1, 0.25, 0.5, 1.0, 2.0]
# 基準格子: リターンの 1 次自己相関が 0 になる最も細かい格子 = 30 秒。
# 98 日の中央値で rho1 は 1s:+0.101 5s:+0.050 10s:+0.026 30s:-0.000 60s:+0.013 で、
# RVol もこの辺りで平ら (10s 497.8 / 30s 498.6 / 60s 504.1 bp)。根拠は本文 3 節。
REF_GRID = 30.0
# 時刻帯 profile だけは形を見る目的なので細かい格子で取る (水準は REF_GRID を見ること)
HOUR_GRID = 5.0
NW_LAGS = 5
SERIES = ("mid", "micro", "spread", "bp", "ofi")
# 差分の報告単位: mid/micro は bp、spread の対数変化は %、板圧力は無次元、OFI は枚
SCALE = {"mid": 1e4, "micro": 1e4, "spread": 1e2, "bp": 1.0, "ofi": 1.0}

MU43 = 2.0 ** (2.0 / 3.0) * gamma(7.0 / 6.0) / gamma(0.5)
Z_FACT = (pi ** 2) / 4.0 + pi - 5.0


def est(r: np.ndarray) -> dict | None:
    """1 日分の差分列から実現量一式を返す。r は NaN を含まないこと。"""
    n = int(r.size)
    if n < 8:
        return None
    r2 = r * r
    rv = float(r2.sum())
    ab = np.abs(r)
    bv = float((pi / 2.0) * (n / (n - 1.0)) * (ab[1:] * ab[:-1]).sum())
    rsp = float(r2[r > 0].sum())
    rsm = float(r2[r < 0].sum())
    rq = float((n / 3.0) * (r2 * r2).sum())
    med = np.median(np.stack([ab[:-2], ab[1:-1], ab[2:]]), axis=0)
    medrv = float(pi / (6.0 - 4.0 * np.sqrt(3.0) + pi) * (n / (n - 2.0)) * (med ** 2).sum())
    rc = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.std() > 0 else np.nan
    # 系列相関に頑健な RV (Bartlett 重みの Newey-West)。この市場のリターンは
    # 正に相関しているので、素の RV は細かい格子で下振れする。
    nw1 = rv + 2.0 * float((r[1:] * r[:-1]).sum())
    nw5 = rv
    for lag in range(1, NW_LAGS + 1):
        if lag < n:
            nw5 += 2.0 * (1.0 - lag / (NW_LAGS + 1.0)) * float((r[lag:] * r[:-lag]).sum())
    a = ab ** (4.0 / 3.0)
    tq = float(n * MU43 ** -3.0 * (n / (n - 2.0)) * (a[2:] * a[1:-1] * a[:-2]).sum())
    rj = (rv - bv) / rv if rv > 0 else np.nan
    den = Z_FACT * max(1.0, tq / bv ** 2) / n if bv > 0 else np.nan
    z = rj / np.sqrt(den) if (den and den > 0) else np.nan
    return {
        "n": n, "rv": rv, "rvol": float(np.sqrt(rv)), "bv": bv, "medrv": medrv, "rho1": rc, "rv_ac1": nw1, "rv_nw5": nw5,
        "rs_up": rsp, "rs_dn": rsm, "rq": rq, "tq": tq,
        "se_rv": float(np.sqrt(2.0 * rq / n)),
        "jump": max(rv - bv, 0.0), "rj": rj, "z": z,
        "mean_abs": float(ab.mean()),
        "max_share": float(r2.max() / rv) if rv > 0 else np.nan,
        "zero_share": float((r == 0).mean()),
        "ret_sum": float(r.sum()),
    }


def levels(d: pl.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """bbo の各更新時点での 5 本の系列 (水準) を作る。"""
    ts = d["ts"].cast(pl.Int64).to_numpy()
    pb = d["best_bid"].to_numpy().astype(np.float64)
    pa = d["best_ask"].to_numpy().astype(np.float64)
    qb = d["bid_sz"].to_numpy().astype(np.float64)
    qa = d["ask_sz"].to_numpy().astype(np.float64)
    mid = 0.5 * (pb + pa)
    tot = qb + qa
    micro = np.where(tot > 0, (qa * pb + qb * pa) / np.where(tot > 0, tot, 1.0), mid)
    lv = {
        "mid": np.log(mid),
        "micro": np.log(micro),
        "spread": np.log(pa - pb),
        "bp": np.where(tot > 0, (qb - qa) / np.where(tot > 0, tot, 1.0), 0.0),
        "ofi": np.cumsum(ofi_series({"bid": pb, "ask": pa, "bsz": qb, "asz": qa})),
    }
    return ts, lv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb = pl.read_parquet(DATA / f"bbo_{tag}.parquet")
    bb, ndrop = clean_bbo(bb)
    print(f"bbo {bb.height:,} 行 (異常 {ndrop:,} 行を除外)", flush=True)
    days = sorted(bb["dt"].unique().to_list())

    rows, hours, noise = [], [], []
    for k, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        ts, lv = levels(d)
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        gapmax = float(np.diff(ts).max()) / 1e9 if ts.size > 1 else 0.0
        lm = lv["mid"]
        park = float((lm.max() - lm.min()) * 1e4)
        oc = float((lm[-1] - lm[0]) * 1e4)
        rvs = {}
        for g in GRIDS:
            if g == 0.0:
                idx = np.arange(ts.size)
            else:
                step = int(round(g * 1e9))
                tg = np.arange(d0, d0 + DAY_NS + 1, step)
                idx = np.searchsorted(ts, tg, side="right") - 1
                idx = idx[idx >= 0]
                if idx.size < 9:
                    continue
            for s in SERIES:
                r = np.diff(lv[s][idx]) * SCALE[s]
                r = r[np.isfinite(r)]
                e = est(r)
                if e is None:
                    continue
                rows.append({"dt": dt, "grid_s": g, "series": s, **e,
                             "park_bp": park, "oc_bp": oc, "gap_max_s": gapmax})
                if s == "mid" and g > 0:
                    rvs[g] = e["rv"]
        # RV(n) = IV + 2 n omega^2 の回帰。i.i.d. 雑音なら傾き > 0 だが、
        # 気配が固まる(stale)市場では傾きが負になる。符号ごと報告する。
        gs = [g for g in NOISE_GRIDS if g in rvs]
        if len(gs) >= 3:
            nn = np.array([86400.0 / g for g in gs])
            yy = np.array([rvs[g] for g in gs])
            sl, ic = np.polyfit(nn, yy, 1)
            noise.append({"dt": dt, "slope": float(sl), "iv_hat": float(ic),
                          "omega_bp": float(np.sqrt(max(sl, 0.0) / 2.0)),
                          "half_spread_bp": float(np.median(
                              (d["best_ask"] - d["best_bid"]).to_numpy()
                              / (0.5 * (d["best_ask"] + d["best_bid"])).to_numpy()) * 1e4 / 2)})
        # 時刻帯ごとの RV (HOUR_GRID・5 系列)。日内の形を見るためのもの。
        tg = np.arange(d0, d0 + DAY_NS + 1, int(HOUR_GRID * 1e9))
        gi = np.searchsorted(ts, tg, side="right") - 1
        gi = gi[gi >= 0]
        hh = np.clip(((tg[-gi.size:][1:] - d0) // (3600 * 10 ** 9)).astype(np.int64), 0, 23)
        for sname in SERIES:
            rr = np.diff(lv[sname][gi]) * SCALE[sname]
            acc = np.zeros(24)
            np.add.at(acc, hh, np.where(np.isfinite(rr), rr * rr, 0.0))
            hours.append({"dt": dt, "series": sname,
                          **{f"h{i:02d}": float(acc[i]) for i in range(24)}})
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt} rows={d.height:,} 最大無更新={gapmax:.1f}s", flush=True)

    pl.DataFrame(rows).write_csv(DATA / f"vol_daily_{tag}.csv")
    pl.DataFrame(noise).write_csv(DATA / f"vol_noise_{tag}.csv")
    pl.DataFrame(hours).write_csv(DATA / f"vol_hour_{tag}.csv")
    print(f"書き出し: vol_daily({len(rows):,}) / vol_noise({len(noise)}) / vol_hour({len(hours)})")


if __name__ == "__main__":
    main()
