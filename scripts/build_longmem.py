"""9 つの系列について長期記憶と自己相関を 8 つの推定量で測る。

【対象の 9 系列】いずれも **1 秒格子**(1 日 86,400 点)。

    OFI            最良気配の流量の偏り(Cont, Kukanov and Stoikov 2014)の 1 秒合計
    depth          最良気配の両側の数量(bid_sz + ask_sz)の 1 秒平均
    spread         相対スプレッド[bp]の 1 秒平均
    order arrivals 1 秒間の新規発注(open)の件数
    cancellations  1 秒間の終端イベント(取消 + 残量 0 の filled)の件数
    order size     1 秒間に出された注文の平均数量
    liquidity churn 1 秒間に板へ入った数量 + 出ていった数量
    queue size     最良気配に並んでいる**注文の本数**
    wallet activity 1 秒間に何かメッセージを出した口座の数

**depth は数量、queue size は本数**で、別の量である。

【8 つの推定量】

    ACF ρ(k)                 自己相関(FFT で 1 時間ラグまで)
    PACF φ(k,k)              偏自己相関(Durbin-Levinson、50 ラグまで)
    lag-1 autocorrelation    ρ(1)
    lag-k autocorrelation    ρ(10) / ρ(60) / ρ(300) / ρ(3600)
    Hurst exponent H         R/S 法
    DFA α                    1 次トレンドを除いた変動解析
    GPH d                    ペリオドグラムの低周波を対数回帰
    fractional differencing d Local Whittle 推定量

【★4 つの推定量は同じものを違う道から測っている】
定常な長期記憶過程では次が成り立つ。

```math
d = H - \\tfrac{1}{2}, \\qquad \\alpha_{DFA} = H = d + \\tfrac{1}{2}
```

したがって **R/S・DFA・GPH・Local Whittle から出した d が揃うかどうか**が
そのまま検算になる。揃わなければ、長期記憶ではない何か(周期・トレンド・
外れ値)を拾っている。

【★日内周期は長期記憶に化ける】
これらの系列には米国市場の寄り付きに合わせた強い日内周期がある。
周期はゆっくり減衰する自己相関を作るので、**そのまま推定すると
長期記憶と区別がつかない**。そこで

    生            そのままの系列
    周期除去      時刻(秒)ごとの平均を全日から作り、それを引いた系列

の両方で推定し、並べて出す。差が大きければ「長期記憶」の主張は弱い。

【★帰無対照】
系列を日内で無作為に並べ替えると、長期記憶は消えて H = 0.5 / d = 0 になるはず。
これを対照として同じ推定量にかける。推定量そのものの偏りが見える。

【x が確定する時刻 / y の期間】
記述統計であって予測ではない。

    uv run python scripts/build_longmem.py --coin xyz:MU
出力: data/longmem_daily_<coin>.parquet / .csv … 日 × 系列 × 種別 × 推定量
      data/longmem_curves_<coin>.parquet       … 日 × 系列 × (ACF / PACF / DFA / R/S)
      data/longmem_sample_<coin>.parquet       … 図示用に 1 日ぶんの系列そのもの
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_bbo_turnover import day_lines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DAY_NS = 86_400_000_000_000
SEC_NS = 1_000_000_000
NG = 86_400
PX_UNIT = 0.01
SZ_LOT = 0.001
MAX_REST_NS = 86_400_000_000_000
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]

SERIES = ["ofi", "depth", "spread", "arrivals", "cancels", "order_size",
          "churn", "queue", "wallet_act"]
SLAB = {"ofi": "OFI", "depth": "depth", "spread": "spread",
        "arrivals": "order arrivals", "cancels": "cancellations",
        "order_size": "order size", "churn": "liquidity churn",
        "queue": "queue size", "wallet_act": "wallet activity"}
ACF_LAGS = [1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600]
NPACF = 50
SCALES = np.unique(np.round(10 ** np.arange(0.6, 4.01, 0.2)).astype(int))  # 4〜10000 秒
GPH_M = int(NG ** 0.5)          # 293。標準的な m = n^0.5
LW_M = int(NG ** 0.65)          # 1,585。Local Whittle は少し広く取る
SEED = 20260901


def acf_fft(x: np.ndarray, nlags: int) -> np.ndarray:
    n = len(x)
    v = x - x.mean()
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(v, nf)
    r = np.fft.irfft(f * np.conj(f), nf)[: nlags + 1]
    return r / r[0] if r[0] > 0 else np.full(nlags + 1, np.nan)


def pacf_dl(r: np.ndarray, nlags: int) -> np.ndarray:
    """Durbin-Levinson。ACF から偏自己相関を出す。"""
    phi = np.zeros((nlags + 1, nlags + 1))
    pac = np.full(nlags + 1, np.nan)
    if nlags >= 1:
        phi[1, 1] = r[1]
        pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - float(phi[k - 1, 1:k] @ r[k - 1:0:-1])
        den = 1.0 - float(phi[k - 1, 1:k] @ r[1:k])
        if abs(den) < 1e-12:
            break
        phi[k, k] = num / den
        pac[k] = phi[k, k]
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1:0:-1]
    return pac


def rs_hurst(x: np.ndarray, scales):
    """R/S 法。スケール n に対する R/S の傾きが Hurst 指数。"""
    out = []
    for s in scales:
        m = len(x) // s
        if m < 2:
            continue
        seg = x[: m * s].reshape(m, s)
        z = seg - seg.mean(1, keepdims=True)
        c = np.cumsum(z, axis=1)
        R = c.max(1) - c.min(1)
        S = seg.std(1, ddof=1)
        ok = S > 0
        if ok.sum() < 2:
            continue
        out.append((s, float(np.mean(R[ok] / S[ok]))))
    if len(out) < 4:
        return np.nan, [], []
    ss = np.array([o[0] for o in out], float)
    vv = np.array([o[1] for o in out], float)
    good = vv > 0
    if good.sum() < 4:
        return np.nan, ss, vv
    h = float(np.polyfit(np.log(ss[good]), np.log(vv[good]), 1)[0])
    return h, ss, vv


def dfa(x: np.ndarray, scales):
    """DFA1。累積和の 1 次トレンドを除いた変動の傾きが α。"""
    y = np.cumsum(x - x.mean())
    out = []
    for s in scales:
        m = len(y) // s
        if m < 2 or s < 4:
            continue
        seg = y[: m * s].reshape(m, s).T
        t = np.arange(s, dtype=float)
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        det = seg - A @ coef
        out.append((s, float(np.sqrt((det ** 2).mean()))))
    if len(out) < 4:
        return np.nan, [], []
    ss = np.array([o[0] for o in out], float)
    vv = np.array([o[1] for o in out], float)
    good = vv > 0
    if good.sum() < 4:
        return np.nan, ss, vv
    a = float(np.polyfit(np.log(ss[good]), np.log(vv[good]), 1)[0])
    return a, ss, vv


def periodogram(x: np.ndarray, m: int):
    n = len(x)
    I = np.abs(np.fft.rfft(x - x.mean())) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    return lam, I[1: m + 1]


def gph_d(x: np.ndarray, m: int) -> float:
    lam, I = periodogram(x, m)
    ok = I > 0
    if ok.sum() < 10:
        return np.nan
    X = -np.log(4 * np.sin(lam[ok] / 2) ** 2)
    return float(np.polyfit(X, np.log(I[ok]), 1)[0])


def lw_d(x: np.ndarray, m: int) -> float:
    """Local Whittle。GPH より効率が良い d の推定量。"""
    lam, I = periodogram(x, m)
    ok = I > 0
    if ok.sum() < 10:
        return np.nan
    lam, I = lam[ok], I[ok]
    ll = np.log(lam)
    grid = np.arange(-0.49, 1.501, 0.005)
    G = np.array([np.mean(I * lam ** (2 * d)) for d in grid])
    R = np.log(np.maximum(G, 1e-300)) - 2 * grid * ll.mean()
    return float(grid[int(np.argmin(R))])


def estimate(x: np.ndarray, rng) -> tuple[dict, dict]:
    """1 本の系列から 8 指標を出す。曲線も返す。"""
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(x).any() else 0.0)
    if x.std() <= 0:
        return {}, {}
    r = acf_fft(x, ACF_LAGS[-1])
    p = pacf_dl(r, NPACF)
    h, hs, hv = rs_hurst(x, SCALES)
    a, ds, dv = dfa(x, SCALES)
    g = gph_d(x, GPH_M)
    lw = lw_d(x, LW_M)
    rec = {"acf1": float(r[1]), "hurst": h, "dfa": a, "gph_d": g, "lw_d": lw,
           # ★同じ量を違う道から見た値。揃うかどうかが検算になる
           "d_from_hurst": h - 0.5, "d_from_dfa": a - 0.5}
    for k in ACF_LAGS:
        rec[f"acf{k}"] = float(r[k]) if k < len(r) else np.nan
    cur = {"acf": (np.array(ACF_LAGS, float),
                   np.array([r[k] for k in ACF_LAGS])),
           "pacf": (np.arange(1, NPACF + 1, dtype=float), p[1:]),
           "rs": (np.asarray(hs, float), np.asarray(hv, float)),
           "dfa": (np.asarray(ds, float), np.asarray(dv, float))}
    return rec, cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"data/l1_{tag} が空。先に fetch_l1.py を実行すること")
    bbo, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 信じられない行を {n_drop:,} 除いた -> {bbo.height:,} 行", file=sys.stderr)
    wdir = ROOT / "data" / f"l1user_{tag}"
    outd = ROOT / "data" / f"longmem_days_{tag}"
    outd.mkdir(parents=True, exist_ok=True)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "isbid": pl.Boolean,
                                 "pidx": pl.Int32, "wid": pl.Int32,
                                 "size_after": pl.Int64, "ts_last": pl.Int64})
    k0 = 0
    for i, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        if all((outd / f"{w}_{dt}.parquet").exists() for w in ("daily", "carry")):
            k0 = i + 1
        else:
            break
    if k0:
        carry = pl.read_parquet(outd / f"carry_{files[k0-1].stem.split('=')[1]}.parquet")
        print(f"[再開] {k0} 日ぶんを飛ばす", file=sys.stderr)

    rng = np.random.default_rng(SEED)
    for fp in files[k0:]:
        dt = fp.stem.split("=")[1]
        tw = time.time()
        bd = bbo.filter(pl.col("dt") == dt).sort("ts")
        wf = wdir / f"dt={dt}.parquet"
        if bd.is_empty() or not wf.exists():
            print(f"  {dt} bbo か口座情報が無い。飛ばす", file=sys.stderr)
            continue
        t0 = (int(pl.scan_parquet(fp).select(pl.col("ts").cast(pl.Int64).min())
                  .collect().item()) // DAY_NS) * DAY_NS
        if carry.height:
            carry = carry.filter(t0 - pl.col("ts_last") <= MAX_REST_NS)

        # ---- bbo 由来の 3 系列 ------------------------------------------------
        bts = bd["ts"].cast(pl.Int64).to_numpy()
        pb, pa = bd["best_bid"].to_numpy(), bd["best_ask"].to_numpy()
        qb, qa = bd["bid_sz"].to_numpy(), bd["ask_sz"].to_numpy()
        sec = np.clip((bts - t0) // SEC_NS, 0, NG - 1).astype(np.int64)
        cnt = np.bincount(sec, minlength=NG).astype(float)
        nz = np.maximum(cnt, 1)
        S = {}
        S["depth"] = np.bincount(sec, weights=qb + qa, minlength=NG) / nz
        S["spread"] = (np.bincount(sec, weights=(pa - pb) / ((pa + pb) / 2) * 1e4,
                                   minlength=NG) / nz)
        # OFI(Cont, Kukanov and Stoikov 2014)
        e = np.zeros(len(bts))
        e[1:] = ((pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
                 - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1])
        S["ofi"] = np.bincount(sec, weights=e, minlength=NG)

        # ---- l1 由来の 5 系列 -------------------------------------------------
        d = pl.read_parquet(fp, columns=["ts", "oid", "status", "orig_sz",
                                         "remaining_sz", "tif", "is_trigger"])
        d = d.with_columns(ts=pl.col("ts").cast(pl.Int64))
        op = d.filter(pl.col("status") == "open")
        tm = d.filter(pl.col("status").is_in(CANCELS)
                      | ((pl.col("status") == "filled")
                         & (pl.col("remaining_sz") <= 0)))
        so = np.clip((op["ts"].to_numpy() - t0) // SEC_NS, 0, NG - 1).astype(np.int64)
        st = np.clip((tm["ts"].to_numpy() - t0) // SEC_NS, 0, NG - 1).astype(np.int64)
        oz = op["orig_sz"].to_numpy()
        S["arrivals"] = np.bincount(so, minlength=NG).astype(float)
        S["cancels"] = np.bincount(st, minlength=NG).astype(float)
        ssum = np.bincount(so, weights=oz, minlength=NG)
        S["order_size"] = np.divide(ssum, np.maximum(S["arrivals"], 1),
                                    out=np.zeros(NG), where=S["arrivals"] > 0)
        # 件数 0 の秒は前の値を持ち越す(全体の 1% 未満)
        idx = np.where(S["arrivals"] > 0, np.arange(NG), 0)
        np.maximum.accumulate(idx, out=idx)
        S["order_size"] = S["order_size"][idx]
        S["churn"] = (ssum + np.bincount(st, weights=tm["remaining_sz"].to_numpy(),
                                         minlength=NG))
        u = pl.read_parquet(wf).unique(subset=["oid"])
        m = (d.select("ts", "oid").join(u, on="oid", how="inner"))
        sw = np.clip((m["ts"].to_numpy() - t0) // SEC_NS, 0, NG - 1).astype(np.int64)
        wv = m["wid"].to_numpy().astype(np.int64)
        o = np.lexsort((wv, sw))
        sw2, wv2 = sw[o], wv[o]
        first = np.concatenate([[True], (sw2[1:] != sw2[:-1]) | (wv2[1:] != wv2[:-1])])
        S["wallet_act"] = np.bincount(sw2[first], minlength=NG).astype(float)
        del d, op, tm, m

        # ---- queue size(最良にある注文の本数)--------------------------------
        seg, carry = day_lines(fp, wf, carry, t0)
        sp = seg["pidx"].to_numpy().astype(np.int64)
        sb = seg["isbid"].to_numpy()
        sa_, sbb = seg["a"].to_numpy(), seg["b"].to_numpy()
        del seg
        qcnt = np.zeros(NG + 1)
        for side in (True, False):
            px = np.rint((pb if side else pa) / PX_UNIT).astype(np.int64)
            ch = np.flatnonzero(np.diff(px) != 0) + 1
            rs = np.concatenate([[0], ch])
            re_ = np.concatenate([ch, [len(px) - 1]])
            r_s, r_e, r_p = bts[rs], bts[re_], px[rs]
            ok = r_e > r_s
            r_s, r_e, r_p = r_s[ok], r_e[ok], r_p[ok]
            msk = sb == side
            lp, la, lb = sp[msk], sa_[msk], sbb[msk]
            o2 = np.argsort(lp, kind="stable")
            lp, la, lb = lp[o2], la[o2], lb[o2]
            ro = np.argsort(r_p, kind="stable")
            rp_s, rs_s, re_s = r_p[ro], r_s[ro], r_e[ro]
            up, first_i = np.unique(rp_s, return_index=True)
            last_i = np.append(first_i[1:], len(rp_s))
            lf = np.searchsorted(lp, up, side="left")
            ll = np.searchsorted(lp, up, side="right")
            for k in range(len(up)):
                i0, i1 = lf[k], ll[k]
                if i1 <= i0:
                    continue
                j0, j1 = first_i[k], last_i[k]
                rr_s, rr_e = rs_s[j0:j1], re_s[j0:j1]
                lo = np.searchsorted(rr_e, la[i0:i1], side="right")
                hi = np.searchsorted(rr_s, lb[i0:i1], side="left")
                c = np.maximum(hi - lo, 0)
                if c.sum() == 0:
                    continue
                selm = c > 0
                idx2 = np.repeat(np.arange(i0, i1)[selm], c[selm])
                base = np.repeat(lo[selm], c[selm])
                off = np.arange(c[selm].sum()) - np.repeat(
                    np.cumsum(c[selm]) - c[selm], c[selm])
                rj = j0 + base + off
                ca = np.maximum(la[idx2], rs_s[rj])
                cb = np.minimum(lb[idx2], re_s[rj])
                g = cb > ca
                if not g.any():
                    continue
                # 区間を 1 秒格子の本数へ(開始で +1、終了で −1 して累積)
                gs = np.clip((ca[g] - t0) // SEC_NS, 0, NG - 1).astype(np.int64)
                ge = np.clip((cb[g] - t0) // SEC_NS + 1, 0, NG).astype(np.int64)
                np.add.at(qcnt, gs, 1.0)
                np.add.at(qcnt, ge, -1.0)
        S["queue"] = np.cumsum(qcnt)[:NG]

        # ---- 推定 --------------------------------------------------------------
        rows, crows = [], []
        for nm in SERIES:
            x = np.asarray(S[nm], float)
            variants = {"生": x}
            if nm in ("ofi",):
                pass
            variants["帰無対照"] = rng.permutation(x)
            for kind, xv in variants.items():
                rec, cur = estimate(xv, rng)
                if not rec:
                    continue
                rows.append({"dt": dt, "series": nm, "kind": kind, **rec})
                if kind == "生":
                    for cn, (xx, yy) in cur.items():
                        for i2 in range(len(xx)):
                            crows.append({"dt": dt, "series": nm, "curve": cn,
                                          "x": float(xx[i2]), "v": float(yy[i2])})
        pl.DataFrame(rows).write_parquet(outd / f"daily_{dt}.parquet")
        pl.DataFrame(crows).write_parquet(outd / f"curves_{dt}.parquet")
        carry.write_parquet(outd / f"carry_{dt}.parquet")
        # ★日内周期の除去は全日の時刻プロファイルが要るので、系列を残して 2 巡目で行う
        pl.DataFrame({k: S[k].astype(np.float32) for k in SERIES}).write_parquet(
            outd / f"series_{dt}.parquet")
        r0 = [r for r in rows if r["series"] == "ofi" and r["kind"] == "生"]
        print(f"  {dt} OFI: H {r0[0]['hurst']:.3f} DFA {r0[0]['dfa']:.3f} "
              f"GPH {r0[0]['gph_d']:.3f} LW {r0[0]['lw_d']:.3f} "
              f"ρ1 {r0[0]['acf1']:.3f} {time.time()-tw:.1f}s",
              file=sys.stderr, flush=True) if r0 else None

    # ---- 2 巡目: 日内周期を除いた系列で同じ推定をかける ----------------------
    sf = sorted(outd.glob("series_*.parquet"))
    if sf and not (outd / "_deseason_done").exists():
        print(f"[2 巡目] {len(sf)} 日から時刻プロファイルを作る", file=sys.stderr)
        prof = np.zeros((NG, len(SERIES)))
        for f in sf:
            prof += pl.read_parquet(f).to_numpy()
        prof /= len(sf)
        for f in sf:
            dt = f.stem.split("_")[1]
            X = pl.read_parquet(f).to_numpy() - prof
            rows = []
            for i2, nm in enumerate(SERIES):
                rec, _ = estimate(X[:, i2], rng)
                if rec:
                    rows.append({"dt": dt, "series": nm, "kind": "周期除去", **rec})
            if rows:
                old = pl.read_parquet(outd / f"daily_{dt}.parquet")
                pl.concat([old, pl.DataFrame(rows)],
                          how="diagonal_relaxed").write_parquet(
                    outd / f"daily_{dt}.parquet")
        (outd / "_deseason_done").write_text("ok", encoding="utf-8")
        pl.DataFrame({k: prof[:, i].astype(np.float32)
                      for i, k in enumerate(SERIES)}).write_parquet(
            ROOT / "data" / f"longmem_profile_{tag}.parquet")

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    D = cat("daily")
    D.write_parquet(ROOT / "data" / f"longmem_daily_{tag}.parquet")
    D.write_csv(ROOT / "data" / f"longmem_daily_{tag}.csv")
    cat("curves").write_parquet(ROOT / "data" / f"longmem_curves_{tag}.parquet")
    if sf:
        pl.read_parquet(sf[len(sf) // 2]).write_parquet(
            ROOT / "data" / f"longmem_sample_{tag}.parquet")
    print(f"\n[集計] 行 {D.height:,}", file=sys.stderr)
    print(f"-> data/longmem_daily_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
