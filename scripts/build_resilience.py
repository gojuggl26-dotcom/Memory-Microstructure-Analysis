"""板の弾力性(resilience)— 流動性ショックからの回復を 12 の指標で測る。

【模型】
ショック直後の depth を D_shock、ショック前の水準を D0 として

    D(t) = D0 + (D_shock − D0) e^{−κ t}

を想定する。正規化した回復度

    R(t) = (D(t) − D_shock) / (D0 − D_shock) = 1 − e^{−κ t}

は 0 から 1 へ向かう。R(t_half) = 1/2 となる時刻から κ = ln 2 / t_half が出る。
**最小二乗で当てずに半減時刻から κ を出す**のは、1 件ごとの経路が階段関数で
雑音が大きく、当てはめが発散しやすいため。集計した平均経路には別途
指数関数を当てて、模型そのものの妥当性を確認する。

【ショックの定義(各時点で判定できる)】
片側の最良気配の数量が 1 イベントで **DROP 以上失われた**ときをショックとする。

    (q_{i−1} − q_i) / q_{i−1} ≥ DROP  かつ  q_{i−1} ≥ MINQ  かつ  D0 ≥ MINQ

D0 は**直前 REF_MS の時間加重平均**(瞬時値は雑音が大きい)。
同じ側で REFRACTORY_MS 以内に続く検出は 1 件にまとめる(同じ事象の重複計上を防ぐ)。

【12 の指標】
    depth recovery rate     κ_D = ln2 / t_half(depth)                [1/s]
    spread recovery rate    κ_S = ln2 / t_half(spread)               [1/s]
    BBO replenishment rate  κ_P = ln2 / t_half(最良気配の価格が戻る)  [1/s]
    queue replenishment rate 最良気配の価格が動かなかった場合の数量の戻り速度 [枚/s]
    half-life of recovery   t_half(depth)                            [ms]
    refill latency          数量が少しでも戻るまでの時間              [ms]
    refill size             D(T) − D_shock  (T = 1 秒)               [枚]
    refill intensity        refill size / T                          [枚/s]
    refill probability      T までに D0 まで戻ったか                  [0/1]
    bid resilience          買い側のショックに対する κ_D
    ask resilience          売り側のショックに対する κ_D
    resilience asymmetry    (κ_bid − κ_ask) / (κ_bid + κ_ask)  ※日ごとの集計量

【x が確定する時刻 / y の期間】
すべての指標は区間 [t0, t0 + T] の中で決まる(T = 1 秒)。したがって

    x が確定する時刻 = t0 + T
    y の期間        = (t0 + T,  t0 + T + h]

とする。latency のように早く確定する指標も、**一律に t0 + T を起点**にして
先読みの疑いを残さない。`shift(-k)` は y にしか使っていない。

【★片側に紐づいた量は符号の情報を持つ】
買い側のショックと売り側のショックは別の事象なので、side を必ず分けて集計する
(100ms 窓の分散のレポートで、片側の量が符号つきリターンと鏡像の関係を作ることを
実測している)。

    uv run python scripts/build_resilience.py --coin xyz:MU
出力: data/resil_<coin>/dt=*.parquet      … ショック 1 件ごとの指標
      data/resil_path_<coin>.parquet      … 平均回復経路(図示・指数当てはめ用)
      data/resil_ols_<coin>.csv           … 指標 × ホライズンの OLS
      data/resil_trans_<coin>.csv         … 指標の帯 × ホライズンの上昇確率
      data/resil_daily_<coin>.csv         … 日ごとの κ_bid / κ_ask / 非対称
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000                       # 1ms を ns で
DROP = 0.5                           # 最良気配の数量がこの割合以上失われたらショック
MINQ = 1.0                           # 小さすぎる板は対象外(枚)
REF_MS = 1_000                       # ショック前の水準を測る窓
REFRACTORY_MS = 200                  # 同じ側でこの間隔以内の再検出はまとめる
T_MS = 1_000                         # 指標を測り切る時刻 T
# 回復経路を評価する時刻(ms)。対数的に散らす
TAU = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300,
                500, 750, 1000, 1500, 2000, 3000, 5000], dtype=np.int64)
IT = int(np.where(TAU == T_MS)[0][0])          # T = 1000ms の位置
HOR = {"10ms": 10, "20ms": 20, "50ms": 50, "100ms": 100, "500ms": 500,
       "1s": 1000, "3s": 3000, "5s": 5000, "10s": 10000, "30s": 30000,
       "50s": 50000, "100s": 100000}
CHUNK = 7

# 回帰と遷移行列にかける指標。帯の切れ目は分布を見る前に機械的に決めた
METRICS = {
    "kappa_depth":      ([0.5, 1, 2, 5, 10, 20], "1/s"),
    "kappa_spread":     ([0.5, 1, 2, 5, 10, 20], "1/s"),
    "kappa_px":         ([0.5, 1, 2, 5, 10, 20], "1/s"),
    "t_half_ms":        ([5, 10, 25, 50, 100, 250], "ms"),
    "latency_ms":       ([1, 2, 5, 10, 25, 50], "ms"),
    "refill_size":      ([1, 5, 20, 100, 500, 2000], "枚"),
    "refill_intensity": ([1, 5, 20, 100, 500, 2000], "枚/s"),
    "queue_repl_rate":  ([1, 5, 20, 100, 500, 2000], "枚/s"),
    "refill_prob":      ([0.5], "0/1"),
    "r_1s":             ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], "—"),
}


def read_chunk(coin: str, days: list[str]):
    p = ROOT / "data" / f"bbo_{coin.replace(':', '_')}.parquet"
    d = pl.scan_parquet(p).filter(pl.col("dt").is_in(days)).collect().sort("ts")
    n0 = d.height
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    return d, n0 - d.height


def twa_prep(ts: np.ndarray, v: np.ndarray) -> np.ndarray:
    """時間加重積分の累積和。C[i] = Σ_{j<i} v[j]·(ts[j+1] − ts[j])。"""
    dt = np.diff(ts).astype(np.float64)
    return np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])


def twa(ts, C, v, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """区間 [a, b) の時間加重平均。a/b は時刻の配列。"""
    ia = np.searchsorted(ts, a, side="right") - 1
    ib = np.searchsorted(ts, b, side="right") - 1
    ia = np.clip(ia, 0, len(ts) - 1)
    ib = np.clip(ib, 0, len(ts) - 1)
    tot = (C[ib] - C[ia]) + v[ib] * (b - ts[ib]) - v[ia] * (a - ts[ia])
    span = (b - a).astype(np.float64)
    return np.where(span > 0, tot / np.maximum(span, 1.0), np.nan)


def cross_time(tau: np.ndarray, R: np.ndarray, lvl: float) -> np.ndarray:
    """R が初めて lvl 以上になる時刻。対数時間で線形補間。届かなければ nan。"""
    ok = R >= lvl
    has = ok.any(axis=1)
    j = np.argmax(ok, axis=1)
    lt = np.log(tau.astype(np.float64))
    out = np.where(has, lt[j], np.nan)
    prev = j - 1
    ip = prev >= 0
    idx = np.arange(len(R))
    r1 = np.where(ip, R[idx, np.maximum(prev, 0)], np.nan)
    r2 = R[idx, j]
    w = np.where(np.isfinite(r1) & (r2 > r1), (lvl - r1) / (r2 - r1), 0.0)
    out = np.where(has & ip, lt[np.maximum(prev, 0)] + w * (lt[j] - lt[np.maximum(prev, 0)]),
                   out)
    return np.exp(out)


DROP_EDGES = [0.6, 0.75, 0.9, 0.97]      # ショックの大きさの帯(D0 に対する落差)
DROP_LAB = ["0.50–0.60", "0.60–0.75", "0.75–0.90", "0.90–0.97", "0.97–1.00"]


def day_shocks(day: str, e: pl.DataFrame, paths: list) -> pl.DataFrame:
    ts = e["ts"].to_numpy()
    if len(ts) < 100:
        return pl.DataFrame()
    bq, aq = e["bid_sz"].to_numpy(), e["ask_sz"].to_numpy()
    bp, ap = e["best_bid"].to_numpy(), e["best_ask"].to_numpy()
    mid = (bp + ap) / 2.0
    spr = (ap - bp) / mid * 1e4                      # bp 単位
    Cb, Ca, Cs = twa_prep(ts, bq), twa_prep(ts, aq), twa_prep(ts, spr)
    out = []

    for side, q, px, C in (("買い", bq, bp, Cb), ("売り", aq, ap, Ca)):
        prev = np.concatenate([[np.nan], q[:-1]])
        drop = np.where(prev > 0, (prev - q) / prev, 0.0)
        cand = np.flatnonzero((drop >= DROP) & (prev >= MINQ))
        if len(cand) == 0:
            continue
        # 不応期: 同じ側で近接する検出は先頭だけ残す
        keep, last = [], -np.inf
        for i in cand:
            if ts[i] - last >= REFRACTORY_MS * MS:
                keep.append(i)
                last = ts[i]
        k = np.asarray(keep)
        t0 = ts[k]

        D0 = twa(ts, C, q, t0 - REF_MS * MS, t0)
        # ★D0 に対する落差も DROP 以上でなければ正規化の分母が潰れる
        #   (これを入れないと R が ±1 万を超える行が出る。実際に踏んだ)
        good = np.isfinite(D0) & (D0 >= MINQ) & ((D0 - q[k]) / D0 >= DROP)
        k, t0, D0 = k[good], t0[good], D0[good]
        if len(k) == 0:
            continue
        Dsh = q[k]
        S0 = twa(ts, Cs, spr, t0 - REF_MS * MS, t0)
        Ssh = spr[k]
        P0 = px[np.maximum(k - 1, 0)]                 # ショック前の最良価格
        gap0 = np.abs(px[k] - P0)

        # ---- 回復経路(searchsorted で厳密な時刻を引く)--------------------
        j = np.searchsorted(ts, t0[:, None] + TAU[None, :] * MS, side="right") - 1
        j = np.clip(j, 0, len(ts) - 1)
        Q = q[j]
        P = px[j]
        S = spr[j]
        den = (D0 - Dsh)[:, None]
        R = (Q - Dsh[:, None]) / den
        t_half = cross_time(TAU, R, 0.5)
        lat = cross_time(TAU, (Q > Dsh[:, None]).astype(float), 0.5)

        # スプレッド: 広がった場合だけ意味を持つ
        dS = (Ssh - S0)
        RS = np.where(dS[:, None] > 0, (Ssh[:, None] - S) / np.where(dS[:, None] > 0, dS[:, None], 1), np.nan)
        t_half_s = np.where(dS > 0, cross_time(TAU, np.nan_to_num(RS, nan=-1.0), 0.5), np.nan)

        # 最良気配の価格が動いた場合だけ意味を持つ
        RP = np.where(gap0[:, None] > 0,
                      1.0 - np.abs(P - P0[:, None]) / np.where(gap0[:, None] > 0, gap0[:, None], 1),
                      np.nan)
        t_half_p = np.where(gap0 > 0, cross_time(TAU, np.nan_to_num(RP, nan=-1.0), 0.5), np.nan)

        refill = Q[:, IT] - Dsh
        same_px = P[:, IT] == px[k]
        rec = np.nanmax(R[:, :IT + 1], axis=1)
        # 平均経路は外れ値に弱いので [-1, 3] に丸めてから足す(丸めた事実は報告する)
        db = np.digitize((D0 - Dsh) / D0, DROP_EDGES)
        paths.append((side, db, np.clip(R, -1.0, 3.0)))
        out.append(pl.DataFrame({
            "dt": [day] * len(k), "side": [side] * len(k), "ts": t0,
            "d0": D0, "d_shock": Dsh, "drop": (D0 - Dsh) / D0,
            "t_half_ms": t_half,
            "kappa_depth": np.log(2.0) / (t_half / 1e3),
            "kappa_spread": np.log(2.0) / (t_half_s / 1e3),
            "kappa_px": np.log(2.0) / (t_half_p / 1e3),
            "latency_ms": lat,
            "refill_size": refill,
            "refill_intensity": refill / (T_MS / 1e3),
            "queue_repl_rate": np.where(same_px, refill / (T_MS / 1e3), np.nan),
            "refill_prob": (rec >= 1.0).astype(np.float64),
            "r_1s": R[:, IT],
            "_idx_end": np.searchsorted(ts, t0 + T_MS * MS, side="right") - 1,
        }))
    if not out:
        return pl.DataFrame()
    return pl.concat(out).sort("ts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い")
    days = sorted(pl.scan_parquet(p).select("dt").unique().collect()["dt"].to_list())
    if a.days:
        days = days[: a.days]
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days}
    outdir = ROOT / "data" / f"resil_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    paths: list = []
    todo = [d for d in days if not (outdir / f"dt={d}.parquet").exists()]
    print(f"[再開] 未処理 {len(todo)} 日", file=sys.stderr)
    n_drop = 0
    for c0 in range(0, len(todo), CHUNK):
        part = todo[c0:c0 + CHUNK]
        E, nd = read_chunk(a.coin, part)
        n_drop += nd
        for day in part:
            e = E.filter(pl.col("dt") == day)
            S = day_shocks(day, e, paths)
            if S.is_empty():
                pl.DataFrame({"dt": [], "side": []}).write_parquet(
                    outdir / f"dt={day}.parquet")
                continue
            # ---- 将来 log リターン(x の確定時刻 t0+T を起点)------------------
            ts = e["ts"].to_numpy()
            lm = np.log((e["best_bid"].to_numpy() + e["best_ask"].to_numpy()) / 2.0)
            ie = S["_idx_end"].to_numpy()
            base = lm[ie]
            te = ts[ie]
            cols = {}
            for hn, hms in HOR.items():
                jj = np.searchsorted(ts, te + hms * MS, side="right") - 1
                ok = jj < len(ts) - 1
                cols[f"y_{hn}"] = np.where(ok, lm[np.clip(jj, 0, len(ts) - 1)] - base, np.nan)
            S = S.with_columns(**{k: pl.Series(v) for k, v in cols.items()},
                               day_type=pl.lit(dtype[day]))
            S.drop("_idx_end").write_parquet(outdir / f"dt={day}.parquet",
                                             compression="zstd")
        print(f"  {part[-1]}  ({c0 + len(part)}/{len(todo)} 日)", file=sys.stderr)

    if paths:
        rows = []
        for side, db, R in paths:
            for b in range(len(DROP_LAB)):
                m = db == b
                if m.sum() < 50:
                    continue
                rows.append(pl.DataFrame({
                    "side": [side] * len(TAU), "drop_bin": [DROP_LAB[b]] * len(TAU),
                    "tau_ms": TAU, "n": [int(m.sum())] * len(TAU),
                    "sum_r": np.nansum(R[m], axis=0),
                    "cnt_r": np.isfinite(R[m]).sum(axis=0)}))
        P = (pl.concat(rows).group_by("side", "drop_bin", "tau_ms")
               .agg(pl.col("n").sum(), pl.col("sum_r").sum(), pl.col("cnt_r").sum())
               .with_columns(r_mean=pl.col("sum_r") / pl.col("cnt_r"))
               .sort("side", "drop_bin", "tau_ms"))
        pp = ROOT / "data" / f"resil_path_{tag}.parquet"
        if pp.exists():           # 再開時は既存分と足す
            P = (pl.concat([pl.read_parquet(pp).drop("r_mean"), P.drop("r_mean")])
                   .group_by("side", "drop_bin", "tau_ms")
                   .agg(pl.col("n").sum(), pl.col("sum_r").sum(), pl.col("cnt_r").sum())
                   .with_columns(r_mean=pl.col("sum_r") / pl.col("cnt_r"))
                   .sort("side", "drop_bin", "tau_ms"))
        P.write_parquet(pp)
        print(f"-> {pp}", file=sys.stderr)
    print(f"[除去] クロス・数量 0 の行 {n_drop:,}", file=sys.stderr)
    print(f"-> {outdir}/dt=*.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
