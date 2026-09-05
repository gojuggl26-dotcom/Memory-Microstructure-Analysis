"""絞った標本は時間的に固まっているか (burstiness) と、自己相関の測定。

    uv run python scripts/build_burst.py --coin xyz:MU

なぜ測るか
----------
Q1 / Q5 の観測が時系列上に散らばっていれば「独立に近い 170 万標本」だが、
連続した塊で採れているなら実質の標本数ははるかに少なく、
プールした標準誤差は嘘になる。リターン自体にも正の自己相関がある
(1 秒刻みで rho1 = +0.10、mu_vol 報告)ので、両方を測って
「どの時間幅までまとめれば標準誤差が落ち着くか」を出す。

測るもの
--------
1. 特徴量とリターンの自己相関 (1 秒格子、ラグ 1〜1800 秒)
2. Q1 / Q5 に入り続ける連 (run) の長さ。独立なら幾何分布になる
3. 間隔の burstiness B = (sd - mean)/(sd + mean) と記憶 M = corr(tau_i, tau_i+1)
   (Goh & Barabasi)。B=0 がポアソン、B>0 が塊、B<0 が規則的
4. Fano 因子 = 窓あたり件数の 分散/平均。ポアソンなら窓幅によらず 1
5. 日・時刻への集中 (Lorenz と Gini)
6. **ブロック頑健な標準誤差**: ブロック幅 w を 1 秒から 1 日まで動かし、
   Q5-Q1 の標準誤差がどこで頭打ちになるかを見る。これが正直な誤差である

時間契約
--------
ここは既存の出力を診断するだけで、新しい予測は作らない。特徴量と
リターンの作り方は build_pred.day_features を**そのまま**使う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_pred import (GRID_N, day_features, quintile, usable)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAXLAG = 1800
FEATS = ("obi", "ofi_10s", "ai_net_10s")
HS = (1.0, 10.0)
BLOCKS = (1, 5, 15, 60, 300, 900, 3600, 14400, 86400)
RUN_EDGES = np.unique(np.round(10.0 ** np.arange(0, 4.01, 0.125)).astype(int))


def acf_day(x: np.ndarray, maxlag: int) -> tuple[np.ndarray, float, int]:
    """1 日ぶんの自己共分散 (ラグ 0..maxlag) と分散と有効長を返す。

    欠損は平均を引いた後 0 で埋める。先頭の数点しか無いので影響は無視できる。
    """
    m = np.isfinite(x)
    n = int(m.sum())
    if n < maxlag * 2:
        return np.zeros(maxlag + 1), 0.0, 0
    y = np.where(m, x - x[m].mean(), 0.0)
    nfft = 1 << int(np.ceil(np.log2(2 * y.size)))
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: maxlag + 1].real / n
    return ac, float(ac[0]), n


def burst_memory(gap: np.ndarray) -> tuple[float, float]:
    if gap.size < 3:
        return np.nan, np.nan
    mu, sd = float(gap.mean()), float(gap.std(ddof=1))
    b = (sd - mu) / (sd + mu) if (sd + mu) > 0 else np.nan
    m = float(np.corrcoef(gap[:-1], gap[1:])[0, 1]) if gap.std() > 0 else np.nan
    return b, m


def gini(v: np.ndarray) -> float:
    v = np.sort(v[v >= 0].astype(float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return np.nan
    return float((2.0 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    # Q5 / Q1 の全体平均は既存の出力から取る (定義のずれを防ぐ)
    C = pl.read_parquet(DATA / f"pred_cells_{tag}.parquet")
    gm = (C.filter(pl.col("q").is_in([0, 4]))
          .group_by("feat", "h", "q")
          .agg(n=pl.col("n").sum(), s=pl.col("sum_r").sum())
          .with_columns(m=pl.col("s") / pl.col("n")))
    G = {(r["feat"], r["h"], r["q"]): (r["m"], r["n"])
         for r in gm.iter_rows(named=True)}

    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    days = sorted(bb["dt"].unique().to_list())
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 行を除外) / {len(days)} 日", flush=True)

    ser = list(FEATS) + ["ret_1s", "absret_1s"]
    acc = {s: [np.zeros(MAXLAG + 1), 0.0, 0] for s in ser}
    runs = {(f, q): np.zeros(RUN_EDGES.size + 1, dtype=np.int64)
            for f in FEATS for q in (0, 4)}
    gaps = {(f, q): [] for f in FEATS for q in (0, 4)}
    fano = {(f, q, w): [0.0, 0.0, 0] for f in FEATS for q in (0, 4)
            for w in BLOCKS if w <= 3600}
    blk = {(f, h, w): 0.0 for f in FEATS for h in HS for w in BLOCKS}
    cnt_day = {(f, q): [] for f in FEATS for q in (0, 4)}
    cnt_hour = {(f, q): np.zeros(24) for f in FEATS for q in (0, 4)}
    dq = {(f, h): [] for f in FEATS for h in HS}      # 日ごとの Q5-Q1
    prev_edges: dict[str, np.ndarray] = {}
    nday = 0

    for k, dt in enumerate(days):
        D = day_features(bb.filter(pl.col("dt") == dt).sort("ts"), FID.get(dt))
        if D is None:
            continue
        F, R = D["F"], D["R"]
        for f in FEATS:
            a1, v1, n1 = acf_day(F[f], MAXLAG)
            acc[f][0] += a1 * n1
            acc[f][1] += v1 * n1
            acc[f][2] += n1
        for nm, x in (("ret_1s", R[1.0]), ("absret_1s", np.abs(R[1.0]))):
            a1, v1, n1 = acf_day(x, MAXLAG)
            acc[nm][0] += a1 * n1
            acc[nm][1] += v1 * n1
            acc[nm][2] += n1

        pe_ok = all(usable(prev_edges.get(f)) for f in FEATS)
        if pe_ok:
            nday += 1
            hh = np.clip(np.arange(GRID_N + 1) // 3600, 0, 23)
            for f in FEATS:
                qq = quintile(F[f], prev_edges[f])
                good = np.isfinite(F[f])
                for q in (0, 4):
                    sel = good & (qq == q)
                    cnt_day[(f, q)].append(int(sel.sum()))
                    np.add.at(cnt_hour[(f, q)], hh[sel], 1.0)
                    # 連 (run) の長さ
                    d1 = np.diff(np.concatenate([[0], sel.view(np.int8), [0]]))
                    st = np.flatnonzero(d1 == 1)
                    en = np.flatnonzero(d1 == -1)
                    rl = en - st
                    if rl.size:
                        runs[(f, q)] += np.bincount(
                            np.searchsorted(RUN_EDGES, rl, side="right"),
                            minlength=RUN_EDGES.size + 1)
                        gaps[(f, q)].append(np.diff(st))
                    for w in BLOCKS:
                        if w > 3600:
                            continue
                        c = sel[:GRID_N].reshape(-1, w).sum(1).astype(float)
                        fano[(f, q, w)][0] += c.sum()
                        fano[(f, q, w)][1] += (c * c).sum()
                        fano[(f, q, w)][2] += c.size
                # ブロック頑健な標準誤差のための積み上げ
                for h in HS:
                    st_ = max(1, int(h))
                    r = R[h]
                    m5, n5 = G.get((f, h, 4), (np.nan, 0))
                    m1, n1_ = G.get((f, h, 0), (np.nan, 0))
                    fin = np.isfinite(r) & good
                    z = np.zeros(GRID_N + 1)
                    s5 = fin & (qq == 4)
                    s1 = fin & (qq == 0)
                    z[s5] = (r[s5] - m5) / n5
                    z[s1] = -(r[s1] - m1) / n1_
                    z[np.arange(GRID_N + 1) % st_ != 0] = 0.0   # 歩幅の外は使わない
                    dq[(f, h)].append(float(z.sum()))
                    for w in BLOCKS:
                        bs = z[:GRID_N].reshape(-1, w).sum(1)
                        blk[(f, h, w)] += float((bs * bs).sum())
        prev_edges = {f: np.nanquantile(F[f][np.isfinite(F[f])],
                                        [0.2, 0.4, 0.6, 0.8])
                      for f in FEATS if np.isfinite(F[f]).any()}
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True)

    rows = []
    for s in ser:
        ac, v, n = acc[s]
        if n == 0:
            continue
        r = ac / max(v, 1e-300)
        for lg in range(1, MAXLAG + 1):
            rows.append({"series": s, "lag_s": lg, "acf": float(r[lg])})
    pl.DataFrame(rows).write_csv(DATA / f"burst_acf_{tag}.csv")

    rr = []
    for (f, q), v in runs.items():
        for i, c in enumerate(v):
            if c:
                rr.append({"feat": f, "q": q, "bin": i,
                           "lo": int(RUN_EDGES[i - 1]) if i else 0,
                           "hi": int(RUN_EDGES[i]) if i < RUN_EDGES.size else -1,
                           "count": int(c)})
    pl.DataFrame(rr).write_csv(DATA / f"burst_run_{tag}.csv")

    sm = []
    for f in FEATS:
        for q in (0, 4):
            g = np.concatenate(gaps[(f, q)]) if gaps[(f, q)] else np.array([])
            b, mem = burst_memory(g.astype(float))
            cd = np.array(cnt_day[(f, q)], dtype=float)
            row = {"feat": f, "q": q, "burstiness": b, "memory": mem,
                   "n_sel": int(cd.sum()), "gini_day": gini(cd),
                   "gini_hour": gini(cnt_hour[(f, q)]),
                   "top10day_share": float(np.sort(cd)[::-1][:10].sum() / cd.sum())
                   if cd.sum() else np.nan}
            for w in BLOCKS:
                if w <= 3600:
                    s_, s2, nb = fano[(f, q, w)]
                    mu = s_ / nb
                    row[f"fano_{w}s"] = float((s2 / nb - mu * mu) / mu) if mu > 0 else np.nan
            sm.append(row)
    pl.DataFrame(sm).write_csv(DATA / f"burst_summary_{tag}.csv")

    br = []
    for f in FEATS:
        for h in HS:
            d = np.array(dq[(f, h)])
            for w in BLOCKS:
                br.append({"feat": f, "h": h, "block_s": w,
                           "se": float(np.sqrt(blk[(f, h, w)]))})
            br.append({"feat": f, "h": h, "block_s": 86400 * 2,
                       "se": float(np.sqrt((d * d).sum()))})
    pl.DataFrame(br).write_csv(DATA / f"burst_block_{tag}.csv")
    print(f"書き出し: burst_acf / burst_run / burst_summary / burst_block ({nday} 日)")


if __name__ == "__main__":
    main()
