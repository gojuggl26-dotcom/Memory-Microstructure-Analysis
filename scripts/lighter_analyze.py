r"""Lighter 特徴量の一括解析(第3段)— 分布・予測力・十分位・相関行列・重複。

【集計の単位】銘柄ごとに推定 → 銘柄をまたいで中央値・符号検定(プールしない)。
【主統計】Spearman ρ(分位境界を使わないので分位のルックアヘッドが原理的に無い)。
【帰無対照】x を銘柄内で巡回シフトして同じ統計を取る。
【y】y_h = (mid(t+h) − mid(t)) / mid(t) × 1e4 [bp]。mid はブック再構成の mid。
    suspect / crossed / book 無効の秒は x からも y からも除く。
    t+h の値は backward asof(その秒が存在しない場合は 2 秒まで手前を許す)。

出力(E:/Memory-lighter/ana/):
  desc_{SYM}.parquet   分布統計
  rho_{SYM}.parquet    Spearman ρ + 帰無、単回帰 β(記述用)
  dec_{SYM}.parquet    十分位ごとの平均・中央値(h=60s、形を見る記述統計)
  corr_{SYM}.npz       特徴量×特徴量の Spearman 行列(間引き標本)
  dup_{SYM}.parquet    |ρ|>0.999 のペア
  samp_{SYM}.parquet   散布図・横断検査用の抽出(2,000 行)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_features import build  # noqa: E402

SRC = Path("E:/Memory-lighter")
OUT = SRC / "ana"
HORS = (10.0, 60.0, 300.0)
NDEC = 10
SAMP = 2000
BP = 1e4


def rank(v: np.ndarray) -> np.ndarray:
    n = len(v)
    o = np.argsort(v, kind="stable")
    r = np.empty(n)
    r[o] = np.arange(1, n + 1)
    s = v[o]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j > i + 1:
            r[o[i:j]] = (i + 1 + j) / 2.0
        i = j
    return r



try:                                     # ★同順位処理は scipy の C 実装を使う
    from scipy.stats import rankdata as _rd

    def rank(v):
        return _rd(v, method="average")
except Exception:
    pass


def corr(x, y):
    if len(x) < 100:
        return np.nan
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def run_symbol(sym: str, rng) -> None:
    d = pl.read_parquet(SRC / f"grid_{sym}.parquet")
    FT = build(d)
    ts = FT["ts_s"].to_numpy().astype(np.float64)
    bad = FT["_bad"].to_numpy()
    mid = FT["_mid"].to_numpy()
    mid = np.where(bad, np.nan, mid)
    cols = [c for c in FT.columns if not c.startswith("_") and c != "ts_s"]
    n = len(ts)

    # --- 目的変数(backward asof) ---
    Y = {}
    for h in HORS:
        j = np.searchsorted(ts, ts + h, side="right") - 1
        ok = (ts[j] >= ts + h - 2) & (ts[j] > ts) & np.isfinite(mid[j]) & np.isfinite(mid)
        Y[h] = np.where(ok, (mid[j] - mid) / mid * BP, np.nan)
    sh = int(rng.integers(n // 5, 4 * n // 5))

    desc, rho, dec = [], [], []
    X = np.empty((n, len(cols)), dtype=np.float64)
    for ci, c in enumerate(cols):
        x = FT[c].to_numpy().astype(np.float64)
        x = np.where(bad, np.nan, x)
        X[:, ci] = x
        fin = np.isfinite(x)
        nv = int(fin.sum())
        row = {"symbol": sym, "feature": c, "n": n, "n_valid": nv,
               "valid_frac": nv / n}
        if nv >= 200:
            v = x[fin]
            q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
            row.update({"mean": float(v.mean()), "sd": float(v.std()),
                        "p1": q[0], "p5": q[1], "p25": q[2], "p50": q[3],
                        "p75": q[4], "p95": q[5], "p99": q[6],
                        "zero_frac": float((v == 0).mean())})
        desc.append(row)
        if nv < 5000 or (nv and np.nanstd(x[fin]) == 0):
            continue
        xs = np.roll(x, sh)
        iqr = np.subtract(*np.percentile(x[fin], [75, 25]))
        for h in HORS:
            m = fin & np.isfinite(Y[h])
            if m.sum() < 3000:
                continue
            xx, yy = x[m], Y[h][m]
            rx, ry = rank(xx), rank(yy)
            ms = np.isfinite(xs) & np.isfinite(Y[h])
            # 記述用 β: y[bp] を x の IQR あたりに直した傾き(全標本 IQR — 記述統計)
            beta = (np.cov(xx, yy)[0, 1] / max(np.var(xx), 1e-18) * iqr
                    if iqr > 0 else np.nan)
            rho.append({
                "symbol": sym, "feature": c, "h": h, "n": int(m.sum()),
                "spearman": corr(rx, ry), "pearson": corr(xx, yy),
                "beta_iqr_bp": float(beta),
                "spearman_pl": corr(rank(xs[ms]), rank(Y[h][ms]))
                if ms.sum() >= 3000 else np.nan,
            })
            if h != 60.0:
                continue
            e = np.quantile(xx, np.linspace(0, 1, NDEC + 1)[1:-1])
            if len(np.unique(e)) < NDEC - 1:
                continue
            qi = np.searchsorted(e, xx, side="right")
            for k in range(NDEC):
                s_ = qi == k
                if s_.sum() < 100:
                    continue
                dec.append({"symbol": sym, "feature": c, "dec": k + 1,
                            "n": int(s_.sum()), "x_mean": float(xx[s_].mean()),
                            "y_mean": float(yy[s_].mean()),
                            "y_med": float(np.median(yy[s_]))})
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(desc).write_parquet(OUT / f"desc_{sym}.parquet")
    pl.DataFrame(rho).write_parquet(OUT / f"rho_{sym}.parquet")
    pl.DataFrame(dec).write_parquet(OUT / f"dec_{sym}.parquet")

    # --- 特徴量×特徴量 Spearman 行列(間引き標本・有効率 50% 以上の列)---
    keep = [ci for ci, c in enumerate(cols)
            if np.isfinite(X[:, ci]).mean() > 0.5]
    sub = X[::3][:, keep]
    R = np.full((len(keep), len(keep)), np.nan, dtype=np.float32)
    ranks = np.empty_like(sub)
    okc = []
    for k2 in range(sub.shape[1]):
        v = sub[:, k2]
        f = np.isfinite(v)
        if f.mean() < 0.5:
            okc.append(False)
            continue
        okc.append(True)
        med = np.nanmedian(v)
        vv = np.where(f, v, med)
        ranks[:, k2] = rank(vv)
    okc = np.array(okc)
    idx = np.flatnonzero(okc)
    Z = ranks[:, idx]
    Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0), 1e-12)
    C = (Z.T @ Z) / len(Z)
    for i2, ii in enumerate(idx):
        R[ii, idx] = C[i2]
    np.savez_compressed(OUT / f"corr_{sym}.npz",
                        corr=R, features=np.array([cols[i] for i in keep]))
    dup = []
    iu = np.triu_indices(len(idx), 1)
    hit = np.flatnonzero(np.abs(C[iu]) > 0.999)
    for k2 in hit[:2000]:
        dup.append({"symbol": sym,
                    "a": cols[keep[idx[iu[0][k2]]]],
                    "b": cols[keep[idx[iu[1][k2]]]],
                    "corr": float(C[iu[0][k2], iu[1][k2]])})
    pl.DataFrame(dup, schema={"symbol": pl.String, "a": pl.String,
                              "b": pl.String, "corr": pl.Float64}
                 ).write_parquet(OUT / f"dup_{sym}.parquet")

    # --- 抽出標本 ---
    m = np.isfinite(Y[60.0]) & ~bad
    ii = np.flatnonzero(m)
    if len(ii) > SAMP:
        ii = np.sort(rng.choice(ii, SAMP, replace=False))
    S = FT[cols][ii.tolist()].with_columns([
        pl.lit(sym).alias("symbol"),
        pl.Series("ts_s", ts[ii].astype(np.int64)),
        pl.Series("y10", Y[10.0][ii].astype(np.float32)),
        pl.Series("y60", Y[60.0][ii].astype(np.float32)),
        pl.Series("y300", Y[300.0][ii].astype(np.float32)),
    ])
    S.write_parquet(OUT / f"samp_{sym}.parquet")
    print(f"{sym}: 行 {n:,} / 特徴量 {len(cols)} / 有効率>50% {len(keep)}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    for sym in a.symbols.split(","):
        run_symbol(sym, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
