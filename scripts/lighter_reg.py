r"""多変量回帰(リッジ)— 特徴量を全部入れて標本外でどこまで行くか。

【設計 — ルックアヘッド対策】
  ・時間分割: 各銘柄の前半 70% で学習、後半 30% で評価(行のシャッフルはしない)
  ・標準化(中央値・IQR)は訓練期間だけで推定
  ・リッジの λ は訓練の末尾 20% を検証にして選ぶ(評価期間は見ない)
  ・y は 60 秒後リターン[bp]を訓練期間の sd で割る(銘柄内標準化 —
    生のままだと値動きの大きい銘柄に引っぱられる。Boros 報告18 の教訓)
  ・y の窓は重ねない(60 秒間隔で間引く)
  ・帰無対照: x を銘柄内で巡回シフトして同じ手順

出力: E:/Memory-lighter/ana/reg_perf.parquet, reg_coef.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_features import build  # noqa: E402

SRC = Path("E:/Memory-lighter")
ANA = SRC / "ana"
BP = 1e4
H = 60.0
LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def ridge_fit(X, y, lam):
    G = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ y)


def run_symbol(sym: str, rng, perf, coefs):
    d = pl.read_parquet(SRC / f"grid_{sym}.parquet")
    FT = build(d)
    ts = FT["ts_s"].to_numpy().astype(np.float64)
    bad = FT["_bad"].to_numpy()
    mid = np.where(bad, np.nan, FT["_mid"].to_numpy())
    cols = [c for c in FT.columns if not c.startswith("_") and c != "ts_s"]
    j = np.searchsorted(ts, ts + H, side="right") - 1
    ok = (ts[j] >= ts + H - 2) & (ts[j] > ts) & np.isfinite(mid[j]) & np.isfinite(mid)
    y = np.where(ok, (mid[j] - mid) / mid * BP, np.nan)

    X = np.column_stack([FT[c].to_numpy().astype(np.float64) for c in cols])
    X[bad] = np.nan
    keep = [i for i in range(X.shape[1]) if np.isfinite(X[:, i]).mean() > 0.8]
    X = X[:, keep]
    names = [cols[i] for i in keep]

    # 60 秒間隔で間引く(y の窓を重ねない)
    pick = np.arange(0, len(ts), 60)
    pick = pick[np.isfinite(y[pick])]
    Xp, yp, tp = X[pick], y[pick], ts[pick]
    fin = np.isfinite(Xp).all(axis=1)
    Xp, yp, tp = Xp[fin], yp[fin], tp[fin]
    n = len(yp)
    if n < 3000:
        print(f"{sym}: 標本 {n} で不足、飛ばす")
        return
    cut = int(n * 0.7)
    med = np.median(Xp[:cut], axis=0)
    iqr = np.subtract(*np.percentile(Xp[:cut], [75, 25], axis=0))
    iqr = np.where(iqr > 1e-12, iqr, np.inf)
    Z = np.clip((Xp - med) / iqr, -5, 5)
    ysd = yp[:cut].std()
    yz = yp / max(ysd, 1e-12)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    for tag in ("real", "placebo"):
        Zt = Z if tag == "real" else np.roll(Z, int(rng.integers(n // 5, 4 * n // 5)),
                                             axis=0)
        v0 = int(cut * 0.8)
        best, bl = -1e18, LAMS[0]
        for lam in LAMS:
            w = ridge_fit(Zt[:v0], yz[:v0], lam)
            p = Zt[v0:cut] @ w
            s = spearman(p, yz[v0:cut])
            if np.isfinite(s) and s > best:
                best, bl = s, lam
        w = ridge_fit(Zt[:cut], yz[:cut], bl)
        p = Zt[cut:] @ w
        yo = yz[cut:]
        sp_ = spearman(p, yo)
        pe = float(np.corrcoef(p, yo)[0, 1])
        r2 = 1 - np.sum((yo - p) ** 2) / np.sum((yo - yo.mean()) ** 2)
        # 日ごとの符号(検定単位は日)
        days = (tp[cut:] // 86400).astype(int)
        pos = tot = 0
        for dd in np.unique(days):
            m2 = days == dd
            if m2.sum() < 50:
                continue
            s2 = spearman(p[m2], yo[m2])
            tot += 1
            pos += int(s2 > 0)
        perf.append({"symbol": sym, "tag": tag, "lam": bl, "n_train": cut,
                     "n_test": n - cut, "n_feat": Z.shape[1],
                     "spearman_oos": sp_, "pearson_oos": pe, "r2_oos": float(r2),
                     "day_pos": pos, "day_n": tot})
        if tag == "real":
            for nm, wv in zip(names, w):
                coefs.append({"symbol": sym, "feature": nm, "w": float(wv)})
    print(f"{sym}: n={n:,} feat={Z.shape[1]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    perf, coefs = [], []
    for sym in a.symbols.split(","):
        run_symbol(sym, rng, perf, coefs)
    tag = a.symbols.split(",")[0]
    pl.DataFrame(perf).write_parquet(ANA / f"reg_perf_{tag}.parquet")
    pl.DataFrame(coefs).write_parquet(ANA / f"reg_coef_{tag}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
