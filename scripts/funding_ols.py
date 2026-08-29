"""funding rate と将来リターンの OLS(予測地平 100ms / 1s / 2s / 3s / 5s)。

【時間契約 — CLAUDE.md 厳禁事項】
  funding は毎時に確定・課金される。時刻 T で判るのは
  **T 以前の直近の時間境界で公表された値**だけなので、backward asof で割り当てる
  (forward / nearest を使うと未来の funding で過去を説明することになる)。
  y は [T, T+h) の対数リターン。x の確定時刻 ≤ y の期間の開始時刻 を満たす。

【標準誤差 — ここが本質】
  funding は 1 時間ごとにしか動かないので、同一時間内の全観測は x が同一値である。
  素の OLS 標準誤差は独立を仮定するため、**有効標本 = 時間数(2,360)であるところを
  数百万として扱い、t 値を桁で過大にする**。よって
  **時間クラスタ頑健標準誤差**を併記し、両方を報告する。

【標本設計】
  5 秒グリッド上で評価する(5s 地平が重複しないため)。crossed 行は除外。

【帰無対照】
  プラセボ = funding を +12 時間ずらして同じ回帰をする。関係が消えなければ混入を疑う。

【出力】 data/funding_ols.json / data/funding_ols_sample.parquet
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

PIPE = Path("C:/Users/ii562/hl-l4-pipeline")
D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
HORIZONS_NS = {"100ms": 100_000_000, "1s": 1_000_000_000, "2s": 2_000_000_000,
               "3s": 3_000_000_000, "5s": 5_000_000_000}
GRID_NS = 5_000_000_000
PLACEBO_SHIFT_NS = 12 * 3600 * 1_000_000_000


def load_funding() -> pl.DataFrame:
    fs = sorted(glob.glob(str(PIPE / "data/funding/dt=*/coin=*/part-000.parquet")))
    f = pl.concat([pl.read_parquet(x) for x in fs], how="diagonal_relaxed").sort("ts_ms")
    return f.select(ft=(pl.col("ts_ms") * 1_000_000).cast(pl.Int64),
                    funding="funding_rate", premium="premium").unique("ft").sort("ft")


def day_grid(dt: str, fund: pl.DataFrame) -> pl.DataFrame | None:
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    if len(m) < 10_000:
        return None
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    mid = m["mid"].to_numpy()
    g = np.arange(ts[0], ts[-1] - GRID_NS, GRID_NS)
    if len(g) < 100:
        return None

    def px(t: np.ndarray) -> np.ndarray:            # backward asof
        return mid[np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(mid) - 1)]

    p0 = px(g)
    out = {"t": g, "hour": g // (3600 * 1_000_000_000)}
    for name, h in HORIZONS_NS.items():
        ok = g + h <= ts[-1]
        out[f"r_{name}"] = np.where(ok, np.log(px(g + h) / p0) * 1e4, np.nan)
    df = pl.DataFrame(out)
    # funding は backward asof(未来を見ない)。プラセボは 12h ずらし
    df = df.join_asof(fund, left_on="t", right_on="ft", strategy="backward")
    df = df.with_columns(t_pl=pl.col("t") + PLACEBO_SHIFT_NS).join_asof(
        fund.rename({"funding": "funding_pl", "premium": "premium_pl"}),
        left_on="t_pl", right_on="ft", strategy="backward")
    return df.drop("t_pl")


def ols_cluster(x: np.ndarray, y: np.ndarray, cl: np.ndarray) -> dict:
    """単回帰 + 定数項。素の SE と クラスタ頑健 SE の両方を返す。"""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    s2 = (e @ e) / (n - 2)
    se_naive = np.sqrt(np.diag(XtX_inv * s2))
    # クラスタ頑健(時間単位)
    meat = np.zeros((2, 2))
    order = np.argsort(cl)
    xs, es = X[order], e[order]
    bounds = np.searchsorted(cl[order], np.unique(cl))
    bounds = np.append(bounds, n)
    for i in range(len(bounds) - 1):
        s = xs[bounds[i]:bounds[i + 1]].T @ es[bounds[i]:bounds[i + 1]]
        meat += np.outer(s, s)
    G = len(bounds) - 1
    adj = G / max(G - 1, 1) * (n - 1) / (n - 2)
    se_cl = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv) * adj)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": float(b[1]), "se_naive": float(se_naive[1]),
            "se_cluster": float(se_cl[1]), "t_naive": float(b[1] / se_naive[1]),
            "t_cluster": float(b[1] / se_cl[1]), "r2": float(r2),
            "n": int(n), "n_clusters": int(G),
            "corr": float(np.corrcoef(x, y)[0, 1])}


def main() -> None:
    fund = load_funding()
    days = sorted({Path(p).parent.name.split("=")[1]
                   for p in glob.glob(str(D / "microprice/dt=*/*.parquet"))})
    parts = [g for g in (day_grid(dt, fund) for dt in days) if g is not None]
    df = pl.concat(parts)
    df.write_parquet(D / "funding_ols_sample.parquet")
    res: dict = {"n_days": len(parts), "grid_s": GRID_NS / 1e9}
    for xcol in ["funding", "premium", "funding_pl", "premium_pl"]:
        res[xcol] = {}
        for name in HORIZONS_NS:
            s = df.select(xcol, f"r_{name}", "hour").drop_nulls()
            x = s[xcol].to_numpy().astype(float)
            y = s[f"r_{name}"].to_numpy().astype(float)
            c = s["hour"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 1000:
                continue
            res[xcol][name] = ols_cluster(x[ok], y[ok], c[ok])
    (D / "funding_ols.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("days", len(parts), "rows", len(df))


if __name__ == "__main__":
    main()
