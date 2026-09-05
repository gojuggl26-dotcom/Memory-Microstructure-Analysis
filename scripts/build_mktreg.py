"""市場の状態で Entry EV を説明する回帰。

    uv run python scripts/build_mktreg.py

    EV ≈ b1·x1 + b2·log(Depth/TradeSize) + b3·OBI_IC + 銘柄固定効果

x1 = (スプレッド/2 − 1ティック) / σ_short。

出す 3 つ
---------
 1. **同時**(同じ窓の説明変数で同じ窓の EV)。これは予測ではなく記述である
 2. **1 期先**(1 つ前の窓の説明変数で次の窓の EV)。これが実際に使える形
 3. 説明変数どうしの相関(x1 と OBI_IC が共線でないかの確認)

標準誤差は **銘柄×日でクラスタ**にする(同じ日の 24 窓は独立でない)。
重みは窓内の建玉数(EV はその平均なので、分散が 1/n に比例する)。
重みなしの結果も併記する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_LOT = 20
SPECS = [
    ("x1 のみ", ["x1"]),
    ("depth のみ", ["log_dot"]),
    ("OBI_IC(mid)のみ", ["obi_ic_mid"]),
    ("OBI_IC(micro)のみ", ["obi_ic_micro"]),
    ("3 つ(mid の IC)", ["x1", "log_dot", "obi_ic_mid"]),
    ("3 つ(micro の IC)", ["x1", "log_dot", "obi_ic_micro"]),
]


def ols_cluster(X, y, g, w=None):
    """クラスタ頑健な標準誤差つきの重み付き最小二乗。"""
    n, p = X.shape
    if w is None:
        w = np.ones(n)
    W = w / w.mean()
    XtX = X.T @ (X * W[:, None])
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ (y * W))
    e = y - X @ b
    meat = np.zeros((p, p))
    for c in np.unique(g):
        m = g == c
        s = (X[m] * (W[m] * e[m])[:, None]).sum(axis=0)
        meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    r2 = 1.0 - float((W * e ** 2).sum() / (W * (y - np.average(y, weights=W)) ** 2).sum())
    return b, se, r2, int(np.unique(g).size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=1)
    a = ap.parse_args()
    P = pl.read_parquet(DATA / "mktpanel.parquet").sort("coin", "dt", "hour")
    P = P.with_columns(log_dot=pl.col("depth_over_trade").log())
    P = P.filter(pl.col("n_lot") >= MIN_LOT, pl.col("ev").is_not_nan(),
                 pl.col("obi_ic_micro").is_not_nan())
    print(f"窓 {P.height:,} 個(建玉 {MIN_LOT} 本以上)  "
          f"銘柄 {P['coin'].n_unique()}  日 {P['dt'].n_unique()}")

    # 1 期先(同じ銘柄・同じ日の中で 1 時間ずらす)
    P = P.with_columns([
        pl.col(c).shift(1).over(["coin", "dt"]).alias(f"lag_{c}")
        for c in ("x1", "log_dot", "obi_ic_mid", "obi_ic_micro")],
        gap=pl.col("hour").diff().over(["coin", "dt"]))
    print()
    print("説明変数どうしの相関(同時)")
    cs = ["x1", "log_dot", "obi_ic_mid", "obi_ic_micro"]
    M = np.corrcoef(np.column_stack([P[c].to_numpy() for c in cs]).T)
    print("        " + "".join(f"{c:>14s}" for c in cs))
    for i, c in enumerate(cs):
        print(f"{c:>8s}" + "".join(f"{M[i, j]:>14.3f}" for j in range(len(cs))))

    out = []
    for kind in ("同時", "1 期先"):
        Q = P if kind == "同時" else P.filter(pl.col("gap") == 1)
        pre = "" if kind == "同時" else "lag_"
        y = Q["ev"].to_numpy()
        w = Q["n_lot"].to_numpy().astype(float)
        g = np.array([f"{a_}|{b_}" for a_, b_ in zip(Q["coin"], Q["dt"])])
        coin = (Q["coin"].to_numpy() == "xyz_MU").astype(float)
        print(f"\n=== {kind} ({Q.height:,} 窓) ===")
        for nm, vs in SPECS:
            cols = [Q[pre + v].to_numpy() for v in vs]
            if any(np.isnan(c).any() for c in cols):
                keep = ~np.any(np.isnan(np.column_stack(cols)), axis=1)
            else:
                keep = np.ones(Q.height, bool)
            Z = np.column_stack(cols)[keep]
            mu, sd = Z.mean(axis=0), Z.std(axis=0)
            sd[sd < 1e-12] = 1.0
            X = np.column_stack([np.ones(keep.sum()), coin[keep],
                                 (Z - mu) / sd])          # 標準化して比較可能に
            b, se, r2, nc = ols_cluster(X, y[keep], g[keep], w[keep])
            t = b / np.maximum(se, 1e-12)
            s = "  ".join(f"{v}={b[2+i]:+.4f}({t[2+i]:+.2f})"
                          for i, v in enumerate(vs))
            print(f"  {nm:20s} R2 {r2:+.4f}  {s}")
            for i, v in enumerate(vs):
                out.append({"kind": kind, "spec": nm, "var": v,
                            "beta": float(b[2 + i]), "se": float(se[2 + i]),
                            "t": float(t[2 + i]), "r2": r2, "n": int(keep.sum()),
                            "clusters": nc})
    pl.DataFrame(out).write_csv(DATA / "mktreg.csv")

    print("\n各変数の五分位ごとの EV(同時・建玉数で加重)")
    for v in ("x1", "log_dot", "obi_ic_mid", "obi_ic_micro"):
        x = P[v].to_numpy()
        y = P["ev"].to_numpy()
        w = P["n_lot"].to_numpy().astype(float)
        e = np.quantile(x, [.2, .4, .6, .8])
        k = np.searchsorted(e, x, side="right")
        s = "  ".join(f"Q{i+1} {np.average(y[k == i], weights=w[k == i]):+7.3f}"
                      for i in range(5))
        print(f"  {v:14s} {s}")
    print(f"\n書き出し {DATA}/mktreg.csv")


if __name__ == "__main__":
    main()
