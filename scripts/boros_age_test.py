r"""特徴量 34(板レベル)の予測力を測る — 板の年齢は将来の金利を語るか。

【Pressure の枠組みが使えない理由】
  Pressure は「残高の差分 ÷ 反対側の最良数量」という**フロー**だった。
  年齢は**水準**であって差分ではないので、同じ構成にはならない。
  ここでは水準をそのまま説明変数にする。

【説明変数(すべて時刻 t で確定する)】
  age_diff      log1p(age_w_long)    − log1p(age_w_short)     どちらの側が古いか
  age_best_diff log1p(age_best_long) − log1p(age_best_short)   最良レベルの新旧
  age_level     log1p((age_w_long + age_w_short)/2)            板全体の古さ
  n_ord_diff    log1p(n_ord_long)    − log1p(n_ord_short)      本数の偏り
  ★log1p を取るのは年齢が 0 から 1e7 秒まで 7 桁にわたるため。

【★符号の事前予測は立てない】
  Pressure には「買い圧が入れば金利は上がる」という機構的な予測があったが、
  年齢にはそれが無い。**両側検定**で扱い、見つかった向きは
  **探索の結果**として報告する(事後の理屈づけをしない)。

【時間契約】 x は t で確定、y は (t, t+k]。shift(-k) は y にしか使わない。
【判定】 市場ごとに推定し符号の一貫性で判定。プールした p 値は使わない。
【帰無対照】 x を市場内で巡回シフト。
【実装可能性】 既定でブロック跨ぎの窓に限る(同一ブロック内は執行不能)。

出力: data/age_test{suffix}.parquet
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HORIZONS = [1, 2, 3, 5, 10, 20, 30]
FEATS = ["age_diff", "age_best_diff", "age_level", "n_ord_diff"]
WINS = (0.01, 0.99)
MIN_N = 500


def ols_nw(x, y, lag):
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {}
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for h in range(1, min(lag, n - 1) + 1):
        G = Xu[h:].T @ Xu[:-h]
        S += (1 - h / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return {"beta": float(b[1]), "se_nw": float(np.sqrt(max(V[1, 1], 0.0))), "n": n}


def build(mid: int):
    pa, pe = DATA / f"age_book_{mid}.parquet", DATA / f"event_book_{mid}.parquet"
    if not (pa.exists() and pe.exists()):
        return None
    A = pl.read_parquet(pa)
    E = pl.read_parquet(pe, columns=["ev_i", "mid_pp", "block", "extrapolated"])
    if A.height != E.height:
        return None
    J = E.join(A.select(["ev_i", "age_w_long", "age_w_short", "age_best_long",
                         "age_best_short", "n_ord_long", "n_ord_short"]),
               on="ev_i", how="inner").filter(~pl.col("extrapolated")).sort("ev_i")
    if J.height < MIN_N + max(HORIZONS) + 2:
        return None
    l1 = lambda c: np.log1p(np.maximum(J[c].to_numpy().astype(float), 0))
    f = {
        "age_diff": l1("age_w_long") - l1("age_w_short"),
        "age_best_diff": l1("age_best_long") - l1("age_best_short"),
        "age_level": np.log1p(np.maximum(
            (J["age_w_long"].to_numpy() + J["age_w_short"].to_numpy()) / 2, 0)),
        "n_ord_diff": l1("n_ord_long") - l1("n_ord_short"),
    }
    ev = J["ev_i"].to_numpy()
    step1 = np.zeros(len(ev), bool)
    step1[1:] = np.diff(ev) == 1
    return f, J["mid_pp"].to_numpy(), step1, J["block"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-windows", action="store_true",
                    help="既定はブロック跨ぎ限定。これを付けると全窓を使う")
    a = ap.parse_args()
    mids = sorted(int(re.search(r"age_book_(\d+)", f).group(1))
                  for f in glob.glob(str(DATA / "age_book_*.parquet")))
    rng = np.random.default_rng(11)
    rows = []
    for mid in mids:
        got = build(mid)
        if got is None:
            continue
        f, mid_pp, step1, blk = got
        cum = np.concatenate([[0], np.cumsum(step1[1:].astype(np.int64))])
        for k in HORIZONS:
            y = np.full(len(mid_pp), np.nan)
            y[:-k] = mid_pp[k:] - mid_pp[:-k]
            clean = np.zeros(len(mid_pp), bool)
            clean[:-k] = (cum[k:] - cum[:-k]) == k
            clean &= step1
            if not a.all_windows:
                span = np.zeros(len(mid_pp), bool)
                span[:-k] = blk[k:] > blk[:-k]
                clean &= span
            for name in FEATS:
                x0 = f[name]
                m = clean & np.isfinite(x0) & np.isfinite(y)
                if m.sum() < MIN_N:
                    continue
                lo, hi = np.quantile(x0[m], WINS)
                xv, yv = np.clip(x0, lo, hi)[m], y[m]
                if np.std(xv) == 0:
                    continue
                o = ols_nw(xv, yv, k)
                if not o:
                    continue
                sh = int(rng.integers(len(xv) // 5, 4 * len(xv) // 5))
                op = ols_nw(np.roll(xv, sh), yv, k)
                rows.append({
                    "market": mid, "k": k, "feature": name, "n": int(m.sum()),
                    "beta": o["beta"], "t_nw": o["beta"] / o["se_nw"] if o["se_nw"] > 0 else np.nan,
                    "pearson": float(stats.pearsonr(xv, yv).statistic),
                    "spearman": float(stats.spearmanr(xv, yv).statistic),
                    "beta_placebo": op.get("beta", np.nan),
                })
        print(f"  market {mid}", flush=True)
    D = pl.DataFrame(rows)
    sfx = "_all" if a.all_windows else ""
    D.write_parquet(DATA / f"age_test{sfx}.parquet")

    N_CELL = len(HORIZONS) * len(FEATS)
    BONF = 0.05 / N_CELL
    print(f"\n市場 {D['market'].n_unique()} / 推定 {D.height:,} 件")
    print(f"多重比較 {N_CELL} セル → Bonferroni p < {BONF:.5f}"
          f"({'ブロック跨ぎ限定' if not a.all_windows else '全窓'})\n")
    print(f"{'特徴量':<16}" + "".join(f"{'k='+str(k):>12}" for k in HORIZONS))
    for name in FEATS:
        row = []
        for k in HORIZONS:
            v = D.filter((pl.col("feature") == name) & (pl.col("k") == k))["beta"].to_numpy()
            v = v[np.isfinite(v)]
            if len(v) < 10:
                row.append(f"{'--':>12}"); continue
            p = int((v > 0).sum())
            z = (p - len(v) / 2) / np.sqrt(len(v) / 4)
            st = "*" if stats.binomtest(p, len(v), 0.5).pvalue < BONF else " "
            row.append(f"{z:+10.1f}{st} ")
        print(f"{name:<16}" + "".join(row))
    mx = 0.0
    for name in FEATS:
        for k in HORIZONS:
            v = D.filter((pl.col("feature") == name) & (pl.col("k") == k))["beta_placebo"].to_numpy()
            v = v[np.isfinite(v)]
            if len(v) >= 10:
                mx = max(mx, abs(((v > 0).sum() - len(v) / 2) / np.sqrt(len(v) / 4)))
    print(f"\n帰無対照の最大 |z| = {mx:.2f}(偶然なら 3 前後)")
    print(f"-> {DATA / f'age_test{sfx}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
