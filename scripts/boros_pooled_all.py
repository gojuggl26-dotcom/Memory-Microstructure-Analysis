"""Boros 全市場を束ねた平滑化スプライン — 標本不足の解消。

【なぜ全市場か】
  KuCoin 7 市場(122,576 イベント)では spread duration が検出できず、
  book_slope_diff はかろうじて 7/7 で検出できた。標本を増やして
  **どちらの結論も確かめ直す**。全 188 市場・**約 1,046 万イベント**(85 倍)。

【計算の工夫 — グラム行列は加法的】
  188 市場・数百万点で λ を入れ子交差検証すると素朴には回らない。
  だが正規方程式は
        β = (BᵀB + λΩ)⁻¹ Bᵀy
  で、**BᵀB と Bᵀy はフォールドごとの和**に分解できる。
  よってフォールド k ごとに G_k = B_kᵀB_k, c_k = B_kᵀy_k を**一度だけ**作れば、
  任意の訓練集合(= フォールドの部分和)と任意の λ に対する解が
  **28×28 の連立方程式を解くだけ**になる。入れ子も実質無料。

【検証】
  ・フォールドは**市場をまとめた 10 群**(市場が跨がないので漏洩しない)
  ・λ は**入れ子**で選ぶ(外側フォールドを見ずに内側だけで決める)
  ・主判定は **市場ごとの標本外相関の符号一貫性**(n≈188 の符号検定)
  ・プラセボは x を市場内で巡回シフト

【出力】 data/pooled_all.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boros_spline import LAMBDAS, bspline_basis, build_basis, penalty_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_FOLD = 10
RNG = np.random.default_rng(20260824)


def spread_duration(ts, sp):
    chg = np.r_[True, sp[1:] != sp[:-1]]
    return ts - np.maximum.accumulate(np.where(chg, ts, -np.inf))


def panel(path: Path, h: int) -> pl.DataFrame | None:
    mid = int(re.search(r"event_book_(\d+)", path.name).group(1))
    d = (pl.read_parquet(path, columns=["ts", "spread_pp", "mid_pp", "slope_diff",
                                        "slope_long", "slope_short", "depth_long",
                                        "depth_short", "ttm_years", "extrapolated"])
           .filter(~pl.col("extrapolated")).sort("ts"))
    if d.height < 500:
        return None
    ts = d["ts"].to_numpy().astype(float)
    sp = d["spread_pp"].to_numpy().astype(float)
    md = d["mid_pp"].to_numpy().astype(float)
    d = d.with_columns(log_dur=pl.Series(np.log1p(spread_duration(ts, sp))))
    g = np.arange(ts[0], ts[-1] - h, h)
    if len(g) < 60:
        return None
    ix = np.searchsorted(ts, g, side="right") - 1
    jx = np.searchsorted(ts, g + h, side="right") - 1
    ok = (ix >= 0) & (jx >= 0)
    ix, jx = ix[ok], jx[ok]
    y = md[jx] - md[ix]
    W = 20
    sd = np.full(len(y), np.nan)
    for k in range(W, len(y)):
        v = float(np.std(y[k - W:k]))
        sd[k] = v if v > 1e-9 else np.nan
    return (d[ix].with_columns(y_z=pl.Series(y / sd), market=pl.lit(mid))
                 .drop_nulls("y_z")
                 .select("market", "y_z", "log_dur", "spread_pp", "slope_diff",
                         "slope_long", "slope_short", "ttm_years"))


def grams(B, y, fid, K):
    """フォールドごとの (BᵀB, Bᵀy, n)。以後はこの和だけで任意の訓練集合を作れる。"""
    return [(B[fid == k].T @ B[fid == k], B[fid == k].T @ y[fid == k], int((fid == k).sum()))
            for k in range(K)]


def solve(G, c, Om, lam):
    A = G + lam * Om
    return np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), c)


def analyse(x, y, mk, tag) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, mk = x[ok], y[ok], mk[ok]
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    x, y = np.clip(x, xlo, xhi), np.clip(y, ylo, yhi)
    knots, B = build_basis(x)
    Om = penalty_matrix(knots, n_basis=B.shape[1])

    ums = np.unique(mk)
    # 市場をまとめて 10 群に(市場は群を跨がない)
    grp = {m: i % N_FOLD for i, m in enumerate(RNG.permutation(ums))}
    fid = np.array([grp[m] for m in mk])
    Gs = grams(B, y, fid, N_FOLD)
    ybar = [float(y[fid == k].mean()) for k in range(N_FOLD)]
    nk = [int((fid == k).sum()) for k in range(N_FOLD)]

    def train(excl):
        G = sum(Gs[k][0] for k in range(N_FOLD) if k not in excl)
        c = sum(Gs[k][1] for k in range(N_FOLD) if k not in excl)
        n = sum(nk[k] for k in range(N_FOLD) if k not in excl)
        m = sum(ybar[k] * nk[k] for k in range(N_FOLD) if k not in excl) / n
        return G, c, m

    def score(excl_outer, lam):
        G, c, m = train(excl_outer)
        b = solve(G, c, Om, lam)
        num = den = 0.0
        for k in excl_outer:
            te = fid == k
            r = y[te] - B[te] @ b
            base = y[te] - m
            num += float(r @ r); den += float(base @ base)
        return 1 - num / den if den > 0 else np.nan

    # --- 入れ子: 外側 k を見ずに、内側だけで λ を決める ---
    num_n = den_n = 0.0
    lam_sel, cor_mk = [], {}
    for k in range(N_FOLD):
        inner = [np.nanmean([score({k, j}, L) for j in range(N_FOLD) if j != k])
                 for L in LAMBDAS]
        Lk = float(LAMBDAS[int(np.nanargmax(np.array(inner)))])
        lam_sel.append(Lk)
        G, c, m = train({k})
        b = solve(G, c, Om, Lk)
        te = fid == k
        p = B[te] @ b
        r = y[te] - p
        base = y[te] - m
        num_n += float(r @ r); den_n += float(base @ base)
        for mm in np.unique(mk[te]):                 # 市場ごとの相関
            s = mk[te] == mm
            if s.sum() >= 50 and np.std(p[s]) > 1e-12 and np.std(y[te][s]) > 1e-12:
                cor_mk[int(mm)] = float(np.corrcoef(p[s], y[te][s])[0, 1])
    r2_nested = 1 - num_n / den_n if den_n > 0 else np.nan

    # --- 全体 λ(参考)と当てはめ曲線 ---
    cvg = [np.nanmean([score({k}, L) for k in range(N_FOLD)]) for L in LAMBDAS]
    lam = float(LAMBDAS[int(np.nanargmax(np.array(cvg)))])
    Gall = sum(g[0] for g in Gs); call = sum(g[1] for g in Gs)
    beta = solve(Gall, call, Om, lam)
    xg = np.linspace(xlo, xhi, 200)
    Bg = bspline_basis(xg, knots)
    edf_ = float(np.trace(np.linalg.solve(Gall + lam * Om + 1e-10 * np.eye(len(Om)), Gall)))

    # --- 市場ブロックのブートストラップ ---
    boot = np.zeros((200, len(xg)))
    idx_by_m = {m: np.flatnonzero(mk == m) for m in ums}
    for bi in range(200):
        pick = RNG.choice(ums, size=len(ums), replace=True)
        idx = np.concatenate([idx_by_m[p] for p in pick])
        Bb = B[idx]
        boot[bi] = Bg @ solve(Bb.T @ Bb, Bb.T @ y[idx], Om, lam)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)

    cors = np.array(list(cor_mk.values()))
    q = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
    bi_ = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    bx = np.array([x[bi_ == i].mean() if (bi_ == i).sum() > 200 else np.nan
                   for i in range(len(q) - 1)])
    by = np.array([y[bi_ == i].mean() if (bi_ == i).sum() > 200 else np.nan
                   for i in range(len(q) - 1)])
    return {"tag": tag, "n": int(len(x)), "n_markets": int(len(ums)),
            "lambda": lam, "lambda_nested": lam_sel, "edf": edf_,
            "r2_nested": float(r2_nested), "cv_curve": [float(v) for v in cvg],
            "lambdas": LAMBDAS.tolist(),
            "cor_med": float(np.median(cors)) if len(cors) else np.nan,
            "n_pos": int((cors > 0).sum()), "n_mk": int(len(cors)),
            "cors_by_market": {str(k): v for k, v in cor_mk.items()},
            "grid_x": xg.tolist(), "grid_f": (Bg @ beta).tolist(),
            "ci_lo": lo95.tolist(), "ci_hi": hi95.tolist(),
            "bin_x": bx.tolist(), "bin_y": by.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="600,3600")
    ap.add_argument("--xs", default="log_dur,spread_pp,slope_diff")
    a = ap.parse_args()
    files = sorted(glob.glob(str(DATA / "event_book_*.parquet")))
    out = {"n_files": len(files), "results": []}
    for h in [int(v) for v in a.horizons.split(",")]:
        ps = []
        for f in files:
            p = panel(Path(f), h)
            if p is not None:
                ps.append(p)
        d = pl.concat(ps, how="diagonal_relaxed")
        y = d["y_z"].to_numpy().astype(float)
        mk = d["market"].to_numpy()
        print(f"\n=== 地平 {h}s: {len(ps)} 市場 / {d.height:,} 点 ===", flush=True)
        for xc in a.xs.split(","):
            r = analyse(d[xc].to_numpy().astype(float), y, mk, f"{xc}@{h}s")
            r["horizon"] = h; r["x"] = xc
            # プラセボ
            xs_ = d[xc].to_numpy().astype(float).copy()
            for m in np.unique(mk):
                i = np.flatnonzero(mk == m)
                if len(i) > 20:
                    xs_[i] = np.roll(xs_[i], max(1, len(i) // 3))
            rp = analyse(xs_, y, mk, f"placebo {xc}@{h}s")
            r["placebo"] = {"r2_nested": rp["r2_nested"], "cor_med": rp["cor_med"],
                            "n_pos": rp["n_pos"], "n_mk": rp["n_mk"]}
            out["results"].append(r)
            print(f"  {xc:<11} n={r['n']:>9,} 市場 {r['n_mk']:>3} λ={r['lambda']:>7.3g} "
                  f"edf={r['edf']:>4.1f} | 入れ子R²={r['r2_nested']:+.5f} "
                  f"相関 {r['cor_med']:+.4f} 正 {r['n_pos']}/{r['n_mk']} "
                  f"| 偽 R²={rp['r2_nested']:+.5f} 相関{rp['cor_med']:+.4f} "
                  f"正 {rp['n_pos']}/{rp['n_mk']}", flush=True)
    (DATA / "pooled_all.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\n-> {DATA / 'pooled_all.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
