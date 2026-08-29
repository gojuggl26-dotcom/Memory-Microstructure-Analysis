"""spread duration は検出可能か — KuCoin 全市場を束ねた平滑化スプライン。

【問い】
  前報(`spline_regression_report.md`)で book_slope / spread の予測力は
  **プラセボの幅に埋もれて検出できなかった**。原因は標本不足(単一市場 44 日、
  実質 1,069〜6,418 点)。ここでは
    (a) 新しい説明変数として **spread duration**(現在のスプレッドが続いている時間)
    (b) **KuCoin 7 市場を束ねる**ことで標本を増やす
  の 2 つを同時に試す。

【spread duration の定義】
  各イベント時点で「spread_pp が最後に変化してからの経過秒」。
  **時刻 t までの情報だけで決まる**ので時間契約に反しない。
  分布は極端に裾が重い(163 で 38% がちょうど 0、6.6% が 1 時間超、最大 31 時間)ため
  **log1p** で扱う。

【★検証の設計 — 市場をまたぐ汎化】
  フォールドを**市場そのもの**にする(1 市場を抜いて学習 → その市場で検証)。
  時間で切るより厳しく、かつ「たまたまその市場だけ」を排除できる。
  補助として暦ブロック 5 分割も出す。

【y の扱い】
  市場ごとに金利水準もボラも違うので、**過去 20 点だけの移動 sd**で標準化する
  (因果的。未来を見ない)。市場固定効果はこの標準化で吸収される。

【出力】 data/duration_pooled.json / charts/boros_duration_pooled.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boros_spline import (LAMBDAS, bspline_basis, build_basis, edf, fit,  # noqa: E402
                          penalty_matrix)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RNG = np.random.default_rng(20260823)
N_BOOT = 300


def spread_duration(ts: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """現在の spread が続いている秒数。t 以前の情報のみ。"""
    chg = np.r_[True, sp[1:] != sp[:-1]]
    seg_start = np.maximum.accumulate(np.where(chg, ts, -np.inf))
    return ts - seg_start


def panel(mid: int, h: int) -> pl.DataFrame | None:
    p = DATA / f"event_book_{mid}.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).filter(~pl.col("extrapolated")).sort("ts")
    if d.height < 2000:
        return None
    ts = d["ts"].to_numpy().astype(float)
    sp = d["spread_pp"].to_numpy().astype(float)
    md = d["mid_pp"].to_numpy().astype(float)
    dur = spread_duration(ts, sp)
    d = d.with_columns(spread_dur=pl.Series(dur),
                       log_dur=pl.Series(np.log1p(dur)))
    # 重ならないグリッド
    g = np.arange(ts[0], ts[-1] - h, h)
    ix = np.searchsorted(ts, g, side="right") - 1
    jx = np.searchsorted(ts, g + h, side="right") - 1
    ok = (ix >= 0) & (jx >= 0)
    ix, jx, g = ix[ok], jx[ok], g[ok]
    y = md[jx] - md[ix]
    out = d[ix].with_columns(y=pl.Series(y), market=pl.lit(mid),
                             day=pl.Series((g // 86400).astype(int)))
    W = 20
    sd = np.full(len(y), np.nan)
    for k in range(W, len(y)):
        v = float(np.std(y[k - W:k]))
        sd[k] = v if v > 1e-9 else np.nan
    return out.with_columns(y_z=pl.Series(y / sd)).drop_nulls("y_z")


def loo_market(x, y, mk, B, Om, lam):
    """1 市場を抜いて学習 → その市場で検証。(R², 市場ごとの相関, 勝ち数)"""
    ums = np.unique(mk)
    num = den = 0.0
    cors = []
    for m in ums:
        tr, te = mk != m, mk == m
        if tr.sum() < 500 or te.sum() < 100:
            continue
        p = B[te] @ fit(B[tr], Om, y[tr], lam)
        r = y[te] - p
        base = y[te] - y[tr].mean()
        num += float(r @ r)
        den += float(base @ base)
        if np.std(p) > 1e-12:
            cors.append(float(np.corrcoef(p, y[te])[0, 1]))
    return (1 - num / den if den > 0 else np.nan, cors)


def analyse(d: pl.DataFrame, xcol: str, ycol: str) -> dict:
    s = d.select([xcol, ycol, "market", "day"]).drop_nulls()
    x = s[xcol].to_numpy().astype(float)
    y = s[ycol].to_numpy().astype(float)
    mk = s["market"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, mk = x[ok], y[ok], mk[ok]
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    x, y = np.clip(x, xlo, xhi), np.clip(y, ylo, yhi)
    knots, B = build_basis(x)
    Om = penalty_matrix(knots, n_basis=B.shape[1])

    cv = [loo_market(x, y, mk, B, Om, L)[0] for L in LAMBDAS]
    cv = np.array(cv)
    best = int(np.nanargmax(cv))
    lam = float(LAMBDAS[best])

    # ★入れ子(nested)LOO — λ を評価と同じ分割で選ぶと楽観側に偏る。
    #   抜いた市場 m を見ずに、残り 6 市場の内部 LOO だけで λ を決めてから m を評価する。
    ums_ = np.unique(mk)
    num_n = den_n = 0.0
    cors_n, lams_n = [], []
    for m in ums_:
        tr, te = mk != m, mk == m
        if tr.sum() < 500 or te.sum() < 100:
            continue
        inner = [loo_market(x[tr], y[tr], mk[tr], B[tr], Om, L)[0] for L in LAMBDAS]
        Lm = float(LAMBDAS[int(np.nanargmax(np.array(inner)))])
        lams_n.append(Lm)
        pr = B[te] @ fit(B[tr], Om, y[tr], Lm)
        r = y[te] - pr
        base = y[te] - y[tr].mean()
        num_n += float(r @ r); den_n += float(base @ base)
        if np.std(pr) > 1e-12:
            cors_n.append(float(np.corrcoef(pr, y[te])[0, 1]))
    r2_nested = 1 - num_n / den_n if den_n > 0 else np.nan

    r2, cors = loo_market(x, y, mk, B, Om, lam)

    # プラセボ: x を市場内で巡回シフト
    xs = x.copy()
    for m in np.unique(mk):
        i = np.flatnonzero(mk == m)
        if len(i) > 20:
            xs[i] = np.roll(x[i], max(1, len(i) // 3))
    _, Bs = build_basis(xs)
    Oms = penalty_matrix(build_basis(xs)[0], n_basis=Bs.shape[1])
    r2_pl, cors_pl = loo_market(xs, y, mk, Bs, Oms, lam)

    beta = fit(B, Om, y, lam)
    xg = np.linspace(xlo, xhi, 200)
    Bg = bspline_basis(xg, knots)
    boot = np.zeros((N_BOOT, len(xg)))
    ums = np.unique(mk)
    for b in range(N_BOOT):                       # 市場ブロックのブートストラップ
        pick = RNG.choice(ums, size=len(ums), replace=True)
        idx = np.concatenate([np.flatnonzero(mk == p) for p in pick])
        boot[b] = Bg @ fit(B[idx], Om, y[idx], lam)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)

    q = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
    bi = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    bx = np.array([x[bi == i].mean() if (bi == i).sum() > 30 else np.nan
                   for i in range(len(q) - 1)])
    by = np.array([y[bi == i].mean() if (bi == i).sum() > 30 else np.nan
                   for i in range(len(q) - 1)])
    return {"x": xcol, "y": ycol, "n": int(len(x)), "n_markets": int(len(ums)),
            "lambda": lam, "edf": edf(B, Om, lam), "cv_curve": cv.tolist(),
            "lambdas": LAMBDAS.tolist(), "r2_loo": float(r2),
            "cors": cors, "cor_med": float(np.median(cors)) if cors else np.nan,
            "n_pos": int(sum(c > 0 for c in cors)), "n_mk": len(cors),
            "r2_nested": float(r2_nested),
            "cor_nested_med": float(np.median(cors_n)) if cors_n else np.nan,
            "n_pos_nested": int(sum(c > 0 for c in cors_n)),
            "lambdas_nested": lams_n,
            "r2_placebo": float(r2_pl),
            "cor_placebo_med": float(np.median(cors_pl)) if cors_pl else np.nan,
            "cors_placebo": cors_pl,
            "grid_x": xg.tolist(), "grid_f": (Bg @ beta).tolist(),
            "ci_lo": lo95.tolist(), "ci_hi": hi95.tolist(),
            "bin_x": bx.tolist(), "bin_y": by.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="150,151,159,163,174,175,182")
    ap.add_argument("--horizons", default="600,3600")
    a = ap.parse_args()
    out = {"results": []}
    for h in [int(v) for v in a.horizons.split(",")]:
        ps = [p for p in (panel(int(m), h) for m in a.markets.split(",")) if p is not None]
        d = pl.concat(ps, how="diagonal_relaxed")
        print(f"\n=== 地平 {h}s: {len(ps)} 市場 / {d.height:,} 点(重ならない標本)===")
        for m, g in sorted(d.group_by("market"), key=lambda z: z[0][0]):
            print(f"    market {m[0]}: {g.height:>6,} 点")
        for xc in ["log_dur", "spread_pp", "slope_diff"]:
            r = analyse(d, xc, "y_z")
            r["horizon"] = h
            out["results"].append(r)
            print(f"  {xc:<11} n={r['n']:>6,} λ={r['lambda']:>7.3g} edf={r['edf']:>4.1f} "
                  f"| LOO R²={r['r2_loo']:+.5f} 入れ子={r['r2_nested']:+.5f} "
                  f"偽={r['r2_placebo']:+.5f} "
                  f"| 相関 {r['cor_med']:+.4f} 入れ子{r['cor_nested_med']:+.4f} "
                  f"偽{r['cor_placebo_med']:+.4f} "
                  f"| 正 {r['n_pos']}/{r['n_mk']}(入れ子 {r['n_pos_nested']})")
    (DATA / "duration_pooled.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\n-> {DATA / 'duration_pooled.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
