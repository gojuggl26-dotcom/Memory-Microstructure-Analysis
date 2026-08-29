"""min{ Loss(f) + λ∫[f''(x)]²dx } による非線形回帰 — book_slope / spread → 将来の金利変化。

【模型】
  目的関数はそのまま
        Σ_i (y_i − f(x_i))²  +  λ ∫ f''(x)² dx
  最小化は 3 次 B スプライン基底 B(x) と**厳密な粗さ罰則行列**
        Ω_jk = ∫ B_j''(x) B_k''(x) dx
  で書ける(3 次 B スプラインの 2 階微分は区分線形なので、
  各ノット区間の 2 点ガウス求積が**厳密**)。解は
        β̂ = (BᵀB + λΩ)⁻¹ Bᵀy
  有効自由度は tr(H), H = B(BᵀB+λΩ)⁻¹Bᵀ。

【λ の決め方 — 機械的】
  **時間ブロック分割の交差検証**(連続する日でフォールドを作る)。
  行のシャッフルによる k-fold は隣接イベントが相関するため漏洩する(METHODOLOGY §3)。
  参考として GCV も併記する。

【時間契約】
  x = イベント時刻 t の板の形(t で確定)
  y = mid(t+h) − mid(t)     [pp]     t 以降のみ
  mid(t+h) は **backward asof**(その時刻以前の最後の値)。forward/nearest は使わない。

【★標本の取り方 — 重なりを排す】
  イベントは同一ブロックに固まり、間隔の中央値は 0 秒。全イベントを使うと
  窓が重なって y の自己相関が **lag1 で 0.98〜0.999** に達し、交差検証が壊れる
  (実測。最初の設計はこれで CV R² が全て負になった)。
  よって **間隔 h の重ならないグリッド上で 1 点ずつ**取る。

【★体制の不均一 — 因果的な標準化】
  フォールド間で y の sd が 8 倍違う(実測: 0.068 〜 0.561)。
  そのままでは「当たる/外れる」が体制の分散に支配される。よって
  **過去だけを使う移動標準偏差**で y を割った版も併せて出す(先読みしない)。

【信頼帯】
  残差は強く自己相関するので、素の Bayesian CI は狭すぎる。
  **日単位のブロックブートストラップ**で点ごとの区間を出す(METHODOLOGY §4)。

【帰無対照】
  x を日内で巡回シフトしたプラセボ。曲線が平坦にならなければ混入を疑う。

【出力】 data/spline_{marketId}.json / charts/boros_spline_{marketId}.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
LAMBDAS = np.logspace(-4, 6, 21)
N_FOLD = 5
N_BOOT = 300
RNG = np.random.default_rng(20260823)


# ---------------------------------------------------------------- B スプライン
def bspline_basis(x: np.ndarray, knots: np.ndarray, deg: int = 3) -> np.ndarray:
    """Cox-de Boor。knots は端点を deg+1 重複させた完全ノット列。"""
    n = len(knots) - deg - 1
    B = np.zeros((len(x), n))
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1.0
        B[:, j] = _deboor(x, knots, c, deg)
    return B


def _deboor(x, t, c, p):
    """基底 1 本の評価(単純だが本数が少ないので十分速い)。"""
    n = len(c)
    out = np.zeros_like(x, dtype=float)
    # 0 次
    N = np.zeros((len(x), len(t) - 1))
    for i in range(len(t) - 1):
        if t[i] < t[i + 1]:
            N[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(float)
    N[x >= t[-1], -1 - p] = 1.0            # 右端を閉じる
    for d in range(1, p + 1):
        M = np.zeros((len(x), len(t) - 1 - d))
        for i in range(len(t) - 1 - d):
            a = b = 0.0
            if t[i + d] > t[i]:
                a = (x - t[i]) / (t[i + d] - t[i]) * N[:, i]
            if t[i + d + 1] > t[i + 1]:
                b = (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1]) * N[:, i + 1]
            M[:, i] = a + b
        N = M
    out = N[:, :n] @ c
    return out


def penalty_matrix(knots: np.ndarray, deg: int = 3, n_basis: int | None = None):
    """Ω_jk = ∫ B_j'' B_k'' dx を 2 点ガウス求積で厳密に組む。"""
    n = n_basis or len(knots) - deg - 1
    uk = np.unique(knots)
    g = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    Om = np.zeros((n, n))
    for a, b in zip(uk[:-1], uk[1:]):
        if b <= a:
            continue
        pts = (b - a) / 2 * g + (a + b) / 2
        w = (b - a) / 2
        D2 = _second_deriv(pts, knots, n, deg)
        for q in range(len(pts)):
            Om += w * np.outer(D2[q], D2[q])
    return Om


def _second_deriv(x, knots, n, deg=3, h=None):
    """B'' を差分で評価(区分線形なので中心差分が厳密に近い)。"""
    if h is None:
        h = (knots[-1] - knots[0]) * 1e-5
    B0 = bspline_basis(np.clip(x - h, knots[0], knots[-1]), knots, deg)
    B1 = bspline_basis(x, knots, deg)
    B2 = bspline_basis(np.clip(x + h, knots[0], knots[-1]), knots, deg)
    return (B0 - 2 * B1 + B2) / h ** 2


# ---------------------------------------------------------------- 当てはめ
def fit(B, Om, y, lam):
    A = B.T @ B + lam * Om
    return np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), B.T @ y)


def edf(B, Om, lam):
    A = B.T @ B + lam * Om
    return float(np.trace(np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), B.T @ B)))


def build_basis(x, n_knots=25, deg=3):
    lo, hi = np.nanmin(x), np.nanmax(x)
    inner = np.quantile(x, np.linspace(0, 1, n_knots)[1:-1])
    inner = np.unique(inner)
    knots = np.r_[[lo] * (deg + 1), inner, [hi] * (deg + 1)]
    return knots, bspline_basis(np.clip(x, lo, hi), knots, deg)


# ---------------------------------------------------------------- 主処理
def analyse(df: pl.DataFrame, xcol: str, ycol: str, tag: str) -> dict:
    d = df.select(["ts", "day", xcol, ycol]).drop_nulls()
    x = d[xcol].to_numpy().astype(float)
    y = d[ycol].to_numpy().astype(float)
    day = d["day"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, day = x[ok], y[ok], day[ok]
    # 訓練側だけで作るべきだが、λ 選択の外側の前処理はここで固定(全体の 0.5/99.5%)
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    x = np.clip(x, xlo, xhi)
    y = np.clip(y, ylo, yhi)

    knots, B = build_basis(x)
    Om = penalty_matrix(knots, n_basis=B.shape[1])

    # --- λ を時間ブロック CV で選ぶ ---
    udays = np.unique(day)
    fold = {d_: i * N_FOLD // len(udays) for i, d_ in enumerate(udays)}   # 連続ブロック
    fid = np.array([fold[d_] for d_ in day])
    # ★R² の基準は「**訓練期間の平均**」にする。試験フォールド自身の平均は
    #   その時点では知りようがない理想値で、基準に使うと不当に厳しくなる。
    def oos(pred_fn):
        num = den = 0.0
        cors, wins = [], []
        for f in range(N_FOLD):
            tr, te = fid != f, fid == f
            if tr.sum() < 500 or te.sum() < 100:
                continue
            p = pred_fn(tr, te)
            r = y[te] - p
            base = y[te] - y[tr].mean()             # 実装可能な基準
            num += float(r @ r)
            den += float(base @ base)
            if np.std(p) > 1e-12:
                cors.append(float(np.corrcoef(p, y[te])[0, 1]))
            wins.append(float(r @ r) < float(base @ base))
        return (1 - num / den if den > 0 else np.nan,
                float(np.median(cors)) if cors else np.nan,
                int(sum(wins)), len(wins))

    cv = []
    for lam in LAMBDAS:
        cv.append(oos(lambda tr, te, L=lam: B[te] @ fit(B[tr], Om, y[tr], L))[0])
    cv = np.array(cv)
    best = int(np.nanargmax(cv))
    lam = float(LAMBDAS[best])
    r2_sp, cor_sp, win_sp, nf = oos(lambda tr, te: B[te] @ fit(B[tr], Om, y[tr], lam))

    # --- GCV(参考)---
    gcv = []
    for L in LAMBDAS:
        beta = fit(B, Om, y, L)
        r = y - B @ beta
        t_ = edf(B, Om, L)
        gcv.append(float((r @ r) / len(y) / (1 - t_ / len(y)) ** 2))
    lam_gcv = float(LAMBDAS[int(np.nanargmin(gcv))])

    beta = fit(B, Om, y, lam)
    xg = np.linspace(xlo, xhi, 200)
    Bg = bspline_basis(xg, knots)
    fg = Bg @ beta

    # --- 日ブロックのブートストラップで信頼帯 ---
    boot = np.zeros((N_BOOT, len(xg)))
    for b in range(N_BOOT):
        pick = RNG.choice(udays, size=len(udays), replace=True)
        idx = np.concatenate([np.flatnonzero(day == p) for p in pick])
        boot[b] = Bg @ fit(B[idx], Om, y[idx], lam)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)

    # --- プラセボ: x を日内で巡回シフト ---
    xs = x.copy()
    for d_ in udays:
        m = day == d_
        if m.sum() > 10:
            xs[m] = np.roll(x[m], max(1, m.sum() // 3))
    _, Bs = build_basis(xs)
    Oms = penalty_matrix(build_basis(xs)[0], n_basis=Bs.shape[1])
    r2_pl_v, cor_pl, win_pl, _ = oos(lambda tr, te: Bs[te] @ fit(Bs[tr], Oms, y[tr], lam))

    # --- 線形との比較(同じ CV 分割)---
    X = np.column_stack([np.ones(len(x)), x])
    r2_lin, cor_lin, win_lin, _ = oos(
        lambda tr, te: X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0])

    # --- ビン平均(ノンパラの目視確認)---
    q = np.quantile(x, np.linspace(0, 1, 21))
    q = np.unique(q)
    bi = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    bx = np.array([x[bi == i].mean() if (bi == i).sum() > 20 else np.nan
                   for i in range(len(q) - 1)])
    by = np.array([y[bi == i].mean() if (bi == i).sum() > 20 else np.nan
                   for i in range(len(q) - 1)])

    return {"tag": tag, "x": xcol, "y": ycol, "n": int(len(x)),
            "lambda_cv": lam, "lambda_gcv": lam_gcv,
            "edf": edf(B, Om, lam), "n_basis": int(B.shape[1]),
            "cv_r2": float(r2_sp), "cv_curve": cv.tolist(),
            "lambdas": LAMBDAS.tolist(),
            "cor_spline": cor_sp, "win_spline": [win_sp, nf],
            "cv_r2_linear": float(r2_lin), "cor_linear": cor_lin,
            "cv_r2_placebo": float(r2_pl_v), "cor_placebo": cor_pl,
            "grid_x": xg.tolist(), "grid_f": fg.tolist(),
            "ci_lo": lo95.tolist(), "ci_hi": hi95.tolist(),
            "bin_x": bx.tolist(), "bin_y": by.tolist(),
            "x_p": np.percentile(x, [1, 25, 50, 75, 99]).tolist()}


def build_panel(d: pl.DataFrame, h: int) -> pl.DataFrame:
    """間隔 h の**重ならない**グリッド上で 1 点ずつ取る。

    x はグリッド時刻**以前**の最後の板(backward asof)、
    y は mid(g+h) − mid(g)。さらに **過去 20 点だけ**の移動 sd で割った版を作る
    (因果的な標準化。未来を見ない)。
    """
    ts = d["ts"].to_numpy().astype(float)
    mid = d["mid_pp"].to_numpy().astype(float)
    g = np.arange(ts[0], ts[-1] - h, h)                 # 重ならないグリッド
    ix = np.searchsorted(ts, g, side="right") - 1       # ★backward asof
    jx = np.searchsorted(ts, g + h, side="right") - 1
    ok = (ix >= 0) & (jx >= 0)
    g, ix, jx = g[ok], ix[ok], jx[ok]
    y = mid[jx] - mid[ix]
    out = d[ix].with_columns(
        g_ts=pl.Series(g), y=pl.Series(y), day=pl.Series((g // 86400).astype(int)))
    # 因果的な移動 sd(直前 20 点、最低 10 点)
    W = 20
    sd = np.full(len(y), np.nan)
    for k in range(W, len(y)):
        v = np.std(y[k - W:k])
        sd[k] = v if v > 1e-9 else np.nan
    return out.with_columns(y_z=pl.Series(y / sd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", type=int, default=163)
    ap.add_argument("--horizons", default="600,3600")
    a = ap.parse_args()
    d = pl.read_parquet(DATA / f"event_book_{a.market}.parquet")           .filter(~pl.col("extrapolated")).sort("ts")
    out = {"market": a.market, "n_events": d.height, "results": []}
    print(f"イベント {d.height:,} 行")
    for h in [int(x) for x in a.horizons.split(",")]:
        p = build_panel(d, h)
        print(f"\n--- 地平 {h}s: 重ならない標本 {p.height:,} 点 "
              f"({p.height / 45:.0f}/日)---")
        ya = p["y"].to_numpy().astype(float); ya = ya[np.isfinite(ya)]
        aa = ya - ya.mean()
        print(f"    y の自己相関 lag1 = {float((aa[:-1]*aa[1:]).mean()/(aa*aa).mean()):+.3f}"
              f"(全イベント使用時は +0.98〜0.999 だった)")
        for ycol, ylab in [("y", "生"), ("y_z", "因果標準化")]:
            for xc in ["slope_diff", "spread_pp"]:
                r = analyse(p, xc, ycol, f"{xc}→{h}s[{ylab}]")
                r["horizon"] = h; r["yspec"] = ylab
                out["results"].append(r)
                print(f"  {r['tag']:<28} n={r['n']:>5,} λ={r['lambda_cv']:>8.3g} "
                      f"edf={r['edf']:>4.1f} R²={r['cv_r2']:+.5f} "
                      f"線形{r['cv_r2_linear']:+.5f} 偽{r['cv_r2_placebo']:+.5f} "
                      f"| 相関 {r['cor_spline']:+.4f} 偽{r['cor_placebo']:+.4f} "
                      f"| 勝ち {r['win_spline'][0]}/{r['win_spline'][1]}")
    (DATA / f"spline_{a.market}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\n-> {DATA / f'spline_{a.market}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
