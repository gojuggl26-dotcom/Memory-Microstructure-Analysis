"""book_slope を核にした予測モデル — 段階 7〜17。

  ベースライン → Lasso 選択 → OLS/ElasticNet → Newey-West → ウォークフォワード
  → OOS MSE/R² → ブロックブートストラップ → FDR 補正 → 取引コスト
  → 経済的有意性 → 最終モデル

【設計上の判断とその理由】
  (a) 目的変数は **y / 前日ボラ** で正規化して当てはめる。
      生の MSE は分散の大きい日に支配され、α 選択が壊れる(variance_report.md)。
      前日ボラなので因果的で、ルックアヘッドにならない。
  (b) 標準化(平均・標準偏差)は**学習期間だけ**から推定して全期間に適用する。
  (c) OOS R² は 1 − Σ(y−ŷ)²/Σ(y−ȳ_train)²。**平均も学習期間のもの**を使う
      (OOS の平均を使うと未来を見たことになる)。
  (d) 推論の単位は「日」。プールした t 統計量は使わない(有効標本は n/14)。
      Newey-West は係数の HAC 標準誤差として、日次ブートストラップは
      OOS R² の区間として、役割を分けて併用する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import ElasticNet, lars_path

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "slope_model"
NW_LAG = 60                    # 1 秒グリッドで 60 = 1 分
RNG = np.random.default_rng(20260815)
B = 5000
FEE_BP = 0.088                 # 実測のメイカー手数料


def r2_oos(y: np.ndarray, p: np.ndarray, ybar_train: float) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - ybar_train) ** 2))


def newey_west(X: np.ndarray, u: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """HAC(Bartlett 核)分散。ラグ l の自己共分散 S_l を l=0..L で重み付けして足す。"""
    S = (X * u[:, None]).astype(np.float64)
    n = S.shape[0]
    M = S.T @ S
    for lag in range(1, L + 1):
        G = S[:-lag].T @ S[lag:]
        M += (1.0 - lag / (L + 1.0)) * (G + G.T)
    return XtX_inv @ M @ XtX_inv * (n / (n - X.shape[1]))


def block_boot(x: np.ndarray, blk: int = 2) -> tuple[float, float, float]:
    n = len(x)
    nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975)), float((m <= 0).mean())


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg。棄却された仮説を True にしたベクトルを返す。"""
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    k = int(np.max(np.where(passed)[0])) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[o[:k]] = True
    return out


def main() -> None:
    meta = json.loads((D / "slope_model_data.json").read_text(encoding="utf-8"))
    SLOPE, CTRL = meta["features"]["slope"], meta["features"]["control"]
    FE = SLOPE + CTRL
    df = pl.read_parquet(OUT / "dataset.parquet")
    dts = df["dt"].to_numpy()
    sp = df["split"].to_numpy()
    y = df["y"].to_numpy().astype(np.float64)
    vp = df["vol_prev"].to_numpy().astype(np.float64)
    X = df.select(FE).to_numpy().astype(np.float64)
    spread = df["spread_bp"].to_numpy().astype(np.float64)
    del df
    tr, va, oo = sp == "train", sp == "val", sp == "oos"
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    yn = y / vp                                  # 因果正規化した目的変数
    ybar_tr, ybar_tr_n = float(y[tr].mean()), float(yn[tr].mean())
    res: dict = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "oos": int(oo.sum())},
                 "features": FE}

    # ===== 段階 7: ベースライン =========================================
    base = {"ゼロ予測": np.zeros(len(y))}
    for nm, c in (("AR(1): r_prev のみ", "r_prev"),
                  ("マイクロプライス乖離 m_bp のみ", "m_bp"),
                  ("book_slope_diff 単独", "slope_diff")):
        j = FE.index(c)
        b = np.polyfit(X[tr, j], yn[tr], 1)
        base[nm] = np.polyval(b, X[:, j]) * vp
    res["baseline"] = {k: {"oos_R2": r2_oos(y[oo], v[oo], ybar_tr),
                           "oos_MSE": float(np.mean((y[oo] - v[oo]) ** 2))}
                       for k, v in base.items()}

    # ===== 段階 8: Lasso による特徴量選択 ===============================
    # LARS で係数経路を出し、**検証期間の MSE** が最小になる点を選ぶ。
    # 学習で選び学習で評価すると必ず全部入りになるので、選択は検証期間で行う。
    print("段階 8: Lasso 経路...", flush=True)
    alphas, _, coefs = lars_path(X[tr], yn[tr] - ybar_tr_n, method="lasso")
    val_mse = np.array([np.mean((yn[va] - ybar_tr_n - X[va] @ coefs[:, i]) ** 2)
                        for i in range(coefs.shape[1])])
    kbest = int(np.argmin(val_mse))
    sel = np.where(np.abs(coefs[:, kbest]) > 1e-12)[0]
    if sel.size == 0:
        sel = np.array([FE.index("slope_diff")])
    res["lasso"] = {"alpha": float(alphas[kbest]), "n_path": int(coefs.shape[1]),
                    "n_selected": int(sel.size),
                    "selected": [FE[i] for i in sel],
                    "dropped": [f for i, f in enumerate(FE) if i not in set(sel.tolist())],
                    "val_mse_at_alpha": float(val_mse[kbest]),
                    "val_mse_null": float(np.mean((yn[va] - ybar_tr_n) ** 2)),
                    "path_alphas": alphas.tolist(), "path_val_mse": val_mse.tolist()}

    # ===== 段階 9: OLS / ElasticNet =====================================
    fit = tr | va                                   # 選択後は学習+検証で当てはめる
    Xs = np.column_stack([np.ones(len(y)), X[:, sel]])
    A = Xs[fit].T @ Xs[fit]
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (Xs[fit].T @ yn[fit])
    p_ols = Xs @ beta * vp
    # ElasticNet は**学習で当てはめ・検証で選ぶ**。sklearn の CV は
    # 折りをシャッフルしうるので使わない(隣接する秒は相関しており漏洩する)。
    # n が 330 万行あるので Gram 行列を事前計算する(座標降下が O(np) から O(p^2) になる)
    Xtr = np.ascontiguousarray(X[tr]); ytr = np.ascontiguousarray(yn[tr] - ybar_tr_n)
    gram = Xtr.T @ Xtr
    best = None
    for l1 in (0.1, 0.5, 0.9):
        for a in np.logspace(-5, -1, 9):
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=3000, precompute=gram,
                           tol=1e-6).fit(Xtr, ytr)
            v = float(np.mean((yn[va] - ybar_tr_n - m.predict(X[va])) ** 2))
            if best is None or v < best[0]:
                best = (v, l1, a, m)
        print(f"  ElasticNet l1={l1} 済", flush=True)
    del Xtr, ytr, gram
    en = best[3]
    p_en = (en.predict(X) + ybar_tr_n) * vp
    res["ols"] = {"coef": {("const" if i == 0 else FE[sel[i - 1]]): float(b)
                           for i, b in enumerate(beta)}}
    res["elasticnet"] = {"alpha": float(best[2]), "l1_ratio": float(best[1]),
                         "val_mse": float(best[0]),
                         "n_nonzero": int((np.abs(en.coef_) > 1e-12).sum()),
                         "nonzero": [FE[i] for i in np.where(np.abs(en.coef_) > 1e-12)[0]]}

    # ===== 段階 10: Newey-West 推論 =====================================
    print("段階 10: Newey-West...", flush=True)
    u = yn[fit] - Xs[fit] @ beta
    V = newey_west(Xs[fit], u, Ainv, NW_LAG)
    se = np.sqrt(np.diag(V))
    zs = beta / se
    pv = 2 * (1 - stats.norm.cdf(np.abs(zs)))
    # 日次係数(推論の単位を日にした場合の照合用)
    dbeta = []
    for d in sorted(set(dts[fit].tolist())):
        m = fit & (dts == d)
        if m.sum() < 500:
            continue
        try:
            dbeta.append(np.linalg.lstsq(Xs[m], yn[m], rcond=None)[0])
        except np.linalg.LinAlgError:
            pass
    dbeta = np.array(dbeta)
    tstat_day = dbeta.mean(0) / (dbeta.std(0, ddof=1) / np.sqrt(len(dbeta)))
    pv_day = 2 * (1 - stats.t.cdf(np.abs(tstat_day), len(dbeta) - 1))

    # ===== 段階 14: FDR 補正(係数の多重性)=============================
    names = ["const"] + [FE[i] for i in sel]
    keep_nw = bh_fdr(pv[1:], 0.05)
    keep_day = bh_fdr(pv_day[1:], 0.05)
    res["inference"] = [{"feature": names[i + 1], "beta": float(beta[i + 1]),
                         "NW_se": float(se[i + 1]), "z": float(zs[i + 1]),
                         "p_NW": float(pv[i + 1]), "FDR_pass_NW": bool(keep_nw[i]),
                         "beta_day_mean": float(dbeta.mean(0)[i + 1]),
                         "t_day": float(tstat_day[i + 1]), "p_day": float(pv_day[i + 1]),
                         "FDR_pass_day": bool(keep_day[i]),
                         "sign_days": int((np.sign(dbeta[:, i + 1])
                                           == np.sign(dbeta.mean(0)[i + 1])).sum()),
                         "n_days": int(len(dbeta))}
                        for i in range(len(sel))]
    res["nw_lag"] = NW_LAG

    # ===== 段階 11: ウォークフォワード検証 ==============================
    # OOS の各日について「その日より前の全日」だけで当てはめて予測する。
    print("段階 11: ウォークフォワード...", flush=True)
    oos_days = sorted(set(dts[oo].tolist()))
    XtX = Xs[fit].T @ Xs[fit]
    Xty = Xs[fit].T @ yn[fit]
    p_wf = np.full(len(y), np.nan)
    wf_rows = []
    jsd = (1 + list(sel).index(FE.index("slope_diff"))
           if FE.index("slope_diff") in set(sel.tolist()) else None)
    for d in oos_days:
        m = dts == d
        bw = np.linalg.solve(XtX, Xty)
        p_wf[m] = Xs[m] @ bw * vp[m]
        wf_rows.append({"dt": d, "n": int(m.sum()), "R2": r2_oos(y[m], p_wf[m], ybar_tr),
                        "beta_slope_diff": (float(bw[jsd]) if jsd is not None else None)})
        XtX += Xs[m].T @ Xs[m]        # 当日を取り込んで翌日へ(拡張窓)
        Xty += Xs[m].T @ yn[m]
    res["walk_forward"] = wf_rows

    # ===== 段階 12: OOS MSE / R² ========================================
    for nm, p in (("OLS(固定)", p_ols), ("ElasticNet(固定)", p_en),
                  ("ウォークフォワード OLS", p_wf)):
        mm = oo & np.isfinite(p)
        res.setdefault("oos", {})[nm] = {
            "R2": r2_oos(y[mm], p[mm], ybar_tr),
            "MSE": float(np.mean((y[mm] - p[mm]) ** 2)),
            "corr": float(np.corrcoef(p[mm], y[mm])[0, 1])}

    # ===== 段階 13: ブロックブートストラップ(日次 OOS R²)===============
    r2d = np.array([w["R2"] for w in wf_rows])
    lo, hi, ple = block_boot(r2d)
    res["bootstrap_R2"] = {
        "mean": float(r2d.mean()), "median": float(np.median(r2d)),
        "pos_days": int((r2d > 0).sum()), "n_days": len(r2d),
        "ci": [lo, hi], "p_le0": ple,
        "sign_test_p": float(stats.binomtest(int((r2d > 0).sum()), len(r2d), 0.5,
                                             alternative="greater").pvalue)}

    # ===== 段階 15〜16: 取引コストと経済的有意性 =========================
    # 方針: 予測が閾値を超えたら 1 単位建て、1 秒後に解消する。
    # 建玉が変わるたびに片道コスト c を払う。**損益が 0 になる c(損益分岐コスト)**を
    # 出せば、実際のスプレッド/手数料と直接比較できる。
    print("段階 15: 経済性...", flush=True)
    econ = []
    for tau_q in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        thr = np.quantile(np.abs(p_wf[oo]), tau_q) if tau_q > 0 else 0.0
        pos = np.where(np.abs(p_wf) > thr, np.sign(p_wf), 0.0)
        g, t_, hs = [], [], []
        for d in oos_days:
            m = oo & (dts == d) & np.isfinite(p_wf)
            g.append(float((pos[m] * y[m]).sum()))
            t_.append(float(np.abs(np.diff(np.concatenate([[0.0], pos[m], [0.0]]))).sum()))
            hs.append(float(np.mean(spread[m]) / 2))
        g, t_ = np.array(g), np.array(t_)
        be = float(g.sum() / max(t_.sum(), 1))          # 損益分岐コスト(片道 bp)
        for cname, c in (("メイカー(手数料のみ)", FEE_BP),
                         ("テイカー(半スプレッド+手数料)", float(np.mean(hs)) + FEE_BP)):
            net = g - c * t_
            lo2, hi2, p2 = block_boot(net)
            econ.append({"閾値分位": tau_q, "コスト前提": cname, "片道コスト_bp": c,
                         "建玉のある割合": float((np.abs(pos[oo]) > 0).mean()),
                         "回転数_日": float(t_.mean()), "粗利_bp_日": float(g.mean()),
                         "純益_bp_日": float(net.mean()), "純益中央値": float(np.median(net)),
                         "黒字日": int((net > 0).sum()), "日数": len(net),
                         "損益分岐コスト_bp": be,
                         "日次シャープ": float(net.mean() / net.std(ddof=1)),
                         "CI": [lo2, hi2], "p_le0": p2})
    res["economics"] = econ
    np.save(OUT / "pred_wf.npy", np.stack([p_wf, y, vp]).astype(np.float32))
    (D / "slope_model_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                encoding="utf-8")

    # ---- 表示 ----
    print("=== 段階 7: ベースライン(OOS)===")
    for k, v in res["baseline"].items():
        print(f"  {k:<28} R2={v['oos_R2']:+.6f}  MSE={v['oos_MSE']:.4f}")
    L = res["lasso"]
    print(f"\n=== 段階 8: Lasso 選択(alpha={L['alpha']:.3e}、経路 {L['n_path']} 点)===")
    print(f"  選択 {L['n_selected']}/{len(FE)} 個: {', '.join(L['selected'])}")
    print(f"  落選: {', '.join(L['dropped'])}")
    print(f"  検証 MSE {L['val_mse_at_alpha']:.6f} vs 定数モデル {L['val_mse_null']:.6f}")
    print(f"\n=== 段階 9・10・14: 係数と推論(NW ラグ {NW_LAG}、FDR q=0.05)===")
    print(f"{'特徴量':<20}{'beta':>11}{'NW z':>9}{'p(NW)':>10}{'FDR':>5}"
          f"{'日次 t':>9}{'p(日)':>9}{'FDR':>5}{'符号一致':>10}")
    for r in res["inference"]:
        print(f"{r['feature']:<20}{r['beta']:>11.5f}{r['z']:>9.2f}{r['p_NW']:>10.2e}"
              f"{'o' if r['FDR_pass_NW'] else 'x':>5}{r['t_day']:>9.2f}{r['p_day']:>9.4f}"
              f"{'o' if r['FDR_pass_day'] else 'x':>5}{r['sign_days']:>6}/{r['n_days']}")
    print(f"\n  ElasticNet: alpha={res['elasticnet']['alpha']:.3e} "
          f"l1={res['elasticnet']['l1_ratio']} 非ゼロ {res['elasticnet']['n_nonzero']}")
    print("\n=== 段階 11・12: OOS 成績 ===")
    for k, v in res["oos"].items():
        print(f"  {k:<22} R2={v['R2']:+.6f}  MSE={v['MSE']:.4f}  相関={v['corr']:+.4f}")
    b = res["bootstrap_R2"]
    print(f"\n=== 段階 13: 日次 OOS R2 のブートストラップ({b['n_days']} 日)===")
    print(f"  平均 {b['mean']:+.6f} 中央 {b['median']:+.6f} 正 {b['pos_days']}/{b['n_days']} "
          f"CI[{b['ci'][0]:+.6f},{b['ci'][1]:+.6f}] p(<=0)={b['p_le0']:.4f} "
          f"符号検定 p={b['sign_test_p']:.4f}")
    print("\n=== 段階 15・16: 取引コストと経済的有意性 ===")
    print(f"{'閾値':>6}{'建玉率':>8}{'回転/日':>9}{'粗利bp/日':>11}"
          f"{'前提':>20}{'コスト':>8}{'純益bp/日':>11}{'黒字':>7}{'SR':>7}")
    for e in econ:
        print(f"{e['閾値分位']:>6.2f}{e['建玉のある割合']:>8.3f}{e['回転数_日']:>9.0f}"
              f"{e['粗利_bp_日']:>11.1f}{e['コスト前提']:>20}{e['片道コスト_bp']:>8.3f}"
              f"{e['純益_bp_日']:>11.1f}{e['黒字日']:>4}/{e['日数']}{e['日次シャープ']:>7.2f}")
    print("\n  損益分岐コスト(片道): " + " / ".join(
        f"閾値{e['閾値分位']:.2f} → {e['損益分岐コスト_bp']:.4f}bp" for e in econ[::2]))


if __name__ == "__main__":
    main()
