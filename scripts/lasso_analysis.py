"""全特徴量(t−1)→ 価格リターン r_t の Lasso 回帰。

★時間契約(CLAUDE.md「厳禁事項: ルックアヘッドバイアス」)
  - パネルは build_panel.py が作成。x は時刻 T までに確定、y は [T, T+1s) のリターン
  - **訓練期間と検証期間は日で分割**(シャッフルしない)。前 70 日で学習、後 29 日で評価
  - 標準化・ウィンザライズの統計量は**訓練期間だけ**から推定して検証期間に適用する
  - α の選択は訓練期間内の**拡大窓(expanding window)**分割で行う。
    行シャッフルの k-fold は隣接窓が相関しているため使わない

出力:
  data/lasso_results.json   係数・正則化パス・標本外指標
  charts/xyz_DRAM_22_lasso.png       係数の大きさと、α を緩めたときに変数が入る順番
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso, LinearRegression, lasso_path

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_1s"
TRAIN_DAYS = 70          # 前 70 日で学習、残り 29 日は触らない
N_FOLD = 5
WINSOR = 0.001           # 訓練期間の 0.1% / 99.9% で切る(外れ 1 日に支配されないように)


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    frames = []
    for k, dt in enumerate(days):
        df = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet")
        frames.append(df.with_columns(pl.lit(k).alias("day_idx")))
    df = pl.concat(frames)
    feats = [c for c in df.columns if c not in ("ts", "y", "day_idx")]
    print(f"rows={df.height:,} features={len(feats)}")

    day = df["day_idx"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float64)
    y = df["y"].to_numpy().astype(np.float64)
    tr = day < TRAIN_DAYS
    te = ~tr

    # --- 前処理はすべて訓練期間だけから推定 ---
    lo = np.quantile(X[tr], WINSOR, axis=0)
    hi = np.quantile(X[tr], 1 - WINSOR, axis=0)
    Xc = np.clip(X, lo, hi)
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0)
    sd[sd == 0] = 1.0
    Z = (Xc - mu) / sd
    ymu = y[tr].mean()
    yc = y - ymu

    # --- 因果的な日次ボラ正規化 ---
    # y の分散は日によって桁違い(立ち上げ期は大きい)。生の MSE を最小化すると
    # 分散の大きい日だけを見た推定になる。そこで **前日** の標準偏差で割る。
    # 前日の値しか使わないのでルックアヘッドにならない。
    sig = np.ones(day.max() + 1)
    for d in range(day.max() + 1):
        m = day == d
        if m.sum() > 100:
            sig[d] = max(float(yc[m].std()), 1e-6)
    sig_lag = np.concatenate([[sig[0]], sig[:-1]])      # 日 d には日 d−1 の σ を使う
    yz = yc / sig_lag[day]
    first_day = day == 0                                 # 前日が無い日は使わない

    def fit_variant(name: str, target: np.ndarray, train_from: int) -> dict:
        """α を訓練期間内の拡大窓で選び、訓練期間だけで学習して標本外を評価する。"""
        trm = (day >= train_from) & (day < TRAIN_DAYS) & (~first_day)
        tem = te & (~first_day)
        alphas = np.logspace(0, -4, 30)
        bounds = np.linspace(train_from + 15, TRAIN_DAYS, N_FOLD + 1).astype(int)
        rel = np.ones((N_FOLD, alphas.size))
        used = np.zeros(N_FOLD, dtype=bool)
        sub = np.zeros(day.size, dtype=bool)
        sub[::5] = True
        for f in range(N_FOLD):
            a, b = bounds[f], bounds[f + 1]
            m_tr = (day >= train_from) & (day < a) & sub & (~first_day)
            m_va = (day >= a) & (day < b) & sub & (~first_day)
            if m_va.sum() < 1000 or m_tr.sum() < 1000:
                continue
            _, cf, _ = lasso_path(Z[m_tr], target[m_tr], alphas=alphas,
                                  max_iter=3000, tol=1e-4)
            pred = Z[m_va] @ cf
            m = np.mean((target[m_va][:, None] - pred) ** 2, axis=0)
            rel[f] = m / float(np.mean(target[m_va] ** 2))
            used[f] = True
            print(f"    fold {f+1}: 学習日 {train_from}–{a} / 検証 {a}–{b} "
                  f"最良α={alphas[int(np.argmin(m))]:.4g} 相対MSE={rel[f].min():.6f}", flush=True)
        curve = rel[used].mean(0)
        alpha = float(alphas[int(np.argmin(curve))])
        mdl = Lasso(alpha=alpha, max_iter=20000, tol=1e-6).fit(Z[trm], target[trm])
        ols = LinearRegression().fit(Z[trm], target[trm])

        def sc(m2, mask):
            p = m2.predict(Z[mask])
            ss = float(np.sum((target[mask] - p) ** 2))
            tss = float(np.sum((target[mask] - target[mask].mean()) ** 2))
            c = float(np.corrcoef(p, target[mask])[0, 1]) if p.std() > 0 else 0.0
            return {"r2": 1 - ss / tss, "corr": c, "n": int(mask.sum())}

        dcor = []
        pt = mdl.predict(Z[tem])
        for d in np.unique(day[tem]):
            m = day[tem] == d
            if m.sum() < 500 or pt[m].std() == 0:
                continue
            dcor.append(float(np.corrcoef(pt[m], target[tem][m])[0, 1]))
        return {
            "name": name, "train_from_day": train_from, "alpha": alpha,
            "n_selected": int((mdl.coef_ != 0).sum()),
            "coef": {f: float(c) for f, c in zip(feats, mdl.coef_)},
            "cv_relative_mse": {f"{a:.5g}": float(v) for a, v in zip(alphas, curve)},
            "in_sample": sc(mdl, trm), "out_of_sample": sc(mdl, tem),
            "ols_out_of_sample": sc(ols, tem),
            "oos_daily_corr": {
                "days": len(dcor),
                "median": float(np.median(dcor)) if dcor else None,
                "q25": float(np.quantile(dcor, .25)) if dcor else None,
                "q75": float(np.quantile(dcor, .75)) if dcor else None,
                "n_days_negative": int(sum(1 for x in dcor if x < 0)),
            },
        }

    variants = []
    print("[A] 生の y・全訓練期間")
    variants.append(fit_variant("A_raw_y_all_train", yc, 0))
    print("[B] 日次ボラ正規化・全訓練期間")
    variants.append(fit_variant("B_volnorm_all_train", yz, 0))
    print("[C] 日次ボラ正規化・体制を合わせた訓練(後半 30 日)")
    variants.append(fit_variant("C_volnorm_recent_train", yz, 40))

    # --- 追加検証 ---
    best = max(variants, key=lambda v: v["oos_daily_corr"]["median"] or -1)
    tem = te & (~first_day)
    Zt, yt, dt_ = Z[tem], yz[tem], day[tem]
    w = np.array([best["coef"][f] for f in feats])
    pred = Zt @ w

    def daily_corr(pr, tg, mask=None):
        out = []
        for d in np.unique(dt_):
            m = dt_ == d if mask is None else (dt_ == d) & mask
            if m.sum() < 500 or pr[m].std() == 0 or tg[m].std() == 0:
                continue
            out.append(float(np.corrcoef(pr[m], tg[m])[0, 1]))
        return out

    # (1) 値動きがあった窓だけ(「動かないことを当てている」分を除く)
    nz = np.abs(yt) > 1e-12
    dc_nz = daily_corr(pred, yt, nz)
    # (2) プラセボ: 説明変数を日内で 5,000 行ずらす。ルックアヘッドが無ければ相関は消える
    pl_pred = np.empty_like(pred)
    for d in np.unique(dt_):
        m = dt_ == d
        pl_pred[m] = np.roll(pred[m], 5000)
    dc_pl = daily_corr(pl_pred, yt)

    res = {
        "n_rows": int(df.height), "n_features": len(feats), "features": feats,
        "train_days": TRAIN_DAYS, "test_days": int(day.max() + 1 - TRAIN_DAYS),
        "y_std_bp": float(y[tr].std()),
        "zero_return_share_test": float(1 - nz.mean()),
        "variants": variants,
        "best_variant": best["name"],
        "robustness_moving_windows_only": {
            "days": len(dc_nz), "median": float(np.median(dc_nz)) if dc_nz else None,
            "n_days_negative": int(sum(1 for x in dc_nz if x < 0))},
        "placebo_shifted_features": {
            "days": len(dc_pl), "median": float(np.median(dc_pl)) if dc_pl else None,
            "n_days_negative": int(sum(1 for x in dc_pl if x < 0))},
        "univariate_corr_train": {
            f: float(np.corrcoef(Z[tr][:, j], yz[tr])[0, 1]) for j, f in enumerate(feats)},
    }
    (D / "lasso_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    print("=== 3 通りの比較 ===")
    for v in variants:
        oc = v["oos_daily_corr"]
        print(f"[{v['name']}] alpha={v['alpha']:.4g} 選択={v['n_selected']}/{len(feats)} "
              f"標本外 R2={v['out_of_sample']['r2']:.5f} "
              f"日次相関 中央値={oc['median'] if oc['median'] is None else round(oc['median'],4)} "
              f"負={oc['n_days_negative']}/{oc['days']}")
    print(f"採用: {best['name']}")
    for f, c in sorted(best["coef"].items(), key=lambda kv: -abs(kv[1])):
        if c != 0:
            print(f"  {f:>22} {c:+.5f}  (単変量 {res['univariate_corr_train'][f]:+.4f})")
    print("値動きのある窓だけ:", res["robustness_moving_windows_only"])
    print("プラセボ(ずらし)  :", res["placebo_shifted_features"])
    print("y=0 の窓の割合    : %.4f" % res["zero_return_share_test"])


if __name__ == "__main__":
    main()
