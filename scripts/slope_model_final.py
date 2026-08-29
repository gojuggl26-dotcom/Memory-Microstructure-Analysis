"""段階 17: 最終モデルの確定。

【監査で判った問題】
  slope 系のみ(OOS R² +0.0399)が全部入り(+0.0372)を上回った。
  だが **「OOS で良かったから採る」は OOS を選択に使うことであり禁じ手**である
  (CLAUDE.md 厳禁事項 5: 結果を見てから選別しない)。

  したがって候補の優劣は**検証期間だけ**で決め、その結果を OOS で 1 回だけ評価する。
  検証で負ける候補を OOS の成績を理由に採用してはならない。

【候補】
  M1 全部入り(Lasso が検証 MSE で選んだ 25 変数)
  M2 slope 系のみ(Lasso 選択のうち book_slope 由来のもの)
  M3 slope_diff 1 本
  M4 slope 系 + マイクロプライス乖離 m_bp(既知の最強統制 1 本だけ)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "slope_model"
RNG = np.random.default_rng(20260815)
B = 5000
FEE_BP = 0.088


def block_boot(x, blk=2):
    n = len(x)
    nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975)), float((m <= 0).mean())


def main() -> None:
    meta = json.loads((D / "slope_model_data.json").read_text(encoding="utf-8"))
    SLOPE, CTRL = meta["features"]["slope"], meta["features"]["control"]
    FE = SLOPE + CTRL
    res = json.loads((D / "slope_model_results.json").read_text(encoding="utf-8"))
    lsel = res["lasso"]["selected"]

    df = pl.read_parquet(OUT / "dataset.parquet").sort(["dt", "ts"])
    dts = df["dt"].to_numpy(); sp = df["split"].to_numpy()
    y = df["y"].to_numpy().astype(np.float64)
    vp = df["vol_prev"].to_numpy().astype(np.float64)
    spread = df["spread_bp"].to_numpy().astype(np.float64)
    X = df.select(FE).to_numpy().astype(np.float64)
    del df
    tr, va, oo = sp == "train", sp == "val", sp == "oos"
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    yn = y / vp
    ybar_tr = float(y[tr].mean())

    cand = {
        "M1 全部入り(25 変数)": lsel,
        "M2 slope 系のみ": [f for f in lsel if f in SLOPE],
        "M3 slope_diff 1 本": ["slope_diff"],
        "M4 slope 系 + m_bp": [f for f in lsel if f in SLOPE] + ["m_bp"],
    }
    out: dict = {"candidates": {}}

    # ---- 選択は検証期間だけで行う(学習で当てはめ・検証で比べる)----------
    for nm, fs in cand.items():
        j = [FE.index(f) for f in fs]
        Xs = np.column_stack([np.ones(len(y)), X[:, j]])
        b = np.linalg.solve(Xs[tr].T @ Xs[tr], Xs[tr].T @ yn[tr])
        pv_ = Xs[va] @ b * vp[va]
        out["candidates"][nm] = {
            "n_feat": len(fs), "features": fs,
            "val_R2": float(1 - np.sum((y[va] - pv_) ** 2)
                            / np.sum((y[va] - ybar_tr) ** 2)),
            "val_MSE": float(np.mean((y[va] - pv_) ** 2))}
    winner = max(out["candidates"], key=lambda k: out["candidates"][k]["val_R2"])
    out["winner"] = winner
    fs = cand[winner]
    j = [FE.index(f) for f in fs]

    # ---- 勝者だけを OOS で 1 回評価(ウォークフォワード)------------------
    Xs = np.column_stack([np.ones(len(y)), X[:, j]])
    fit = tr | va
    XtX = Xs[fit].T @ Xs[fit]; Xty = Xs[fit].T @ yn[fit]
    p = np.full(len(y), np.nan)
    oos_days = sorted(set(dts[oo].tolist()))
    wf = []
    for d in oos_days:
        m = dts == d
        bw = np.linalg.solve(XtX, Xty)
        p[m] = Xs[m] @ bw * vp[m]
        wf.append({"dt": d, "R2": float(1 - np.sum((y[m] - p[m]) ** 2)
                                        / np.sum((y[m] - ybar_tr) ** 2))})
        XtX += Xs[m].T @ Xs[m]; Xty += Xs[m].T @ yn[m]
    r2d = np.array([w["R2"] for w in wf])
    lo, hi, ple = block_boot(r2d)
    out["oos"] = {
        "R2_pooled": float(1 - np.sum((y[oo] - p[oo]) ** 2) / np.sum((y[oo] - ybar_tr) ** 2)),
        "MSE": float(np.mean((y[oo] - p[oo]) ** 2)),
        "corr": float(np.corrcoef(p[oo], y[oo])[0, 1]),
        "R2_daily_mean": float(r2d.mean()), "R2_daily_median": float(np.median(r2d)),
        "pos_days": int((r2d > 0).sum()), "n_days": len(r2d), "ci": [lo, hi], "p_le0": ple,
        "sign_test_p": float(stats.binomtest(int((r2d > 0).sum()), len(r2d), 0.5,
                                             alternative="greater").pvalue),
        "daily": wf}
    out["final_coef"] = {("const" if i == 0 else fs[i - 1]): float(v)
                         for i, v in enumerate(np.linalg.solve(XtX, Xty))}
    out["standardization"] = {f: {"mean": float(mu[FE.index(f)]), "sd": float(sd[FE.index(f)])}
                              for f in fs}

    # ---- 経済性(勝者のみ)----------------------------------------------
    econ = []
    for tq in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        thr = np.quantile(np.abs(p[oo]), tq) if tq > 0 else 0.0
        pos = np.where(np.abs(p) > thr, np.sign(p), 0.0)
        g, t_, hs = [], [], []
        for d in oos_days:
            m = oo & (dts == d)
            g.append(float((pos[m] * y[m]).sum()))
            t_.append(float(np.abs(np.diff(np.concatenate([[0.], pos[m], [0.]]))).sum()))
            hs.append(float(np.mean(spread[m]) / 2))
        g, t_ = np.array(g), np.array(t_)
        be = float(g.sum() / max(t_.sum(), 1))
        row = {"閾値分位": tq, "回転数_日": float(t_.mean()), "粗利_bp_日": float(g.mean()),
               "損益分岐コスト_bp": be}
        for cn, c in (("maker", FEE_BP), ("taker", float(np.mean(hs)) + FEE_BP)):
            net = g - c * t_
            l2, h2, p2 = block_boot(net)
            row[cn] = {"cost_bp": c, "net_bp_day": float(net.mean()),
                       "median": float(np.median(net)), "pos_days": int((net > 0).sum()),
                       "sharpe": float(net.mean() / net.std(ddof=1)), "ci": [l2, h2],
                       "p_le0": p2}
        econ.append(row)
    out["economics"] = econ
    (D / "slope_model_final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                              encoding="utf-8")

    print("=== 段階 17: 候補の比較(選択は検証期間のみ)===")
    print(f"{'候補':<24}{'変数':>5}{'検証 R2':>11}{'検証 MSE':>11}")
    for k, v in out["candidates"].items():
        mk = " ←採用" if k == winner else ""
        print(f"{k:<24}{v['n_feat']:>5}{v['val_R2']:>11.6f}{v['val_MSE']:>11.4f}{mk}")
    o = out["oos"]
    print(f"\n=== 採用モデルの OOS(28 日・ウォークフォワード)===")
    print(f"  R2(プール) {o['R2_pooled']:+.6f}   相関 {o['corr']:+.4f}   MSE {o['MSE']:.4f}")
    print(f"  日次 R2 平均 {o['R2_daily_mean']:+.6f} 中央 {o['R2_daily_median']:+.6f} "
          f"正 {o['pos_days']}/{o['n_days']}")
    print(f"  CI[{o['ci'][0]:+.6f},{o['ci'][1]:+.6f}] p(<=0)={o['p_le0']:.4f} "
          f"符号検定 p={o['sign_test_p']:.4f}")
    print("\n=== 経済的有意性 ===")
    print(f"{'閾値':>6}{'回転/日':>9}{'粗利bp/日':>11}{'分岐コスト':>11}"
          f"{'maker純益':>11}{'黒字':>7}{'SR':>6}{'taker純益':>11}{'黒字':>7}{'SR':>6}")
    for e in econ:
        print(f"{e['閾値分位']:>6.2f}{e['回転数_日']:>9.0f}{e['粗利_bp_日']:>11.1f}"
              f"{e['損益分岐コスト_bp']:>11.4f}"
              f"{e['maker']['net_bp_day']:>11.1f}{e['maker']['pos_days']:>4}/28"
              f"{e['maker']['sharpe']:>6.2f}"
              f"{e['taker']['net_bp_day']:>11.1f}{e['taker']['pos_days']:>4}/28"
              f"{e['taker']['sharpe']:>6.2f}")
    print(f"\n  テイカーの片道コスト実測 = {econ[0]['taker']['cost_bp']:.4f} bp "
          f"(半スプレッド + 手数料 {FEE_BP})")


if __name__ == "__main__":
    main()
