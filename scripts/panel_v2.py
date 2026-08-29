"""新特徴量をパネルへ統合し(panel_v2)、全 99 日で検証したうえで Lasso を回し直す。

検証:
  1. book_slope_diff の y との相関が 8 日での −0.2313 を全 99 日でも再現するか
  2. 既存 25 変数への線形従属性(全期間)
  3. 新旧の Lasso 比較(同じ時間分割・同じ手順)

★時間契約は panel_1s と同一。学習は前 70 日、後 28 日は一切触らない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso, lasso_path

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_1s"
NEW = D / "new_features"
OUT = D / "panel_v2"
ADD = ["book_slope_bid", "book_slope_ask", "book_slope_diff",
       "hhi_new_bid", "hhi_new_ask", "hhi_diff",
       "top1_share_new_bid", "top1_share_new_ask", "top1_diff",
       "reject_rate", "alo_reject_rate"]
TRAIN_DAYS = 70
WINSOR = 0.001


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for k, dt in enumerate(days):
        p = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet")
        fl = PANEL / f"dt={dt}" / "resync_flag.parquet"
        if fl.exists():
            p = (p.join(pl.read_parquet(fl), on="ts", how="left")
                  .with_columns(pl.col("is_resync_contaminated").fill_null(False))
                  .filter(~pl.col("is_resync_contaminated"))
                  .drop("is_resync_contaminated"))
        nf = NEW / f"{dt}.parquet"
        if not nf.exists():
            continue
        j = p.join(pl.read_parquet(nf).select(["ts", *ADD]), on="ts", how="inner")
        d = OUT / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        j.write_parquet(d / "part-000.parquet", compression="zstd")
        frames.append(j.with_columns(pl.lit(k).alias("day")))
    df = pl.concat(frames)
    old = [c for c in df.columns if c not in ("ts", "y", "day", *ADD)]
    feats = old + ADD
    print(f"rows={df.height:,} 既存={len(old)} 追加={len(ADD)}")

    day = df["day"].to_numpy()
    y = df["y"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float32)

    # ---- 1. 日次相関(全 99 日) ----
    res: dict = {"n_rows": int(df.height), "old_features": old, "added": ADD}
    cor = {}
    for j, c in enumerate(feats):
        v = []
        for k in range(int(day.max()) + 1):
            m = day == k
            if m.sum() < 1000 or X[m, j].std() == 0 or y[m].std() == 0:
                continue
            v.append(float(np.corrcoef(X[m, j], y[m])[0, 1]))
        if v:
            cor[c] = {"median": float(np.median(v)), "days": len(v),
                      "n_neg": int(sum(1 for z in v if z < 0))}
    res["daily_corr_full_period"] = cor

    # ---- 2. 既存への線形従属性(全期間) ----
    def prep(A):
        lo, hi = np.quantile(A, WINSOR, axis=0), np.quantile(A, 1 - WINSOR, axis=0)
        A = np.clip(A, lo, hi)
        s = A.std(0)
        s[s == 0] = 1.0
        return (A - A.mean(0)) / s
    io = [feats.index(c) for c in old]
    ia = [feats.index(c) for c in ADD]
    Xs = prep(X[:, io])
    Zs = prep(X[:, ia])
    Xd = np.column_stack([np.ones(len(Xs), dtype=np.float32), Xs])
    B = np.linalg.solve(Xd.T @ Xd + 1e-6 * np.eye(Xd.shape[1]), Xd.T @ Zs)
    r2 = 1 - (Zs - Xd @ B).var(0) / np.maximum(Zs.var(0), 1e-12)
    res["r2_on_existing"] = {c: float(r2[i]) for i, c in enumerate(ADD)}

    # ---- 3. Lasso(lasso_report.md の変種 C と同じ手順) ----
    sig = np.ones(int(day.max()) + 1)
    for d0 in range(sig.size):
        m = day == d0
        if m.sum() > 100:
            sig[d0] = max(float(y[m].std()), 1e-6)
    yz = y / np.concatenate([[sig[0]], sig[:-1]])[day]
    first = day == 0
    sub = np.zeros(day.size, dtype=bool)
    sub[::5] = True

    def run(cols: list[str], tag: str) -> dict:
        ii = [feats.index(c) for c in cols]
        tr = (day >= 40) & (day < TRAIN_DAYS) & (~first)
        te = (day >= TRAIN_DAYS) & (~first)
        lo, hi = (np.quantile(X[tr][:, ii], WINSOR, axis=0),
                  np.quantile(X[tr][:, ii], 1 - WINSOR, axis=0))
        Ztr = np.clip(X[tr][:, ii], lo, hi)
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd[sd == 0] = 1.0
        Ztr = ((Ztr - mu) / sd).astype(np.float64)
        Zte = ((np.clip(X[te][:, ii], lo, hi) - mu) / sd).astype(np.float64)
        ttr = yz[tr] - yz[tr].mean()
        tte = yz[te] - yz[tr].mean()
        alphas = np.logspace(0, -4, 30)
        bounds = np.linspace(55, TRAIN_DAYS, 6).astype(int)
        rel = np.ones((5, alphas.size))
        used = np.zeros(5, dtype=bool)
        for f in range(5):
            a, b = bounds[f], bounds[f + 1]
            m_tr = (day >= 40) & (day < a) & sub & (~first)
            m_va = (day >= a) & (day < b) & sub & (~first)
            if m_va.sum() < 1000 or m_tr.sum() < 1000:
                continue
            Za = ((np.clip(X[m_tr][:, ii], lo, hi) - mu) / sd).astype(np.float64)
            Zb = ((np.clip(X[m_va][:, ii], lo, hi) - mu) / sd).astype(np.float64)
            ta = yz[m_tr] - yz[tr].mean()
            tb = yz[m_va] - yz[tr].mean()
            _, cf, _ = lasso_path(Za, ta, alphas=alphas, max_iter=3000, tol=1e-4)
            rel[f] = np.mean((tb[:, None] - Zb @ cf) ** 2, axis=0) / float(np.mean(tb ** 2))
            used[f] = True
        alpha = float(alphas[int(np.argmin(rel[used].mean(0)))])
        mdl = Lasso(alpha=alpha, max_iter=20000, tol=1e-6).fit(Ztr, ttr)
        p = mdl.predict(Zte)
        dc = []
        dte = day[te]
        for d0 in np.unique(dte):
            m = dte == d0
            if m.sum() < 500 or p[m].std() == 0:
                continue
            dc.append(float(np.corrcoef(p[m], tte[m])[0, 1]))
        return {"tag": tag, "alpha": alpha, "n_features": len(cols),
                "n_selected": int((mdl.coef_ != 0).sum()),
                "coef": {c: float(v) for c, v in zip(cols, mdl.coef_) if v != 0},
                "oos_daily_corr_median": float(np.median(dc)),
                "oos_days_negative": int(sum(1 for z in dc if z < 0)), "oos_days": len(dc)}

    res["lasso_old"] = run(old, "既存 25")
    res["lasso_new"] = run(feats, "既存 25 + 追加 11")
    (D / "panel_v2_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    print("\n=== 追加特徴量の全 99 日での成績 ===")
    print(f"{'特徴量':>22} {'yとの相関':>10} {'符号逆':>9} {'既存へのR²':>10}")
    for c in ADD:
        v = cor.get(c)
        if v:
            print(f"{c:>22} {v['median']:>+10.4f} {v['n_neg']:>4}/{v['days']} "
                  f"{res['r2_on_existing'][c]:>10.4f}")
    print("\n=== Lasso 比較(標本外 28 日) ===")
    for k in ("lasso_old", "lasso_new"):
        r = res[k]
        print(f"[{r['tag']}] alpha={r['alpha']:.4g} 選択={r['n_selected']}/{r['n_features']} "
              f"日次相関 中央値={r['oos_daily_corr_median']:.4f} "
              f"負={r['oos_days_negative']}/{r['oos_days']}")
    print("\n新モデルの係数(絶対値順):")
    for c, v in sorted(res["lasso_new"]["coef"].items(), key=lambda kv: -abs(kv[1])):
        mark = " ★新規" if c in ADD else ""
        print(f"  {c:>22} {v:+.5f}{mark}")


if __name__ == "__main__":
    main()
