"""book_slope モデルの監査 — OOS R² = 0.045 が本物かを疑う。

1 秒リターンの予測で R² 4.5%(相関 0.22)は極めて高い。
報告する前に、次の 4 つで潰す。

  A. プラセボ: 説明変数を**日内で 1 時間ずらす**。分布も自己相関も同じで
     時点の対応だけが壊れる。ここで R² が残れば手順の artifact である
  B. 外れ値依存: 上位 0.1% の |y| を除いても R² が残るか
  C. 分解: slope 系だけ / 統制だけ / 両方 で OOS R² を比べる。
     book_slope の寄与を分離する
  D. 地平の反証: y を 1 秒**前**のリターンに置き換える(未来ではなく過去)。
     これで高い R² が出るなら「同時性」を拾っているだけ
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "slope_model"
SHIFT_SEC = 3600


def fit_eval(Xf, yn, y, vp, fit, ev, ybar_tr):
    Xs = np.column_stack([np.ones(Xf.shape[0]), Xf])
    b = np.linalg.solve(Xs[fit].T @ Xs[fit], Xs[fit].T @ yn[fit])
    p = Xs[ev] @ b * vp[ev]
    return float(1.0 - np.sum((y[ev] - p) ** 2) / np.sum((y[ev] - ybar_tr) ** 2)), p


def main() -> None:
    meta = json.loads((D / "slope_model_data.json").read_text(encoding="utf-8"))
    SLOPE, CTRL = meta["features"]["slope"], meta["features"]["control"]
    FE = SLOPE + CTRL
    res = json.loads((D / "slope_model_results.json").read_text(encoding="utf-8"))
    sel = [FE.index(f) for f in res["lasso"]["selected"]]

    df = pl.read_parquet(OUT / "dataset.parquet").sort(["dt", "ts"])
    dts = df["dt"].to_numpy()
    sp = df["split"].to_numpy()
    ts = df["ts"].to_numpy()
    y = df["y"].to_numpy().astype(np.float64)
    vp = df["vol_prev"].to_numpy().astype(np.float64)
    X = df.select(FE).to_numpy().astype(np.float64)
    del df
    tr, va, oo = sp == "train", sp == "val", sp == "oos"
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    yn = y / vp
    ybar_tr = float(y[tr].mean())
    fit = tr | va
    out: dict = {}

    # ---- 本体(比較の基準)------------------------------------------
    base_r2, p_base = fit_eval(X[:, sel], yn, y, vp, fit, oo, ybar_tr)
    out["本体"] = base_r2

    # ---- A. プラセボ: 日内で 1 時間ずらす -----------------------------
    # 「その日の説明変数を 1 時間後の行に貼る」。日をまたがないので体制は同じ。
    Xp = X.copy()
    for d in sorted(set(dts.tolist())):
        m = np.where(dts == d)[0]
        k = int(np.searchsorted(ts[m], ts[m][0] + SHIFT_SEC * 1_000_000_000))
        if k <= 0 or k >= len(m):
            continue
        Xp[m] = np.roll(X[m], k, axis=0)
    out["プラセボ(説明変数を日内で+1h)"] = fit_eval(Xp[:, sel], yn, y, vp, fit, oo, ybar_tr)[0]
    del Xp

    # ---- B. 外れ値依存 -------------------------------------------------
    thr = np.quantile(np.abs(y[oo]), 0.999)
    keep = oo & (np.abs(y) <= thr)
    r2k = float(1.0 - np.sum((y[keep] - p_base[np.abs(y[oo]) <= thr]) ** 2)
                / np.sum((y[keep] - ybar_tr) ** 2))
    out["上位 0.1% の |y| を除外"] = r2k
    # 二乗和の集中度
    e2 = (y[oo] - ybar_tr) ** 2
    o = np.argsort(-e2)
    out["Σy² に占める上位 0.1% の割合"] = float(e2[o[:max(1, len(o) // 1000)]].sum() / e2.sum())

    # ---- C. 分解: slope 系だけ / 統制だけ --------------------------------
    si = [i for i in sel if FE[i] in SLOPE]
    ci = [i for i in sel if FE[i] in CTRL]
    out["slope 系のみ"] = fit_eval(X[:, si], yn, y, vp, fit, oo, ybar_tr)[0]
    out["統制のみ(slope 抜き)"] = fit_eval(X[:, ci], yn, y, vp, fit, oo, ybar_tr)[0]
    out["slope_diff 1 本のみ"] = fit_eval(X[:, [FE.index("slope_diff")]], yn, y, vp,
                                          fit, oo, ybar_tr)[0]

    # ---- D. 地平の反証: 過去リターンを当てにいく --------------------------
    # r_prev は「T−1s → T」。説明変数が T 時点の状態なら、
    # 過去を説明する力(同時性)は未来を予測する力より**強いはず**。
    yb = X[:, FE.index("r_prev")] * sd[FE.index("r_prev")] + mu[FE.index("r_prev")]
    ybn = yb / vp
    sel2 = [i for i in sel if FE[i] != "r_prev"]
    ybar_b = float(yb[tr].mean())
    out["【参考】同じ x で過去 1 秒を説明"] = fit_eval(
        X[:, sel2], ybn, yb, vp, fit, oo, ybar_b)[0]
    out["【参考】未来 1 秒(r_prev 抜き)"] = fit_eval(
        X[:, sel2], yn, y, vp, fit, oo, ybar_tr)[0]

    # ---- 日次の内訳(1 日に支配されていないか)---------------------------
    daily = []
    for d in sorted(set(dts[oo].tolist())):
        m = oo & (dts == d)
        pm = p_base[dts[oo] == d]
        daily.append({"dt": d, "R2": float(1 - np.sum((y[m] - pm) ** 2)
                                           / np.sum((y[m] - ybar_tr) ** 2))})
    out["日次 R² の最小/最大"] = [min(x["R2"] for x in daily), max(x["R2"] for x in daily)]

    (D / "slope_model_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    print("=== 監査 ===")
    for k, v in out.items():
        if isinstance(v, list):
            print(f"  {k:<34} [{v[0]:+.6f}, {v[1]:+.6f}]")
        else:
            print(f"  {k:<34} {v:+.6f}")


if __name__ == "__main__":
    main()
