"""日次ボラティリティは翌日を予測するか (item 47 の後段)。

HAR-RV (Corsi 2009) を土台に、跳び成分・半分散・他 4 系列のボラを 1 つずつ
足して増分を測る。

時間契約
--------
- 目的変数 y = log RVol(日 t+1)。説明変数はすべて **日 t の終了時点で確定**する
  量だけで作る (日 t までの RVol、その 5 日平均・22 日平均、日 t の跳び成分など)。
- 同じ日の RV で同じ日のリターンを説明するようなことはしない。
- 標本外は**時間で前後に切る** (前 60% で推定・後 40% で評価)。行を混ぜた
  k-fold は隣接日が相関しているので使わない。
- 標準化・回帰係数はすべて訓練期間だけで推定して評価期間に当てる。

検出力について
--------------
標本は 98 日、22 日成分を作ると使えるのは 75 日、標本外は 30 日しかない。
「有意でない」は「効果がない」ではない。t 値と一緒に標本数を必ず読むこと。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vol import REF_GRID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HAC_LAGS = 5
TRAIN_FRAC = 0.60
# 帰無対照: 追加項を円状にずらして同じ回帰を回す。1 回だけだと引きが悪いので
# 10〜65 日の全ずらしで t の分布を作り、経験的な棄却域を求める。
PLACEBO_SHIFTS = list(range(10, 66))


def hac_se(X: np.ndarray, e: np.ndarray, q: int) -> np.ndarray:
    """Newey-West (Bartlett) の頑健標準誤差。"""
    xi = X * e[:, None]
    S = xi.T @ xi
    for l in range(1, q + 1):
        w = 1.0 - l / (q + 1.0)
        G = xi[l:].T @ xi[:-l]
        S += w * (G + G.T)
    xtxi = np.linalg.pinv(X.T @ X)
    return np.sqrt(np.diag(xtxi @ S @ xtxi))


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    r2 = 1.0 - e.var() / y.var()
    return b, e, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_csv(DATA / f"vol_daily_{tag}.csv").filter(
        pl.col("grid_s") == REF_GRID)

    W = D.pivot(on="series", index="dt", values="rvol").sort("dt")
    M = D.filter(pl.col("series") == "mid").sort("dt")
    W = W.join(M.select("dt", "rv", "bv", "rs_up", "rs_dn", "oc_bp"), on="dt")

    lg = lambda v: np.log(np.maximum(v, 1e-12))
    y_all = lg(W["mid"].to_numpy())
    n = y_all.size
    jump = lg(np.sqrt(np.maximum(W["rv"].to_numpy() - W["bv"].to_numpy(), 0)) + 1e-9)
    add = {"跳び成分 √(RV−BV)": jump,
           "下向き半分散 √RS−": lg(np.sqrt(W["rs_dn"].to_numpy())),
           "上向き半分散 √RS+": lg(np.sqrt(W["rs_up"].to_numpy())),
           "microprice のボラ": lg(W["micro"].to_numpy()),
           "スプレッドのボラ": lg(W["spread"].to_numpy()),
           "板圧力のボラ": lg(W["bp"].to_numpy()),
           "OFI のボラ": lg(W["ofi"].to_numpy())}

    # HAR の 3 成分 (すべて日 t までの後ろ向き平均)
    d1 = y_all
    d5 = np.array([y_all[max(0, i - 4):i + 1].mean() for i in range(n)])
    d22 = np.array([y_all[max(0, i - 21):i + 1].mean() for i in range(n)])
    s = 22                      # 22 日平均が揃うまで捨てる
    idx = np.arange(s, n - 1)   # t = idx, 目的変数は t+1
    y = y_all[idx + 1]
    base = np.column_stack([np.ones(idx.size), d1[idx], d5[idx], d22[idx]])
    ntr = int(round(idx.size * TRAIN_FRAC))

    rows = []
    b0, e0, r2_0 = ols(y, base)
    se0 = hac_se(base, e0, HAC_LAGS)
    for i, nm in enumerate(["定数", "RVol(t)", "RVol(5日平均)", "RVol(22日平均)"]):
        rows.append({"model": "HAR", "term": nm, "coef": float(b0[i]),
                     "t": float(b0[i] / se0[i]), "r2": r2_0, "n": int(idx.size),
                     "oos_r2_vs_mean": np.nan, "oos_r2_vs_rw": np.nan,
                     "oos_r2_vs_best_const": np.nan, "oos_r2_vs_har": np.nan,
                     "placebo_p": np.nan, "placebo_t_hi": np.nan})

    # 標本外。基準は 3 つ置く:
    #   (1) 訓練期間の平均 = 何もしない
    #   (2) ランダムウォーク = 明日は今日と同じ
    #   (3) 評価窓の真の平均 = 定数予測の理論上限 (反則。これに勝てば本物)
    yt = y[ntr:]
    btr, _, _ = ols(y[:ntr], base[:ntr])
    f_har = base[ntr:] @ btr
    B = {"mean": np.full(yt.size, y[:ntr].mean()),
         "rw": base[ntr:, 1],
         "best_const": np.full(yt.size, yt.mean())}
    sse = {k: float(((yt - v) ** 2).sum()) for k, v in B.items()}
    sse_har = float(((yt - f_har) ** 2).sum())
    rows.append({"model": "HAR", "term": "(標本外)", "coef": np.nan, "t": np.nan,
                 "r2": r2_0, "n": int(idx.size),
                 "oos_r2_vs_mean": 1.0 - sse_har / sse["mean"],
                 "oos_r2_vs_rw": 1.0 - sse_har / sse["rw"],
                 "oos_r2_vs_best_const": 1.0 - sse_har / sse["best_const"],
                 "oos_r2_vs_har": 0.0, "placebo_p": np.nan,
                 "placebo_t_hi": np.nan})

    for nm, v in add.items():
        X = np.column_stack([base, v[idx]])
        b, e, r2 = ols(y, X)
        se = hac_se(X, e, HAC_LAGS)
        tobs = float(b[-1] / se[-1])
        bt, _, _ = ols(y[:ntr], X[:ntr])
        s_add = float(((yt - X[ntr:] @ bt) ** 2).sum())
        # 帰無分布: ずらした追加項でも同じ大きさの t が出るか
        tn = []
        for sh in PLACEBO_SHIFTS:
            Xp = np.column_stack([base, np.roll(v, sh)[idx]])
            bp_, ep_, _ = ols(y, Xp)
            sep = hac_se(Xp, ep_, HAC_LAGS)
            tn.append(abs(float(bp_[-1] / sep[-1])))
        tn = np.array(tn)
        rows.append({"model": nm, "term": "追加項", "coef": float(b[-1]),
                     "t": tobs, "r2": r2, "n": int(idx.size),
                     "oos_r2_vs_mean": 1.0 - s_add / sse["mean"],
                     "oos_r2_vs_rw": 1.0 - s_add / sse["rw"],
                     "oos_r2_vs_best_const": 1.0 - s_add / sse["best_const"],
                     "oos_r2_vs_har": 1.0 - s_add / sse_har,
                     "placebo_p": float((tn >= abs(tobs)).mean()),
                     "placebo_t_hi": float(np.quantile(tn, 0.95))})

    R = pl.DataFrame(rows)
    R.write_csv(DATA / f"vol_signal_{tag}.csv")
    print(f"標本 {idx.size} 日 (訓練 {ntr} / 評価 {idx.size - ntr})  "
          f"検定数 {len(add)}  帰無対照は各 {len(PLACEBO_SHIFTS)} 通りのずらし")
    print("placebo_p = ずらした追加項で |t| が観測以上になった割合 "
          "(これが経験的な p 値。名目の t は信用しない)")
    with pl.Config(tbl_rows=30, tbl_width_chars=140, fmt_str_lengths=30):
        print(R)


if __name__ == "__main__":
    main()
