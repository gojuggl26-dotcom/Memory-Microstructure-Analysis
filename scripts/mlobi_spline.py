"""各レベルの OBI を説明変数とした加法スプライン回帰(L1〜L10)。

【何をするか】
  線形の MLOBI は  y = α + Σ_L w_L · I_L  だった。
  これを各レベルごとに非線形な関数へ置き換える(加法モデル / GAM)。

      y = α + Σ_{L=1..10} f_L(I_L) + ε ,   f_L は 3 次 B スプライン

  重み 1 個ではなく**曲線 1 本**を各段に割り当てる。
  「不均衡が ±1 に近い極端な局面でだけ効く」といった非線形性を捉えられる。

【比較対象】
  線形 w3(重回帰 OLS) / 線形 w4(Ridge) / スプライン(節点数を変えた 3 通り)

【★ルックアヘッドの排除】
  ・節点は**最初の 20 日(全評価日にとって学習期間)の分位**に固定する
  ・係数は評価日より前のデータだけで推定(拡張窓)
  ・R² の分母の平均も学習期間のもの
  ・スプライン基底は節点だけで決まるので、評価日のデータは一切使わない

【自由度の管理】
  節点 K に対し 1 段あたり K+4 個の基底。10 段で 10(K+4) 個。
  K=3 なら 70 個。観測は 1 日 85,000 点 × 20 日以上なので母数比は十分小さいが、
  **線形(10 個)より 7 倍**なので過学習の余地は増える。標本外で確かめる。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import BSpline

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "obi_levels"
NL = 10
BURN = 7
MIN_TRAIN = 20
DEG = 3
KNOTS = [2, 3, 5, 8, 12, 20]          # 内部節点の数(飽和点を探す)
RIDGE_S = 1e-6             # 数値安定用の微小リッジ


def basis_for(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """節点列 t に対する B スプライン設計行列(最後の 1 本は識別のため落とす)。"""
    m = BSpline.design_matrix(np.clip(x, t[DEG], t[-DEG - 1]), t, DEG).toarray()
    return m[:, :-1]


def make_knots(vals: np.ndarray, k: int) -> np.ndarray:
    q = np.quantile(vals, np.linspace(0, 1, k + 2))
    lo, hi = q[0], q[-1]
    if hi - lo < 1e-9:
        lo, hi = lo - 1e-3, hi + 1e-3
    inner = np.unique(q[1:-1])
    return np.concatenate([[lo] * (DEG + 1), inner, [hi] * (DEG + 1)])


def main() -> None:
    files = sorted(glob.glob(str(OUT / "*.npz")))[BURN:]
    n = len(files)
    print(f"=== {n} 日 / 加法スプライン y = α + Σ f_L(I_L) ===")

    # ---- 節点は最初の MIN_TRAIN 日から作る(全評価日にとって過去)----
    samp = []
    for f in files[:MIN_TRAIN]:
        z = np.load(f)
        I = z["I"].astype(np.float64)
        samp.append(I[::20])          # 間引いて分位を推定
    samp = np.vstack(samp)
    knots = {k: [make_knots(samp[:, L], k) for L in range(NL)] for k in KNOTS}
    print(f"節点は最初の {MIN_TRAIN} 日の分位から作成(以降固定)")
    for k in KNOTS:
        print(f"  内部節点 {k} → 基底 {sum(len(t) - DEG - 2 for t in knots[k])} 個")

    # ---- 各日の十分統計量を一度だけ作る ----
    print("\n各日の十分統計量を作成中...", flush=True)
    models = {"線形(10 変数)": None} | {f"スプライン K={k}": k for k in KNOTS}
    stat = {m: [] for m in models}
    for f in files:
        z = np.load(f)
        I = z["I"].astype(np.float64); y = z["y"].astype(np.float64)
        for m, k in models.items():
            X = (np.column_stack([np.ones(len(y)), I]) if k is None
                 else np.column_stack([np.ones(len(y))]
                                      + [basis_for(I[:, L], knots[k][L]) for L in range(NL)]))
            stat[m].append({"XtX": X.T @ X, "Xty": X.T @ y,
                            "yty": float(y @ y), "sy": float(y.sum()), "n": len(y)})
    print("完了\n")

    def r2_oos(s, b, ybar):
        sse = s["yty"] - 2 * b @ s["Xty"] + b @ s["XtX"] @ b
        sst = s["yty"] - 2 * ybar * s["sy"] + s["n"] * ybar ** 2
        return float(1 - sse / sst)

    res = {m: [] for m in models}
    for i in range(MIN_TRAIN, n):
        for m in models:
            tr = stat[m][:i]
            A = sum(s["XtX"] for s in tr); c = sum(s["Xty"] for s in tr)
            ybar = sum(s["sy"] for s in tr) / sum(s["n"] for s in tr)
            b = np.linalg.solve(A + RIDGE_S * np.trace(A) / len(A) * np.eye(len(A)), c)
            res[m].append(r2_oos(stat[m][i], b, ybar))

    base = np.array(res["線形(10 変数)"])
    print(f"標本外 R²(評価 {len(base)} 日・係数は学習期間のみ)")
    print(f"{'模型':<20}{'母数':>6}{'OOS R²':>11}{'線形=1':>9}"
          f"{'線形超え':>10}{'符号検定 p':>12}")
    out = {}
    for m in models:
        a = np.array(res[m])
        npar = len(stat[m][0]["Xty"])
        rel = np.median(np.where(np.abs(base) > 1e-12, a / base, np.nan))
        d_ = a - base
        pos = int((d_ > 0).sum())
        p = float(stats.binomtest(pos, len(d_), .5, alternative="greater").pvalue)
        mk = " ★" if p < 0.05 else ""
        print(f"{m:<20}{npar:>6}{np.median(a):>11.5f}{rel:>9.3f}"
              f"{pos:>6}/{len(d_):<3}{p:>12.4f}{mk}")
        out[m] = {"n_param": npar, "oos_r2": float(np.median(a)), "rel": float(rel),
                  "pos": pos, "n": len(d_), "p": p}
    print("\n★ = 線形を有意に上回る")

    # ---- 採用候補の形を出す(全期間で当てはめ、実装の参考)----
    kbest = max(KNOTS, key=lambda k: out[f"スプライン K={k}"]["rel"])
    m = f"スプライン K={kbest}"
    A = sum(s["XtX"] for s in stat[m]); c = sum(s["Xty"] for s in stat[m])
    b = np.linalg.solve(A + RIDGE_S * np.trace(A) / len(A) * np.eye(len(A)), c)
    grid = np.linspace(-0.95, 0.95, 21)
    shapes = {}
    o = 1
    for L in range(NL):
        t = knots[kbest][L]
        nb = len(t) - DEG - 2
        f_ = basis_for(grid, t) @ b[o:o + nb]
        shapes[f"L{L+1}"] = (f_ - f_.mean()).tolist()
        o += nb
    out["shape_grid"] = grid.tolist()
    out["shapes"] = shapes
    out["knots_best"] = kbest
    print(f"\n【参考】K={kbest} の推定形状(全期間当てはめ・中心化、I_L = −0.95 → +0.95)")
    for L in (0, 1, 4, 9):
        v = shapes[f"L{L+1}"]
        print(f"  L{L+1:<2} " + " ".join(f"{v[j]:+.4f}" for j in (0, 5, 10, 15, 20)))
    (D / "mlobi_spline.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
