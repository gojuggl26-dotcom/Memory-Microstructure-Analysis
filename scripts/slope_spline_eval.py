"""L10 book_slope のスプライン回帰を 7 つの予測地平で評価する。

【模型】
  線形       y = α + b_B·S^Bid + b_A·S^Ask
  スプライン  y = α + f_B(S^Bid) + f_A(S^Ask)

【★評価】
  真の標本外 R²(係数は学習期間のみ、分母の平均も学習期間)。拡張窓。
  節点は最初の 20 日の分位に固定 → 全評価日にとって過去のデータのみ。

【★注意】
  地平を伸ばすと y の窓が重なる(1s ごとに評価点があるのに 15s 先を見る等)。
  重なりは誤差の自己相関を生むが、**日次の符号一貫性**で推論するので
  プールした t 統計量のような影響は受けない。ただし
  「独立な観測 n 個ぶんの情報がある」とは言えないので、R² の水準は
  地平が長いほど楽観側に出る可能性がある。この点は報告に明記する。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import BSpline

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "slope_spline"
DEG = 3
KNOTS = 8
MIN_TRAIN = 20
RIDGE_S = 1e-6
HZN = ["1ms", "100ms", "1s", "3s", "5s", "10s", "15s"]


def make_knots(v, k):
    q = np.quantile(v, np.linspace(0, 1, k + 2))
    lo, hi = q[0], q[-1]
    if hi - lo < 1e-9:
        lo, hi = lo - 1e-3, hi + 1e-3
    return np.concatenate([[lo] * (DEG + 1), np.unique(q[1:-1]), [hi] * (DEG + 1)])


def basis(x, t):
    return BSpline.design_matrix(np.clip(x, t[DEG], t[-DEG - 1]), t, DEG).toarray()[:, :-1]


def main() -> None:
    fs = sorted(glob.glob(str(OUT / "*.npz")))
    n = len(fs)
    print(f"=== {n} 日 / イベント粒度 / L10 book_slope ===")
    samp_b, samp_a = [], []
    for f in fs[:MIN_TRAIN]:
        z = np.load(f)
        samp_b.append(z["sb"][::10]); samp_a.append(z["sa"][::10])
    tb = make_knots(np.concatenate(samp_b), KNOTS)
    ta = make_knots(np.concatenate(samp_a), KNOTS)
    print(f"節点は最初の {MIN_TRAIN} 日の分位(K={KNOTS})。以降固定\n")

    stat = {("線形", h): [] for h in range(len(HZN))}
    stat |= {("スプライン", h): [] for h in range(len(HZN))}
    npts = []
    for f in fs:
        z = np.load(f)
        sb = z["sb"].astype(np.float64); sa = z["sa"].astype(np.float64)
        Y = z["Y"].astype(np.float64)
        Xl = np.column_stack([np.ones(len(sb)), sb, sa])
        Xs = np.column_stack([np.ones(len(sb)), basis(sb, tb), basis(sa, ta)])
        npts.append(len(sb))
        for h in range(len(HZN)):
            y = Y[:, h]
            m = np.isfinite(y)
            for nm, X in (("線形", Xl), ("スプライン", Xs)):
                Xm, ym = X[m], y[m]
                stat[(nm, h)].append({"XtX": Xm.T @ Xm, "Xty": Xm.T @ ym,
                                      "yty": float(ym @ ym), "sy": float(ym.sum()),
                                      "n": int(m.sum())})

    def r2(s, b, ybar):
        sse = s["yty"] - 2 * b @ s["Xty"] + b @ s["XtX"] @ b
        sst = s["yty"] - 2 * ybar * s["sy"] + s["n"] * ybar ** 2
        return float(1 - sse / sst) if sst > 0 else 0.0

    res = {k: [] for k in stat}
    for i in range(MIN_TRAIN, n):
        for k in stat:
            tr = stat[k][:i]
            A = sum(s["XtX"] for s in tr); c = sum(s["Xty"] for s in tr)
            tot = sum(s["n"] for s in tr)
            ybar = sum(s["sy"] for s in tr) / max(tot, 1)
            b = np.linalg.solve(A + RIDGE_S * np.trace(A) / len(A) * np.eye(len(A)), c)
            res[k].append(r2(stat[k][i], b, ybar))

    print(f"評価 {n - MIN_TRAIN} 日 / 1 日あたり平均 {int(np.mean(npts)):,} 点\n")
    # ★日ごとの比 b/a は分母が 0 近傍で暴れるので、中央値の比と差で報告する
    print(f"{'地平':<8}{'線形 R²':>12}{'スプライン R²':>14}{'中央値の比':>11}"
          f"{'差の中央値':>12}{'線形超え':>10}{'p':>10}")
    out = {}
    for h, nm in enumerate(HZN):
        a = np.array(res[("線形", h)]); b_ = np.array(res[("スプライン", h)])
        ma, mb = float(np.median(a)), float(np.median(b_))
        rel = mb / ma if abs(ma) > 1e-12 else np.nan
        d_ = b_ - a
        pos = int((d_ > 0).sum())
        p = float(stats.binomtest(pos, len(d_), .5, alternative="greater").pvalue)
        mk = " ★" if p < 0.05 else ""
        print(f"{nm:<8}{ma:>12.5f}{mb:>14.5f}{rel:>11.3f}{np.median(d_):>12.5f}"
              f"{pos:>6}/{len(d_):<3}{p:>10.4f}{mk}")
        out[nm] = {"linear": ma, "spline": mb, "rel": float(rel),
                   "diff": float(np.median(d_)), "pos": pos, "n": len(d_), "p": p,
                   "spline_pos": int((b_ > 0).sum()), "linear_pos": int((a > 0).sum())}
    print("\n★ = スプラインが線形を有意に上回る")

    # 形状(全期間当てはめ・参考)
    h1 = HZN.index("1s")
    A = sum(s["XtX"] for s in stat[("スプライン", h1)])
    c = sum(s["Xty"] for s in stat[("スプライン", h1)])
    b = np.linalg.solve(A + RIDGE_S * np.trace(A) / len(A) * np.eye(len(A)), c)
    nb = len(tb) - DEG - 2
    gb = np.linspace(np.quantile(np.concatenate(samp_b), .02),
                     np.quantile(np.concatenate(samp_b), .98), 21)
    ga = np.linspace(np.quantile(np.concatenate(samp_a), .02),
                     np.quantile(np.concatenate(samp_a), .98), 21)
    fb = basis(gb, tb) @ b[1:1 + nb]; fa = basis(ga, ta) @ b[1 + nb:]
    out["shape"] = {"grid_bid": gb.tolist(), "f_bid": (fb - fb.mean()).tolist(),
                    "grid_ask": ga.tolist(), "f_ask": (fa - fa.mean()).tolist()}
    print(f"\n【参考】1s 地平の推定形状(全期間当てはめ・中心化)")
    print(f"  S^Bid {gb[0]:.2f}→{gb[-1]:.2f} bp : " +
          " ".join(f"{(fb-fb.mean())[j]:+.4f}" for j in (0, 5, 10, 15, 20)))
    print(f"  S^Ask {ga[0]:.2f}→{ga[-1]:.2f} bp : " +
          " ".join(f"{(fa-fa.mean())[j]:+.4f}" for j in (0, 5, 10, 15, 20)))
    # ★バックテスト用: **最初の MIN_TRAIN 日だけ**で当てはめた模型を保存する。
    #   バックテスト期間(それ以降の日)にとって完全に過去のデータのみ。
    A = sum(s["XtX"] for s in stat[("スプライン", h1)][:MIN_TRAIN])
    c = sum(s["Xty"] for s in stat[("スプライン", h1)][:MIN_TRAIN])
    bt = np.linalg.solve(A + RIDGE_S * np.trace(A) / len(A) * np.eye(len(A)), c)
    out["model_for_backtest"] = {
        "train_days": [Path(f).stem for f in fs[:MIN_TRAIN]],
        "knots_bid": tb.tolist(), "knots_ask": ta.tolist(),
        "coef": bt.tolist(), "deg": DEG, "horizon": "1s"}
    (D / "slope_spline.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
