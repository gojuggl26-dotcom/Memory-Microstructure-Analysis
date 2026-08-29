"""期間体制 5 通り × OBI 5 種の制限付き 3 次スプライン回帰(knot は機械選択)。

【模型】
  y = α + f(OBI) + ε      f は制限付き 3 次スプライン(restricted cubic spline)
  y = (log mid(T+1s) − log mid(T)) × 1e4 [bp]、x = 時刻 T の板から作る OBI。
  **x の確定時刻 = y の期間の開始時刻**(CLAUDE.md 時間契約)。

【期間体制 5 通り】
  98 日を暦順に 5 等分(約 20 日ずつ)。この市場はスプレッドが 142.7bp → 0.46bp と
  変わっており、暦順ブロックがそのまま体制の推移になる。境界は日数で機械的に決める。

【★knot の機械的調節 — 2 段階とも自動】
  (1) 位置: 訓練分割内の x の**等間隔分位点**([5%, 95%] を K 等分)に置く。
      分位点は**訓練分割だけ**から計算する(全標本から作ると漏洩 — CLAUDE.md 実装義務 4)
  (2) 個数: K ∈ {0(線形), 3, 4, 5, 6, 7, 9, 12} を
      **日単位 GroupKFold(5 分割)の標本外 R²** で選ぶ。
      行のシャッフルは隣接窓が相関するため禁止(CLAUDE.md)。日ごと丸ごと分割する。

【前処理】
  y・x とも訓練分割の 0.1%/99.9% でウィンザライズ(報告 29 と同じ手順)。
  resync 汚染行と crossed 行は除外(crossed は panel 生成時に除外済み)。

【帰無対照】
  プラセボ = x を日内で +3600 秒ずらして同じ手順。
  予測力が残るならルックアヘッド以外の混入を疑う。

【出力】 data/obi_spline_regimes.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OBIS = ["depth_one_level", "obi_2", "obi_3", "obi_5", "obi_10"]
LABEL = {"depth_one_level": "OBI(L=1)", "obi_2": "OBI(L=2)", "obi_3": "OBI(L=3)",
         "obi_5": "OBI(L=5)", "obi_10": "OBI(L=10)"}
# 初回は上限 12 で K* が張り付いた(= 格子が選択を拘束していた)ので拡張した。
# 「knot を機械的に調節する」以上、格子の端が選ばれた状態で止めてはいけない。
K_GRID = [0, 3, 4, 5, 6, 7, 9, 12, 16, 20, 25, 32, 40]
N_REGIME = 5
N_FOLD = 5
PLACEBO_SHIFT = 3600          # 1 秒グリッドなので 3600 行 = 1 時間


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """制限付き 3 次スプライン基底(Harrell)。K 個の knot → K-1 列(線形項を含む)。"""
    if len(knots) < 3:
        return x[:, None]
    t = knots
    K = len(t)
    denom = (t[-1] - t[-2])
    cols = [x]
    for j in range(K - 2):
        c = (np.maximum(x - t[j], 0) ** 3
             - np.maximum(x - t[-2], 0) ** 3 * (t[-1] - t[j]) / denom
             + np.maximum(x - t[-1], 0) ** 3 * (t[-2] - t[j]) / denom)
        cols.append(c / (t[-1] - t[0]) ** 2)      # 桁を揃える
    return np.column_stack(cols)


def fit_predict(xtr, ytr, xte, K):
    """訓練分割で knot・ウィンザ限界・係数を決め、試験分割を予測する。"""
    lo, hi = np.percentile(ytr, [0.1, 99.9])
    ytr = np.clip(ytr, lo, hi)
    xlo, xhi = np.percentile(xtr, [0.1, 99.9])
    xtr_c, xte_c = np.clip(xtr, xlo, xhi), np.clip(xte, xlo, xhi)
    knots = (np.percentile(xtr_c, np.linspace(5, 95, K)) if K >= 3
             else np.array([]))
    if K >= 3:                                   # 重複 knot を潰す
        knots = np.unique(knots)
        if len(knots) < 3:
            knots = np.array([])
    Btr = rcs_basis(xtr_c, knots)
    Bte = rcs_basis(xte_c, knots)
    Xtr = np.column_stack([np.ones(len(Btr)), Btr])
    Xte = np.column_stack([np.ones(len(Bte)), Bte])
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta, knots, beta


def r2(y, yhat):
    return 1.0 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_select(x, y, day, ks=K_GRID):
    """日単位 GroupKFold で K を選ぶ。返り値: {K: 標本外 R²}"""
    days = np.unique(day)
    fold = {d: i % N_FOLD for i, d in enumerate(days)}
    fid = np.array([fold[d] for d in day])
    out = {}
    for K in ks:
        num = den = 0.0
        for f in range(N_FOLD):
            tr, te = fid != f, fid == f
            if tr.sum() < 5000 or te.sum() < 1000:
                continue
            yh, _, _ = fit_predict(x[tr], y[tr], x[te], K)
            num += float(((y[te] - yh) ** 2).sum())
            den += float(((y[te] - y[te].mean()) ** 2).sum())
        out[K] = 1.0 - num / den if den > 0 else float("nan")
    return out


def per_day_compare(x, y, day, kbest):
    """K* と線形の標本外 SSE を日ごとに比べる(符号検定用)。CV 分割は cv_select と同一。"""
    days = np.unique(day)
    fold = {d: i % N_FOLD for i, d in enumerate(days)}
    fid = np.array([fold[d] for d in day])
    win = tot = 0
    for f in range(N_FOLD):
        tr, te = fid != f, fid == f
        if tr.sum() < 5000 or te.sum() < 1000:
            continue
        yh0, _, _ = fit_predict(x[tr], y[tr], x[te], 0)
        yhk, _, _ = fit_predict(x[tr], y[tr], x[te], kbest)
        yt, dt_ = y[te], day[te]
        for d in np.unique(dt_):
            m = dt_ == d
            if m.sum() < 500:
                continue
            tot += 1
            win += int(((yt[m] - yhk[m]) ** 2).sum() < ((yt[m] - yh0[m]) ** 2).sum())
    return win, tot


def main() -> None:
    fs = sorted(glob.glob(str(D / "panel_v2/dt=*/*.parquet")))
    cols = ["ts", "y"] + OBIS
    frames = []
    for f in fs:
        dt = f.split("dt=")[1][:10]
        p = pl.read_parquet(f, columns=cols)
        rs = glob.glob(str(D / f"panel_1s/dt={dt}/*.parquet"))
        # panel_1s は日によってスキーマが違う(旧版が混在)。列がある日だけ使う
        if rs and "is_resync_contaminated" in pl.scan_parquet(rs[0]).collect_schema().names():
            flag = pl.read_parquet(rs[0], columns=["ts", "is_resync_contaminated"])
            p = p.join(flag, on="ts", how="left").filter(
                pl.col("is_resync_contaminated").fill_null(False).not_()).drop(
                "is_resync_contaminated")
        frames.append(p.with_columns(dt=pl.lit(dt)))
    df = pl.concat(frames)
    days = sorted(df["dt"].unique().to_list())
    edges = np.array_split(np.arange(len(days)), N_REGIME)
    res: dict = {"n_days": len(days), "n_rows": len(df), "k_grid": K_GRID,
                 "regimes": []}
    for ri, idx in enumerate(edges):
        ds = [days[i] for i in idx]
        g = df.filter(pl.col("dt").is_in(ds))
        rec = {"regime": ri + 1, "days": f"{ds[0]}〜{ds[-1]}", "n_days": len(ds),
               "n_rows": len(g), "obi": {}}
        day = g["dt"].to_numpy()
        y_all = g["y"].to_numpy().astype(float)
        for c in OBIS:
            x_all = g[c].to_numpy().astype(float)
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            x, y, dd = x_all[ok], y_all[ok], day[ok]
            cv = cv_select(x, y, dd)
            valid = {k: v for k, v in cv.items() if np.isfinite(v)}
            kbest = max(valid, key=valid.get)
            # 選んだ K で全期間に当てて曲線の形を出す(訓練=当該体制全体)
            deciles = np.percentile(x, np.arange(5, 100, 10))
            yh, knots, _ = fit_predict(x, y, deciles, kbest)
            # プラセボ: x を日内で 1 時間ずらす
            xs = np.concatenate([np.roll(x[dd == d], PLACEBO_SHIFT)
                                 for d in np.unique(dd)])
            ys = np.concatenate([y[dd == d] for d in np.unique(dd)])
            ds_ = np.concatenate([dd[dd == d] for d in np.unique(dd)])
            cv_pl = cv_select(xs, ys, ds_, ks=[0, kbest] if kbest else [0])
            rec["obi"][c] = {
                "k_best": int(kbest), "cv": {str(k): v for k, v in cv.items()},
                "r2_linear": cv[0], "r2_spline": cv[kbest],
                "gain": (cv[kbest] / cv[0] if cv[0] > 0 else float("nan")),
                "knots": [float(v) for v in knots],
                "curve_x": [float(v) for v in deciles],
                "curve_y": [float(v) for v in yh],
                "placebo_r2": {str(k): v for k, v in cv_pl.items()},
                "corr": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x)),
                "k_at_grid_max": bool(kbest == max(K_GRID))}
            w, t = per_day_compare(x, y, dd, kbest) if kbest else (0, 0)
            rec["obi"][c]["days_spline_wins"] = [w, t]
            print(f"体制{ri+1} {LABEL[c]:>10}: K*={kbest:>2} "
                  f"R²(線形)={cv[0]:.5f} R²(spline)={cv[kbest]:.5f} "
                  f"偽={cv_pl.get(kbest, float('nan')):.5f} 勝ち日 {w}/{t}", flush=True)
        res["regimes"].append(rec)
    (D / "obi_spline_regimes.json").write_text(json.dumps(res, indent=1),
                                               encoding="utf-8")
    print("完了")


if __name__ == "__main__":
    main()
