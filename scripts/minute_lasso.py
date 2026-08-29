"""Lasso で残った 9 変数を、1 分刻みのフォールドで回帰する(99 日分)。

2 通り:

  A. 分内当てはめ   … その 1 分(≒60 行)だけで Lasso を推定する。
                      係数が分ごとにどれだけ動くか、どの変数が何割の分で選ばれるかを見る。
                      当てはめ(in-sample)なので R² は上振れする。安定性の診断用

  B. 分単位ウォークフォワード … 分 m で推定 → **分 m+1 を予測**。
                      標準化の統計量も分 m のものだけを使う(★ルックアヘッドなし)。
                      直前の 1 分だけで学習する版と、直前 K 分で学習する版を比べる。
                      3 日ブロックで学習した既存モデル(minute_relmse.parquet)とも比較

標準化: 各分で X と y をその分の平均・標準偏差で正規化する(A)。
B では**学習側の分**の統計量を検証側にも適用する。

resync アーティファクト(variance_report.md §2)を含む行は除外。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_1s"
FEATS = ["r_prev", "depth_one_level", "voi_sum", "cancel_diff_n", "oi",
         "ofi_sum", "obi_3", "obi_2", "obi_5"]          # lasso_report.md で残った 9 変数
ALPHAS = (0.0161, 0.05)
MIN_N = 40
TRAIL_K = 30                                            # B の「直前 K 分」版
MIN_NS = 60_000_000_000


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    xs, ys, ms = [], [], []
    for dt in days:
        d = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet",
                            columns=["ts", "y", *FEATS]).sort("ts")
        fl = PANEL / f"dt={dt}" / "resync_flag.parquet"
        if fl.exists():
            d = (d.join(pl.read_parquet(fl), on="ts", how="left")
                  .with_columns(pl.col("is_resync_contaminated").fill_null(False))
                  .filter(~pl.col("is_resync_contaminated")).drop("is_resync_contaminated"))
        xs.append(d.select(FEATS).to_numpy().astype(np.float64))
        ys.append(d["y"].to_numpy().astype(np.float64))
        ms.append((d["ts"].to_numpy() // MIN_NS))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ms)


def main() -> None:
    X, y, minute = load()
    order = np.argsort(minute, kind="stable")
    X, y, minute = X[order], y[order], minute[order]
    uniq, start = np.unique(minute, return_index=True)
    end = np.append(start[1:], minute.size)
    print(f"rows={y.size:,} 分={uniq.size:,} 特徴量={len(FEATS)}", flush=True)

    res: dict = {"features": FEATS, "n_minutes_total": int(uniq.size)}

    # ---------- A. 分内当てはめ ----------
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        coefs = np.full((uniq.size, len(FEATS)), np.nan)
        r2 = np.full(uniq.size, np.nan)
        nn = np.zeros(uniq.size, dtype=np.int32)
        mdl = Lasso(alpha=alpha, max_iter=2000, tol=1e-4)
        for i, (a, b) in enumerate(zip(start, end)):
            n = b - a
            nn[i] = n
            if n < MIN_N:
                continue
            xb, yb = X[a:b], y[a:b]
            sx = xb.std(0)
            sy = yb.std()
            if sy <= 0 or np.any(sx <= 0):
                continue
            z = (xb - xb.mean(0)) / sx
            t = (yb - yb.mean()) / sy
            mdl.fit(z, t)
            coefs[i] = mdl.coef_
            p = z @ mdl.coef_ + mdl.intercept_
            r2[i] = 1.0 - float(np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
        ok = np.isfinite(r2)
        c = coefs[ok]
        res[f"in_minute_alpha_{alpha}"] = {
            "n_minutes": int(ok.sum()), "sec": time.perf_counter() - t0,
            "r2_median": float(np.median(r2[ok])), "r2_q90": float(np.quantile(r2[ok], .9)),
            "per_feature": {
                f: {"selected_share": float((c[:, j] != 0).mean()),
                    "median": float(np.median(c[:, j])),
                    "q25": float(np.quantile(c[:, j], .25)),
                    "q75": float(np.quantile(c[:, j], .75)),
                    "share_positive": float((c[:, j] > 0).mean()),
                    "share_negative": float((c[:, j] < 0).mean())}
                for j, f in enumerate(FEATS)},
        }
        print(f"[A] alpha={alpha}: {ok.sum():,} 分  {time.perf_counter()-t0:.0f} 秒  "
              f"R² 中央値 {np.median(r2[ok]):.4f}", flush=True)
        if alpha == ALPHAS[0]:
            np.save(D / "minute_lasso_coefs.npy", coefs)
            pl.DataFrame({"minute": uniq, "n": nn, "r2": r2,
                          **{f: coefs[:, j] for j, f in enumerate(FEATS)}}
                         ).write_parquet(D / "minute_lasso_coefs.parquet", compression="zstd")

    # ---------- B. 分単位ウォークフォワード ----------
    alpha = ALPHAS[0]
    mdl = Lasso(alpha=alpha, max_iter=2000, tol=1e-4)
    for name, k in (("prev1", 1), (f"prev{TRAIL_K}", TRAIL_K)):
        t0 = time.perf_counter()
        rel, cor, nsel = [], [], []
        for i in range(k, uniq.size):
            if uniq[i] - uniq[i - 1] != 1:          # 連続する分でないと因果性が崩れる
                continue
            a_tr, b_tr = start[i - k], end[i - 1]
            a_te, b_te = start[i], end[i]
            if (b_tr - a_tr) < MIN_N * k or (b_te - a_te) < MIN_N:
                continue
            if k > 1 and (uniq[i - 1] - uniq[i - k]) != k - 1:
                continue
            xtr, ytr = X[a_tr:b_tr], y[a_tr:b_tr]
            sx, sy = xtr.std(0), ytr.std()
            if sy <= 0 or np.any(sx <= 0):
                continue
            mx, my = xtr.mean(0), ytr.mean()
            mdl.fit((xtr - mx) / sx, (ytr - my) / sy)
            pred = (((X[a_te:b_te] - mx) / sx) @ mdl.coef_ + mdl.intercept_) * sy + my
            yt = y[a_te:b_te]
            sst = float(np.sum(yt ** 2))
            if sst <= 0:
                continue
            rel.append(float(np.sum((yt - pred) ** 2) / sst))
            if pred.std() > 0 and yt.std() > 0:
                cor.append(float(np.corrcoef(pred, yt)[0, 1]))
            nsel.append(int((mdl.coef_ != 0).sum()))
        r = np.array(rel)
        res[f"walkforward_{name}"] = {
            "n_minutes": int(r.size), "sec": time.perf_counter() - t0,
            "relmse_median": float(np.median(r)), "share_lt_1": float((r < 1).mean()),
            "relmse_q25": float(np.quantile(r, .25)), "relmse_q75": float(np.quantile(r, .75)),
            "corr_median": float(np.median(cor)) if cor else None,
            "n_selected_median": float(np.median(nsel)) if nsel else None,
        }
        print(f"[B] {name}: {r.size:,} 分  相対MSE 中央値 {np.median(r):.4f}  "
              f"<1 の割合 {(r < 1).mean():.3f}  {time.perf_counter()-t0:.0f} 秒", flush=True)

    # 既存(3 日ブロックで学習)との比較
    mr = pl.read_parquet(D / "minute_relmse.parquet").filter(
        (pl.col("n") >= MIN_N) & (pl.col("n_bad") == 0))
    rb = mr["relmse_raw"].to_numpy()
    rb = rb[np.isfinite(rb)]
    res["block_trained_baseline"] = {
        "n_minutes": int(rb.size), "relmse_median": float(np.median(rb)),
        "share_lt_1": float((rb < 1).mean())}
    (D / "minute_lasso.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("walk", "block"))},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
