"""99 日を 33 等分(3 日ずつ)し、ブロックごとの「分散」と「相対 MSE」の関係を測る。

問い: 相対 MSE(= MSE / 何もしないモデルの MSE)は、そのブロックの分散の大きさに
      影響されているか。

理屈:
  - 相関は尺度不変。分散が変わっても値は変わらない
  - 相対 MSE は、モデルが**そのブロック内で推定されていれば**尺度不変になる
  - しかし**別の期間で推定した固定モデル**を当てると尺度依存になる。
    訓練期のボラで較正された予測を分散の違うブロックに当てるので、
      ・検証ブロックの分散が訓練期より大きい → 予測が小さすぎ → 相対 MSE → 1(効かなく見える)
      ・検証ブロックの分散が訓練期より小さい → 予測が大きすぎ → 相対 MSE > 1(悪化して見える)
  この機構が働いているかを、生の y と日次ボラ正規化 y で比べて確認する。

分散の算出(最高解像度):
  イベント単位(BBO 変化ごと)に加え、1ms / 10ms / 100ms / 1s / 10s / 60s の
  等間隔グリッド上の対数リターンの分散。グリッドはゼロを含む全点で数える
  (疎表現で厳密計算。return_acf.py と同じ方式)。

★時間契約: 評価は拡大窓。ブロック k の評価には**ブロック 0..k−1 だけで学習**した
  モデルを使う。前処理(ウィンザライズ・標準化・ボラ)も訓練側だけから作る。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_1s"
N_BLOCK = 33
CLEAN = "--clean" in __import__("sys").argv
ALPHA = 0.0161          # lasso_report.md の変種 C と同じ α に固定して尺度効果だけを見る
WINSOR = 0.001
NS_DAY = 86_400_000_000_000
STEPS = {"1ms": 1_000_000, "10ms": 10_000_000, "100ms": 100_000_000,
         "1s": 1_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000}


def grid_var(tau: np.ndarray, lp: np.ndarray, t0: int, step: int) -> tuple[float, int]:
    """歩幅 step のグリッド上のリターン分散(ゼロ点も母数に含める厳密計算)。"""
    if tau.size < 3:
        return float("nan"), 0
    d = np.diff(lp)
    j = -((t0 - tau[1:]) // step)
    n_grid = NS_DAY // step
    j_first = -((t0 - tau[0]) // step)
    keep = (j > j_first) & (j <= n_grid) & (d != 0.0)
    n = int(n_grid - j_first)
    if n <= 10 or keep.sum() < 3:
        return float("nan"), n
    pos, inv = np.unique(j[keep], return_inverse=True)
    val = np.bincount(inv, weights=d[keep])
    s1, s2 = float(val.sum()), float((val * val).sum())
    return (s2 / n - (s1 / n) ** 2) * 1e8, n      # bp² 単位


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    per_day = []
    for k, dt in enumerate(days):
        t0 = int(datetime.strptime(dt, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
        mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                              columns=["ts", "mid", "is_crossed"])
              .filter(~pl.col("is_crossed")).sort("ts"))
        tau = mp["ts"].to_numpy()
        lp = np.log(mp["mid"].to_numpy())
        ev = np.diff(lp) * 1e4                     # イベント単位のリターン(bp)
        row = {"dt": dt, "day": k, "n_events": int(tau.size),
               "var_event": float(ev.var()) if ev.size > 2 else float("nan")}
        for name, step in STEPS.items():
            v, n = grid_var(tau, lp, t0, step)
            row[f"var_{name}"] = v
        per_day.append(row)
        if (k + 1) % 20 == 0:
            print(f"分散 {k+1}/{len(days)}", flush=True)
    dv = pl.DataFrame(per_day)

    # ---- パネルを読み、33 ブロックに割る ----
    frames = []
    for k, dt in enumerate(days):
        f = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet").with_columns(pl.lit(k).alias("day"))
        fl = PANEL / f"dt={dt}" / "resync_flag.parquet"
        if fl.exists():
            f = f.join(pl.read_parquet(fl), on="ts", how="left").with_columns(
                pl.col("is_resync_contaminated").fill_null(False))
        else:
            f = f.with_columns(pl.lit(False).alias("is_resync_contaminated"))
        frames.append(f)
    df = pl.concat(frames)
    if CLEAN:
        # ★resync アーティファクト(全体の 0.18% だが Σy² の 79%)を除く
        before = df.height
        df = df.filter(~pl.col("is_resync_contaminated"))
        print(f"resync 汚染を除去: {before:,} -> {df.height:,} 行")
    df = df.drop("is_resync_contaminated")
    feats = [c for c in df.columns if c not in ("ts", "y", "day")]
    day = df["day"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float32)   # 8.5M x 25 は float64 だと 1.6GB
    y = df["y"].to_numpy().astype(np.float64)
    n_days = int(day.max()) + 1
    blk_of_day = np.minimum((np.arange(n_days) * N_BLOCK) // n_days, N_BLOCK - 1)
    blk = blk_of_day[day]
    dv = dv.with_columns(pl.Series("block", blk_of_day[dv["day"].to_numpy()]))

    # 日次ボラ(前日の σ。因果的)
    sig = np.ones(n_days)
    for d0 in range(n_days):
        m = day == d0
        if m.sum() > 100:
            sig[d0] = max(float(y[m].std()), 1e-6)
    sig_lag = np.concatenate([[sig[0]], sig[:-1]])
    yz = y / sig_lag[day]

    sub = np.zeros(day.size, dtype=bool)
    sub[::5] = True
    rows = []
    for b in range(5, N_BLOCK):                      # 最初の 5 ブロックは学習に使う
        tr = (blk < b) & sub
        te = blk == b
        if tr.sum() < 50_000 or te.sum() < 5_000:
            continue
        lo = np.quantile(X[tr], WINSOR, axis=0).astype(np.float32)
        hi = np.quantile(X[tr], 1 - WINSOR, axis=0).astype(np.float32)
        Ztr = np.clip(X[tr], lo, hi)
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd[sd == 0] = 1.0
        Ztr = ((Ztr - mu) / sd).astype(np.float64)
        Zte = ((np.clip(X[te], lo, hi) - mu) / sd).astype(np.float64)
        out = {"block": b}
        for tag, target in (("raw", y - y[tr].mean()), ("volnorm", yz - yz[tr].mean())):
            mdl = Lasso(alpha=ALPHA, max_iter=5000, tol=1e-5).fit(Ztr, target[tr])
            p = mdl.predict(Zte)
            t_ = target[te]
            out[f"relmse_{tag}"] = float(np.mean((t_ - p) ** 2) / np.mean(t_ ** 2))
            out[f"corr_{tag}"] = float(np.corrcoef(p, t_)[0, 1]) if p.std() > 0 else np.nan
            out[f"pred_sd_{tag}"] = float(p.std())
            out[f"target_sd_{tag}"] = float(t_.std())
            out[f"train_target_sd_{tag}"] = float(target[tr].std())
        out["var_ratio_test_over_train"] = out["target_sd_raw"] ** 2 / out["train_target_sd_raw"] ** 2
        rows.append(out)
        print(f"block {b:2d}: relMSE raw={out['relmse_raw']:.4f} "
              f"volnorm={out['relmse_volnorm']:.4f} corr={out['corr_raw']:+.4f} "
              f"分散比(検証/訓練)={out['var_ratio_test_over_train']:.2f}", flush=True)

    res = pl.DataFrame(rows)
    blkvar = dv.group_by("block").agg(
        [pl.col(c).mean().alias(c) for c in dv.columns if c.startswith("var_")]
        + [pl.col("dt").min().alias("dt_from"), pl.col("dt").max().alias("dt_to")]
    ).sort("block")
    merged = res.join(blkvar, on="block", how="left")
    merged.write_csv(D / ("variance_blocks_clean.csv" if CLEAN else "variance_blocks.csv"))
    dv.write_csv(D / ("variance_daily_clean.csv" if CLEAN else "variance_daily.csv"))

    # ---- 分散と相対 MSE の関係 ----
    summary = {"n_blocks": N_BLOCK, "block_days": len(days) / N_BLOCK, "alpha": ALPHA,
               "relation": {}}
    lv = np.log(merged["var_1s"].to_numpy())
    for tag in ("raw", "volnorm"):
        r = merged[f"relmse_{tag}"].to_numpy()
        c = merged[f"corr_{tag}"].to_numpy()
        summary["relation"][tag] = {
            "corr_logvar_vs_relmse": float(np.corrcoef(lv, r)[0, 1]),
            "corr_logvar_vs_corr": float(np.corrcoef(lv, c)[0, 1]),
            "relmse_median": float(np.median(r)), "relmse_min": float(r.min()),
            "relmse_max": float(r.max()),
            "n_blocks_worse_than_null": int((r > 1).sum()),
        }
    summary["relation"]["var_ratio_vs_relmse_raw"] = float(
        np.corrcoef(np.log(merged["var_ratio_test_over_train"].to_numpy()),
                    merged["relmse_raw"].to_numpy())[0, 1])
    (D / ("variance_blocks_clean.json" if CLEAN else "variance_blocks.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
