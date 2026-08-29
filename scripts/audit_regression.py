"""回帰結果の統計的正当性を監査する。

検証する脅威:
  T1 Fama-MacBeth の SE は日次 β の独立同分布を仮定するが、β は明らかにドリフトしている
     → 日次系列の自己相関を測り、Newey-West とブロック・ブートストラップで SE を取り直す
  T2 日次 β 同士は「同じ母数の反復推定」ではないかもしれない
     → 各日の HAC SE を出し、逆分散加重平均・Cochran の Q・I² で異質性を検定する
  T3 外れ値依存(クロス除外後も残るか)
     → ウィンザライズ β・順位相関(Spearman)・符号一致率で頑健性を見る
  T4 機械的相関の疑い(x と y が同じ mid を共有していないか)
     → x = −(I−1/2)·spread は mid が解析的に消えるので構造上は無い。
        実証的にはプラセボ(x を日内で大きくずらす)で β ≈ 0 を確認する
  T5 有効標本サイズ: x の自己相関で n がどれだけ水増しされているか
  T6 標本外: 前半 50 日で推定した β を後半 49 日に当てて当たるか
  T7 経済的有意性の判定に体制平均の β を使ったのは誤りではないか
     → 体制別 β で |β·m| > ハーフスプレッドの割合を測り直す
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/microprice")
OUT = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
H_NS = 1_000_000_000                      # 主対象は 1 秒地平
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9)]
RNG = np.random.default_rng(20260814)


def band_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}bp" if hi < 1e8 else f"{lo:g}bp+"


def ols_hac(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    """β と Newey-West SE(定数項つき単回帰)。"""
    n = x.size
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    u = y - X @ b
    g = X * u[:, None]
    S = g.T @ g
    for j in range(1, min(lag, n - 1) + 1):
        w = 1.0 - j / (lag + 1.0)
        G = g[j:].T @ g[:-j]
        S += w * (G + G.T)
    Ainv = np.linalg.inv(XtX)
    V = Ainv @ S @ Ainv
    return float(b[1]), float(math.sqrt(max(V[1, 1], 0.0)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.empty(x.size)
    rx[np.argsort(x, kind="stable")] = np.arange(x.size)
    ry = np.empty(y.size)
    ry[np.argsort(y, kind="stable")] = np.arange(y.size)
    return float(np.corrcoef(rx, ry)[0, 1])


def nw_series(v: np.ndarray, lag: int) -> float:
    """系列 v の平均の Newey-West 標準誤差(日次 β の系列に使う)。"""
    n = v.size
    d = v - v.mean()
    s = float(d @ d) / n
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        s += 2.0 * w * float(d[j:] @ d[:-j]) / n
    return math.sqrt(max(s, 0.0) / n)


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    rows = []
    band_half = {}          # (band, half) -> 十分統計量
    band_econ = {}          # band -> [n, Σ|m|, Σhalf_spread, cnt(|βm|>half)] は後段で
    band_all = {}
    acf_acc = np.zeros(51)
    acf_n = 0

    for i, dt in enumerate(days, 1):
        df = (pl.read_parquet(SRC / f"dt={dt}" / "part-000.parquet",
                              columns=["ts", "mid", "microprice", "spread_bp", "is_crossed"])
              .filter(~pl.col("is_crossed")).sort("ts"))
        n0 = df.height
        if n0 < 5000:
            continue
        ts = df["ts"].to_numpy()
        mid = df["mid"].to_numpy()
        logm = np.log(mid)
        x_all = (mid - df["microprice"].to_numpy()) / mid * 1e4
        sp_all = df["spread_bp"].to_numpy()

        tgt = ts + H_NS
        idx = np.nonzero(tgt <= ts[-1])[0]
        if idx.size < 5000:
            continue
        j = np.searchsorted(ts, tgt[idx], side="right") - 1
        y = (logm[j] - logm[idx]) * 1e4
        x = x_all[idx]
        sp = sp_all[idx]

        rate = n0 / max(1e-9, (ts[-1] - ts[0]) / 1e9)      # ticks/sec
        lag = int(min(200, max(4, 2 * rate)))               # 1 秒重複ぶんのラグ
        beta, se = ols_hac(x, y, lag)

        # T3 頑健性
        lo, hi = np.quantile(x, [0.01, 0.99])
        ylo, yhi = np.quantile(y, [0.01, 0.99])
        xw, yw = np.clip(x, lo, hi), np.clip(y, ylo, yhi)
        bw = float(np.cov(xw, yw, ddof=1)[0, 1] / np.var(xw, ddof=1)) if np.var(xw) > 0 else np.nan
        sub = RNG.choice(idx.size, size=min(120_000, idx.size), replace=False)
        sp_corr = spearman(x[sub], y[sub])
        nz = (np.abs(x) > 1e-12) & (np.abs(y) > 1e-12)
        sign_rate = float((np.sign(x[nz]) != np.sign(y[nz])).mean()) if nz.sum() else np.nan

        # T4 プラセボ: x を日内で 5,000 ティックずらす(同じ日・同じ分布・時点だけ無関係)
        xs = np.roll(x, 5000)
        b_pl = float(np.cov(xs, y, ddof=1)[0, 1] / np.var(xs, ddof=1))

        # T5 x の自己相関(有効標本サイズ用)
        d = x - x.mean()
        v0 = float(d @ d)
        if v0 > 0:
            for k in range(51):
                acf_acc[k] += float(d[k:] @ d[:d.size - k]) / v0 if k else 1.0
            acf_n += 1

        # T6/T7 体制別の蓄積(前半/後半)
        half = 0 if i <= 50 else 1
        for blo, bhi in BANDS:
            sel = (sp >= blo) & (sp < bhi)
            if sel.sum() < 100:
                continue
            xb, yb, spb = x[sel], y[sel], sp[sel]
            a = np.array([sel.sum(), xb.sum(), yb.sum(), (xb * xb).sum(),
                          (yb * yb).sum(), (xb * yb).sum(), np.abs(xb).sum(), (spb / 2).sum()])
            k = band_name(blo, bhi)
            band_half[(k, half)] = band_half.get((k, half), 0) + a
            band_all[k] = band_all.get(k, 0) + a
            band_econ.setdefault(k, []).append((xb, spb))

        rows.append({"dt": dt, "n": int(idx.size), "beta": beta, "se_hac": se,
                     "beta_wins": bw, "spearman": sp_corr, "sign_rate": sign_rate,
                     "beta_placebo": b_pl})
        if i % 20 == 0:
            print(f"{i}/{len(days)}", flush=True)

    b = np.array([r["beta"] for r in rows])
    se = np.array([r["se_hac"] for r in rows])
    res: dict = {"n_days": len(rows)}

    # ---- T1 日次 β 系列の自己相関と SE ----
    d = b - b.mean()
    ac = [float(d[k:] @ d[:d.size - k]) / float(d @ d) for k in range(1, 6)]
    se_iid = b.std(ddof=1) / math.sqrt(b.size)
    se_nw = nw_series(b, 5)
    # ブロック・ブートストラップ(ブロック長 5 日、5,000 回)
    L = 5
    nb = int(np.ceil(b.size / L))
    starts = RNG.integers(0, b.size - L + 1, size=(5000, nb))
    boot = np.stack([np.concatenate([b[s:s + L] for s in row])[:b.size] for row in starts])
    res["T1_fama_macbeth"] = {
        "mean": float(b.mean()), "median": float(np.median(b)),
        "acf_daily_beta_lag1_5": ac,
        "se_iid": se_iid, "t_iid": float(b.mean() / se_iid),
        "se_newey_west_days": se_nw, "t_newey_west_days": float(b.mean() / se_nw),
        "boot_mean_ci95": [float(np.quantile(boot.mean(1), 0.025)),
                           float(np.quantile(boot.mean(1), 0.975))],
        "boot_median_ci95": [float(np.quantile(np.median(boot, 1), 0.025)),
                             float(np.quantile(np.median(boot, 1), 0.975))],
        "boot_p_mean_ge0": float((boot.mean(1) >= 0).mean()),
        "n_days_negative": int((b < 0).sum()),
        "sign_test_p_iid": float(2 * 0.5 ** b.size * sum(
            math.comb(b.size, k) for k in range(int((b >= 0).sum()) + 1))),
    }

    # ---- T2 異質性(逆分散加重・Cochran の Q) ----
    w = 1.0 / se ** 2
    bw_mean = float((w * b).sum() / w.sum())
    se_fe = math.sqrt(1.0 / w.sum())
    Q = float((w * (b - bw_mean) ** 2).sum())
    dfree = b.size - 1
    res["T2_heterogeneity"] = {
        "inverse_variance_mean": bw_mean, "se": se_fe, "t": bw_mean / se_fe,
        "cochran_Q": Q, "df": dfree, "I2": max(0.0, (Q - dfree) / Q),
        "median_daily_se": float(np.median(se)),
        "n_days_signif_neg": int(((b + 1.96 * se) < 0).sum()),
        "n_days_signif_pos": int(((b - 1.96 * se) > 0).sum()),
    }

    # ---- T3 頑健推定 ----
    res["T3_robustness"] = {
        "median_beta_winsorized": float(np.nanmedian([r["beta_wins"] for r in rows])),
        "median_spearman": float(np.nanmedian([r["spearman"] for r in rows])),
        "median_sign_disagree_rate": float(np.nanmedian([r["sign_rate"] for r in rows])),
        "n_days_spearman_neg": int(sum(1 for r in rows if r["spearman"] < 0)),
    }

    # ---- T4 プラセボ ----
    pl_b = np.array([r["beta_placebo"] for r in rows])
    res["T4_placebo"] = {
        "median_beta_placebo": float(np.median(pl_b)),
        "mean_beta_placebo": float(pl_b.mean()),
        "n_days_negative": int((pl_b < 0).sum()),
        "se_iid": float(pl_b.std(ddof=1) / math.sqrt(pl_b.size)),
        "t_iid": float(pl_b.mean() / (pl_b.std(ddof=1) / math.sqrt(pl_b.size))),
    }

    # ---- T5 有効標本サイズ ----
    acf = acf_acc / max(acf_n, 1)
    infl = 1.0 + 2.0 * float(acf[1:].sum())
    res["T5_effective_n"] = {
        "acf_x_lag1_5": [float(v) for v in acf[1:6]],
        "variance_inflation_50lags": infl,
        "n_effective_per_day": None,
    }

    # ---- T6 標本外(体制別に前半で推定 → 後半で評価) ----
    oos = {}
    for k in {kk for kk, _ in band_half}:
        a0, a1 = band_half.get((k, 0)), band_half.get((k, 1))
        if a0 is None or a1 is None:
            continue
        def fit(a):
            n, sx, sy, sxx, syy, sxy, _, _ = a
            vx = sxx / n - (sx / n) ** 2
            vy = syy / n - (sy / n) ** 2
            cxy = sxy / n - (sx / n) * (sy / n)
            return n, cxy / vx, cxy / math.sqrt(vx * vy), sx / n, sy / n, vx, vy, cxy
        n0, b0, c0, mx0, my0, vx0, vy0, cxy0 = fit(a0)
        n1, b1, c1, mx1, my1, vx1, vy1, cxy1 = fit(a1)
        # 前半の β を後半に当てたときの説明力(R²_oos = 1 - SSE/SST)
        a_hat = my0 - b0 * mx0
        sse = (a1[4] - 2 * b0 * a1[5] - 2 * a_hat * a1[2] + b0 * b0 * a1[3]
               + 2 * b0 * a_hat * a1[1] + n1 * a_hat ** 2)
        sst = a1[4] - n1 * my1 ** 2
        oos[k] = {"n_first": int(n0), "n_second": int(n1),
                  "beta_first": b0, "beta_second": b1,
                  "corr_first": c0, "corr_second": c1,
                  "r2_oos": float(1 - sse / sst)}
    res["T6_out_of_sample"] = oos

    # ---- T7 体制別 β での経済的有意性 ----
    econ = {}
    for k, chunks in band_econ.items():
        n, sx, sy, sxx, syy, sxy, sabs, shalf = band_all[k]
        vx = sxx / n - (sx / n) ** 2
        beta_k = (sxy / n - (sx / n) * (sy / n)) / vx
        cnt = 0
        tot = 0
        for xb, spb in chunks:
            cnt += int((np.abs(beta_k * xb) > spb / 2).sum())
            tot += xb.size
        econ[k] = {"n": int(n), "beta_band": beta_k,
                   "mean_abs_m_bp": sabs / n, "mean_half_spread_bp": shalf / n,
                   "mean_pred_move_bp": abs(beta_k) * sabs / n,
                   "frac_pred_gt_half_spread": cnt / tot}
    res["T7_economics_by_band"] = econ

    (OUT / "audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                    encoding="utf-8")
    pl.DataFrame(rows).write_csv(OUT / "audit_daily.csv")
    print(json.dumps({k: v for k, v in res.items() if k != "T6_out_of_sample"},
                     indent=2, ensure_ascii=False, default=float))
    print(json.dumps(res["T6_out_of_sample"], indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
