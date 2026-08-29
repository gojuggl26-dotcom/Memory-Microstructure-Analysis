"""r_{t+h} = α + β_A·S^Ask_t + β_B·S^Bid_t + γ·X_t + ε の推定と検定。

粒度: 1 秒グリッド、深さ帯 25bp(book_slope_grain_report.md の実測に基づく)。
地平: h = 1 秒。

★時間契約(CLAUDE.md 厳禁事項)
    S^Ask_t, S^Bid_t, X_t はすべて **T 時点までに確定**した情報
    r_{t+h} = ( log mid(T+h) − log mid(T) ) × 10⁴  … T から先だけ
    resync アーティファクトとクロス行は除外。y は 0.1%/99.9% でウィンザライズ

検定は 3 系統を併記する(regression_report.md §9 の監査に従う):
    z (Newey-West)   … 重複なし(h = グリッド幅)なので L は保険で 5
    t₉₈ (日次クラスター) … 日内の任意の系列相関を許容。G = 99 で自由度 98
    Fama-MacBeth     … 日ごとに推定して平均。ブロック・ブートストラップの CI つき
n = 800 万超では naive な t はどんな微小値も有意にするため、
**日次推定の符号一貫性と経済的大きさを併せて報告する**。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_v2"
SLOPE = D / "slope25"
WINSOR = 0.001
DROP = ["book_slope_bid", "book_slope_ask", "book_slope_diff"]   # 全レベル版は外す
RNG = np.random.default_rng(20260815)


def betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / (tiny if abs(1 + aa * d) < tiny else 1 + aa * d)
        c = tiny if abs(1 + aa / c) < tiny else 1 + aa / c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / (tiny if abs(1 + aa * d) < tiny else 1 + aa * d)
        c = tiny if abs(1 + aa / c) < tiny else 1 + aa / c
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def p_t(stat: float, df: int) -> float:
    x = df / (df + stat * stat)
    a, b = df / 2.0, 0.5
    if x <= 0:
        return 0.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lb)
    return (bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2)
            else 1.0 - bt * betacf(b, a, 1 - x) / b)


def p_z(stat: float) -> float:
    return math.erfc(abs(stat) / math.sqrt(2.0))


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    frames = []
    for k, dt in enumerate(days):
        p = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet").drop(DROP)
        s = pl.read_parquet(SLOPE / f"{dt}.parquet").select(
            ["ts", "s_bid", "s_ask"]).rename({"s_bid": "S_bid", "s_ask": "S_ask"})
        frames.append(p.join(s, on="ts", how="inner").with_columns(pl.lit(k).alias("day")))
    df = pl.concat(frames)
    other = [c for c in df.columns if c not in ("ts", "y", "day", "S_bid", "S_ask")]
    print(f"rows={df.height:,}  X の変数={len(other)}")

    day = df["day"].to_numpy()
    y = df["y"].to_numpy()
    lo, hi = np.quantile(y, WINSOR), np.quantile(y, 1 - WINSOR)
    y = np.clip(y, lo, hi)                       # ★y のウィンザライズ
    S = df.select(["S_ask", "S_bid"]).to_numpy()
    Xo = df.select(other).to_numpy()

    def prep(A):
        l, h = np.quantile(A, WINSOR, axis=0), np.quantile(A, 1 - WINSOR, axis=0)
        A = np.clip(A, l, h)
        return A
    S = prep(S)
    Xo = prep(Xo)
    res: dict = {"n": int(len(y)), "days": len(days), "other_vars": other,
                 "band_bp": 25, "horizon": "1s"}

    def fit(Z: np.ndarray, names: list[str], tag: str) -> dict:
        A = np.column_stack([np.ones(len(Z)), Z])
        XtX = A.T @ A
        b = np.linalg.solve(XtX, A.T @ y)
        u = y - A @ b
        Ainv = np.linalg.inv(XtX)
        g = A * u[:, None]
        # Newey-West(L = 5。地平とグリッドが等しいので重複は無いが保険)
        Smat = g.T @ g
        for j in range(1, 6):
            G = g[j:].T @ g[:-j]
            Smat += (1 - j / 6) * (G + G.T)
        V_nw = Ainv @ Smat @ Ainv
        # 日次クラスター
        Sc = np.zeros_like(Smat)
        for d0 in np.unique(day):
            gg = g[day == d0].sum(0)
            Sc += np.outer(gg, gg)
        G_ = len(np.unique(day))
        ssc = G_ / (G_ - 1) * (len(y) - 1) / (len(y) - A.shape[1])
        V_cl = Ainv @ Sc @ Ainv * ssc
        # Fama-MacBeth(日ごとに推定)
        fm = []
        for d0 in np.unique(day):
            m = day == d0
            if m.sum() < 500:
                continue
            Ad = A[m]
            try:
                fm.append(np.linalg.solve(Ad.T @ Ad, Ad.T @ y[m]))
            except np.linalg.LinAlgError:
                continue
        fm = np.array(fm)
        out = {"tag": tag, "n_params": len(names), "coef": {}, "r2": float(
            1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2))}
        for i, nm in enumerate(names, start=1):
            se_nw = math.sqrt(max(V_nw[i, i], 0))
            se_cl = math.sqrt(max(V_cl[i, i], 0))
            f = fm[:, i]
            se_fm = f.std(ddof=1) / math.sqrt(f.size)
            # ブロック・ブートストラップ(5 日ブロック、5,000 回)
            L = 5
            nb = int(np.ceil(f.size / L))
            st = RNG.integers(0, f.size - L + 1, size=(5000, nb))
            boot = np.stack([np.concatenate([f[s:s + L] for s in row])[:f.size]
                             for row in st]).mean(1)
            out["coef"][nm] = {
                "beta": float(b[i]),
                "z_newey_west": float(b[i] / se_nw), "p_z": p_z(b[i] / se_nw),
                "t98_cluster": float(b[i] / se_cl), "p_t98": p_t(b[i] / se_cl, G_ - 1),
                "fm_mean": float(f.mean()), "fm_t98": float(f.mean() / se_fm),
                "fm_p": p_t(f.mean() / se_fm, f.size - 1),
                "fm_median": float(np.median(f)),
                "fm_days_negative": int((f < 0).sum()), "fm_days": int(f.size),
                "boot_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                "se_nw": se_nw, "se_cluster": se_cl, "se_fm": se_fm,
            }
        return out

    res["model_simple"] = fit(S, ["S_ask", "S_bid"], "S のみ")
    res["model_full"] = fit(np.column_stack([S, Xo]), ["S_ask", "S_bid"] + other, "S + X")
    # 標準化版(係数の比較用)
    Zs = np.column_stack([S, Xo])
    sd = Zs.std(0); sd[sd == 0] = 1
    res["model_full_standardized"] = fit((Zs - Zs.mean(0)) / sd,
                                         ["S_ask", "S_bid"] + other, "S + X(標準化)")
    (D / "slope_regression.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    for key in ("model_simple", "model_full"):
        r = res[key]
        print(f"\n=== {r['tag']}  (R² = {r['r2']:.5f}) ===")
        print(f"{'':>8} {'β':>12} {'z(NW)':>9} {'t₉₈(クラスタ)':>13} {'FM t₉₈':>9} "
              f"{'FM 日次中央値':>13} {'符号逆':>8}")
        for nm in ("S_ask", "S_bid"):
            c = r["coef"][nm]
            print(f"{nm:>8} {c['beta']:>+12.6f} {c['z_newey_west']:>9.1f} "
                  f"{c['t98_cluster']:>13.2f} {c['fm_t98']:>9.2f} "
                  f"{c['fm_median']:>+13.6f} {c['fm_days_negative']:>4}/{c['fm_days']}")
            print(f"{'':>8} p(NW)={c['p_z']:.2e}  p(t₉₈)={c['p_t98']:.2e}  "
                  f"p(FM)={c['fm_p']:.2e}  BS95%CI=[{c['boot_ci95'][0]:+.5f},"
                  f"{c['boot_ci95'][1]:+.5f}]")
    r = res["model_full_standardized"]
    print("\n=== 標準化係数(1SD あたり bp) ===")
    for nm, c in sorted(r["coef"].items(), key=lambda kv: -abs(kv[1]["beta"]))[:8]:
        print(f"  {nm:>22} {c['beta']:+.5f}")


if __name__ == "__main__":
    main()
