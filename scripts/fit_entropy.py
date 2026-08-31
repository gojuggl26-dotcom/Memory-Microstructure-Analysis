"""エントロピー 13 種が将来の値動きを予測するかを測る。

【判定の設計】
エントロピーは「散らばり」の指標であって向きを持たない。したがって素直な
目的変数は**将来のボラティリティ**(|r|)であり、**向き**(r)には効かないはずである。
両方を測って書き分ける。判定は次の 4 段で行う。

  (1) 効果量   … 重ならない標本での Spearman 順位相関 ρ
  (2) 頑健性   … 日ごとに ρ を出し、**符号検定**(98 日)で判定する。
                  1 秒格子の隣接点は強く相関するので、プールした z は信用しない。
  (3) 帰無対照 … x を日の中で**巡回シフト**(+12 時間)して同じ手順を通す。
                  シフトしても残る相関は、因果ではなく日内の形が作っている。
  (4) 統制     … 活動量(板の本数 n_ord・イベント数 n_ev)を順位で回した後の
                  **偏順位相関**。エントロピーは母集団が大きいほど機械的に
                  上がるので、これを外さないと「注文が多い日は荒れる」の
                  言い換えを掴んでしまう。

【★予測と同時性の書き分け】
同じ手順を**過去のリターン**([T−h, T))に対しても回す。前向きより後ろ向きが
大きければ、その特徴量は値動きの**後始末**であって前触れではない。
定義は reports/predicting_definition.md に従う。

【重ならない標本】
格子は 1 秒なので、地平 h の窓 [T, T+h) は h 個の格子で重なる。重なった標本で
相関を測ると自由度が h 倍に水増しされる。よって**stride = h** で間引く。

【多重比較】
13 特徴量 × 6 地平 × 2 目的変数 = **156 セル**を全件報告する。
Bonferroni 閾値は 0.05 / 156 = 3.2e−4(符号検定の |z| で 3.60)。

【出力】
    data/entropy_fit_<coin>.csv     156 セルの全結果(効果量・符号検定・対照)
    data/entropy_corr_<coin>.csv    13 特徴量 + 統制 2 の相関行列
    data/entropy_daily_<coin>.csv   日ごとの平均(時系列の安定性を見る)
    data/entropy_summary_<coin>.json 見出しの数値

実行例:
    uv run python scripts/fit_entropy.py --coin xyz:MU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entropy import FEATURES, HORIZONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("n_ord", "n_ev")
SHIFT_S = 12 * 3600          # 帰無対照の巡回シフト量(秒)
MIN_DAY_N = 200              # 日ごとの相関を採用する最小標本
SPREAD_BP = 1.245            # 往復スプレッドの中央値(mu_obi_ofi_report.md 実測)
POOL_CAP = 1_000_000         # プールした効果量を測るときの標本上限(下記 ★)

# ★ 効果量(rho_pooled / rho_partial)は 100 万点で十分に決まる。
#   h=1 では全期間で 840 万点あり、順位付けだけで 1 セット 10 秒かかるので
#   等間隔に間引いて上限を掛ける。判定に使う日ごとの符号検定は間引かない。


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """欠損を除いた Spearman 順位相関。標本が足りなければ nan。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    xr = stats.rankdata(x[m])
    yr = stats.rankdata(y[m])
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """統制 Z を順位で回帰して外したあとの、x と y の順位相関。"""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if m.sum() < 50:
        return np.nan
    xr = stats.rankdata(x[m])
    yr = stats.rankdata(y[m])
    Zr = np.column_stack([stats.rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(len(xr)), Zr])
    # 最小二乗で統制成分を落とす
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    ex, ey = xr - A @ bx, yr - A @ by
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def sign_test(rhos: np.ndarray) -> tuple[int, int, float, float]:
    """日ごとの ρ の符号検定。(正の日, 有効な日, z, 両側 p) を返す。"""
    r = rhos[np.isfinite(rhos)]
    n = r.size
    if n < 10:
        return 0, n, np.nan, np.nan
    k = int((r > 0).sum())
    z = (k - n / 2) / np.sqrt(n / 4)
    p = 2 * stats.norm.sf(abs(z))
    return k, n, float(z), float(p)


def abschg_analysis(d: pl.DataFrame, days, by_day, tag) -> pl.DataFrame:
    """★十分位を見て初めて判った U 字への追試。

    signed な Δh_depth と将来 |r| の関係は**単調でない**。十分位に切ると
    |r| は Δh_depth の両端で高く中央で最小になる(U 字)。順位相関はこの
    U 字の左右差だけを拾っており、関係の強さを取り逃がしている。
    そこで **|Δh_depth|**(変化の大きさ)を説明変数にして測り直す。

    これは十分位表を見てから追加した分析なので、探索の一部として
    別枠で報告する(セル数 6。156 セルの Bonferroni とは別に数える)。
    """
    rows = []
    for h in HORIZONS:
        dr, db, dp = [], [], []
        PX, PY, PZ = [], [], []
        for dt in days:
            s = by_day[(dt,)].sort("ts")
            x = np.abs(s["h_chg"].to_numpy().astype(np.float64))
            r = s[f"r{h}"].to_numpy().astype(np.float64)
            y = np.abs(r)
            rb = (np.concatenate([np.full(h, np.nan), r[:-h]])
                  if r.size > h else np.full_like(r, np.nan))
            xs = np.roll(x, SHIFT_S)
            Z = np.column_stack([s[c].to_numpy().astype(np.float64)
                                 for c in CONTROLS])
            sl = slice(None, None, h)
            dr.append(spearman(x[sl], y[sl]))
            db.append(spearman(x[sl], np.abs(rb)[sl]))
            dp.append(spearman(xs[sl], y[sl]))
            PX.append(x[sl]); PY.append(y[sl]); PZ.append(Z[sl])
        X, Y, ZZ = np.concatenate(PX), np.concatenate(PY), np.concatenate(PZ)
        if X.size > POOL_CAP:
            st = int(np.ceil(X.size / POOL_CAP))
            X, Y, ZZ = X[::st], Y[::st], ZZ[::st]
        k, n, z, pv = sign_test(np.array(dr))
        _, _, zp, _ = sign_test(np.array(dp))
        rows.append({"horizon_s": h, "rho_day_med": float(np.nanmedian(dr)),
                     "backward_rho_med": float(np.nanmedian(db)),
                     "rho_partial": partial_spearman(X, Y, ZZ),
                     "pos_days": k, "n_days": n, "z": z, "p": pv,
                     "placebo_z": zp})
    out = pl.DataFrame(rows)
    out.write_csv(ROOT / "data" / f"entropy_abschg_{tag}.csv")
    return out


def decile_table(d: pl.DataFrame, tag) -> pl.DataFrame:
    """Δh_depth の十分位ごとの、次の 1 秒の |r|。U 字を数字で示す。"""
    x = d["h_chg"].to_numpy().astype(np.float64)
    y = np.abs(d["r1"].to_numpy().astype(np.float64))
    xs = np.roll(x, SHIFT_S)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, xs = x[m], y[m], xs[m]
    rows = []
    for name, xx in (("actual", x), ("placebo", xs)):
        q = np.percentile(xx, np.arange(10, 100, 10))
        b = np.searchsorted(q, xx, side="right")
        for i in range(10):
            s = y[b == i] * 1e4
            rows.append({"kind": name, "decile": i + 1,
                         "x_med": float(np.median(xx[b == i])),
                         "absr_med_bp": float(np.median(s)) if s.size else np.nan,
                         "absr_mean_bp": float(s.mean()) if s.size else np.nan,
                         "n": int(s.size)})
    out = pl.DataFrame(rows)
    out.write_csv(ROOT / "data" / f"entropy_decile_{tag}.csv")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = pl.read_parquet(ROOT / "data" / f"entropy_{tag}.parquet")
    d = d.filter(pl.col("ok"))
    days = d["dt"].unique().sort().to_list()
    print(f"[load] {d.height:,} 行 / {len(days)} 日", file=sys.stderr)

    # ---------- 特徴量どうしの相関(冗長性の実測) ----------
    cols = list(FEATURES) + list(CONTROLS)
    M = np.column_stack([d[c].to_numpy().astype(np.float64) for c in cols])
    ok = np.isfinite(M).all(axis=1)
    R = np.corrcoef(np.column_stack(
        [stats.rankdata(M[ok, j]) for j in range(M.shape[1])]).T)
    (pl.DataFrame({"feature": cols,
                   **{c: R[:, j] for j, c in enumerate(cols)}})
     .write_csv(ROOT / "data" / f"entropy_corr_{tag}.csv"))
    print(f"[corr] 相関行列 {len(cols)}×{len(cols)}(有効 {ok.sum():,} 行)",
          file=sys.stderr)

    # ---------- 日ごとの平均 ----------
    (d.group_by("dt").agg(
        [pl.col(f).mean().alias(f) for f in FEATURES]
        + [pl.col("n_ord").mean(), pl.col("n_ev").mean(),
           pl.len().alias("n_grid")])
     .sort("dt").write_csv(ROOT / "data" / f"entropy_daily_{tag}.csv"))

    # ---------- 156 セルの本体 ----------
    by_day = {dt: sub for dt, sub in d.group_by("dt")}
    part = ROOT / "data" / f"entropy_fit_{tag}.partial.csv"
    rows = []
    done: set[tuple] = set()
    if part.exists():        # 途中で止まっても続きから走れるようにする
        prev = pl.read_csv(part)
        rows = prev.to_dicts()
        done = {(r["feature"], r["horizon_s"], r["target"]) for r in rows}
        print(f"[resume] 済み {len(done)} セル", file=sys.stderr)
    for f in FEATURES:
        for h in HORIZONS:
            for tgt in ("dir", "vol"):
                if (f, h, tgt) in done:
                    continue
                pooled_x, pooled_y, pooled_Z, dayr, dayr_pl, dayr_bw = \
                    [], [], [], [], [], []
                for dt in days:
                    s = by_day[(dt,)].sort("ts")
                    x = s[f].to_numpy().astype(np.float64)
                    r = s[f"r{h}"].to_numpy().astype(np.float64)
                    y = np.abs(r) if tgt == "vol" else r
                    # 後ろ向き(過去 h 秒のリターン)= 同時性の検査
                    rb = np.concatenate([np.full(h, np.nan), r[:-h]]) \
                        if r.size > h else np.full_like(r, np.nan)
                    yb = np.abs(rb) if tgt == "vol" else rb
                    xs = np.roll(x, SHIFT_S)          # 帰無対照(巡回シフト)
                    Z = np.column_stack([s[c].to_numpy().astype(np.float64)
                                         for c in CONTROLS])
                    sl = slice(None, None, h)         # ★重ならない標本
                    xa, ya, yba, xsa, Za = (x[sl], y[sl], yb[sl], xs[sl], Z[sl])
                    if np.isfinite(xa).sum() >= MIN_DAY_N:
                        dayr.append(spearman(xa, ya))
                        dayr_pl.append(spearman(xsa, ya))
                        dayr_bw.append(spearman(xa, yba))
                    pooled_x.append(xa); pooled_y.append(ya); pooled_Z.append(Za)
                X = np.concatenate(pooled_x)
                Y = np.concatenate(pooled_y)
                ZZ = np.concatenate(pooled_Z)
                if X.size > POOL_CAP:            # ★等間隔に間引く(無作為でない)
                    st = int(np.ceil(X.size / POOL_CAP))
                    X, Y, ZZ = X[::st], Y[::st], ZZ[::st]
                k, n, z, p = sign_test(np.array(dayr))
                kp, npl, zp, pp = sign_test(np.array(dayr_pl))
                kb, nb, zb, pb = sign_test(np.array(dayr_bw))
                rows.append({
                    "feature": f, "horizon_s": h, "target": tgt,
                    "rho_pooled": spearman(X, Y),
                    "rho_partial": partial_spearman(X, Y, ZZ),
                    "rho_day_med": float(np.nanmedian(dayr)) if dayr else np.nan,
                    "pos_days": k, "n_days": n, "z": z, "p": p,
                    "placebo_pos": kp, "placebo_z": zp,
                    "backward_rho_med": float(np.nanmedian(dayr_bw))
                    if dayr_bw else np.nan,
                    "backward_z": zb,
                    "n_obs": int(np.isfinite(X).sum()),
                })
            pl.DataFrame(rows).write_csv(part)     # 1 地平ごとに保存
            print(f"  {f:9s} h={h:3d}s 済", file=sys.stderr)

    fit = pl.DataFrame(rows)
    fit.write_csv(ROOT / "data" / f"entropy_fit_{tag}.csv")

    # ---- 十分位で U 字に気づいたので、変化の大きさでも測り直す ----
    dec = decile_table(d, tag)
    ab = abschg_analysis(d, days, by_day, tag)
    print(f"[abschg] |Δh_depth| の追試 {ab.height} 地平 -> "
          f"data/entropy_abschg_{tag}.csv", file=sys.stderr)

    # ---------- 見出しの数値 ----------
    nb_cells = fit.height
    thr = 0.05 / nb_cells
    zthr = float(stats.norm.isf(thr / 2))
    passed = fit.filter(pl.col("p") < thr)
    vol = fit.filter(pl.col("target") == "vol")
    dr = fit.filter(pl.col("target") == "dir")
    best_vol = vol.sort(pl.col("z").abs(), descending=True).head(1)
    best_dir = dr.sort(pl.col("z").abs(), descending=True).head(1)
    # 向きの効果を bp へ翻訳(1σ の x に対する mid 変化の期待値)
    summ = {
        "n_rows": d.height, "n_days": len(days), "n_cells": nb_cells,
        "bonferroni_p": thr, "bonferroni_z": zthr,
        "n_pass": passed.height,
        "n_pass_vol": passed.filter(pl.col("target") == "vol").height,
        "n_pass_dir": passed.filter(pl.col("target") == "dir").height,
        "best_vol": best_vol.to_dicts()[0] if best_vol.height else None,
        "best_dir": best_dir.to_dicts()[0] if best_dir.height else None,
        "max_abs_rho_dir": float(dr["rho_day_med"].abs().max()),
        "max_abs_rho_vol": float(vol["rho_day_med"].abs().max()),
        "spread_bp": SPREAD_BP,
        "corr_rate_cond": float(R[cols.index("h_rate"), cols.index("h_cond")]),
        "corr_rate_trans": float(R[cols.index("h_rate"), cols.index("h_trans")]),
        "corr_cond_trans": float(R[cols.index("h_cond"), cols.index("h_trans")]),
        "corr_depth_plevel": float(R[cols.index("h_depth"),
                                     cols.index("h_plevel")]),
        "corr_depth_nord": float(R[cols.index("h_depth"), cols.index("n_ord")]),
        "abschg_1s": ab.filter(pl.col("horizon_s") == 1).to_dicts()[0],
        "decile_ratio": float(
            dec.filter((pl.col("kind") == "actual") & (pl.col("decile") == 1))
               ["absr_med_bp"][0]
            / dec.filter((pl.col("kind") == "actual") & (pl.col("decile") == 10))
                 ["absr_med_bp"][0]),
        "decile_ratio_placebo": float(
            dec.filter((pl.col("kind") == "placebo") & (pl.col("decile") == 1))
               ["absr_med_bp"][0]
            / dec.filter((pl.col("kind") == "placebo") & (pl.col("decile") == 10))
                 ["absr_med_bp"][0]),
    }
    (ROOT / "data" / f"entropy_summary_{tag}.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[fit] {nb_cells} セル / Bonferroni 通過 {passed.height} "
          f"(ボラ {summ['n_pass_vol']} / 向き {summ['n_pass_dir']})",
          file=sys.stderr)
    print(f"-> data/entropy_fit_{tag}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
