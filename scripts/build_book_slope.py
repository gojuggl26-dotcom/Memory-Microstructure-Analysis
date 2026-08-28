"""イベントごとに Book Slope を出し、将来の log リターンに回帰する(OLS / GLS)。

【定義】

板の傾き = 「価格が mid からどれだけ離れると、どれだけの数量が積まれているか」。
最良気配だけを使う場合、mid からの距離は両側とも半スプレッドに等しいので、

    h_t     = (P^ask − P^bid) / (2·mid) × 10^4          … 半スプレッド[bp]
    S^bid_t = Q^bid_t / h_t,   S^ask_t = Q^ask_t / h_t  … 片側の傾き[枚/bp]
    BookSlope_t = S^bid_t − S^ask_t = (Q^bid_t − Q^ask_t) / h_t

**★恒等式(スクリプト内で数値検算する)**

    BookSlope_t / (S^bid_t + S^ask_t) = OBI_t

つまり **傾きを正規化すると OBI そのものになる**。BookSlope が OBI に足しているのは
「厚みをスプレッドで割った量」= 板の密度である。したがって
[OBI と OFI のレポート]の焼き直しにならないよう、正規化前の BookSlope を主役にし、
OBI を対照として同じ回帰にかける。

【★最良気配だけを使っている理由】
本来の板の傾きは複数階層(Næs–Skjeltorp 2006 等)で測る。しかし多階層の板
(l2/book_bp, l2/book_px)は **S3 Deep Archive にあり直接読めない**うえ、
1 秒スナップショットなので **イベント単位のホライズンに使えない**。
per-event で手元にあるのは bbo(最良気配)だけである。この制約は結果の解釈に効く。

【★尺度の正規化】
BookSlope は枚/bp の単位を持ち、h が小さい(スプレッドが狭い)ときに発散する。
日によって尺度が大きく変わるので、**直前 W イベントの標準偏差**で割って無次元にする。
`shift(1)` を入れて自分自身を尺度に含めない(全標本から標準化パラメータを作らない)。

【x が確定する時刻 / y の期間】
    x = BookSlope_z_t(または OBI_t) … 時刻 t の板だけで決まる。t で確定する
    y = ln(mid_{t+k} / mid_t) × 10^4  … 期間は (t, t+k]。t より後にしか判らない
先読みは無い。`shift(-k)` は y にしか使っていない。

【★重なる窓と誤差構造】
k イベント先のリターンを 1 イベントずらしながら並べると、隣り合う観測は
期間を k−1 だけ共有する。誤差は **MA(k−1)** になる。したがって

  - OLS の古典的な標準誤差は使えない → **HAC(Newey–West, ラグ k)** を主とする
  - **GLS の AR(1) はこの構造に対して誤設定**である。指示により算出はするが、
    標準誤差の根拠にはしない
  - 決定的な対照として **重ならない部分標本**(日内で k 個ごとに 1 つ)も回帰する。
    ここでは重なりが無いので古典的な標準誤差が使える

    uv run python scripts/build_book_slope.py --coin xyz:MU
出力: data/book_slope_fits_<coin>.parquet  … 回帰の結果
      data/book_slope_bins_<coin>.parquet  … 帯ごとの平均 log リターン(図示用)
      data/book_slope_meta_<coin>.parquet  … スプレッド等の要約
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
# ユーザ指定は「1,3,5,10,2,30,50,100,300,500」。10 と 30 の間の "2" は 20 の誤記と
# 読めるが、2 と 20 の両方を入れて どちらの読みでも答えられるようにしてある。
HORIZONS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 300, 500]
ROLL_W = 5_000
HAC_CAP = 500
SEED = 20260828

# 図示用の帯。対称・等間隔・端は開区間という機械的な規則で先に決める
BIN_EDGES = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0]


def load(coin: str) -> pl.DataFrame:
    tag = coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い。先に fetch_bbo.py を実行すること")
    d = pl.read_parquet(p).sort("ts")
    n0 = d.height
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    print(f"[load] {n0:,} 行 -> {d.height:,} 行(クロス・数量 0 を除去 {n0 - d.height:,})",
          file=sys.stderr)

    d = d.with_columns(
        mid=(pl.col("best_bid") + pl.col("best_ask")) / 2,
    ).with_columns(
        half_bp=(pl.col("best_ask") - pl.col("best_bid")) / (2 * pl.col("mid")) * 1e4,
        obi=(pl.col("bid_sz") - pl.col("ask_sz")) / (pl.col("bid_sz") + pl.col("ask_sz")),
    ).with_columns(
        slope=(pl.col("bid_sz") - pl.col("ask_sz")) / pl.col("half_bp"),
        slope_tot=(pl.col("bid_sz") + pl.col("ask_sz")) / pl.col("half_bp"),
    )

    # ★恒等式の検算: BookSlope / (S^bid + S^ask) = OBI
    lhs = (d["slope"] / d["slope_tot"]).to_numpy()
    rhs = d["obi"].to_numpy()
    err = float(np.nanmax(np.abs(lhs - rhs)))
    print(f"[検算] BookSlope/(S_bid+S_ask) = OBI の最大誤差 {err:.3e}", file=sys.stderr)
    if err > 1e-12:
        sys.exit("恒等式が成り立たない。定義かデータを疑うこと")

    # 尺度は直前 W イベントだけから作る。shift(1) で自分を除く
    d = d.with_columns(
        sc=pl.col("slope").rolling_std(ROLL_W, min_samples=ROLL_W // 2).shift(1)
    ).with_columns(
        slope_z=pl.when(pl.col("sc") > 0).then(pl.col("slope") / pl.col("sc")).otherwise(None)
    )
    return d


def ols_from_sums(n, Sx, Sy, Sxx, Sxy, Syy):
    den = n * Sxx - Sx * Sx
    if den <= 0 or n < 3:
        return np.nan, np.nan, np.nan
    b = (n * Sxy - Sx * Sy) / den
    a = (Sy - b * Sx) / n
    dy = n * Syy - Sy * Sy
    r2 = (n * Sxy - Sx * Sy) ** 2 / (den * dy) if dy > 0 else np.nan
    return a, b, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d = load(a.coin)

    days = sorted(d["dt"].unique().to_list())
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {x: ("立会日" if x in sess else "閉場日") for x in days}
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    mid = d["mid"].to_numpy()
    dt_arr = d["dt"].to_numpy()
    X = {"BookSlope_z": d["slope_z"].to_numpy(), "OBI": d["obi"].to_numpy()}
    hb = d["half_bp"].to_numpy()
    st = d["slope_tot"].to_numpy()
    print(f"[尺度] 半スプレッド 中央 {np.median(hb):.3f} bp / "
          f"両側の傾き 中央 {np.median(st):,.1f} 枚/bp", file=sys.stderr)
    print(f"[BookSlope] 生の中央 {np.nanmedian(d['slope'].to_numpy()):+,.2f} 枚/bp / "
          f"z の |値|>2 が {np.nanmean(np.abs(X['BookSlope_z']) > 2) * 100:.2f}%", file=sys.stderr)

    meta = pl.DataFrame({
        "half_spread_bp_median": [float(np.median(hb))],
        "spread_bp_median": [float(np.median(hb) * 2)],
        "slope_tot_median": [float(np.median(st))],
        "n_events": [int(len(mid))],
    })
    meta.write_parquet(ROOT / "data" / f"book_slope_meta_{tag}.parquet")

    # 日ごとのスライス
    idx = {}
    for day in days:
        m = np.nonzero(dt_arr == day)[0]
        idx[day] = (m[0], m[-1] + 1)

    rng = np.random.default_rng(SEED)
    fits, bins_rows = [], []

    for xname, xall in X.items():
        for dtl in ["立会日", "閉場日"]:
            dl = [x for x in days if dtype[x] == dtl]
            for k in HORIZONS:
                # ---- pass 1: 十分統計量 -------------------------------------
                n = Sx = Sy = Sxx = Sxy = Syy = 0.0
                Sxp = Sxyp = 0.0                       # プラセボ用
                for day in dl:
                    s, e = idx[day]
                    if e - s <= k:
                        continue
                    xi = xall[s:e - k]
                    yi = np.log(mid[s + k:e] / mid[s:e - k]) * 1e4
                    ok = np.isfinite(xi) & np.isfinite(yi)
                    xi, yi = xi[ok], yi[ok]
                    if len(xi) < 100:
                        continue
                    n += len(xi); Sx += xi.sum(); Sy += yi.sum()
                    Sxx += float(xi @ xi); Sxy += float(xi @ yi); Syy += float(yi @ yi)
                    # 帰無対照: x を日内で巡回シフト(x の分布と自己相関は保つ)
                    xp = np.roll(xi, int(rng.integers(len(xi) // 5, 4 * len(xi) // 5)))
                    Sxp += xp.sum(); Sxyp += float(xp @ yi)
                if n < 1000:
                    continue
                alpha, beta, r2 = ols_from_sums(n, Sx, Sy, Sxx, Sxy, Syy)
                _, beta_pl, _ = ols_from_sums(n, Sxp, Sy, Sxx, Sxyp, Syy)
                xbar = Sx / n
                Sxx_c = Sxx - n * xbar * xbar          # Σ(x−x̄)²

                # ---- pass 2: 残差 → HAC / DW / ρ / 重ならない部分標本 --------
                L = min(k, HAC_CAP)
                G = np.zeros(L + 1)
                dw_num = ee_lag = ee_sum = 0.0
                s2 = 0.0
                nn = Snx = Sny = Snxx = Snxy = Snyy = 0.0   # 重ならない部分標本
                for day in dl:
                    s, e = idx[day]
                    if e - s <= k:
                        continue
                    xi = xall[s:e - k]
                    yi = np.log(mid[s + k:e] / mid[s:e - k]) * 1e4
                    ok = np.isfinite(xi) & np.isfinite(yi)
                    xi, yi = xi[ok], yi[ok]
                    if len(xi) < 100:
                        continue
                    ei = yi - alpha - beta * xi
                    s2 += float(ei @ ei)
                    u = (xi - xbar) * ei
                    G[0] += float(u @ u)
                    for j in range(1, L + 1):
                        if len(u) > j:
                            G[j] += float(u[:-j] @ u[j:])
                    de = np.diff(ei)
                    dw_num += float(de @ de)
                    ee_lag += float(ei[:-1] @ ei[1:])
                    ee_sum += float(ei @ ei)
                    # 重ならない部分標本: k 個ごとに 1 つ取る
                    xs, ys = xi[::k], yi[::k]
                    nn += len(xs); Snx += xs.sum(); Sny += ys.sum()
                    Snxx += float(xs @ xs); Snxy += float(xs @ ys); Snyy += float(ys @ ys)

                w = 1.0 - np.arange(1, L + 1) / (L + 1.0)
                S_hac = G[0] + 2.0 * float(w @ G[1:])
                se_hac = np.sqrt(max(S_hac, 0.0)) / Sxx_c
                se_ols = np.sqrt(s2 / (n - 2) / Sxx_c)
                dw = dw_num / s2 if s2 > 0 else np.nan
                rho = ee_lag / ee_sum if ee_sum > 0 else 0.0

                _, beta_no, r2_no = ols_from_sums(nn, Snx, Sny, Snxx, Snxy, Snyy)
                if nn > 3:
                    an, bn, _ = ols_from_sums(nn, Snx, Sny, Snxx, Snxy, Snyy)
                    ssr = Snyy - 2 * (an * Sny + bn * Snxy) + \
                        (an * an * nn + 2 * an * bn * Snx + bn * bn * Snxx)
                    se_no = np.sqrt(max(ssr, 0) / (nn - 2) /
                                    max(Snxx - nn * (Snx / nn) ** 2, 1e-12))
                else:
                    se_no = np.nan

                # ---- pass 3: GLS(AR(1) の準差分。日内でのみ)-----------------
                gn = gSx = gSy = gSxx = gSxy = gSyy = 0.0
                for day in dl:
                    s, e = idx[day]
                    if e - s <= k:
                        continue
                    xi = xall[s:e - k]
                    yi = np.log(mid[s + k:e] / mid[s:e - k]) * 1e4
                    ok = np.isfinite(xi) & np.isfinite(yi)
                    xi, yi = xi[ok], yi[ok]
                    if len(xi) < 100:
                        continue
                    xg = xi[1:] - rho * xi[:-1]
                    yg = yi[1:] - rho * yi[:-1]
                    gn += len(xg); gSx += xg.sum(); gSy += yg.sum()
                    gSxx += float(xg @ xg); gSxy += float(xg @ yg); gSyy += float(yg @ yg)
                a_g, b_gls, r2_g = ols_from_sums(gn, gSx, gSy, gSxx, gSxy, gSyy)
                ssr_g = gSyy - 2 * (a_g * gSy + b_gls * gSxy) + \
                    (a_g * a_g * gn + 2 * a_g * b_gls * gSx + b_gls * b_gls * gSxx)
                se_gls = np.sqrt(max(ssr_g, 0) / (gn - 2) /
                                 max(gSxx - gn * (gSx / gn) ** 2, 1e-12))

                fits.append({
                    "x": xname, "day_type": dtl, "k": k, "n": int(n),
                    "beta_ols": beta, "se_ols": se_ols, "t_ols": beta / se_ols,
                    "se_hac": se_hac, "t_hac": beta / se_hac,
                    "beta_gls": b_gls, "se_gls": se_gls, "t_gls": b_gls / se_gls,
                    "beta_nonoverlap": beta_no, "se_nonoverlap": se_no,
                    "t_nonoverlap": beta_no / se_no if se_no == se_no else np.nan,
                    "n_nonoverlap": int(nn),
                    "r2": r2, "r2_nonoverlap": r2_no, "rho": rho, "dw": dw,
                    "beta_placebo": beta_pl,
                })
                print(f"  {xname:<12} {dtl} k={k:<4} n={int(n):>11,}  "
                      f"beta {beta:+.4f}  HAC t {beta/se_hac:+8.1f}  "
                      f"GLS {b_gls:+.4f}  非重複 {beta_no:+.4f}  "
                      f"R2 {r2:.5f}  rho {rho:+.3f}", file=sys.stderr)

            # ---- 図示用: 帯ごとの平均 log リターン ---------------------------
            for k in HORIZONS:
                cnt = np.zeros(len(BIN_EDGES) + 1)
                ssum = np.zeros(len(BIN_EDGES) + 1)
                for day in dl:
                    s, e = idx[day]
                    if e - s <= k:
                        continue
                    xi = xall[s:e - k]
                    yi = np.log(mid[s + k:e] / mid[s:e - k]) * 1e4
                    ok = np.isfinite(xi) & np.isfinite(yi)
                    xi, yi = xi[ok], yi[ok]
                    b = np.digitize(xi, BIN_EDGES)
                    cnt += np.bincount(b, minlength=len(BIN_EDGES) + 1)
                    ssum += np.bincount(b, weights=yi, minlength=len(BIN_EDGES) + 1)
                for i in range(len(cnt)):
                    if cnt[i] >= 500:
                        bins_rows.append({"x": xname, "day_type": dtl, "k": k, "bin_i": i,
                                          "n": int(cnt[i]), "mean_bp": ssum[i] / cnt[i]})

    F = pl.DataFrame(fits)
    F.write_parquet(ROOT / "data" / f"book_slope_fits_{tag}.parquet")
    B = pl.DataFrame(bins_rows)
    B.write_parquet(ROOT / "data" / f"book_slope_bins_{tag}.parquet")
    F.write_csv(ROOT / "data" / f"book_slope_fits_{tag}.csv")

    # ---- まとめ表示 ----------------------------------------------------------
    print("\n=== BookSlope_z / 立会日 ===", file=sys.stderr)
    s = F.filter((pl.col("x") == "BookSlope_z") & (pl.col("day_type") == "立会日"))
    print(f"{'k':>5}{'beta(OLS)':>12}{'t(古典)':>11}{'t(HAC)':>10}{'beta(GLS)':>12}"
          f"{'beta(非重複)':>14}{'R2':>10}{'DW':>8}{'プラセボ':>10}", file=sys.stderr)
    for r in s.sort("k").iter_rows(named=True):
        print(f"{r['k']:>5}{r['beta_ols']:>12.4f}{r['t_ols']:>11.1f}{r['t_hac']:>10.1f}"
              f"{r['beta_gls']:>12.4f}{r['beta_nonoverlap']:>14.4f}{r['r2']:>10.5f}"
              f"{r['dw']:>8.3f}{r['beta_placebo']:>10.4f}", file=sys.stderr)
    print(f"\n-> data/book_slope_fits_{tag}.parquet / _bins_ / _meta_", file=sys.stderr)


if __name__ == "__main__":
    main()
