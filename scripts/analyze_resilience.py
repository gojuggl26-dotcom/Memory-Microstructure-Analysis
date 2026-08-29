"""弾力性の指標を将来 log リターンへ回帰し、確率遷移行列も作る。

入力は build_resilience.py が書いた 1 ショック 1 行の表。

【回帰】
指標 × 側 × 日区分 × ホライズンごとに OLS。指標は右に大きく歪むものが多いので
**生と log(1+x) の両方**で当てる(100ms 窓の分散のレポートで、生のままだと
関係が消えることを実測している)。目的変数は 3 つ。

    符号つき  log mid_{Te+h} − log mid_{Te}     Te = ショック時刻 + 1 秒
    絶対値    |符号つき|
    二乗      符号つき^2

【確率遷移行列】
指標の帯 × ホライズンで **P(上昇 | 動いた)** を出す。価格は離散なので
短いホライズンでは「変化なし」が多数を占め、生の P(上昇) は 0.5 を下回って見える。
無条件値(その側・その日区分・そのホライズンでの全体値)を対照に置く。

【不確かさ】
同じ日のショックどうしは相関するので、**日を単位にしたブートストラップ**で
無条件値からの差の区間を出す。

    uv run python scripts/analyze_resilience.py --coin xyz:MU
出力: data/resil_ols_<coin>.csv    data/resil_trans_<coin>.csv
      data/resil_daily_<coin>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_resilience import HOR, METRICS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 400
SEED = 20260829


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy, syy = (x * x).sum(), (x * y).sum(), (y * y).sum()
    den = n * sxx - sx * sx
    dy = n * syy - sy * sy
    if n < 30 or den <= 0 or dy <= 0:
        return {}
    b1 = (n * sxy - sx * sy) / den
    b0 = (sy - b1 * sx) / n
    r = (n * sxy - sx * sy) / np.sqrt(den * dy)
    sse = max(syy - b0 * sy - b1 * sxy, 0.0)
    se = np.sqrt(sse / (n - 2) * n / den)
    return {"n": n, "beta": b1, "r": r, "r2": r * r,
            "t": b1 / se if se > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(SEED)
    files = sorted((ROOT / "data" / f"resil_{tag}").glob("dt=*.parquet"))
    S = pl.concat([pl.read_parquet(f) for f in files if f.stat().st_size > 900],
                  how="diagonal_relaxed")
    S = S.filter(pl.col("side").is_not_null())
    days = sorted(S["dt"].unique().to_list())
    print(f"[読み] ショック {S.height:,} 件 / {len(days)} 日", file=sys.stderr)
    print(S.group_by("side", "day_type").len().sort("side", "day_type"), file=sys.stderr)

    # ---- 回帰 ---------------------------------------------------------------
    rows = []
    for met in METRICS:
        for side in ("買い", "売り"):
            for dty in ("立会日", "閉場日"):
                sub = S.filter((pl.col("side") == side) & (pl.col("day_type") == dty))
                if sub.height < 100:
                    continue
                xv = sub[met].to_numpy().astype(float)
                fx = np.isfinite(xv)
                if fx.sum() < 100:
                    continue
                xf_all = {"生": xv[fx], "log1p": np.log1p(np.maximum(xv[fx], 0))}
                for hn in HOR:
                    yv = sub[f"y_{hn}"].to_numpy().astype(float)[fx]
                    m = np.isfinite(yv)
                    if m.sum() < 100:
                        continue
                    y0 = yv[m]
                    for xf, xx in ((k, v[m]) for k, v in xf_all.items()):
                        for yk, yy in (("符号つき", y0), ("絶対値", np.abs(y0)),
                                       ("二乗", y0 * y0)):
                            d = ols(xx, yy)
                            if d:
                                rows.append({"metric": met, "side": side,
                                             "day_type": dty, "hor": hn,
                                             "hor_ms": HOR[hn], "x_form": xf,
                                             "y_kind": yk} | d)
    O = pl.DataFrame(rows).sort("metric", "side", "day_type", "x_form", "y_kind", "hor_ms")
    O.write_csv(ROOT / "data" / f"resil_ols_{tag}.csv")
    O.write_parquet(ROOT / "data" / f"resil_ols_{tag}.parquet")
    print(f"[OLS] {O.height:,} 行", file=sys.stderr)

    # ---- 確率遷移行列 -------------------------------------------------------
    di = {d: i for i, d in enumerate(days)}
    day_ix = np.array([di[d] for d in S["dt"].to_list()])
    trs = []
    for met, (edges, unit) in METRICS.items():
        xv = S[met].to_numpy().astype(float)
        b = np.where(np.isfinite(xv), np.digitize(np.nan_to_num(xv), edges), -1)
        nb = len(edges) + 1
        for side in ("買い", "売り"):
            for dty in ("立会日", "閉場日"):
                sel = (S["side"].to_numpy() == side) & (S["day_type"].to_numpy() == dty)
                if sel.sum() < 500:
                    continue
                for hn in HOR:
                    y = S[f"y_{hn}"].to_numpy().astype(float)
                    ok = sel & np.isfinite(y) & (b >= 0)
                    if ok.sum() < 500:
                        continue
                    up, dn = y > 0, y < 0
                    base = up[ok].sum() / max((up | dn)[ok].sum(), 1)
                    for bi in range(nb):
                        m = ok & (b == bi)
                        nu, nd = int(up[m].sum()), int(dn[m].sum())
                        if nu + nd < 200:
                            continue
                        p = nu / (nu + nd)
                        # 日を単位にした再標本で「無条件との差」の区間を出す。
                        # ★日ごとの上昇・下落数を先に畳んでおく。畳まずに毎回
                        #   bincount を呼ぶと 130 万回になり 10 分で終わらない
                        nd_ = len(days)
                        dix = day_ix[m]
                        uu = np.bincount(dix, weights=up[m].astype(float), minlength=nd_)
                        dd = np.bincount(dix, weights=dn[m].astype(float), minlength=nd_)
                        pick = rng.integers(0, nd_, (N_BOOT, nd_))
                        U, Dn = uu[pick].sum(1), dd[pick].sum(1)
                        tot = U + Dn
                        bs = np.where(tot > 0, U / np.maximum(tot, 1), np.nan)
                        lo, hi = np.nanpercentile(bs - base, [2.5, 97.5])
                        trs.append({"metric": met, "unit": unit, "side": side,
                                    "day_type": dty, "hor": hn, "hor_ms": HOR[hn],
                                    "bin_i": bi, "n_up": nu, "n_dn": nd,
                                    "n": int(m.sum()), "p_up": p, "base": base,
                                    "diff": p - base, "lo": lo, "hi": hi})
    T = pl.DataFrame(trs).sort("metric", "side", "day_type", "hor_ms", "bin_i")
    T.write_csv(ROOT / "data" / f"resil_trans_{tag}.csv")
    T.write_parquet(ROOT / "data" / f"resil_trans_{tag}.parquet")
    print(f"[遷移] {T.height:,} 行", file=sys.stderr)

    # ---- 日ごとの κ と非対称 -------------------------------------------------
    D = (S.filter(pl.col("kappa_depth").is_finite())
          .group_by("dt", "side", "day_type")
          .agg(pl.col("kappa_depth").median().alias("kappa"),
               pl.col("t_half_ms").median().alias("t_half"),
               pl.len().alias("n"))
          .pivot(on="side", index=["dt", "day_type"], values=["kappa", "t_half", "n"])
          .sort("dt"))
    kb = [c for c in D.columns if c.startswith("kappa") and "買い" in c][0]
    ka = [c for c in D.columns if c.startswith("kappa") and "売り" in c][0]
    D = D.with_columns(
        asym=((pl.col(kb) - pl.col(ka)) / (pl.col(kb) + pl.col(ka))).alias("asym"))
    D.write_csv(ROOT / "data" / f"resil_daily_{tag}.csv")
    print(f"[日次] {D.height} 日  非対称の中央 {D['asym'].median():+.4f}", file=sys.stderr)
    print(f"\n-> data/resil_ols_{tag}.csv / resil_trans_{tag}.csv / resil_daily_{tag}.csv",
          file=sys.stderr)


if __name__ == "__main__":
    main()
