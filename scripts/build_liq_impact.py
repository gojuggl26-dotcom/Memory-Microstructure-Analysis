"""清算の前後で価格はどう動いたか — イベントスタディと 3 通りの帰無対照。

    uv run python scripts/build_liq_impact.py --coin xyz:MU

入力: data/liqev_<coin>.parquet / data/bbo_<coin>.parquet
出力: data/liqimp_<coin>.csv       前後のリターン(向き別・対照別)
      data/liqimp_cond_<coin>.csv  「直前の動きを揃えた」条件つき比較
      data/_liqmid_<coin>.npz      1 秒格子の mid(再利用のため)

★何を測っているか
------------------
清算のラウンド時刻 τ(同じミリ秒に起きる清算をまとめたもの)について

    後ろ向き  r(τ-Δ → τ)   清算に至るまでに価格がどれだけ動いたか
    前向き    r(τ → τ+Δ)   清算のあと価格がどうなったか

を 1 秒格子の mid で測る。**τ は公開テープに載る時刻**なので、前向きの
リターンは原理的には観測可能な情報に対する将来である(ただし後述のとおり
費用は一切引いていないので、儲かるという主張ではない)。

★帰無対照を 3 つ置く
--------------------
1. 無作為   標本期間の任意の秒から同数を引く
2. 層別     同じ (日, 時) の枠から引く。時間帯と地合いの濃淡を揃える
3. 条件つき **直前 5 分の動きを十分位で揃えて**比べる。これが本命で、
            「価格が既に動いていたから清算が出た」分を差し引く

★板がクロスしている秒は除く(MAINTENANCE §3)。

x が確定する時刻 / y の期間: 後ろ向きは τ 以前、前向きは τ より後。
両者を混ぜない。条件付けに使う直前の動きも τ 以前で閉じている。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
NS = 10 ** 9
LAGS = [5, 30, 60, 300, 1800, 3600]


def midgrid(tag: str) -> tuple[np.ndarray, int]:
    fp = D / f"_liqmid_{tag}.npz"
    if fp.exists():
        z = np.load(fp)
        return z["mid"], int(z["t0"])
    b = (pl.scan_parquet(D / f"bbo_{tag}.parquet")
         .filter(pl.col("best_ask") > pl.col("best_bid"))
         .select("ts", mid=(pl.col("best_bid") + pl.col("best_ask")) / 2)
         .sort("ts").collect())
    bt = b["ts"].to_numpy() // NS
    bm = b["mid"].to_numpy()
    t0, t1 = int(bt[0]), int(bt[-1])
    grid = np.arange(t0, t1 + 1)
    j = np.searchsorted(bt, grid, side="right") - 1
    mid = np.where(j >= 0, bm[np.maximum(j, 0)], np.nan)
    # 1 時間以上更新が無い区間は欠測にする(休場の穴を実在の価格として扱わない)
    stale = grid - bt[np.maximum(j, 0)]
    mid[stale > 3600] = np.nan
    np.savez_compressed(fp, mid=mid, t0=t0)
    print(f"[mid] 1 秒格子 {mid.size:,} 点 / 欠測 {np.isnan(mid).mean()*100:.2f}%",
          file=sys.stderr)
    return mid, t0


def rets(mid: np.ndarray, idx: np.ndarray, lag: int, fwd: bool) -> np.ndarray:
    k = idx + lag if fwd else idx - lag
    ok = (k >= 0) & (k < mid.size)
    out = np.full(idx.size, np.nan)
    a = mid[idx[ok]]
    b = mid[k[ok]]
    r = np.log(b / a) * 1e4 * (1 if fwd else -1)
    out[ok] = r
    return out


def boot(x: np.ndarray, day: np.ndarray, rng, n=2000) -> tuple[float, float, float]:
    """日単位のブロック bootstrap で中央値の 95% 区間を出す。"""
    m = np.isfinite(x)
    x, day = x[m], day[m]
    if x.size < 30:
        return np.nan, np.nan, np.nan
    uq = np.unique(day)
    gl = [np.flatnonzero(day == d) for d in uq]
    s = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, uq.size, uq.size)
        s[i] = np.median(x[np.concatenate([gl[j] for j in pick])])
    return float(np.median(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(0)
    mid, t0 = midgrid(tag)

    ev = pl.read_parquet(D / f"liqev_{tag}.parquet").sort("ts")
    ts = ev["ts"].dt.epoch("s").to_numpy()
    rt, inv = np.unique(ts, return_inverse=True)
    rnd = (pl.DataFrame({"r": inv, "usd": ev["notional"], "long": (ev["lside"] == "long")})
           .group_by("r").agg(pl.col("usd").sum(), pl.len().alias("n"),
                              pl.col("long").mean().alias("ls")).sort("r"))
    ridx = rt - t0
    okr = (ridx >= 0) & (ridx < mid.size) & np.isfinite(mid[np.clip(ridx, 0, mid.size - 1)])
    print(f"[imp] ラウンド {rt.size:,} / mid が引けた {int(okr.sum()):,}", file=sys.stderr)
    ridx, rnd, rt = ridx[okr], rnd.filter(pl.Series(okr)), rt[okr]
    lsh = rnd["ls"].to_numpy()
    grp = np.where(lsh >= 0.9, "long", np.where(lsh <= 0.1, "short", "mix"))
    day = (rt // 86400)

    # 対照 1: 無作為 / 対照 2: 同じ (日, 時) / 対照 3: プラセボ(+1 時間)
    fin = np.flatnonzero(np.isfinite(mid))
    n_null = 20 * ridx.size
    r_rand = rng.choice(fin, n_null, replace=True)
    hr = (rt // 3600)
    r_strat = np.empty(n_null, np.int64)
    for i in range(20):
        jitter = rng.integers(0, 3600, ridx.size)
        r_strat[i * ridx.size:(i + 1) * ridx.size] = (hr * 3600 - t0) + jitter
    r_strat = np.clip(r_strat, 0, mid.size - 1)
    r_strat = r_strat[np.isfinite(mid[r_strat])]
    r_plac = np.clip(ridx + 3600, 0, mid.size - 1)
    r_plac = r_plac[np.isfinite(mid[r_plac])]

    print(f"[imp] ラウンドの向き: ロング {int((grp=='long').sum()):,} / "
          f"ショート {int((grp=='short').sum()):,} / 混在 {int((grp=='mix').sum()):,}",
          file=sys.stderr)
    rows = []
    sets = [("清算(ロング)", ridx[grp == "long"], day[grp == "long"]),
            ("清算(ショート)", ridx[grp == "short"], day[grp == "short"]),
            ("対照1 無作為", r_rand, (r_rand + t0) // 86400),
            ("対照2 同じ時間枠", r_strat, (r_strat + t0) // 86400),
            ("対照3 プラセボ +1h", r_plac, (r_plac + t0) // 86400)]
    for name, ix, dy in sets:
        for lag in LAGS:
            for fwd in (False, True):
                r = rets(mid, ix, lag, fwd)
                med, lo_, hi_ = boot(r, dy, rng, n=600)
                rows.append({"set": name, "n": int(np.isfinite(r).sum()),
                             "dir": "fwd" if fwd else "bwd", "lag_s": lag,
                             "median_bp": med, "lo": lo_, "hi": hi_,
                             "mean_bp": float(np.nanmean(r))})
    imp = pl.DataFrame(rows)
    imp.write_csv(D / f"liqimp_{tag}.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(imp.filter(pl.col("lag_s").is_in([60, 300]))
              .with_columns(k=pl.col("dir") + "_" + pl.col("lag_s").cast(pl.Utf8))
              .pivot(values="median_bp", index="set", on="k")
              .select("set", "bwd_300", "bwd_60", "fwd_60", "fwd_300"))

    # ---- 条件つき: 直前 5 分の動きを揃えて比べる ------------------------
    # 1 秒格子の全点でリターンを作り、清算の前後 5 分を除いた点を対照の母集団にする。
    lg = np.log(mid)
    def sh(k):                       # k>0 なら未来、k<0 なら過去
        o = np.full(lg.size, np.nan)
        if k >= 0:
            o[:lg.size - k] = lg[k:]
        else:
            o[-k:] = lg[:lg.size + k]
        return o
    B300 = (lg - sh(-300)) * 1e4
    F60 = (sh(60) - lg) * 1e4
    F300 = (sh(300) - lg) * 1e4
    near = np.zeros(lg.size, bool)
    for d in range(-300, 301):
        k = np.clip(ridx + d, 0, lg.size - 1)
        near[k] = True
    pool = np.isfinite(B300) & np.isfinite(F300) & ~near
    print(f"\n[条件つき] 対照の母集団 {int(pool.sum()):,} 秒"
          f"(清算の前後 5 分 {int(near.sum()):,} 秒を除外)")

    EDG = np.array([-1e9, -150, -100, -70, -50, -35, -25, -15, -8, -3, 0,
                    3, 8, 15, 25, 35, 50, 70, 100, 150, 1e9])
    ge = np.digitize(B300[ridx], EDG) - 1
    gn = np.digitize(np.where(pool, B300, np.nan), EDG) - 1
    gn[~pool] = -1
    cond_rows = []
    for d in range(EDG.size - 1):
        me = (ge == d) & np.isfinite(F300[ridx])
        mn = gn == d
        if me.sum() < 10:
            continue
        cond_rows.append({
            "bin_lo_bp": float(EDG[d]), "bin_hi_bp": float(EDG[d + 1]),
            "n_liq": int(me.sum()), "n_null": int(mn.sum()),
            "liq_bwd300": float(np.nanmedian(B300[ridx][me])),
            "null_bwd300": float(np.nanmedian(B300[mn])),
            "liq_long_share": float(np.nanmean((grp == "long")[me])),
            "liq_fwd60": float(np.nanmedian(F60[ridx][me])),
            "null_fwd60": float(np.nanmedian(F60[mn])),
            "liq_fwd300": float(np.nanmedian(F300[ridx][me])),
            "null_fwd300": float(np.nanmedian(F300[mn])),
        })
    cond = pl.DataFrame(cond_rows).with_columns(
        diff60=pl.col("liq_fwd60") - pl.col("null_fwd60"),
        diff300=pl.col("liq_fwd300") - pl.col("null_fwd300"),
        match_gap=(pl.col("liq_bwd300") - pl.col("null_bwd300")).abs())
    cond.write_csv(D / f"liqimp_cond_{tag}.csv")
    print("直前 5 分の動きの帯ごとに、その後 1 分・5 分の中央値(bp)")
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(cond.select("bin_lo_bp", "bin_hi_bp", "n_liq", "n_null",
                          "liq_bwd300", "null_bwd300", "match_gap",
                          "liq_long_share", "liq_fwd60", "null_fwd60", "diff60",
                          "liq_fwd300", "null_fwd300", "diff300"))
    w = cond["n_liq"].to_numpy()
    print(f"件数で重みづけた差: 1 分 {np.average(cond['diff60'], weights=w):+.2f} bp / "
          f"5 分 {np.average(cond['diff300'], weights=w):+.2f} bp")

    # ---- 直前の動きを差し引いた「超過」リターンを、日ブロック bootstrap で ----
    ex_rows = []
    gi = np.digitize(B300[ridx], EDG) - 1
    for lag in LAGS:
        F = (sh(lag) - lg) * 1e4
        base = np.full(EDG.size - 1, np.nan)
        gnull = np.digitize(np.where(pool, B300, np.nan), EDG) - 1
        gnull[~pool] = -1
        for d in range(EDG.size - 1):
            m = gnull == d
            if m.sum() > 200:
                base[d] = np.nanmedian(F[m])
        ex = F[ridx] - base[np.clip(gi, 0, EDG.size - 2)]
        for nm, k in (("ロング", grp == "long"), ("ショート", grp == "short")):
            med, lo_, hi_ = boot(ex[k], day[k], rng, n=1000)
            raw, _, _ = boot(F[ridx][k], day[k], rng, n=200)
            ex_rows.append({"side": nm, "lag_s": lag, "n": int(np.isfinite(ex[k]).sum()),
                            "raw_bp": raw, "excess_bp": med, "lo": lo_, "hi": hi_})
    exd = pl.DataFrame(ex_rows)
    exd.write_csv(D / f"liqexcess_{tag}.csv")
    print("\n[超過] 直前 5 分の動きが同じ秒の中央値を引いた、清算後のリターン(bp)")
    with pl.Config(tbl_rows=30, tbl_width_chars=140):
        print(exd)


if __name__ == "__main__":
    main()
