r"""Lighter × Binance のリードラグ — 図と日クラスタ区間。

    uv run python scripts/xv_lighter_report.py --sym MU

出力: charts/lighter_MU_xv_leadlag.png / data/xvl_asym_<sym>.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from _chartstyle import C1, C2, C3, C4, CM, GRID, D, plt, save

RNG = np.random.default_rng(20260915)
NB = 3000
SESS = ["00-08 夜", "08-13.5 プレ", "13.5-20 現物", "20-24 アフター"]


def boot_ci(days, vals, nb=NB):
    u = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in u}
    out = np.empty(nb)
    for b in range(nb):
        sel = np.concatenate([idx[d] for d in RNG.choice(u, u.size, True)])
        out[b] = vals[sel].mean()
    return float(vals.mean()), float(np.quantile(out, .025)), \
        float(np.quantile(out, .975))


def asym(P, group="全体"):
    Q = P.filter(pl.col("group") == group)
    W = (Q.filter(pl.col("lag_ms") > 0)
         .join(Q.filter(pl.col("lag_ms") < 0)
               .with_columns((-pl.col("lag_ms")).alias("lag_ms"))
               .rename({"corr": "cn"}).select("dt", "lag_ms", "cn"),
               on=["dt", "lag_ms"], how="inner")
         .with_columns((pl.col("corr") - pl.col("cn")).alias("a")))
    rows = []
    for lg in sorted(W["lag_ms"].unique().to_list()):
        X = W.filter(pl.col("lag_ms") == lg)
        m, lo, hi = boot_ci(X["dt"].to_numpy(), X["a"].to_numpy())
        rows.append({"group": group, "lag_ms": lg, "asym": m, "lo": lo,
                     "hi": hi, "n_days": X.height})
    return pl.DataFrame(rows)


def mean_ccf(P, group="全体"):
    return (P.filter(pl.col("group") == group).group_by("lag_ms")
            .agg(pl.col("corr").mean().alias("m"),
                 pl.col("corr").std().alias("s"), pl.len().alias("n"))
            .sort("lag_ms"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="MU")
    a = ap.parse_args()
    s = a.sym

    P = pl.read_parquet(D / f"xvl_ccfday_{s}_local_c100.parquet")
    PE = pl.read_parquet(D / f"xvl_ccfday_{s}_exch_c100.parquet")
    PS = pl.read_parquet(D / f"xvl_ccfday_{s}_local_c100_strict.parquet")
    f_bt = D / f"xvl_ccfday_{s}_local_c100_btrade.parquet"
    PT = pl.read_parquet(f_bt) if os.path.exists(f_bt) else None
    V = pl.read_csv(D / f"xvl_dev_{s}_local_c100.csv")
    DD = pl.read_csv(D / f"xvl_daily_{s}_local_c100.csv").filter(pl.col("ok"))

    A = pl.concat([asym(P, g) for g in ["全体"] + SESS])
    A.write_csv(D / f"xvl_asym_{s}.csv")
    g0 = A.filter(pl.col("group") == "全体").sort("lag_ms")

    print(f"Lighter {s} × Binance {s}USDT / {P['dt'].n_unique()} 日\n")
    print("【1】非対称 A(k) = corr(Binance が k 先) − corr(Lighter が k 先)")
    print(f"{'ラグ(ms)':>9s}{'A(k)':>9s}{'95% 区間':>20s}{'n 日':>6s}")
    for r in g0.iter_rows(named=True):
        if r["lag_ms"] > 600 and r["lag_ms"] % 500:
            continue
        print(f"{r['lag_ms']:>9d}{r['asym']:>9.4f}"
              f"  [{r['lo']:+.4f},{r['hi']:+.4f}]{r['n_days']:>6d}")
    pos = g0.filter(pl.col("lo") > 0)
    print(f"\n0 を跨がないラグ {pos.height}/{g0.height} 本")

    c0 = mean_ccf(P)
    lag0 = float(c0.filter(pl.col("lag_ms") == 0)["m"][0])
    print(f"\n【2】同時点(ラグ 0)の相関 = {lag0:.4f}")
    VAR = [("ローカル受信", P), ("取引所打刻", PE), ("両側が実更新", PS)]
    if PT is not None:
        VAR.append(("Binance=約定値", PT))
    for nm, Q in VAR:
        q = mean_ccf(Q)
        v = {int(r): float(m) for r, m in zip(q["lag_ms"], q["m"])}
        print(f"  {nm:>10s}  −100ms {v.get(-100, np.nan):.4f} / "
              f"0 {v.get(0, np.nan):.4f} / +100ms {v.get(100, np.nan):.4f} / "
              f"非対称 {v.get(100, np.nan)-v.get(-100, np.nan):+.4f}")

    hl = V["half_life_ms"].drop_nulls().drop_nans().to_numpy()
    print(f"\n【3】乖離 |d| 中央 {V['dev_abs_p50'].median():.2f}bp / "
          f"90% 点 {V['dev_abs_p90'].median():.2f}bp / "
          f"基差 {V['basis_p50'].median():+.2f}bp / "
          f"半減期 中央 {np.median(hl):.0f}ms")
    print(f"\n【4】打刻→受信の中央(日ごと)。両会場のずれが同じなら"
          f"共通のクロック誤差として相殺する")
    dl = DD["lgt_lat_ms_p50"].to_numpy()
    db = DD["bnc_lat_ms_p50"].to_numpy()
    print(f"  Lighter 中央 {np.median(dl):+.1f}ms / Binance {np.median(db):+.1f}ms")
    print(f"  日ごとの差 |Lighter − Binance| 中央 {np.median(np.abs(dl-db)):.2f}ms"
          f" / 最大 {np.max(np.abs(dl-db)):.2f}ms")

    # ================= 図 =================
    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    b = ax[0, 0]
    x, m = c0["lag_ms"].to_numpy(), c0["m"].to_numpy()
    se = c0["s"].to_numpy() / np.sqrt(c0["n"].to_numpy())
    b.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=C1, alpha=.2, lw=0)
    b.plot(x, m, lw=2.0, color=C1)
    b.axvline(0, color=CM, lw=1.0, ls="--")
    b.axhline(0, color=GRID, lw=1.0)
    b.set_xlabel("ラグ (ms)。正 = Binance が先")
    b.set_ylabel("100ms リターンの相関")
    b.set_title(f"★(a) ほぼ同時に動く — ラグ 0 で {lag0:.2f}",
                fontsize=9.5, loc="left")

    b = ax[0, 1]
    b.fill_between(g0["lag_ms"], g0["lo"], g0["hi"], color=C2, alpha=.22, lw=0)
    b.plot(g0["lag_ms"], g0["asym"], lw=2.0, color=C2)
    b.axhline(0, color=CM, lw=1.2)
    b.set_xlabel("ラグ (ms)")
    b.set_ylabel("A(k) = corr(+k) − corr(−k)")
    b.set_title("★(b) 非対称は ±100ms にだけ集中する", fontsize=9.5,
                loc="left")

    b = ax[0, 2]
    cfg = [("ローカル受信", P, C1, "-"), ("取引所打刻", PE, C2, "--"),
           ("両側が実更新", PS, C3, ":")]
    if PT is not None:
        cfg.append(("Binance=約定値", PT, C4, "-."))
    for nm, Q, c, ls in cfg:
        q = mean_ccf(Q)
        b.plot(q["lag_ms"], q["m"], lw=1.8, color=c, ls=ls, label=nm)
    b.axvline(0, color=CM, lw=1.0, ls="--")
    b.set_xlim(-700, 700)
    b.set_xlabel("ラグ (ms)")
    b.set_ylabel("相関")
    b.set_title("★(c) 時計・絞り・気配/約定値を替えても同じ", fontsize=9.5,
                loc="left")
    b.legend(fontsize=8, frameon=False)

    # (d) Hyperliquid との比較
    b = ax[1, 0]
    f_hl = D / "xv_ccfday_xyz_MU_c100.parquet"
    if os.path.exists(f_hl):
        H = mean_ccf(pl.read_parquet(f_hl))
        b.plot(H["lag_ms"], H["m"], lw=2.0, color=C4,
               label="Hyperliquid xyz:MU(99 日)")
    b.plot(x, m, lw=2.0, color=C1, label=f"Lighter {s}(28 日)")
    b.axvline(0, color=CM, lw=1.0, ls="--")
    b.set_yscale("symlog", linthresh=0.01)
    b.set_yticks([0, 0.001, 0.01, 0.1, 0.6],
                 ["0", "0.001", "0.01", "0.1", "0.6"], fontsize=8)
    b.set_xlabel("ラグ (ms)。正 = Binance が先")
    b.set_ylabel("相関(対数目盛)")
    b.set_title("★(d) 同じ Binance に対し、2 つの DEX は別物",
                fontsize=9.5, loc="left")
    b.legend(fontsize=8, frameon=False)

    b = ax[1, 1]
    CLIP = 4000
    nout = int((hl > CLIP).sum())
    b.hist(np.clip(hl, 0, CLIP), bins=40, color=C1, alpha=.85, lw=0)
    b.axvline(np.median(hl), color=C2, lw=2.0,
              label=f"Lighter 中央 {np.median(hl):.0f}ms")
    b.axvline(179, color=C4, lw=1.8, ls="--", label="HL 中央 179ms")
    b.set_xlabel(f"乖離 d の半減期 (ms)。{CLIP}ms 超の {nout} 日は右端")
    b.set_ylabel("日数")
    b.set_title("(e) 乖離は小さく、長生きする", fontsize=9.5, loc="left")
    b.legend(fontsize=8, frameon=False)

    b = ax[1, 2]
    xi = np.arange(DD.height)
    b.plot(xi, dl, "o-", lw=1.4, ms=3.5, color=C1, label="Lighter")
    b.plot(xi, db, "o-", lw=1.4, ms=3.5, color=C2, label="Binance")
    b.set_xticks(xi[::5], [d[5:] for d in DD["dt"].to_list()[::5]], fontsize=8)
    b.set_xlabel("日付 (2026)")
    b.set_ylabel("打刻 → 受信 の中央 (ms)")
    b.set_title("★(f) 時計は数百 ms ずれるが、両会場で同じ",
                fontsize=9.5, loc="left")
    b.legend(fontsize=8, frameon=False)

    save(fig, f"lighter_{s}_xv_leadlag.png",
         f"Lighter {s} × Binance {s}USDT — 会場間リードラグ"
         f"(28 日・100ms 格子・同一ホスト記録)")


if __name__ == "__main__":
    main()
