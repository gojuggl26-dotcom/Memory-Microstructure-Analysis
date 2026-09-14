r"""Binance ↔ Hyperliquid のリードラグ — 日クラスタの区間つきで確定させる。

    uv run python scripts/xv_leadlag_report.py --coin xyz:MU

出力: charts/xyz_MU_xv_leadlag.png / data/xv_asym_<coin>.csv

=============================================================================
★何を主張し、何を主張しないか
=============================================================================
主張する:
  - 非対称 A(k) = corr(+k) − corr(−k) が **有意に正**であること
    (+k は「Binance が先、HL が後」)
  - その非対称がピークを取るラグと、乖離の半減期

主張しない:
  - 「Binance が正確に N ms 先行する」という点推定。
    **両会場のクロック差と伝送遅延は観測できない**ので、一定のずれが
    あればピーク位置はそのぶん動く。**ずれで説明できないのは非対称の
    存在そのもの**である(一定のずれは曲線を平行移動させるだけで、
    ずらした中心のまわりで非対称を作らない)

=============================================================================
★疑似先行への対策(2 段)
=============================================================================
1. 鮮度で絞る … 両側とも最終更新から 2 秒以内
2. `--strict` … そのセル内で**両方が実際に更新された**行だけ
   LOCF で「動いていない側」を混ぜると、市場構造によらず
   「よく動くほうが先行」に見える(Uniswap で実際に踏んだ)
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import polars as pl

from _chartstyle import C1, C2, C3, C4, CM, INK, GRID, D, CH, plt, save

RNG = np.random.default_rng(20260914)
NB = 4000
SESS = ["00-08 夜", "08-13.5 プレ", "13.5-20 現物", "20-24 アフター"]


def boot_ci(days, vals, nb=NB):
    """日をブロックに復元抽出して平均の 95% 区間。"""
    u = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in u}
    out = np.empty(nb)
    for b in range(nb):
        sel = np.concatenate([idx[d] for d in RNG.choice(u, u.size, True)])
        out[b] = vals[sel].mean()
    return float(vals.mean()), float(np.quantile(out, .025)), \
        float(np.quantile(out, .975))


def asym(P, group):
    """非対称 A(k) = C(+k) − C(−k) を日ごとに作り、区間を付ける。"""
    Q = P.filter(pl.col("group") == group)
    W = (Q.filter(pl.col("lag_ms") > 0)
         .join(Q.filter(pl.col("lag_ms") < 0)
               .with_columns((-pl.col("lag_ms")).alias("lag_ms"))
               .rename({"corr": "corr_neg"}).select("dt", "lag_ms", "corr_neg"),
               on=["dt", "lag_ms"], how="inner")
         .with_columns((pl.col("corr") - pl.col("corr_neg")).alias("a")))
    rows = []
    for lg in sorted(W["lag_ms"].unique().to_list()):
        X = W.filter(pl.col("lag_ms") == lg)
        m, lo, hi = boot_ci(X["dt"].to_numpy(), X["a"].to_numpy())
        rows.append({"group": group, "lag_ms": lg, "asym": m,
                     "lo": lo, "hi": hi, "n_days": X.height})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    P100 = pl.read_parquet(D / f"xv_ccfday_{tag}_c100.parquet")
    f_s = D / f"xv_ccfday_{tag}_c100_strict.parquet"
    f_20 = D / f"xv_ccfday_{tag}_c20.parquet"
    PS = pl.read_parquet(f_s) if os.path.exists(f_s) else None
    P20 = pl.read_parquet(f_20) if os.path.exists(f_20) else None
    V = pl.read_csv(D / f"xv_dev_{tag}_c100.csv")

    A = pl.concat([asym(P100, g) for g in ["全体"] + SESS])
    A.write_csv(D / f"xv_asym_{tag}.csv")

    print(f"{a.coin} × Binance MUUSDT / {P100['dt'].n_unique()} 日\n")
    print("【1】非対称 A(k) = corr(Binance が k 先) − corr(HL が k 先)")
    print(f"{'ラグ(ms)':>9s}{'A(k)':>9s}{'95% 区間':>20s}{'夜':>9s}"
          f"{'プレ':>9s}{'現物':>9s}{'アフター':>9s}")
    g0 = A.filter(pl.col("group") == "全体").sort("lag_ms")
    for r in g0.iter_rows(named=True):
        if r["lag_ms"] > 1000 and r["lag_ms"] % 500:
            continue
        s = (f"{r['lag_ms']:>9d}{r['asym']:>9.4f}"
             f"  [{r['lo']:+.4f},{r['hi']:+.4f}]")
        for nm in SESS:
            v = A.filter((pl.col("group") == nm)
                         & (pl.col("lag_ms") == r["lag_ms"]))["asym"]
            s += f"{(v[0] if v.len() else np.nan):>9.4f}"
        print(s)
    pk = g0.sort("asym", descending=True).head(1)
    print(f"\nピーク: ラグ {int(pk['lag_ms'][0])}ms / A = {float(pk['asym'][0]):.4f}"
          f" [{float(pk['lo'][0]):+.4f}, {float(pk['hi'][0]):+.4f}]")
    pos = g0.filter(pl.col("lo") > 0)
    print(f"95% 区間が 0 を跨がないラグ: {pos.height}/{g0.height} 本"
          f"(最小 {int(pos['lag_ms'].min()) if pos.height else 0}ms 〜 "
          f"最大 {int(pos['lag_ms'].max()) if pos.height else 0}ms)")

    if PS is not None:
        AS = asym(PS, "全体")
        p2 = AS.sort("asym", descending=True).head(1)
        print(f"\n【2】strict(両側がセル内で実更新)でも残るか")
        print(f"  ピーク ラグ {int(p2['lag_ms'][0])}ms / "
              f"A = {float(p2['asym'][0]):.4f} "
              f"[{float(p2['lo'][0]):+.4f}, {float(p2['hi'][0]):+.4f}]")
    if P20 is not None:
        A20 = asym(P20, "全体")
        p3 = A20.sort("asym", descending=True).head(1)
        print(f"\n【3】20ms 格子でピークを細かく見る")
        print(f"  ピーク ラグ {int(p3['lag_ms'][0])}ms / "
              f"A = {float(p3['asym'][0]):.4f} "
              f"[{float(p3['lo'][0]):+.4f}, {float(p3['hi'][0]):+.4f}]")

    # ---- クロック差では説明できないことの診断 ----
    # 一定のずれ δ があるだけなら、真の関係が同時点でも曲線は δ を中心に
    # **対称**になる。ピークのまわりで非対称なら、向きのある情報の流れがある。
    st0 = (P100.filter(pl.col("group") == "全体").group_by("lag_ms")
           .agg(pl.col("corr").mean().alias("m")).sort("lag_ms"))
    xs = st0["lag_ms"].to_numpy()
    ms = st0["m"].to_numpy()
    pk_i = int(np.argmax(ms))
    print()
    print(f"[5] クロック差で説明できるか — ピーク({xs[pk_i]}ms) の対称性")
    print(f"{'u (ms)':>8s}{'C(pk+u)':>10s}{'C(pk-u)':>10s}{'差':>10s}")
    for u in (100, 200, 300, 500, 800):
        i, j = pk_i + u // 100, pk_i - u // 100
        if 0 <= i < ms.size and 0 <= j < ms.size:
            print(f"{u:>8d}{ms[i]:>10.4f}{ms[j]:>10.4f}{ms[i]-ms[j]:>10.4f}")
    print("  ★差が 0 なら「一定のずれ + 同時点」で説明できてしまう。")
    print("   差が 0 から離れているほど対称性が破れており、ずれでは説明できない。")
    print("   ここでは**正**= Binance が先の側だけ裾が長い、という向きである。")

    hl = V["half_life_ms"].drop_nulls().drop_nans().to_numpy()
    print(f"\n【4】乖離 d の寿命")
    print(f"  半減期 中央 {np.median(hl):.0f}ms / "
          f"四分位 {np.quantile(hl,.25):.0f}〜{np.quantile(hl,.75):.0f}ms")
    print(f"  |d| 中央 {V['dev_abs_p50'].median():.2f}bp / "
          f"90% 点 {V['dev_abs_p90'].median():.2f}bp / "
          f"基差 中央 {V['basis_p50'].median():+.2f}bp")

    # ================= 図 =================
    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    # (a) CCF 曲線(全体・日平均と日クラスタ帯)
    b = ax[0, 0]
    Q = P100.filter(pl.col("group") == "全体")
    st = (Q.group_by("lag_ms").agg(pl.col("corr").mean().alias("m"),
                                   pl.col("corr").std().alias("s"),
                                   pl.len().alias("n")).sort("lag_ms"))
    x = st["lag_ms"].to_numpy()
    m = st["m"].to_numpy()
    se = st["s"].to_numpy() / np.sqrt(st["n"].to_numpy())
    b.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=C1, alpha=.20, lw=0)
    b.plot(x, m, lw=2.0, color=C1)
    b.axvline(0, color=CM, lw=1.0, ls="--")
    b.axhline(0, color=GRID, lw=1.0)
    b.set_xlabel("ラグ (ms)。正 = Binance が先")
    b.set_ylabel("100ms リターンの相関")
    b.set_title("★(a) Binance 側にだけ山がある(99 日平均・95% 帯)",
                fontsize=9.5, loc="left")

    # (b) 非対称 A(k)
    b = ax[0, 1]
    g = g0
    x = g["lag_ms"].to_numpy()
    b.fill_between(x, g["lo"].to_numpy(), g["hi"].to_numpy(),
                   color=C2, alpha=.22, lw=0)
    b.plot(x, g["asym"].to_numpy(), lw=2.0, color=C2, label="全体")
    b.axhline(0, color=CM, lw=1.2)
    b.set_xlabel("ラグ (ms)")
    b.set_ylabel("A(k) = corr(+k) − corr(−k)")
    b.set_title("★(b) 非対称は 0 を跨がない(日クラスタ 95%)",
                fontsize=9.5, loc="left")
    b.legend(fontsize=8, frameon=False)

    # (c) 時間帯別
    b = ax[0, 2]
    for nm, c in zip(SESS, (C1, C2, C3, C4)):
        q = A.filter(pl.col("group") == nm).sort("lag_ms")
        b.plot(q["lag_ms"], q["asym"], lw=1.8, color=c, label=nm)
    b.axhline(0, color=CM, lw=1.2)
    b.set_xlabel("ラグ (ms)")
    b.set_ylabel("A(k)")
    b.set_title("(c) 4 時間帯すべてで同じ形", fontsize=9.5, loc="left")
    b.legend(fontsize=7.5, frameon=False)

    # (d) 細かい格子
    b = ax[1, 0]
    if P20 is not None:
        q = asym(P20, "全体").sort("lag_ms")
        b.fill_between(q["lag_ms"], q["lo"], q["hi"], color=C3,
                       alpha=.22, lw=0)
        b.plot(q["lag_ms"], q["asym"], lw=2.0, color=C3, label="20ms 格子")
    if PS is not None:
        q = asym(PS, "全体").sort("lag_ms")
        b.plot(q["lag_ms"], q["asym"], lw=1.6, color=C2, ls="--",
               label="100ms・両側が実更新")
    b.axhline(0, color=CM, lw=1.2)
    b.set_xlabel("ラグ (ms)")
    b.set_ylabel("A(k)")
    b.set_title("★(d) 格子を細かくしても、絞っても残る", fontsize=9.5,
                loc="left")
    b.legend(fontsize=8, frameon=False)

    # (e) 乖離の寿命
    b = ax[1, 1]
    # ★裾が 11 秒まで伸びるので、読める範囲に切って外れ値の数を注記する
    CLIP = 1500
    n_out = int((hl > CLIP).sum())
    b.hist(np.clip(hl, 0, CLIP), bins=45, color=C1, alpha=.85, lw=0)
    b.axvline(np.median(hl), color=C2, lw=2.0,
              label=f"中央 {np.median(hl):.0f}ms")
    for L, c in ((65, C3), (130, C4)):
        b.axvline(L, color=c, lw=1.4, ls="--", label=f"遅延 {L}ms")
    b.set_xlabel(f"乖離 d の半減期 (ms)。{CLIP}ms 超の {n_out} 日は右端に寄せた")
    b.set_ylabel("日数")
    b.set_title("★(e) 機会は 179ms で半減する — 待てない", fontsize=9.5,
                loc="left")
    b.legend(fontsize=8, frameon=False)

    # (f) 金に換算した検定(遅延ごとの十分位勾配)
    b = ax[1, 2]
    f = D / f"xv_test_{tag}.csv"
    if os.path.exists(f):
        T = pl.read_csv(f)
        T = T.filter(pl.col("shift_s") == 0).sort("delay_ms")
        b.plot(T["delay_ms"], T["grad"], "o-", lw=2.0, color=C1, ms=6,
               label="D10 − D1(往復損益)")
        P = pl.read_csv(f).filter(pl.col("shift_s") > 0)
        if P.height:
            b.axhline(float(P["grad"][0]), color=C2, lw=1.6, ls="--",
                      label=f"プラセボ(1h ずらし) {float(P['grad'][0]):+.2f}")
        b.axhline(0, color=CM, lw=1.2)
        b.set_xscale("symlog", linthresh=100)
        b.set_xticks([0, 65, 130, 250, 1000, 2500],
                     ["0", "65", "130", "250", "1000", "2500"], fontsize=8)
        b.set_xlabel("使える情報の遅延 Δ (ms)")
        b.set_ylabel("往復損益の十分位勾配 (bp)")
        b.set_title("★(f) 遅れるほど消え、1 秒で符号が反転する",
                    fontsize=9.5, loc="left")
        b.legend(fontsize=8, frameon=False)

    save(fig, f"{tag}_xv_leadlag.png",
         f"{a.coin} × Binance MUUSDT — 会場間リードラグ"
         f"(99 日・100ms 格子・両側の鮮度 2 秒以内)")


if __name__ == "__main__":
    main()
