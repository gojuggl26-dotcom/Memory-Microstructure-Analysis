r"""Derive — ウォレットの実績を「技術」と「運」に分解する。

=============================================================================
問い
=============================================================================
上位 3 ウォレットが実現損益の 103% を占めている。これは平均を見ても意味がなく、
**持続する wallet alpha があるか**を検定すべき状況である。

  r_{i,t} = α_i + β'X_{i,t} + ε_{i,t}

  i = ウォレット、t = 日。目的変数は 4 つを別々に:
    (a) 純損益 / 名目 [bp]   (b) 純損益 [$]
    (c) オプションのみ [$]   (d) perp のみ [$]

=============================================================================
★設計の要点
=============================================================================
1. **共通ショックを落とす。**日固定効果 X_{i,t} = 日ダミーを入れると、α_i は
   「同じ日に他のウォレットより良かったか」になる。相場が動いた日に vega を
   持っていただけの「運」は日効果に吸われる。
   日固定効果なしの素の平均も併記して、どれだけ共通要因だったかを見る。

2. **★階層的縮約(empirical Bayes)。**α_i ~ N(μ_α, σ_α²) を仮定し、
   標本の少ないウォレットの見かけの大勝ちを 0 方向へ縮める:

       α_i^EB = μ_α + B_i (α̂_i − μ_α),   B_i = σ_α² / (σ_α² + s_i²)

   σ_α² はモーメント法で推定: Var(α̂) − mean(s_i²) を下限 0 で切る。
   事後の分散 τ_i² = (1/σ_α² + 1/s_i²)^{-1} から P(α_i > 0) = Φ(α_i^EB / τ_i)。
   ★s_i は日次残差の標準誤差(日を独立と見なす)。日内の相関は残るので
     過小評価の可能性がある。これは明記する。

3. **★ウォークフォワードで持続性を見る。**形成期 6 か月で順位を付け、
   次の 6 か月で同じ順位が保たれるかを測る。窓を 1 か月ずつ転がす。
     - Spearman rank IC(形成期の順位 vs 評価期の順位)
     - 上位四分位の残存率(形成期 Q4 → 評価期 Q4 の確率)
     - 帰無対照: 評価期のウォレット標識を並べ替えて同じ統計を取る

4. **母集団は結果を見る前に決める。**形成期・評価期の両方で
   **20 日以上**活動したウォレットのみ。これを満たさない者は
   「小標本の大勝ち」そのものなので、順位付けの対象にしない。
   併せて**メイカー主体の部分集合**(maker 比 70% 以上・200 約定以上)でも同じ検定を回す。

5. **★bp は名目で加重する。**日次 bp の単純平均を取ると、
   名目が極小の日の bp が効き過ぎる。実際、最初に単純平均で出したときは
   総損益 −$133,523 のウォレットが +1,180bp で 1 位になった。
   正しくは **Σ損益 / Σ名目**(= 名目加重平均)を使う。

出力: E:/Memory-derive/alpha/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "alpha"
BP = 1e4
MIN_DAYS = 20                      # ★事前に決めた活動日数の下限
FORM_M, EVAL_M = 6, 6              # 形成期 / 評価期(か月)


# ---------------------------------------------------------------- パネル
def build_panel() -> pl.DataFrame:
    """wallet × day のパネル。オプション売買 + 満期決済 + perp。"""
    T = pl.read_parquet(SRC / "ledger_trades.parquet")
    S = pl.read_parquet(SRC / "ledger_settle.parquet")
    W = pl.read_parquet(SRC / "wallet_map.parquet")
    P = pl.read_parquet(SRC / "ledger_perp.parquet")

    opt = (T.with_columns((pl.col("amt") * pl.col("index")).alias("no"))
           .group_by(["wallet", "day"]).agg([
               pl.col("realized").sum().alias("pnl_opt"),
               pl.col("no").sum().alias("notional_opt"),
               pl.len().alias("n_opt"),
               (pl.col("role") == "maker").mean().alias("maker_share"),
               pl.col("fee").sum().alias("fee"),
           ]))
    # 満期決済は subaccount 単位。wallet に付け替え、満期日を day にする
    se = (S.join(W, on="sub", how="left")
          .filter(pl.col("wallet").is_not_null())
          .with_columns(
              pl.when(pl.col("expiry") > 0)
              .then(pl.from_epoch(pl.col("expiry"), time_unit="s")
                    .dt.strftime("%Y-%m-%d"))
              .otherwise(pl.col("ins").str.split("-").list.get(1)
                         .str.strptime(pl.Date, "%Y%m%d").dt.strftime("%Y-%m-%d"))
              .alias("day"))
          .group_by(["wallet", "day"]).agg(pl.col("pnl").sum().alias("pnl_settle")))
    pp = (P.group_by(["wallet", "day"]).agg([
        pl.col("realized").sum().alias("pnl_perp"),
        pl.col("notional").sum().alias("notional_perp"),
        pl.col("n").sum().alias("n_perp")]))
    D = (opt.join(se, on=["wallet", "day"], how="full", coalesce=True)
            .join(pp, on=["wallet", "day"], how="full", coalesce=True)
            .fill_null(0.0))
    D = D.with_columns([
        (pl.col("pnl_opt") + pl.col("pnl_settle")).alias("pnl_option_side"),
        (pl.col("pnl_opt") + pl.col("pnl_settle")
         + pl.col("pnl_perp")).alias("pnl_total"),
        (pl.col("notional_opt") + pl.col("notional_perp")).alias("notional"),
    ])
    D = D.with_columns(
        (pl.col("pnl_total") / pl.when(pl.col("notional") > 0)
         .then(pl.col("notional")).otherwise(None) * BP).alias("bp"))
    return D.sort(["wallet", "day"])


# ------------------------------------------------ 日固定効果を落とした残差
def demean_by_day(D: pl.DataFrame, col: str, wtd: bool = False) -> pl.DataFrame:
    """その日の(加重)平均を引く = 日固定効果 X_{i,t} を落とす。

    ★共通ショック(相場が動いた日は誰でも勝つ/負ける)を吸わせる。
      残る差は「同じ日に他者より良かったか」= 技術の候補。
    """
    if wtd:
        g = (D.group_by("day").agg(
            (pl.col(col) * pl.col("notional")).sum().alias("s"),
            pl.col("notional").sum().alias("w"))
            .with_columns((pl.col("s") / pl.col("w")).alias("daymean"))
            .select(["day", "daymean"]))
    else:
        g = D.group_by("day").agg(pl.col(col).mean().alias("daymean"))
    return (D.join(g, on="day", how="left")
            .with_columns((pl.col(col) - pl.col("daymean")).alias(f"{col}_dm"))
            .drop("daymean"))


# ---------------------------------------------------- 経験ベイズ縮約
def shrink(a: np.ndarray, s: np.ndarray) -> dict:
    """α̂_i と標準誤差 s_i から階層的縮約。

    α_i ~ N(μ, σ²) を仮定。σ² はモーメント法:
        Var(α̂) = σ² + mean(s²)  ->  σ² = max(Var(α̂) − mean(s²), 0)
    """
    ok = np.isfinite(a) & np.isfinite(s) & (s > 0)
    a, s = a[ok], s[ok]
    mu = float(np.average(a, weights=1.0 / s ** 2)) if len(a) else 0.0
    var_hat = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    sig2 = max(var_hat - float(np.mean(s ** 2)), 0.0)
    B = sig2 / (sig2 + s ** 2) if sig2 > 0 else np.zeros_like(s)
    eb = mu + B * (a - mu)
    tau2 = 1.0 / (1.0 / sig2 + 1.0 / s ** 2) if sig2 > 0 else s ** 2 * 0.0
    tau = np.sqrt(tau2) if sig2 > 0 else np.full_like(s, np.inf)
    from scipy.stats import norm
    p = norm.cdf(eb / np.where(tau > 0, tau, np.inf))
    return {"ok": ok, "mu": mu, "sigma": float(np.sqrt(sig2)),
            "eb": eb, "B": B, "tau": tau, "p_gt0": p, "raw": a, "se": s}


def wallet_alpha(D: pl.DataFrame, col: str, min_days: int,
                 weighted: bool = False) -> pl.DataFrame:
    """ウォレットごとの α̂ と標準誤差。

    weighted=True のとき α̂ = Σ(x·w)/Σw(名目加重)。
    ★bp を単純平均すると、名目が極小の日の bp が効き過ぎる。
      実測で「総損益 −$133,523 のウォレットが +1,180bp で 1 位」という
      おかしな順位が出た。
    """
    d = D.filter(pl.col(col).is_not_null())
    if weighted:
        d = d.filter(pl.col("notional") > 0)
        # ★加重平均の標準誤差は非加重の sd/√n ではない。
        #   α̂ = Σwx/Σw に対し Var(α̂) = Σw²(x−α̂)² / (Σw)²。
        #   ここを取り違えると s_i が過大になり、
        #   モーメント法の σ_α² = Var(α̂) − mean(s²) が 0 に潰れて
        #   全ウォレットが μ に完全縮約される(実際に起きた)。
        g = (d.group_by("wallet").agg([
            ((pl.col(col) * pl.col("notional")).sum()
             / pl.col("notional").sum()).alias("a"),
            pl.col(col).alias("_x"), pl.col("notional").alias("_w"),
            pl.len().alias("nd"),
            pl.col("notional").sum().alias("notional"),
            pl.col("pnl_total").sum().alias("pnl_total"),
        ]).filter(pl.col("nd") >= min_days))
        g = g.with_columns(
            ((pl.col("_w").list.eval(pl.element() ** 2).list.sum() * 0).alias("_z")))
        # polars の list 演算より numpy のほうが素直なので取り出して計算する
        xs = g["_x"].to_list(); ws = g["_w"].to_list(); aa = g["a"].to_list()
        ses = []
        for x, w, am in zip(xs, ws, aa):
            x = np.asarray(x, float); w = np.asarray(w, float)
            sw = w.sum()
            ses.append(float(np.sqrt((w ** 2 * (x - am) ** 2).sum()) / sw)
                       if sw > 0 else np.nan)
        g = g.drop(["_x", "_w", "_z"]).with_columns(pl.Series("se", ses))
    else:
        g = (d.group_by("wallet").agg([
            pl.col(col).mean().alias("a"),
            pl.col(col).std().alias("sd"),
            pl.len().alias("nd"),
            pl.col("notional").sum().alias("notional"),
            pl.col("pnl_total").sum().alias("pnl_total"),
        ]).filter(pl.col("nd") >= min_days))
        g = g.with_columns((pl.col("sd") / pl.col("nd").sqrt()).alias("se"))
    st = shrink(g["a"].to_numpy(), g["se"].to_numpy())
    g = g.filter(pl.Series(st["ok"]))
    return g.with_columns([
        pl.Series("alpha_eb", st["eb"]),
        pl.Series("shrink_B", st["B"]),
        pl.Series("tau", st["tau"]),
        pl.Series("p_alpha_gt0", st["p_gt0"]),
    ]), st


# -------------------------------------------------- ウォークフォワード
def months(D: pl.DataFrame) -> list[str]:
    return sorted(D["day"].str.slice(0, 7).unique().to_list())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def walk_forward(D: pl.DataFrame, col: str, rng, min_days: int,
                 weighted: bool = False, track: set | None = None) -> pl.DataFrame:
    ms = months(D)
    rows = []
    for k in range(len(ms) - FORM_M - EVAL_M + 1):
        f0, f1 = ms[k], ms[k + FORM_M - 1]
        e0, e1 = ms[k + FORM_M], ms[k + FORM_M + EVAL_M - 1]
        mo = D.with_columns(pl.col("day").str.slice(0, 7).alias("mon"))
        F = mo.filter((pl.col("mon") >= f0) & (pl.col("mon") <= f1))
        E = mo.filter((pl.col("mon") >= e0) & (pl.col("mon") <= e1))
        if weighted:
            F = F.filter(pl.col("notional") > 0)
            E = E.filter(pl.col("notional") > 0)
            agg_f = ((pl.col(col) * pl.col("notional")).sum()
                     / pl.col("notional").sum()).alias("f")
            agg_e = ((pl.col(col) * pl.col("notional")).sum()
                     / pl.col("notional").sum()).alias("e")
        else:
            agg_f = pl.col(col).mean().alias("f")
            agg_e = pl.col(col).mean().alias("e")
        fa = (F.group_by("wallet").agg([agg_f, pl.len().alias("nf")])
              .filter(pl.col("nf") >= min_days))
        ea = (E.group_by("wallet").agg([agg_e, pl.len().alias("ne")])
              .filter(pl.col("ne") >= min_days))
        J = fa.join(ea, on="wallet", how="inner")
        if J.height < 8:
            continue
        x, y = J["f"].to_numpy(), J["e"].to_numpy()
        ic = spearman(x, y)
        # 上位四分位の残存
        qf = np.quantile(x, 0.75)
        qe = np.quantile(y, 0.75)
        top = x >= qf
        keep = (top & (y >= qe)).sum() / max(top.sum(), 1)
        # 帰無対照: 評価期の標識を並べ替え
        ics, keeps = [], []
        for _ in range(500):
            yy = rng.permutation(y)
            ics.append(spearman(x, yy))
            keeps.append((top & (yy >= np.quantile(yy, 0.75))).sum()
                         / max(top.sum(), 1))
        # ★上位 3 者が評価期で何分位にいるか(直接追跡)
        tr = {}
        if track:
            wl = J["wallet"].to_list()
            from scipy.stats import rankdata
            pe = rankdata(y) / len(y)          # 評価期の順位(0〜1)
            for w in track:
                if w in wl:
                    tr[f"pct_{w[:10]}"] = float(pe[wl.index(w)])
        rows.append({"form": f"{f0}..{f1}", "eval": f"{e0}..{e1}",
                     "n_wallet": J.height, "ic": ic, **tr,
                     "ic_null_p95": float(np.nanpercentile(ics, 95)),
                     "top_q_keep": float(keep),
                     "keep_null_mean": float(np.mean(keeps)),
                     "keep_null_p95": float(np.percentile(keeps, 95))})
    return pl.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=MIN_DAYS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260912)

    # ★検定対象の 3 者は「実現損益の上位 3」= 結果を見て選んだ集合である。
    #   だからこそ「たまたまか、再現するか」を別期間で検定する意味がある。
    L = pl.read_parquet(SRC / "mm" / "wallet_pnl.parquet")
    MKset = (L.filter((pl.col("maker_share") >= 0.7) & (pl.col("n_rows") >= 200))
             .sort("pnl_realized", descending=True))
    MKSET_ORDER_LIST = MKset["wallet"].to_list()[:3]
    TOP3 = set(MKSET_ORDER_LIST)
    MKW = set(MKset["wallet"].to_list())
    print('追跡する上位 3: ' + ', '.join(w[:12] for w in MKSET_ORDER_LIST))

    D = build_panel()
    D.write_parquet(OUT / "panel.parquet")
    print(f"パネル {D.height:,} 行 / ウォレット {D['wallet'].n_unique():,} / "
          f"日 {D['day'].n_unique():,}")

    TARGETS = [("bp", "純損益/名目 [bp]", True),
               ("pnl_total", "純損益 [$]", False),
               ("pnl_option_side", "オプションのみ [$]", False),
               ("pnl_perp", "perp のみ [$]", False)]

    summ = []
    for col, lab, wtd in TARGETS:
        Dd = demean_by_day(D, col, wtd=wtd)
        for spec, c in (("素", col), ("日固定効果あり", f"{col}_dm")):
            g, st = wallet_alpha(Dd, c, a.min_days, weighted=wtd)
            if g.height == 0:
                continue
            g = g.sort("alpha_eb", descending=True)
            g.write_parquet(OUT / f"alpha_{col}_{'dm' if spec!='素' else 'raw'}.parquet")
            npos = int((g["p_alpha_gt0"] > 0.95).sum())
            nneg = int((g["p_alpha_gt0"] < 0.05).sum())
            summ.append({"target": lab, "spec": spec, "n": g.height,
                         "sigma_alpha": st["sigma"], "mu": st["mu"],
                         "shrink_median": float(g["shrink_B"].median()),
                         "n_p95": npos, "n_p05": nneg})
            print(f"\n=== {lab} / {spec} ===")
            print(f"  ウォレット {g.height} / σ_α = {st['sigma']:.4g} / "
                  f"μ = {st['mu']:.4g} / 縮約係数 B の中央値 {g['shrink_B'].median():.3f}")
            print(f"  ★P(α>0) > 0.95 のウォレット {npos} / < 0.05 のウォレット {nneg}")
            print(f"  {'wallet':<16}{'素のalpha':>12}{'縮約後':>12}{'B':>7}"
                  f"{'P(a>0)':>9}{'日数':>6}{'総損益':>14}")
            for r in g.head(6).iter_rows(named=True):
                print(f"  {r['wallet'][:14]:<16}{r['a']:>12.4g}{r['alpha_eb']:>12.4g}"
                      f"{r['shrink_B']:>7.3f}{r['p_alpha_gt0']:>9.3f}"
                      f"{r['nd']:>6}${r['pnl_total']:>13,.0f}")
            for r in g.tail(2).iter_rows(named=True):
                print(f"  {r['wallet'][:14]:<16}{r['a']:>12.4g}{r['alpha_eb']:>12.4g}"
                      f"{r['shrink_B']:>7.3f}{r['p_alpha_gt0']:>9.3f}"
                      f"{r['nd']:>6}${r['pnl_total']:>13,.0f}")
    pl.DataFrame(summ).write_parquet(OUT / "alpha_summary.parquet")

    # ---------------- ウォークフォワード ----------------
    print("\n" + "=" * 78)
    print(f"ウォークフォワード(形成 {FORM_M} か月 → 評価 {EVAL_M} か月、1 か月ずつ前進)")
    allwf = []
    for col, lab, wtd in TARGETS:
        Dd = demean_by_day(D, col, wtd=wtd)
        for spec, c in (("素", col), ("日固定効果あり", f"{col}_dm")):
            wf = walk_forward(Dd, c, rng, a.min_days, weighted=wtd, track=TOP3)
            if wf.height == 0:
                continue
            wf = wf.with_columns([pl.lit(lab).alias("target"),
                                  pl.lit(spec).alias("spec")])
            allwf.append(wf)
            ic = wf["ic"].to_numpy()
            kp = wf["top_q_keep"].to_numpy()
            kn = wf["keep_null_mean"].to_numpy()
            print(f"\n--- {lab} / {spec} ({wf.height} 窓) ---")
            print(f"  Spearman IC: 中央 {np.nanmedian(ic):+.3f} / "
                  f"平均 {np.nanmean(ic):+.3f} / 正の窓 "
                  f"{int(np.nansum(ic > 0))}/{len(ic)}")
            print(f"    帰無の 95 分位 中央 {np.nanmedian(wf['ic_null_p95'].to_numpy()):+.3f}"
                  f" → 実測が上回った窓 "
                  f"{int(np.nansum(ic > wf['ic_null_p95'].to_numpy()))}/{len(ic)}")
            print(f"  上位四分位の残存: 実測 {np.nanmean(kp):.3f} / "
                  f"帰無 {np.nanmean(kn):.3f} / "
                  f"帰無 95 分位を超えた窓 "
                  f"{int(np.nansum(kp > wf['keep_null_p95'].to_numpy()))}/{len(kp)}")
    if allwf:
        pl.concat(allwf).write_parquet(OUT / "walkforward.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
