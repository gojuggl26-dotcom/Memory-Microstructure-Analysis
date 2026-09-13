"""十分位分析の集計 — 所有 7 銘柄 × 84 特徴量 × 9 ホライズン × mid/micro。

    uv run python scripts/analyze_decile_all.py

入力: data/decile_sum_<coin>_bbo.csv / data/decile_null_<coin>_bbo.csv
      data/featlib_stats_xyz_MU.csv(ファミリー分類。84 本すべてが載っている)
出力: data/decile_all_report.csv     銘柄 × ホライズン × 価格建ての集計
      data/decile_all_top_<coin>.csv 銘柄ごとの上位
      data/decile_all_robust.csv     銘柄をまたいで再現する (特徴量, 目的変数)
      data/decile_all_xcoin.csv      銘柄ペアごとの順位相関
      data/decile_all_cost.csv       銘柄ごとの費用と 3 段階の判定

## なぜ 84 本なのか(229 本ではなく)

229 本の `featlib` は **L1(板の全階層の再構成)** を入力にしている。L1 が
手元にあるのは MU と INTC だけで、他 5 銘柄ぶんを S3 から引くと約 99GB・
$2.6 の課金が要る(CLAUDE.md の費用ゲート対象。未承認)。
そこで **bbo(最良気配)と fills(約定)だけで作れる 84 本**に絞り、
7 銘柄を**同じ土俵**で比べる。落ちるのは第 2 階層以深に依存するファミリー
(板の形・隙間・弾力性、取消・指値フローの階層別)である。

## MU では featbbo のほうが正確

MU の共通 21 日で featlib と featbbo を突き合わせると、最良気配の**数量**由来の
28 列で r < 0.999 だった(`qa1` r=0.911、`obi1` r=0.966、`delta1_bp` r=0.524)。
featlib の第 1 階層数量は L1 から組み直した梯子、featbbo は取引所が配信する
`bid_sz`/`ask_sz` そのもの。**後者が真値**なので、全銘柄を featbbo でそろえる。

## 判定の基準(2 銘柄版と同じ)

* **Bonferroni**: 閾値は自由度 = 日数 − 1 の t 分布から取る。
* **帰無対照**: 特徴量だけ 1 営業日ずらした系列。
* **費用**: 銘柄ごとに 2 × (中央スプレッド / 2 + テイカー手数料 0.79bp)。
  2.66bp(MU)〜15.49bp(KIOXIA)と 6 倍違うので単一の値は当てない。
  判定は |D10 − D1| ではなく **片側 max(|D1|, |D10|)**(2 建玉ぶんの差を
  1 往復の費用と比べるのは甘い)。
* **銘柄横断の再現**: 1 銘柄で通っただけの規則は多重比較の残りかすでありうる。
  7 銘柄のうち何銘柄で通り、符号がそろうかを主たる判定にする。

## polars の NaN 比較

`pl.col("t") > 閾値` は NaN に true を返す。`.is_finite()` を必ず付ける。

x が確定する時刻 / y の期間: 入力の時点で担保済み(build_decile.py 参照)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
HZ = ["100ms", "300ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"]
FEE = 0.79                    # テイカー手数料(bp、片道)
FIN = pl.col("t").is_finite() & pl.col("spread_bp").is_finite()
NMIN = 5                      # 7 銘柄中いくつで通れば「再現した」とみなすか


def costs() -> dict[str, float]:
    """銘柄ごとの往復費用 = 2 × (中央スプレッド / 2 + テイカー手数料)。

    229 本版は 2 銘柄とも 2.83bp という単一の値を使っていたが、これは xyz:MU の
    半スプレッドから作った値で、xyz:KIOXIA(中央スプレッド 13.9bp)には当てはまらない。
    """
    out = {}
    for c in COINS:
        s = float(pl.scan_parquet(str(D / f"featbbo_xyz_{c}" / "dt=*.parquet"))
                  .select("spread_bp").collect()["spread_bp"].median())
        out[c] = 2 * (s / 2 + FEE)
    return out


def main() -> None:
    fam = pl.read_csv(D / "featlib_stats_xyz_MU.csv").select("feature", "family")
    COST = costs()
    rows, keep, thrs, crows = [], {}, {}, []

    for c in COINS:
        tag = f"xyz_{c}"
        S = (pl.read_csv(D / f"decile_sum_{tag}_bbo.csv")
             .join(fam, on="feature", how="left"))
        N = pl.read_csv(D / f"decile_null_{tag}_bbo.csv")
        nd = int(S["n_days"].max())
        Sv, Nv = S.filter(FIN), N.filter(FIN)
        thr = float(stats.t.ppf(1 - 0.05 / Sv.height / 2, nd - 1))
        thrs[c] = thr
        keep[c] = Sv
        nf, nfv = S["feature"].n_unique(), Sv["feature"].n_unique()
        print(f"\n===== xyz:{c} =====")
        print(f"日数 {nd} / 特徴量 {nf} 本のうち十分位が切れたのは {nfv} 本 / "
              f"検定 {Sv.height:,} 通り / Bonferroni の |t| 閾値 {thr:.2f}")
        for nm, T in (("実測", Sv), ("プラセボ(1 営業日ずらし)", Nv)):
            a = T.filter(pl.col("t").abs() > 1.96).height
            b = T.filter(pl.col("t").abs() > thr).height
            g = T.filter((pl.col("t").abs() > thr)
                         & (pl.col("spread_bp").abs() > COST[c])).height
            v = np.abs(T["spread_bp"].to_numpy())
            print(f"  {nm:<24} |t|>1.96 {a:>5} ({a/max(T.height,1)*100:4.1f}%) / "
                  f"|t|>{thr:.2f} {b:>5} ({b/max(T.height,1)*100:4.1f}%) / "
                  f"かつ |差|>{COST[c]:.2f}bp {g:>3}  |差| 中央 {np.median(v):.3f}bp "
                  f"最大 {v.max():.2f}bp")
        for h in HZ:
            for px in ("mid", "micro"):
                q = Sv.filter(pl.col("target") == f"fwd_{px}_{h}")
                p = Nv.filter(pl.col("target") == f"fwd_{px}_{h}")
                rows.append({
                    "coin": f"xyz:{c}", "horizon": h, "px": px, "n_days": nd,
                    "n_feat": q.height,
                    "n_bonf": q.filter(pl.col("t").abs() > thr).height,
                    "n_bonf_null": p.filter(pl.col("t").abs() > thr).height,
                    "n_over_cost": q.filter((pl.col("t").abs() > thr)
                                            & (pl.col("spread_bp").abs()
                                               > COST[c])).height,
                    "max_abs_spread": float(q["spread_bp"].abs().max()),
                    "p90_abs_spread": float(q["spread_bp"].abs().quantile(0.9)),
                    "thr": thr})
        top = (Sv.filter(pl.col("t").abs() > thr)
               .with_columns(ab=pl.col("spread_bp").abs()).sort("ab", descending=True))
        top.write_csv(D / f"decile_all_top_{tag}.csv")

        # ★判定は「片側」で行う。|D10 − D1| は 2 建玉ぶんの差なので、
        #   1 往復の費用と比べるのは甘い。片側に建てて畳む実費と比べる。
        sig = top.with_columns(one=pl.max_horizontal(pl.col("d1").abs(),
                                                     pl.col("d10").abs()))
        mid = sig.filter(pl.col("target").str.starts_with("fwd_mid_"))
        best = mid.sort("one", descending=True).head(1).to_dicts()
        crows.append({
            "coin": f"xyz:{c}", "cost_bp": COST[c], "n_sig": sig.height,
            "n_spread_over": sig.filter(pl.col("ab") > COST[c]).height,
            "n_oneside_over_mid": mid.filter(pl.col("one") > COST[c]).height,
            "n_oneside_over_micro": sig.filter(
                pl.col("target").str.starts_with("fwd_micro_")
                & (pl.col("one") > COST[c])).height,
            "max_oneside_bp": float(sig["one"].max()),
            "max_oneside_mid_bp": float(mid["one"].max()),
            "best_mid_feature": best[0]["feature"] if best else None,
            "best_mid_target": best[0]["target"] if best else None})
        print("\n  |D10 − D1| の上位 6(Bonferroni 通過)")
        with pl.Config(tbl_width_chars=185, float_precision=3, tbl_rows=8):
            print(top.head(6).select("feature", "family", "target", "d1", "d10",
                                     "spread_bp", "t", "monotone_rho"))

    R = pl.DataFrame(rows)
    R.write_csv(D / "decile_all_report.csv")
    print("\n===== ホライズン別 Bonferroni 通過本数(mid 建て)=====")
    with pl.Config(tbl_width_chars=150):
        print(R.filter(pl.col("px") == "mid")
              .pivot(values="n_bonf", index="horizon", on="coin")
              .join(pl.DataFrame({"horizon": HZ, "i": range(len(HZ))}),
                    on="horizon").sort("i").drop("i"))
    print(f"プラセボの通過は全銘柄合計 {R['n_bonf_null'].sum()} 本 / "
          f"銘柄ごとの費用を |D10 − D1| で超えたセルは {R['n_over_cost'].sum()} 本")

    # ---- ★費用の判定(3 段階)-------------------------------------------
    C = pl.DataFrame(crows)
    C.write_csv(D / "decile_all_cost.csv")
    print("\n===== 費用の判定 =====")
    print("  (1) 有意          |t| > Bonferroni 閾値")
    print("  (2) 差が費用超    |D10 − D1| > 往復費用     ← 甘い(2 建玉を 1 往復と比較)")
    print("  (3) 片側が費用超  max(|D1|,|D10|) > 往復費用 ← 正しい")
    with pl.Config(tbl_rows=10, tbl_width_chars=170, float_precision=2):
        print(C)
    print(f"(3) の合計: mid {C['n_oneside_over_mid'].sum()} / "
          f"micro {C['n_oneside_over_micro'].sum()}")

    # ---- 銘柄をまたいだ再現 ---------------------------------------------
    long = pl.concat([
        keep[c].select("feature", "target", "spread_bp", "t", "monotone_rho", "family")
        .with_columns(coin=pl.lit(c), sig=pl.col("t").abs() > thrs[c])
        for c in COINS])
    G = (long.group_by("feature", "target").agg(
        family=pl.col("family").first(),
        n_coin=pl.len(),
        n_sig=pl.col("sig").sum(),
        n_pos=(pl.col("spread_bp") > 0).sum(),
        med_spread=pl.col("spread_bp").median(),
        min_spread=pl.col("spread_bp").min(),
        max_spread=pl.col("spread_bp").max(),
        med_rho=pl.col("monotone_rho").median())
        .with_columns(same_sign=pl.max_horizontal(
            "n_pos", pl.col("n_coin") - pl.col("n_pos"))))
    Rb = (G.filter((pl.col("n_sig") >= NMIN)
                   & (pl.col("same_sign") == pl.col("n_coin")))
          .with_columns(ab=pl.col("med_spread").abs()).sort("ab", descending=True))
    Rb.write_csv(D / "decile_all_robust.csv")
    nmid = Rb.filter(pl.col("target").str.starts_with("fwd_mid_")).height
    print(f"\n===== {NMIN} 銘柄以上で Bonferroni 通過 かつ 全銘柄で符号一致 =====")
    print(f"該当 {Rb.height} 通り(全 {G.height} 通り中) / うち mid 建て {nmid} 通り"
          "(費用との比較は銘柄ごとに違うので §費用の判定を見ること)")
    with pl.Config(tbl_width_chars=190, float_precision=3, tbl_rows=25):
        print(Rb.head(25).select("feature", "family", "target", "n_sig",
                                 "med_spread", "min_spread", "max_spread", "med_rho"))
    print("\n  ファミリー別の内訳")
    with pl.Config(tbl_rows=15, tbl_width_chars=120):
        print(Rb.group_by("family").agg(n=pl.len(), med=pl.col("ab").median())
              .sort("n", descending=True))

    # ---- 銘柄ペアの順位相関 ----------------------------------------------
    xco = []
    for i, a in enumerate(COINS):
        for b in COINS[i + 1:]:
            j = keep[a].join(keep[b], on=["feature", "target"],
                             how="inner", suffix="_b")
            for px in ("mid", "micro"):
                for h in ["ALL"] + HZ:
                    q = (j.filter(pl.col("target").str.starts_with(f"fwd_{px}_"))
                         if h == "ALL"
                         else j.filter(pl.col("target") == f"fwd_{px}_{h}"))
                    x, y = q["spread_bp"].to_numpy(), q["spread_bp_b"].to_numpy()
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() < 30:
                        continue
                    r = float(np.corrcoef(np.argsort(np.argsort(x[m])),
                                          np.argsort(np.argsort(y[m])))[0, 1])
                    xco.append({"a": a, "b": b, "px": px, "horizon": h,
                                "n": int(m.sum()), "rank_corr": r,
                                "same_sign": float(np.mean(
                                    np.sign(x[m]) == np.sign(y[m])))})
    X = pl.DataFrame(xco)
    X.write_csv(D / "decile_all_xcoin.csv")
    print("\n===== 銘柄ペアの順位相関(mid 建て・全ホライズンまとめ)=====")
    with pl.Config(tbl_rows=25, tbl_width_chars=120, float_precision=3):
        print(X.filter((pl.col("px") == "mid") & (pl.col("horizon") == "ALL"))
              .sort("rank_corr", descending=True)
              .select("a", "b", "n", "rank_corr", "same_sign"))


if __name__ == "__main__":
    main()
