"""十分位分析の集計 — 2 銘柄 × 229 特徴量 × 9 ホライズン × mid/micro。

    uv run python scripts/analyze_decile.py

入力: data/decile_sum_<coin>.csv / data/decile_null_<coin>.csv /
      data/featlib_stats_<coin>.csv(分類)
出力: data/decile_report.csv     銘柄 × ホライズン × 価格建ての集計
      data/decile_top_<coin>.csv 上位の特徴量(Bonferroni 通過・|差| 順)
      data/decile_xcoin.csv      2 銘柄の一致

## ★polars の NaN 比較

`pl.col("t") > 閾値` は **NaN に対して true を返す**(NaN を最大値として扱う仕様)。
十分位が潰れた特徴量の `t` は NaN なので、`.is_finite()` を付けないと
「プラセボでも 25% が Bonferroni を通る」という偽の結果になる。実際に一度踏んだ。

## 判定の基準

* **Bonferroni**: 検定は(有効な特徴量 × 18 目的変数)通り。閾値は
  自由度 = 日数 − 1 の t 分布から取る(正規近似では甘い)。
* **帰無対照**: 特徴量だけ 1 営業日ずらした系列。白色雑音を同じ配管に通すと
  |t|>1.96 が 5.8%・Bonferroni 通過 0 本になることを確認済み(配管の検算)。
* **費用**: 往復 2.83bp(半スプレッド 0.62 + テイカー手数料 0.79、×2)。
  「有意」と「費用を引いて残る」を必ず分けて数える。

x が確定する時刻 / y の期間: 入力の時点で担保済み(build_decile.py 参照)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
COINS = {"xyz:MU": "xyz_MU", "xyz:INTC": "xyz_INTC"}
HZ = ["100ms", "300ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"]
COST = 2.83
FIN = pl.col("t").is_finite() & pl.col("spread_bp").is_finite()


def load(tag: str):
    s = pl.read_csv(D / f"decile_sum_{tag}.csv")
    n = pl.read_csv(D / f"decile_null_{tag}.csv")
    fam = (pl.read_csv(D / f"featlib_stats_{tag}.csv").select("feature", "family")
           if (D / f"featlib_stats_{tag}.csv").exists() else None)
    if fam is not None:
        s = s.join(fam, on="feature", how="left")
    return s, n


def main() -> None:
    rows, tops, xco = [], {}, []
    for coin, tag in COINS.items():
        S, N = load(tag)
        nd = int(S["n_days"].max())
        Sv, Nv = S.filter(FIN), N.filter(FIN)
        ncell = Sv.height
        thr = float(stats.t.ppf(1 - 0.05 / ncell / 2, nd - 1))
        deg = S.height - ncell
        nf = S["feature"].n_unique()
        nfv = Sv["feature"].n_unique()
        print(f"\n===== {coin} =====")
        print(f"特徴量 {nf} 本のうち十分位が切れたのは {nfv} 本"
              f"(残り {nf-nfv} 本は前日の分位が潰れて切れない)")
        print(f"検定 {ncell:,} 通り / 日数 {nd} / Bonferroni の |t| 閾値 {thr:.2f}")
        for nm, T in (("実測", Sv), ("プラセボ(1 営業日ずらし)", Nv)):
            a = T.filter(pl.col("t").abs() > 1.96).height
            b = T.filter(pl.col("t").abs() > thr).height
            c = T.filter((pl.col("t").abs() > thr)
                         & (pl.col("spread_bp").abs() > COST)).height
            v = np.abs(T["spread_bp"].to_numpy())
            print(f"  {nm:<24} |t|>1.96 {a:>5} ({a/ncell*100:4.1f}%) / "
                  f"|t|>{thr:.2f} {b:>5} ({b/ncell*100:4.1f}%) / "
                  f"かつ |差|>{COST}bp {c:>3}  |差| 中央 {np.median(v):.3f}bp "
                  f"最大 {v.max():.2f}bp")
        for h in HZ:
            for px in ("mid", "micro"):
                q = Sv.filter(pl.col("target") == f"fwd_{px}_{h}")
                p = Nv.filter(pl.col("target") == f"fwd_{px}_{h}")
                rows.append({
                    "coin": coin, "horizon": h, "px": px, "n_feat": q.height,
                    "n_bonf": q.filter(pl.col("t").abs() > thr).height,
                    "n_bonf_null": p.filter(pl.col("t").abs() > thr).height,
                    "n_over_cost": q.filter((pl.col("t").abs() > thr)
                                            & (pl.col("spread_bp").abs() > COST)).height,
                    "max_abs_spread": float(q["spread_bp"].abs().max()),
                    "p90_abs_spread": float(q["spread_bp"].abs().quantile(0.9)),
                    "thr": thr})
        top = (Sv.filter(pl.col("t").abs() > thr)
               .with_columns(ab=pl.col("spread_bp").abs())
               .sort("ab", descending=True))
        top.write_csv(D / f"decile_top_{tag}.csv")
        tops[coin] = Sv
        print("\n  |D10 − D1| の上位 8(Bonferroni 通過)")
        with pl.Config(tbl_width_chars=185, float_precision=3, tbl_rows=10):
            print(top.head(8).select("feature", "family", "target", "d1", "d10",
                                     "spread_bp", "t", "monotone_rho"))

    R = pl.DataFrame(rows)
    R.write_csv(D / "decile_report.csv")
    print("\n===== ホライズン別(Bonferroni 通過 / 有効本数)=====")
    piv = R.with_columns(k=pl.col("coin") + " " + pl.col("px")) \
           .pivot(values="n_bonf", index="horizon", on="k")
    with pl.Config(tbl_width_chars=150):
        print(piv.join(pl.DataFrame({"horizon": HZ, "i": range(len(HZ))}),
                       on="horizon").sort("i").drop("i"))
    print("プラセボはどのホライズンでも通過 "
          f"{R['n_bonf_null'].sum()} 本 / 費用 {COST}bp を超えたセルは "
          f"{R['n_over_cost'].sum()} 本")

    # ---- 2 銘柄の一致 ----------------------------------------------------
    a, b = tops["xyz:MU"], tops["xyz:INTC"]
    j = a.join(b, on=["feature", "target"], how="inner", suffix="_intc")
    print("\n===== 2 銘柄で同じ向きか(共通の特徴量 × 目的変数)=====")
    for h in HZ:
        for px in ("mid", "micro"):
            q = j.filter(pl.col("target") == f"fwd_{px}_{h}")
            x, y = q["spread_bp"].to_numpy(), q["spread_bp_intc"].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 30:
                continue
            r = float(np.corrcoef(np.argsort(np.argsort(x[m])),
                                  np.argsort(np.argsort(y[m])))[0, 1])
            sg = float(np.mean(np.sign(x[m]) == np.sign(y[m])))
            xco.append({"horizon": h, "px": px, "n": int(m.sum()),
                        "rank_corr": r, "same_sign": sg})
    X = pl.DataFrame(xco)
    X.write_csv(D / "decile_xcoin.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=120, float_precision=3):
        print(X.pivot(values=["rank_corr", "same_sign"], index="horizon", on="px")
              .join(pl.DataFrame({"horizon": HZ, "i": range(len(HZ))}),
                    on="horizon").sort("i").drop("i"))


if __name__ == "__main__":
    main()
