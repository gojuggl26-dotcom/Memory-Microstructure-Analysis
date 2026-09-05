"""板の出どころを l2/bbo から「l1 の組み直し」へ替えると、特徴量はどれだけ変わるか。

    uv run python scripts/build_featlib.py --coin xyz:MU --bbo-suffix _l1 \
        --out-suffix _l1 --days 10
    uv run python scripts/featlib_bbo_effect.py --coin xyz:MU

出力: data/featlib_bbo_effect_<tag>.csv と charts/<tag>_featlib_bbo_effect.png

なぜ要るか
----------
xyz:INTC は `l2/bbo` が DEEP_ARCHIVE で読めないため、板を `l1` から組み直して
特徴量を作る。組み直した**最良気配**が実物とどれだけ一致するかは
`validate_bbo_l1.py` で測ったが、本当に知りたいのは
**「その差が特徴量にどれだけ乗るか」**である。ここでは xyz:MU で
両方の板から同じ特徴量を作り、1 本ずつ突き合わせる。

見る量は 3 つ。

  1. 2 版の**相関**(同じ格子点どうし)…… 形が保たれているか
  2. 中央値の**相対差**              …… 水準がずれていないか
  3. 目的変数との**相関の差**        …… 結論が変わるか(これが本命)

x が確定する時刻 / y の期間: 該当なし(2 つの作り方の突合)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, D, plt, save  # noqa: E402

TGT = "fwd_micro_10s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    A = D / f"featlib_{tag}"
    B = D / f"featlib_{tag}_l1"
    days = sorted({p.stem.split("=")[1] for p in A.glob("dt=*.parquet")}
                  & {p.stem.split("=")[1] for p in B.glob("dt=*.parquet")})
    if not days:
        sys.exit("共通日が無い。先に --out-suffix _l1 で build_featlib を回すこと")
    print(f"共通 {len(days)} 日 ({days[0]} 〜 {days[-1]})")

    cols = None
    acc = {}
    for d in days:
        x = pl.read_parquet(A / f"dt={d}.parquet")
        y = pl.read_parquet(B / f"dt={d}.parquet")
        if cols is None:
            cols = [c for c in x.columns if c in y.columns
                    and c not in ("dt", "sec") and not c.startswith("fwd_")]
            acc = {c: {"r": [], "rel": [], "ra": [], "rb": []} for c in cols}
        ya = x[TGT].to_numpy().astype(np.float64)
        yb = y[TGT].to_numpy().astype(np.float64)
        for c in cols:
            u = x[c].to_numpy().astype(np.float64)
            v = y[c].to_numpy().astype(np.float64)
            m = np.isfinite(u) & np.isfinite(v)
            if m.sum() < 500:
                continue
            su, sv = u[m].std(), v[m].std()
            if su > 1e-15 and sv > 1e-15:
                acc[c]["r"].append(float(np.mean((u[m] - u[m].mean())
                                                 * (v[m] - v[m].mean())) / (su * sv)))
            mu_, mv_ = np.median(np.abs(u[m])), np.median(np.abs(v[m]))
            if mu_ > 1e-12:
                acc[c]["rel"].append(float((mv_ - mu_) / mu_))
            for k, (z, yy) in (("ra", (u, ya)), ("rb", (v, yb))):
                mm = np.isfinite(z) & np.isfinite(yy)
                if mm.sum() > 500:
                    sz, sy = z[mm].std(), yy[mm].std()
                    if sz > 1e-15 and sy > 1e-15:
                        acc[c][k].append(float(np.mean((z[mm] - z[mm].mean())
                                                       * (yy[mm] - yy[mm].mean()))
                                               / (sz * sy)))

    rows = []
    for c in cols:
        g = acc[c]
        rows.append({"feature": c,
                     "corr_two_versions": float(np.mean(g["r"])) if g["r"] else np.nan,
                     "rel_level": float(np.mean(g["rel"])) if g["rel"] else np.nan,
                     "r_real": float(np.mean(g["ra"])) if g["ra"] else np.nan,
                     "r_l1": float(np.mean(g["rb"])) if g["rb"] else np.nan})
    t = pl.DataFrame(rows).with_columns(
        dr=(pl.col("r_l1") - pl.col("r_real")))
    t.write_csv(D / f"featlib_bbo_effect_{tag}.csv")

    ok = t.filter(pl.col("corr_two_versions").is_finite())
    print(f"\n2 版の相関: 中央 {ok['corr_two_versions'].median():.4f} / "
          f"下位 5% {ok['corr_two_versions'].quantile(0.05):.4f} / "
          f"0.9 未満 {ok.filter(pl.col('corr_two_versions') < 0.9).height} 本")
    with pl.Config(tbl_rows=16, fmt_str_lengths=26):
        print("\n一致が悪い順:")
        print(ok.sort("corr_two_versions").head(14).select(
            "feature", "corr_two_versions", "rel_level", "r_real", "r_l1"))
    k = t.filter(pl.col("r_real").is_finite() & pl.col("r_l1").is_finite())
    print(f"\n目的変数との相関: 順位相関 "
          f"{np.corrcoef(k['r_real'], k['r_l1'])[0,1]:.4f} / "
          f"|差| 中央 {k['dr'].abs().median():.5f} / 最大 {k['dr'].abs().max():.5f}")
    print("差の大きい順:")
    with pl.Config(tbl_rows=10, fmt_str_lengths=26):
        print(k.sort(pl.col("dr").abs(), descending=True).head(8).select(
            "feature", "r_real", "r_l1", "dr"))

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.1))
    a0 = ax[0]
    v = ok["corr_two_versions"].to_numpy()
    a0.hist(v, bins=np.linspace(min(0.0, float(v.min())), 1.0, 60), color=C1)
    a0.set_xlabel("同じ格子点で見た 2 版の相関")
    a0.set_ylabel("特徴量の本数")
    a0.set_title(f"A. 形はどれだけ保たれるか(中央 {np.median(v):.3f})")

    a1 = ax[1]
    a1.scatter(k["r_real"], k["r_l1"], s=13, color=C3, linewidths=0)
    lim = float(np.nanmax(np.abs(np.concatenate([k["r_real"].to_numpy(),
                                                 k["r_l1"].to_numpy()])))) * 1.12
    a1.plot([-lim, lim], [-lim, lim], color=CM, lw=0.8, ls="--")
    a1.axhline(0, color=CM, lw=0.6)
    a1.axvline(0, color=CM, lw=0.6)
    a1.set_xlabel("l2/bbo を使ったときの r")
    a1.set_ylabel("l1 の組み直しを使ったときの r")
    a1.set_title(f"B. 結論は変わるか({TGT})")

    a2 = ax[2]
    s = ok.sort("corr_two_versions").head(18)
    yy = np.arange(s.height)
    a2.barh(yy, s["corr_two_versions"], color=C2, height=0.66)
    a2.set_yticks(yy)
    a2.set_yticklabels(s["feature"], fontsize=6.5)
    a2.invert_yaxis()
    a2.set_xlabel("2 版の相関")
    a2.set_title("C. 一致が悪い 18 本")
    save(fig, f"{tag}_featlib_bbo_effect.png",
         f"{a.coin}: 板の出どころを l2/bbo → l1 の組み直しに替えたときの特徴量の変化"
         f"({len(days)} 日)")


if __name__ == "__main__":
    main()
