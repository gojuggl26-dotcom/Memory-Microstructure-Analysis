r"""報酬帯(特徴量 55)の検証と利用。

【何を確かめるか】
  (1) 帯は実際に効いているのか — 発注は帯の内側に寄るのか
      a. 市場をまたいだ相関(帯の幅 vs 発注距離)。**規模の交絡がある**
      b. ★同一市場内で左右の帯が違う場合の比較。市場規模を完全に統制できる
      c. 帯の縁での密度の不連続(バンチング)
  (2) 帯の内側に置くのは誰か — 支配的口座 vs その他
  (3) 帯内の注文は挙動が違うか — 約定率・寿命
  (4)(別スクリプト)信号は帯の内側の流動性から来るのか外側からか

【★前提の限界】
  帯は API の**現在エポック**(2026-08-21)の値。窓(2026-05-04〜08-10)当時の
  値は取得できない(epoch 系のパラメータは 4 通り試して全て無効)。
  よって (1) は「いまの帯が当時も同じだった」という仮定の下での検証である。
  (1b) の同一市場内比較は規模の交絡を除けるが、この仮定は除けない。
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load():
    inc = pl.read_parquet(DATA / "incentive_range.parquet").filter(pl.col("ok"))
    band = {r["market"]: (r["range_long"], r["range_short"], r["budget_long"],
                          r["budget_short"])
            for r in inc.iter_rows(named=True)}
    rows = []
    for f in sorted(glob.glob(str(DATA / "band_place_*.parquet"))):
        m = int(re.search(r"band_place_(\d+)", f).group(1))
        if m not in band or band[m][0] is None:
            continue
        P = pl.read_parquet(f)
        E = DATA / f"event_book_{m}.parquet"
        sp = np.nan
        if E.exists():
            e = pl.read_parquet(E, columns=["spread_pp", "mid_pp", "extrapolated"]
                                ).filter(~pl.col("extrapolated"))
            sp = float(e["spread_pp"].median()) if e.height else np.nan
        for sd, key in ((0, 0), (1, 1)):
            v = P.filter(pl.col("side") == sd)
            d = v["dist_pp"].to_numpy()
            d = d[np.isfinite(d) & (d > 0)]
            if len(d) < 200:
                continue
            R = band[m][key] * 100
            rows.append({
                "market": m, "side": "long" if sd == 0 else "short",
                "band_pp": R, "n": len(d),
                "med_dist": float(np.median(d)),
                "p75_dist": float(np.percentile(d, 75)),
                "in_band_frac": float(np.mean(d <= R)),
                "med_spread": sp,
                "budget": band[m][2 + key],
            })
    return pl.DataFrame(rows), pl.read_parquet(DATA / "incentive_range.parquet")


def partial(x, y, z):
    r = stats.rankdata
    x, y, z = r(x), r(y), r(z)
    Z = np.column_stack([np.ones(len(z)), z])
    ex = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ey = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rho = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = rho * np.sqrt((n - 3) / max(1 - rho ** 2, 1e-12))
    return rho, 2 * (1 - stats.t.cdf(abs(t), n - 3))


def main() -> int:
    D, inc = load()
    print(f"市場×側 {D.height} 件 / 市場 {D['market'].n_unique()}\n")

    # --- (1a) 市場をまたいだ相関 ---
    b = D["band_pp"].to_numpy(); d = D["med_dist"].to_numpy()
    s = D["med_spread"].to_numpy()
    m = np.isfinite(b) & np.isfinite(d) & np.isfinite(s)
    r0, p0 = stats.spearmanr(b[m], d[m])
    r1, p1 = partial(b[m], d[m], s[m])
    print("=== (1a) 市場をまたいだ相関 — 帯の幅 vs 発注距離 ===")
    print(f"  素の Spearman            ρ={r0:+.3f} (p={p0:.2e}, n={m.sum()})")
    print(f"  spread を統制した偏相関   ρ={r1:+.3f} (p={p1:.2e})")
    r2, p2 = stats.spearmanr(s[m], d[m])
    print(f"  参考: spread vs 発注距離  ρ={r2:+.3f} (p={p2:.2e})")
    r3, p3 = stats.spearmanr(b[m], s[m])
    print(f"  参考: 帯の幅 vs spread    ρ={r3:+.3f} (p={p3:.2e})  ← ここが交絡")

    # --- (1b) ★同一市場内で左右を比べる ---
    print("\n=== (1b) ★同一市場内の左右比較(市場規模を完全に統制)===")
    piv = (D.pivot(values=["band_pp", "med_dist", "in_band_frac"],
                   index="market", on="side")
           .drop_nulls())
    rl = piv["band_pp_long"].to_numpy(); rs = piv["band_pp_short"].to_numpy()
    dl = piv["med_dist_long"].to_numpy(); ds = piv["med_dist_short"].to_numpy()
    for thr in (1.0, 1.2, 1.5, 2.0):
        sel = (np.maximum(rl, rs) / np.minimum(rl, rs)) >= thr
        if sel.sum() < 10:
            continue
        # 帯が広い側のほうが発注距離も大きいか
        wider_long = rl[sel] > rs[sel]
        farther_long = dl[sel] > ds[sel]
        agree = int((wider_long == farther_long).sum())
        n = int(sel.sum())
        pv = stats.binomtest(agree, n, 0.5).pvalue
        print(f"  帯の比 ≥{thr:.1f} の市場 {n:>3}: 一致 {agree:>3}/{n:<3} = {agree/n:.1%}  "
              f"二項検定 p={pv:.4f}")
    print("  (一致 = 帯が広い側のほうが発注距離も大きい。帯が効いているなら 50% を超えるはず)")

    # --- (1c) 帯の内側に何割いるか ---
    print("\n=== (1c) 帯の内側にある発注の割合 ===")
    f = D["in_band_frac"].to_numpy()
    print(f"  中央 {np.median(f):.1%}  p10 {np.percentile(f,10):.1%}  p90 {np.percentile(f,90):.1%}")
    print(f"  予算 >0 の市場だけ: ", end="")
    bd = D.filter(pl.col("budget") > 0)
    if bd.height:
        fb = bd["in_band_frac"].to_numpy()
        print(f"{np.median(fb):.1%}(n={bd.height})  予算 0: "
              f"{np.median(D.filter(pl.col('budget')==0)['in_band_frac'].to_numpy()):.1%}")
        u = stats.mannwhitneyu(fb, D.filter(pl.col("budget") == 0)["in_band_frac"].to_numpy())
        print(f"  Mann-Whitney p={u.pvalue:.4f}  ← 予算が付いている市場ほど帯内に寄るか")
    else:
        print("(該当なし)")

    # --- (2)(3) 誰が帯内に置くか / 帯内の注文は違うか ---
    print("\n=== (2)(3) 帯内の発注は誰が置き、どうなるか ===")
    rows2 = []
    for f_ in sorted(glob.glob(str(DATA / "band_place_*.parquet"))):
        mid = int(re.search(r"band_place_(\d+)", f_).group(1))
        P = pl.read_parquet(f_)
        if P.height < 500:
            continue
        ib = P["in_band"].to_numpy()
        ok = np.array([x is not None for x in ib])
        ibb = np.array([bool(x) if x is not None else False for x in ib])
        t1 = P["is_top1"].to_numpy()
        term = P["terminal"].to_numpy()
        lt = P["lifetime_s"].to_numpy()
        rec = {"market": mid}
        good = True
        for tag, sel in (("in", ok & ibb), ("out", ok & ~ibb)):
            if sel.sum() < 50:
                good = False
                break
            rec[f"n_{tag}"] = int(sel.sum())
            rec[f"top1_{tag}"] = float(t1[sel].mean())
            rec[f"fill_{tag}"] = float((term[sel] == "filled").mean())
            rec[f"life_{tag}"] = float(np.nanmedian(lt[sel]))
        if good:
            rows2.append(rec)
    A = pl.DataFrame(rows2)
    print(f"  両群が揃った市場: {A.height}")
    print(f"{'量':<16}{'帯内(中央)':>14}{'帯外(中央)':>14}{'帯内>帯外の市場':>18}{'符号検定 p':>13}")
    for base, lab, fmt in (("top1", "上位1者の割合", "{:.1%}"),
                           ("fill", "約定率", "{:.2%}"),
                           ("life", "寿命中央[s]", "{:.0f}")):
        a = A[f"{base}_in"].to_numpy(); b = A[f"{base}_out"].to_numpy()
        m2 = np.isfinite(a) & np.isfinite(b)
        a, b = a[m2], b[m2]
        n = len(a); k = int((a > b).sum())
        pv = stats.binomtest(k, n, 0.5).pvalue if n else np.nan
        print(f"{lab:<16}{fmt.format(np.median(a)):>14}{fmt.format(np.median(b)):>14}"
              f"{f'{k}/{n}':>18}{pv:>13.2e}")
    print("\n  ★README が交絡として挙げてきたのは「報酬帯の内側は報酬目的の指値で埋まる」。")
    print("    上位1者の割合・約定率・寿命が帯の内外で違えば、その懸念は実在する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
