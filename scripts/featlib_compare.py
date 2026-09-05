"""2 銘柄の特徴量ライブラリを、**共通の日**だけで並べて比べる。

    uv run python scripts/featlib_compare.py --a xyz:INTC --b xyz:MU

出力: data/featlib_cmp_<A>_<B>.csv と charts/<A>_vs_<B>_featlib.png

なぜ共通の日に限るか
--------------------
標本期間が違うまま「INTC は 99 日 / MU は 21 日」で比べると、
市場全体の地合いの違いを銘柄の違いと取り違える。両方がそろっている日だけに
そろえる。**日の選び方は結果を見る前に「共通日すべて」と決めている**
(生存バイアスを作らない)。

比べる特徴量は 22 本を**あらかじめ**決めてある(下の `KEY`)。
結果を見てから選び直すことはしない。

時間契約: `x` は格子点 `T` まで、`y` は `(T, T+10s]`。日単位でクラスタした t。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

KEY = ["spread_bp", "spread_tick", "depth_bbo", "qb1", "qa1", "depth_cum10",
       "obi1", "obi10", "delta1_bp", "micro_slope", "ofi_1s", "ofi_z",
       "cancel_rate", "add_intensity", "cancel_intensity", "trade_imb",
       "signed_vol", "aggr_mom", "liq_conc", "gap12_b", "elast_b",
       "t_since_mid_move"]
TGT = "fwd_micro_10s"


def scan(tag: str, days: list[str]):
    """共通日だけを読み、分位と日ごとの相関を返す。"""
    pct = {k: [] for k in KEY}
    rr = {k: [] for k in KEY}
    ev = []
    for d in days:
        fp = D / f"featlib_{tag}" / f"dt={d}.parquet"
        cols = list(dict.fromkeys(KEY + [TGT, "mid"]))   # ★重複を落とす
        t = pl.read_parquet(fp, columns=cols)
        y = t[TGT].to_numpy().astype(np.float64)
        fy = np.isfinite(y)
        ev.append({"dt": d, "mid": float(np.nanmedian(t["mid"].to_numpy())),
                   "spread_bp": float(np.nanmedian(t["spread_bp"].to_numpy()))})
        for k in KEY:
            x = t[k].to_numpy().astype(np.float64)
            fx = np.isfinite(x)
            if fx.sum() > 100:
                pct[k].append(np.percentile(x[fx], [10, 25, 50, 75, 90]))
            m = fx & fy
            if m.sum() > 500:
                xx, yy = x[m], y[m]
                sx, sy = xx.std(), yy.std()
                if sx > 1e-15 and sy > 1e-15:
                    rr[k].append(float(np.mean((xx - xx.mean()) * (yy - yy.mean()))
                                       / (sx * sy)))
    return pct, rr, pl.DataFrame(ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    ta, tb = args.a.replace(":", "_"), args.b.replace(":", "_")
    da = {p.stem.split("=")[1] for p in (D / f"featlib_{ta}").glob("dt=*.parquet")}
    db = {p.stem.split("=")[1] for p in (D / f"featlib_{tb}").glob("dt=*.parquet")}
    days = sorted(da & db)
    print(f"共通 {len(days)} 日 ({days[0]} 〜 {days[-1]}) / "
          f"{args.a} {len(da)} 日, {args.b} {len(db)} 日")
    pa, ra, ea = scan(ta, days)
    pb, rb, eb = scan(tb, days)

    rows = []
    for k in KEY:
        A = np.array(pa[k]) if pa[k] else np.full((1, 5), np.nan)
        B = np.array(pb[k]) if pb[k] else np.full((1, 5), np.nan)
        va, vb = np.array(ra[k]), np.array(rb[k])

        def ct(v):
            v = v[np.isfinite(v)]
            if v.size < 3:
                return np.nan, np.nan
            return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(v.size))
        ma, tA = ct(va)
        mb, tB = ct(vb)
        rows.append({"feature": k,
                     "a_p10": np.median(A[:, 0]), "a_p50": np.median(A[:, 2]),
                     "a_p90": np.median(A[:, 4]),
                     "b_p10": np.median(B[:, 0]), "b_p50": np.median(B[:, 2]),
                     "b_p90": np.median(B[:, 4]),
                     "a_r": ma, "a_t": tA, "b_r": mb, "b_t": tB})
    t = pl.DataFrame(rows)
    t.write_csv(D / f"featlib_cmp_{ta}_{tb}.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=30):
        print(t.select("feature", "a_p50", "b_p50", "a_r", "a_t", "b_r", "b_t"))
    print(f"\n{args.a}: mid 中央 {ea['mid'].median():.3f}  "
          f"spread 中央 {ea['spread_bp'].median():.3f}bp")
    print(f"{args.b}: mid 中央 {eb['mid'].median():.3f}  "
          f"spread 中央 {eb['spread_bp'].median():.3f}bp")

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.6))
    y = np.arange(len(KEY))
    a0 = ax[0]
    sa = np.where(np.abs(t["a_p50"]) > 1e-12, t["a_p50"], np.nan)
    sb = np.where(np.abs(t["b_p50"]) > 1e-12, t["b_p50"], np.nan)
    a0.barh(y, np.log10(np.abs(np.asarray(sb, float)) + 1e-9)
            - np.log10(np.abs(np.asarray(sa, float)) + 1e-9),
            color=CM, height=0.6)
    a0.axvline(0, color=C2, lw=1.0)
    a0.set_yticks(y)
    a0.set_yticklabels(KEY, fontsize=7)
    a0.invert_yaxis()
    a0.set_xlabel(f"log10( |{args.b} の中央値| / |{args.a} の中央値| )")
    a0.set_title(f"A. 水準の違い(右へ行くほど {args.b} が大きい)")

    a1 = ax[1]
    a1.barh(y, t["a_r"], color=C1, height=0.4, label=args.a)
    a1.barh(y + 0.42, t["b_r"], color=C2, height=0.4, label=args.b)
    a1.axvline(0, color=CM, lw=0.8)
    a1.set_yticks(y + 0.21)
    a1.set_yticklabels(KEY, fontsize=7)
    a1.invert_yaxis()
    a1.set_xlabel(f"{TGT} との相関")
    a1.set_title("B. 予測相関(共通日・符号つき)")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[2]
    a2.scatter(t["b_r"], t["a_r"], s=22, color=C1, linewidths=0)
    lim = float(np.nanmax(np.abs(np.concatenate([t["a_r"].to_numpy(),
                                                 t["b_r"].to_numpy()])))) * 1.15
    a2.plot([-lim, lim], [-lim, lim], color=CM, lw=0.8, ls="--")
    a2.axhline(0, color=CM, lw=0.6)
    a2.axvline(0, color=CM, lw=0.6)
    for i, k in enumerate(KEY):
        if abs(t["a_r"][i] or 0) > 0.03 or abs(t["b_r"][i] or 0) > 0.03:
            a2.annotate(k, (t["b_r"][i], t["a_r"][i]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    a2.set_xlabel(f"{args.b} の r")
    a2.set_ylabel(f"{args.a} の r")
    a2.set_title("C. 同じ特徴量が両銘柄で同じ向きか")
    save(fig, f"{ta}_vs_{tb}_featlib.png",
         f"{args.a} と {args.b} の特徴量比較(共通 {len(days)} 日)")


if __name__ == "__main__":
    main()
