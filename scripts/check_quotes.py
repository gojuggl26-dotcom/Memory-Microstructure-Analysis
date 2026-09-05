"""仮想発注候補テーブル(build_quotes.py の出力)の全期間検査。

出す数字はすべて分子と分母を明示する。日次の推移と最悪日を必ず添える
(中央値だけ見て 1 日の破綻を見逃す事故を避けるため)。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

BULK = Path("E:/Memory-quotes")
OUT = Path("data")
FEE_BP = 0.088


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    print(f"{len(files)} 日", flush=True)

    lf = pl.scan_parquet([str(f) for f in files])
    cols = lf.collect_schema().names()
    expl = [c for c in cols if not c.startswith("label_")
            and c not in ("dt", "ts", "sec", "side")]
    labs = [c for c in cols if c.startswith("label_")]

    # ---- 日次 ----
    day = (lf.group_by("dt").agg(
        n=pl.len(),
        n_cand=pl.col("ts").n_unique(),
        fill=pl.col("label_filled_60s").mean(),
        lat_med=pl.col("label_fill_lat_s").median(),
        edge_med=pl.col("label_edge_bp").median(),
        spr_med=pl.col("spread_bp").median(),
        mk1_med=pl.col("label_markout_1s_bp").median(),
        pnl1_mean=pl.col("label_pnl_1s_bp").mean(),
        ts_min=pl.col("ts").min(), ts_max=pl.col("ts").max(),
    ).sort("dt").collect())
    day.write_parquet(OUT / f"quotes_day_{tag}.parquet")

    N = int(day["n"].sum())
    print(f"\n総行数 {N:,} 行 = 候補時点 {int(day['n_cand'].sum()):,} × 2 面")
    print(f"1 日あたり 行数 中央値 {day['n'].median():,.0f} / "
          f"最小 {day['n'].min():,} ({day.filter(pl.col('n')==day['n'].min())['dt'][0]}) / "
          f"最大 {day['n'].max():,} ({day.filter(pl.col('n')==day['n'].max())['dt'][0]})")

    # ---- 欠損 ----
    nul = lf.select([pl.col(c).is_null().mean().alias(c) for c in cols]).collect()
    miss = {c: float(nul[c][0]) for c in cols}
    nn = lf.select([pl.col(c).is_nan().sum().alias(c) for c in cols
                    if lf.collect_schema()[c] in (pl.Float32, pl.Float64)]).collect()
    assert all(int(nn[c][0] or 0) == 0 for c in nn.columns), "NaN が残っている"""
    print("\n欠損率 上位(分母は全 {:,} 行)".format(N))
    for c, v in sorted(miss.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {c:24s} {100*v:6.2f}%")

    # ---- 説明変数の分布 ----
    qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    st = lf.select(
        [pl.col(c).mean().alias(f"{c}|mean") for c in expl]
        + [pl.col(c).std().alias(f"{c}|std") for c in expl]
        + [pl.col(c).quantile(q).alias(f"{c}|q{q}") for c in expl for q in qs]
        + [pl.col(c).min().alias(f"{c}|min") for c in expl]
        + [pl.col(c).max().alias(f"{c}|max") for c in expl]).collect()
    def val(x):
        # ★ `x or np.nan` は 0.0 が偽値なので 0 を欠損に化けさせる。使わない。
        return float("nan") if x is None else float(x)

    rows = []
    for c in expl:
        r = {"col": c, "miss": miss[c]}
        r["mean"] = val(st[f"{c}|mean"][0])
        r["std"] = val(st[f"{c}|std"][0])
        for q in qs:
            r[f"q{q}"] = val(st[f"{c}|q{q}"][0])
        r["min"] = val(st[f"{c}|min"][0])
        r["max"] = val(st[f"{c}|max"][0])
        rows.append(r)
    S = pl.DataFrame(rows)
    S.write_parquet(OUT / f"quotes_stats_{tag}.parquet")

    # ---- ラベル ----
    L = lf.select(
        [pl.col(c).mean().alias(f"{c}|m") for c in labs]
        + [pl.col(c).median().alias(f"{c}|md") for c in labs]
        + [pl.col(c).std().alias(f"{c}|s") for c in labs]).collect()
    print("\nラベル(分母は非欠損行)")
    for c in labs:
        print(f"  {c:24s} 平均 {float(L[f'{c}|m'][0]):+9.4f} "
              f"中央 {float(L[f'{c}|md'][0]):+9.4f} 標準偏差 {float(L[f'{c}|s'][0]):9.4f} "
              f"欠損 {100*miss[c]:5.2f}%")

    # ---- 整合性の検査 ----
    print("\n整合性")
    chk = lf.select(
        neg_age=(pl.col("bbo_age_s") < 0).sum(),
        neg_spr=(pl.col("spread_abs") <= 0).sum(),
        lat_bad=((pl.col("label_fill_lat_s") <= 0)
                 | (pl.col("label_fill_lat_s") > 60)).sum(),
        lat_orphan=(pl.col("label_fill_lat_s").is_not_null()
                    & (pl.col("label_filled_60s") == 0)).sum(),
        fill_orphan=((pl.col("label_filled_60s") == 1)
                     & pl.col("label_fill_lat_s").is_null()).sum(),
        obi_out=((pl.col("obi1").abs() > 1.0000001)).sum(),
        q_neg=(pl.col("queue_ahead_qty") < 0).sum(),
        pnl_bad=((pl.col("label_pnl_1s_bp")
                  - (pl.col("label_edge_bp") - FEE_BP
                     + pl.col("label_markout_1s_bp"))).abs() > 1e-2).sum(),
    ).collect()
    for c in chk.columns:
        v = int(chk[c][0])
        print(f"  {c:14s} {v:,} 件 ({100*v/N:.4f}%)  {'OK' if v == 0 else '★要確認'}")

    # ---- 最悪日 ----
    print("\n日次の振れ(最悪日を明示)")
    for c, f in (("fill", "{:.3f}"), ("spr_med", "{:.3f}"),
                 ("edge_med", "{:+.3f}"), ("pnl1_mean", "{:+.4f}")):
        v = day[c].to_numpy().astype(float)
        if not np.isfinite(v).any():
            print(f"  {c:10s} 全日 欠損"); continue
        i0, i1 = int(np.nanargmin(v)), int(np.nanargmax(v))
        print(f"  {c:10s} 中央 {f.format(float(np.nanmedian(v)))}  "
              f"最小 {f.format(v[i0])} ({day['dt'][i0]})  "
              f"最大 {f.format(v[i1])} ({day['dt'][i1]})")

    # ---- 前半後半 ----
    h = len(day) // 2
    print("\n前半 49 日 / 後半 49 日")
    for c in ("n", "fill", "spr_med", "edge_med", "pnl1_mean"):
        v = day[c].to_numpy().astype(float)
        print(f"  {c:10s} {np.nanmean(v[:h]):12.4f} -> {np.nanmean(v[h:]):12.4f}")

    json.dump({"days": len(files), "rows": N,
               "checks": {c: int(chk[c][0]) for c in chk.columns},
               "miss": miss},
              open(OUT / f"quotes_check_{tag}.json", "w"), indent=1)
    print(f"\n書き出し {OUT}/quotes_day_{tag}.parquet, quotes_stats_{tag}.parquet")


if __name__ == "__main__":
    main()
