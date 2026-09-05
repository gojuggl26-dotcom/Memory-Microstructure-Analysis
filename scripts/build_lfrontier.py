"""遅延フロンティア(実測できる範囲)をまとめる。

    uv run python scripts/build_lfrontier.py --coin xyz:MU

3 つを 1 つの表にする。

 1. 戦略の EV を遅延の関数として(在庫シミュレータを遅延ごとに回した結果)
 2. シグナルの減衰(`build_decay.py`)
 3. ブロック間隔の実測と、「次のブロックで入るなら P(L<50ms) はいくつか」の算術

新規発注の実効遅延そのものは**過去データからは測れない**(自分の注文を出して
みないと分からない)。ここで出すのは「いくつなら成立するか」の閾値であって、
「いくつである」ではない。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NW_LAGS = 14
BLK_DAYS = ["2026-07-15", "2026-07-29", "2026-08-06"]


def nw(v):
    n = v.size
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, NW_LAGS + 1):
        g += 2.0 * (1 - lg / (NW_LAGS + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    te = {f.stem.split("=")[1] for f in
          sorted(Path("E:/Memory-quotes").joinpath(tag).glob("dt=*.parquet"))[59:]}
    rows = []
    for f in sorted(DATA.glob(f"inv_days_{tag}_q1*gated.csv")):
        m = re.search(r"lat(\d+)", f.name)
        L = int(m.group(1)) if m else 0
        D = pl.read_csv(f).filter(pl.col("dt").is_in(list(te)))
        v = D["pair_pnl_mean"].to_numpy()
        v = v[np.isfinite(v)]
        mm, se = nw(v)
        rows.append({"lag_ms": L, "pairs_day": int(D["n_pair"].sum()) / D.height,
                     "ev_pair": mm, "se": se, "t": mm / se,
                     "per_day_bp": float(D["total_bp"].sum()) / D.height})
    F = pl.DataFrame(rows).sort("lag_ms")
    F.write_csv(DATA / f"lfrontier_{tag}.csv")
    x = F["lag_ms"].to_numpy().astype(float)
    y = F["ev_pair"].to_numpy()
    be = np.nan
    for i in range(1, x.size):
        if y[i - 1] > 0 >= y[i]:
            be = x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i])
            break
    print(F)
    print(f"\n★ 損益分岐の遅延 ≈ {be:.0f} ms")

    gaps = []
    for dt in BLK_DAYS:
        d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                      .filter(pl.col("dt") == dt).collect())[0].sort("ts")
        t = d["ts"].cast(pl.Int64).to_numpy()
        g = np.diff(t)
        gaps.append(g[g > 0] / 1e6)
    G = np.concatenate(gaps)
    one = G[(G > 40) & (G < 95)]              # 1 ブロックとみなせる山
    blk = float(np.median(one))
    print(f"ブロック間隔の実測 {blk:.2f} ms(1 ブロックの山 {one.size:,} 件)"
          f"  p10 {np.quantile(one,0.1):.1f} / p90 {np.quantile(one,0.9):.1f} "
          f"/ 標準偏差 {one.std():.2f} ms")
    kk = np.round(G / blk).astype(int)
    print("  板が動いたブロックどうしの間隔(何ブロック空いたか):",
          " ".join(f"{v}:{100*np.mean(kk == v):.1f}%" for v in (1, 2, 3, 4)),
          f" 5 以上:{100*np.mean(kk >= 5):.1f}%")
    out = []
    for d in range(0, 61, 5):
        p1 = max(0.0, min(1.0, (be - d) / blk)) if be == be else np.nan
        out.append({"net_delay_ms": d, "p_under_be_1block": p1,
                    "p_under_be_2block": 0.0})
    pl.DataFrame(out).write_csv(DATA / f"lfrontier_block_{tag}.csv")
    print("次のブロックで入る場合に、損益分岐を切る確率")
    for r in out[::2]:
        print(f"  ネットワーク遅延 {r['net_delay_ms']:>2} ms → "
              f"P(L < {be:.0f}ms) = {100*r['p_under_be_1block']:.0f}%")
    print(f"  2 ブロックかかると L は最小でも {blk:.0f} ms なので 0%")
    pl.DataFrame({"gap_ms": G[G < 500]}).write_parquet(
        DATA / f"lfrontier_gaps_{tag}.parquet")
    print(f"\n書き出し {DATA}/lfrontier_{tag}.csv ほか")


if __name__ == "__main__":
    main()
