"""パネルの板の再構成を、独立な経路(`orderlife_*`)と突き合わせて検査する。

    uv run python scripts/audit_l4feat.py --coin xyz:INTC [--dt 2026-05-05 -n 20]

何を検査するか
--------------
`build_l4feat.py` の板は**イベントの差分を価格 × 格子の 2 次元で累積**して
組み立てている。この仕掛けが正しいかを、まったく別の作り方をした
`orderlife_*`(1 行 = 指値 1 本、開始と終了の時刻を持つ)から

    格子点 T の最良買いに居る注文 = { ts_open < T < ts_close かつ 同じ価格 }

を素朴に数えて突き合わせる。**本数・年齢・数量**の 3 つを見る。

★ 突き合わせに要る 2 つの落とし穴(両方とも実際に踏んだ)

  1. **打ち切り注文を入れ忘れない。** 窓の終わりまで一度も約定も取消も
     されなかった注文は `orderlife_*/dt=*.parquet` には現れず、
     `orderlife_*_censored.parquet` にだけ居る。これを外すと
     「パネルにだけ 3 本ある」ように見える(実測 20 点中 2 点で発生)
  2. **数量は部分約定で減る。** `orig_lots` は発注時の数量なので、
     部分約定のあった注文では板に残っている量と一致しない。
     数量の一致は「その水準に部分約定した注文が居ないとき」だけ厳密に見る

時間契約もここで見る: 比較は `ts_open < T`(等号を入れない)で行う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import PX_UNIT, SZ_LOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = Path("E:/Memory-l4feat")
DAY_NS = 86_400_000_000_000
INF = 2 ** 62


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    ap.add_argument("--dt", default=None)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    days = sorted(p.name[3:13] for p in (SRC / tag).glob("dt=*.parquet"))
    dt = a.dt or days[1]
    t0 = int(np.datetime64(dt, "ns").astype("int64"))
    P = pl.read_parquet(SRC / tag / f"dt={dt}.parquet",
                        columns=["sec", "best_bid", "qb_l1", "nb_l1", "age_b1"])
    C = ["ts_open", "ts_close", "is_bid", "pidx", "orig_lots", "filled_lots"]
    lf = (pl.scan_parquet(str(DATA / f"orderlife_{tag}" / "dt=*.parquet"))
          .select(C)
          .filter((pl.col("ts_close") > t0) & (pl.col("ts_open") < t0 + DAY_NS))
          .collect())
    cen = DATA / f"orderlife_{tag}_censored.parquet"
    if cen.exists():
        c = (pl.read_parquet(cen)
             .filter(pl.col("ts_open") < t0 + DAY_NS)
             .with_columns(ts_close=pl.lit(INF, pl.Int64),
                           filled_lots=pl.lit(0, pl.Int64))
             .select(C))
        lf = pl.concat([lf, c])
    print(f"{a.coin} {dt}: 板に居た注文 {lf.height:,} 本"
          f"(うち打ち切り {c.height if cen.exists() else 0:,})", flush=True)

    to, tc = lf["ts_open"].to_numpy(), lf["ts_close"].to_numpy()
    pi, ib = lf["pidx"].to_numpy(), lf["is_bid"].to_numpy()
    ol = lf["orig_lots"].to_numpy().astype(float)
    fl = lf["filled_lots"].to_numpy().astype(float)
    rng = np.random.default_rng(a.seed)
    secs = rng.integers(7200, 86000, a.n)
    okn = oka = okq = tot = qtot = 0
    for s in secs:
        s = int(s)
        bb = P["best_bid"][s]
        if bb is None or not np.isfinite(bb):
            continue
        T = t0 + s * 1_000_000_000
        p = int(round(float(bb) / PX_UNIT))
        m = ib & (pi == p) & (to < T) & (tc > T)
        n = int(m.sum())
        if n == 0:
            continue
        tot += 1
        age = float((T - to[m]).mean()) / 1e9
        gn, ga, gq = (float(P["nb_l1"][s]), float(P["age_b1"][s]),
                      float(P["qb_l1"][s]))
        okn += abs(n - gn) < 1e-6
        oka += abs(age - ga) < 0.02
        if (fl[m] == 0).all():                # 部分約定が無いときだけ数量を見る
            qtot += 1
            q = float(ol[m].sum()) * SZ_LOT
            okq += abs(q - gq) < 0.02
            if abs(q - gq) >= 0.02:
                print(f"  数量が違う sec={s} 手 {q:.3f} 対 列 {gq:.3f}")
        if abs(n - gn) > 1e-6 or abs(age - ga) > 0.02:
            print(f"  差 sec={s} 本数 {n} 対 {gn:.0f} / 年齢 {age:.2f} 対 {ga:.2f}")
    print(f"\n本数 {okn}/{tot}  年齢(±0.02 秒) {oka}/{tot}  "
          f"数量(部分約定なしの {qtot} 点) {okq}/{qtot}")
    if okn == tot and oka == tot and okq == qtot:
        print("=== 合格: 板の再構成は独立な経路と一致した ===")
    else:
        print("=== 不一致あり ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
