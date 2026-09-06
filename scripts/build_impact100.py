"""sweep インパクトの非対称を 100 ms 格子で作り直す(候補 2 の「新鮮な信号」)。

    uv run python scripts/build_impact100.py --coin xyz:MU

既存の `impact_*.parquet` は **5 秒格子**である。発注時点で使うと信号の古さは
中央 2.49 秒になり、実測した半減期(約 250 ms)から見て強度が 3% しか残らない。
そこで板の再構成(`build_obi_levels`、100 ms 格子・10 水準)から

    I_buy(Q)  = ask 側を Q 枚食べたときの、mid からの出来値の乖離(bp)
    I_sell(Q) = bid 側を Q 枚食べたときの同じ量
    A = I_sell(Q) − I_buy(Q)

を作り直す。Q は 5 枚(既報と同じ)。

時間契約: ラダーは**格子セルの開始時刻の状態**(= その時刻以前)で作られている
(`mu_quotes_table_report.md` §4-B で実測確認済み)。先読みはしていない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (GRID_NS, PX_UNIT, SZ_LOT, clean_bbo,
                              day_features as ladder_day)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
Q = 5.0          # 枚


def sweep(dep, tick, mid, q):
    """各格子点で q 枚を食べたときの、mid からの出来値乖離(bp)。

    dep[:, k] は最良から k 水準目の数量(ロット)。価格は mid から
    (0.5*spread + k*tick) 離れているが、ここでは最良からの距離を使い、
    最良自体の mid からの距離は呼び出し側で足す。
    """
    n, L = dep.shape
    rem = np.full(n, q)
    cost = np.zeros(n)
    for k in range(L):
        d = np.nan_to_num(dep[:, k]) * SZ_LOT
        take = np.minimum(rem, d)
        cost += take * (k * tick)
        rem -= take
    # 足りない分は最終水準の価格で埋める(保守的)
    cost += rem * ((L - 1) * tick)
    return cost / q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((DATA / f"l1_{tag}").glob("dt=*.parquet"))
    carry = pl.DataFrame()
    out = []
    for k, f in enumerate(files):
        dt = f.stem.split("=")[1]
        bb = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                       .filter(pl.col("dt") == dt).collect())[0].sort("ts")
        if bb.height < 1000:
            continue
        keep = {"step": 1, "gi": [], "qb": [], "qa": [], "ok": []}
        _, _, _, _, _, carry, _ = ladder_day(f, bb, carry, keep=keep)
        gi = np.concatenate(keep["gi"])
        o = np.argsort(gi)
        gi = gi[o]
        Qb = np.concatenate(keep["qb"])[o]
        Qa = np.concatenate(keep["qa"])[o]
        ok = np.concatenate(keep["ok"])[o]
        del keep
        ts = bb["ts"].cast(pl.Int64).to_numpy()
        d0 = int(ts[0]) // 86_400_000_000_000 * 86_400_000_000_000
        gt = d0 + gi.astype(np.int64) * GRID_NS
        j = np.clip(np.searchsorted(ts, gt, side="right") - 1, 0, ts.size - 1)
        pb, pa = bb["best_bid"].to_numpy()[j], bb["best_ask"].to_numpy()[j]
        mid = 0.5 * (pb + pa)
        tick = np.where(mid >= 1000.0, 0.1, 0.01)
        half = (pa - pb) / 2.0
        # 買いは ask 側、売りは bid 側を食べる
        ib = (half + sweep(Qa, tick, mid, Q)) / mid * 1e4      # I_buy
        isl = (half + sweep(Qb, tick, mid, Q)) / mid * 1e4     # I_sell
        m = ok & np.isfinite(ib) & np.isfinite(isl)
        out.append(pl.DataFrame({
            "dt": [dt] * int(m.sum()), "ts": gt[m],
            "imp_b_q5": isl[m].astype(np.float32),
            "imp_a_q5": ib[m].astype(np.float32)}))
        del Qb, Qa, bb
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(files)}] {dt} 格子 {int(m.sum()):,}", flush=True)
    T = pl.concat(out)
    T.write_parquet(DATA / f"impact100_{tag}.parquet")
    print(f"\n{T.height:,} 点(100 ms 格子)  書き出し "
          f"{DATA}/impact100_{tag}.parquet")


if __name__ == "__main__":
    main()
