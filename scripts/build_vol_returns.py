"""基準格子のリターン系列そのものを保存する (item 47 の検証用)。

跳び検定の帰無対照を作るのに、その日の 0 リターンの**並び**が要る。0 は静かな
時間帯に固まって出るので、無作為にばらまく帰無は BV を壊しすぎて z を過大に
出す。実際の並びを保ったまま非 0 部分だけ差し替える対照を作るため、格子上の
リターンを一度だけ書き出しておく。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_vol import DAY_NS, REF_GRID, SCALE, SERIES, levels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 行を除外)", flush=True)
    out = []
    days = sorted(bb["dt"].unique().to_list())
    for k, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        ts, lv = levels(d)
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        tg = np.arange(d0, d0 + DAY_NS + 1, int(REF_GRID * 1e9))
        gi = np.searchsorted(ts, tg, side="right") - 1
        gi = gi[gi >= 0]
        m = len(gi) - 1
        out.append(pl.DataFrame({"dt": [dt] * m, "i": np.arange(m, dtype=np.int32),
                                 **{s: (np.diff(lv[s][gi]) * SCALE[s]).astype(np.float64)
                                    for s in SERIES}}))
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True)
    R = pl.concat(out)
    R.write_parquet(DATA / f"vol_returns_{tag}.parquet")
    print(f"書き出し vol_returns_{tag}.parquet  {R.height:,} 行 "
          f"({REF_GRID:g} 秒格子・{len(days)} 日)")


if __name__ == "__main__":
    main()
