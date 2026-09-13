"""入口の門の**帰無対照** — 同じ日・同じ側で、同じ本数だけ無作為に通す門。

    uv run python scripts/build_mmgate_null.py --coin xyz:MU

出力: data/entrygate_<coin>_mmgrnd.parquet

## なぜ要るか

門を入れると 1 組あたり損益は必ず上がる。しかしその原因は 2 つありうる。

1. 門が**良い場面を選んでいる**(本物)
2. 門が**発注を減らしただけ**(偽)。出す本数が減れば、残った組は
   たまたま条件の良い時間帯に偏りうるし、そもそも組数が減るので
   分散が下がって「単価が良く見える」だけということが起こる

この 2 つは、**通す本数だけを合わせて中身を無作為にした門**と比べれば分かれる。
無作為の門でも同じだけ上がるなら、門の中身には価値が無い。

日と側ごとに本数を合わせるのは、日内の発注数の偏りと買い売りの非対称を
そのまま残すためである(全期間で本数だけ合わせると、門が特定の日に
集中していた事実が消えてしまう)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    G = pl.read_parquet(DATA / f"entrygate_{tag}_mmg.parquet")
    rng = np.random.default_rng(a.seed)
    out = []
    for key, g in sorted(G.partition_by("dt", "side", as_dict=True).items(),
                         key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        k = int(g["pass"].sum())
        v = np.zeros(g.height, np.int8)
        if k:
            v[rng.choice(g.height, size=k, replace=False)] = 1
        out.append(g.with_columns(pl.Series("pass", v)))
    R = pl.concat(out).sort("dt", "t")
    R.write_parquet(DATA / f"entrygate_{tag}_mmgrnd.parquet", compression="zstd")
    print(f"{a.coin}: 発注 {G.height:,} / 通した本数 実 {int(G['pass'].sum()):,}"
          f" = 無作為 {int(R['pass'].sum()):,}(日 × 側ごとに一致)")
    print(f"書き出し data/entrygate_{tag}_mmgrnd.parquet")


if __name__ == "__main__":
    main()
