"""候補テーブルの NaN を null に統一する後処理。

polars は null を無視して集計するが NaN は伝播する。両者が混ざっていると
mean は NaN、median は値、という不整合が出て集計が静かに壊れる。
「該当なし」は null で表す方に寄せる。値は一切変えない。
"""
import argparse
import os
from pathlib import Path

import polars as pl

BULK = Path("E:/Memory-quotes")
SUB = Path("data")


def fix(d: Path) -> tuple[int, int]:
    n_file = n_cell = 0
    for f in sorted(d.glob("dt=*.parquet")):
        T = pl.read_parquet(f)
        fl = [c for c, t in T.schema.items() if t in (pl.Float32, pl.Float64)]
        cnt = int(T.select(pl.sum_horizontal(
            [pl.col(c).is_nan().sum() for c in fl]))[0, 0] or 0)
        if cnt == 0:
            continue
        T = T.with_columns([pl.when(pl.col(c).is_nan()).then(None)
                            .otherwise(pl.col(c)).alias(c) for c in fl])
        tmp = f.with_suffix(".tmp")           # 読み取り中の同じ経路へは書けない
        T.write_parquet(tmp, compression="zstd")
        del T
        os.replace(tmp, f)
        n_file += 1
        n_cell += cnt
    return n_file, n_cell


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    for d in (BULK / tag, SUB / f"quotes_sub_{tag}"):
        print(d, "->", fix(d), flush=True)
