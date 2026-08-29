r"""帯内/帯外に分解した板を、Pressure 解析が読める形にまとめる。

band_book_{mid}(帯内残量)と event_book_{mid}(全体)を ev_i で結合し、
    ib_side   = 帯内の残量
    oob_side  = depth_side − ib_side   (帯外の残量)
    bq_side   = 最良 1 レベル(分母に使う。元のまま)
    depth_side= 片側総量(参照)
を持つ event_book_band_{mid}.parquet を書く。

★これで「信号は報酬帯の内側の流動性から来ているのか、外側から来ているのか」を
  板を再生し直さずに測れる。README が全レポートで交絡として挙げてきた問いに、
  除外板と同じ形式(同じ時間契約・同じ帰無対照)で答えられる。

★ oob は差で作るので、丸め誤差で僅かに負になり得る。0 未満は 0 に丸める
  (実測での発生率も出す)。
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="")
    a = ap.parse_args()
    if a.markets:
        mids = [int(x) for x in a.markets.split(",")]
    else:
        mids = sorted(int(re.search(r"band_book_(\d+)", f).group(1))
                      for f in glob.glob(str(OUT / "band_book_*.parquet")))
    n_neg = n_tot = 0
    made = 0
    for mid in mids:
        pb, pe = OUT / f"band_book_{mid}.parquet", OUT / f"event_book_{mid}.parquet"
        if not (pb.exists() and pe.exists()):
            continue
        B = pl.read_parquet(pb)
        E = pl.read_parquet(pe, columns=["ev_i", "ts", "block", "mid_pp", "extrapolated",
                                         "bq_long", "bq_short",
                                         "depth_long", "depth_short"])
        if B.height != E.height:
            print(f"market {mid}: 行数が違う {B.height} vs {E.height} — 飛ばす")
            continue
        J = E.join(B.select(["ev_i", "ib_long", "ib_short",
                             "ibn_long", "ibn_short"]), on="ev_i", how="inner")
        if J.height != E.height:
            print(f"market {mid}: 結合で行が落ちた — 飛ばす")
            continue
        for s in ("long", "short"):
            neg = (pl.col(f"depth_{s}") - pl.col(f"ib_{s}")) < -1e-9
            n_neg += int(J.select(neg.sum()).item())
            n_tot += J.height
        J = J.with_columns([
            (pl.col("depth_long") - pl.col("ib_long")).clip(lower_bound=0).alias("oob_long"),
            (pl.col("depth_short") - pl.col("ib_short")).clip(lower_bound=0).alias("oob_short"),
        ])
        J.write_parquet(OUT / f"event_book_band_{mid}.parquet")
        made += 1
    print(f"作成 {made} 市場")
    print(f"oob が負になった件数: {n_neg:,} / {n_tot:,} = {n_neg/max(n_tot,1):.6%}(0 に丸めた)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
