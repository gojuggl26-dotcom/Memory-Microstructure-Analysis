"""毎ティックの OBI を 1 分グリッドへ貼り付けた軽量版を作る(配布・目視確認用)。

全解像度(41.8M 行 / 3.1GB)は手元にしか置けないので、各分の境界時点での
直近の OBI(L=1..10) を CSV にしておく。stale_ns で鮮度が分かる。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

OUT = Path("C:/Users/ii562/Downloads/Memory")
OBI = OUT / "data" / "obi"
STEP = 60_000_000_000
NS_DAY = 86_400_000_000_000
COLS = ["best_bid", "best_ask", "mid"] + [f"obi_{i}" for i in range(1, 11)] + [
    "cum_bid_10", "cum_ask_10", "n_lv_bid", "n_lv_ask"]


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in OBI.iterdir() if p.is_dir())
    parts = []
    for dt in days:
        t0 = int(datetime.strptime(dt, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
        df = pl.read_parquet(OBI / f"dt={dt}" / "part-000.parquet").sort("ts")
        g = pl.DataFrame({"ts": list(range(t0, t0 + NS_DAY, STEP))}, schema={"ts": pl.Int64})
        j = g.join_asof(
            df.select(["ts", *COLS]).rename({"ts": "ts_src"}),
            left_on="ts", right_on="ts_src", strategy="backward",
        )
        parts.append(
            j.with_columns((pl.col("ts") - pl.col("ts_src")).alias("stale_ns"),
                           pl.lit(dt).alias("dt")).drop("ts_src").drop_nulls(["obi_1"])
        )
    out = pl.concat(parts).with_columns(
        # CSV を無駄に太らせないよう有効桁で丸める(OBI は 1e-6 で十分)
        [pl.col(f"obi_{i}").round(6) for i in range(1, 11)]
        + [pl.col(c).round(4) for c in ("cum_bid_10", "cum_ask_10")]
    )
    out.write_csv(OUT / "data" / "obi_1m.csv")
    out.write_parquet(OUT / "data" / "obi_1m.parquet", compression="zstd")
    print(out.height, "rows")


if __name__ == "__main__":
    main()
