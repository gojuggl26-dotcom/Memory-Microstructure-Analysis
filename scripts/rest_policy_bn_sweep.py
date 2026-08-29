"""遅延を細かく刻んで損益分岐点を出す — 東京コロケが効くかの直接判定。

【問い】
  実測(報告 46)で、遅延は開発機 85.8ms → 東京 EC2 13.2ms に落ちる。
  では **E[PnL] がゼロを超えるのは何 ms までか**。信号あり・なしで分けて出す。

【構成】 k=0(報告 37 の最良方策)、遅延 {0, 25, 50, 75, 100} ms、
        信号 = なし / 2bp・保持 500ms(本セッションの最良)

【出力】 data/rest_policy_bn_sweep.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from rest_policy_bn import (D, LAT_READ_NS, MS, WIN_NS, binance_signal,  # noqa: E402
                            simulate)
from capacity_v3 import load                                             # noqa: E402
from backtester_v3 import load_model                                     # noqa: E402

LATS = [0, 25, 50, 75, 100]
SIGS = [(np.inf, 0), (2.0, 500)]
K = 0


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    q = float(pl.read_csv(D / "capacity_100usd.csv")
              .filter(pl.col("lev") == 1)["qty"][0])
    out = D / "rest_policy_bn_sweep.csv"
    rows = []
    if out.exists():
        old = pl.read_csv(out)
        done = {dt for dt, c in old.group_by("dt").len().iter_rows()
                if c == len(LATS) * len(SIGS)}
        rows = old.filter(pl.col("dt").is_in(list(done))).to_dicts()
        days = [d_ for d_ in days if d_ not in done]
        print(f"再開: 完了 {len(done)} 日 / 残り {len(days)} 日", flush=True)
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        sg = binance_signal(dt, WIN_NS, LAT_READ_NS)
        if sg is None:
            continue
        for lat in LATS:
            for thr, hold in SIGS:
                r = simulate(d, q, 5.0 * q, K, lat * MS, lat * MS,
                             sg[0], sg[1], thr, hold * MS)
                if r:
                    rows.append({"dt": dt, "lat_ms": lat, "thr_bp": float(thr),
                                 "hold_ms": hold, **r})
        print(f"{i+1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(out)
    pl.DataFrame(rows).write_csv(out)
    print("完了")


if __name__ == "__main__":
    main()
