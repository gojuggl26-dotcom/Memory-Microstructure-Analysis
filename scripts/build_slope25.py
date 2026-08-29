"""25bp 帯の book slope を 1 秒グリッドで全 99 日ぶん算出する。

    S^side_t = Σ_{i: 距離_i ≤ 25bp} (距離_i[bp] × 数量_i) / Σ_{i: 距離_i ≤ 25bp} 数量_i

粒度は book_slope_grain_report.md の実測に基づく(深さ 25bp・地平 1 秒が最良)。
★時間契約: T 時点の板の状態のみ(backward)。未来は一切参照しない。
途中再開できるよう日ごとにファイルを書く。
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
BOOKPX = Path("data/l2_v99/book_px")
OUT = D / "slope25"
STEP = 1_000_000_000
NS_DAY = 86_400_000_000_000
BAND_BP = 25.0


def run_day(dt: str) -> None:
    f = OUT / f"{dt}.parquet"
    if f.exists():
        return
    t0 = int(datetime.strptime(dt, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    grid = np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)
    df = (pl.read_parquet(BOOKPX / f"dt={dt}" / "part-000.parquet",
                          columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list(); sd = df["side"].to_list()
    px = df["px"].to_list(); sz = df["total_sz"].to_list()
    n = grid.size
    keys: dict[str, list] = {"B": [], "A": []}
    vals: dict[str, list] = {"B": [], "A": []}
    out = {"s_bid": np.zeros(n), "s_ask": np.zeros(n),
           "q_bid": np.zeros(n), "q_ask": np.zeros(n)}
    gi = i = 0
    N = len(ts)
    while gi < n:
        while i < N and ts[i] <= grid[gi]:
            s = sd[i]
            k = -px[i] if s == "B" else px[i]
            arr, va = keys[s], vals[s]
            j = bisect_left(arr, k)
            hit = j < len(arr) and arr[j] == k
            if sz[i] > 0:
                if hit:
                    va[j] = sz[i]
                else:
                    arr.insert(j, k); va.insert(j, sz[i])
            elif hit:
                del arr[j]; del va[j]
            i += 1
        for s, tag in (("B", "bid"), ("A", "ask")):
            arr, va = keys[s], vals[s]
            if not arr:
                out[f"s_{tag}"][gi] = out[f"s_{tag}"][gi - 1] if gi else 0.0
                out[f"q_{tag}"][gi] = 0.0
                continue
            best = -arr[0] if s == "B" else arr[0]
            num = den = 0.0
            for kk, vv in zip(arr, va):
                p = -kk if s == "B" else kk
                dd = abs(p - best) / best * 1e4
                if dd > BAND_BP:
                    break                       # 距離順に並んでいるので以降は帯の外
                num += dd * vv
                den += vv
            out[f"s_{tag}"][gi] = num / den if den > 0 else 0.0
            out[f"q_{tag}"][gi] = den
        gi += 1
        if i >= N and gi < n:
            for key in out:
                out[key][gi:] = out[key][gi - 1]
            break
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts": grid, **out}).write_parquet(f, compression="zstd")


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D / "microprice").iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        days = sys.argv[1:]
    for k, dt in enumerate(days, 1):
        run_day(dt)
        if k % 5 == 0 or k == len(days):
            print(f"[{k}/{len(days)}] {dt}", flush=True)
    print("done", len(list(OUT.glob('*.parquet'))))


if __name__ == "__main__":
    main()
