"""承認された新特徴量を全 99 日で算出する。

  1. book_slope_bid / _ask / _diff   … 板の傾き(最良気配からの距離の数量加重平均)
  2. hhi_new_bid / _ask / hhi_diff   … 新規注文数量の参加者集中(ハーフィンダール)
  3. top1_share_new_bid / _ask / top1_diff … 最大参加者のシェア
  4. reject_rate / alo_reject_rate   … ★ts_close 基準に修正(棄却注文は ts_open が null)

★時間契約(CLAUDE.md 厳禁事項): 板の傾きは T 時点の状態、
  参加者・棄却は窓 [T−Δ, T) の集計。どちらも T までに確定している。
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
BOOKPX = Path("data/l2_v99/book_px")
OUT = D / "new_features"
STEP = 1_000_000_000
NS_DAY = 86_400_000_000_000
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
DEPTH_SCAN = 60          # 傾きの計算に使う最大レベル数(それ以深は距離が遠すぎて寄与しない)


def grid_of(dt: str) -> np.ndarray:
    t0 = int(datetime.strptime(dt, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    return np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)


def bsum(ts: np.ndarray, val: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, ts, side="right")
    o = np.zeros(grid.size + 1)
    np.add.at(o, np.clip(idx, 0, grid.size), val)
    return o[: grid.size]


def book_slope(dt: str, grid: np.ndarray) -> dict:
    """T 時点の板を再生し、最良気配からの距離の数量加重平均を側別に出す。"""
    df = (pl.read_parquet(BOOKPX / f"dt={dt}" / "part-000.parquet",
                          columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list()
    sd = df["side"].to_list()
    px = df["px"].to_list()
    sz = df["total_sz"].to_list()
    n = grid.size
    keys: dict[str, list] = {"B": [], "A": []}
    vals: dict[str, list] = {"B": [], "A": []}
    out = {"book_slope_bid": np.zeros(n), "book_slope_ask": np.zeros(n)}
    gi = 0
    i = 0
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
                    arr.insert(j, k)
                    va.insert(j, sz[i])
            elif hit:
                del arr[j]
                del va[j]
            i += 1
        for s, tag in (("B", "bid"), ("A", "ask")):
            arr, va = keys[s], vals[s]
            if not arr:
                out[f"book_slope_{tag}"][gi] = (out[f"book_slope_{tag}"][gi - 1]
                                                if gi else 0.0)
                continue
            best = -arr[0] if s == "B" else arr[0]
            num = den = 0.0
            for kk, vv in zip(arr[:DEPTH_SCAN], va[:DEPTH_SCAN]):
                p = -kk if s == "B" else kk
                num += abs(p - best) / best * 1e4 * vv
                den += vv
            out[f"book_slope_{tag}"][gi] = num / den if den > 0 else 0.0
        gi += 1
        if i >= N and gi < n:                       # 以降は板が動かない
            for tag in ("bid", "ask"):
                out[f"book_slope_{tag}"][gi:] = out[f"book_slope_{tag}"][gi - 1]
            break
    out["book_slope_diff"] = out["book_slope_bid"] - out["book_slope_ask"]
    return out


def wallet_and_reject(dt: str, grid: np.ndarray) -> dict:
    df = pl.read_parquet(
        LIFE / f"dt={dt}" / "part-000.parquet",
        columns=["user", "side", "orig_sz", "ts_open", "ts_close",
                 "terminal_status", "tif", "is_rejected", "is_trigger"])
    n = grid.size
    out: dict[str, np.ndarray] = {}

    # ---- 参加者集中(板に載った新規注文のみ) ----
    rest = df.filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                     & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                     & pl.col("ts_open").is_not_null())
    win = np.clip(np.searchsorted(grid, rest["ts_open"].to_numpy(), side="right"), 0, n)
    t = pl.DataFrame({"w": win, "u": rest["user"].to_numpy(),
                      "sz": rest["orig_sz"].to_numpy(),
                      "b": rest["side"].to_numpy() == "B"}).filter(pl.col("w") < n)
    for side, tag in ((True, "bid"), (False, "ask")):
        g = (t.filter(pl.col("b") == side)
              .group_by(["w", "u"]).agg(sz=pl.col("sz").sum())
              .group_by("w").agg(tot=pl.col("sz").sum(),
                                 sq=(pl.col("sz") ** 2).sum(),
                                 top=pl.col("sz").max()))
        w_ = g["w"].to_numpy()
        tot = g["tot"].to_numpy()
        a = np.zeros(n)
        a[w_] = np.where(tot > 0, g["sq"].to_numpy() / np.maximum(tot ** 2, 1e-12), 0.0)
        out[f"hhi_new_{tag}"] = a
        a = np.zeros(n)
        a[w_] = np.where(tot > 0, g["top"].to_numpy() / np.maximum(tot, 1e-12), 0.0)
        out[f"top1_share_new_{tag}"] = a
    out["hhi_diff"] = out["hhi_new_bid"] - out["hhi_new_ask"]
    out["top1_diff"] = out["top1_share_new_bid"] - out["top1_share_new_ask"]

    # ---- 棄却率(★ts_close 基準。棄却注文は ts_open が null) ----
    rej = df.filter(pl.col("is_rejected"))
    acc = df.filter((~pl.col("is_rejected")) & pl.col("ts_open").is_not_null())
    n_rej = bsum(rej["ts_close"].to_numpy(), np.ones(rej.height), grid)
    n_acc = bsum(acc["ts_open"].to_numpy(), np.ones(acc.height), grid)
    tot = n_rej + n_acc
    out["reject_rate"] = np.where(tot > 0, n_rej / np.maximum(tot, 1), 0.0)

    alo_r = rej.filter(pl.col("terminal_status") == "badAloPxRejected")
    alo_a = acc.filter(pl.col("tif") == "Alo")
    a_r = bsum(alo_r["ts_close"].to_numpy(), np.ones(alo_r.height), grid)
    a_a = bsum(alo_a["ts_open"].to_numpy(), np.ones(alo_a.height), grid)
    at = a_r + a_a
    out["alo_reject_rate"] = np.where(at > 0, a_r / np.maximum(at, 1), 0.0)
    out["n_reject"] = n_rej
    return out


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D / "microprice").iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        days = days[: int(sys.argv[1])]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, dt in enumerate(days, 1):
        if (OUT / f"{dt}.parquet").exists():      # 途中再開: 既に出来ている日は飛ばす
            continue
        grid = grid_of(dt)
        f = book_slope(dt, grid)
        f.update(wallet_and_reject(dt, grid))
        pl.DataFrame({"ts": grid, **{k: v.astype(np.float64) for k, v in f.items()}}
                     ).write_parquet(OUT / f"{dt}.parquet", compression="zstd")
        if i % 10 == 0 or i == len(days):
            print(f"[{i:3d}/{len(days)}] {dt}", flush=True)
    print("done", len(days))


if __name__ == "__main__":
    main()
