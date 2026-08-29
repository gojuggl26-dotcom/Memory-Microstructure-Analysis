"""L10 の book_slope に加法スプライン回帰を当て、7 つの予測地平で評価する。

【book_slope(L10)の定義】
    S^side = Σ_{i≤10} q_i · |p_i − best| / Σ_{i≤10} q_i      [bp]
  最良気配から数えて 10 段までを使い、**数量加重の平均距離**を測る。
  既存の slope25 は「25bp 帯」で切っていたが、本報告は**段数**で切る。

【模型】
    線形       y = α + b_B·S^Bid + b_A·S^Ask
    スプライン  y = α + f_B(S^Bid) + f_A(S^Ask)      f は 3 次 B スプライン(加法)

【予測地平】 1ms / 100ms / 1s / 3s / 5s / 10s / 15s
  1ms・100ms を測るため **1 秒グリッドではなくイベント粒度**で評価点を取る。
  book_px の更新時刻を 5 個に 1 個間引いて評価点とし、
  そこから各地平の中値を backward asof で引く。

【★ルックアヘッドの排除】
  ・板の状態は評価時刻 t までのイベントのみ(ts ≤ t)
  ・y は t から t+h への前向きリターン。x と重ならない
  ・節点は最初の 20 日の分位に固定(全評価日にとって過去)
  ・係数は拡張窓で学習期間のみ。R² の分母の平均も学習期間のもの
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
BOOKPX = Path("data/l2_v99/book_px")
OUT = D / "slope_spline"
NL = 10
STRIDE = 5                     # 板イベントを何個に 1 個評価点にするか
HZ = [("1ms", 1_000_000), ("100ms", 100_000_000), ("1s", 1_000_000_000),
      ("3s", 3_000_000_000), ("5s", 5_000_000_000), ("10s", 10_000_000_000),
      ("15s", 15_000_000_000)]


def run_day(dt: str) -> dict | None:
    f = OUT / f"{dt}.npz"
    if f.exists() and "ts" in np.load(f).files:
        return {"dt": dt, "cached": True}
    bp = BOOKPX / f"dt={dt}" / "part-000.parquet"
    mpf = D / f"microprice/dt={dt}/part-000.parquet"
    if not bp.exists() or not mpf.exists():
        return None
    mp = (pl.read_parquet(mpf, columns=["ts", "mid", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    if mp.height < 5000:
        return None
    mts = mp["ts"].to_numpy(); lm = np.log(mp["mid"].to_numpy())

    df = (pl.read_parquet(bp, columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list(); sd = df["side"].to_list()
    px = df["px"].to_list(); sz = df["total_sz"].to_list()
    keys: dict[str, list] = {"B": [], "A": []}
    vals: dict[str, list] = {"B": [], "A": []}
    et, sb, sa = [], [], []
    N = len(ts)
    i = 0
    emit = 0
    while i < N:
        t_now = ts[i]
        while i < N and ts[i] == t_now:            # 同一時刻はまとめて反映
            s = sd[i]
            k = -px[i] if s == "B" else px[i]      # 最良が先頭に来るよう符号反転
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
        emit += 1
        if emit % STRIDE:
            continue
        kb, ka = keys["B"][:NL], keys["A"][:NL]
        vb, va_ = vals["B"][:NL], vals["A"][:NL]
        if not kb or not ka:
            continue
        best_b, best_a = -kb[0], ka[0]
        mid_ = (best_b + best_a) / 2
        if mid_ <= 0:
            continue
        wb = sum(vb); wa = sum(va_)
        if wb <= 0 or wa <= 0:
            continue
        # 数量加重の平均距離(bp)
        s_b = sum(v * (best_b - (-k)) for k, v in zip(kb, vb)) / wb / mid_ * 1e4
        s_a = sum(v * (k - best_a) for k, v in zip(ka, va_)) / wa / mid_ * 1e4
        et.append(t_now); sb.append(s_b); sa.append(s_a)
    if len(et) < 5000:
        return None
    et = np.array(et, dtype=np.int64)
    sb = np.array(sb); sa = np.array(sa)

    j0 = np.searchsorted(mts, et, side="right") - 1
    ok = j0 >= 0
    et, sb, sa, j0 = et[ok], sb[ok], sa[ok], j0[ok]
    Y = np.full((len(et), len(HZ)), np.nan)
    for h, (nm, ns) in enumerate(HZ):
        j1 = np.searchsorted(mts, et + ns, side="right") - 1
        v = np.where(j1 >= 0, (lm[np.clip(j1, 0, len(lm) - 1)] - lm[j0]) * 1e4, np.nan)
        v[et + ns > mts[-1]] = np.nan
        Y[:, h] = v
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, ts=et.astype(np.int64),
                        sb=sb.astype(np.float32), sa=sa.astype(np.float32),
                        Y=Y.astype(np.float32))
    return {"dt": dt, "n": len(et)}


def main() -> None:
    for dt in sys.argv[1:]:
        r = run_day(dt)
        print(f"{dt}: {r.get('n', '既存'):>9} 点" if r else f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
