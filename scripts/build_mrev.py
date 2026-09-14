"""平均回帰の分析用に 1 分足(mid / microprice / 約定値)を作る。

    uv run python scripts/build_mrev.py --coin xyz:MU

出力: data/mrev_bars_<coin>.parquet
      (dt, minute, mid, micro, trade, spread_bp, qb, qa, obi, n_trade,
       buy_vol, sell_vol, n_quote)

## なぜ約定値も作るのか

**平均回帰を約定値で測ってはいけない。** 買いと売りが交互に来るだけで
リターンに強い負の自己相関が出る(bid–ask bounce)。これは平均回帰ではなく
測り方の副産物である。本スクリプトは mid / microprice と**並べて**約定値も出し、
その差を図で見せるために使う(`analyze_mrev.py` §1)。

## バーの作り方

1 分の**終値**は「その分の最後に観測された気配」。`searchsorted(..., "right") − 1`
の後ろ向き参照なので、バー時刻 T の値は T までの情報だけで確定する。
気配が 1 件も無い分は欠測にする(前の分を持ち越さない。持ち越すと
人工的な自己相関が入る)。

    mid   = (bid + ask) / 2
    micro = mid + (ask − bid)/2 × (qb − qa)/(qb + qa)

## 銘柄ごとの入力

* 通常: `data/bbo_<coin>.parquet`(L2 の bbo をローカルに組んだもの)
* `xyz:DRAM`: `data/DRAM/microprice_1s.parquet`(1 秒格子。旧レイアウトのため
  bbo を組み直していない。1 秒格子の最後の点をその分の終値とする)

## 時間契約

バー T の値は T までの情報だけ。目的変数(将来リターン)は `analyze_mrev.py`
の側で作る。ここでは先読みは一切していない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000
NMIN = 1440


def day_bars(ts, pb, pa, qb, qa, d0):
    """1 分の終値(後ろ向き)。気配が無い分は NaN。"""
    grid = d0 + (np.arange(NMIN, dtype=np.int64) + 1) * 60_000_000_000
    j = np.searchsorted(ts, grid, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    b, a = pb[j], pa[j]
    sb, sa = qb[j], qa[j]
    mid = 0.5 * (b + a)
    tot = np.maximum(sb + sa, 1e-12)
    obi = (sb - sa) / tot
    micro = mid + (a - b) / 2 * obi
    # その分に 1 件も気配が無い(= 前の分と同じ添字)なら欠測にする
    fresh = np.empty(NMIN, bool)
    fresh[0] = ok[0]
    fresh[1:] = ok[1:] & (j[1:] != j[:-1])
    nq = np.diff(np.concatenate([[0], j + 1]))
    m = np.where(ok & fresh, mid, np.nan)
    return (m, np.where(ok & fresh, micro, np.nan),
            np.where(ok, (a - b) / mid * 1e4, np.nan),
            np.where(ok, sb, np.nan), np.where(ok, sa, np.nan),
            np.where(ok, obi, np.nan), np.maximum(nq, 0))


def day_trades(ft, fpx, fsz, fbuy, d0):
    """1 分の約定終値と、符号つき出来高・件数。"""
    grid = d0 + (np.arange(NMIN, dtype=np.int64) + 1) * 60_000_000_000
    j = np.searchsorted(ft, grid, side="right") - 1
    ok = j >= 0
    close = np.where(ok, fpx[np.maximum(j, 0)], np.nan)
    fresh = np.empty(NMIN, bool)
    fresh[0] = ok[0]
    fresh[1:] = ok[1:] & (j[1:] != j[:-1])
    close = np.where(fresh, close, np.nan)
    lo = np.concatenate([[0], j[:-1] + 1])
    cb = np.concatenate([[0.0], np.cumsum(np.where(fbuy, fsz, 0.0))])
    cs = np.concatenate([[0.0], np.cumsum(np.where(~fbuy, fsz, 0.0))])
    hi = np.maximum(j + 1, 0)
    lo = np.maximum(lo, 0)
    return (close, cb[hi] - cb[lo], cs[hi] - cs[lo],
            np.maximum(hi - lo, 0))


def from_bbo(tag):
    b = pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
    days = sorted(b.select(pl.col("dt").unique()).collect()["dt"].to_list())
    fp = DATA / f"fills_{tag}.parquet"
    out = []
    for k, dt in enumerate(days):
        d = clean_bbo(b.filter(pl.col("dt") == dt).collect())[0].sort("ts")
        if d.height < 60:
            continue
        ts = d["ts"].cast(pl.Int64).to_numpy()
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        m, mc, sp, sb, sa, ob, nq = day_bars(
            ts, d["best_bid"].to_numpy(), d["best_ask"].to_numpy(),
            d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy(), d0)
        if fp.exists():
            F = (pl.scan_parquet(fp)
                 .filter(pl.col("crossed") & (pl.col("dt") == dt))
                 .select("ts", "px", "sz", "side").collect().sort("ts"))
            tr, bv, sv, nt = day_trades(
                F["ts"].cast(pl.Int64).to_numpy(), F["px"].to_numpy(),
                F["sz"].to_numpy(), (F["side"] == "B").to_numpy(), d0)
        else:
            tr = bv = sv = np.full(NMIN, np.nan)
            nt = np.zeros(NMIN)
        out.append(pl.DataFrame({
            "dt": [dt] * NMIN, "minute": np.arange(NMIN, dtype=np.int32),
            "mid": m, "micro": mc, "trade": tr, "spread_bp": sp,
            "qb": sb, "qa": sa, "obi": ob,
            "n_quote": nq.astype(np.int32), "n_trade": np.asarray(nt, np.int32),
            "buy_vol": bv, "sell_vol": sv}))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True, file=sys.stderr)
    return pl.concat(out)


def from_dram():
    """xyz:DRAM は 1 秒格子の表しか無いので、その最後の点を分の終値にする。"""
    t = (pl.scan_parquet(DATA / "DRAM" / "microprice_1s.parquet")
         .select("ts", "best_bid", "best_ask", "bid_sz", "ask_sz",
                 "mid", "microprice", "spread_bp", "stale_ns", "dt")
         .collect().sort("ts"))
    days = sorted(t["dt"].unique().to_list())
    out = []
    for k, dt in enumerate(days):
        d = t.filter(pl.col("dt") == dt)
        ts = d["ts"].cast(pl.Int64).to_numpy()
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        # 1 秒格子は前の値を持ち越してあるので、stale_ns で「その分に本当に
        # 気配が動いたか」を判定する(持ち越しは欠測にする)。
        grid = d0 + (np.arange(NMIN, dtype=np.int64) + 1) * 60_000_000_000
        j = np.searchsorted(ts, grid, side="right") - 1
        ok = j >= 0
        j = np.maximum(j, 0)
        st = d["stale_ns"].to_numpy()[j]
        fresh = ok & (st < 60_000_000_000)      # 1 分以内に更新があった
        g = lambda c: np.where(fresh, d[c].to_numpy()[j], np.nan)  # noqa: E731
        out.append(pl.DataFrame({
            "dt": [dt] * NMIN, "minute": np.arange(NMIN, dtype=np.int32),
            "mid": g("mid"), "micro": g("microprice"),
            "trade": np.full(NMIN, np.nan),
            "spread_bp": np.where(ok, d["spread_bp"].to_numpy()[j], np.nan),
            "qb": np.where(ok, d["bid_sz"].to_numpy()[j], np.nan),
            "qa": np.where(ok, d["ask_sz"].to_numpy()[j], np.nan),
            "obi": np.full(NMIN, np.nan),
            "n_quote": np.zeros(NMIN, np.int32),
            "n_trade": np.zeros(NMIN, np.int32),
            "buy_vol": np.full(NMIN, np.nan),
            "sell_vol": np.full(NMIN, np.nan)}))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True, file=sys.stderr)
    B = pl.concat(out)
    q = B["qb"].to_numpy(), B["qa"].to_numpy()
    tot = np.maximum(q[0] + q[1], 1e-12)
    return B.with_columns(obi=pl.Series((q[0] - q[1]) / tot))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    B = from_dram() if a.coin == "xyz:DRAM" else from_bbo(tag)
    B.write_parquet(DATA / f"mrev_bars_{tag}.parquet", compression="zstd")
    m = B["mid"].to_numpy()
    print(f"[mrev] {a.coin}: {B['dt'].n_unique()} 日 × {NMIN} 分 = {B.height:,} 行 / "
          f"mid が埋まった分 {np.isfinite(m).sum():,}"
          f"({np.isfinite(m).mean()*100:.1f}%)")
    print(f"       中央スプレッド {float(B['spread_bp'].median()):.2f} bp / "
          f"1 分あたり気配更新 中央 {float(B['n_quote'].median()):.0f} 件")


if __name__ == "__main__":
    main()
