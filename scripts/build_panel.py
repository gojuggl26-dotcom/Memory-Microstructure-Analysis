"""Lasso 回帰用のパネル(1 秒グリッド)を作る。

★時間契約(CLAUDE.md「厳禁事項: ルックアヘッドバイアス」に従う)

    説明変数 x : 時刻 T までに確定している情報だけ
                 - 窓集計は  [T−Δ, T)  … 窓が閉じるのが T なので T で判る
                 - 板の状態は T 時点の直近値(backward asof)
    目的変数 y : r_t = ( log mid(T+Δ) − log mid(T) ) × 10⁴   [bp]
                 … x が確定した T から先の 1 窓分だけ

    x の期間と y の期間は T で接し、重ならない。負のシフトは y の生成にしか使わない。

Δ = 1 秒。列は以下。

  板の状態(T 時点)          m_bp, spread_bp, depth_one_level, obi_2/3/5/10,
                             log_depth, stale_ns
  窓集計 [T−Δ, T)           voi_sum, ofi_sum, n_events,
                             oi(約定インバランス), vol_trade, n_trades,
                             lambda_cancel_bid/ask, cancel_diff_n, cancel_diff_vol,
                             lambda_new_buy/sell, new_intensity_diff,
                             q_ahead_new_bid/ask(新規注文の平均キュー位置)
  モメンタム統制             r_prev = ( log mid(T) − log mid(T−Δ) ) × 10⁴
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
STEP = 1_000_000_000
NS_DAY = 86_400_000_000_000
NON_RESTING_TIF = {"Ioc", "FrontendMarket", "LiquidationMarket"}


def day_grid(dt: str) -> np.ndarray:
    t0 = int(datetime.strptime(dt, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    return np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)


def bucket_sum(ts: np.ndarray, val: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """窓 [T−Δ, T) の合計。T に立つので「T で確定」を満たす。"""
    idx = np.searchsorted(grid, ts, side="right")      # ts < T の最初の T の位置
    out = np.zeros(grid.size + 1)
    np.add.at(out, np.clip(idx, 0, grid.size), val)
    return out[: grid.size]


def build_day(dt: str) -> pl.DataFrame | None:
    grid = day_grid(dt)
    n = grid.size

    # ---- 板の状態(T 時点の直近値)と目的変数 ----
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "mid", "microprice", "spread_bp",
                                   "bid_sz", "ask_sz", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    if mp.height < 5000:
        return None
    ts = mp["ts"].to_numpy()
    mid = mp["mid"].to_numpy()
    lm = np.log(mid)

    def last_at(u: np.ndarray) -> np.ndarray:
        j = np.searchsorted(ts, u, side="right") - 1
        return np.where((j >= 0) & (u >= ts[0]), j, -1)

    j_now = last_at(grid)
    ok = j_now >= 0
    j_fwd = last_at(grid + STEP)
    j_prev = last_at(grid - STEP)
    valid = ok & (j_fwd >= 0) & (grid + STEP <= ts[-1])

    y = np.full(n, np.nan)
    y[valid] = (lm[j_fwd[valid]] - lm[j_now[valid]]) * 1e4        # ★ T から T+Δ
    r_prev = np.full(n, np.nan)
    m_prev = valid & (j_prev >= 0)
    r_prev[m_prev] = (lm[j_now[m_prev]] - lm[j_prev[m_prev]]) * 1e4

    jj = np.clip(j_now, 0, ts.size - 1)
    micro = mp["microprice"].to_numpy()
    bsz, asz = mp["bid_sz"].to_numpy(), mp["ask_sz"].to_numpy()
    cols = {
        "m_bp": (mid[jj] - micro[jj]) / mid[jj] * 1e4,
        "spread_bp": mp["spread_bp"].to_numpy()[jj],
        "depth_one_level": (bsz[jj] - asz[jj]) / (bsz[jj] + asz[jj]),
        "log_depth": np.log(bsz[jj] + asz[jj]),
        "stale_ns": (grid - ts[jj]).astype(np.float64),
    }

    # ---- OBI の深い段(T 時点の直近値) ----
    ob = (pl.read_parquet(D / f"obi/dt={dt}/part-000.parquet",
                          columns=["ts", "obi_2", "obi_3", "obi_5", "obi_10",
                                   "best_bid", "best_ask"])
          .filter(pl.col("best_ask") > pl.col("best_bid")).sort("ts"))
    ots = ob["ts"].to_numpy()
    jo = np.clip(np.searchsorted(ots, grid, side="right") - 1, 0, ots.size - 1)
    for c in ("obi_2", "obi_3", "obi_5", "obi_10"):
        cols[c] = ob[c].to_numpy()[jo]

    # ---- 窓集計: 板イベント由来(VOI / OFI / イベント数) ----
    ft = (pl.read_parquet(D / f"features/dt={dt}/part-000.parquet",
                          columns=["ts", "voi", "ofi_cont", "is_crossed"])
          .filter(~pl.col("is_crossed")))
    fts = ft["ts"].to_numpy()
    cols["voi_sum"] = bucket_sum(fts, np.nan_to_num(ft["voi"].to_numpy()), grid)
    cols["ofi_sum"] = bucket_sum(fts, np.nan_to_num(ft["ofi_cont"].to_numpy()), grid)
    cols["n_events"] = bucket_sum(fts, np.ones(fts.size), grid)

    # ---- 窓集計: 約定(oi_grid の 1s は境界が同じなので 1 窓ずらして使う) ----
    og = pl.read_parquet(D / f"oi_grid/step=1s/dt={dt}/part-000.parquet").sort("ts")
    o_ts = og["ts"].to_numpy() + STEP        # 窓 [T−Δ,T) の値を T の行に置く
    pos = np.searchsorted(grid, o_ts)
    inb = (pos < n) & (grid[np.clip(pos, 0, n - 1)] == o_ts)
    for src, dst in (("order_imbalance", "oi"), ("n_trades", "n_trades")):
        v = np.zeros(n)
        v[pos[inb]] = np.nan_to_num(og[src].to_numpy()[inb])
        cols[dst] = v
    v = np.zeros(n)
    v[pos[inb]] = (og["vol_buy"].to_numpy() + og["vol_sell"].to_numpy())[inb]
    cols["vol_trade"] = v

    # ---- 窓集計: フロー強度とキャンセル ----
    fg = pl.read_parquet(D / f"flow_grid/step=1s/dt={dt}/part-000.parquet").sort("ts")
    f_ts = fg["ts"].to_numpy() + STEP
    pos = np.searchsorted(grid, f_ts)
    inb = (pos < n) & (grid[np.clip(pos, 0, n - 1)] == f_ts)
    for c in ("lambda_cancel_bid", "lambda_cancel_ask",
              "lambda_new_buy", "lambda_new_sell", "new_intensity_diff"):
        v = np.zeros(n)
        v[pos[inb]] = np.nan_to_num(fg[c].to_numpy()[inb])
        cols[c] = v
    cg = pl.read_parquet(D / f"cancel_grid/step=1s/dt={dt}/part-000.parquet").sort("ts")
    c_ts = cg["ts"].to_numpy() + STEP
    pos = np.searchsorted(grid, c_ts)
    inb = (pos < n) & (grid[np.clip(pos, 0, n - 1)] == c_ts)
    for c in ("cancel_diff_n", "cancel_diff_vol"):
        v = np.zeros(n)
        v[pos[inb]] = np.nan_to_num(cg[c].to_numpy()[inb].astype(float))
        cols[c] = v

    # ---- 窓集計: 新規注文のキュー位置(側別の平均) ----
    life = (pl.read_parquet(LIFE / f"dt={dt}" / "part-000.parquet",
                            columns=["oid", "side", "ts_open", "tif",
                                     "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(list(NON_RESTING_TIF)).fill_null(False))))
    qi = pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                         columns=["oid", "q_ahead_open", "open_unknown"])
    q = (life.join(qi, on="oid", how="inner")
             .filter((~pl.col("open_unknown")) & pl.col("ts_open").is_not_null()))
    for side, name in (("B", "q_ahead_new_bid"), ("A", "q_ahead_new_ask")):
        sub = q.filter(pl.col("side") == side)
        t_ = sub["ts_open"].to_numpy()
        v_ = np.nan_to_num(sub["q_ahead_open"].to_numpy().astype(float))
        s = bucket_sum(t_, v_, grid)
        c = bucket_sum(t_, np.ones(t_.size), grid)
        cols[name] = np.where(c > 0, s / np.maximum(c, 1), 0.0)

    cols["r_prev"] = r_prev
    out = pl.DataFrame({"ts": grid, "y": y, **{k: v.astype(np.float64)
                                               for k, v in cols.items()}})
    # NaN は null と別物なので is_finite で落とす(y=NaN の行が残ると回帰が壊れる)
    return out.filter(pl.col("y").is_finite() & pl.col("r_prev").is_finite())


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D / "microprice").iterdir() if p.is_dir())
    base = OUT / "data" / "panel_1s"
    base.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, dt in enumerate(days, 1):
        df = build_day(dt)
        if df is None or df.is_empty():
            continue
        d = base / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / "part-000.parquet", compression="zstd")
        total += df.height
        if i % 10 == 0 or i == len(days):
            print(f"[{i:3d}/{len(days)}] {dt} rows={df.height:>7,} total={total:>10,}", flush=True)
    print("done", total)


if __name__ == "__main__":
    main()
