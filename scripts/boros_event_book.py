"""Boros のイベント列を再生し、**毎イベント**の板特徴量(book_slope / spread)を出す。

【再生規則】
  `E:\\Boros-history\\scripts\\mm_pnl.py` の `replay()` と同一。あちらは約定と中値だけを
  返すので、ここでは**全イベント時点の板の形**を出す。踏んではいけない罠は 4 つ:

    1. `filled` / `oob_purged` の `from_id`/`to_id` は **1 本ではなく範囲**
    2. 注文 ID は bit63 が立つので **UInt64 必須**(Int64 では溢れる)
    3. **LONG 側は tick がビット反転**(`unpack` が処理)
    4. 数量は **wei の int のまま**足し引きする。float だと残渣が幻の最良気配を作る

【出す量】
  spread_pp       = best_short_rate − best_long_rate            [pp]
  mid_pp          = (best_short_rate + best_long_rate) / 2      [pp]
  book_slope_side = Σ_i |rate_i − best_side| · size_i / Σ_i size_i   [pp]
      その側の流動性が**自分の最良気配からどれだけ離れて置かれているか**の
      数量加重平均。大きいほど「近くが薄い」。DRAM 案件の `book_slope` と同じ定義
      (あちらは bp、ここは金利なので pp)
  book_slope_diff = book_slope_long − book_slope_short
      正 = LONG 側の流動性が遠い = LONG が弱い
  bq_long / bq_short = **その側の最良気配 1 レベルに載っている数量** [YU]
      depth_* が片側の総量なのに対し、こちらは最良レベル単体。
      Pressure = ΔOFM / 反対側の最良数量 を作るのに要る
  ew{τ}_long / ew{τ}_short = **最良気配を最も重くした指数減衰の加重数量** [YU]
      w_i = exp(−d_i / τ)、d_i = |tick_i − tick_best| は**最良からの tick 距離**。
      Q_s = Σ_i w_i · q_i。
      τ → 0 で bq_*(最良 1 レベルのみ)、τ → ∞ で depth_*(片側総量)に一致する
      ので、両者を連続につなぐ 1 パラメータ族になっている。
      ★距離を pp でなく tick で測るのは、pp だと金利水準で刻み幅が変わり
        (幾何級数)市場間で τ の意味が揃わないため(spread_states 報告 §1 と同じ理由)。
  ev_i = その市場の全イベント列における通し番号
      板が片側空・クロスの間は行を出さないので、**連続する 2 行が
      連続イベントとは限らない**。差分を取る側がこれを見て段差を除ける。

【検算】
  公式アーカイブの断面と**最良気配**を突合する(mm_pnl と同じ不変量)。
  注文集合が合っていても気配だけずれる事故が実際にあったため。

【出力】 data/event_book_{marketId}.parquet

【--exclude-top N(頑健性検査)】
  口座パネル(data/accounts_panel.parquet)の発注量上位 N 口座を**板から除いた**
  第 2 の板を同時に再生し、`*_x` 列として出す。
  Boros は上位 1 者が発注量の中央 79.7%(164/188 市場で 50% 超)を占めるので、
  「板の形が金利を予測する」という既存の結論が**その 1 口座の癖を測っていただけ**
  でないかを確かめるために要る。

  ★行の出し入れは**常に全体の板**の条件(両側非空・非クロス)で決める。
    除外板が空/クロスの時点は `*_x` を NaN にする。こうしないと行が揃わず、
    既存の結果と 1 対 1 で比較できない。
  出力は data/event_book_ex_{marketId}.parquet
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Boros-history")                 # ★読み取り専用で扱う
OUT = Path(__file__).resolve().parent.parent / "data"
WAD = 1e18

# 指数減衰の減衰長[tick]。幾何級数で 1〜64 を張る。
# ★結果を見る前に、板の幾何(spread 中央 18 tick)から決めたグリッドである。
#   tau=1 はほぼ最良 1 レベル、tau=64 はほぼ片側総量に相当する。
TAUS = (1, 4, 16, 64)

# exp(−d/τ) を整数 tick 距離で引くための表。tick 距離は整数なので事前計算できる。
# 打ち切りは τ=64 で exp(−2048/64)=1.3e-14 と無視できる水準。
# (10.4M イベント × 約 20 レベル × 4 τ = 8 億回の exp を避けるため)
_DMAX = 2048
EXP_TAB = [[math.exp(-d / t) for d in range(_DMAX + 1)] for t in TAUS]


def market_dir(cat: str, mid: int) -> Path | None:
    base = SRC / "data" / "parquet" / cat
    for d in base.glob(f"{mid}-*"):
        return d
    return None


def unpack(oid: int) -> tuple[int, int]:
    """注文 ID から (side, tick)。LONG(side=0)は tick がビット反転している。"""
    e = (oid >> 40) & 0xFFFF
    side = (oid >> 56) & 1
    if side == 0:
        e = (~e) & 0xFFFF
    t = e ^ 0x8000
    return side, (t - (1 << 16) if t >= (1 << 15) else t)


def block_clock(mid: int):
    d = market_dir("order-book", mid)
    fs = sorted(glob.glob(str(d / "raw" / "*.parquet"))) if d else []
    if not fs:
        return None
    raw = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed").sort("blockNumber")
    b = raw["blockNumber"].to_list()
    t = raw["blockTimestamp"].to_list()
    if len(b) < 2:
        return None

    def f(blk: int) -> float:
        i = bisect.bisect_left(b, blk)
        if i <= 0:
            r = (t[1] - t[0]) / (b[1] - b[0])
            return t[0] + (blk - b[0]) * r
        if i >= len(b):
            r = (t[-1] - t[-2]) / (b[-1] - b[-2])
            return t[-1] + (blk - b[-1]) * r
        w = (blk - b[i - 1]) / (b[i] - b[i - 1])
        return t[i - 1] + w * (t[i] - t[i - 1])

    return f, b[-1], t[-1]


def archive_best(mid: int) -> list[tuple[int, int, int]]:
    d = market_dir("order-book", mid)
    fs = sorted(glob.glob(str(d / "raw" / "*.parquet"))) if d else []
    if not fs:
        return []
    raw = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed").sort("blockNumber")
    out = []
    for r in raw.iter_rows(named=True):
        L, S = r["long"] or [], r["short"] or []
        if L and S:
            out.append((r["blockNumber"], max(e["t"] for e in L), min(e["t"] for e in S)))
    return out


def run(mid: int, tick_step: int, maturity: int, excl: set[str] | None = None) -> pl.DataFrame:
    ck = block_clock(mid)
    if ck is None:
        raise SystemExit(f"market {mid}: 断面が無い")
    clock, last_real_block, _ = ck
    ev = (pl.read_parquet(SRC / f"data/chain/events/marketId={mid}.parquet")
            .sort(["block_number", "log_index"]))

    def rate(t: int) -> float:
        return (1.00005 ** (t * tick_step) - 1.0) * 100.0

    live: dict[int, int] = {}
    depth: list[dict[int, int]] = [{}, {}]
    # 除外板(--exclude-top のときだけ使う)。除外口座の注文を載せない。
    EX = excl or set()
    depth_x: list[dict[int, int]] = [{}, {}]
    is_ex: dict[int, bool] = {}          # order_id -> 除外口座のものか

    def touch(side: int, tick: int, delta: int, skip_x: bool = False) -> None:
        d = depth[side]
        v = d.get(tick, 0) + delta
        if v <= 0:
            d.pop(tick, None)
        else:
            d[tick] = v
        if EX and not skip_x:
            dx = depth_x[side]
            vx = dx.get(tick, 0) + delta
            if vx <= 0:
                dx.pop(tick, None)
            else:
                dx[tick] = vx

    def slope(side: int, best_tick: int, dep=None) -> tuple[float, float, int, list[float], float]:
        """(数量加重の平均距離[pp], 総数量[YU], レベル数, τ ごとの加重数量[YU],
           数量加重の平均 tick 距離)

        1 回の走査で slope と全 τ の指数加重を同時に作る(レベル数だけ回る)。
        """
        d = (dep or depth)[side]
        if not d:
            return (float("nan"), 0.0, 0, [0.0] * len(TAUS), float("nan"))
        b = rate(best_tick)
        num = 0.0
        tot = 0
        dnum = 0.0
        ew = [0.0] * len(TAUS)
        for tk, sz in d.items():
            num += abs(rate(tk) - b) * sz
            tot += sz
            dt_ = abs(tk - best_tick)          # ★最良からの tick 距離
            dnum += dt_ * sz
            if dt_ <= _DMAX:
                for j in range(len(TAUS)):
                    ew[j] += EXP_TAB[j][dt_] * sz
        return (num / tot if tot else float("nan"), tot / WAD, len(d),
                [v / WAD for v in ew], dnum / tot if tot else float("nan"))

    snap = archive_best(mid)
    snap_i = snap_bad = snap_chk = 0
    rows: list[dict] = []

    for ev_i, r in enumerate(ev.iter_rows(named=True)):
        while snap_i < len(snap) and snap[snap_i][0] < r["block_number"]:
            sb, a_bl, a_bs = snap[snap_i]
            if depth[0] and depth[1]:
                snap_chk += 1
                if (max(depth[0]), min(depth[1])) != (a_bl, a_bs):
                    snap_bad += 1
            snap_i += 1

        k, blk = r["event"], r["block_number"]
        if k == "placed":
            oid = r["order_id"]
            sz = int(r["size_raw"])
            sd, tk = unpack(oid)
            live[oid] = sz
            ex = bool(EX) and r["account"] in EX
            is_ex[oid] = ex
            touch(sd, tk, sz, skip_x=ex)
        elif k in ("cancelled", "forced_cancelled"):
            oid = r["order_id"]
            sz = live.pop(oid, None)
            if sz is not None:
                sd, tk = unpack(oid)
                touch(sd, tk, -sz, skip_x=is_ex.pop(oid, False))
        elif k in ("filled", "oob_purged"):
            lo, hi = r["from_id"], r["to_id"]          # ★範囲であって 1 本ではない
            for a in [x for x in live if lo <= x <= hi]:
                sz = live.pop(a)
                sd, tk = unpack(a)
                touch(sd, tk, -sz, skip_x=is_ex.pop(a, False))
        elif k == "partially_filled":
            oid = r["order_id"]
            if oid in live:
                d = int(r["size_raw"])
                sd, tk = unpack(oid)
                live[oid] -= d
                touch(sd, tk, -d, skip_x=is_ex.get(oid, False))
                if live[oid] <= 0:
                    live.pop(oid)
                    is_ex.pop(oid, None)

        # --- このイベント直後の板の形 ---
        if not depth[0] or not depth[1]:
            continue
        bl, bs = max(depth[0]), min(depth[1])
        r_l, r_s = rate(bl), rate(bs)
        if r_s <= r_l:                                  # クロスは捨てる(DRAM 案件と同じ規約)
            continue
        sl_l, dep_l, nl_l, ew_l, td_l = slope(0, bl)
        sl_s, dep_s, nl_s, ew_s, td_s = slope(1, bs)
        ts = clock(blk)
        # --- 除外板(--exclude-top のときだけ)---
        # 行の出し入れは全体の板で決めているので、除外板が空/クロスなら NaN を置く。
        xc: dict = {}
        if EX:
            nan = float("nan")
            if depth_x[0] and depth_x[1]:
                blx, bsx = max(depth_x[0]), min(depth_x[1])
                rlx, rsx = rate(blx), rate(bsx)
                if rsx > rlx:
                    slx_l, dpx_l, nlx_l, ewx_l, tdx_l = slope(0, blx, depth_x)
                    slx_s, dpx_s, nlx_s, ewx_s, tdx_s = slope(1, bsx, depth_x)
                    xc = {"best_long_x": rlx, "best_short_x": rsx,
                          "spread_pp_x": rsx - rlx, "mid_pp_x": (rsx + rlx) / 2,
                          "slope_long_x": slx_l, "slope_short_x": slx_s,
                          "slope_diff_x": slx_l - slx_s,
                          "depth_long_x": dpx_l, "depth_short_x": dpx_s,
                          "bq_long_x": depth_x[0][blx] / WAD,
                          "bq_short_x": depth_x[1][bsx] / WAD,
                          "n_lv_long_x": nlx_l, "n_lv_short_x": nlx_s,
                          "tickdist_long_x": tdx_l, "tickdist_short_x": tdx_s,
                          **{f"ew{t}_long_x": ewx_l[j] for j, t in enumerate(TAUS)},
                          **{f"ew{t}_short_x": ewx_s[j] for j, t in enumerate(TAUS)}}
            if not xc:
                xc = {c: nan for c in
                      ["best_long_x", "best_short_x", "spread_pp_x", "mid_pp_x",
                       "slope_long_x", "slope_short_x", "slope_diff_x",
                       "depth_long_x", "depth_short_x", "bq_long_x", "bq_short_x",
                       "tickdist_long_x", "tickdist_short_x"]}
                xc.update({"n_lv_long_x": 0, "n_lv_short_x": 0})
                xc.update({f"ew{t}_long_x": nan for t in TAUS})
                xc.update({f"ew{t}_short_x": nan for t in TAUS})
        rows.append({
            **xc,
            "ts": ts, "block": blk, "event": k, "ev_i": ev_i,
            "best_long": r_l, "best_short": r_s,
            # 最良レベル 1 本の数量。depth[side] は wei の int なので WAD で割る
            "bq_long": depth[0][bl] / WAD, "bq_short": depth[1][bs] / WAD,
            "spread_pp": r_s - r_l, "mid_pp": (r_s + r_l) / 2,
            "slope_long": sl_l, "slope_short": sl_s,
            "slope_diff": sl_l - sl_s,
            "depth_long": dep_l, "depth_short": dep_s,
            "n_lv_long": nl_l, "n_lv_short": nl_s,
            **{f"ew{t}_long": ew_l[j] for j, t in enumerate(TAUS)},
            **{f"ew{t}_short": ew_s[j] for j, t in enumerate(TAUS)},
            "tickdist_long": td_l, "tickdist_short": td_s,
            "ttm_years": max(maturity - ts, 0) / (365 * 86400),
            "extrapolated": blk > last_real_block,
        })

    print(f"  [不変量] 最良気配をアーカイブと突合: {snap_chk:,} 断面中 "
          f"不一致 {snap_bad:,}" + ("  ★要調査" if snap_bad else "  OK"))
    return pl.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="163")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--exclude-top", type=int, default=0,
                    help="発注量上位 N 口座を除いた板も同時に再生する(頑健性検査)")
    a = ap.parse_args()
    top_by_market: dict[int, set[str]] = {}
    if a.exclude_top:
        panel = pl.read_parquet(OUT / "accounts_panel.parquet")
        for (m,), g in panel.group_by(["market"]):
            top = (g.sort("vol_placed", descending=True)
                    .head(a.exclude_top)["account"].to_list())
            top_by_market[int(m)] = set(top)
        print(f"上位 {a.exclude_top} 口座を除外する市場: {len(top_by_market)}")
    cfg = {m["marketId"]: m for m in
           json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for mid in [int(x) for x in a.markets.split(",")]:
        p = OUT / (f"event_book_ex_{mid}.parquet" if a.exclude_top
                   else f"event_book_{mid}.parquet")
        if p.exists() and not a.force:               # 再開可能
            print(f"market {mid}: 済み — 飛ばす", flush=True)
            continue
        m = cfg.get(mid)
        if m is None:
            print(f"market {mid}: config に無い — 飛ばす", flush=True)
            continue
        print(f"\n=== market {mid} {m['symbol']} ===", flush=True)
        try:
            d = run(mid, m["tickStep"], m["maturity"],
                    top_by_market.get(mid) if a.exclude_top else None)
        except SystemExit as e:
            print(f"  飛ばす: {e}", flush=True)
            continue
        except Exception as e:
            print(f"  ★失敗 {type(e).__name__}: {e}", flush=True)
            continue
        if d.height == 0:
            print("  行が無い — 飛ばす", flush=True)
            continue
        d.write_parquet(p)
        v = d.filter(~pl.col("extrapolated"))
        print(f"  イベント時点 {d.height:,} 行(外挿区間を除くと {v.height:,})")
        print(f"  spread_pp   中央 {v['spread_pp'].median():.4f}  "
              f"p10 {v['spread_pp'].quantile(0.1):.4f}  p90 {v['spread_pp'].quantile(0.9):.4f}")
        print(f"  slope_long  中央 {v['slope_long'].median():.4f}   "
              f"slope_short 中央 {v['slope_short'].median():.4f}")
        print(f"  slope_diff  中央 {v['slope_diff'].median():+.4f}  "
              f"sd {v['slope_diff'].std():.4f}")
        print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
