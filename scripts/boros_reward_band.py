r"""特徴量 55 — メイカー報酬帯の内外を算出する。

=============================================================================
帯の定義(Boros OpenAPI の原文。推測ではない)
=============================================================================
  `incentiveRange`:
    "Half-width of the incentive band around the mid implied APR for this side,
     expressed as a decimal APR (e.g. `0.005` = ±50bps).
     Resting size within this band qualifies."

  つまり side ごとに半幅 R_side(APR 小数)が与えられ、
  **mid からの距離が R_side 以内に置かれている残量**が報酬対象になる。
  約定ではなく **resting(板に載っている量)** が対象で、トラックは毎時。

  本スクリプトでは pp 単位に直して使う: R_pp = incentiveRange × 100。
  long は bid 側(mid より下)、short は ask 側(mid より上)なので、
    long  が対象 ⇔ mid − rate_i ≤ R_long_pp
    short が対象 ⇔ rate_i − mid ≤ R_short_pp

=============================================================================
★二つの版を出す理由
=============================================================================
  (a) 静的帯 — API の現在値を過去に当てる
      エンドポイントは marketId しか受け取らず**現在エポックしか返さない**
      (OpenAPI で確認)。窓(2026-05-04〜08-10)当時の帯は取得できない。
      よって「帯は概ね変わっていない」という**仮定**の下で当てる。
      エポックは週次(実測 2026-08-21 00:00Z)なので、この仮定は弱くない。

  (b) 実証帯 — 発注密度の不連続から帯を検出する
      報酬が「帯の内側の残量」に払われるなら、合理的なメイカーは
      **帯の内側ぎりぎり**に置く(約定リスクを最小にしつつ報酬を取る)。
      したがって mid からの距離の分布に**内側での山と外側での崖**が出るはず。
      これは仮定を要さず、窓当時のデータだけで測れる。
      (a) が正しければ (b) の検出位置は (a) の帯と一致するはずで、相互検証になる。

=============================================================================
出力
=============================================================================
  data/band_book_{mid}.parquet
      毎イベントの帯内残量。event_book_{mid}.parquet と行が 1 対 1 で対応する
      (同じ出力条件・同じ ev_i)。
        ib_long / ib_short          帯内の数量[YU]
        ibn_long / ibn_short        帯内のレベル数
        ib_share_long / _short      帯内 ÷ 片側総量
  data/band_place_{mid}.parquet
      発注ごとの mid からの距離。バンチング検出と口座別の行動に使う。
        dist_pp   自分の側から見た mid からの距離[pp](正 = mid から遠い)
        in_band   静的帯の内側か
        is_top1   発注量 1 位の口座か
        lifetime_s / terminal   その注文の顛末
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Boros-history")                 # ★読み取り専用
OUT = Path(__file__).resolve().parent.parent / "data"
WAD = 1e18


def market_dir(cat: str, mid: int) -> Path | None:
    for d in (SRC / "data" / "parquet" / cat).glob(f"{mid}-*"):
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
    raw = pl.concat([pl.read_parquet(f, columns=["blockNumber", "blockTimestamp"])
                     for f in fs], how="diagonal_relaxed").sort("blockNumber")
    b, t = raw["blockNumber"].to_list(), raw["blockTimestamp"].to_list()
    if len(b) < 2:
        return None

    def f(blk: int) -> float:
        i = bisect.bisect_left(b, blk)
        if i <= 0:
            r = (t[1] - t[0]) / (b[1] - b[0]); return t[0] + (blk - b[0]) * r
        if i >= len(b):
            r = (t[-1] - t[-2]) / (b[-1] - b[-2]); return t[-1] + (blk - b[-1]) * r
        w = (blk - b[i - 1]) / (b[i] - b[i - 1])
        return t[i - 1] + w * (t[i] - t[i - 1])
    return f, b[-1]


def run(mid: int, tick_step: int, maturity: int,
        r_long: float, r_short: float, top1: str | None):
    ck = block_clock(mid)
    if ck is None:
        raise SystemExit("断面が無い")
    clock, last_real_block = ck
    ev = (pl.read_parquet(SRC / f"data/chain/events/marketId={mid}.parquet")
            .sort(["block_number", "log_index"]))

    def rate(t: int) -> float:
        return (1.00005 ** (t * tick_step) - 1.0) * 100.0

    RL, RS = r_long * 100.0, r_short * 100.0        # APR 小数 → pp
    live: dict[int, int] = {}
    meta: dict[int, tuple] = {}                     # oid -> (account, ts, dist_pp, in_band)
    depth: list[dict[int, int]] = [{}, {}]

    def touch(side: int, tick: int, delta: int) -> None:
        d = depth[side]
        v = d.get(tick, 0) + delta
        if v <= 0:
            d.pop(tick, None)
        else:
            d[tick] = v

    def band(side: int, best_tick: int, mid_rate: float):
        """(帯内数量[YU], 帯内レベル数, 片側総量[YU])"""
        d = depth[side]
        if not d:
            return 0.0, 0, 0.0
        R = RL if side == 0 else RS
        ib = 0
        n = 0
        tot = 0
        for tk, sz in d.items():
            tot += sz
            dist = (mid_rate - rate(tk)) if side == 0 else (rate(tk) - mid_rate)
            if dist <= R:                     # 負(mid を跨ぐ)も帯内として扱う
                ib += sz
                n += 1
        return ib / WAD, n, tot / WAD

    rows, places = [], []
    for ev_i, r in enumerate(ev.iter_rows(named=True)):
        k, blk = r["event"], r["block_number"]
        # 現在の mid(発注の距離を測るために、イベント処理の**前**に取る)
        mid_before = None
        if depth[0] and depth[1]:
            bl0, bs0 = max(depth[0]), min(depth[1])
            rl0, rs0 = rate(bl0), rate(bs0)
            if rs0 > rl0:
                mid_before = (rl0 + rs0) / 2

        if k == "placed":
            oid = r["order_id"]; sz = int(r["size_raw"])
            sd, tk = unpack(oid)
            live[oid] = sz
            ts = clock(blk)
            dist = np.nan; inb = None
            if mid_before is not None:
                dist = (mid_before - rate(tk)) if sd == 0 else (rate(tk) - mid_before)
                inb = bool(dist <= (RL if sd == 0 else RS))
            meta[oid] = (r["account"], ts, dist, inb, sd, sz / WAD)
            touch(sd, tk, sz)
        elif k in ("cancelled", "forced_cancelled"):
            oid = r["order_id"]; sz = live.pop(oid, None)
            if sz is not None:
                sd, tk = unpack(oid); touch(sd, tk, -sz)
            m = meta.pop(oid, None)
            if m is not None:
                places.append((m[0], m[2], m[3], m[4], m[5],
                               clock(blk) - m[1], "cancelled"))
        elif k in ("filled", "oob_purged"):
            lo, hi = r["from_id"], r["to_id"]
            for a in [x for x in live if lo <= x <= hi]:
                sz = live.pop(a); sd, tk = unpack(a); touch(sd, tk, -sz)
                m = meta.pop(a, None)
                if m is not None:
                    places.append((m[0], m[2], m[3], m[4], m[5],
                                   clock(blk) - m[1], "filled"))
        elif k == "partially_filled":
            oid = r["order_id"]
            if oid in live:
                dsz = int(r["size_raw"]); sd, tk = unpack(oid)
                live[oid] -= dsz; touch(sd, tk, -dsz)
                if live[oid] <= 0:
                    live.pop(oid)
                    m = meta.pop(oid, None)
                    if m is not None:
                        places.append((m[0], m[2], m[3], m[4], m[5],
                                       clock(blk) - m[1], "filled"))

        # --- このイベント直後の帯内残量 ---
        if not depth[0] or not depth[1]:
            continue
        bl, bs = max(depth[0]), min(depth[1])
        r_l, r_s = rate(bl), rate(bs)
        if r_s <= r_l:
            continue
        mrate = (r_l + r_s) / 2
        ibl, nbl, dl = band(0, bl, mrate)
        ibs, nbs, ds = band(1, bs, mrate)
        rows.append({
            "ev_i": ev_i, "ts": clock(blk), "block": blk,
            "ib_long": ibl, "ib_short": ibs,
            "ibn_long": nbl, "ibn_short": nbs,
            "ib_share_long": ibl / dl if dl > 0 else np.nan,
            "ib_share_short": ibs / ds if ds > 0 else np.nan,
            "extrapolated": blk > last_real_block,
        })

    # 窓終端で生存している注文も右打ち切りとして記録する
    for oid, m in meta.items():
        places.append((m[0], m[2], m[3], m[4], m[5], np.nan, "censored"))

    P = pl.DataFrame(
        {"account": [p[0] for p in places], "dist_pp": [p[1] for p in places],
         "in_band": [p[2] for p in places], "side": [p[3] for p in places],
         "size": [p[4] for p in places], "lifetime_s": [p[5] for p in places],
         "terminal": [p[6] for p in places]})
    if top1 is not None and P.height:
        P = P.with_columns((pl.col("account") == top1).alias("is_top1"))
    else:
        P = P.with_columns(pl.lit(False).alias("is_top1"))
    return pl.DataFrame(rows), P.with_columns(pl.lit(mid).alias("market"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="163")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = {m["marketId"]: m for m in
           json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))}
    inc = pl.read_parquet(OUT / "incentive_range.parquet")
    band = {r["market"]: (r["range_long"], r["range_short"])
            for r in inc.iter_rows(named=True) if r["ok"]}
    panel = pl.read_parquet(OUT / "accounts_panel.parquet")
    top1 = {int(m): g.sort("vol_placed", descending=True)["account"][0]
            for (m,), g in panel.group_by(["market"])}

    for mid in [int(x) for x in a.markets.split(",")]:
        pb = OUT / f"band_book_{mid}.parquet"
        if pb.exists() and not a.force:
            print(f"market {mid}: 済み — 飛ばす", flush=True); continue
        m = cfg.get(mid); br = band.get(mid)
        if m is None or br is None or br[0] is None:
            print(f"market {mid}: 設定/帯が無い — 飛ばす", flush=True); continue
        try:
            B, P = run(mid, m["tickStep"], m["maturity"], br[0], br[1], top1.get(mid))
        except SystemExit as e:
            print(f"market {mid}: 飛ばす({e})", flush=True); continue
        except Exception as e:
            print(f"market {mid}: ★失敗 {type(e).__name__}: {e}", flush=True); continue
        if B.height == 0:
            continue
        B.write_parquet(pb)
        P.write_parquet(OUT / f"band_place_{mid}.parquet")
        v = B.filter(~pl.col("extrapolated"))
        print(f"market {mid:>4}: 帯 L={br[0]*100:.3f}pp S={br[1]*100:.3f}pp  "
              f"行 {B.height:,}  帯内シェア 中央 L={v['ib_share_long'].median():.1%} "
              f"S={v['ib_share_short'].median():.1%}  発注 {P.height:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
