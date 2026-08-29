r"""特徴量 34(板レベル)— 毎イベントの「板に載っている注文の年齢」。

【なぜ板レベルにするか】
  注文ごとの生存時間(`band_place_*`)は**終端が判ってから**しか確定しない。
  予測に使うには「いま板にある注文が、いつから置かれているか」= **年齢**でなければ
  時間契約を満たさない(年齢は時刻 t で確定する)。

【出す量(片側ごと)】
  age_w_side    数量加重の平均年齢[秒] = t − (Σ size·ts_placed)/(Σ size)
  age_best_side 最良レベルに載っている注文の数量加重平均年齢[秒]
  n_ord_side    板に載っている注文の本数(レベル数 n_lv とは別)

【計算量】
  全注文を毎イベント走査すると 10.4M × 約 100 本 = 10 億回になるので、
  **和を逐次更新**する: 片側と (側,tick) ごとに Σsize と Σ(size·ts) を持ち、
  発注で足し、取消・約定で引く。1 イベントあたり O(1)。
  部分約定は size を按分して引く(ts_placed は変えない = 元の注文の年齢を保つ)。

【★数値誤差への対処(実測で踏んだ)】
  素朴に Σ(size·ts) を積むと ts が 1.78e9 なので桁落ちし、
  **年齢が負になる**行が 0.7〜1.5% 出た(年齢は定義上非負)。対策は 2 つ:
    (1) 時刻は市場の最初のブロック時刻 t0 からの相対秒で持つ(桁を 1.78e9 → 1e7 に落とす)
    (2) 10 万イベントごとに live から**厳密に再計算**して累積誤差を切る
  それでも残る微小な負値は 0 に丸め、**丸めた割合を出力**して監査できるようにする。

【出力】 data/age_book_{mid}.parquet(event_book_{mid} と行が 1 対 1、同じ ev_i)
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

SRC = Path("E:/Boros-history")
OUT = Path(__file__).resolve().parent.parent / "data"
WAD = 1e18


def market_dir(cat: str, mid: int):
    for d in (SRC / "data" / "parquet" / cat).glob(f"{mid}-*"):
        return d
    return None


def unpack(oid: int) -> tuple[int, int]:
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


def run(mid: int, tick_step: int) -> pl.DataFrame:
    ck = block_clock(mid)
    if ck is None:
        raise SystemExit("断面が無い")
    clock, last_real_block = ck
    ev = (pl.read_parquet(SRC / f"data/chain/events/marketId={mid}.parquet")
            .sort(["block_number", "log_index"]))

    def rate(t: int) -> float:
        return (1.00005 ** (t * tick_step) - 1.0) * 100.0

    live: dict[int, list] = {}                  # oid -> [size_wei, ts_rel, side, tick]
    T0 = None                                   # 基準時刻(最初のブロック)。桁落ち対策
    depth: list[dict[int, int]] = [{}, {}]
    # 逐次更新する和(float で持つ。size は wei/WAD にしてから足す)
    ssz = [0.0, 0.0]                            # Σ size
    sst = [0.0, 0.0]                            # Σ size·ts_placed
    n_ord = [0, 0]
    lvl: list[dict[int, list]] = [{}, {}]       # (side)[tick] -> [Σsize, Σsize·ts]

    def add(side: int, tick: int, sz: float, ts: float, sign: int) -> None:
        ssz[side] += sign * sz
        sst[side] += sign * sz * ts
        n_ord[side] += sign if sign > 0 else 0
        d = lvl[side].setdefault(tick, [0.0, 0.0])
        d[0] += sign * sz
        d[1] += sign * sz * ts
        if d[0] <= 1e-15:
            lvl[side].pop(tick, None)

    def touch(side: int, tick: int, delta: int) -> None:
        d = depth[side]
        v = d.get(tick, 0) + delta
        if v <= 0:
            d.pop(tick, None)
        else:
            d[tick] = v

    def rebuild() -> None:
        """live から和を厳密に作り直す(累積誤差を切る)。"""
        ssz[0] = ssz[1] = 0.0
        sst[0] = sst[1] = 0.0
        n_ord[0] = n_ord[1] = 0
        lvl[0].clear(); lvl[1].clear()
        for o in live.values():
            sz = o[0] / WAD
            ssz[o[2]] += sz; sst[o[2]] += sz * o[1]; n_ord[o[2]] += 1
            d = lvl[o[2]].setdefault(o[3], [0.0, 0.0])
            d[0] += sz; d[1] += sz * o[1]

    rows = []
    n_clip = 0
    for ev_i, r in enumerate(ev.iter_rows(named=True)):
        k, blk = r["event"], r["block_number"]
        ts_abs = clock(blk)
        if T0 is None:
            T0 = ts_abs
        ts = ts_abs - T0                        # ★相対秒で扱う
        if ev_i and ev_i % 100_000 == 0:
            rebuild()
        if k == "placed":
            oid = r["order_id"]; szw = int(r["size_raw"]); sd, tk = unpack(oid)
            sz = szw / WAD
            live[oid] = [szw, ts, sd, tk]
            touch(sd, tk, szw); add(sd, tk, sz, ts, +1)
        elif k in ("cancelled", "forced_cancelled"):
            oid = r["order_id"]; o = live.pop(oid, None)
            if o is not None:
                touch(o[2], o[3], -o[0]); add(o[2], o[3], o[0] / WAD, o[1], -1)
                n_ord[o[2]] -= 1
        elif k in ("filled", "oob_purged"):
            lo, hi = r["from_id"], r["to_id"]
            for a in [x for x in live if lo <= x <= hi]:
                o = live.pop(a)
                touch(o[2], o[3], -o[0]); add(o[2], o[3], o[0] / WAD, o[1], -1)
                n_ord[o[2]] -= 1
        elif k == "partially_filled":
            oid = r["order_id"]; o = live.get(oid)
            if o is not None:
                dsz = int(r["size_raw"])
                touch(o[2], o[3], -dsz); add(o[2], o[3], dsz / WAD, o[1], -1)
                o[0] -= dsz
                if o[0] <= 0:
                    live.pop(oid, None); n_ord[o[2]] -= 1

        if not depth[0] or not depth[1]:
            continue
        bl, bs = max(depth[0]), min(depth[1])
        if rate(bs) <= rate(bl):
            continue
        rec = {"ev_i": ev_i, "ts": ts_abs, "block": blk,
               "extrapolated": blk > last_real_block}
        for sd, lab, bt in ((0, "long", bl), (1, "short", bs)):
            s = ssz[sd]
            a = (ts - sst[sd] / s) if s > 1e-15 else np.nan
            if np.isfinite(a) and a < 0:
                n_clip += 1; a = 0.0            # 年齢は定義上非負
            rec[f"age_w_{lab}"] = a
            rec[f"n_ord_{lab}"] = max(n_ord[sd], 0)
            d = lvl[sd].get(bt)
            ab = (ts - d[1] / d[0]) if (d and d[0] > 1e-15) else np.nan
            if np.isfinite(ab) and ab < 0:
                n_clip += 1; ab = 0.0
            rec[f"age_best_{lab}"] = ab
        rows.append(rec)
    D = pl.DataFrame(rows)
    D = D.with_columns(pl.lit(n_clip / max(4 * D.height, 1)).alias("clip_rate"))
    return D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="163")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = {m["marketId"]: m for m in
           json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))}
    for mid in [int(x) for x in a.markets.split(",")]:
        p = OUT / f"age_book_{mid}.parquet"
        if p.exists() and not a.force:
            continue
        m = cfg.get(mid)
        if m is None:
            continue
        try:
            d = run(mid, m["tickStep"])
        except SystemExit as e:
            print(f"market {mid}: 飛ばす({e})", flush=True); continue
        except Exception as e:
            print(f"market {mid}: ★失敗 {type(e).__name__}: {e}", flush=True); continue
        if d.height == 0:
            continue
        d.write_parquet(p)
        v = d.filter(~pl.col("extrapolated"))
        print(f"market {mid:>4}: 行 {d.height:,}  加重年齢 中央 "
              f"L={v['age_w_long'].median():.0f}s S={v['age_w_short'].median():.0f}s  "
              f"最良 L={v['age_best_long'].median():.0f}s S={v['age_best_short'].median():.0f}s  "
              f"本数 L={v['n_ord_long'].median():.0f} S={v['n_ord_short'].median():.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
