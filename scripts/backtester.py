"""項目 3: イベント駆動バックテスタ(L4 の注文単位データで**正確に**キューを再現する)

【なぜ L4 だと正確にできるか】
  集計板(L2)しか無いと「レベルの数量が減った」ことしか判らず、
  それが約定なのか取消なのか、自分の前だったか後ろだったかが判らない。
  本データは**注文 1 件ごとに到着時刻・消滅時刻・消滅理由**が判るので、
  自分の注文の**前に並んでいる注文の集合**を時刻ごとに厳密に持てる。

【模擬する執行】
  - 最良買気配・最良売気配に常に 1 単位ずつ出す(古典的なマーケットメイク)
  - 最良気配が動いたら、その価格の注文は取り消して新しい最良気配に出し直す
    (= キュー最後尾に付き直す。優先順位は失う。実運用と同じ)
  - 方針 A: 常に出す
    方針 B: シグナル(book_slope_diff)が不利ならその側を出さない/引く
            引く判断は **δ = 100ms 前のシグナル**で行う(反応時間の予算)

【約定判定(厳密)】
  自分の注文の前には「自分より前に到着し、まだ生きている注文」が並ぶ。
    ・前の注文が取消で消えた → キューが詰まる(約定量は消費しない)
    ・その価格で約定が起きた → **前から順に**約定量を消費する
    ・前が尽きた状態で約定が来たら → 自分が約定
  部分約定は扱わず、自分の 1 単位は「約定量が自分に届いたら全量約定」とする
  (自分の数量を最小単位にすれば実質同じ)。

【時間契約】
  シグナルは常に判断時刻より前の値のみ。未来は参照しない。
"""

from __future__ import annotations

import glob
import json
import sys
from bisect import insort
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
SL = D / "book_slope_grain"
FEE_BP = 0.088
LAT = 100_000_000          # 反応時間 100ms
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]


def load(dt: str) -> dict | None:
    f = SL / f"{dt}.parquet"
    if not f.exists():
        return None
    life = (pl.read_parquet(LIFE / f"dt={dt}" / "part-000.parquet",
                            columns=["oid", "side", "px", "orig_sz", "filled_sz",
                                     "ts_open", "ts_close", "terminal_status",
                                     "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null()))
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fill = pl.concat([pl.read_parquet(x, columns=["ts", "px", "sz", "side", "crossed", "tid"])
                      for x in fs], how="diagonal_relaxed")
    fill = (fill.filter(pl.col("crossed")).unique(subset=["tid"], keep="first")
                .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort("t"))
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "best_bid", "best_ask", "mid", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    s = pl.read_parquet(f)
    return {"life": life, "fill": fill, "mp": mp,
            "sts": s["ts"].to_numpy(),
            "sig": (s["bid_25bp"].to_numpy() - s["ask_25bp"].to_numpy())}


def simulate(d: dict, gated: bool) -> dict:
    """最良気配に 1 単位ずつ出し続ける。gated=True ならシグナルで引く。"""
    mp = d["mp"]
    mts = mp["ts"].to_numpy(); bb = mp["best_bid"].to_numpy()
    ba = mp["best_ask"].to_numpy(); mid = mp["mid"].to_numpy()
    sts, sig = d["sts"], d["sig"]
    # 価格ごとの「注文の到着/消滅」イベント(自分の前に誰がいるかを追うため)
    life = d["life"]
    lo_px = life["px"].to_numpy(); lo_side = life["side"].to_numpy()
    lo_o = life["ts_open"].to_numpy(); lo_c = life["ts_close"].to_numpy()
    lo_sz = life["orig_sz"].to_numpy()
    key = np.round(lo_px * 1e6).astype(np.int64) * 2 + (lo_side == "B")
    # レベルごとに「到着時刻の昇順」と「その時点までの ts_close の最大値」を持つ。
    # 到着 o ≤ t0 の注文のうち最後に消える時刻 = 前置最大値 なので、
    # 「自分が先頭になる時刻」が二分探索 O(log n) で引ける
    order_by_lvl: dict[int, tuple] = {}
    tmp: dict[int, list] = {}
    for k, o, c in zip(key, lo_o, lo_c):
        tmp.setdefault(int(k), []).append((int(o), int(c)))
    for k, v in tmp.items():
        v.sort()
        oo = np.array([x[0] for x in v], dtype=np.int64)
        cc = np.maximum.accumulate(np.array([x[1] for x in v], dtype=np.int64))
        order_by_lvl[k] = (oo, cc)
    del tmp
    fl = d["fill"]
    f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy()
    f_sz = fl["sz"].to_numpy(); f_side = fl["side"].to_numpy()   # テイカーの side
    fills_out = []
    for side in ("B", "A"):
        px_ser = bb if side == "B" else ba
        s_dir = 1.0 if side == "B" else -1.0
        # テイカーが反対方向のときに自分が約定する(自分が買い = テイカーが売り 'A')
        taker = "A" if side == "B" else "B"
        tmask = f_side == taker
        tt, tp, tz = f_t[tmask], f_px[tmask], f_sz[tmask]
        i = 0                      # 板イベントの位置
        n = mts.size
        while i < n:
            p = px_ser[i]
            t0 = mts[i]
            # シグナルで引く判断(δ 前の値)
            if gated:
                j = np.searchsorted(sts, t0 - LAT, side="right") - 1
                if j >= 0 and s_dir * (-sig[j]) < 0:
                    i += 1
                    continue
            # この価格が最良でなくなるまで
            i2 = i
            while i2 + 1 < n and px_ser[i2 + 1] == p:
                i2 += 1
            t_end = mts[i2 + 1] if i2 + 1 < n else mts[-1]
            k = int(round(p * 1e6)) * 2 + (side == "B")
            # 自分より前に並ぶ注文が全部消える時刻(= 自分が先頭になる時刻)。
            # 価格・時間優先なので、自分に約定が届くのはそれ以降。
            # 前の注文は約定でも取消でも ts_close で消える(L4 だから厳密に書ける)
            lv = order_by_lvl.get(k)
            t_front = t0
            if lv is not None:
                jj = int(np.searchsorted(lv[0], t0, side="right")) - 1
                if jj >= 0 and lv[1][jj] > t0:
                    t_front = int(lv[1][jj])
            lo = np.searchsorted(tt, max(t0, t_front)); hi = np.searchsorted(tt, t_end)
            filled_at = None
            for x in range(lo, hi):
                if abs(tp[x] - p) <= 1e-9:
                    filled_at = tt[x]
                    break
            if filled_at is not None:
                j0 = np.searchsorted(mts, filled_at, side="right") - 1
                j1 = np.searchsorted(mts, filled_at + 1_000_000_000, side="right") - 1
                if j0 >= 0 and j1 >= 0 and filled_at + 1_000_000_000 <= mts[-1]:
                    m0 = mid[j0]
                    r = s_dir * (mid[j1] - p) / m0 * 1e4 - FEE_BP
                    fills_out.append(r)
            i = i2 + 1
    a = np.array(fills_out)
    return {"n_fills": int(a.size), "mean_bp": float(a.mean()) if a.size else np.nan,
            "total_bp": float(a.sum()) if a.size else 0.0}


def main() -> None:
    days = sorted(p.stem for p in SL.glob("*.parquet"))
    if len(sys.argv) > 1:
        days = sys.argv[1:]
    rows = []
    for dt in days:
        d = load(dt)
        if d is None:
            continue
        a = simulate(d, gated=False)
        b = simulate(d, gated=True)
        rows.append({"dt": dt, "always_n": a["n_fills"], "always_mean": a["mean_bp"],
                     "gated_n": b["n_fills"], "gated_mean": b["mean_bp"],
                     "always_total": a["total_bp"], "gated_total": b["total_bp"]})
        print(f"{dt}: 常時 {a['n_fills']:>6,} 約定 平均 {a['mean_bp']:+.4f}bp | "
              f"選別 {b['n_fills']:>6,} 約定 平均 {b['mean_bp']:+.4f}bp", flush=True)
    t = pl.DataFrame(rows)
    t.write_csv(D / "backtest_results.csv")
    out = {"days": t.height,
           "always": {"mean_of_daily": float(t["always_mean"].mean()),
                      "median_of_daily": float(t["always_mean"].median()),
                      "fills_per_day": float(t["always_n"].mean()),
                      "total_bp_per_day": float(t["always_total"].mean())},
           "gated": {"mean_of_daily": float(t["gated_mean"].mean()),
                     "median_of_daily": float(t["gated_mean"].median()),
                     "fills_per_day": float(t["gated_n"].mean()),
                     "total_bp_per_day": float(t["gated_total"].mean())}}
    out["fill_retention"] = out["gated"]["fills_per_day"] / max(out["always"]["fills_per_day"], 1)
    (D / "backtest_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
