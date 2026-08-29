"""規模の測定 — 1 件あたりの数量を上げたとき損益がどう劣化するか。

【インパクト模型なしで測れる制約】
  数量を Q に上げても、**実際に流れてきた約定量以上は取れない**。
  価格・時間優先で自分が先頭になった後、到来する約定を
      取れる量 = min(残数量, その約定の数量)
  と順に食っていく。約定が細切れなら Q を上げても埋まらず、
  最良気配が動けば残りは取消になる。
  これが**部分約定による自然な上限**であり、他者の反応を仮定せずに測れる。

  ★v3 は 1 単位の全量約定として扱っていた。本script はこれを実装する。

【在庫上限の扱い】
  Q を上げたのに上限を据え置くと、1 回の約定で即座に上限へ達する。
  「何回ぶんの余裕を持つか」を揃えるため **上限 = INV_MULT × Q** とする。

【★測れないこと】
  自分が板に居ることで**他者の行動が変わる効果**は入らない。
  したがってここで出る数字は**上限**である。
  capacity_report.md のとおり、他者の注文数量の中央値は 53 単位(板の 7.1%)なので、
  Q が数十単位に達すれば前提は確実に崩れる。
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from backtester_v3 import (D, LIFE, FEE_BP, LAT, TICK, NON_RESTING,   # noqa: E402
                           load_model, predict)

SP = D / "slope_spline"
INV_MULT = 5.0
SIZES = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
# ★価格は定数で持たない。DRAM は 42.8〜71.1 USD の間を動くので、
#   bp 換算の分母はその日の中値から取る(定数 5.6 を置いていたのは誤りだった)。


def load(dt, model):
    f = SP / f"{dt}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if "ts" not in z.files:
        return None
    life = (pl.read_parquet(LIFE / f"dt={dt}" / "part-000.parquet",
                            columns=["side", "px", "ts_open", "ts_close",
                                     "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null() & pl.col("ts_close").is_not_null()))
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fl = pl.concat([pl.read_parquet(x, columns=["ts", "px", "sz", "side", "crossed", "tid"])
                    for x in fs], how="diagonal_relaxed")
    # ★決定性: fills の ts は ms 精度で同時刻タイが大量にある。unique() は順序を
    #   保存しないため、同一 ms 内の処理順が実行ごとに変わり、浮動小数の加算順の
    #   違いで建玉がちょうど在庫上限(例 5.0)の境界を跨ぐ判定が揺れる。
    #   tid で全順序を固定する(報告 35 §7-6)。
    fill = (fl.filter(pl.col("crossed")).unique(subset=["tid"], keep="first",
                                                maintain_order=True)
              .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort(["t", "tid"]))
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "best_bid", "best_ask", "mid", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    px = life["px"].to_numpy(); sd = life["side"].to_numpy()
    o = life["ts_open"].to_numpy(); c = life["ts_close"].to_numpy()
    key = np.round(px * 1e6).astype(np.int64) * 2 + (sd == "B")
    tmp: dict[int, list] = {}
    for k, oo, cc in zip(key, o, c):
        tmp.setdefault(int(k), []).append((int(oo), int(cc)))
    lvl = {k: (np.array([x[0] for x in sorted(v)], dtype=np.int64),
               np.maximum.accumulate(np.array([x[1] for x in sorted(v)], dtype=np.int64)))
           for k, v in tmp.items()}
    return {"lvl": lvl, "fill": fill, "mp": mp}


def simulate(d, qty, inv_limit):
    mp = d["mp"]
    mts = mp["ts"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    mid = mp["mid"].to_numpy()
    fl = d["fill"]
    f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy()
    f_sz = fl["sz"].to_numpy(); f_side = fl["side"].to_numpy()
    trades = {}
    for s in ("B", "A"):
        m = f_side == ("A" if s == "B" else "B")
        trades[s] = (f_t[m], f_px[m], f_sz[m])

    intervals = []
    for side, ser in (("B", bb), ("A", ba)):
        i = 0; n = mts.size
        while i < n:
            p = ser[i]; j = i
            while j + 1 < n and ser[j + 1] == p:
                j += 1
            intervals.append((mts[i], mts[j + 1] if j + 1 < n else mts[-1], side, p, i))
            i = j + 1
    intervals.sort()

    cash = 0.0; pos = 0.0
    n_fill = 0; vol = 0.0; n_quote = 0; n_full = 0
    for t0, t_end, side, p, i0 in intervals:
        s_dir = 1.0 if side == "B" else -1.0
        if inv_limit > 0 and s_dir * pos >= inv_limit:
            continue
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:            # A 気配改善
            px_ = p + TICK if side == "B" else p - TICK
        n_quote += 1
        k_ = int(round(px_ * 1e6)) * 2 + (side == "B")
        lv = d["lvl"].get(k_)
        t_front = t0
        if px_ == p and lv is not None:
            jj = int(np.searchsorted(lv[0], t0, side="right")) - 1
            if jj >= 0 and lv[1][jj] > t0:
                t_front = int(lv[1][jj])
        tt, tp, tz = trades[side]
        lo = np.searchsorted(tt, max(t0, t_front)); hi = np.searchsorted(tt, t_end)
        # ★部分約定: 先頭に立った後、到来する約定を min(残数量, 約定数量) で順に食う
        rem = qty
        got = False
        for x in range(lo, hi):
            if abs(tp[x] - px_) > 1e-9:
                continue
            take = min(rem, float(tz[x]))
            if take <= 0:
                continue
            cash += -s_dir * px_ * take - FEE_BP / 1e4 * px_ * take
            pos += s_dir * take
            vol += take; n_fill += 1; rem -= take; got = True
            if rem <= 1e-9:
                break
        if got and rem <= 1e-9:
            n_full += 1
    if vol <= 0:
        return None
    pnl = cash + pos * mid[-1]
    px_ref = float(np.median(mid))          # その日の価格水準(bp 換算の分母)
    return {"qty": qty, "n_quote": n_quote, "n_fill": n_fill, "n_full": n_full,
            "volume": vol, "pnl_usd": float(pnl), "px_ref": px_ref,
            "notional_usd": float(vol * px_ref),
            "pnl_bp_per_unit": float(pnl / (vol * px_ref) * 1e4),
            "fill_ratio": float(vol / (n_quote * qty)) if n_quote else 0.0,
            "end_inv": float(pos)}


def main() -> None:
    model = load_model()
    days = sys.argv[1:] or sorted(p.stem for p in SP.glob("*.npz"))
    days = [x for x in days if x not in set(model[4])]
    rows = []
    for dt in days:
        d = load(dt, model)
        if d is None:
            continue
        for q in SIZES:
            r = simulate(d, q, INV_MULT * q)
            if r:
                rows.append({"dt": dt, **r})
        print(f"{dt} 済", flush=True)
        pl.DataFrame(rows).write_csv(D / "capacity_v3.csv")
    pl.DataFrame(rows).write_csv(D / "capacity_v3.csv")


if __name__ == "__main__":
    main()
