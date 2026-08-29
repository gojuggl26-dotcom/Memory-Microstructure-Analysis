"""監査: 発注・取消の遅延を入れると E[PnL] はどうなるか。

【何を検査するか】
  capacity_v3.simulate は最良気配が変わった瞬間 t0 に注文が板に載り、
  区間終端 t_end に即時取消される(遅延ゼロ)。シグナル判定には δ=100ms を
  課しているのに、**発注と取消には課していない**。これは 2 つの楽観を生む:
    (i)  [t0, t0+δ) に来る約定を取れてしまう(実際は板に届いていない)
    (ii) [t_end, t_end+δ) の stale 約定(気配が動いた後の pick-off)を受けない
  (i) は利益の過大、(ii) は損失の過小。両方入れた値が実装可能な下界に近い。

【時間契約】
  発注価格の決定は t0 時点の板(その時点で既知)。注文は t0+lat に有効化、
  取消は t_end+lat に効く。約定判定は [max(t0+lat, t_front), t_end+lat) の
  実約定列。未来の情報は判定に使わない。

【x が確定する時刻 / y の期間】
  予測ではなく執行シミュレーション。全判定は判定時点までの情報のみ。
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from capacity_v3 import load, TICK, FEE_BP          # noqa: E402
from backtester_v3 import load_model                # noqa: E402

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
MS = 1_000_000


def simulate_lat(d, qty, inv_limit, lat_open, lat_cancel):
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
    n_fill = 0; vol = 0.0; n_quote = 0
    for t0, t_end, side, p, i0 in intervals:
        s_dir = 1.0 if side == "B" else -1.0
        if inv_limit > 0 and s_dir * pos >= inv_limit:
            continue
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:
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
        lo = np.searchsorted(tt, max(t0 + lat_open, t_front))
        hi = np.searchsorted(tt, t_end + lat_cancel)
        rem = qty
        for x in range(lo, hi):
            if abs(tp[x] - px_) > 1e-9:
                continue
            take = min(rem, float(tz[x]))
            if take <= 0:
                continue
            cash += -s_dir * px_ * take - FEE_BP / 1e4 * px_ * take
            pos += s_dir * take
            vol += take; n_fill += 1; rem -= take
            if rem <= 1e-9:
                break
    if vol <= 0:
        return None
    pnl = cash + pos * mid[-1]
    return {"n_quote": n_quote, "n_fill": n_fill, "volume": float(vol),
            "pnl_usd": float(pnl), "end_inv": float(pos)}


def main():
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    q1 = float(pl.read_csv(D / "capacity_100usd.csv")
               .filter(pl.col("lev") == 1)["qty"][0])          # 0.372...
    # (名前, qty, lat_open, lat_cancel) [ns]
    CFG = [("lat0",        q1, 0,        0),
           ("lat50",       q1, 50 * MS,  50 * MS),
           ("lat100",      q1, 100 * MS, 100 * MS),
           ("lat300",      q1, 300 * MS, 300 * MS),
           ("open100only", q1, 100 * MS, 0),        # (i) 早い約定の喪失のみ
           ("cancel100only", q1, 0,      100 * MS), # (ii) stale pick-off のみ
           ("q1_lat0",     1.0, 0,       0),
           ("q1_lat100",   1.0, 100 * MS, 100 * MS)]
    rows = []
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        for nm, q, lo_, lc_ in CFG:
            r = simulate_lat(d, q, 5.0 * q, lo_, lc_)
            if r:
                rows.append({"dt": dt, "cfg": nm, "qty": q,
                             "lat_open_ms": lo_ // MS, "lat_cancel_ms": lc_ // MS, **r})
        print(f"{i+1}/{len(days)} {dt} 済", flush=True)
        pl.DataFrame(rows).write_csv(D / "audit_latency.csv")
    pl.DataFrame(rows).write_csv(D / "audit_latency.csv")
    print("完了", len(days), "日")


if __name__ == "__main__":
    main()
