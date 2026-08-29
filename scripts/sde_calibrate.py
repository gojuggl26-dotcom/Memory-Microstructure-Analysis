"""SDE 模型の較正 — 実シミュレーションを計測して素過程のパラメータを採る。

【なぜ計測するか】
  SDE を組むには (a) 約定の到来強度 λ、(b) 1 約定あたりのスプレッド獲得 δ、
  (c) 約定数量の分布、(d) 在庫過程の性質 が要る。
  これらを文献値や当て推量で置くと「仮定した答えが出る」だけになるので、
  **報告 34 と同一のシミュレーション(capacity_v3, Q=0.372)を計装して実測**する。

【時間契約】
  予測模型ではない。各約定の粗利は「約定時刻の中値」と「約定価格」の差であり、
  どちらも約定時点で確定している。未来の情報は使わない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from capacity_v3 import load, TICK, FEE_BP          # noqa: E402
from backtester_v3 import load_model                # noqa: E402

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")


def simulate_instr(d, qty, inv_limit):
    """capacity_v3.simulate と同一の規則。加えて約定ごとの記録と在庫経路を返す。"""
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
    n_quote = 0
    rec_t = []; rec_s = []; rec_sz = []; rec_gross = []      # 約定ごと
    inv_t = []; inv_q = []                                    # 在庫経路(約定時点)
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
        lo = np.searchsorted(tt, max(t0, t_front)); hi = np.searchsorted(tt, t_end)
        rem = qty
        for x in range(lo, hi):
            if abs(tp[x] - px_) > 1e-9:
                continue
            take = min(rem, float(tz[x]))
            if take <= 0:
                continue
            cash += -s_dir * px_ * take - FEE_BP / 1e4 * px_ * take
            pos += s_dir * take
            rem -= take
            ft = int(tt[x])
            j0 = int(np.searchsorted(mts, ft, side="right")) - 1
            m0 = mid[max(j0, 0)]
            rec_t.append(ft); rec_s.append(s_dir); rec_sz.append(take)
            rec_gross.append(s_dir * (m0 - px_) / m0 * 1e4)     # 粗利 [bp]
            inv_t.append(ft); inv_q.append(pos)
            if rem <= 1e-9:
                break
    if not rec_t:
        return None
    rt = np.array(rec_t); rs = np.array(rec_s)
    rz = np.array(rec_sz); rg = np.array(rec_gross)
    iq = np.array(inv_q)
    dt_ns = np.diff(rt)
    # 在庫の時間加重(約定間は一定)
    w = np.diff(np.append(rt, mts[-1])).astype(np.float64)
    tw = w.sum()
    return {
        "pnl_usd": float(cash + pos * mid[-1]),
        "n_quote": n_quote, "n_fill": len(rt), "volume": float(rz.sum()),
        "gross_bp_mean": float(np.average(rg, weights=rz)),
        "gross_bp_med": float(np.median(rg)),
        "sz_mean": float(rz.mean()), "sz_sd": float(rz.std(ddof=1)),
        "sz_p50": float(np.median(rz)), "sz_full_frac": float((rz > qty - 1e-9).mean()),
        "iat_med_ms": float(np.median(dt_ns) / 1e6) if dt_ns.size else np.nan,
        "iat_mean_ms": float(dt_ns.mean() / 1e6) if dt_ns.size else np.nan,
        "buy_frac": float((rs > 0).mean()),
        "q_abs_tw": float(np.average(np.abs(iq), weights=w)),
        "q_max_abs": float(np.abs(iq).max()),
        "q_at_limit_frac": float(np.average(np.abs(iq) >= inv_limit - 1e-9, weights=w)),
        "q_end": float(iq[-1]),
        "mid_first": float(mid[0]), "mid_last": float(mid[-1]),
        "_fills": (rt, rs, rz, rg), "_inv": (np.array(inv_t), iq, mts[-1]),
    }


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    QTY = float(pl.read_csv(D / "capacity_100usd.csv")
                .filter(pl.col("lev") == 1)["qty"][0])
    INV = 5.0 * QTY
    rows = []
    keep_sz = []; keep_gross = []; keep_iat = []
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        r = simulate_instr(d, QTY, INV)
        if r is None:
            continue
        rt, rs, rz, rg = r.pop("_fills")
        r.pop("_inv")
        keep_sz.append(rz); keep_gross.append(rg)
        keep_iat.append(np.diff(rt) / 1e9)
        rows.append({"dt": dt, "qty": QTY, "inv_limit": INV, **r})
        print(f"{i+1}/{len(days)} {dt} pnl {r['pnl_usd']:+.3f} "
              f"gross {r['gross_bp_mean']:.4f}bp n {r['n_fill']}", flush=True)
        pl.DataFrame(rows).write_csv(D / "sde_calib_daily.csv")
    pl.DataFrame(rows).write_csv(D / "sde_calib_daily.csv")
    np.savez_compressed(D / "sde_calib_fills.npz",
                        sz=np.concatenate(keep_sz),
                        gross=np.concatenate(keep_gross),
                        iat_s=np.concatenate(keep_iat))
    print("完了")


if __name__ == "__main__":
    main()
