"""遅延 100ms の世界を SDE の言葉で分解する。

【問い】
  報告 35 は「発注・取消に 100ms の遅延を入れると E[PnL] が +3.21 → −0.59 になる」
  と測った。SDE の分解 E[PnL] = λ v S (δ − f − β σ_Δ) で見ると、遅延は
    ・λ を落とす(約定機会を逃す)      … 実測 5,691 → 3,672 件/日
    ・δ を落とす(良い約定だけ逃す)    … 未測定
    ・β を上げる(stale 約定を食らう)  … 未測定
  のどれで効いているのか。**λ の低下だけでは符号は反転しない**(全体に掛かる係数
  だから)。したがって δ か β のどちらかが必ず悪化している。それを実測する。

【方法】
  audit_latency.py と同じ遅延つきエンジンに、sde_calibrate.py と同じ計装を入れ、
  1 約定あたりの粗利 δ を測る。β は残差として逆算する。

【時間契約】audit_latency.py と同一。注文は t0+lat に有効化、取消は t_end+lat。
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
MS = 1_000_000


def sim(d, qty, inv_limit, lat_open, lat_cancel):
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
    iv = []
    for side, ser in (("B", bb), ("A", ba)):
        i = 0; n = mts.size
        while i < n:
            p = ser[i]; j = i
            while j + 1 < n and ser[j + 1] == p:
                j += 1
            iv.append((mts[i], mts[j + 1] if j + 1 < n else mts[-1], side, p, i))
            i = j + 1
    iv.sort()
    cash = 0.0; pos = 0.0
    rec_sz = []; rec_g = []
    for t0, t_end, side, p, i0 in iv:
        sd = 1.0 if side == "B" else -1.0
        if inv_limit > 0 and sd * pos >= inv_limit:
            continue
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:
            px_ = p + TICK if side == "B" else p - TICK
        k_ = int(round(px_ * 1e6)) * 2 + (side == "B")
        lv = d["lvl"].get(k_); t_front = t0
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
            cash += -sd * px_ * take - FEE_BP / 1e4 * px_ * take
            pos += sd * take; rem -= take
            ft = int(tt[x])
            j0 = int(np.searchsorted(mts, ft, side="right")) - 1
            m0 = mid[max(j0, 0)]
            rec_sz.append(take); rec_g.append(sd * (m0 - px_) / m0 * 1e4)
            if rem <= 1e-9:
                break
    if not rec_sz:
        return None
    z = np.array(rec_sz); g = np.array(rec_g)
    return {"pnl_usd": float(cash + pos * mid[-1]), "n_fill": len(z),
            "volume": float(z.sum()),
            "gross_bp_w": float(np.average(g, weights=z)),
            "sz_mean": float(z.mean()),
            "mid_med": float(np.median(mid))}


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    q = float(pl.read_csv(D / "capacity_100usd.csv")
              .filter(pl.col("lev") == 1)["qty"][0])
    rows = []
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        for nm, lo_, lc_ in [("lat0", 0, 0), ("lat100", 100 * MS, 100 * MS)]:
            r = sim(d, q, 5.0 * q, lo_, lc_)
            if r:
                rows.append({"dt": dt, "cfg": nm, **r})
        print(f"{i+1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(D / "sde_latency_calib.csv")
    df = pl.DataFrame(rows)
    df.write_csv(D / "sde_latency_calib.csv")

    print("\n=== 遅延の分解(30 日平均)===")
    print(f"{'':10}{'E[PnL]':>10}{'λ/日':>9}{'約定量':>9}{'δ [bp]':>10}")
    agg = {}
    for nm in ("lat0", "lat100"):
        s = df.filter(pl.col("cfg") == nm)
        w = s["volume"].to_numpy()
        agg[nm] = {"pnl": float(s["pnl_usd"].mean()),
                   "lam": float(s["n_fill"].mean()),
                   "vol": float(s["volume"].mean()),
                   "delta": float(np.average(s["gross_bp_w"].to_numpy(), weights=w)),
                   "px": float(s["mid_med"].mean()),
                   "v": float(np.average(s["sz_mean"].to_numpy(), weights=w))}
        a = agg[nm]
        print(f"  {nm:<8}{a['pnl']:>10.3f}{a['lam']:>9,.0f}{a['v']:>9.4f}{a['delta']:>10.4f}")

    # β を残差として逆算: PnL = λ v S (δ − f − β σ_Δ)
    rv = pl.read_csv(D / "sde_realized_vol.csv")
    sig = float(rv.filter(pl.col("dt").is_in(days))["rv30"].median())
    sig_d_bp = sig / np.sqrt(86400) * 1e4
    print(f"\n  σ_Δ(1 秒)= {sig_d_bp:.4f} bp")
    for nm in ("lat0", "lat100"):
        a = agg[nm]
        base = a["lam"] * a["v"] * a["px"]
        net_bp = a["pnl"] / base * 1e4
        beta = (a["delta"] - FEE_BP - net_bp) / sig_d_bp
        a["net_bp"] = net_bp; a["beta"] = beta
        print(f"  {nm:<8} 純益 {net_bp:+.4f} bp/約定 → 逆算 β = {beta:.4f}"
              f"  (逆選択コスト {beta*sig_d_bp:.4f} bp)")
    d0, d1 = agg["lat0"], agg["lat100"]
    print(f"\n=== 遅延 100ms が壊すもの ===")
    print(f"  λ  {d0['lam']:,.0f} → {d1['lam']:,.0f} 件/日  ({d1['lam']/d0['lam']-1:+.1%})")
    print(f"  δ  {d0['delta']:.4f} → {d1['delta']:.4f} bp   ({d1['delta']/d0['delta']-1:+.1%})")
    print(f"  β  {d0['beta']:.4f} → {d1['beta']:.4f}        ({d1['beta']/d0['beta']-1:+.1%})")
    print(f"\n  ★λ だけの低下なら E[PnL] は {d0['pnl']*d1['lam']/d0['lam']:+.3f} のはず"
          f"(実測 {d1['pnl']:+.3f})。差 {d1['pnl']-d0['pnl']*d1['lam']/d0['lam']:+.3f} は"
          f" δ と β の劣化による。")


if __name__ == "__main__":
    main()
