"""市場の状態で Entry EV を説明できるか(仮説の検定用パネル)。

    uv run python scripts/build_mktpanel.py --coins xyz:MU xyz:INTC

仮説
----
    EV ≈ f( (Spread/2 − 1tick) / σ_short ,  Depth / TradeSize ,  OBI の IC )

「price-improvement 後の余裕が大きく、逆行が小さく、食い切られにくく、
OBI の方向予測力が高い市場ほど Entry EV が高い」。

作るもの
--------
銘柄 × 日 × 時刻(UTC 1 時間)の窓ごとに

    room_bp        = スプレッド/2 − 1 ティック(bp)
    sigma_short_bp = 1 秒刻みの mid 対数リターンの標準偏差(bp)
    x1             = room_bp / sigma_short_bp
    depth_over_trade = 最良気配の数量の中央値 ÷ 約定 1 件の数量の中央値
    obi_ic_mid     = corr(OBI_t, r^mid_{t→t+1s})
    obi_ic_micro   = corr(OBI_t, r^micro_{t→t+1s})
    ev             = その窓で建てた玉の往復損益の平均(在庫シミュレータの実績)

★ 時間契約について
-------------------
説明変数と EV を**同じ窓**で取ると、それは予測ではなく同時性である。
そこで **1 つ前の窓の説明変数で次の窓の EV を説明する**版も必ず作り、
両方を並べて報告する。IC 自体も窓内の先読み(t→t+1s)を含むので、
「市場の性質を記述する量」であって発注時に使える予測子ではない。

★ OBI の IC は 2 通り出す
-------------------------
既報のとおり OBI が mid を予測して見えるのは、**mid が既知の microprice へ
追いつく機械的な動き**が大半である。mid に対する IC はスプレッドの大きさと
機械的に相関するので、x1 と共線になりうる。microprice に対する IC を併記する。
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
HOUR_NS = 3_600_000_000_000
IC_H = 1.0                      # IC を測る前向きホライズン(秒)


def day_panel(tag: str, dt: str, lots: pl.DataFrame) -> list[dict]:
    d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                  .filter(pl.col("dt") == dt).collect())[0].sort("ts")
    if d.height < 1000:
        return []
    ts = d["ts"].cast(pl.Int64).to_numpy()
    pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
    qb, qa = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
    mid = 0.5 * (pb + pa)
    sq = qb + qa
    obi = np.where(sq > 0, (qb - qa) / np.maximum(sq, 1e-12), 0.0)
    micro = mid + 0.5 * (pa - pb) * obi
    tick = np.where(mid >= 1000.0, 0.1, 0.01)
    nb = ts.size
    d0 = int(ts[0]) // DAY_NS * DAY_NS

    F = pl.scan_parquet(DATA / f"fills_{tag}.parquet").filter(
        pl.col("crossed") & (pl.col("dt") == dt)).select("ts", "sz").collect()
    ft = F["ts"].cast(pl.Int64).to_numpy()
    fsz = F["sz"].to_numpy()

    # 前向き 1 秒の後ろ向き asof(t+1s 以前の最後の板)
    jf = np.clip(np.searchsorted(ts, ts + int(IC_H * 1e9), side="right") - 1,
                 0, nb - 1)
    r_mid = np.log(mid[jf] / mid) * 1e4
    r_mic = np.log(micro[jf] / micro) * 1e4

    # 1 秒格子の mid(σ_short 用)
    ng = 86400
    tg = d0 + np.arange(ng, dtype=np.int64) * 10 ** 9
    gj = np.clip(np.searchsorted(ts, tg, side="right") - 1, 0, nb - 1)
    midg = mid[gj]
    rg = np.diff(np.log(midg)) * 1e4

    L = lots.filter(pl.col("dt") == dt)
    lt = L["t_in"].to_numpy() if L.height else np.zeros(0, np.int64)
    lp = L["pnl"].to_numpy().astype(float) if L.height else np.zeros(0)

    out = []
    for h in range(24):
        lo, hi = d0 + h * HOUR_NS, d0 + (h + 1) * HOUR_NS
        m = (ts >= lo) & (ts < hi)
        n = int(m.sum())
        if n < 200:
            continue
        gs, ge = h * 3600, (h + 1) * 3600 - 1
        rr = rg[gs:ge]
        rr = rr[np.isfinite(rr)]
        if rr.size < 600:
            continue
        sig = float(rr.std())
        hs = float(np.median((pa[m] - pb[m]) / mid[m] * 1e4)) / 2.0
        tk = float(np.median(tick[m] / mid[m] * 1e4))
        fm = (ft >= lo) & (ft < hi)
        nf = int(fm.sum())
        if nf < 20:
            continue
        tsz = float(np.median(fsz[fm]))
        dep = float(np.median(sq[m] / 2.0))
        o, rm, rc = obi[m], r_mid[m], r_mic[m]
        v = np.isfinite(o) & np.isfinite(rm) & np.isfinite(rc)
        if v.sum() < 200 or o[v].std() < 1e-9:
            continue
        ic_m = float(np.corrcoef(o[v], rm[v])[0, 1])
        ic_c = (float(np.corrcoef(o[v], rc[v])[0, 1])
                if rc[v].std() > 1e-12 else np.nan)
        lm = (lt >= lo) & (lt < hi)
        nl = int(lm.sum())
        out.append({
            "coin": tag, "dt": dt, "hour": h, "n_bbo": n, "n_fill": nf,
            "half_spread_bp": hs, "tick_bp": tk, "room_bp": hs - tk,
            "sigma_short_bp": sig, "x1": (hs - tk) / max(sig, 1e-9),
            "depth": dep, "trade_size": tsz,
            "depth_over_trade": dep / max(tsz, 1e-12),
            "obi_ic_mid": ic_m, "obi_ic_micro": ic_c,
            "n_lot": nl,
            "ev": float(lp[lm].mean()) if nl else np.nan})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["xyz:MU", "xyz:INTC"])
    a = ap.parse_args()
    rows = []
    for c in a.coins:
        tag = c.replace(":", "_")
        lots = pl.read_parquet(DATA / f"inv_lots_{tag}_q1.parquet")
        days = sorted(lots["dt"].unique().to_list())
        print(f"{c}: {len(days)} 日", flush=True)
        for k, dt in enumerate(days):
            rows += day_panel(tag, dt, lots)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(days)}", flush=True)
    P = pl.DataFrame(rows)
    P.write_parquet(DATA / "mktpanel.parquet")
    P.write_csv(DATA / "mktpanel.csv")
    print(f"\n窓 {P.height:,} 個(銘柄 {P['coin'].n_unique()} × 日 × 時刻)")
    print(P.select("x1", "depth_over_trade", "obi_ic_mid", "obi_ic_micro",
                   "ev", "n_lot").describe())
    print(f"\n書き出し {DATA}/mktpanel.parquet")


if __name__ == "__main__":
    main()
