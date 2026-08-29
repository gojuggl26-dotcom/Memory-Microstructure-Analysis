"""Hyperliquid(xyz:DRAM)と Binance(DRAMUSDT)のリードラグ。

【符号の定義 — 混乱しやすいので明示する】
    ρ(k) = corr( r_HL(t), r_BN(t + kΔ) )
      k > 0 で山 → HL の動きが後の Binance に相関 = **HL が先行**
      k < 0 で山 → Binance が先行
  Δ は標本化間隔。複数の Δ で測る(Epps 効果の確認)。

【非同期性の扱い】
  両者とも共通グリッド上に **backward asof**(その時刻以前の最新値)で載せる。
  未来を見ないので、リードラグの推定に先読みは入らない。
  Binance 側は約定価格(bookTicker が存在しないため)。気配跳ねは
  相関を減衰させるが、**山の位置は偏らせない**(跳ねは独立雑音のため)。

【★時計の問題 — 本分析の最大の限界】
  HL の時刻は L1 合意の時刻、Binance の時刻はその取引所サーバの時刻で、
  **両者が絶対時刻で揃っている保証はない**。一定の時計ずれは
  リードラグと**区別できない**。よって
    ・絶対のラグは「時計ずれ込み」として報告する
    ・日ごと・時間帯ごとの**変動**を併せて出す(時計ずれなら不変のはず)
    ・プラセボ(片方を 1 時間ずらす)で相関が消えることを確認する

【出力】 data/leadlag_hl_binance.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
# (標本化間隔 ns, 片側ラグ本数)
GRIDS = [(100_000_000, 50), (250_000_000, 40), (1_000_000_000, 30),
         (5_000_000_000, 12)]
PLACEBO_NS = 3600 * 1_000_000_000


def hl_series(dt: str):
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    if len(m) < 5000:
        return None
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    return ts, m["mid"].to_numpy()


def bn_series(dt: str, col: str = "mid_proxy"):
    f = D / f"binance/dt={dt}.parquet"
    if not f.exists():
        return None
    b = pl.read_parquet(f).sort("ts_ms")
    if len(b) < 2000:
        return None
    return b["ts_ms"].to_numpy() * 1_000_000, b[col].to_numpy()


def grid_returns(ts_a, pa, ts_b, pb, step, shift_b=0):
    """共通グリッドに backward asof で載せて対数リターン(bp)を返す。"""
    lo = max(ts_a[0], ts_b[0] + shift_b) + step
    hi = min(ts_a[-1], ts_b[-1] + shift_b)
    if hi - lo < 3600 * 1_000_000_000:              # 1 時間未満の重なりは捨てる
        return None
    g = np.arange(lo, hi, step)
    ia = np.searchsorted(ts_a, g, side="right") - 1
    ib = np.searchsorted(ts_b + shift_b, g, side="right") - 1
    ok = (ia >= 0) & (ib >= 0)
    if ok.sum() < 1000:
        return None
    la = np.log(pa[ia[ok]])
    lb = np.log(pb[ib[ok]])
    return np.diff(la) * 1e4, np.diff(lb) * 1e4


def ccf(ra, rb, L):
    """ρ(k) = corr(ra(t), rb(t+k)) を k=-L..L で返す。"""
    out = np.full(2 * L + 1, np.nan)
    for i, k in enumerate(range(-L, L + 1)):
        if k >= 0:
            x, y = ra[:len(ra) - k], rb[k:]
        else:
            x, y = ra[-k:], rb[:len(rb) + k]
        if len(x) < 500 or x.std() == 0 or y.std() == 0:
            continue
        out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def main() -> None:
    days = sorted(p.stem.split("=")[1]
                  for p in (D / "binance").glob("dt=*.parquet"))
    res: dict = {"days": [], "grids": {}}
    store = {s: {"sum": None, "n": 0, "peaks": [], "peak_rho": []} for s, _ in GRIDS}
    store_pl = {s: {"sum": None, "n": 0} for s, _ in GRIDS}
    used = []
    for dt in days:
        h = hl_series(dt)
        b = bn_series(dt)
        if h is None or b is None:
            continue
        used.append(dt)
        for step, L in GRIDS:
            r = grid_returns(h[0], h[1], b[0], b[1], step)
            if r is None:
                continue
            c = ccf(r[0], r[1], L)
            s = store[step]
            s["sum"] = c if s["sum"] is None else np.nansum([s["sum"], c], axis=0)
            s["n"] += 1
            if np.isfinite(c).any():
                j = int(np.nanargmax(np.abs(c)))
                s["peaks"].append((j - L) * step / 1e9)
                s["peak_rho"].append(float(c[j]))
            rp = grid_returns(h[0], h[1], b[0], b[1], step, shift_b=PLACEBO_NS)
            if rp is not None:
                cp = ccf(rp[0], rp[1], L)
                sp = store_pl[step]
                sp["sum"] = cp if sp["sum"] is None else np.nansum([sp["sum"], cp], axis=0)
                sp["n"] += 1
        if len(used) % 10 == 0:
            print(f"{len(used)} 日処理", flush=True)
    res["days"] = used
    for step, L in GRIDS:
        s, sp = store[step], store_pl[step]
        if s["n"] == 0:
            continue
        mean = (s["sum"] / s["n"]).tolist()
        res["grids"][str(step)] = {
            "step_ms": step / 1e6, "L": L, "n_days": s["n"],
            "lags_s": [(k - L) * step / 1e9 for k in range(2 * L + 1)],
            "ccf_mean": mean,
            "ccf_placebo": (sp["sum"] / sp["n"]).tolist() if sp["n"] else None,
            "peak_lag_per_day": s["peaks"], "peak_rho_per_day": s["peak_rho"]}
    (D / "leadlag_hl_binance.json").write_text(json.dumps(res, indent=1),
                                               encoding="utf-8")
    print(f"完了: {len(used)} 日")


if __name__ == "__main__":
    main()
