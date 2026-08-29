"""リードラグの精密化 — 時計ずれと本物の情報伝播を切り分ける。

【切り分けの論理】
  一定の時計ずれ δ は CCF を**剛体的に平行移動**するだけである。したがって
    ・山の位置だけでは、本物のリードと時計ずれを区別できない
    ・しかし **山から遠いラグ(±3〜5 秒)での左右非対称**は、
      0.2 秒の平行移動では作れない。そこに非対称が残るなら**本物**である
  よって (a) 細かいグリッドで山を詰める、(b) 遠いラグの非対称比を測る、
  (c) 市場状態(時間帯・ボラ)でラグが動くか見る(時計ずれなら不変)。

【経済的な翻訳】
  ρ と各系列の SD から「Binance の動きを見て予測できる HL の変化幅(bp)」を出し、
  δ*(=0.363bp、報告 36)およびハーフスプレッドと比べる。

【出力】 data/leadlag_detail.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
FINE_STEP = 50_000_000        # 50 ms
FINE_L = 40                   # ±2 秒
SESS = {"us": (13, 21), "off": None}


def hl_series(dt):
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


def bn_series(dt):
    f = D / f"binance/dt={dt}.parquet"
    if not f.exists():
        return None
    b = pl.read_parquet(f).sort("ts_ms")
    if len(b) < 2000:
        return None
    return b["ts_ms"].to_numpy() * 1_000_000, b["mid_proxy"].to_numpy()


def series_on_grid(ts_a, pa, ts_b, pb, step):
    """全時間の共通グリッドでリターンを作る。時間帯の選別は後段の mask で行う
    (グリッドを先に間引くと、時間帯の境目をまたぐ差分が偽の跳ねになる)。"""
    lo = max(ts_a[0], ts_b[0]) + step
    hi = min(ts_a[-1], ts_b[-1])
    if hi - lo < 3600 * 1_000_000_000:
        return None
    g = np.arange(lo, hi, step)
    ia = np.searchsorted(ts_a, g, side="right") - 1
    ib = np.searchsorted(ts_b, g, side="right") - 1
    ok = (ia >= 0) & (ib >= 0)
    if ok.sum() < 5000:
        return None
    g = g[ok]
    ra = np.diff(np.log(pa[ia[ok]])) * 1e4
    rb = np.diff(np.log(pb[ib[ok]])) * 1e4
    hour = ((g[1:] // (3600 * 1_000_000_000)) % 24).astype(np.int8)
    return ra, rb, hour


def ccf(ra, rb, L, mask=None):
    """mask を渡すと、両端が mask 内にある対だけで相関を取る(時間帯の分割用)。"""
    out = np.full(2 * L + 1, np.nan)
    for i, k in enumerate(range(-L, L + 1)):
        if k >= 0:
            x, y = ra[:len(ra) - k], rb[k:]
            m = (mask[:len(ra) - k] & mask[k:]) if mask is not None else None
        else:
            x, y = ra[-k:], rb[:len(rb) + k]
            m = (mask[-k:] & mask[:len(rb) + k]) if mask is not None else None
        if m is not None:
            x, y = x[m], y[m]
        if len(x) < 500 or x.std() == 0 or y.std() == 0:
            continue
        out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def main() -> None:
    days = sorted(p.stem.split("=")[1] for p in (D / "binance").glob("dt=*.parquet"))
    acc = {"all": [], "us": [], "off": []}
    sd = {"hl": [], "bn": []}
    for i, dt in enumerate(days):
        h, b = hl_series(dt), bn_series(dt)
        if h is None or b is None:
            continue
        r = series_on_grid(h[0], h[1], b[0], b[1], FINE_STEP)
        if r is None:
            continue
        ra, rb, hour = r
        us = (hour >= 13) & (hour < 21)
        for key, m in [("all", None), ("us", us), ("off", ~us)]:
            c = ccf(ra, rb, FINE_L, m)
            if not np.isfinite(c).any():        # 全 NaN の日は捨てる
                continue
            acc[key].append(c)
        sd["hl"].append(float(ra.std()))
        sd["bn"].append(float(rb.std()))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(days)}", flush=True)
    lags = [(k - FINE_L) * FINE_STEP / 1e9 for k in range(2 * FINE_L + 1)]
    res = {"step_ms": FINE_STEP / 1e6, "lags_s": lags,
           "sd_hl_bp": float(np.median(sd["hl"])), "sd_bn_bp": float(np.median(sd["bn"]))}
    for key in acc:
        if not acc[key]:
            continue
        M = np.vstack(acc[key])
        mean = np.nanmean(M, axis=0)
        pk = np.array([lags[int(np.nanargmax(np.abs(row)))] for row in M])
        res[key] = {"n_days": int(M.shape[0]), "ccf_mean": mean.tolist(),
                    "peak_lag_s": float(lags[int(np.nanargmax(np.abs(mean)))]),
                    "peak_rho": float(np.nanmax(np.abs(mean))),
                    "peak_per_day": pk.tolist()}
    (D / "leadlag_detail.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("完了")


if __name__ == "__main__":
    main()
