"""リードラグの経済的な使い道 — 「Binance を見て HL の次を予測できるか」。

【問い】
  相関があることと、使えることは別である(報告 42 の教訓)。
  マーケットメイカーにとっての実用形は
    「直前 Δ の Binance の動きを見て、これから h の間の HL の動きを予測し、
      不利なら気配を引く」
  なので、そのままの形で測る。

【時間契約】
  x = Binance の [t−Δ, t) のリターン(t の時点で確定)
  y = HL の [t, t+h) のリターン(t 以降)
  **x の確定時刻 = y の期間の開始時刻**。backward asof のみ使用。

【判定】
  x の十分位ごとの E[y] を bp で出し、δ*(0.363bp)・ハーフスプレッド(0.287bp)・
  1 ティック(0.184bp)と比べる。回避規則として意味があるのは
  「予測が 1 ティックを超える」場合である。

【帰無対照】 x を 1 時間ずらしたプラセボ。

【出力】 data/leadlag_econ.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
STEP = 50_000_000                      # グリッド 50ms
DX = [5, 10]                           # x の窓: 250ms, 500ms(グリッド本数)
HY = [2, 5, 10, 20]                    # y の地平: 100ms, 250ms, 500ms, 1s
PLACEBO = 3600 * 1_000_000_000


def load(dt):
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    f2 = D / f"binance/dt={dt}.parquet"
    if not fs or not f2.exists():
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    b = pl.read_parquet(f2).sort("ts_ms")
    if len(m) < 5000 or len(b) < 2000:
        return None
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    return ts, m["mid"].to_numpy(), b["ts_ms"].to_numpy() * 1_000_000, \
        b["mid_proxy"].to_numpy()


def grid(tsh, ph, tsb, pb, shift=0):
    lo = max(tsh[0], tsb[0] + shift) + STEP * max(DX)
    hi = min(tsh[-1], tsb[-1] + shift) - STEP * max(HY)
    if hi - lo < 3600 * 1_000_000_000:
        return None
    g = np.arange(lo, hi, STEP)
    ih = np.searchsorted(tsh, g, side="right") - 1
    ib = np.searchsorted(tsb + shift, g, side="right") - 1
    ok = (ih >= 0) & (ib >= 0)
    if ok.sum() < 5000:
        return None
    return np.log(ph[ih[ok]]) * 1e4, np.log(pb[ib[ok]]) * 1e4


def main() -> None:
    days = sorted(p.stem.split("=")[1] for p in (D / "binance").glob("dt=*.parquet"))
    acc: dict = {}
    for i, dt in enumerate(days):
        d = load(dt)
        if d is None:
            continue
        for tag, shift in [("real", 0), ("placebo", PLACEBO)]:
            gr = grid(*d, shift=shift)
            if gr is None:
                continue
            lh, lb = gr
            n = len(lh)
            for dx in DX:
                x = lb[dx:] - lb[:-dx]                      # Binance の直前 dx の動き
                for hy in HY:
                    m = n - dx - hy
                    if m < 5000:
                        continue
                    xx = x[:m]
                    yy = lh[dx + hy:dx + hy + m] - lh[dx:dx + m]   # HL の直後 hy
                    key = f"{tag}_dx{dx}_hy{hy}"
                    a = acc.setdefault(key, {"x": [], "y": []})
                    # 日ごとに 5 万点に間引く(全部持つとメモリが持たない)
                    idx = np.linspace(0, m - 1, min(m, 50_000)).astype(int)
                    a["x"].append(xx[idx].astype(np.float32))
                    a["y"].append(yy[idx].astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(days)}", flush=True)
    res: dict = {"step_ms": STEP / 1e6, "n_days": len(days)}
    for key, a in acc.items():
        x = np.concatenate(a["x"]).astype(np.float64)
        y = np.concatenate(a["y"]).astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        q = np.quantile(x, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 9)
        dec = [float(y[b == j].mean()) for j in range(10)]
        decx = [float(x[b == j].mean()) for j in range(10)]
        beta = float(np.polyfit(x, y, 1)[0])
        res[key] = {"n": int(len(x)), "corr": float(np.corrcoef(x, y)[0, 1]),
                    "beta": beta, "sd_x": float(x.std()), "sd_y": float(y.std()),
                    "decile_x": decx, "decile_y": dec}
    (D / "leadlag_econ.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("完了")


if __name__ == "__main__":
    main()
