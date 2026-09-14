r"""現物(指数価格)は Binance perp の上流か — 1 分足での検証。

    uv run python scripts/xv_index_test.py --coin xyz:MU

出力: data/xv_index_<coin>.csv / 標準出力

=============================================================================
★何を確かめたいか
=============================================================================
提案: 「参照を Binance ではなく本場(Nasdaq)のプレマーケットにすれば
      レイテンシが改善するのでは」

情報の連鎖が  現物 → Binance perp → HL perp  なら、上流へ遡るほど先行は増える。
だが本当にそうかは測らないと判らない。手元で測れるのは:

  Binance `indexPriceKlines` = **原資産(現物)から算出される指数価格**
  Binance `klines`           = MUUSDT perp 自身の値段
  HL の BBO                   = xyz:MU

★**1 分足でしか無い。**200ms の先行は原理的に測れない。ここで判るのは
  「1 分スケールで連鎖の向きがどうなっているか」と
  「指数が動かない時間帯はどこか」だけである。**それ以上を主張しない。**

=============================================================================
★先に判っている制約
=============================================================================
- 米国現物は **08:00〜00:00 UTC** しか無い(プレ 08:00-13:30 / 立会 13:30-20:00 /
  アフター 20:00-00:00)。**00:00〜08:00 UTC は現物が存在しない**
- その夜間こそ会場間の先行が最も強かった(CCF +200ms で 0.0607、全体 0.0329)
- 手元の現物 1 分足(`cash1m_MU.csv`)は **2026-08-13 以降**しかなく、
  標本期間 05-04〜08-10 に届かない(Yahoo の 30 日保持)
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = Path("E:/Binance-perp-data/data/history/parquet")
BP = 1e4
NS = 1_000_000_000
M = 60 * NS


def kl(kind, sym, day, tail=""):
    f = BASE / f"kind={kind}" / f"symbol={sym}" / f"{sym}-{kind}{tail}-{day}.parquet"
    if not f.exists():
        return None
    D = pl.read_parquet(f, columns=["open_time", "close", "high", "low"])
    t = D["open_time"].to_numpy().astype(np.int64)
    t = t * (1_000_000 if t[0] < 10 ** 14 else 1_000)      # ms/us -> ns
    # ★`open_time` は足の**開始**。`close` が判るのは 60 秒後である。
    #   揃えないと、それだけでラグ +1 分に偽のピークが出る(実際に出した)。
    return t + M, D["close"].to_numpy().astype(np.float64)


def locf(ts, v, q):
    i = np.searchsorted(ts, q, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(ts) - 1)
    return np.where(ok, v[i], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sym", default="")
    ap.add_argument("--lags", type=int, default=5, help="±何分")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    sym = a.sym or (a.coin.split(":")[-1] + "USDT")
    L = a.lags

    fs = sorted(glob.glob(str(DATA / f"bbo_l1_{tag}" / "*.parquet")))
    days = [os.path.basename(f).split("=")[1].split(".")[0] for f in fs]
    print(f"{a.coin} × {sym} / 1 分足 / ラグ ±{L} 分", flush=True)

    # 指数が「動く」時間帯の分布
    hour_move = np.zeros(24)
    hour_n = np.zeros(24)
    KEYS = ["idx_hl", "prp_hl", "idx_prp"]
    C = {k: np.zeros(2 * L + 1) for k in KEYS}
    N = {k: np.zeros(2 * L + 1) for k in C}
    # 時間帯別の 指数→perp(夜は米国現物が無いので向きが変わりうる)
    SS = [("00-08 夜", 0, 8), ("08-13.5 プレ", 8, 13.5),
          ("13.5-20 現物", 13.5, 20), ("20-24 アフター", 20, 24)]
    CS = np.zeros((len(SS), 2 * L + 1))
    NS_ = np.zeros_like(CS)
    rows = []
    for f, day in zip(fs, days):
        ix = kl("indexPriceKlines", sym, day, "-1m")
        pr = kl("klines", sym, day, "-1m")
        if ix is None or pr is None:
            continue
        B = pl.read_parquet(f)
        bt = B["ts"].to_numpy()
        bb, ba = B["best_bid"].to_numpy(), B["best_ask"].to_numpy()
        m = np.isfinite(bb) & np.isfinite(ba) & (ba > bb) & (bb > 0)
        bt, hm = bt[m], 0.5 * (bb[m] + ba[m])
        d0 = (bt[0] // (86400 * NS)) * 86400 * NS
        g = np.arange(d0, d0 + 86400 * NS, M, dtype=np.int64)
        vi = locf(ix[0], ix[1], g)
        vp = locf(pr[0], pr[1], g)
        vh = locf(bt, hm, g)
        ok = np.isfinite(vi) & np.isfinite(vp) & np.isfinite(vh)
        ri = np.concatenate([[np.nan], np.diff(np.log(vi))]) * BP
        rp = np.concatenate([[np.nan], np.diff(np.log(vp))]) * BP
        rh = np.concatenate([[np.nan], np.diff(np.log(vh))]) * BP
        hr = ((g - d0) // (3600 * NS)).astype(int)
        for h in range(24):
            s = ok & (hr == h)
            if s.sum():
                hour_n[h] += s.sum()
                hour_move[h] += np.sum(np.abs(ri[s]) > 1e-9)
        hrf = (g - d0) / (3600 * NS)
        for si, (_, lo, hi) in enumerate(SS):
            ms = ok & (hrf >= lo) & (hrf < hi)
            for k in range(-L, L + 1):
                xx = ri[max(0, -k): ri.size - max(0, k)]
                yy = rp[max(0, k): rp.size - max(0, -k)]
                gg = (ms[max(0, -k): ms.size - max(0, k)]
                      & ms[max(0, k): ms.size - max(0, -k)]
                      & np.isfinite(xx) & np.isfinite(yy))
                if gg.sum() < 50:
                    continue
                c = np.corrcoef(xx[gg], yy[gg])[0, 1]
                if np.isfinite(c):
                    CS[si, k + L] += c
                    NS_[si, k + L] += 1
        for nm, x, y in (("idx_hl", ri, rh), ("prp_hl", rp, rh),
                         ("idx_prp", ri, rp)):
            for k in range(-L, L + 1):
                xx = x[max(0, -k): x.size - max(0, k)]
                yy = y[max(0, k): y.size - max(0, -k)]
                gg = (ok[max(0, -k): ok.size - max(0, k)]
                      & ok[max(0, k): ok.size - max(0, -k)]
                      & np.isfinite(xx) & np.isfinite(yy))
                if gg.sum() < 50:
                    continue
                c = np.corrcoef(xx[gg], yy[gg])[0, 1]
                if np.isfinite(c):
                    C[nm][k + L] += c
                    N[nm][k + L] += 1
        rows.append({"dt": day, "cells": int(ok.sum()),
                     "idx_move_pct": float(100 * np.mean(
                         np.abs(ri[ok]) > 1e-9))})
    D = pl.DataFrame(rows)
    D.write_csv(DATA / f"xv_index_{tag}.csv")
    print(f"\n有効 {D.height} 日 / 指数が動くセルの割合 "
          f"{D['idx_move_pct'].mean():.1f}%")

    print("\n【1】指数価格(= 現物)が動く時間帯 — UTC 時ごとの割合")
    s = ""
    for h in range(24):
        p = 100 * hour_move[h] / max(hour_n[h], 1)
        s += f"{h:02d}:{p:5.1f}%  "
        if h % 6 == 5:
            print("  " + s)
            s = ""
    print("  ★00-08 UTC は米国現物が存在しない。指数はそこで止まる。")

    print(f"\n【2】1 分足の相互相関(正のラグ = 左が先、右が後)")
    print(f"{'ラグ(分)':>9s}{'指数→HL':>11s}{'perp→HL':>11s}{'指数→perp':>12s}")
    for k in range(-L, L + 1):
        j = k + L
        v = [C[n][j] / N[n][j] if N[n][j] else np.nan
             for n in ("idx_hl", "prp_hl", "idx_prp")]
        print(f"{k:>9d}{v[0]:>11.4f}{v[1]:>11.4f}{v[2]:>12.4f}")
    print()
    print("[3] 指数 -> perp を時間帯で分ける(正 = 指数が先)")
    print(f"{'ラグ(分)':>9s}" + "".join(f"{n:>13s}" for n, _, _ in SS))
    for k in range(-L, L + 1):
        j = k + L
        row = f"{k:>9d}"
        for si in range(len(SS)):
            row += (f"{CS[si, j]/NS_[si, j]:>13.4f}" if NS_[si, j]
                    else f"{'-':>13s}")
        print(row)
    print()
    print("★夜(00-08)で指数が perp に遅れるなら、その時間帯の指数は")
    print("  現物ではなく perp 由来である。現物を参照先にはできない。")
    print("\n★ラグ 0 が突出し、±1 分が小さいなら、現物 → perp の遅れは"
          "\n  **1 分より短く、この粒度では測れない**ということ。"
          "\n  200ms の先行を測るには sub-second の現物フィードが要る。")


if __name__ == "__main__":
    main()
