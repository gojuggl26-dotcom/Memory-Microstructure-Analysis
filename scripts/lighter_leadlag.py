r"""Lighter × Binance リードラグの徹底測定(12 ペア)。

=============================================================================
設計
=============================================================================
【系列】両取引所とも BBO の mid(bookTicker / ticker)。板の奥は使わない。
【時計】2 通りで全部やる:
    server = 取引所が打つイベント時刻(Binance E[ms] / Lighter last_updated_at[µs])
    local  = このマシンの受信時刻(local_ns / recv_ns — ★同一マシンなので共通の物差し)
  取引所の時計はサーバ間でずれるので、server だけで結論しない。
【グリッド】Δ=1s(±30s の CCF・偏相関)と Δ=0.25s(±3s の細部・ホライズン曲線)。
  LOCF(backward asof)。直近更新が 10 秒より古い点は無効。
【単位】日 × 銘柄で推定 → 銘柄内で日をまたいで集計 → 銘柄横断は中央値。
【帰無対照】日ずらし(Binance の日 d × Lighter の日 d+1、同じ時刻帯)。
  日内の同時性は殺し、日周パターンだけ残る対照。
【層別】米国現物立会(13:30–20:00 UTC・平日)と、それ以外。

★リターンの符号規約: r(t) = log mid(t) − log mid(t−Δ) [bp]。
  CCF の k>0 は「Binance の r(t) と Lighter の r(t+kΔ)」= Binance 先行の証拠。

出力(E:/Memory-lighter/ll/):
  ccf.parquet      sym, day, clock, dt, k, session, rho, n
  partial.parquet  sym, day, clock, 方向, plain, partial, n
  hcurve.parquet   sym, day, clock, h, rho, n
  events_{SYM}.npz イベントスタディ(両方向の平均経路)
  clock.parquet    受信 − サーバ時刻の日次中央値(両取引所)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-lighter")
OUT = SRC / "ll"
BP = 1e4
PAIRS = ["DRAM", "MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD",
         "AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "XAG", "XAU"]
KS_1S = list(range(-30, 31))
KS_25 = list(range(-12, 13))
HCURVE = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
STALE = 10.0


def day_bounds(d0: str, d1: str):
    a = datetime.strptime(d0, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(d1, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = []
    t = a
    while t <= b:
        days.append((t.strftime("%Y-%m-%d"), t.timestamp(), t.timestamp() + 86400,
                     t.weekday()))
        t = datetime.fromtimestamp(t.timestamp() + 86400, tz=timezone.utc)
    return days


def rank(v: np.ndarray) -> np.ndarray:
    n = len(v)
    o = np.argsort(v, kind="stable")
    r = np.empty(n)
    r[o] = np.arange(1, n + 1)
    s = v[o]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j > i + 1:
            r[o[i:j]] = (i + 1 + j) / 2.0
        i = j
    return r



try:                                     # ★同順位処理は scipy の C 実装を使う
    from scipy.stats import rankdata as _rd

    def rank(v):
        return _rd(v, method="average")
except Exception:
    pass


def sp(x, y, min_n=300):
    m = np.isfinite(x) & np.isfinite(y)
    nm = int(m.sum())
    if nm < min_n:
        return np.nan, nm
    rx, ry = rank(x[m]), rank(y[m])
    sx, sy = rx.std(), ry.std()
    if sx <= 0 or sy <= 0:
        return np.nan, nm
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy)), nm


def partial_sp(y, x, c):
    """rank 化 → c を線形に除いた残差どうしの相関。"""
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(c)
    if m.sum() < 300:
        return np.nan, np.nan, int(m.sum())
    ry, rx, rc = rank(y[m]), rank(x[m]), rank(c[m])
    rc = rc - rc.mean()
    den = (rc * rc).sum()
    if den <= 0:
        return np.nan, np.nan, int(m.sum())
    ex = rx - rx.mean() - rc * ((rx * rc).sum() / den)
    ey = ry - ry.mean() - rc * ((ry * rc).sum() / den)
    if ex.std() <= 0 or ey.std() <= 0:
        return np.nan, np.nan, int(m.sum())
    plain = float(((rx - rx.mean()) * (ry - ry.mean())).mean()
                  / (rx.std() * ry.std()))
    part = float((ex * ey).mean() / (ex.std() * ey.std()))
    return plain, part, int(m.sum())


def grid_mid(t: np.ndarray, mid: np.ndarray, T: np.ndarray):
    j = np.searchsorted(t, T, side="right") - 1
    ok = j >= 0
    jj = np.maximum(j, 0)
    stale = T - t[jj]
    v = np.where(ok & (stale <= STALE), mid[jj], np.nan)
    return v


def load(sym: str):
    L = pl.read_parquet(SRC / f"bbo_{sym}.parquet")
    B = pl.read_parquet(SRC / f"bnb_{sym}.parquet")
    lt_s = L["ts_us"].to_numpy() / 1e6
    lt_l = L["recv_ns"].to_numpy() / 1e9
    lm = (L["bid"].to_numpy() + L["ask"].to_numpy()) / 2
    bt_s = B["event_ms"].to_numpy() / 1e3
    bt_l = B["local_ns"].to_numpy() / 1e9
    bm = (B["bid"].to_numpy() + B["ask"].to_numpy()) / 2
    # server 時刻は単調とは限らない(復元順)→ ソート
    oL = np.argsort(lt_s, kind="stable")
    oB = np.argsort(bt_s, kind="stable")
    return ((lt_s[oL], lt_l[oL], lm[oL]), (bt_s[oB], bt_l[oB], bm[oB]))


def run_symbol(sym: str, d0: str, d1: str, ccf, par, hcv, clk):
    (lt_s, lt_l, lm), (bt_s, bt_l, bm) = load(sym)
    days = day_bounds(d0, d1)
    ev_L = []          # Binance イベント → Lighter 経路
    ev_B = []          # Lighter イベント → Binance 経路
    ev_selfB = []
    ev_selfL = []

    for di, (day, t0, t1, wd) in enumerate(days):
        # 時計の診断(その日の受信 − サーバ)
        mL = (lt_s >= t0) & (lt_s < t1)
        mB = (bt_s >= t0) & (bt_s < t1)
        if mL.sum() < 1000 or mB.sum() < 1000:
            continue
        clk.append({"symbol": sym, "day": day,
                    "lighter_lag_ms": float(np.median((lt_l[mL] - lt_s[mL]) * 1e3)),
                    "binance_lag_ms": float(np.median((bt_l[mB] - bt_s[mB]) * 1e3)),
                    "nL": int(mL.sum()), "nB": int(mB.sum())})
        for clock, (Lt, Bt) in (("server", (lt_s, bt_s)),
                                ("local", (lt_l, bt_l))):
            # ---- Δ=1s: CCF ±30s + セッション層別 + 偏相関 ----
            T = np.arange(t0, t1, 1.0)
            gL = grid_mid(Lt, lm, T)
            gB = grid_mid(Bt, bm, T)
            rL = np.r_[np.nan, np.diff(np.log(gL))] * BP
            rB = np.r_[np.nan, np.diff(np.log(gB))] * BP
            sod = T - t0
            us = (wd < 5) & (sod >= 13.5 * 3600) & (sod < 20 * 3600)
            for sess, msk in (("all", np.ones(len(T), bool)),
                              ("US", us), ("OFF", ~us)):
                if clock == "local" and sess != "all":
                    continue
                a = np.where(msk, rB, np.nan)
                b = np.where(msk, rL, np.nan)
                for k in KS_1S:
                    if k >= 0:
                        r, nm = sp(a[:len(a) - k or None], b[k:])
                    else:
                        r, nm = sp(a[-k:], b[:len(b) + k])
                    ccf.append({"symbol": sym, "day": day, "clock": clock,
                                "dt": 1.0, "k": float(k), "session": sess,
                                "rho": r, "n": nm})
            pl_, pt, nm = partial_sp(rL[2:], rB[1:-1], rL[1:-1])
            par.append({"symbol": sym, "day": day, "clock": clock,
                        "dir": "B→L", "plain": pl_, "partial": pt, "n": nm})
            pl_, pt, nm = partial_sp(rB[2:], rL[1:-1], rB[1:-1])
            par.append({"symbol": sym, "day": day, "clock": clock,
                        "dir": "L→B", "plain": pl_, "partial": pt, "n": nm})

            # ---- Δ=0.25s: 細部 CCF + ホライズン曲線 ----
            T4 = np.arange(t0, t1, 0.25)
            gL4 = grid_mid(Lt, lm, T4)
            gB4 = grid_mid(Bt, bm, T4)
            rL4 = np.r_[np.nan, np.diff(np.log(gL4))] * BP
            rB4 = np.r_[np.nan, np.diff(np.log(gB4))] * BP
            for k in KS_25:
                if k >= 0:
                    r, nm = sp(rB4[:len(rB4) - k or None], rL4[k:])
                else:
                    r, nm = sp(rB4[-k:], rL4[:len(rL4) + k])
                ccf.append({"symbol": sym, "day": day, "clock": clock,
                            "dt": 0.25, "k": float(k), "session": "all",
                            "rho": r, "n": nm})
            # Binance 直近 1s リターン → Lighter の h 先リターン
            rb1 = np.full(len(T4), np.nan)
            rb1[4:] = (np.log(gB4[4:]) - np.log(gB4[:-4])) * BP
            lgL = np.log(gL4)
            for h in HCURVE:
                s = int(round(h / 0.25))
                yy = np.full(len(T4), np.nan)
                yy[:len(T4) - s] = (lgL[s:] - lgL[:-s]) * BP
                r, nm = sp(rb1, yy)
                hcv.append({"symbol": sym, "day": day, "clock": clock,
                            "h": h, "rho": r, "n": nm})

            # ---- イベントスタディ(server のみ)----
            if clock != "server":
                continue
            for src_r, src_g, dst_g, store_dst, store_self in (
                    (rb1, gB4, gL4, ev_L, ev_selfB),
                    (np.r_[np.full(4, np.nan),
                           (np.log(gL4[4:]) - np.log(gL4[:-4])) * BP],
                     gL4, gB4, ev_B, ev_selfL)):
                ab = np.abs(src_r)
                fin = np.isfinite(ab)
                if fin.sum() < 10000:
                    continue
                thr = max(3.0, float(np.nanpercentile(ab[fin], 99.9)))
                cand = np.flatnonzero(fin & (ab >= thr))
                last = -1e9
                pre, post = 20, 60          # −5s..+15s(0.25s 刻み)
                for i2 in cand:
                    if T4[i2] - last < 30.0:
                        continue
                    if i2 < pre or i2 + post >= len(T4):
                        continue
                    w_dst = dst_g[i2 - pre:i2 + post + 1]
                    w_src = src_g[i2 - pre:i2 + post + 1]
                    if not (np.isfinite(w_dst).all() and np.isfinite(w_src).all()):
                        continue
                    last = T4[i2]
                    sgn = np.sign(src_r[i2])
                    store_dst.append(sgn * (np.log(w_dst) - np.log(dst_g[i2])) * BP)
                    store_self.append(sgn * (np.log(w_src) - np.log(src_g[i2])) * BP)

        # ---- 帰無対照: 日ずらし(server・Δ=1s・k=−30..30)----
        if di + 1 < len(days):
            day2, u0, u1, _ = days[di + 1]
            T = np.arange(t0, t1, 1.0)
            gB = grid_mid(bt_s, bm, T)
            gL = grid_mid(lt_s, lm, np.arange(u0, u1, 1.0))
            rB = np.r_[np.nan, np.diff(np.log(gB))] * BP
            rL = np.r_[np.nan, np.diff(np.log(gL))] * BP
            for k in (-10, -5, -2, -1, 0, 1, 2, 5, 10):
                if k >= 0:
                    r, nm = sp(rB[:len(rB) - k or None], rL[k:])
                else:
                    r, nm = sp(rB[-k:], rL[:len(rL) + k])
                ccf.append({"symbol": sym, "day": day, "clock": "placebo_dayshift",
                            "dt": 1.0, "k": float(k), "session": "all",
                            "rho": r, "n": nm})

    np.savez_compressed(
        OUT / f"events_{sym}.npz",
        b2l=np.array(ev_L) if ev_L else np.zeros((0, 81)),
        b_self=np.array(ev_selfB) if ev_selfB else np.zeros((0, 81)),
        l2b=np.array(ev_B) if ev_B else np.zeros((0, 81)),
        l_self=np.array(ev_selfL) if ev_selfL else np.zeros((0, 81)))
    print(f"{sym}: 完了(イベント B→L {len(ev_L)} / L→B {len(ev_B)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(PAIRS))
    ap.add_argument("--start", default="2026-08-18")
    ap.add_argument("--end", default="2026-09-05")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ccf, par, hcv, clk = [], [], [], []
    for sym in a.symbols.split(","):
        run_symbol(sym, a.start, a.end, ccf, par, hcv, clk)
    tag = f"_{a.tag}" if a.tag else ""
    pl.DataFrame(ccf).write_parquet(OUT / f"ccf{tag}.parquet")
    pl.DataFrame(par).write_parquet(OUT / f"partial{tag}.parquet")
    pl.DataFrame(hcv).write_parquet(OUT / f"hcurve{tag}.parquet")
    pl.DataFrame(clk).write_parquet(OUT / f"clock{tag}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
