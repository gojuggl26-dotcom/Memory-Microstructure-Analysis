r"""会場間リードラグ — Binance MUUSDT(perp)と Hyperliquid xyz:MU。

    uv run python scripts/xv_leadlag.py --coin xyz:MU
    uv run python scripts/xv_leadlag.py --coin xyz:MU --cell 100 --days 10

出力:
    data/xv_ccf_<coin>.csv      … ラグ別の相互相関(全体 / 時間帯別)
    data/xv_dev_<coin>.csv      … 乖離 d の日次統計と半減期
    data/xv_daily_<coin>.csv    … 日次の素性(更新率・基差・欠測)

=============================================================================
★この測定が「関門」である理由
=============================================================================
提案された戦略の利益源は **外部の Fair Value − HL の現在価格** である。
したがって成立条件は 2 つだけ:

  (1) Binance が HL より**先に**動く(先行が無ければ FV は FV でない)
  (2) その乖離が**自分の遅延より長く**生きる(すぐ埋まるなら間に合わない)

xyz:MU では信号の半減期が約 250ms で、2.49 秒古いだけで候補 2 の勾配が
消えた前例がある(`mu_cand2_fresh_report.md`)。**先に半減期を測る。**

=============================================================================
★踏んではいけない罠(先に書いておく)
=============================================================================
1. **時刻規約が違う。** Binance `transact_time` は約定が起きたマッチング
   エンジン時刻(ms)、HL の ts はブロック/イベント時刻(ns)。**両会場の
   クロック差と伝送遅延は観測できない。**よってラグは「曲線」として出し、
   点推定を「Binance が N ms 先行」と断言しない。戦略側では**自分が知れる
   時刻**(= Binance 打刻 + 保守的な遅延)でしか使わない

2. **片側が疎だと偽の先行が出る。** Binance の株式 perp は 24/7 だが、
   米国時間外は約定が薄い。LOCF で格子に写すと「動いていない側が遅行」に
   見える。**両側の鮮度(最終更新からの経過)で必ず絞る**
   (Uniswap で実際に踏んだ: 1 秒格子だと AMM の更新は 8% しかなく、
    素朴に写すと市場構造によらず「CEX 先行」が出た)

3. **基差を全標本の中央値で抜かない。** HL と Binance には資金調達と
   需給由来の持続的な差がある。中心は**過去 30 分の移動中央値**で取る
   (厳禁事項 §4: 全標本から標準化のパラメータを作らない)

4. **最終約定値は mid ではない。** Binance 側を生の約定値のまま使うと、
   自分の板幅ぶんの**跳ね返り**が乖離に入る。取れない見かけの乖離で
   半減期も |d| も水増しされるので、符号つき約定から実効半スプレッド h を
   推定して `mid ≒ price − sgn·h` に直す。推定は Glosten–Harris 型の
   Δp = h·Δsgn + ε の OLS(日ごと。**その日の中で閉じた測定**であって
   予測には使わない)

5. **クロス板を混ぜない。** best_ask < best_bid の行は除く
   (少数の異常行が結論を支配する — CLAUDE.md の既知の実例)
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
BNC = Path("E:/Binance-perp-data/data/history/parquet/kind=aggTrades")
BP = 1e4
NS = 1_000_000_000
# 時間帯(UTC)。提案の区切りに合わせる
SESS = [("00-08 夜", 0, 8), ("08-13.5 プレ", 8, 13.5),
        ("13.5-20 現物", 13.5, 20), ("20-24 アフター", 20, 24)]


def sess_of(hour):
    out = np.full(hour.shape, -1, dtype=np.int8)
    for i, (_, a, b) in enumerate(SESS):
        out[(hour >= a) & (hour < b)] = i
    return out


def load_hl(f):
    D = pl.read_parquet(f)
    ts = D["ts"].to_numpy()
    bid = D["best_bid"].to_numpy()
    ask = D["best_ask"].to_numpy()
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bid > 0)
    return ts[ok], 0.5 * (bid[ok] + ask[ok])


def load_bnc(sym, day):
    f = BNC / f"symbol={sym}" / f"{sym}-aggTrades-{day}.parquet"
    if not f.exists():
        return None, None, None
    D = pl.read_parquet(f, columns=["transact_time", "price", "quantity",
                                    "is_buyer_maker"])
    t = D["transact_time"].to_numpy().astype(np.int64) * 1_000_000  # ms -> ns
    px = D["price"].to_numpy().astype(np.float64)
    # is_buyer_maker=True は「買い手がメイカー」= テイカーは売り
    sgn = np.where(D["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    sv = sgn * D["quantity"].to_numpy().astype(np.float64)
    o = np.argsort(t, kind="stable")
    return t[o], px[o], sv[o]


def debounce(px, sgn):
    """符号つき約定から実効半スプレッド h を出し、mid ≒ px − sgn·h に直す。

    Δp_t = h·(s_t − s_{t−1}) + ε の切片なし OLS。h は板の半値幅に相当する。
    """
    dp = np.diff(px)
    ds = np.diff(sgn)
    den = float((ds * ds).sum())
    h = float((dp * ds).sum() / den) if den > 0 else 0.0
    h = max(h, 0.0)
    return px - sgn * h, h


def grid_locf(t_src, v_src, grid):
    """格子へ backward asof で写す。値と「最終更新からの経過 ns」を返す。"""
    i = np.searchsorted(t_src, grid, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(t_src) - 1)
    v = np.where(ok, v_src[i], np.nan)
    age = np.where(ok, grid - t_src[i], np.inf)
    return v, age


def roll_med(x, w):
    """過去 w 点の移動中央値(自分を含む)。未来は見ない。"""
    return (pl.Series(x).rolling_median(w, min_samples=max(10, w // 20))
            .to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sym", default="")
    ap.add_argument("--cell", type=int, default=100, help="格子 (ms)")
    ap.add_argument("--lags", type=int, default=20, help="±何セル")
    ap.add_argument("--fresh", type=float, default=2.0,
                    help="両側の鮮度の上限 (秒)")
    ap.add_argument("--strict", action="store_true",
                    help="両側が**そのセル内で実際に更新された**行だけを使う")
    ap.add_argument("--sfx", default="")
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    sym = a.sym or (a.coin.split(":")[-1] + "USDT")
    cell = a.cell * 1_000_000                       # ns
    nlag = a.lags

    fs = sorted(glob.glob(str(DATA / f"bbo_l1_{tag}" / "*.parquet")))
    days = [os.path.basename(f).split("=")[1].split(".")[0] for f in fs]
    if a.days:
        fs, days = fs[:a.days], days[:a.days]
    print(f"{a.coin} × {sym} / 格子 {a.cell}ms / ラグ ±{nlag} セル "
          f"(±{nlag*a.cell/1000:g}s) / 鮮度 {a.fresh}s 以内", flush=True)

    ccf_n = np.zeros((len(SESS) + 1, 2 * nlag + 1))
    ccf_s = np.zeros_like(ccf_n)
    daily, devrows, per_day = [], [], []
    for f, day in zip(fs, days):
        th, mh = load_hl(f)
        tb, pb, sb = load_bnc(sym, day)
        h_bnc = np.nan
        if tb is not None and tb.size > 1000:
            pb, h_bnc = debounce(pb, np.sign(sb))
        if tb is None or th.size < 1000 or tb.size < 1000:
            daily.append({"dt": day, "ok": False, "n_hl": int(th.size),
                          "n_bnc": 0 if tb is None else int(tb.size)})
            continue
        d0 = (th[0] // (86400 * NS)) * 86400 * NS
        grid = np.arange(d0, d0 + 86400 * NS, cell, dtype=np.int64)
        vh, ah = grid_locf(th, mh, grid)
        vb, ab = grid_locf(tb, pb, grid)
        good = np.isfinite(vh) & np.isfinite(vb) \
            & (ah <= a.fresh * NS) & (ab <= a.fresh * NS)
        lh, lb = np.log(vh), np.log(vb)
        # 基差を過去 30 分の移動中央値で抜く(未来を見ない)
        w = max(10, int(30 * 60 * 1000 / a.cell))
        raw = (lb - lh) * BP
        dev = raw - roll_med(np.where(good, raw, np.nan), w)
        rh = np.concatenate([[np.nan], np.diff(lh)]) * BP
        rb = np.concatenate([[np.nan], np.diff(lb)]) * BP
        hour = ((grid - d0) / (3600 * NS)).astype(np.float64)
        sid = sess_of(hour)

        # ---- 相互相関: corr(rb[t], rh[t+k]) ----
        # k>0 なら「Binance が先、HL が後」= Binance 先行
        for k in range(-nlag, nlag + 1):
            x = rb[max(0, -k): rb.size - max(0, k)]
            y = rh[max(0, k): rh.size - max(0, -k)]
            g = good[max(0, -k): good.size - max(0, k)] \
                & good[max(0, k): good.size - max(0, -k)]
            g &= np.isfinite(x) & np.isfinite(y)
            if g.sum() < 100:
                continue
            c = np.corrcoef(x[g], y[g])[0, 1]
            j = k + nlag
            ccf_n[0, j] += 1
            ccf_s[0, j] += c
            per_day.append({"dt": day, "group": "全体", "lag_ms": int(k * a.cell),
                            "corr": float(c), "n": int(g.sum())})
            ss = sid[max(0, -k): sid.size - max(0, k)]
            for i in range(len(SESS)):
                gg = g & (ss == i)
                if gg.sum() < 100:
                    continue
                cc = np.corrcoef(x[gg], y[gg])[0, 1]
                if np.isfinite(cc):
                    ccf_n[i + 1, j] += 1
                    ccf_s[i + 1, j] += cc
                    per_day.append({"dt": day, "group": SESS[i][0],
                                    "lag_ms": int(k * a.cell),
                                    "corr": float(cc), "n": int(gg.sum())})

        # ---- 乖離の半減期(AR(1) を切片なしで)----
        dd = np.where(good, dev, np.nan)
        x0, x1 = dd[:-1], dd[1:]
        m = np.isfinite(x0) & np.isfinite(x1)
        phi = float((x0[m] * x1[m]).sum() / max((x0[m] ** 2).sum(), 1e-12)) \
            if m.sum() > 100 else np.nan
        hl_ms = (np.log(0.5) / np.log(phi) * a.cell
                 if np.isfinite(phi) and 0 < phi < 1 else np.nan)
        devrows.append({
            "dt": day, "phi": phi, "half_life_ms": hl_ms,
            "dev_abs_p50": float(np.nanmedian(np.abs(dd))),
            "dev_abs_p90": float(np.nanquantile(np.abs(dd), 0.90)),
            "dev_sd": float(np.nanstd(dd)),
            "basis_p50": float(np.nanmedian(np.where(good, raw, np.nan))),
        })
        daily.append({
            "dt": day, "ok": True, "n_hl": int(th.size), "n_bnc": int(tb.size),
            "cells": int(grid.size), "usable": int(good.sum()),
            "usable_pct": float(100 * good.mean()),
            "hl_fresh_pct": float(100 * np.mean(ah <= a.fresh * NS)),
            "bnc_fresh_pct": float(100 * np.mean(ab <= a.fresh * NS)),
            "hl_move_pct": float(100 * np.mean(np.abs(rh[good]) > 1e-9)),
            "bnc_move_pct": float(100 * np.mean(np.abs(rb[good]) > 1e-9)),
            "bnc_half_spread_bp": float(BP * h_bnc / np.nanmedian(pb)),
        })
        if len(daily) % 20 == 0:
            print(f"  {len(daily)}/{len(fs)} {day}", flush=True)

    SF = a.sfx or (f"_c{a.cell}" + ("_strict" if a.strict else ""))
    pl.DataFrame(per_day).write_parquet(DATA / f"xv_ccfday_{tag}{SF}.parquet")
    D = pl.DataFrame(daily, infer_schema_length=None)
    D.write_csv(DATA / f"xv_daily_{tag}{SF}.csv")
    V = pl.DataFrame(devrows, infer_schema_length=None)
    V.write_csv(DATA / f"xv_dev_{tag}{SF}.csv")
    lags = np.arange(-nlag, nlag + 1) * a.cell
    rows = []
    names = ["全体"] + [s[0] for s in SESS]
    for i, nm in enumerate(names):
        for j, lg in enumerate(lags):
            if ccf_n[i, j] > 0:
                rows.append({"group": nm, "lag_ms": int(lg),
                             "corr": ccf_s[i, j] / ccf_n[i, j],
                             "n_days": int(ccf_n[i, j])})
    C = pl.DataFrame(rows)
    C.write_csv(DATA / f"xv_ccf_{tag}{SF}.csv")

    ok = D.filter(pl.col("ok"))
    print(f"\n有効 {ok.height} 日 / 全 {D.height} 日")
    print(f"使えるセル {ok['usable_pct'].mean():.1f}% "
          f"(HL 鮮度 {ok['hl_fresh_pct'].mean():.1f}% / "
          f"Binance 鮮度 {ok['bnc_fresh_pct'].mean():.1f}%)")
    print(f"セル内で動く割合: HL {ok['hl_move_pct'].mean():.1f}% / "
          f"Binance {ok['bnc_move_pct'].mean():.1f}%")
    print(f"Binance の実効半スプレッド 中央 "
          f"{ok['bnc_half_spread_bp'].median():.2f}bp(跳ね返りとして除去済み)")
    print(f"\n乖離 |d| 中央 {V['dev_abs_p50'].median():.2f}bp / "
          f"90% 点 {V['dev_abs_p90'].median():.2f}bp / "
          f"基差 中央 {V['basis_p50'].median():+.2f}bp")
    hlm = V["half_life_ms"].drop_nulls().drop_nans()
    print(f"乖離の半減期 中央 {hlm.median():.0f}ms "
          f"(四分位 {hlm.quantile(0.25):.0f}〜{hlm.quantile(0.75):.0f}ms)")
    print(f"\n{'ラグ(ms)':>10s}  " + "".join(f"{n:>12s}" for n in names))
    print("  ★正のラグ = Binance が先、HL が後")
    for lg in lags:
        if abs(lg) > 1000 and lg % 500:
            continue
        r = C.filter(pl.col("lag_ms") == int(lg))
        s = f"{int(lg):>10d}  "
        for nm in names:
            v = r.filter(pl.col("group") == nm)["corr"]
            s += f"{(v[0] if v.len() else float('nan')):>12.4f}"
        print(s)
    print(f"\n書き出し {DATA}/xv_{{ccf,dev,daily}}_{tag}.csv")


if __name__ == "__main__":
    main()
