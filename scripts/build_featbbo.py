"""板(bbo)と約定(fills)だけで作れる特徴量を、全所有銘柄ぶん作る。

    uv run python scripts/build_featbbo.py --coin xyz:AMD

出力: data/featbbo_<coin>/dt=*.parquet(1 秒格子で作って 5 秒おきに間引く)

## なぜ要るか

`build_featlib.py` の 229 本は **L1(注文 1 本ごとのイベント列)から板を
10 水準まで組み直して**作る。L1 が手元にあるのは `xyz:MU` と `xyz:INTC` だけで、
残りの銘柄は板の最良気配(`bbo_*.parquet`)と約定(`fills_*.parquet`)しか無い。
S3 から L1 を取り直すと 5 銘柄で 26GB・約 \\$2.6 かかるので、まず
**費用ゼロで作れる範囲**を全銘柄でそろえる。

## 何が作れて、何が作れないか

| 分類 | bbo + fills で | |
|---|---|---|
| 1 価格と最良気配 | **全部** | bbo |
| 2 板の偏り OBI | **最良気配の分だけ** | 多水準(obi2/3/5/10 等)は不可 |
| 3 板の厚み | **最良気配の分だけ** | 累積の厚みは不可 |
| 4 マイクロプライス | **最良気配の分だけ** | micro2/5/10・micro_slope は不可 |
| 5 板の形 / 6 隙間 / 7 弾力性 | 不可 | 多水準が要る |
| 8 OFI | **ほぼ全部** | 多水準の `ofi_lv*` だけ不可 |
| 9 指値フロー / 10 取消 / 11 到着 | 不可 | L1 のイベント列が要る |
| 12 攻撃的な約定フロー | **全部** | fills |

定義は `build_featlib.py` からそのまま持ってきている(補助関数は import する)ので、
`featlib_*` と同名の列は同じ式である。ただし **`featlib` の最良気配の数量は
L1 から組み直したラダーの最良段**、こちらは **bbo の `bid_sz`/`ask_sz`** なので、
同じ列名でも出どころが違う。比較するときは全銘柄をこちら(`featbbo`)でそろえる。

## 時間契約

すべて格子点 T までの情報だけで作る。窓は後ろ向き、差分は過去との差。
`ofi_pct` の基準は**前日の分布**(全標本の分位を使わない)。
約定は `[T−1s, T)` に入ったものを格子点 T に載せる(featlib の 2026-09-05 の
修正と同じ。`[T, T+1)` に入れると目的変数 (T, T+h] と重なる)。

x が確定する時刻 / y の期間: 本スクリプトは x だけを作る。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_featlib import (GRID_S, KEEP_EVERY, NG, OFI_WIN, rstd,  # noqa: E402
                           rsum, safe, streak, tsince, zsc)
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000


def day_feats(d: pl.DataFrame, fd, prev_ofi):
    ts = d["ts"].cast(pl.Int64).to_numpy()
    d0 = int(ts[0]) // DAY_NS * DAY_NS
    tg = d0 + np.arange(NG, dtype=np.int64) * 10 ** 9
    pb = d["best_bid"].to_numpy()
    pa = d["best_ask"].to_numpy()
    qb1 = d["bid_sz"].to_numpy()
    qa1 = d["ask_sz"].to_numpy()
    gidx = np.searchsorted(ts, tg, side="right") - 1
    okg = gidx >= 0
    gi = np.maximum(gidx, 0)
    F = {}

    # --- 1. 価格と最良気配 ------------------------------------------------
    F["best_bid"] = np.where(okg, pb[gi], np.nan)
    F["best_ask"] = np.where(okg, pa[gi], np.nan)
    mid = 0.5 * (F["best_bid"] + F["best_ask"])
    F["mid"] = mid
    F["log_mid"] = np.log(mid)
    F["spread_abs"] = F["best_ask"] - F["best_bid"]
    tick = np.where(mid >= 1000.0, 0.1, 0.01)
    F["tick"] = tick
    F["spread_tick"] = F["spread_abs"] / tick
    F["spread_bp"] = F["spread_abs"] / mid * 1e4
    F["spread_rel"] = F["spread_abs"] / mid
    for nm, v in (("bid", F["best_bid"]), ("ask", F["best_ask"]),
                  ("mid", mid), ("spread", F["spread_abs"])):
        F[f"d_{nm}"] = np.diff(v, prepend=v[0])
    F["t_since_mid_move"] = tsince(np.abs(F["d_mid"]) > 1e-12)
    F["t_since_bbo_chg"] = tsince((np.abs(F["d_bid"]) > 1e-12)
                                  | (np.abs(F["d_ask"]) > 1e-12))
    F["bbo_age"] = F["t_since_bbo_chg"]
    F["spread_duration"] = tsince(np.abs(F["d_spread"]) > 1e-12)

    # --- 2/3. 最良気配の偏りと厚み ----------------------------------------
    QB = np.where(okg, qb1[gi], np.nan)
    QA = np.where(okg, qa1[gi], np.nan)
    F["qb1"] = QB
    F["qa1"] = QA
    F["cum_b1"] = QB
    F["cum_a1"] = QA
    F["depth_bbo"] = QB + QA
    F["dollar_depth"] = QB * F["best_bid"] + QA * F["best_ask"]
    o1 = safe(QB - QA, QB + QA, np.nan)
    F["obi1"] = o1
    F["obi_abs"] = np.abs(o1)
    F["obi_sign"] = np.sign(o1)
    F["depth_diff"] = QB - QA
    F["depth_ratio"] = safe(QB, QA, np.nan)
    F["log_depth_ratio"] = np.log(np.maximum(F["depth_ratio"], 1e-9))
    d1 = np.diff(o1, prepend=o1[0])
    F["d_obi"] = d1
    F["obi_vel"] = d1 / GRID_S
    F["obi_acc"] = np.diff(F["obi_vel"], prepend=F["obi_vel"][0])
    F["obi_persist"] = tsince(np.concatenate(
        [[True], np.sign(o1[1:]) != np.sign(o1[:-1])]))
    F["obi_streak"] = streak(np.sign(np.nan_to_num(o1)))
    F["d_depth"] = np.diff(F["depth_bbo"], prepend=F["depth_bbo"][0])
    F["depth_vol"] = rstd(np.nan_to_num(F["depth_bbo"]), 60)

    # --- 4. マイクロプライス(最良気配) ----------------------------------
    hs = F["spread_bp"] / 2.0
    F["micro1"] = mid * (1.0 + hs * o1 / 1e4)
    F["delta1"] = mid * hs * o1 / 1e4
    F["delta1_bp"] = hs * o1
    F["obi_x_spread"] = o1 * F["spread_bp"]
    m1 = F["micro1"]
    F["d_micro"] = np.diff(m1, prepend=m1[0])
    F["micro_ret"] = np.diff(np.log(m1), prepend=0.0) * 1e4
    F["micro_mom"] = rsum(np.nan_to_num(F["micro_ret"]), 10)
    F["micro_persist"] = tsince(np.concatenate(
        [[True], np.sign(F["micro_ret"][1:]) != np.sign(F["micro_ret"][:-1])]))
    F["micro_minus_bid"] = (m1 - F["best_bid"]) / mid * 1e4
    F["micro_minus_ask"] = (m1 - F["best_ask"]) / mid * 1e4

    # --- 8. OFI -----------------------------------------------------------
    e = np.concatenate([[0.0], (
        (pb[1:] >= pb[:-1]) * qb1[1:] - (pb[1:] <= pb[:-1]) * qb1[:-1]
        - ((pa[1:] <= pa[:-1]) * qa1[1:] - (pa[1:] >= pa[:-1]) * qa1[:-1]))])
    cofi = np.cumsum(e)
    cofi_g = np.where(okg, cofi[gi], np.nan)
    for w in OFI_WIN:
        j = np.searchsorted(ts, tg - int(w * 1e9), side="right") - 1
        F[f"ofi_{w:g}s"] = cofi_g - np.where(j >= 0, cofi[np.maximum(j, 0)], np.nan)
    of1 = F["ofi_1s"]
    F["ofi_dollar"] = of1 * mid
    F["ofi_norm"] = safe(of1, F["depth_bbo"], np.nan)
    cnt = np.concatenate([[0], np.ones(ts.size - 1)]).cumsum()
    cg = np.where(okg, cnt[gi], np.nan)
    j1 = np.searchsorted(ts, tg - 10 ** 9, side="right") - 1
    nev = cg - np.where(j1 >= 0, cnt[np.maximum(j1, 0)], np.nan)
    F["ofi_per_event"] = safe(of1, nev, np.nan)
    F["ofi_per_time"] = of1 / 1.0
    F["ofi_cum"] = cofi_g
    al = 2.0 / 11.0
    ew = np.zeros(NG)
    v = np.nan_to_num(of1)
    for i in range(1, NG):
        ew[i] = al * v[i] + (1 - al) * ew[i - 1]
    F["ofi_ewma"] = ew
    F["d_ofi"] = np.diff(of1, prepend=of1[0])
    F["ofi_vel"] = F["d_ofi"] / GRID_S
    F["ofi_acc"] = np.diff(F["ofi_vel"], prepend=F["ofi_vel"][0])
    F["ofi_sign"] = np.sign(of1)
    F["ofi_persist"] = tsince(np.concatenate(
        [[True], np.sign(of1[1:]) != np.sign(of1[:-1])]))
    F["ofi_streak"] = streak(np.sign(np.nan_to_num(of1)))
    F["ofi_reversal"] = (np.sign(of1) != np.sign(
        np.concatenate([[of1[0]], of1[:-1]]))).astype(float)
    F["ofi_z"] = zsc(np.nan_to_num(of1), 300)
    F["ofi_shock"] = (np.abs(F["ofi_z"]) > 3).astype(float)
    if prev_ofi is None:
        F["ofi_pct"] = np.full(NG, np.nan)
    else:
        F["ofi_pct"] = (np.searchsorted(prev_ofi, np.nan_to_num(of1),
                                        side="right") / prev_ofi.size)
    nxt_ofi = np.sort(np.nan_to_num(of1))

    # --- 12. 攻撃的な約定フロー ------------------------------------------
    if fd is not None:
        ft = fd["ts"].cast(pl.Int64).to_numpy()
        fsz = fd["sz"].to_numpy()
        fpx = fd["px"].to_numpy()
        fbuy = fd["side"].to_numpy() == "B"
        fs_ = np.clip((ft - d0) // 10 ** 9 + 1, 0, NG - 1)
        for nm, m in (("buy", fbuy), ("sell", ~fbuy)):
            v_ = np.zeros(NG)
            c_ = np.zeros(NG)
            np.add.at(v_, fs_[m], fsz[m])
            np.add.at(c_, fs_[m], 1.0)
            F[f"trade_{nm}_vol"] = v_
            F[f"trade_{nm}_cnt"] = c_
        dv = np.zeros(NG)
        np.add.at(dv, fs_, fsz * fpx * np.where(fbuy, 1.0, -1.0))
        F["trade_dollar_imb"] = dv
    else:
        for c in ("trade_buy_vol", "trade_sell_vol", "trade_buy_cnt",
                  "trade_sell_cnt", "trade_dollar_imb"):
            F[c] = np.zeros(NG)
    bvl, svl = F["trade_buy_vol"], F["trade_sell_vol"]
    bct, sct = F["trade_buy_cnt"], F["trade_sell_cnt"]
    F["trade_imb"] = safe(bvl - svl, bvl + svl)
    F["signed_vol"] = bvl - svl
    F["signed_cnt"] = bct - sct
    F["aggr_buy_int"] = rsum(bct, 10) / 10.0
    F["aggr_sell_int"] = rsum(sct, 10) / 10.0
    F["aggr_imb"] = safe(rsum(bvl, 10) - rsum(svl, 10), rsum(bvl, 10) + rsum(svl, 10))
    F["aggr_mom"] = rsum(F["signed_vol"], 10)
    F["aggr_acc"] = np.diff(F["aggr_mom"], prepend=F["aggr_mom"][0])
    F["aggr_shock"] = (np.abs(zsc(F["signed_vol"], 300)) > 3).astype(float)
    F["size_wgt_imb"] = safe(bvl * bvl - svl * svl, bvl * bvl + svl * svl)
    return F, nxt_ofi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--bbo-suffix", default="")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    # ★大きい銘柄(SKHX / SNDK)は板と約定を一度に読むとメモリが足りない。
    #   日ごとに読んで日ごとに捨てる。clean_bbo の規則 2(直前 21 行の中央値)は
    #   日の先頭 21 行だけ全体読みと結果が変わりうるが、1 日 35 万行に対して 21 行なので
    #   実質同じである。
    bl = pl.scan_parquet(DATA / f"bbo_{tag}{a.bbo_suffix}.parquet")
    days = sorted(bl.select("dt").unique().collect()["dt"].to_list())
    fl = pl.scan_parquet(DATA / f"fills_{tag}.parquet")
    out = DATA / f"featbbo_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[featbbo] {a.coin}: {len(days)} 日(日ごとに読む)", file=sys.stderr)

    prev_ofi = None
    nwrite = 0
    for k, dt in enumerate(days):
        d, ndrop = clean_bbo(bl.filter(pl.col("dt") == dt).collect())
        d = d.sort("ts")
        if d.height < 1000:
            print(f"  {dt}: 板が {d.height} 行しかないので飛ばす", file=sys.stderr)
            continue
        fd = (fl.filter((pl.col("dt") == dt) & pl.col("crossed"))
              .select("ts", "px", "sz", "side").collect().sort("ts"))
        fd = fd if fd.height else None
        F, prev_ofi = day_feats(d, fd, prev_ofi)
        sl = slice(0, NG, KEEP_EVERY)
        T = pl.DataFrame({"dt": [dt] * len(range(0, NG, KEEP_EVERY)),
                          "sec": np.arange(0, NG, KEEP_EVERY, dtype=np.int32),
                          **{c: np.asarray(v, dtype=np.float32)[sl]
                             for c, v in F.items()}})
        T.write_parquet(out / f"dt={dt}.parquet")
        nwrite += 1
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}  特徴量 {len(F)} 本", flush=True,
                  file=sys.stderr)
    print(f"[featbbo] {nwrite} 日 × {len(F)} 本 -> {out.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
