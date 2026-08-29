"""生データ(L2 板 + 約定)から、1 イベントごとの 4 系列を算出する。

    (1) bid_ask_spread   = price_ask − price_bid                       [USDC]
    (2) order_imbalance  = (Volume_buy − Volume_sell)
                           / (Volume_buy + Volume_sell)                 ← 約定量ベース
    (3) depth_one_level  = (Quantity_bid − Quantity_ask)
                           / (Quantity_bid + Quantity_ask)              ← 板の数量ベース
    (4) voi              = ΔQuantity_bid_t − ΔQuantity_ask_t
        ただし ΔQuantity は **t−1 と t の最良気配(価格と数量の両方)** から決める:
            ΔQ^b_t = 0                        (P^b_t < P^b_{t−1})  気配が下がった = 旧レベルは消滅
                   = Q^b_t − Q^b_{t−1}        (P^b_t = P^b_{t−1})  同価格なら素の差分
                   = Q^b_t                    (P^b_t > P^b_{t−1})  気配が上がった = 全量が新規
            ΔQ^a_t は売り側で対称(不等号を逆に)
        これは Shen (2015) の VOI。素の差分版は voi_raw_diff として併記する

イベント = 最良気配 (best_bid, best_ask, bid_sz, ask_sz) が変化した瞬間(ns)。
(2) だけ板ではなく約定(node_fills)が情報源なので、直前イベントから当該イベントまでの
区間 (ts_{t−1}, ts_t] に起きた約定を集計する。約定の方向はテイカー(crossed=True)の側で決める。

(4) の注意: 指定どおりの素の差分は、最良気配の**価格が変わった時に別の価格水準の数量を
引き算する**ため符号が反転しうる。比較用に Cont-Kukanov-Stoikov の価格条件つき OFI
(`ofi_cont`)も併せて出す。どちらを使うかは利用側の判断。

日跨ぎ: ΔQuantity は前日最終イベントを引き継いで計算する(初日の初回のみ null)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/microprice")   # ts, best_bid/ask, bid/ask_sz
FILLS = Path("data/fills_v99")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")

SCHEMA = {
    "ts": pl.Int64, "best_bid": pl.Float64, "best_ask": pl.Float64,
    "bid_sz": pl.Float64, "ask_sz": pl.Float64,
    "bid_ask_spread": pl.Float64, "bid_ask_spread_bp": pl.Float64,
    "depth_one_level": pl.Float64,
    "d_vol_bid": pl.Float64, "d_vol_ask": pl.Float64, "voi": pl.Float64,
    "voi_raw_diff": pl.Float64, "ofi_cont": pl.Float64,
    "vol_buy": pl.Float64, "vol_sell": pl.Float64, "n_trades": pl.Int32,
    "order_imbalance": pl.Float64,
    "is_crossed": pl.Boolean,
}


def load_taker_fills(dt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """テイカー行だけ返す(ts_ns, sz, is_buy)。取引 1 件はメイカー/テイカーの 2 行で入る。"""
    p = FILLS / f"dt={dt}"
    files = sorted(p.rglob("*.parquet")) if p.exists() else []
    if not files:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=bool)
    df = pl.concat([pl.read_parquet(f, columns=["ts", "sz", "side", "crossed", "tid"])
                    for f in files], how="diagonal_relaxed")
    df = (df.filter(pl.col("crossed"))                 # テイカー = 板を食った側
            .unique(subset=["tid"], keep="first")
            .with_columns(pl.col("ts").cast(pl.Int64).alias("ts_ns"))
            .sort("ts_ns"))
    return (df["ts_ns"].to_numpy(), df["sz"].to_numpy(),
            (df["side"].to_numpy() == "B"))


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    out_base = OUT_DIR / "data" / "features"
    out_base.mkdir(parents=True, exist_ok=True)
    carry: dict | None = None          # 前日最終イベントの気配と数量
    daily = []

    for i, dt in enumerate(days, 1):
        df = pl.read_parquet(
            SRC / f"dt={dt}" / "part-000.parquet",
            columns=["ts", "best_bid", "best_ask", "bid_sz", "ask_sz", "spread_bp", "is_crossed"],
        ).sort("ts")
        ts = df["ts"].to_numpy()
        bb, ba = df["best_bid"].to_numpy(), df["best_ask"].to_numpy()
        qb, qa = df["bid_sz"].to_numpy(), df["ask_sz"].to_numpy()
        n = ts.size

        # ---- (1) スプレッド ----
        spread = ba - bb

        # ---- (3) 板の 1 段目インバランス ----
        depth1 = (qb - qa) / (qb + qa)

        # ---- (4) ΔQuantity と VOI(前日から引き継ぐ) ----
        pb = np.empty(n)
        pa = np.empty(n)
        pqb = np.empty(n)
        pqa = np.empty(n)
        pb[1:], pa[1:], pqb[1:], pqa[1:] = bb[:-1], ba[:-1], qb[:-1], qa[:-1]
        if carry is None:
            pb[0] = pa[0] = pqb[0] = pqa[0] = np.nan
        else:
            pb[0], pa[0], pqb[0], pqa[0] = carry["bb"], carry["ba"], carry["qb"], carry["qa"]
        # 素の差分(初版の指定式。価格が動くと別レベルの数量を引くので比較用に残す)
        d_bid_raw = qb - pqb
        d_ask_raw = qa - pqa
        voi_raw = d_bid_raw - d_ask_raw

        # ★ t−1 と t の最良気配(価格+数量)から決める ΔQuantity = Shen (2015) の VOI
        d_bid = np.where(bb > pb, qb, np.where(bb < pb, 0.0, qb - pqb))
        d_ask = np.where(ba < pa, qa, np.where(ba > pa, 0.0, qa - pqa))
        voi = d_bid - d_ask

        # Cont-Kukanov-Stoikov の OFI(気配が不利に動いた時に 0 でなく −Q_{t−1} を使う版)
        e_b = np.where(bb > pb, qb, np.where(bb < pb, -pqb, qb - pqb))
        e_a = np.where(ba < pa, qa, np.where(ba > pa, -pqa, qa - pqa))
        ofi = e_b - e_a
        bad = np.isnan(pb)
        for a in (d_bid, d_ask, voi, ofi):
            a[bad] = np.nan

        # ---- (2) 約定量インバランス: 区間 (ts_{t−1}, ts_t] の約定を集計 ----
        f_ts, f_sz, f_buy = load_taker_fills(dt)
        vol_b = np.zeros(n)
        vol_s = np.zeros(n)
        n_tr = np.zeros(n, dtype=np.int32)
        assigned = 0
        if f_ts.size:
            idx = np.searchsorted(ts, f_ts, side="left")   # 約定を「以降で最初のイベント」に割り当てる
            ok = idx < n
            assigned = int(ok.sum())
            ii, ss, bu = idx[ok], f_sz[ok], f_buy[ok]
            np.add.at(vol_b, ii[bu], ss[bu])
            np.add.at(vol_s, ii[~bu], ss[~bu])
            np.add.at(n_tr, ii, 1)
        tot = vol_b + vol_s
        with np.errstate(invalid="ignore", divide="ignore"):
            oi = np.where(tot > 0, (vol_b - vol_s) / tot, np.nan)

        out = pl.DataFrame({
            "ts": ts, "best_bid": bb, "best_ask": ba, "bid_sz": qb, "ask_sz": qa,
            "bid_ask_spread": spread, "bid_ask_spread_bp": df["spread_bp"].to_numpy(),
            "depth_one_level": depth1,
            "d_vol_bid": d_bid, "d_vol_ask": d_ask, "voi": voi,
            "voi_raw_diff": voi_raw, "ofi_cont": ofi,
            "vol_buy": vol_b, "vol_sell": vol_s, "n_trades": n_tr,
            "order_imbalance": oi,
            "is_crossed": df["is_crossed"].to_numpy(),
        }).cast(SCHEMA).with_columns(
            # NaN のままだと is_not_null() をすり抜ける。欠測は null に統一する
            [pl.col(c).fill_nan(None) for c in
             ("order_imbalance", "d_vol_bid", "d_vol_ask", "voi", "voi_raw_diff",
              "ofi_cont")]
        )
        d = out_base / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        out.write_parquet(d / "part-000.parquet", compression="zstd", statistics=True)

        carry = {"bb": bb[-1], "ba": ba[-1], "qb": qb[-1], "qa": qa[-1]}
        daily.append({
            "dt": dt, "n_events": n,
            "n_trades_taker": int(f_ts.size), "n_trades_assigned": assigned,
            "vol_buy": float(vol_b.sum()), "vol_sell": float(vol_s.sum()),
            "events_with_trade": int((n_tr > 0).sum()),
            "spread_mean": float(np.nanmean(spread)),
            "depth1_mean": float(np.nanmean(depth1)),
            "voi_mean": float(np.nanmean(voi)), "voi_std": float(np.nanstd(voi)),
            "voi_raw_std": float(np.nanstd(voi_raw)),
            "ofi_mean": float(np.nanmean(ofi)), "ofi_std": float(np.nanstd(ofi)),
            "oi_mean": float(np.nanmean(oi)) if np.isfinite(oi).any() else None,
        })
        print(f"[{i:3d}/{len(days)}] {dt} events={n:>8,} trades={f_ts.size:>7,} "
              f"({assigned:,} 割当) 約定のあったイベント={int((n_tr > 0).sum()):>7,}", flush=True)

    pl.DataFrame(daily).write_csv(OUT_DIR / "data" / "features_daily.csv")
    print(json.dumps({"days": len(daily),
                      "events": sum(d["n_events"] for d in daily),
                      "trades": sum(d["n_trades_taker"] for d in daily),
                      "assigned": sum(d["n_trades_assigned"] for d in daily)}, indent=1))


if __name__ == "__main__":
    main()
