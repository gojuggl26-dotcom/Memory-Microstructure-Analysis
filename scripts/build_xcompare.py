"""所有銘柄を同じ物差しで横並びにする。

    uv run python scripts/build_xcompare.py --coins xyz:MU,xyz:INTC,xyz:AMD,...

2 つの表を作る:
  1. 市場の姿(xcompare_stats.csv)… スプレッド・厚み・約定頻度・ボラ・
     OBI の持続・気配の入れ替わりの、60 秒標本の中央値と四分位
  2. 予測力の横並び(xcompare_pred.csv)… 共通の主要特徴量が、各銘柄で
     どれだけ将来のマイクロプライスを説明するか(1s と 10s の前向き順位相関)

xfeat パネル(build_xfeat.py)と、その予測力(fit_xfeat.py --stage pool)が
先に要る。x が確定する時刻 / y の期間は xfeat と同じ。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = Path("E:/Memory-xfeat")

# 市場の姿を測る量(60 秒標本の分位を出す)
STAT_COLS = ["spread_bp", "spread_tick", "mid", "bid_sz", "ask_sz",
             "rv_10s", "delta_bp", "obi_abs", "bbo_chg_10s", "msg_rate_10s",
             "trd_rate_10s", "vol_10s", "trd_sz_10s", "vpin_10s", "ofi_norm_10s"]
# 横並びにする主要特徴量
KEY = ["delta_bp", "obi1", "obi_not1", "micro_bid", "ofi_1s", "ofi_10s",
       "ofi_norm_10s", "vol_imb_10s", "trd_imb_10s", "sgn_vol_10s",
       "rv_10s", "spread_bp", "obi1__dev", "delta_bp__dev", "obi1__x__spread_bp"]

# 原資産(レポート用)
UNDERLYING = {
    "xyz:MU": "マイクロン(メモリ)", "xyz:INTC": "インテル(ロジック/ファウンドリ)",
    "xyz:AMD": "AMD(ロジック)", "xyz:KIOXIA": "キオクシア(NAND)",
    "xyz:SKHX": "SK ハイニックス(メモリ)", "xyz:SMSN": "サムスン電子(メモリ)",
    "xyz:SNDK": "サンディスク(NAND)", "xyz:DRAM": "DRAM(メモリ・HIP-3 発)",
}


def q(x, p):
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if x.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", required=True)
    a = ap.parse_args()
    coins = a.coins.split(",")
    stat_rows, pred_rows = [], []
    for coin in coins:
        tag = coin.replace(":", "_")
        base = SRC / tag
        if not base.exists():
            print(f"[skip] {coin} パネルなし", flush=True)
            continue
        days = sorted(p.name[3:13] for p in base.glob("dt=*.parquet"))
        # 60 秒標本で市場の姿
        acc = {c: [] for c in STAT_COLS}
        for d in days:
            t = pl.read_parquet(base / f"dt={d}.parquet",
                                columns=[c for c in STAT_COLS if c])
            for c in STAT_COLS:
                acc[c].append(t[c].to_numpy()[::60])
        row = {"coin": coin, "underlying": UNDERLYING.get(coin, "?"), "days": len(days)}
        for c in STAT_COLS:
            x = np.concatenate(acc[c])
            row[f"{c}_med"] = q(x, 0.5)
            row[f"{c}_p25"] = q(x, 0.25)
            row[f"{c}_p75"] = q(x, 0.75)
        stat_rows.append(row)
        # 予測力
        fp = DATA / f"xfeat_pred_{tag}.csv"
        if fp.exists():
            pr = pl.read_csv(fp)
            for c in KEY:
                for h in ("1s", "10s"):
                    s = pr.filter((pl.col("col") == c) & (pl.col("h") == h))
                    if s.height:
                        pred_rows.append(dict(coin=coin, col=c, h=h,
                                              r_fwd=float(s["r_fwd"][0]),
                                              r_bwd=float(s["r_bwd"][0]),
                                              r_plc=float(s["r_plc"][0])))
        print(f"  {coin}: {len(days)} 日", flush=True)
    pl.DataFrame(stat_rows).write_csv(DATA / "xcompare_stats.csv")
    if pred_rows:
        pl.DataFrame(pred_rows).write_csv(DATA / "xcompare_pred.csv")
    # 要約を表示
    st = pl.DataFrame(stat_rows)
    print(st.select("coin", "mid_med", "spread_bp_med", "bid_sz_med",
                    "trd_rate_10s_med", "rv_10s_med"))


if __name__ == "__main__":
    main()
