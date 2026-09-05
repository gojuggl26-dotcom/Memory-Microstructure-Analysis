"""在庫シミュレータの全設定を評価期間で集計する。

    uv run python scripts/build_inv_summary.py --coin xyz:MU

`build_inventory.py` が設定ごとに書いた日次 CSV を読み、**評価期間 39 日だけ**で
1 組あたりの往復損益と日次 Newey-West の標準誤差を出す。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NW_LAGS = 14


def nw(v):
    n = v.size
    if n < 5:
        return np.nan, np.nan, n
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, NW_LAGS + 1):
        g += 2.0 * (1 - lg / (NW_LAGS + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    te = {f.stem.split("=")[1] for f in
          sorted(Path("E:/Memory-quotes") .joinpath(tag).glob("dt=*.parquet"))[59:]}
    rows = []
    for f in sorted(DATA.glob(f"inv_days_{tag}_*.csv")):
        nm = f.stem.replace(f"inv_days_{tag}_", "")
        if re.search(r"_d\d+$", nm):          # 試験実行は除く
            continue
        D = pl.read_csv(f)
        D = D.filter(pl.col("dt").is_in(list(te)))
        if not D.height:
            continue
        v = D["pair_pnl_mean"].to_numpy()
        v = v[np.isfinite(v)]
        m, se, n = nw(v)
        npr = int(D["n_pair"].sum())
        nf = int(D["n_fill"].sum())
        # ★ ev_pair は「メイカー同士で相殺できた組」だけの単価であり、
        #   強制決済された建玉を含まない。timeout を短くすると強制決済が
        #   88% に達し、残った組は速くて有利なものばかりになるので
        #   ev_pair が +0.39 と最良に見える(総額は 1 日 −38,068 bp)。
        #   主指標は必ず「建てた玉 1 本あたり」と「1 日あたり総額」で見る。
        nto = int(D["n_timeout"].sum()) if "n_timeout" in D.columns else 0
        nlot = npr + int(D["forced_n"].sum()) + nto
        r = {"cfg": nm, "days": n, "pairs": npr, "pairs_day": npr / D.height,
             "lots": nlot, "lots_day": nlot / D.height,
             "ev_lot": float(D["total_bp"].sum()) / max(nlot, 1),
             "ev_pair": m, "se": se, "t": m / se if se == se and se > 0 else np.nan,
             "off60": float(np.nanmean(D["p_off_60s"].to_numpy())),
             "t_off_med": float(np.nanmedian(D["t_off_med"].to_numpy())),
             "forced_rate": float(D["forced_n"].sum()) / max(nf, 1),
             "total_bp": float(D["total_bp"].sum()),
             "per_day_bp": float(D["total_bp"].sum()) / D.height}
        if "n_timeout" in D.columns:
            r["timeout_rate"] = float(D["n_timeout"].sum()) / max(nf, 1)
            r["timeout_cost"] = (float(D["timeout_cost_bp"].sum())
                                 / max(int(D["n_timeout"].sum()), 1))
        rows.append(r)
    S = pl.DataFrame(rows).sort("ev_pair", descending=True)
    S.write_csv(DATA / f"inv_summary_{tag}.csv")
    print(S.select("cfg", "pairs_day", "ev_pair", "se", "t", "off60",
                   "forced_rate", "per_day_bp"))
    print(f"\n書き出し {DATA}/inv_summary_{tag}.csv")


if __name__ == "__main__":
    main()
