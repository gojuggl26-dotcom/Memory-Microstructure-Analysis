r"""★凍結した構成を、未見期間(2026-09-06 以降)で 1 回だけ評価する。

=============================================================================
なぜこれが要るか(要求 2・5)
=============================================================================
08-18〜09-05 の期間は私が既に全部見ている。特徴量の順位表も相関行列も
リードラグも、その期間を見た上で書いた。したがって
**その期間の walk-forward も「探索的」にしかならない**。
選択過程を含めた補正の代わりに、**一度も見ていない期間での最終検定**を使う。

  凍結したもの: config/lighter_frozen.json(モデル・閾値の規則・数量・決済・費用)
  未見期間     : 2026-09-06 以降の完全な日だけ(部分日は使わない)
  ★この評価は 1 回だけ。結果が悪くても構成を変えて再試行しない。
    再試行したらそれは探索であって検定ではない。

閾値と符号は「テスト日の直前 7 日」で毎回引き直す(凍結したのは**規則**であって
数値ではない。実運用が再較正するのと同じ。較正窓はテスト日より厳密に前)。

出力: E:/Memory-lighter/wf/oos.parquet(日 × 銘柄の損益)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_wf import (SYMS, block_mask, day_str, evaluate,
                        prep)  # noqa: E402

SRC = Path("E:/Memory-lighter/wf")
ROOT = Path(__file__).resolve().parent.parent
OOS_START = "2026-09-06"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMS))
    ap.add_argument("--oos-days", default="2026-09-06,2026-09-07")
    ap.add_argument("--shard", default="", help="銘柄ごとに別ファイルへ")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config" / "lighter_frozen.json").read_text("utf-8"))
    sname = cfg["model"]
    q = float(cfg["threshold_rule"]["q"])
    wdays = int(cfg["threshold_rule"]["window_days"])
    h = float(cfg["exit"]["seconds"])
    oos = a.oos_days.split(",")
    print(f"凍結構成: {sname} / q={q} / h={h}s / 較正窓 {wdays} 日")
    print(f"未見日: {oos}")

    rows = []
    for sym in a.symbols.split(","):
        try:
            ts, days, X, cols, Y, COST = prep(sym)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        ci = {c: i for i, c in enumerate(cols)}
        if sname not in ci:
            print(f"{sym}: 凍結した signal {sname} が無い")
            continue
        sig = X[:, ci[sname]].astype(np.float64)
        for dstr in oos:
            d = int(np.datetime64(dstr).astype("datetime64[D]").astype(int))
            cal0, cal1 = d - wdays, d - 1
            m_cal = block_mask(days, ts, cal0, cal1)
            m_te = block_mask(days, ts, d, d)
            if m_cal.sum() < 50000 or m_te.sum() < 5000:
                print(f"{sym} {dstr}: 較正 {int(m_cal.sum())} / "
                      f"テスト {int(m_te.sum())} 行、飛ばす")
                continue
            v = sig[m_cal]
            yv = Y[60.0][m_cal]
            g = np.isfinite(v) & np.isfinite(yv)
            if g.sum() < 5000:
                continue
            c = np.corrcoef(v[g], yv[g])[0, 1]
            if not np.isfinite(c) or c == 0:
                continue
            sgn = float(np.sign(c))
            thr = float(np.quantile(np.abs(v[np.isfinite(v)]), q))
            res, nfire = evaluate(sig, thr, sgn, ts, days, Y, COST, h, m_te)
            for dd, (tot, n, gro, cst) in res.items():
                rows.append({"symbol": sym, "day": day_str(dd),
                             "n_trade": n, "pnl_sum": tot,
                             "pnl_mean": tot / max(n, 1),
                             "gross_sum": gro, "cost_sum": cst,
                             "thr": thr, "sign": sgn, "signal": sname,
                             "q": q, "h": h})
            print(f"  {sym} {dstr}: 建玉 {nfire} / "
                  f"純 {sum(v2[0] for v2 in res.values()):+.1f} "
                  f"= 粗 {sum(v2[2] for v2 in res.values()):+.1f} "
                  f"− 費 {sum(v2[3] for v2 in res.values()):.1f} bp "
                  f"(符号 {sgn:+.0f}, 閾値 {thr:.4g})", flush=True)
    O = pl.DataFrame(rows)
    O.write_parquet(SRC / (f"oos_{a.shard}.parquet" if a.shard
                           else "oos.parquet"))

    print("\n===== 未見期間の結果(この評価は 1 回だけ)=====")
    if O.height == 0:
        print("行が無い")
        return 0
    tot = float(O["pnl_sum"].sum()); n = int(O["n_trade"].sum())
    print(f"銘柄 {O['symbol'].n_unique()} / 日 {O['day'].n_unique()} / "
          f"取引 {n:,}")
    print(f"総額 {tot:+.2f} bp / 1 取引あたり {tot / max(n, 1):+.4f} bp")
    print(f"『何もしない』= 0 との比較: {'勝ち' if tot > 0 else '負け'}")
    rng = np.random.default_rng(7)
    days = O["day"].unique().to_list()
    by = {d: O.filter(pl.col("day") == d) for d in days}
    t = np.array([by[d]["pnl_sum"].sum() for d in days])
    print(f"日別総額: " + " / ".join(f"{d} {x:+.2f}" for d, x in zip(days, t)))
    if len(days) >= 2:
        idx = rng.integers(0, len(days), size=(5000, len(days)))
        bt = t[idx].sum(axis=1)
        print(f"日ブートストラップ 95% CI: "
              f"[{np.percentile(bt, 2.5):+.2f}, {np.percentile(bt, 97.5):+.2f}] bp")
        print(f"★日が {len(days)} 日しかないので、この CI は 2 点の再抽出に過ぎず"
              f"検出力はほぼ無い。期間が伸びたら同じ凍結構成で測り直すこと")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
