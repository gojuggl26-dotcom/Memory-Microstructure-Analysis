"""入口の門を入れたときのメイカー損益を、7 銘柄の**評価期間だけ**で比べる。

    uv run python scripts/analyze_mmgate.py

入力: data/inv_days_xyz_<coin>_q1.csv                門なし
      data/inv_days_xyz_<coin>_q1_gatedby_mmg.csv    門あり(`build_mmgate.py` の門)
      data/inv_days_xyz_<coin>_q1_gatedby_mmgrnd.csv 無作為の門(帰無対照)
      data/inv_days_xyz_<coin>_q1_lat130*.csv        遅延つき
      data/mmgate_fit_<coin>.csv                     学習・評価の別
出力: data/mmgate_all.csv

## 判定の設計

* **評価期間だけを使う。** 門は前 60% の日だけで学習してある。学習期間を混ぜると
  標本内の当てはまりを成績として報告することになる。
* **対照は 3 つ置く。** (a) 何もしない = 0(メイカーは出さなければ損益 0)、
  (b) 門なし、(c) **無作為の門**(同じ日・同じ側で同じ本数だけ無作為に通す)。
  ★(c) が要る理由: 門を入れると発注が減るだけで 1 組あたり損益は上がりうる。
  実測でも `xyz:KIOXIA` は無作為の門だけで +1.98 → +2.97 bp に上がった。
  **門の中身の価値は「門あり − 無作為の門」**であって「門あり − 門なし」ではない。
* **同じ日で対にして比べる。** 門あり / 門なしを日ごとに引き算し、その差の系列に
  Newey-West をかける。
* **単価と総額を分ける。** 門は組数を減らすので、単価が上がっても総額は減りうる。
* **最悪日を出す。** 中央値だけでは尾が見えない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
NW_LAGS = 14


def nw(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 5:
        return np.nan, np.nan, n
    e = v - v.mean()
    s = float(e @ e) / n
    for L in range(1, min(NW_LAGS, n - 1) + 1):
        s += 2 * (1 - L / (NW_LAGS + 1)) * float(e[L:] @ e[:-L]) / n
    return float(v.mean()), float(np.sqrt(max(s, 1e-18) / n)), n


def load(tag, sfx):
    f = DATA / f"inv_days_{tag}{sfx}.csv"
    return pl.read_csv(f).sort("dt") if f.exists() else None


def main() -> None:
    rows = []
    for c in COINS:
        tag = f"xyz_{c}"
        base = load(tag, "_q1")
        if base is None:
            continue
        fit = DATA / f"mmgate_fit_{tag}.csv"
        if fit.exists():
            F = pl.read_csv(fit)
            ev_days = set(F.filter(pl.col("train") == 0)["dt"].to_list())
        else:                       # 門が無い銘柄は後ろ 40% を評価期間とする
            d = base["dt"].to_list()
            ev_days = set(d[int(round(len(d) * 0.60)):])
        for lab, sfx in (("門なし", "_q1"),
                         ("無作為の門", "_q1_gatedby_mmgrnd"),
                         ("門あり", "_q1_gatedby_mmg"),
                         ("門なし+130ms", "_q1_lat130"),
                         ("門あり+130ms", "_q1_lat130_gatedby_mmg")):
            D = load(tag, sfx)
            if D is None:
                continue
            E = D.filter(pl.col("dt").is_in(list(ev_days)))
            if E.height < 5:
                continue
            pair = E["pair_pnl_mean"].to_numpy()
            npair = E["n_pair"].to_numpy().astype(float)
            tot = E["total_bp"].to_numpy()
            m, se, nd = nw(pair)
            mt, set_, _ = nw(tot)
            # ★ NW(ラグ 14)は 19 日しかない xyz:KIOXIA では仮定が苦しい。
            #   ラグ 0 の素の標準誤差と、分布を仮定しない符号検定も併記する。
            pf = pair[np.isfinite(pair)]
            se0 = (float(np.std(pf, ddof=1) / np.sqrt(pf.size))
                   if pf.size > 2 else np.nan)
            npos = int(np.sum(tot > 0))
            psign = float(stats.binomtest(npos, len(tot), 0.5).pvalue)
            rows.append({
                "coin": f"xyz:{c}", "cfg": lab, "n_days": nd,
                "se_lag0": se0, "t_lag0": m / se0 if se0 > 0 else np.nan,
                "p_sign": psign,
                "pairs_day": float(npair.mean()),
                "ev_pair_bp": float(np.nansum(pair * npair)
                                    / max(np.nansum(npair), 1)),
                "ev_pair_nw": m, "se": se, "t": m / se if se > 0 else np.nan,
                "day_bp": mt, "se_day": set_,
                "t_day": mt / set_ if set_ > 0 else np.nan,
                "p_day_pos": float(np.mean(tot > 0)),
                "worst_day_bp": float(tot.min()),
                "p_off_60s": float(E["p_off_60s"].to_numpy().mean()),
            })
        # 門あり − 対照 を同じ日で対にする。対照は「門なし」と「無作為の門」の
        # 両方。無作為の門との差が、門の**中身**の価値である。
        g = load(tag, "_q1_gatedby_mmg")
        for ctl_lab, ctl_sfx in (("門なし", "_q1"),
                                 ("無作為の門", "_q1_gatedby_mmgrnd")):
            ctl = load(tag, ctl_sfx)
            if g is None or ctl is None:
                continue
            j = (ctl.select("dt", "pair_pnl_mean", "total_bp")
                 .join(g.select("dt", "pair_pnl_mean", "total_bp"),
                       on="dt", suffix="_g")
                 .filter(pl.col("dt").is_in(list(ev_days))))
            dpair = (j["pair_pnl_mean_g"] - j["pair_pnl_mean"]).to_numpy()
            dtot = (j["total_bp_g"] - j["total_bp"]).to_numpy()
            m, se, nd = nw(dpair)
            mt, set_, _ = nw(dtot)
            rows.append({"coin": f"xyz:{c}", "cfg": f"差(門あり−{ctl_lab})",
                         "n_days": nd, "pairs_day": np.nan,
                         "se_lag0": float(np.nanstd(dpair, ddof=1)
                                          / np.sqrt(max(nd, 1))),
                         "t_lag0": (m / (np.nanstd(dpair, ddof=1)
                                         / np.sqrt(max(nd, 1)))
                                    if nd > 2 else np.nan),
                         "p_sign": float(stats.binomtest(
                             int(np.sum(dtot > 0)), len(dtot), 0.5).pvalue),
                         "ev_pair_bp": float(np.nanmean(dpair)),
                         "ev_pair_nw": m, "se": se,
                         "t": m / se if se > 0 else np.nan,
                         "day_bp": mt, "se_day": set_,
                         "t_day": mt / set_ if set_ > 0 else np.nan,
                         "p_day_pos": float(np.mean(dtot > 0)),
                         "worst_day_bp": float(dtot.min()),
                         "p_off_60s": np.nan})

    S = pl.DataFrame(rows)
    S.write_csv(DATA / "mmgate_all.csv")
    nt = S.filter(~pl.col("cfg").str.starts_with("差(")).height
    thr = float(stats.norm.ppf(1 - 0.05 / nt / 2))
    print(f"評価期間だけ。設定 {nt} 通り / Bonferroni の |t| 閾値 {thr:.2f}\n")
    print(f"{'銘柄':<11}{'設定':<16}{'日数':>5}{'組/日':>9}{'1 組 bp':>10}"
          f"{'NW se':>8}{'t':>8}{'t(ラグ0)':>9}{'符号検定 p':>12}"
          f"{'日次 bp':>11}{'黒字日':>7}{'最悪日':>11}")
    for c in COINS:
        for r in S.filter(pl.col("coin") == f"xyz:{c}").iter_rows(named=True):
            sig = "★" if abs(r["t"]) > thr else " "
            print(f"xyz:{c:<7}{r['cfg']:<16}{r['n_days']:>5}"
                  f"{r['pairs_day']:>9,.0f}{r['ev_pair_bp']:>10.3f}"
                  f"{r['se']:>8.3f}{r['t']:>7.2f}{sig}"
                  f"{r['t_lag0']:>9.2f}{r['p_sign']:>12.2g}"
                  f"{r['day_bp']:>11,.0f}"
                  f"{r['p_day_pos']*100:>6.0f}%{r['worst_day_bp']:>11,.0f}")
        print()


if __name__ == "__main__":
    main()
