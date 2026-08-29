r"""boros_pressure.py の市場別推定を集計する。

【判定の単位】
  全市場をプールした p 値は集塊を無視するので使わない。
  **市場ごとに推定した係数の符号の一貫性**(二項検定)を主判定にする。

【事前予測の向き】(結果を見る前に決めたもの)
  Boros の rate 空間では LONG が bid、SHORT が ask に相当する。したがって
      long  → 係数 **正**(買い側に数量が入れば金利は上がる)
      short → 係数 **負**
      diff  → 係数 **正**
  重みづけ τ をどれにしても向きの予測は同じ。
  表には両側二項検定の p を出し、予測どおりの向きかを別途記す。

【多重比較】 7 地平 × 18 系列 × 2 推定量 = 252 セル。Bonferroni 閾値 0.05/252 を併記。
  ★前報(84 セル)と合わせて同じデータを見ているので、閾値は**今回の全格子**で取る。

【帰無対照】 市場内で x を巡回シフトしたプラセボ。実測と同じ集計を通す。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

HORIZONS = [1, 2, 3, 5, 10, 20, 30]
TAGS = ["bq", "ew1", "ew4", "ew16", "ew64", "depth"]
TAGLAB = {"bq": "τ→0 最良のみ", "ew1": "τ=1", "ew4": "τ=4",
          "ew16": "τ=16 ★主", "ew64": "τ=64", "depth": "τ→∞ 総量"}
SIDES = ["long", "short", "diff"]
SIDEPRED = {"long": +1, "short": -1, "diff": +1}
SIGNALS = [f"{t}_{s}" for t in TAGS for s in SIDES]
PRED = {f"{t}_{s}": SIDEPRED[s] for t in TAGS for s in SIDES}
LABEL = {f"{t}_{s}": f"{TAGLAB[t]} {s}" for t in TAGS for s in SIDES}
# 7 地平 × 18 系列 × 2 推定量。格子は全件報告する(CLAUDE.md §C-10)
N_CELL = len(HORIZONS) * len(SIGNALS) * 2
BONF = 0.05 / N_CELL


def sign_test(v: np.ndarray) -> tuple[int, int, float, float]:
    """(正の数, 全体, z, 両側 p)"""
    v = v[np.isfinite(v)]
    n = len(v)
    p = int((v > 0).sum())
    if n == 0:
        return 0, 0, np.nan, np.nan
    z = (p - n / 2) / np.sqrt(n / 4)
    return p, n, z, float(stats.binomtest(p, n, 0.5).pvalue)


def tradeable() -> dict:
    """★実装可能性(CLAUDE.md §D-16)。

    Boros はオンチェーンなので、**同一ブロック内のイベントの間には入れない**。
    x を見てから y の期間に注文を出せるのは、区間 (t, t+k] がブロック境界を
    跨いでいるときだけである。跨がない窓では「予測できる」と言えても
    **執行できない**。地平ごとにその割合を測る。
    """
    out = {}
    tot = {k: [0, 0] for k in HORIZONS}
    for f in sorted(glob.glob(str(DATA / "event_book_*.parquet"))):
        e = pl.read_parquet(f, columns=["block", "ev_i", "extrapolated"]
                            ).filter(~pl.col("extrapolated"))
        if e.height < 200:
            continue
        b = e["block"].to_numpy()
        for k in HORIZONS:
            if len(b) <= k:
                continue
            tot[k][0] += int((b[k:] > b[:-k]).sum())   # 窓がブロックを跨ぐ
            tot[k][1] += len(b) - k
    for k in HORIZONS:
        n1, n0 = tot[k]
        out[k] = {"span_block": n1, "windows": n0,
                  "frac": n1 / n0 if n0 else np.nan}
    return out


def main() -> int:
    d = pl.read_parquet(DATA / "pressure_results.parquet")
    print(f"市場 {d['market'].n_unique()} / 推定 {d.height:,} 件")
    print(f"多重比較: {N_CELL} セル → Bonferroni 閾値 p < {BONF:.5f}\n")

    out = []
    for sg in SIGNALS:
        for k in HORIZONS:
            s = d.filter((pl.col("signal") == sg) & (pl.col("k") == k))
            if s.height == 0:
                continue
            b = s["beta"].to_numpy()
            bg = s["beta_gls"].to_numpy()
            bp = s["beta_placebo"].to_numpy()
            po, no, zo, pvo = sign_test(b)
            pg, ng, zg, pvg = sign_test(bg)
            pp, np_, zp, pvp = sign_test(bp)
            ratio = s["se_nw"].to_numpy() / np.where(s["se_plain"].to_numpy() > 0,
                                                     s["se_plain"].to_numpy(), np.nan)
            disagree = np.mean(np.sign(b) != np.sign(bg))
            out.append({
                "signal": sg, "k": k, "n_markets": no,
                "pred_sign": PRED[sg],
                "ols_pos": po, "ols_z": zo, "ols_p": pvo,
                "gls_pos": pg, "gls_z": zg, "gls_p": pvg,
                "placebo_pos": pp, "placebo_z": zp, "placebo_p": pvp,
                "med_pearson": float(np.nanmedian(s["pearson"].to_numpy())),
                "med_spearman": float(np.nanmedian(s["spearman"].to_numpy())),
                "med_t_nw": float(np.nanmedian(s["t_nw"].to_numpy())),
                "med_t_plain": float(np.nanmedian(s["t_plain"].to_numpy())),
                "med_se_ratio_nw_plain": float(np.nanmedian(ratio)),
                "med_rho_ar1": float(np.nanmedian(s["rho_ar1"].to_numpy())),
                "sign_disagree_ols_gls": float(disagree),
                "med_n": int(np.median(s["n"].to_numpy())),
                "med_n_eff": int(np.median(s["n_eff"].to_numpy())),
                "med_frac_zero": float(np.median(s["frac_zero"].to_numpy())),
            })

    r = pl.DataFrame(out)
    tr = tradeable()
    (DATA / "pressure_summary.json").write_text(
        json.dumps({"cells": out, "tradeable": tr}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    for sg in SIGNALS:
        s = r.filter(pl.col("signal") == sg)
        if s.height == 0:
            continue
        z0 = s["med_frac_zero"].to_numpy()[0]
        print(f"=== {LABEL[sg]}  事前予測の向き: {'+' if PRED[sg] > 0 else '−'}"
              f"  |  x が厳密に 0 の割合 {z0:.1%} ===")
        print(f"{'k':>3} {'市場':>4} {'OLS 正':>9} {'z':>6} {'p':>10} "
              f"{'GLS 正':>9} {'p':>10} {'プラセボ':>9} {'ρ(Pear)':>9} {'ρ(Spear)':>9} {'有効n':>8}")
        for row in s.iter_rows(named=True):
            star = "*" if row["ols_p"] < BONF else " "
            gstar = "*" if row["gls_p"] < BONF else " "
            print(f"{row['k']:>3} {row['n_markets']:>4} "
                  f"{row['ols_pos']:>4}/{row['n_markets']:<4} {row['ols_z']:>6.2f} "
                  f"{row['ols_p']:>9.2e}{star} "
                  f"{row['gls_pos']:>4}/{row['n_markets']:<4} {row['gls_p']:>9.2e}{gstar}"
                  f"{row['placebo_pos']:>5}/{row['n_markets']:<4} "
                  f"{row['med_pearson']:>+9.4f} {row['med_spearman']:>+9.4f} "
                  f"{row['med_n_eff']:>8,}")
        print()

    # --- ★τ の梯子: 減衰長を変えると信号がどう動くか ------------------------
    print("=== ★τ の梯子 — 最良をどれだけ重くするかで信号がどう動くか ===")
    print("    値は符号検定の z(事前予測の向きを正にそろえてある)")
    hdr = "  ".join(f"k={k:<4}" for k in HORIZONS)
    for s in SIDES:
        print(f"\n  [{s}]  予測の向き {'+' if SIDEPRED[s] > 0 else '−'}")
        print(f"{'重みづけ':<14} {hdr}   {'ゼロ率':>7} {'有効n':>9}")
        for t in TAGS:
            row, z0, ne = [], None, None
            for k in HORIZONS:
                q = r.filter((pl.col("signal") == f"{t}_{s}") & (pl.col("k") == k))
                if q.height == 0:
                    row.append("   --  "); continue
                v = q.row(0, named=True)
                z = v["ols_z"] * SIDEPRED[s]      # 予測どおりなら正になるよう符号を揃える
                star = "*" if v["ols_p"] < BONF else " "
                row.append(f"{z:+6.2f}{star}")
                z0, ne = v["med_frac_zero"], v["med_n_eff"]
            print(f"{TAGLAB[t]:<14} " + " ".join(row) +
                  f"   {z0:>6.1%} {ne:>9,}")

    print("\n=== ★標準誤差の分解 — 素の SE が外れる原因は不均一分散か系列相関か ===")
    print("    White/素 が 1 から離れる = 不均一分散 / NW/White が 1 から離れる = 系列相関")
    print(f"{'k':>3} {'White/素':>10} {'NW/White':>10} {'NW/素':>8}  (全系列・全市場の中央値)")
    for k in HORIZONS:
        s = d.filter(pl.col("k") == k)
        p_ = s["se_plain"].to_numpy(); w_ = s["se_white"].to_numpy()
        n_ = s["se_nw"].to_numpy()
        ok = (p_ > 0) & (w_ > 0)
        print(f"{k:>3} {np.median(w_[ok]/p_[ok]):>10.3f} "
              f"{np.median(n_[ok]/w_[ok]):>10.3f} {np.median(n_[ok]/p_[ok]):>8.3f}")

    print("\n=== 推定量の性質 ===")
    print(f"{'k':>3} {'NW/素 の SE 比':>14} {'素の t 中央':>12} {'NW の t 中央':>12} "
          f"{'AR1 ρ 中央':>11} {'OLS と GLS で符号不一致':>22}")
    for k in HORIZONS:
        s = r.filter((pl.col("k") == k) & (pl.col("signal") == "ew16_diff"))
        if s.height == 0:
            continue
        row = s.row(0, named=True)
        print(f"{k:>3} {row['med_se_ratio_nw_plain']:>14.2f} "
              f"{row['med_t_plain']:>12.3f} {row['med_t_nw']:>12.3f} "
              f"{row['med_rho_ar1']:>11.3f} {row['sign_disagree_ols_gls']:>21.1%}")
    print("\n=== ★実装可能性 — 窓 (t, t+k] がブロックを跨ぐ割合 ===")
    print("    跨がない窓は、予測できても**その間に注文を出せない**")
    for k in HORIZONS:
        v = tr[k]
        print(f"  k={k:>2}  {v['frac']:>7.2%}  ({v['span_block']:,} / {v['windows']:,} 窓)")

    print(f"\n-> {DATA / 'pressure_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
