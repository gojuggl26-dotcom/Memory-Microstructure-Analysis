"""レベル別 OBI の情報量を集計する — 最良気配を 1 とした比。

【推論の単位】
  日。R² は日ごとに算出し、その中央値を報告する。
  プールした値は少数日に支配されるため使わない(regression_report.md §9)。

【比の取り方】
  ★日ごとに比 R²_L / R²_1 を作ってから中央値を取る。
    中央値どうしを割ると、日によって L1 の水準が違う影響が消えない。
  併せて符号一貫性(β が L1 と同符号だった日数)も出す。
  R² は符号を持たないので、これが無いと「効いている」とは言えない。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "obi_levels"
NL = 10
BURN = 7


def main() -> None:
    fs = sorted(glob.glob(str(OUT / "*.json")))
    days = [json.loads(Path(f).read_text(encoding="utf-8")) for f in fs][BURN:]
    n = len(days)
    if n < 8:
        print("日数不足")
        return
    print(f"=== {n} 日 / 1 秒グリッド / y = 1 秒先のリターン ===\n")

    lv_r2 = np.array([[d["level"][L]["r2"] for L in range(NL)] for d in days])
    lv_b = np.array([[d["level"][L]["beta"] for L in range(NL)] for d in days])
    cu_r2 = np.array([[d["cumulative"][L]["r2"] for L in range(NL)] for d in days])
    cu_b = np.array([[d["cumulative"][L]["beta"] for L in range(NL)] for d in days])
    inc = np.array([[d["incremental"][L]["delta"] for L in range(NL)] for d in days])
    nz = np.array([[d["level"][L]["nonzero"] for L in range(NL)] for d in days])

    def ratio(a):
        base = a[:, [0]]
        r = np.where(base > 1e-12, a / np.maximum(base, 1e-12), np.nan)
        return np.nanmedian(r, 0)

    res = {"n_days": n, "days": [d["dt"] for d in days]}
    print("【1】レベル別 OBI — L 段目だけの不均衡 I_L = (q_b^L − q_a^L)/(q_b^L + q_a^L)")
    print(f"{'段':>3}{'R²(中央値)':>13}{'★L1=1 の比':>13}{'β':>10}"
          f"{'β が L1 と同符号':>16}{'p':>9}{'その段が空でない率':>18}")
    rl = ratio(lv_r2)
    for L in range(NL):
        s1 = np.sign(np.median(lv_b[:, 0]))
        same = int((np.sign(lv_b[:, L]) == s1).sum())
        p = float(stats.binomtest(max(same, n - same), n, .5).pvalue)
        print(f"{L+1:>3}{np.median(lv_r2[:, L]):>13.5f}{rl[L]:>13.3f}"
              f"{np.median(lv_b[:, L]):>10.4f}{same:>12}/{n:<3}{p:>9.4f}"
              f"{np.median(nz[:, L])*100:>17.1f}%")
    res["level"] = {"r2_median": lv_r2.mean(0).tolist(), "ratio_to_L1": rl.tolist()}

    print("\n【2】累積 OBI(L) — 1 段目から L 段目まで(既存の obi テーブルと同じ定義)")
    print(f"{'段':>3}{'R²(中央値)':>13}{'★L1=1 の比':>13}{'β':>10}{'β が L1 と同符号':>16}")
    rc = ratio(cu_r2)
    for L in range(NL):
        s1 = np.sign(np.median(cu_b[:, 0]))
        same = int((np.sign(cu_b[:, L]) == s1).sum())
        print(f"{L+1:>3}{np.median(cu_r2[:, L]):>13.5f}{rc[L]:>13.3f}"
              f"{np.median(cu_b[:, L]):>10.4f}{same:>12}/{n:<3}")
    res["cumulative"] = {"r2_median": cu_r2.mean(0).tolist(), "ratio_to_L1": rc.tolist()}

    print("\n【3】増分 R² — 1..L−1 の重回帰に L 段目を足したときの増加")
    print("   「そのレベルが**新たに**持ち込む情報」。上の単変量 R² と違い重複を除いてある")
    print(f"{'段':>3}{'増分 R²':>12}{'★L1=1 の比':>13}{'正の日':>10}{'p':>9}")
    ri = ratio(inc)
    for L in range(NL):
        pos = int((inc[:, L] > 0).sum())
        p = float(stats.binomtest(pos, n, .5, alternative="greater").pvalue)
        print(f"{L+1:>3}{np.median(inc[:, L]):>12.5f}{ri[L]:>13.3f}{pos:>6}/{n:<3}{p:>9.4f}")
    res["incremental"] = {"delta_median": np.median(inc, 0).tolist(), "ratio_to_L1": ri.tolist()}

    tot = np.median(np.cumsum(inc, 1)[:, -1])
    print(f"\n  10 段全部を使った重回帰の R²(中央値) = {tot:.5f}")
    print(f"  1 段目だけの R²                      = {np.median(lv_r2[:, 0]):.5f}"
          f"  → 10 段で {tot/max(np.median(lv_r2[:,0]),1e-12):.2f} 倍")
    (D / "obi_levels_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
