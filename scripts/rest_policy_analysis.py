"""「トレード頻度を落とすべきか」の判定。

【判定の設計(CLAUDE.md §B)】
  ・主判定は **施策 vs 何もしない**の対応差で行う(0 との比較ではない)。
    「何もしない」= 現行の追いかける方策 k=0
  ・**格子は全セル報告**する。9 個の k × 2 個の遅延 = 18 セルを探索しているので
    Bonferroni 閾値を併記する
  ・**発火率でなく実行可能性の下限**を課す: 発注レート ≤ 6.6 操作/秒 を満たさない
    セルは、どれだけ成績が良くても採用できない
  ・**標本外の方策選択**を行う: k を「その日より前の日だけ」で選び、その日で評価する。
    全期間を見てから最良 k を選ぶのは選択バイアスそのもの

【決定的な検証】
  頻度を落とす主張の核心は「レースに参加しなければ遅延は効かない」である。
  よって判定は **遅延 100ms のもとで k>0 が k=0 に勝つか**。
"""
from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
RATE_LIMIT = 6.6            # 操作/秒(取引所仕様。報告 33 §4。本検証では未再検証)

df = pl.read_csv(D / "rest_policy.csv")
ks = sorted(df["k_ticks"].unique().to_list())
lats = sorted(df["lat_ms"].unique().to_list())
days = sorted(df["dt"].unique().to_list())
n = len(days)
print(f"標本 {n} 日 / k {ks} / 遅延 {lats} ms  = {len(ks)*len(lats)} セル")


def series(k, lat, col="pnl_usd"):
    return (df.filter((pl.col("k_ticks") == k) & (pl.col("lat_ms") == lat))
            .sort("dt")[col].to_numpy())


for lat in lats:
    print(f"\n{'='*78}")
    print(f"=== 遅延 {lat} ms ===")
    base = series(0, lat)                     # 対照 = 現行(追いかける)
    print(f"{'k':>4}{'E[PnL]':>10}{'中央値':>9}{'黒字':>7}{'発注/日':>10}"
          f"{'操作/秒':>9}{'δ bp':>8}{'約定/日':>9}{'日中最小':>10}"
          f"{'Δ対k=0':>10}{'改善日':>8}{'p(対応)':>10}")
    for k in ks:
        u = series(k, lat)
        d = u - base
        pl_ = df.filter((pl.col("k_ticks") == k) & (pl.col("lat_ms") == lat))
        ops = float(pl_["ops_per_s"].median())
        pt = stats.ttest_1samp(d, 0).pvalue if k != 0 else np.nan
        mark = "  " if ops <= RATE_LIMIT else " ✗"      # ✗ = 発注レート超過
        print(f"{k:>4}{u.mean():>10.3f}{np.median(u):>9.3f}"
              f"{int((u>0).sum()):>5}/{n:<2}{float(pl_['n_place'].median()):>10,.0f}"
              f"{ops:>8.2f}{mark}{float(pl_['gross_bp'].median()):>8.4f}"
              f"{float(pl_['n_fill'].median()):>9,.0f}"
              f"{float(pl_['eq_min'].median()):>10.2f}"
              f"{d.mean():>10.3f}{int((d>0).sum()):>6}/{n:<2}"
              f"{pt:>10.4f}" if k != 0 else
              f"{k:>4}{u.mean():>10.3f}{np.median(u):>9.3f}"
              f"{int((u>0).sum()):>5}/{n:<2}{float(pl_['n_place'].median()):>10,.0f}"
              f"{ops:>8.2f}{mark}{float(pl_['gross_bp'].median()):>8.4f}"
              f"{float(pl_['n_fill'].median()):>9,.0f}"
              f"{float(pl_['eq_min'].median()):>10.2f}"
              f"{'—':>10}{'—':>8}{'(対照)':>10}")

    # 実行可能なセルだけに絞った判定
    feas = [k for k in ks
            if float(df.filter((pl.col("k_ticks") == k) & (pl.col("lat_ms") == lat))
                     ["ops_per_s"].median()) <= RATE_LIMIT]
    print(f"\n  発注レートを満たす k: {feas}")
    if feas:
        best = max(feas, key=lambda k: series(k, lat).mean())
        u = series(best, lat); d = u - base
        p = stats.ttest_1samp(d, 0).pvalue
        ps = stats.binomtest(int((d > 0).sum()), n, 0.5, "greater").pvalue
        print(f"  実行可能なうち最良 k={best}: E[PnL] {u.mean():+.3f}"
              f"  (対 k=0 で {d.mean():+.3f}, p={p:.4f}, 符号 {int((d>0).sum())}/{n} p={ps:.4f})")
        print(f"  ※ {len(ks)*len(lats)} セル探索の Bonferroni 閾値 = "
              f"{0.05/(len(ks)*len(lats)):.4f}"
              f" → {'通る' if p < 0.05/(len(ks)*len(lats)) else '通らない'}")
        print(f"  絶対値の判定(0 と比較): p={stats.ttest_1samp(u,0).pvalue:.4f}"
              f"  黒字 {int((u>0).sum())}/{n}")

# --- 標本外の方策選択 ---
print(f"\n{'='*78}")
print("=== 標本外の方策選択(k を過去の日だけで選び、当日で評価)===")
for lat in lats:
    feas = [k for k in ks
            if float(df.filter((pl.col("k_ticks") == k) & (pl.col("lat_ms") == lat))
                     ["ops_per_s"].median()) <= RATE_LIMIT]
    if not feas:
        print(f"  遅延 {lat}ms: 実行可能な k が無い")
        continue
    mat = {k: series(k, lat) for k in feas}
    base = series(0, lat)
    oos = []; chosen = []
    for j in range(5, n):                    # 最初の 5 日は選択用
        sc = {k: mat[k][:j].mean() for k in feas}
        k_sel = max(sc, key=sc.get)
        chosen.append(k_sel); oos.append(mat[k_sel][j])
    oos = np.array(oos)
    b = base[5:]
    d = oos - b
    print(f"  遅延 {lat}ms  標本外 {len(oos)} 日: E[PnL] {oos.mean():+.3f}"
          f"  (現行 k=0 は {b.mean():+.3f}, 差 {d.mean():+.3f},"
          f" 改善 {int((d>0).sum())}/{len(d)}, p={stats.ttest_1samp(d,0).pvalue:.4f})")
    print(f"    選ばれた k: {chosen}")

# --- 遅延に対する頑健性(これが本題)---
print(f"\n{'='*78}")
print("=== ★遅延に対する頑健性: 遅延を入れると各 k はどれだけ落ちるか ===")
print(f"{'k':>4}{'遅延0':>10}{'遅延100ms':>12}{'低下':>10}{'低下率':>9}{'操作/秒':>9}")
for k in ks:
    a = series(k, 0).mean(); b = series(k, 100).mean()
    ops = float(df.filter((pl.col("k_ticks") == k) & (pl.col("lat_ms") == 100))
                ["ops_per_s"].median())
    print(f"{k:>4}{a:>10.3f}{b:>12.3f}{b-a:>10.3f}"
          f"{(b/a-1)*100 if a != 0 else float('nan'):>8.0f}%{ops:>9.2f}")
