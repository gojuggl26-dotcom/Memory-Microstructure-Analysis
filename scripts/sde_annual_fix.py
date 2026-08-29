"""★年次分布の訂正 — SDE 版にも母数の不確実性を入れる。

【何が問題だったか】
  sde_simulate.py の年次集計は、実測 30 日の母数(λ_d, σ_d)を等確率で
  ブロック復元抽出して 365 日を作った。これは
  **「真の日次分布が観測した 30 日そのものだったら」**という条件付き分布で、
  **30 日から推定した平均自体の誤差**を含まない。
  結果、年次 sd が 82 USD という非現実的に狭い区間になった。

  ★これは報告 34 で一度犯して報告 35 で訂正したのと**同じ誤り**である
  (旧版の年次 sd 83 USD → 訂正版 301 USD)。SDE に置き換えても、
  「30 日しか観測していない」という事実は消えない。

【訂正】二重ブロックブートストラップ
  ① 30 日の**日型**をブロック復元抽出して「あり得た母集団」を作る(母数の不確実性)
  ② そこから 365 日を抽出し、各日について SDE が生成した 2,000 本の
     条件付き分布から 1 本引く(経路の不確実性 + 日中のブラウン運動)

  分散は Var = 365·σ²_within + 365²·s²/n となり、第 2 項が支配的になる。

【時間契約】予測模型ではない。既に生成した条件付き分布の再集計。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
CAP = 100.0
NB = 1000          # ★年次 1,000 本(ユーザー指定)
YR = 365
BLK = 2
rng = np.random.default_rng(20260818)

z = np.load(D / "sde_day_ensemble.npz")
day_pnl = z["pnl"]          # (30, 2000)
day_eqmin = z["eq_min"]     # (30, 2000)
nd, npath = day_pnl.shape
day_mean = day_pnl.mean(axis=1)

print(f"SDE の日型 {nd} 種 × 各 {npath:,} 本")
print(f"  日型ごとの平均: {day_mean.mean():+.3f} ± {day_mean.std(ddof=1):.3f} USD/日")
print(f"  日型内のばらつき(平均): {day_pnl.std(axis=1, ddof=1).mean():.3f}")


def blocks(n_src, n_out, size):
    nb = int(np.ceil(n_out / BLK))
    st = rng.integers(0, n_src, size=(size, nb))
    ix = (st[:, :, None] + np.arange(BLK)[None, None, :]) % n_src
    return ix.reshape(size, -1)[:, :n_out]


def annual(double: bool):
    if double:
        pop = blocks(nd, nd, NB)                 # ① 母集団の再抽出 (NB, 30)
        sel = blocks(nd, YR, NB)                 # ② そこから 365 日
        didx = np.take_along_axis(pop, sel, axis=1)
    else:
        didx = blocks(nd, YR, NB)
    pick = rng.integers(0, npath, size=(NB, YR))
    d = day_pnl[didx, pick]
    e = day_eqmin[didx, pick]
    return d, e


for nm, dbl in [("単一(旧・母数誤差なし)", False), ("★二重(訂正)", True)]:
    d, e = annual(dbl)
    ann = d.sum(axis=1)
    cum = np.cumsum(d, axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0), axis=1)
    mdd = (peak - cum).max(axis=1)
    ruin = ((cum - d + e) <= -CAP).any(axis=1)
    qs = [1, 5, 10, 25, 50, 70, 75, 80, 90, 95, 99]
    print(f"\n=== {nm} ===")
    print(f"  E[年次] {ann.mean():+,.0f} USD  sd {ann.std(ddof=1):,.0f}"
          f"  → APY {ann.mean()/CAP*100:,.0f}%")
    print("  APY 分位: " + "  ".join(f"p{q} {np.percentile(ann,q)/CAP*100:,.0f}%" for q in qs))
    print(f"  P(年次<0) {(ann<0).mean()*100:.2f}%"
          f"   P(日中に資金 100 USD 消尽) {ruin.mean()*100:.2f}%")
    print(f"  最大ドローダウン 中央値 {np.median(mdd):.0f}"
          f"  p90 {np.percentile(mdd,90):.0f}  p99 {np.percentile(mdd,99):.0f}"
          f"  最悪 {mdd.max():.0f}")
    if dbl:
        out = {"mean": float(ann.mean()), "sd": float(ann.std(ddof=1)),
               "apy_mean": float(ann.mean() / CAP * 100),
               "prob_neg": float((ann < 0).mean()), "prob_ruin": float(ruin.mean()),
               "mdd_med": float(np.median(mdd)), "mdd_p90": float(np.percentile(mdd, 90)),
               "mdd_p99": float(np.percentile(mdd, 99)), "mdd_max": float(mdd.max()),
               "apy_pct": {f"p{q}": float(np.percentile(ann, q) / CAP * 100) for q in qs}}
        (D / "sde_annual_fixed.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(D / "sde_annual_fixed.npz", ann=ann, mdd=mdd, ruin=ruin)

# 分散の分解
s_e = day_mean.std(ddof=1) / np.sqrt(nd)
within = day_pnl.std(axis=1, ddof=1).mean()
v1 = YR * (within ** 2 + day_mean.var(ddof=1))
v2 = YR ** 2 * s_e ** 2
print(f"\n=== 年次分散の内訳 ===")
print(f"  経路の不確実性 365·σ²      = {v1:>12,.0f}  (sd {np.sqrt(v1):>6.0f})")
print(f"  母数の不確実性 365²·s²/n   = {v2:>12,.0f}  (sd {np.sqrt(v2):>6.0f})")
print(f"  合計                        = {v1+v2:>12,.0f}  (sd {np.sqrt(v1+v2):>6.0f})")
print(f"  → 母数側が全体の {v2/(v1+v2)*100:.0f}%")
print(f"\n参考: 報告 35 の実測ブートストラップ(30 日の実測損益から)"
      f" = 平均 1,170 USD / sd 273")
