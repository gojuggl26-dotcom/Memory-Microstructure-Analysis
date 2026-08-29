"""テスト資金 100 USD で DRAM 単体を運用した場合の E[PnL] と APY 分布。

【資金の使われ方】
  在庫上限 = 5×Q 単位 → 最大建玉 = 5×Q×価格 [USD]。
  これが証拠金 100 USD で賄えるかが制約。レバレッジ L なら
      Q_max = 100 × L / (5 × 価格)
  レバレッジは「最大まで使う」前提なので、清算余裕を持たない上限値である。
"""
import sys; sys.path.insert(0,'scripts')
import numpy as np, polars as pl
from pathlib import Path
from capacity_v3 import load, simulate
from backtester_v3 import load_model

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PX = 53.7                      # 期間中央値。Q の決定にのみ使う(損益は実価格)
CAP = 100.0
LEV = [1, 3, 5, 10]
SIZES = {L: CAP * L / (5 * PX) for L in LEV}
print("テスト資金 100 USD / 在庫上限 = 5×Q 単位 / 価格中央値 53.7 USD")
for L, q in SIZES.items():
    print(f"  レバレッジ {L:>2}× → Q = {q:.3f} 単位 (1 件 {q*PX:>5.1f} USD, 最大建玉 {5*q*PX:>6.0f} USD)")
print()

model = load_model()
days = sorted(p.stem for p in (D/"slope_spline").glob("*.npz"))
days = [x for x in days if x not in set(model[4])]
rows = []
for i, dt in enumerate(days):
    d = load(dt, model)
    if d is None:
        continue
    for L, q in SIZES.items():
        r = simulate(d, q, 5.0 * q)
        if r:
            rows.append({"dt": dt, "lev": L, **r})
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(days)} 日", flush=True)
    pl.DataFrame(rows).write_csv(D / "capacity_100usd.csv")
pl.DataFrame(rows).write_csv(D / "capacity_100usd.csv")
print(f"完了 {len(set(r['dt'] for r in rows))} 日")
