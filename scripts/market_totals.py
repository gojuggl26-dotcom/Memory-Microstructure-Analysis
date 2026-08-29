"""事業性の分母を測る: 市場全体の出来高・スプレッド・手数料の総額。

【★訂正 2026-08-17】
  初版は 99 日全体から 4 日おきに標本した中央値を分母にしていた。
  DRAM の出来高は 5 月 14.8 万 → 7 月 329 万単位/日 と 22 倍成長しているため、
  この「中央値」は閑散期に引きずられ、評価窓(2026-07-10〜08-08)の実勢を
  6.5 倍過小評価していた。しかも戦略側の約定量(評価窓)と割ってシェアを
  出しており、分子と分母の期間が食い違っていた(CLAUDE.md §A-2 違反)。
  本版は日次の市場集計を全 99 日分作り(market_volume_daily.csv)、
  **評価窓の分母**で計算する。旧値は比較のため併記する。
"""
import glob
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
EV = ("2026-07-10", "2026-08-08")          # 戦略評価と同じ 30 日

f = D / "market_volume_daily.csv"
if not f.exists():                          # 全 99 日の日次集計(初回のみ)
    rows = []
    for dd in sorted(glob.glob("data/fills_v99/dt=*")):
        dt = dd.split("=")[1]
        fs = glob.glob(dd + "/**/*.parquet", recursive=True)
        if not fs:
            continue
        fl = pl.concat([pl.read_parquet(x, columns=["px", "sz", "crossed", "tid"])
                        for x in fs], how="diagonal_relaxed")
        tr = fl.filter(pl.col("crossed")).unique(subset=["tid"], keep="first")
        rows.append({"dt": dt, "n_tr": tr.height, "vol": float(tr["sz"].sum()),
                     "notional": float((tr["sz"] * tr["px"]).sum())})
    pl.DataFrame(rows).sort("dt").write_csv(f)
df = pl.read_csv(f)
ds = pl.read_csv(D / "daily_summary.csv")

ev = df.filter((pl.col("dt") >= EV[0]) & (pl.col("dt") <= EV[1]))
spr_ev = float(ds.filter((pl.col("dt") >= EV[0]) & (pl.col("dt") <= EV[1]))
               ["spread_bp_med"].median())
n_ev = float(ev["notional"].median())

print("=== DRAM 市場全体: 月別中央値(出来高は 22 倍成長している)===")
for a, b, nm in [("2026-05-04", "2026-05-31", "5月"), ("2026-06-01", "2026-06-30", "6月"),
                 ("2026-07-01", "2026-07-31", "7月"), ("2026-08-01", "2026-08-10", "8月")]:
    s = df.filter((pl.col("dt") >= a) & (pl.col("dt") <= b))
    print(f"  {nm}: {s.height:>2} 日  取引 {float(s['n_tr'].median()):>8,.0f} 件/日"
          f"  出来高 {float(s['vol'].median()):>10,.0f} 単位/日"
          f"  想定元本 {float(s['notional'].median())/1e6:>6.1f}M USD/日")

print(f"\n=== 評価窓 {EV[0]}〜{EV[1]}(戦略の数字と同じ期間)===")
print(f"  取引件数   中央値 {float(ev['n_tr'].median()):>10,.0f} 件/日")
print(f"  出来高     中央値 {float(ev['vol'].median()):>10,.0f} 単位/日")
print(f"  想定元本   中央値 {n_ev:>12,.0f} USD/日  (年 {n_ev*365/1e9:.1f}B USD)")
print(f"  スプレッド 中央値 {spr_ev:>6.3f} bp")
print(f"  (参考: 旧版の 99 日混合中央値は 出来高 {float(df['vol'].median()):,.0f} / "
      f"想定元本 {float(df['notional'].median())/1e6:.1f}M : 6.5 倍の過小)")

print("\n=== 手数料の総額(実測レート × 評価窓の想定元本)===")
for nm, bp in (("取引所", 0.543), ("deployer", 0.227),
               ("builder", 0.076), ("メイカー支払", 0.088)):
    d = n_ev * bp / 1e4
    print(f"  {nm:<12} {bp:>6.3f} bp → {d:>8,.0f} USD/日 = {d*365:>11,.0f} USD/年")

c = pl.read_csv(D / "capacity_v3.csv")
mv = float(ev["vol"].median())
print("\n=== マーケットメイクの取り分と市場に占める割合(分母 = 評価窓)===")
print(f"{'数量':>5}{'約定量/日':>11}{'市場比':>9}{'損益USD/日':>12}{'年USD':>10}{'Sharpe':>9}")
for q in sorted(c["qty"].unique().to_list()):
    s = c.filter(pl.col("qty") == q); u = s["pnl_usd"].to_numpy()
    sh = u.mean() / u.std(ddof=1) * np.sqrt(365)
    print(f"{q:>5.0f}{float(s['volume'].median()):>11,.0f}"
          f"{float(s['volume'].median())/mv*100:>8.2f}%"
          f"{u.mean():>12.2f}{u.mean()*365:>10,.0f}{sh:>9.1f}")
