"""キュー内位置 QI の検証・分布・約定確率との関係。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
Q = D / "queue_qi"
# 発注時キュー位置(数量)のバケット
EDGES = [0, 1e-9, 10, 50, 100, 250, 500, 1000, 2500, 5000, 1e18]
LBL = ["0(先頭)", "0-10", "10-50", "50-100", "100-250", "250-500",
       "500-1k", "1k-2.5k", "2.5k-5k", "5k+"]

def main() -> None:
    lf = pl.scan_parquet(str(Q / "**" / "*.parquet"))
    tot = lf.select(
        n=pl.len(), filled=pl.col("is_filled").sum(),
        unknown=pl.col("open_unknown").sum(),
        qi_null=pl.col("qi_close").is_null().sum(),
    ).collect().to_dicts()[0]

    # 検証: 約定した注文は前が捌けている
    val = lf.filter(pl.col("is_filled")).select(
        n=pl.len(),
        qi_med=pl.col("qi_close").median(),
        qi_lt_m09=(pl.col("qi_close") < -0.9).mean(),
        q_ahead_zero=(pl.col("q_ahead_close") == 0).mean(),
    ).collect().to_dicts()[0]
    can = lf.filter(~pl.col("is_filled")).select(
        n=pl.len(), qi_med=pl.col("qi_close").median(),
        qi_lt_m09=(pl.col("qi_close") < -0.9).mean(),
        qi_gt_p09=(pl.col("qi_close") > 0.9).mean(),
        q_ahead_zero=(pl.col("q_ahead_close") == 0).mean(),
    ).collect().to_dicts()[0]

    # 発注時キュー位置 → 約定確率
    q = (lf.filter(~pl.col("open_unknown"))
           .with_columns(pl.col("q_ahead_open").cut(EDGES[1:-1], labels=LBL).alias("b"))
           .group_by("b")
           .agg(n=pl.len(), fill_rate=pl.col("is_filled").mean(),
                qi_close_med=pl.col("qi_close").median())
           .collect())
    order = {l: i for i, l in enumerate(LBL)}
    rows = sorted(q.to_dicts(), key=lambda r: order.get(str(r["b"]), 99))

    # qi_close のヒストグラム(0.05 刻み)
    h = (lf.select(pl.col("qi_close").mul(20).round().truediv(20).alias("b"),
                   pl.col("is_filled"))
           .group_by("b").agg(n=pl.len(), filled=pl.col("is_filled").sum())
           .sort("b").collect())

    res = {"total": tot, "filled": val, "canceled": can,
           "by_queue_position": rows,
           "qi_hist": h.drop_nulls("b").to_dicts()}
    (D / "queue_qi_summary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "qi_hist"},
                     indent=1, ensure_ascii=False, default=float)[:2600])

if __name__ == "__main__":
    main()
