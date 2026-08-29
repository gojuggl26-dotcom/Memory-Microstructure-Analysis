"""遅延 300ms 帯での k スイープ — 「遅いことを前提にした方策」に正の期待値があるか。

【問い】
  報告 35 の遅延格子は k=-1(毎回建て直す)だけで測っており、
  300ms が 50/100ms より良いという非単調性が出た(本セッションで
  対応検定 p=0.016 と確認)。しかし k=-1 は「遅いのに追いかける」形であり、
  遅延前提の設計ではない。**遅いことを前提に置く方策**(k>0)を
  300ms のもとで測らないと「遅延前提の戦略」の可否は判定できない。

【時間契約】
  rest_policy.simulate_rest と同一。建て直しの判定はその時刻の板のみ、
  注文は t+lat_open に有効化、取消は判定時刻 +lat_cancel に効く。

【出力】
  data/rest_policy_lat300.csv(rest_policy.csv と同一スキーマ)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rest_policy import D, MS, load, load_model, simulate_rest  # noqa: E402

LAT_MS = 300
KS = [-1, 0, 2, 5, 10, 20, 50, 100]
OUT = D / f"rest_policy_lat{LAT_MS}.csv"


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    q = float(pl.read_csv(D / "capacity_100usd.csv")
              .filter(pl.col("lev") == 1)["qty"][0])
    rows = []
    if OUT.exists():                       # 途中で止まっても続きから走る
        old = pl.read_csv(OUT)
        done = {dt for dt, c in old.group_by("dt").len().iter_rows()
                if c == len(KS)}
        rows = old.filter(pl.col("dt").is_in(list(done))).to_dicts()
        days = [d_ for d_ in days if d_ not in done]
        print(f"再開: 完了 {len(done)} 日 / 残り {len(days)} 日", flush=True)
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        for k in KS:
            r = simulate_rest(d, q, 5.0 * q, k, LAT_MS * MS, LAT_MS * MS)
            if r:
                rows.append({"dt": dt, **r})
        print(f"{i + 1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(OUT)
    pl.DataFrame(rows).write_csv(OUT)
    print("完了")


if __name__ == "__main__":
    main()
