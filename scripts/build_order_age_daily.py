"""時間帯どうしの差が、日ごとに見ても同じ向きに出るかを確かめる。

`build_order_age.py` が出すのは 67〜68 日をひとまとめにした分布である。
まとめた中央値の差は、**少数の日が全体を引っ張っているだけ**でも出てしまう。
そこで立会日ごとに窓ごとの中央値を出し、符号検定にかける。

対照は「開場後 1 時間」。他の 4 窓それぞれについて

    その日の中央値 > 開場後の中央値

となった日を数え、両側の二項検定(p = 1/2)で p 値を出す。日をまたいだ独立性は
厳密には成り立たない(地合いは連日相関する)ので、p 値は**目安**として読む。
向きが 60 日以上そろっているかどうか、を主に見る。

対象と除外は `build_order_age.py` と同じ(Alo / Gtc、拒否・孤児・右打ち切りを除く)。

x が確定する時刻 / y の期間: 該当なし(記述統計)。

    uv run python scripts/build_order_age_daily.py --all
出力: data/order_age_daily_<coin>.csv
"""

from __future__ import annotations

import argparse
import datetime as dtm
import time
from math import comb

import polars as pl

from build_order_age import (COINS, RESTING_TIF, ROOT, WINDOWS, lifecycle_dir,
                             sessions, window_expr)

WNAMES = [w[0] for w in WINDOWS]          # 5 つの主たる窓の名前だけ
BASE = "開場後 1 時間"


def sign_p(k: int, n: int) -> float:
    """両側二項検定(p=1/2)。n が小さいので厳密に足す。"""
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    tail = sum(comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run(coin: str) -> None:
    tag = coin.replace(":", "_")
    parts = sorted(p for p in lifecycle_dir(coin).glob("dt=*")
                   if (p / "_SUCCESS").exists())
    keep, _ = sessions([p.name[3:] for p in parts])
    rows, t0 = [], time.time()
    for i, p in enumerate(parts):
        dtd = dtm.date.fromisoformat(p.name[3:])
        g = (
            pl.scan_parquet(sorted(p.glob("part-*.parquet")))
            .select("ts_open", "lifetime_ns", "tif",
                    "is_rejected", "is_orphan", "is_censored")
            .filter(pl.col("ts_open").is_not_null())
            .with_columns(w=window_expr())
            .filter(pl.col("w").is_not_null())
            .filter(pl.col("w").is_in(WNAMES))
            .with_columns(d=pl.col("ts_open").cast(pl.Datetime("ns")).dt.date())
            .filter(pl.col("d").is_in(list(keep)))
            # 注文行は「閉じた日」の区画に載るので、開いた日が区画日と同じものだけ残す。
            # 5 つの窓はすべて 04:00 UTC 以降なので UTC 日 = 東部日であり、
            # ここで落ちるのは「日をまたいで生きた注文」= 全体の 0.05% 未満
            .filter(pl.col("d") == pl.lit(dtd))
            .filter(~pl.col("is_rejected") & ~pl.col("is_orphan")
                    & ~pl.col("is_censored")
                    & pl.col("tif").is_in(RESTING_TIF)
                    & pl.col("lifetime_ns").is_not_null())
            .group_by("d", "w")
            .agg(n=pl.len(), p50=(pl.col("lifetime_ns") / 1e9).median())
            .collect()
        )
        rows.extend(g.iter_rows(named=True))
        if (i + 1) % 25 == 0 or i + 1 == len(parts):
            print(f"  {i + 1:>3}/{len(parts)} 日  {time.time() - t0:6.1f}s")

    D = pl.DataFrame(rows).sort("d", "w")
    D.write_csv(ROOT / "data" / f"order_age_daily_{tag}.csv")

    wide = D.pivot(on="w", index="d", values="p50").drop_nulls()
    print(f"\n=== {coin} ===  立会日 {wide.height} 日  対照 = {BASE}")
    print(f"{'時間帯':<24}{'対照より長い日':>14}{'割合':>9}{'p 値':>11}"
          f"{'中央値の比の中央値':>20}")
    for w in WNAMES:
        if w == BASE or w not in wide.columns:
            continue
        r = (wide[w] / wide[BASE]).to_numpy()
        k = int((r > 1).sum()); n = len(r)
        print(f"{w:<24}{k:>10}/{n:<3}{k / n * 100:>8.1f}%{sign_p(k, n):>11.2e}"
              f"{float(pl.Series(r).median()):>19.2f} 倍")
    print(f"[out] data/order_age_daily_{tag}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for c in (COINS if a.all else [a.coin]):
        if c is None:
            raise SystemExit("--coin か --all を指定すること")
        run(c)


if __name__ == "__main__":
    main()
