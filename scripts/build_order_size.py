"""4 つの時間帯ごとに、成行注文 1 本あたりのサイズの分布を出す。

## 「オーダーサイズ」に何を使ったか

板に置かれた指値注文の当初数量(`orig_sz`)は L2 の `lifecycle` 表にあるが、
この表は WORK_BUCKET 上で **DEEP_ARCHIVE** に移されており、復元しないと読めない
(復元は数時間かかり費用も発生する)。そこで手元にある約定記録から作れる

    **成行注文 1 本のサイズ** = 同一 ns・同一ユーザーのテイカー約定の合計数量

を「オーダーサイズ」として使う。板を何段も食った注文は 1 本として合算される。
これは **執行された攻撃的注文のサイズ** であって、板に置かれて待っている
指値注文のサイズではない。両者は別物なので、結論をそちらへ広げないこと。

比較のため、約定 1 件ごとのサイズ(板の 1 段ぶん)も併せて出す。

## 時間帯(すべて UTC)

標本期間は全日が米国東部夏時間なので、ニューヨークの寄り付きは 13:30 UTC、
引けは 20:00 UTC にあたる。

| 呼び名 | UTC | 米国東部時間 |
|---|---|---|
| 開場後 2 時間 | 13:30–15:30 | 9:30–11:30 |
| 昼 11 時から 2 時間 | 11:00–13:00 | 7:00–9:00(寄り付き前) |
| 閉場前 2 時間 | 18:00–20:00 | 14:00–16:00 |
| 深夜 11 時から 2 時間 | 23:00–翌 01:00 | 19:00–21:00(引け後) |

深夜の窓だけ日付をまたぐので、分単位の判定で「23 時台以降 または 1 時前」とする。

## 全曜日

曜日で絞らず 99 日すべてを込みにする。ただし「開場後」「閉場前」は原市場が
開いている日にしか意味がないので、立会日と休場日に分けた値も併せて出す。

x が確定する時刻 / y の期間: 該当なし(分布の記述統計)。

    uv run python scripts/build_order_size.py --coin xyz:MU
出力: data/order_size_stats_<coin>.csv, data/order_size_hist_<coin>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

# (名前, 開始分, 終了分)。終了は含まない。深夜は日付をまたぐ
WINDOWS = [
    ("開場後 2 時間", 13 * 60 + 30, 15 * 60 + 30),
    ("昼 11 時から 2 時間", 11 * 60, 13 * 60),
    ("閉場前 2 時間", 18 * 60, 20 * 60),
    ("深夜 11 時から 2 時間", 23 * 60, 25 * 60),      # 23:00–翌 01:00
]
# 対数ヒストグラムの階級(枚)。分布を見る前に機械的に決める
# 1 桁あたり 5 階級。これより細かくすると丸い数量(1, 5, 10 枚など)への集中で
# 棘だらけになり、4 本の比較が読めなくなる
EDGES = np.logspace(-3, 4, 36)


def in_window(mo: np.ndarray, lo: int, hi: int) -> np.ndarray:
    if hi <= 24 * 60:
        return (mo >= lo) & (mo < hi)
    return (mo >= lo) | (mo < hi - 24 * 60)          # 日付をまたぐ窓


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                        columns=["ts", "user", "sz", "crossed"])
        .filter(pl.col("crossed"))
        .with_columns(
            d=pl.col("ts").dt.date(),
            mo=(pl.col("ts").dt.hour().cast(pl.Int32) * 60
                + pl.col("ts").dt.minute().cast(pl.Int32)),
        )
    )
    # 成行注文 1 本 = 同一 ns・同一ユーザー。順序は結果に効かないが決定的にしておく
    o = (
        f.group_by("ts", "user")
        .agg(sz=pl.col("sz").sum(), d=pl.col("d").first(), mo=pl.col("mo").first(),
             n_fill=pl.len())
        .sort("ts", "user")
    )
    print(f"[件数] 約定 {f.height:,} 件 → 成行注文 {o.height:,} 本"
          f"(1 本あたり {f.height / o.height:.2f} 件)")

    cal = xc.get_calendar("XNYS")
    days = o["d"].unique().sort().to_list()
    sess = {x.date() for x in cal.sessions_in_range(str(days[0]), str(days[-1]))}

    o = o.with_columns(open_day=pl.col("d").is_in(list(sess)))
    mo = o["mo"].to_numpy()
    sz = o["sz"].to_numpy()
    fsz = f["sz"].to_numpy()
    fmo = f["mo"].to_numpy()
    isopen = o["open_day"].to_numpy()

    rows, hrows = [], []
    print(f"\n{'時間帯':<22}{'区分':<8}{'本数':>12}{'中央値':>10}{'平均':>10}"
          f"{'第1四分位':>10}{'第3四分位':>10}{'99%点':>10}{'最大':>11}")
    for name, lo, hi in WINDOWS:
        m_all = in_window(mo, lo, hi)
        for lab, m in (("全曜日", m_all), ("立会日", m_all & isopen), ("休場日", m_all & ~isopen)):
            x = sz[m]
            if len(x) == 0:
                continue
            rows.append({
                "window": name, "day_type": lab, "n": int(len(x)),
                "median": float(np.median(x)), "mean": float(x.mean()),
                "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
                "q99": float(np.quantile(x, 0.99)), "max": float(x.max()),
                "share_ge_10": float((x >= 10).mean()), "total_sz": float(x.sum()),
            })
            print(f"{name:<22}{lab:<8}{len(x):>12,}{np.median(x):>10.3f}{x.mean():>10.3f}"
                  f"{np.quantile(x, 0.25):>10.3f}{np.quantile(x, 0.75):>10.3f}"
                  f"{np.quantile(x, 0.99):>10.2f}{x.max():>11.1f}")
        # 約定 1 件ごと(板の 1 段ぶん)も出す
        mf = in_window(fmo, lo, hi)
        xf = fsz[mf]
        rows.append({
            "window": name, "day_type": "全曜日(約定 1 件)", "n": int(len(xf)),
            "median": float(np.median(xf)), "mean": float(xf.mean()),
            "q25": float(np.quantile(xf, 0.25)), "q75": float(np.quantile(xf, 0.75)),
            "q99": float(np.quantile(xf, 0.99)), "max": float(xf.max()),
            "share_ge_10": float((xf >= 10).mean()), "total_sz": float(xf.sum()),
        })

        # ヒストグラムと補相補累積分布(全曜日)
        x = sz[m_all]
        cnt, _ = np.histogram(x, bins=EDGES)
        for i in range(len(cnt)):
            hrows.append({"window": name, "lo": float(EDGES[i]), "hi": float(EDGES[i + 1]),
                          "count": int(cnt[i])})

    S = pl.DataFrame(rows)
    S.write_csv(ROOT / "data" / f"order_size_stats_{tag}.csv")
    pl.DataFrame(hrows).write_csv(ROOT / "data" / f"order_size_hist_{tag}.csv")

    print("\n=== 全曜日での窓どうしの比較(成行注文 1 本)===")
    g = S.filter(pl.col("day_type") == "全曜日")
    base = g.filter(pl.col("window") == "開場後 2 時間").to_dicts()[0]
    for r in g.iter_rows(named=True):
        print(f"  {r['window']:<22} 中央値 {r['median']:6.3f} 枚"
              f"(開場後比 {r['median'] / base['median']:.2f} 倍) / "
              f"平均 {r['mean']:6.3f}(同 {r['mean'] / base['mean']:.2f} 倍) / "
              f"10 枚以上の割合 {r['share_ge_10'] * 100:5.2f}%")

    print("\n=== 分布の裾: 上位 1% が担う数量の割合(全曜日)===")
    for name, lo, hi in WINDOWS:
        x = np.sort(sz[in_window(mo, lo, hi)])[::-1]
        k = max(1, int(len(x) * 0.01))
        print(f"  {name:<22} 上位 1%({k:,} 本)が全数量の {x[:k].sum() / x.sum() * 100:5.2f}%")

    print(f"\n[out] data/order_size_stats_{tag}.csv, data/order_size_hist_{tag}.csv")


if __name__ == "__main__":
    main()
