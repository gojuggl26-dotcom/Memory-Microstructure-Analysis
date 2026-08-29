"""注文ごとのキュー内位置と QI を算出する。

    QI(o, t) = (Q_ahead − Q_behind) / (Q_ahead + Q_behind)

  Q_ahead  : 同じ (side, px) の板で、自分より**先に**並んでいる注文の残量合計
  Q_behind : 同じ板で、自分より**後に**並んだ注文の残量合計

L4(注文単位)でしか作れない指標。各注文について 2 時点で出す。

  発注時   Q_behind = 0 なので QI = +1 で自明。意味があるのは
           **q_ahead_open = 発注時に自分の前にあった数量(= キュー位置)**
  消滅時   qi_close。約定で消えた注文は前が捌けているので −1 付近、
           取消で消えた注文は前が残っていれば +1 付近になるはず(§検証)

方式: (side, px) ごとに「生存中の注文を到着順に並べたリスト」を持ち、
イベント(到着・消滅)を時刻順に流す。到着は必ず末尾に付くので追加は O(1)、
消滅は bisect で位置を出して前後の数量を合計する。生存数は板 1 段あたり数十なので軽い。

近似: 残量は orig_sz 一定として扱う(取消で終わる注文が 99.0% を占め、
その残量は発注数量そのもの)。部分約定してから取り消された注文だけ過大評価になる。

日跨ぎ: lifecycle は ts_close 基準の日次パーティション。前日に発注された注文も
当日ファイルに入るので到着イベントとして正しく流れる。逆に「当日発注・翌日消滅」の
注文はその日のファイルに無いため板の状態から漏れる(2026-07-29 で 0.007%)。
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("data/l2_v99/lifecycle")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
DAYS_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NON_RESTING_TIF = {"Ioc", "FrontendMarket", "LiquidationMarket"}


def run_day(dt: str) -> pl.DataFrame:
    df = (pl.read_parquet(
            SRC / f"dt={dt}" / "part-000.parquet",
            columns=["oid", "side", "px", "orig_sz", "filled_sz", "ts_open", "ts_close",
                     "terminal_status", "tif", "is_rejected", "is_trigger"])
          .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                  & (~pl.col("tif").is_in(list(NON_RESTING_TIF)).fill_null(False)))
          .sort("ts_open", nulls_last=False))
    n = df.height
    if n < 10:
        return pl.DataFrame()
    # ts_open が null の注文(前日以前に発注され、当日の状態復元では開始時刻が不明)は
    # 「当日の全注文より先に並んでいた」として日の先頭に置く。自身の発注時キュー位置は
    # 測れないので open_unknown で除外できるようにする
    unknown_open = df["ts_open"].is_null().to_numpy()
    if unknown_open.any():
        df = df.with_columns(pl.col("ts_open").fill_null(pl.col("ts_close").min() - 1))
    # (side, px) を整数キーに畳む。px は 1e-6 まで一意に表せる
    lvl = (df.select(
        (pl.col("px").mul(1_000_000).round().cast(pl.Int64) * 2
         + (pl.col("side") == "B").cast(pl.Int64)).alias("k"))["k"].to_list())
    o_ts = df["ts_open"].to_list()
    c_ts_np = df["ts_close"].to_numpy()
    sz = df["orig_sz"].to_list()
    dep_order = np.argsort(c_ts_np, kind="stable").tolist()
    c_ts = c_ts_np[dep_order].tolist()

    seqs: dict[int, list[int]] = {}
    szs: dict[int, list[float]] = {}
    q_ahead_open = [0.0] * n
    n_ahead_open = [0] * n
    q_ahead_close = [0.0] * n
    q_behind_close = [0.0] * n

    ai = 0          # 到着側の走査位置
    di = 0          # 消滅側の走査位置
    while ai < n or di < n:
        # 同時刻なら到着を先に処理する(発注と同 ns の取消でも自分は板に載る)
        if di >= n or (ai < n and o_ts[ai] <= c_ts[di]):
            k = lvl[ai]
            s = seqs.get(k)
            if s is None:
                seqs[k] = [ai]
                szs[k] = [sz[ai]]
            else:
                q_ahead_open[ai] = sum(szs[k])
                n_ahead_open[ai] = len(s)
                s.append(ai)
                szs[k].append(sz[ai])
            ai += 1
        else:
            j = dep_order[di]
            k = lvl[j]
            s = seqs.get(k)
            if s:
                i = bisect_left(s, j)
                if i < len(s) and s[i] == j:
                    z = szs[k]
                    q_ahead_close[j] = sum(z[:i])
                    q_behind_close[j] = sum(z[i + 1:])
                    del s[i]
                    del z[i]
            di += 1

    a = np.array(q_ahead_close)
    b = np.array(q_behind_close)
    tot = a + b
    qi = np.where(tot > 0, (a - b) / tot, np.nan)
    return pl.DataFrame({
        "oid": df["oid"],
        "is_filled": df["terminal_status"] == "filled",
        "open_unknown": unknown_open,
        "q_ahead_open": np.where(unknown_open, np.nan,
                                 np.array(q_ahead_open)).astype(np.float32),
        "n_ahead_open": np.where(unknown_open, -1,
                                 np.array(n_ahead_open)).astype(np.int32),
        "q_ahead_close": a.astype(np.float32),
        "q_behind_close": b.astype(np.float32),
        "qi_close": qi.astype(np.float32),
    }).with_columns(pl.col("qi_close").fill_nan(None),
                    pl.col("q_ahead_open").fill_nan(None))


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    if DAYS_LIMIT:
        days = days[:DAYS_LIMIT]
    base = OUT_DIR / "data" / "queue_qi"
    base.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, dt in enumerate(days, 1):
        out = run_day(dt)
        if out.is_empty():
            continue
        d = base / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        out.write_parquet(d / "part-000.parquet", compression="zstd", statistics=True)
        total += out.height
        print(f"[{i:3d}/{len(days)}] {dt} orders={out.height:>10,} total={total:>12,}", flush=True)
    print("done", total)


if __name__ == "__main__":
    main()
