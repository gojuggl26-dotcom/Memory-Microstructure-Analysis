"""xyz:DRAM の板を価格レベル差分から再生し、毎ティックの OBI(L=1..10)を算出する。

    OBI(L) = (Σ_{i=1..L} q_i^b − Σ_{i=1..L} q_i^a) / (Σ_{i=1..L} q_i^b + Σ_{i=1..L} q_i^a)

入力: L2 book_px(ts, side, px, total_sz, is_snapshot)
      = 変化した (side, px) レベルだけを ns 精度で出した絶対値表現の差分。
      各日の先頭に is_snapshot=True の全レベル行が入るので日単位で独立に再生できる。

出力: OUT_DIR/data/obi/dt=YYYY-MM-DD/part-000.parquet
      ts, best_bid, best_ask, mid, obi_1..obi_10, cum_bid_10, cum_ask_10,
      n_lv_bid, n_lv_ask, dwell_ns

行の出し方: 上位 10 レベルの状態(気配値と各 OBI)が変化した ts にだけ行を出す。
10 段より深いだけの変化は OBI(1..10) を動かさないので落とす。
dwell_ns(次の行までの時間)が入っているので、任意の時点の値は前方補完で復元できる。

レベル数が L に満たない側は、存在する分だけを合計する(片側 0 レベルなら OBI=±1)。
判別できるよう n_lv_bid / n_lv_ask(10 で打ち切り)を併せて出す。
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path

import polars as pl

SRC = Path("data/l2_v99/book_px")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
DAYS_ARG = sys.argv[2] if len(sys.argv) > 2 else None  # 検証用: "2026-05-17,2026-07-29"
NS_DAY = 86_400_000_000_000
L_MAX = 10

SCHEMA = {
    "ts": pl.Int64, "best_bid": pl.Float64, "best_ask": pl.Float64, "mid": pl.Float64,
    **{f"obi_{i}": pl.Float64 for i in range(1, L_MAX + 1)},
    "cum_bid_10": pl.Float64, "cum_ask_10": pl.Float64,
    "n_lv_bid": pl.UInt8, "n_lv_ask": pl.UInt8, "dwell_ns": pl.Int64,
}


def day_end(dt: str) -> int:
    d = datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp()) * 1_000_000_000 + NS_DAY


def run_day(path: Path, dt: str) -> pl.DataFrame:
    df = (
        pl.read_parquet(path, columns=["ts", "side", "px", "total_sz", "is_snapshot"])
        # 同 ts 内はスナップショット行を先に当てる。異なるレベルの行は絶対値なので順不同で可
        .sort(["ts", "is_snapshot"], descending=[False, True])
    )
    ts_a = df["ts"].to_list()
    side_a = df["side"].to_list()
    px_a = df["px"].to_list()
    sz_a = df["total_sz"].to_list()

    # 買い板は -px を昇順に持つ(添字 0 が最良買気配)。売り板は px 昇順
    b_px: list[float] = []
    b_sz: list[float] = []
    a_px: list[float] = []
    a_sz: list[float] = []

    out_ts: list[int] = []
    rows: list[tuple] = []
    prev_key: tuple | None = None
    n = len(ts_a)
    i = 0
    while i < n:
        ts = ts_a[i]
        # --- 同一 ts のレベル更新をすべて適用 ---
        while i < n and ts_a[i] == ts:
            if side_a[i] == "B":
                pxs, szs, key = b_px, b_sz, -px_a[i]
            else:
                pxs, szs, key = a_px, a_sz, px_a[i]
            sz = sz_a[i]
            j = bisect_left(pxs, key)
            hit = j < len(pxs) and pxs[j] == key
            if sz > 0.0:
                if hit:
                    szs[j] = sz
                else:
                    pxs.insert(j, key)
                    szs.insert(j, sz)
            elif hit:
                del pxs[j]
                del szs[j]
            i += 1

        nb, na = len(b_px), len(a_px)
        if nb == 0 and na == 0:
            continue
        # --- 上位 10 レベルの累積 ---
        cb = list(accumulate(b_sz[:L_MAX]))
        ca = list(accumulate(a_sz[:L_MAX]))
        last_b = cb[-1] if cb else 0.0
        last_a = ca[-1] if ca else 0.0
        obis = []
        for k in range(L_MAX):
            sb = cb[k] if k < len(cb) else last_b
            sa = ca[k] if k < len(ca) else last_a
            tot = sb + sa
            obis.append((sb - sa) / tot if tot > 0.0 else 0.0)
        bb = -b_px[0] if nb else None
        ba = a_px[0] if na else None
        key = (bb, ba, *obis)
        if key == prev_key:      # 上位 10 段に変化なし(深部だけの更新)
            continue
        prev_key = key
        out_ts.append(ts)
        rows.append((
            bb, ba, (bb + ba) / 2 if nb and na else None, *obis,
            last_b, last_a, min(nb, L_MAX), min(na, L_MAX),
        ))

    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    cols = list(zip(*rows))
    names = ["best_bid", "best_ask", "mid"] + [f"obi_{i}" for i in range(1, L_MAX + 1)] + [
        "cum_bid_10", "cum_ask_10", "n_lv_bid", "n_lv_ask"]
    out = pl.DataFrame({"ts": out_ts, **{k: list(v) for k, v in zip(names, cols)}})
    return out.with_columns(
        (pl.col("ts").shift(-1).fill_null(day_end(dt)) - pl.col("ts")).alias("dwell_ns")
    ).cast(SCHEMA)


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    if DAYS_ARG:
        want = set(DAYS_ARG.split(","))
        days = [d for d in days if d in want]
    base = OUT_DIR / "data" / "obi"
    base.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, dt in enumerate(days, 1):
        out = run_day(SRC / f"dt={dt}" / "part-000.parquet", dt)
        d = base / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        out.write_parquet(d / "part-000.parquet", compression="zstd", statistics=True)
        total += out.height
        print(f"[{i:3d}/{len(days)}] {dt} rows={out.height:>9,} total={total:>12,}", flush=True)
    print("done", total)


if __name__ == "__main__":
    main()
