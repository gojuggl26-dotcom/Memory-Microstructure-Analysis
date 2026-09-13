"""特徴量ライブラリの格子点に、9 つのホライズンの前向きリターンを足す。

    uv run python scripts/build_fwd.py --coin xyz:MU

出力: data/fwd_<coin>.parquet(dt, sec, fwd_mid_*, fwd_micro_* の 18 列)

なぜ作り直すか
--------------
`featlib_<coin>` は 1 秒格子で作って 5 秒おきに間引いてあり、前向きリターンは
**1s / 10s / 60s の 3 本しか無い**。100ms・300ms・500ms は 1 秒格子では作れない。
そこで**同じ板の生データ**(`bbo_<coin>.parquet` に `clean_bbo` をかけたもの。
featlib が読んでいるものと同一)から、格子点ごとに 9 ホライズンぶん作り直す。

    mid   = (pb + pa) / 2
    micro = mid + (pa − pb)/2 × (qb − qa)/(qb + qa)      ← featlib の micro1 と同じ式
    fwd_x_h = log( x(T + h) / x(T) ) × 1e4               単位 bp

`x(T)` は **T 以前の最後の気配**(backward)。featlib の格子点の値と同じ規約。
既存の `fwd_mid_1s / 10s / 60s` と突き合わせて一致を確かめる(`--check`)。

時間契約: 説明変数は T までで確定し、目的変数は (T, T+h]。T + h がその日を
はみ出す格子点は欠測にする(featlib の既定と同じ)。

x が確定する時刻 / y の期間: 本スクリプトは y だけを作る。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
DAY_NS = 86_400_000_000_000
KEEP_EVERY = 5
HZ = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]


def hname(h: float) -> str:
    return f"{int(h*1000)}ms" if h < 1 else f"{int(h)}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--bbo-suffix", default="")
    ap.add_argument("--check", action="store_true", default=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb, ndrop = clean_bbo(pl.read_parquet(D / f"bbo_{tag}{a.bbo_suffix}.parquet"))
    print(f"[fwd] bbo {bb.height:,} 行(clean_bbo で {ndrop} 行を除外)", file=sys.stderr)
    days = sorted(p.stem.split("=")[1] for p in Path(D / f"featlib_{tag}").glob("dt=*.parquet"))
    print(f"[fwd] featlib の {len(days)} 日に合わせて作る", file=sys.stderr)

    sec = np.arange(0, 86400, KEEP_EVERY, dtype=np.int64)
    out = []
    for k, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        if d.height < 100:
            continue
        ts = d["ts"].to_numpy()
        pb = d["best_bid"].to_numpy()
        pa = d["best_ask"].to_numpy()
        qb = d["bid_sz"].to_numpy()
        qa = d["ask_sz"].to_numpy()
        mid = (pb + pa) / 2.0
        mic = mid + (pa - pb) / 2.0 * (qb - qa) / (qb + qa)
        day0 = int(np.datetime64(dt + "T00:00:00", "ns").astype("int64"))
        grid = day0 + sec * 1_000_000_000

        def at(tt):
            j = np.searchsorted(ts, tt, side="right") - 1
            ok = j >= 0
            j = np.maximum(j, 0)
            return np.where(ok, mid[j], np.nan), np.where(ok, mic[j], np.nan), ok

        m0, c0, ok0 = at(grid)
        row = {"dt": [dt] * sec.size, "sec": sec}
        for h in HZ:
            mh, ch, okh = at(grid + int(h * 1_000_000_000))
            inday = (sec + h) <= 86399
            m = ok0 & okh & inday
            row[f"fwd_mid_{hname(h)}"] = np.where(m, np.log(mh / m0) * 1e4, np.nan)
            row[f"fwd_micro_{hname(h)}"] = np.where(m, np.log(ch / c0) * 1e4, np.nan)
        out.append(pl.DataFrame(row))
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True, file=sys.stderr)

    t = pl.concat(out).sort("dt", "sec")
    t.write_parquet(D / f"fwd_{tag}.parquet", compression="zstd")
    print(f"[fwd] {t.height:,} 行 × {len(t.columns)} 列 -> fwd_{tag}.parquet", file=sys.stderr)

    if a.check:
        fl = (pl.scan_parquet(str(D / f"featlib_{tag}" / "dt=*.parquet"))
              .select("dt", "sec", "fwd_mid_1s", "fwd_mid_10s", "fwd_mid_60s",
                      "fwd_micro_1s", "fwd_micro_10s", "fwd_micro_60s")
              .collect().sort("dt", "sec"))
        j = t.join(fl, on=["dt", "sec"], how="inner", suffix="_old")
        print("\n[検算] 既存の featlib の前向きリターンと突き合わせる")
        for c in ("fwd_mid_1s", "fwd_mid_10s", "fwd_mid_60s",
                  "fwd_micro_1s", "fwd_micro_10s", "fwd_micro_60s"):
            x = j[c].to_numpy()
            y = j[c + "_old"].to_numpy().astype(np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            d_ = np.abs(x[m] - y[m])
            print(f"   {c:<16} n={m.sum():>9,}  相関 {np.corrcoef(x[m], y[m])[0,1]:.6f}  "
                  f"|差| 中央 {np.median(d_):.4f}bp / 99% 点 {np.percentile(d_, 99):.3f}bp "
                  f"/ 最大 {d_.max():.2f}bp")


if __name__ == "__main__":
    main()
