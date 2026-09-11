"""Hyperliquid の公開 info API から perp のローソク足を落とす。

    uv run python scripts/fetch_hlcandle.py --coin xyz:MU --interval 1h

出力: data/hlcandle_<coin>_<interval>.csv

なぜ要るか
----------
手元の 1 秒 mid(`_liqmid_*.npz`、`bbo_*.parquet` 由来)は S3 から落とした
2026-05-04 〜 08-09 の 98 日しか無い。公開 API の 1 時間足なら**上場から現在まで**
取れるので、日次の標本を 66 営業日から 90 営業日へ伸ばせる。1 分足・5 分足は
直近数日しか残っていない(実測)。

★足の値は**最終約定値**であって mid ではない。手元の mid と突き合わせて
どれだけずれるかを `build_pricedisc.py` で必ず検算する。

x が確定する時刻 / y の期間: 該当なし(データ取得)。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.hyperliquid.xyz/info"
HDR = {"User-Agent": "python-urllib", "Content-Type": "application/json"}
STEP = {"1m": 3, "5m": 14, "15m": 40, "1h": 100, "4h": 400, "1d": 2000}


def post(body: dict):
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=40) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    s = int(dt.datetime.fromisoformat(a.start).replace(
        tzinfo=dt.timezone.utc).timestamp() * 1000)
    e = int(dt.datetime.fromisoformat(a.end).replace(
        tzinfo=dt.timezone.utc).timestamp() * 1000) if a.end else int(time.time() * 1000)
    step = STEP[a.interval] * 86_400_000

    rows, cur = [], s
    while cur < e:
        d = post({"type": "candleSnapshot",
                  "req": {"coin": a.coin, "interval": a.interval,
                          "startTime": cur, "endTime": min(cur + step, e)}})
        rows.extend(d or [])
        print(f"  {dt.datetime.fromtimestamp(cur/1000, dt.timezone.utc):%Y-%m-%d}: "
              f"{len(d or [])} 本 (累計 {len(rows)})", file=sys.stderr)
        cur += step
        time.sleep(0.35)

    t = (pl.DataFrame(rows)
         .select(ts=(pl.col("t") // 1000).cast(pl.Int64),
                 open=pl.col("o").cast(pl.Float64), high=pl.col("h").cast(pl.Float64),
                 low=pl.col("l").cast(pl.Float64), close=pl.col("c").cast(pl.Float64),
                 volume=pl.col("v").cast(pl.Float64), n_trades=pl.col("n"))
         .unique(subset="ts", keep="last").sort("ts")
         .with_columns(dtm=pl.from_epoch("ts", time_unit="s")))
    tag = a.coin.replace(":", "_")
    out = ROOT / "data" / f"hlcandle_{tag}_{a.interval}.csv"
    t.write_csv(out)
    print(f"[hl] {t.height:,} 本 {t['dtm'].min()} 〜 {t['dtm'].max()} -> {out.name}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
