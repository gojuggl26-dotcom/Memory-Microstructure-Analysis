"""原資産(米国上場株)の時間足・日足を Yahoo の公開 chart API から落とす。

    uv run python scripts/fetch_cash.py --symbols MU NQ=F SMH

出力: data/cash_<SYM>_1h.csv(プレ/アフター込み)と data/cash_<SYM>_1d.csv

鍵も課金も要らない。`includePrePost=true` を付けると

    08:00〜13:00 UTC  プレマーケット(1 時間ごと)
    13:30 UTC         レギュラーの寄り付きから 1 時間 ★
    14:30〜19:30 UTC  レギュラー(1 時間ごと)
    20:00〜23:00 UTC  アフターマーケット

という並びになる(標本期間 2026-05〜08 は全て米国夏時間なので
寄り付き 13:30 UTC / 引け 20:00 UTC で固定)。

★注意: 時間外の足の high / low には明らかな誤気配が混ざる
(例 2026-06-01 の 20:00 UTC 足の low = 928.41、前後は 1,035 前後)。
**時間外は open と close しか使わない**こと。volume は時間外では 0 が返る。

1 分足・5 分足は 30〜60 日より前を要求すると 422 が返る(本標本では使えない)。

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
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
       "?period1={p1}&period2={p2}&interval={iv}&includePrePost={pp}")


def fetch(sym: str, iv: str, p1: int, p2: int, pp: bool) -> pl.DataFrame:
    u = URL.format(sym=urllib.parse.quote(sym), p1=p1, p2=p2, iv=iv,
                   pp="true" if pp else "false")
    req = urllib.request.Request(u, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as f:
        d = json.load(f)
    r = d["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    t = pl.DataFrame({
        "ts": r["timestamp"], "open": q["open"], "high": q["high"],
        "low": q["low"], "close": q["close"], "volume": q["volume"],
    }).with_columns(
        dtm=pl.from_epoch("ts", time_unit="s"),
        symbol=pl.lit(r["meta"]["symbol"]),
    )
    return t.filter(pl.col("close").is_not_null()).sort("ts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["MU", "NQ=F", "SMH"])
    ap.add_argument("--start", default="2026-04-25")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    p1 = int(dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc).timestamp()
             ) if a.end else int(time.time())

    for sym in a.symbols:
        tag = sym.replace("=", "").replace("^", "")
        for iv, pp in (("1h", True), ("1d", False)):
            t = fetch(sym, iv, p1, p2, pp)
            out = ROOT / "data" / f"cash_{tag}_{iv}.csv"
            t.write_csv(out)
            print(f"[cash] {sym:<6} {iv}: {t.height:>5} 本 "
                  f"{t['dtm'].min()} 〜 {t['dtm'].max()} -> {out.name}", file=sys.stderr)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
