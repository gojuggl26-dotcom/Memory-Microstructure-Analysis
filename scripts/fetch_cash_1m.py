"""原資産の 1 分足を yfinance から落とす。

    uv run python scripts/fetch_cash_1m.py --symbols MU

出力: data/cash1m_<TAG>.csv(ts は UTC の epoch 秒。列は cash_*_1h.csv と同じ)

★取れる期間は 30 日しかない
----------------------------
Yahoo は 1 分足を**直近 30 日**しか保持しない。実測(2026-09-12 時点)で

    2026-08-13 より前を要求すると YFPricesMissingError
    1 回のリクエストで要求できるのは 7 日まで

したがってこのスクリプトは 7 日窓を順に叩いて繋ぐ。**本リポジトリの
perp の標本(2026-05-04 〜 08-10)とは重ならない**ので、過去の分析に
そのまま足すことはできない。今日から先へ向けて貯めていく用途になる。

`prepost=True` を付けると 04:00〜20:00 ET(= 08:00〜24:00 UTC、米国夏時間)の
時間外も含めて 1 日 960 本が返る。レギュラーは 09:30〜16:00 ET の 390 本。

x が確定する時刻 / y の期間: 該当なし(データ取得)。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 7          # 1 リクエストの上限(日)
KEEP = 30          # Yahoo が 1 分足を保持する日数


def one(sym: str, s: dt.date, e: dt.date):
    d = yf.Ticker(sym).history(start=s.isoformat(), end=e.isoformat(), interval="1m",
                               prepost=True, auto_adjust=False, raise_errors=True)
    if not len(d):
        return None
    return pl.DataFrame({
        "ts": [int(x.timestamp()) for x in d.index],
        "open": d["Open"].to_numpy(), "high": d["High"].to_numpy(),
        "low": d["Low"].to_numpy(), "close": d["Close"].to_numpy(),
        "volume": d["Volume"].to_numpy().astype("int64"),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["MU"])
    ap.add_argument("--days", type=int, default=KEEP)
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    start = end - dt.timedelta(days=a.days)

    for sym in a.symbols:
        parts, miss = [], []
        cur = start
        while cur <= end:
            hi = min(cur + dt.timedelta(days=CHUNK), end + dt.timedelta(days=1))
            try:
                t = one(sym, cur, hi)
                if t is not None:
                    parts.append(t)
                    print(f"  {cur} 〜 {hi}: {t.height:>5} 本", file=sys.stderr)
                else:
                    miss.append((cur, hi))
            except Exception as ex:
                miss.append((cur, hi))
                print(f"  {cur} 〜 {hi}: {type(ex).__name__}", file=sys.stderr)
            cur = hi
            time.sleep(1.0)

        if not parts:
            print(f"[1m] {sym}: 1 本も取れなかった", file=sys.stderr)
            continue
        t = (pl.concat(parts).unique(subset="ts", keep="last").sort("ts")
             .with_columns(dtm=pl.from_epoch("ts", time_unit="s"),
                           symbol=pl.lit(sym)))
        tag = sym.replace("=", "").replace("^", "").replace(".", "_")
        out = ROOT / "data" / f"cash1m_{tag}.csv"
        t.write_csv(out)

        # ---- 品質の確認 -------------------------------------------------
        u = (t.with_columns(day=(pl.col("ts") // 86400) * 86400,
                            sec=pl.col("ts") % 86400)
             .with_columns(seg=pl.when(pl.col("sec") < 13 * 3600 + 1800)
                           .then(pl.lit("プレ(08:00-13:30 UTC)"))
                           .when(pl.col("sec") < 20 * 3600)
                           .then(pl.lit("レギュラー(13:30-20:00)"))
                           .otherwise(pl.lit("アフター(20:00-24:00)"))))
        d = (u.group_by("day").agg(pl.len().alias("n"),
                                   (pl.col("seg") == "レギュラー(13:30-20:00)")
                                   .sum().alias("n_reg"),
                                   pl.col("volume").sum()).sort("day"))
        print(f"[1m] {sym}: {t.height:,} 本 / {d.height} 日 "
              f"{t['dtm'].min()} 〜 {t['dtm'].max()} -> {out.name}", file=sys.stderr)
        print(f"     1 日の本数: 中央 {int(d['n'].median())} "
              f"(時間外込みで満杯なら 960)/ 最小 {int(d['n'].min())} "
              f"/ 最大 {int(d['n'].max())}", file=sys.stderr)
        print(f"     レギュラーの本数: 中央 {int(d['n_reg'].median())}(満杯なら 390)"
              f"/ 最小 {int(d['n_reg'].min())}", file=sys.stderr)
        print(u.group_by("seg").agg(pl.len().alias("本数"),
                                    (pl.col("volume") == 0).mean().alias("出来高 0 の割合"),
                                    pl.col("volume").median().alias("出来高の中央値")
                                    ).sort("seg"))
        if miss:
            print(f"     取れなかった窓 {len(miss)} 個(週末だけの窓と保持期間の外)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
