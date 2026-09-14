r"""DefiLlama から ETH ステーキング系プールの利回り履歴を全件取得する。

    uv run python scripts/defillama_yield_history.py

入力: data/staking/yields.json (defillama_staking_fetch.py の出力)
出力: data/staking/yield_history.parquet   … long 形式(1 行 = 1 プール 1 日)
      data/staking/yield_history_meta.json … 取得結果の内訳

=============================================================================
★ 使うのは yields.llama.fi/chart/<pool> であって Pro API ではない
=============================================================================
DefiLlama の TVL 履歴(/protocols 系)は有料だが、**利回り履歴を返す
`yields.llama.fi/chart/<pool>` は無料**である。2026-09-14 に実測して確認した。

★ 429 が高頻度で出る
公開枠なので連続で叩くと HTTP 429 になる。データが無いのではなく叩きすぎで、
間隔を空ければ取れる。指数バックオフで必ず取り切ること。**429 を「データなし」
として黙って落とすと、欠損なのか未収録なのか後から区別できなくなる。**

★ 開始日は「プロトコルの開始日」ではない
返ってくる最古の日付は DefiLlama が収録を始めた日である。Lido は 2020-12 から
稼働しているが履歴は 2022-05-03 から。多くのプールが 2023-01-16 に揃うのは
そこが収録開始日だからで、プロトコルがその日に始まったわけではない。
それ以前が要るならオンチェーンから取る(scripts/onchain_lst_rates.py)。

★ pricePerShare は半分以上 null
Lido 0/1596、Rocket Pool 0/1338。これが無いプールでは DefiLlama の apy 定義を
そのまま使うしかなく、自前で利回りを再計算できない。null 率を meta に残す。
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "staking" / "yields.json"
OUT = ROOT / "data" / "staking" / "yield_history.parquet"
META = ROOT / "data" / "staking" / "yield_history_meta.json"

BASE = "https://yields.llama.fi/chart/"
PAUSE = 0.6          # 通常時の間隔(秒)
MAX_RETRY = 6        # 429 のときの再試行回数
BACKOFF = 3.0        # 初回待機。以後 2 倍ずつ


def fetch(pool: str) -> list[dict]:
    """1 プールの履歴を取る。429 は指数バックオフで粘る。"""
    wait = BACKOFF
    for attempt in range(MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(BASE + pool, timeout=60) as r:
                return json.loads(r.read()).get("data", [])
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == MAX_RETRY:
                raise
            print(f"    429 -> {wait:.0f}s 待機して再試行 ({attempt+1}/{MAX_RETRY})", flush=True)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    pools = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"対象 {len(pools)} プール / {len(set(p['project'] for p in pools))} プロジェクト")

    frames: list[pl.DataFrame] = []
    meta: list[dict] = []
    for i, p in enumerate(pools, 1):
        tag = f"{p['project']}/{p['symbol']}"
        try:
            rows = fetch(p["pool"])
        except Exception as e:                                   # noqa: BLE001
            print(f"[{i:2d}/{len(pools)}] {tag:<40} 失敗: {e}", flush=True)
            meta.append({"pool": p["pool"], "project": p["project"], "symbol": p["symbol"],
                         "ok": False, "error": str(e)[:200]})
            time.sleep(PAUSE)
            continue

        if not rows:
            print(f"[{i:2d}/{len(pools)}] {tag:<40} 空", flush=True)
            meta.append({"pool": p["pool"], "project": p["project"], "symbol": p["symbol"],
                         "ok": True, "n": 0})
            time.sleep(PAUSE)
            continue

        d = pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.lit(p["pool"]).alias("pool"),
            pl.lit(p["project"]).alias("project"),
            pl.lit(p["symbol"]).alias("symbol"),
            pl.lit(p["chain"]).alias("chain"),
        )
        # timestamp は ISO 文字列。日付に落とす(1 日 1 点なので日付が主キー)
        d = d.with_columns(
            pl.col("timestamp").str.slice(0, 10).str.to_date().alias("date")
        )
        for c in ("apy", "apyBase", "apyReward", "tvlUsd", "pricePerShare", "apyBase7d", "il7d"):
            if c not in d.columns:
                d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
            else:
                d = d.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        d = d.select("pool", "project", "symbol", "chain", "date",
                     "apy", "apyBase", "apyReward", "tvlUsd", "pricePerShare")
        frames.append(d)

        pps = int(d["pricePerShare"].is_not_null().sum())
        print(f"[{i:2d}/{len(pools)}] {tag:<40} {d.height:>5}点 "
              f"{d['date'].min()}〜{d['date'].max()} pps={pps}", flush=True)
        meta.append({"pool": p["pool"], "project": p["project"], "symbol": p["symbol"],
                     "chain": p["chain"], "ok": True, "n": d.height,
                     "first": str(d["date"].min()), "last": str(d["date"].max()),
                     "pps_notnull": pps, "tvlUsd_now": p["tvlUsd"]})
        time.sleep(PAUSE)

    if not frames:
        raise SystemExit("1 件も取れなかった")

    all_d = pl.concat(frames, how="vertical_relaxed").sort("project", "symbol", "date")
    all_d.write_parquet(OUT, compression="zstd")
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [m for m in meta if m.get("ok") and m.get("n")]
    ng = [m for m in meta if not m.get("ok")]
    print()
    print(f"[write] {OUT}  {all_d.height:,} 行 / {all_d['pool'].n_unique()} プール")
    print(f"  期間 {all_d['date'].min()} 〜 {all_d['date'].max()}")
    print(f"  成功 {len(ok)} / 失敗 {len(ng)}")
    if ng:
        print("  ★失敗したプール(未収録ではなく取得失敗。再実行すること):")
        for m in ng:
            print(f"    {m['project']}/{m['symbol']}: {m.get('error','')[:80]}")


if __name__ == "__main__":
    main()
