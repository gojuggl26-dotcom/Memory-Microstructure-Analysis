r"""口座パネル — 特徴量 48〜54(参加者・集中度・メイカー/テイカー・寿命)。

【なぜこれを先にやるか】
  `account` は 500 万件の `placed` すべてに付いているのに、既存レポート 0〜7 は
  一度も使っていない。市場 163 では発注口座が 41 しかなく上位 2 者で 80.6% を占める。
  **「板の形が金利を予測する」という既存の結論が、数口座の癖を測っていただけ**
  という可能性を、これで初めて検証できる。

【素材と結合の仕方】
  placed               … account / order_id / side / tick / order_index / size / rate
  cancelled            … account が**無い** → 同じ order_id の placed から結合する
  partially_filled     … order_id あり(部分約定)
  filled               … order_id が無く from_id〜to_id の**範囲**。範囲に入る
                         生存注文をすべて消す(板再生と同じ規則)
  market_orders_filled … **テイカーの account**。板からの除去は同 tx の filled 側が担う。
                         ★`size_raw` は**実約定量ではない**。実測(市場 163)で 375 件すべてが
                           2^128 を超え、先頭桁が 2^256 と一致する 1e39 級の番兵値だった
                           (= 「上限なし」の指定)。**テイカー量は同一 tx のメイカー側
                           除去量の合計から求める**
  otc_swap             … **板を経由しない**ので口座パネルからは除外する(件数だけ記録)

【出す量(市場 × 口座)】
  n_placed / vol_placed / n_cancelled / vol_cancelled / n_filled_maker / vol_filled_maker
  n_taker / vol_taker / median_size / cancel_rate / fill_rate
  median_lifetime_s / p90_lifetime_s / first_ts / last_ts / active_days
  median_order_index(キューのどこに置くか)

【出す量(市場)】
  n_accounts / HHI / top1_share / top2_share / top5_share
  n_maker_only / n_taker_only / n_both

★時刻はブロック時刻(秒)。ブロック内の順序は log_index に従うが、
  これは**ログの出力順であってマッチング順の保証ではない**ので、
  サブブロックの因果は主張しない。

出力: data/accounts_market.parquet(市場サマリ)/ data/accounts_panel.parquet(口座別)
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Boros-history")          # ★読み取り専用
OUT = Path(__file__).resolve().parent.parent / "data"
WAD = 1e18


def block_clock(mid: int):
    """blockNumber → 秒。板再生スクリプトと同一の線形内挿。"""
    d = None
    for p in (SRC / "data/parquet/order-book").glob(f"{mid}-*"):
        d = p
        break
    fs = sorted(glob.glob(str(d / "raw" / "*.parquet"))) if d else []
    if not fs:
        return None
    raw = pl.concat([pl.read_parquet(f, columns=["blockNumber", "blockTimestamp"])
                     for f in fs], how="diagonal_relaxed").sort("blockNumber")
    b = raw["blockNumber"].to_list()
    t = raw["blockTimestamp"].to_list()
    if len(b) < 2:
        return None

    def f(blk: int) -> float:
        i = bisect.bisect_left(b, blk)
        if i <= 0:
            r = (t[1] - t[0]) / (b[1] - b[0])
            return t[0] + (blk - b[0]) * r
        if i >= len(b):
            r = (t[-1] - t[-2]) / (b[-1] - b[-2])
            return t[-1] + (blk - b[-1]) * r
        w = (blk - b[i - 1]) / (b[i] - b[i - 1])
        return t[i - 1] + w * (t[i] - t[i - 1])
    return f


def run(mid: int) -> tuple[pl.DataFrame, dict] | None:
    p = SRC / f"data/chain/events/marketId={mid}.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).sort(["block_number", "log_index"])
    if d.height < 200:
        return None
    clock = block_clock(mid)
    if clock is None:
        return None

    # --- 注文台帳: order_id -> (account, size, tick, side, order_index, ts) ---
    live: dict[int, list] = {}
    owner: dict[int, str] = {}          # order_id -> account(取消の帰属に要る)
    acc: dict[str, dict] = {}
    n_otc = 0
    # tx ごとに「メイカー側から消えた量」と「テイカーの口座」を集める。
    # market_orders_filled の size_raw は番兵値なので使えない。
    tx_removed: dict[str, float] = {}
    tx_takers: dict[str, set] = {}
    n_tx_multi_taker = 0

    def rec(a, key, v=1.0):
        if a is None:
            return
        s = acc.setdefault(a, {k: 0.0 for k in
                               ("n_placed", "vol_placed", "n_cancelled", "vol_cancelled",
                                "n_filled_maker", "vol_filled_maker", "n_taker", "vol_taker")})
        s[key] += v

    sizes: dict[str, list] = {}
    oidx: dict[str, list] = {}
    lifet: dict[str, list] = {}
    seen: dict[str, list] = {}

    for r in d.iter_rows(named=True):
        ev = r["event"]
        ts = clock(r["block_number"])
        if ev == "otc_swap":
            n_otc += 1
            continue
        if ev == "placed":
            a = r["account"]
            sz = int(r["size_raw"]) / WAD
            oid = r["order_id"]
            live[oid] = [a, sz, ts]
            owner[oid] = a
            rec(a, "n_placed"); rec(a, "vol_placed", sz)
            sizes.setdefault(a, []).append(sz)
            if r["order_index"] is not None:
                oidx.setdefault(a, []).append(int(r["order_index"]))
            s = seen.setdefault(a, [ts, ts]); s[0] = min(s[0], ts); s[1] = max(s[1], ts)
        elif ev in ("cancelled", "forced_cancelled"):
            oid = r["order_id"]
            o = live.pop(oid, None)
            a = o[0] if o else owner.get(oid)
            if o:
                rec(a, "n_cancelled"); rec(a, "vol_cancelled", o[1])
                lifet.setdefault(a, []).append(ts - o[2])
            else:
                rec(a, "n_cancelled")
            if a:
                s = seen.setdefault(a, [ts, ts]); s[1] = max(s[1], ts)
        elif ev == "partially_filled":
            oid = r["order_id"]
            o = live.get(oid)
            if o:
                dsz = int(r["size_raw"]) / WAD
                rec(o[0], "n_filled_maker"); rec(o[0], "vol_filled_maker", dsz)
                o[1] -= dsz
                tx_removed[r["tx_hash"]] = tx_removed.get(r["tx_hash"], 0.0) + dsz
                if o[1] <= 0:
                    live.pop(oid, None)
                    lifet.setdefault(o[0], []).append(ts - o[2])
        elif ev in ("filled", "oob_purged"):
            lo, hi = r["from_id"], r["to_id"]
            for oid in [x for x in live if lo <= x <= hi]:
                o = live.pop(oid)
                rec(o[0], "n_filled_maker"); rec(o[0], "vol_filled_maker", o[1])
                tx_removed[r["tx_hash"]] = tx_removed.get(r["tx_hash"], 0.0) + o[1]
                lifet.setdefault(o[0], []).append(ts - o[2])
        elif ev == "market_orders_filled":
            a = r["account"]
            rec(a, "n_taker")          # 量は tx 単位で後から付ける
            if a:
                tx_takers.setdefault(r["tx_hash"], set()).add(a)
            if a:
                s = seen.setdefault(a, [ts, ts]); s[0] = min(s[0], ts); s[1] = max(s[1], ts)

    # --- テイカー量を tx 単位で帰属する ---
    for tx, takers in tx_takers.items():
        vol = tx_removed.get(tx, 0.0)
        if len(takers) == 1:
            rec(next(iter(takers)), "vol_taker", vol)
        else:
            n_tx_multi_taker += 1        # 複数テイカーの tx は按分せず計上しない
    if not acc:
        return None
    rows = []
    for a, s in acc.items():
        lt = np.array(lifet.get(a, []), dtype=float)
        sz = np.array(sizes.get(a, []), dtype=float)
        oi = np.array(oidx.get(a, []), dtype=float)
        fs, ls = seen.get(a, [np.nan, np.nan])
        term = s["n_cancelled"] + s["n_filled_maker"]
        rows.append({
            "market": mid, "account": a, **{k: v for k, v in s.items()},
            "median_size": float(np.median(sz)) if sz.size else np.nan,
            "median_order_index": float(np.median(oi)) if oi.size else np.nan,
            "cancel_rate": s["n_cancelled"] / term if term else np.nan,
            "fill_rate": s["n_filled_maker"] / term if term else np.nan,
            "median_lifetime_s": float(np.median(lt)) if lt.size else np.nan,
            "p90_lifetime_s": float(np.percentile(lt, 90)) if lt.size else np.nan,
            "first_ts": fs, "last_ts": ls,
            "active_days": (ls - fs) / 86400 if np.isfinite(fs) else np.nan,
        })
    panel = pl.DataFrame(rows)

    v = panel["vol_placed"].to_numpy()
    tot = v.sum()
    sh = np.sort(v / tot)[::-1] if tot > 0 else np.array([0.0])
    mk = (panel["n_placed"].to_numpy() > 0)
    tk = (panel["n_taker"].to_numpy() > 0)
    summ = {
        "market": mid, "n_accounts": int(panel.height),
        "n_events": int(d.height), "n_otc": n_otc,
        "hhi": float((sh ** 2).sum()),
        "top1_share": float(sh[0]) if sh.size else np.nan,
        "top2_share": float(sh[:2].sum()) if sh.size else np.nan,
        "top5_share": float(sh[:5].sum()) if sh.size else np.nan,
        "n_tx_multi_taker": n_tx_multi_taker,
        "n_maker_only": int((mk & ~tk).sum()), "n_taker_only": int((~mk & tk).sum()),
        "n_both": int((mk & tk).sum()),
        "vol_placed_total": float(tot),
        "vol_taker_total": float(panel["vol_taker"].sum()),
    }
    return panel, summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="")
    a = ap.parse_args()
    if a.markets:
        mids = [int(x) for x in a.markets.split(",")]
    else:
        mids = sorted(int(re.search(r"marketId=(\d+)", f).group(1))
                      for f in glob.glob(str(SRC / "data/chain/events/marketId=*.parquet")))
    panels, summs = [], []
    for m in mids:
        try:
            got = run(m)
        except Exception as e:
            print(f"  market {m}: ★失敗 {type(e).__name__}: {e}", flush=True)
            continue
        if got is None:
            continue
        p, s = got
        panels.append(p); summs.append(s)
        print(f"  market {m:>4}: 口座 {s['n_accounts']:>4}  top1 {s['top1_share']:.1%}  "
              f"HHI {s['hhi']:.3f}", flush=True)
    if not panels:
        print("結果なし"); return 1
    P = pl.concat(panels, how="diagonal_relaxed")
    S = pl.DataFrame(summs)
    OUT.mkdir(parents=True, exist_ok=True)
    P.write_parquet(OUT / "accounts_panel.parquet")
    S.write_parquet(OUT / "accounts_market.parquet")
    print(f"\n市場 {S.height} / 口座エントリ {P.height:,}")
    print(f"-> {OUT/'accounts_panel.parquet'} / {OUT/'accounts_market.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
