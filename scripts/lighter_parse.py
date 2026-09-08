r"""Lighter 生記録 → 1 秒グリッド基礎特徴量 + BBO 系列(第1段)。

入力(E:\Lighter　データ、読み取りのみ):
  data/raw/ws/order_book/{SYM}/dt=*/*.jsonl.gz   50ms 刻みの L2 差分(size は絶対値、0=消滅)
  data/raw/ws/trade/{SYM}/...                    約定(is_maker_ask で テイカー方向が確定)
  data/raw/ws/ticker/{SYM}/...                   BBO 変化ストリーム(照合と OFI 用)

出力(E:/Memory-lighter/、ローカルのみ・.gitignore 対象):
  grid_{SYM}.parquet   1 秒グリッド ~100 列(板状態 + 秒内フロー + 品質フラグ)
  bbo_{SYM}.parquet    ticker の全 BBO 変化(リードラグ用: サーバ µs + 受信 ns)
  meta_{SYM}.json      枠数・切断・snapshot・切れた gz などの台帳

=============================================================================
★実装上の事実(reports/step0_recon.md・granularity_ceiling_20260826.md より)
=============================================================================
- L2 は 50ms バッチ。バッチ内の複数イベントは 1 フレームに集約され分離できない。
  よって秒内フローは「フレーム差分の合計」であり、真のイベント数ではない
- 鎖の検証は begin_nonce == 直前フレームの nonce。切れたら n_gap を数え、
  suspect を立てる。suspect は snapshot か、ticker BBO と 10 秒連続一致で解除
- `subscribed/order_book` が全量 snapshot(鎖の再開点)。フローに数えない
- 再起動時の .r1/.r2 ファイルは名前順で正しく時系列に並ぶ
- 切れた gz(EOFError)は「そこまで読んで次のファイルへ」(gzip 追記トラップの回避構造)
- 時刻は last_updated_at(エンジン µs)。recv_ns(受信、このマシンの時計)との差を
  lag として毎秒記録する

★時間契約: 全列は「その秒の終わりまでに判る値」だけで作る。未来は見ない。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
import zlib
from array import array
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

RAW = r"E:/Lighter　データ/data/raw/ws"
OUT = Path("E:/Memory-lighter")
SYMS = ["DRAM", "MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD",
        "AAPL", "AMZN", "H100", "MSFT", "NVDA", "TSLA", "XAG", "XAU"]
BP = 1e4
KS = (1, 2, 3, 5, 10, 20)
WBPS = (5, 10, 25, 50, 100)


def files_for(sym: str, kind: str, d0: str, d1: str) -> list[str]:
    fs = []
    for p in sorted(glob.glob(f"{RAW}/{kind}/{sym}/dt=*/*.jsonl.gz")):
        dt = p.replace("\\", "/").split("dt=")[1][:10]
        if d0 <= dt <= d1:
            fs.append(p)
    return fs


def iter_lines(paths: list[str], meta: dict):
    """gz を順に読む。切れた・壊れたファイルは読めたところまでで次へ。

    ★kill された gz は EOFError だけでなく zlib.error(invalid block type)にもなる。
      08-23 の WS 再接続嵐で 300 件実在する。
    """
    for p in paths:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line
        except (EOFError, OSError, zlib.error):
            meta["truncated_gz"] += 1


class Cols:
    """列名 → array('d') の束。行はまとめて追記する。"""

    def __init__(self, names):
        self.names = list(names)
        self.a = {n: array("d") for n in self.names}

    def add(self, row: dict):
        for n in self.names:
            self.a[n].append(row.get(n, float("nan")))


# ---------------------------------------------------------------- ticker 走査
def pass_ticker(sym: str, d0: str, d1: str, meta: dict):
    """BBO 変化列(リードラグ用)と、秒ごとの ticker 由来量を作る。

    OFI は Cont 型を BBO 更新ごとに積む:
      bid 側: b_t > b_{t-1} → +Bsz_t / b_t == → +ΔBsz / b_t < → −Bsz_{t-1}
      ask 側は符号を逆に。
    """
    import numpy as np
    ts_us = array("q"); recv = array("q")
    bidp = array("d"); askp = array("d"); bids = array("d"); asks_ = array("d")
    per = {}   # ts_s -> [n_bbo, ofi, mid_first, mid_last, mid_hi, mid_lo,
               #          path, spr_sum, n_spr, lag_sum]
    pb = pa = pbs = pas = None
    for line in iter_lines(files_for(sym, "ticker", d0, d1), meta):
        try:
            d = json.loads(line)
        except Exception:
            meta["bad_json"] += 1
            continue
        m = d["m"]
        tk = m.get("ticker")
        if not tk:
            continue
        lu = int(m.get("last_updated_at") or tk.get("last_updated_at") or 0)
        if lu <= 0:
            continue
        b, a = tk.get("b"), tk.get("a")
        if not b or not a:
            continue
        bp_, ap_ = float(b["price"]), float(a["price"])
        bs_, as__ = float(b["size"]), float(a["size"])
        rn = int(d["recv_ns"])
        ts_us.append(lu); recv.append(rn)
        bidp.append(bp_); askp.append(ap_); bids.append(bs_); asks_.append(as__)
        s = lu // 1_000_000
        st = per.get(s)
        mid = (bp_ + ap_) / 2
        if st is None:
            st = per[s] = [0, 0.0, mid, mid, mid, mid, 0.0, 0.0, 0, 0.0]
        ofi = 0.0
        if pb is not None:
            if bp_ > pb:
                ofi += bs_
            elif bp_ == pb:
                ofi += bs_ - pbs
            else:
                ofi -= pbs
            if ap_ < pa:
                ofi -= as__
            elif ap_ == pa:
                ofi -= as__ - pas
            else:
                ofi += pas
            pmid = (pb + pa) / 2
            st[6] += abs(mid - pmid)
        st[0] += 1
        st[1] += ofi
        st[3] = mid
        if mid > st[4]:
            st[4] = mid
        if mid < st[5]:
            st[5] = mid
        st[7] += (ap_ - bp_)
        st[8] += 1
        st[9] += (rn / 1e6 - lu / 1e3)      # 受信遅延 ms
        pb, pa, pbs, pas = bp_, ap_, bs_, as__

    import polars as pl
    bbo = pl.DataFrame({
        "ts_us": np.asarray(ts_us, dtype=np.int64),
        "recv_ns": np.asarray(recv, dtype=np.int64),
        "bid": np.asarray(bidp), "ask": np.asarray(askp),
        "bid_sz": np.asarray(bids, dtype=np.float32),
        "ask_sz": np.asarray(asks_, dtype=np.float32),
    })
    bbo.write_parquet(OUT / f"bbo_{sym}.parquet")
    meta["ticker_rows"] = bbo.height
    return per


# ------------------------------------------------------------- order_book 走査
def side_new(is_bid: bool):
    return {}, []                     # dict px->sz, sorted price list


def pass_orderbook(sym: str, d0: str, d1: str, meta: dict, tick_per: dict,
                   grid: Cols):
    bidd, bidl = side_new(True)       # bidl 昇順(最良 = 末尾)
    askd, askl = side_new(False)      # askl 昇順(最良 = 先頭)
    tot_b = tot_a = 0.0
    book_valid = False
    suspect = 0.0
    match_run = 0
    last_nonce = None
    cur_s = None
    n_frames = n_gap = n_lvl = 0
    inc_b = inc_a = dec_b = dec_a = 0.0
    incn_b = incn_a = decn_b = decn_a = 0.0
    lag_sum = 0.0
    snap_in_s = 0.0

    def apply(levels, is_bid, is_update, mid_pre):
        nonlocal tot_b, tot_a, inc_b, inc_a, dec_b, dec_a
        nonlocal incn_b, incn_a, decn_b, decn_a, n_lvl
        d, lst = (bidd, bidl) if is_bid else (askd, askl)
        for lv in levels:
            px = float(lv["price"]); sz = float(lv["size"])
            old = d.get(px, 0.0)
            if sz == old:
                continue
            near = (mid_pre > 0 and abs(px - mid_pre) / mid_pre * BP <= 10.0)
            dlt = sz - old
            if is_update:
                n_lvl += 1
                if dlt > 0:
                    if is_bid:
                        inc_b += dlt
                        if near:
                            incn_b += dlt
                    else:
                        inc_a += dlt
                        if near:
                            incn_a += dlt
                else:
                    if is_bid:
                        dec_b -= dlt
                        if near:
                            decn_b -= dlt
                    else:
                        dec_a -= dlt
                        if near:
                            decn_a -= dlt
            if is_bid:
                tot_b += dlt
            else:
                tot_a += dlt
            if sz == 0.0:
                del d[px]
                i = bisect_left(lst, px)
                if i < len(lst) and lst[i] == px:
                    lst.pop(i)
            else:
                if old == 0.0:
                    insort(lst, px)
                d[px] = sz

    def close_second(s):
        """秒 s の行を確定する。"""
        nonlocal n_frames, n_gap, n_lvl, inc_b, inc_a, dec_b, dec_a
        nonlocal incn_b, incn_a, decn_b, decn_a, lag_sum, suspect, match_run
        nonlocal snap_in_s
        row = {"ts_s": float(s), "n_frames": float(n_frames),
               "n_gap": float(n_gap), "suspect": suspect,
               "snap": snap_in_s, "n_lvl_ch": float(n_lvl),
               "ob_lag_ms": lag_sum / n_frames if n_frames else float("nan"),
               "inc_b": inc_b, "inc_a": inc_a, "dec_b": dec_b, "dec_a": dec_a,
               "incn_b": incn_b, "incn_a": incn_a,
               "decn_b": decn_b, "decn_a": decn_a}
        ok = book_valid and bidl and askl
        if ok:
            bb = bidl[-1]; ba = askl[0]
            row["bb"] = bb; row["ba"] = ba
            row["crossed"] = 1.0 if bb >= ba else 0.0
            row["bb_sz"] = bidd[bb]; row["ba_sz"] = askd[ba]
            row["tot_b"] = tot_b; row["tot_a"] = tot_a
            mid = (bb + ba) / 2
            nb = len(bidl); na = len(askl)
            # 上位 K 累積
            cb = 0.0
            for k in range(1, min(KS[-1], nb) + 1):
                cb += bidd[bidl[nb - k]]
                if k in KS:
                    row[f"D{k}_b"] = cb
            ca = 0.0
            for k in range(1, min(KS[-1], na) + 1):
                ca += askd[askl[k - 1]]
                if k in KS:
                    row[f"D{k}_a"] = ca
            if bb < ba:
                # mid から X bp 以内の量と、100bp 以内のレベル数
                for w in WBPS:
                    lo = mid * (1 - w / BP)
                    i = bisect_left(bidl, lo)
                    row[f"W{w}_b"] = sum(bidd[p] for p in bidl[i:])
                    hi = mid * (1 + w / BP)
                    j = bisect_right(askl, hi)
                    row[f"W{w}_a"] = sum(askd[p] for p in askl[:j])
                row["nlv100_b"] = float(nb - bisect_left(bidl, mid * (1 - 100 / BP)))
                row["nlv100_a"] = float(bisect_right(askl, mid * (1 + 100 / BP)))
                # 形状(上位 20 レベル)
                for tag, lvls, szd in (("b", bidl[max(0, nb - 20):][::-1], bidd),
                                       ("a", askl[:20], askd)):
                    q = [szd[p] for p in lvls]
                    tq = sum(q)
                    if tq > 0 and len(lvls) >= 2:
                        dist = [abs(p - mid) / mid * BP for p in lvls]
                        row[f"wavg_bp_{tag}"] = sum(dd * qq for dd, qq in zip(dist, q)) / tq
                        row[f"px_range20_bp_{tag}"] = dist[-1] - dist[0]
                        q10 = q[:10]; t10 = sum(q10)
                        if t10 > 0:
                            row[f"hhi10_{tag}"] = sum((x / t10) ** 2 for x in q10)
                            mx = max(q10)
                            row[f"wallsh10_{tag}"] = mx / t10
                            row[f"walldist10_bp_{tag}"] = dist[q10.index(mx)]
                        row[f"gap12_bp_{tag}"] = abs(lvls[1] - lvls[0]) / mid * BP
                        gaps = [abs(lvls[i2 + 1] - lvls[i2]) for i2 in range(min(9, len(lvls) - 1))]
                        row[f"maxgap10_bp_{tag}"] = max(gaps) / mid * BP
        else:
            row["crossed"] = float("nan")
        # ticker 由来
        tk = tick_per.get(s)
        if tk is not None:
            row["n_bbo"] = float(tk[0]); row["ofi"] = tk[1]
            row["mid_first"] = tk[2]; row["mid_last"] = tk[3]
            row["mid_hi"] = tk[4]; row["mid_lo"] = tk[5]
            row["path"] = tk[6]
            row["spr_mean"] = tk[7] / tk[8] if tk[8] else float("nan")
            row["tk_lag_ms"] = tk[9] / tk[0] if tk[0] else float("nan")
        grid.add(row)
        n_frames = n_gap = n_lvl = 0
        inc_b = inc_a = dec_b = dec_a = 0.0
        incn_b = incn_a = decn_b = decn_a = 0.0
        lag_sum = 0.0
        snap_in_s = 0.0

    for line in iter_lines(files_for(sym, "order_book", d0, d1), meta):
        try:
            d = json.loads(line)
        except Exception:
            meta["bad_json"] += 1
            continue
        m = d["m"]
        typ = m.get("type", "")
        ob = m.get("order_book")
        if ob is None:
            continue
        lu = int(m.get("last_updated_at") or ob.get("last_updated_at") or 0)
        if lu <= 0:
            continue
        s = lu // 1_000_000
        if cur_s is None:
            cur_s = s
        while s > cur_s:
            close_second(cur_s)
            cur_s += 1
            if cur_s < s and (s - cur_s) > 60:
                # 60 秒を超える空白は行を出さない(切断とみなす)
                meta["silence_gaps"] += 1
                cur_s = s
        if typ.startswith("subscribed/"):
            bidd.clear(); bidl.clear(); askd.clear(); askl.clear()
            tot_b = tot_a = 0.0
            apply(ob.get("bids") or [], True, False, 0.0)
            apply(ob.get("asks") or [], False, False, 0.0)
            book_valid = True
            suspect = 0.0
            match_run = 0
            last_nonce = ob.get("nonce")
            meta["snapshots"] += 1
            snap_in_s = 1.0
            n_frames += 1
            continue
        bn = ob.get("begin_nonce")
        if last_nonce is not None and bn is not None and bn != last_nonce:
            n_gap += 1
            meta["nonce_gaps"] += 1
            suspect = 1.0
            match_run = 0
        last_nonce = ob.get("nonce", last_nonce)
        mid_pre = 0.0
        if bidl and askl:
            mid_pre = (bidl[-1] + askl[0]) / 2
        apply(ob.get("bids") or [], True, True, mid_pre)
        apply(ob.get("asks") or [], False, True, mid_pre)
        n_frames += 1
        lag_sum += d["recv_ns"] / 1e6 - lu / 1e3
        # suspect の解除: ticker BBO と一致が 10 秒続いたら信用する
        if suspect and bidl and askl:
            tk = tick_per.get(s)
            if tk is not None and abs((bidl[-1] + askl[0]) / 2 - tk[3]) < 1e-9:
                match_run += 1
                if match_run >= 10:
                    suspect = 0.0
            else:
                match_run = 0
    if cur_s is not None:
        close_second(cur_s)


# ------------------------------------------------------------------ trade 走査
def pass_trades(sym: str, d0: str, d1: str, meta: dict):
    per = {}   # ts_s -> [tb_vol, ts_vol, tb_cnt, ts_cnt, usd, max_trd, n_liq]
    for line in iter_lines(files_for(sym, "trade", d0, d1), meta):
        try:
            d = json.loads(line)
        except Exception:
            meta["bad_json"] += 1
            continue
        m = d["m"]
        for t in (m.get("trades") or []):
            ts = int(t.get("timestamp") or 0)
            if ts <= 0:
                continue
            s = ts // 1000
            st = per.get(s)
            if st is None:
                st = per[s] = [0.0, 0.0, 0, 0, 0.0, 0.0, 0]
            sz = float(t["size"])
            if t.get("is_maker_ask"):
                st[0] += sz; st[2] += 1          # テイカー買い
            else:
                st[1] += sz; st[3] += 1          # テイカー売り
            st[4] += float(t.get("usd_amount") or 0.0)
            if sz > st[5]:
                st[5] = sz
        nl = len(m.get("liquidation_trades") or [])
        if nl:
            # 清算はどの秒か: 同メッセージの trades の秒に寄せる(近似)
            if per:
                per[max(per)][6] += nl
    meta["trade_seconds"] = len(per)
    return per


GRID_COLS = (
    ["ts_s", "n_frames", "n_gap", "suspect", "snap", "n_lvl_ch", "ob_lag_ms",
     "inc_b", "inc_a", "dec_b", "dec_a", "incn_b", "incn_a", "decn_b", "decn_a",
     "bb", "ba", "crossed", "bb_sz", "ba_sz", "tot_b", "tot_a"]
    + [f"D{k}_{t}" for k in KS for t in ("b", "a")]
    + [f"W{w}_{t}" for w in WBPS for t in ("b", "a")]
    + ["nlv100_b", "nlv100_a"]
    + [f"{n}_{t}" for n in ("wavg_bp", "px_range20_bp", "hhi10", "wallsh10",
                            "walldist10_bp", "gap12_bp", "maxgap10_bp")
       for t in ("b", "a")]
    + ["n_bbo", "ofi", "mid_first", "mid_last", "mid_hi", "mid_lo", "path",
       "spr_mean", "tk_lag_ms",
       "tb_vol", "ts_vol", "tb_cnt", "ts_cnt", "usd_vol", "max_trd", "n_liq"]
)


def run_symbol(sym: str, d0: str, d1: str) -> None:
    import numpy as np
    import polars as pl
    t0 = time.time()
    meta = {"symbol": sym, "d0": d0, "d1": d1, "truncated_gz": 0, "bad_json": 0,
            "snapshots": 0, "nonce_gaps": 0, "silence_gaps": 0}
    tick_per = pass_ticker(sym, d0, d1, meta)
    trade_per = pass_trades(sym, d0, d1, meta)
    grid = Cols(GRID_COLS)
    pass_orderbook(sym, d0, d1, meta, tick_per, grid)
    df = pl.DataFrame({n: np.asarray(grid.a[n]) for n in grid.names})
    # trade 列を合流
    tsec = np.asarray(df["ts_s"].to_numpy(), dtype=np.int64)
    tv = np.full((len(tsec), 7), 0.0)
    for i, s in enumerate(tsec):
        st = trade_per.get(int(s))
        if st is not None:
            tv[i] = st
    for j, n in enumerate(["tb_vol", "ts_vol", "tb_cnt", "ts_cnt",
                           "usd_vol", "max_trd", "n_liq"]):
        df = df.with_columns(pl.Series(n, tv[:, j]))
    df.write_parquet(OUT / f"grid_{sym}.parquet")
    meta["grid_rows"] = df.height
    meta["sec"] = round(time.time() - t0, 1)
    with open(OUT / f"meta_{sym}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{sym}: grid {df.height:,} 行 / ticker {meta.get('ticker_rows', 0):,} 行 "
          f"/ snapshot {meta['snapshots']} / gap {meta['nonce_gaps']} "
          f"/ 切れgz {meta['truncated_gz']} / {meta['sec']}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMS))
    ap.add_argument("--start", default="2026-08-18")
    ap.add_argument("--end", default="2026-09-05")
    ap.add_argument("--out", default="", help="出力先を変える(既存を壊さない)")
    a = ap.parse_args()
    if a.out:
        globals()["OUT"] = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in a.symbols.split(","):
        run_symbol(sym, a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
