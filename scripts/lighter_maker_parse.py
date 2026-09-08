r"""Lighter maker 検証の第 1 段 — 約定ストリームの取り込みと板の監査。

    uv run python scripts/lighter_maker_parse.py                    # 全 12 銘柄
    uv run python scripts/lighter_maker_parse.py --audit MU --day 2026-09-05

入力(E:\Lighter　データ、読み取りのみ):
  data/raw/ws/trade/{SYM}/dt=*/*.jsonl.gz        約定(is_maker_ask でテイカー方向が確定)
  data/raw/ws/order_book/{SYM}/dt=*/*.jsonl.gz   50ms L2 差分(監査でのみ読む)

出力(E:/Memory-lighter/mk/、ローカルのみ):
  trades_{SYM}.parquet         recv_ns / ts_ms / px / sz / is_maker_ask / trade_id
  audit_trades.csv             受信順とサーバ順の逆転・重複・受信遅れ・日別件数
  audit_book_{DAY}.json        板の減少のうち約定で説明できる割合(Q1 の前提の検証)

=============================================================================
★なぜ order_book の全再生を主経路にしないか
=============================================================================
主構成の指値は **BBO** に置く。BBO の表示数量は ticker が直接くれる
(`bbo_{SYM}.parquet` に既にある)。strict-trade モデル(Q1)は
「live 時点の同価格表示数量を前方数量とし、**約定だけ**がそれを削る」ので、
板の全水準は要らない。order_book は銘柄あたり 988MB あるのに対し trade は 28MB で
35 倍の差がある。板は **Q1 の前提が正しいかの監査**と、BBO 以外に置く感応度でだけ読む。

★時間契約: 本段は変換のみ。時刻は 2 本を別々に持つ。
  recv_ns … このマシンが受け取った時刻(= 実際に観測可能な時刻。発注判断に使う)
  ts_ms  … 取引所エンジンの時刻(監査と照合にだけ使う。発注時刻には使わない)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
import zlib
from bisect import bisect_left, insort
from pathlib import Path

import numpy as np
import polars as pl

RAW = r"E:/Lighter　データ/data/raw/ws"
OUT = Path("E:/Memory-lighter/mk")
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "DRAM",
        "XAU", "XAG", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT"]
D0, D1 = "2026-08-18", "2026-09-07"


def files_for(sym, kind, d0, d1):
    fs = []
    for p in sorted(glob.glob(f"{RAW}/{kind}/{sym}/dt=*/*.jsonl.gz")):
        dt = p.replace("\\", "/").split("dt=")[1][:10]
        if d0 <= dt <= d1:
            fs.append(p)
    return fs


def iter_lines(paths, meta):
    """gz を順に読む。kill された gz は EOFError にも zlib.error にもなる。"""
    for p in paths:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line
        except (EOFError, zlib.error, OSError):
            meta["broken_gz"] += 1


def parse_trades(sym, d0, d1):
    meta = {"sym": sym, "broken_gz": 0, "frames": 0, "rows": 0, "dup": 0}
    recv, tms, px, sz, mka, tid = [], [], [], [], [], []
    for line in iter_lines(files_for(sym, "trade", d0, d1), meta):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("m") or {}
        ts = m.get("trades")
        if not ts:
            continue
        meta["frames"] += 1
        r = int(d["recv_ns"])
        for t in ts:
            recv.append(r)
            tms.append(int(t.get("timestamp", 0)))
            px.append(float(t["price"]))
            sz.append(float(t["size"]))
            mka.append(bool(t["is_maker_ask"]))
            tid.append(int(t["trade_id"]))
    if not recv:
        return pl.DataFrame(), meta
    T = pl.DataFrame({"recv_ns": np.asarray(recv, np.int64),
                      "ts_ms": np.asarray(tms, np.int64),
                      "px": np.asarray(px, np.float64),
                      "sz": np.asarray(sz, np.float64),
                      "is_maker_ask": np.asarray(mka, bool),
                      "trade_id": np.asarray(tid, np.int64)})
    n0 = T.height
    T = T.unique(subset=["trade_id"], keep="first")
    meta["dup"] = n0 - T.height
    # ★受信時刻で明示的に並べ替える(受信時刻系列は受信時刻でソートする)
    T = T.sort(["recv_ns", "trade_id"])
    meta["rows"] = T.height
    a = T["ts_ms"].to_numpy()
    meta["inv_server_order"] = int((np.diff(a) < 0).sum())
    meta["inv_frac"] = float(meta["inv_server_order"] / max(len(a) - 1, 1))
    lag = (T["recv_ns"].to_numpy() / 1e6) - a
    meta["lag_ms_p50"] = float(np.median(lag))
    meta["lag_ms_p10"] = float(np.quantile(lag, 0.10))
    meta["lag_ms_p90"] = float(np.quantile(lag, 0.90))
    return T, meta


def audit_book(sym, day):
    """Q1 の前提の監査 — 最良気配の数量減少のうち、約定で説明できる割合。

    板を snapshot から再生し、50ms フレームごとに「最良の表示数量が減った量」と
    「同じフレーム区間に受信した同価格の約定量」を突き合わせる。
    Q1 は取消由来の減少を前方数量から引かないので、説明できない減少が多いほど
    Q1 は保守的(= 約定しにくい側)に外れる。
    """
    meta = {"broken_gz": 0}
    T, _ = parse_trades(sym, day, day)
    z64, z = np.zeros(0, np.int64), np.zeros(0)
    trec = T["recv_ns"].to_numpy() if T.height else z64
    tpx = T["px"].to_numpy() if T.height else z
    tsz = T["sz"].to_numpy() if T.height else z
    tmk = T["is_maker_ask"].to_numpy() if T.height else np.zeros(0, bool)
    bidd, askd = {}, {}
    bidl, askl = [], []
    o = {"frames": 0, "nonce_gap": 0, "snap": 0,
         "dec_b": 0.0, "dec_a": 0.0, "exp_b": 0.0, "exp_a": 0.0,
         "batch_multi": 0, "n_dec_b": 0, "n_dec_a": 0}
    last_nonce = None
    prev_recv = 0

    def apply(levels, d, lst):
        for lv in levels:
            p = float(lv["price"])
            s = float(lv["size"])
            if s == 0.0:
                if p in d:
                    del d[p]
                    i = bisect_left(lst, p)
                    if i < len(lst) and lst[i] == p:
                        lst.pop(i)
            else:
                if p not in d:
                    insort(lst, p)
                d[p] = s

    for line in iter_lines(files_for(sym, "order_book", day, day), meta):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("m") or {}
        ob = m.get("order_book")
        if ob is None:
            continue
        r = int(d["recv_ns"])
        snap = str(m.get("type", "")).startswith("subscribed/")
        if snap:
            o["snap"] += 1
            bidd.clear(); askd.clear(); bidl.clear(); askl.clear()
            last_nonce = None
        else:
            n0 = ob.get("begin_nonce")
            if last_nonce is not None and n0 is not None and n0 != last_nonce:
                o["nonce_gap"] += 1
        last_nonce = ob.get("nonce")
        bb0 = bidl[-1] if bidl else None
        ba0 = askl[0] if askl else None
        sb0 = bidd.get(bb0, 0.0) if bb0 is not None else 0.0
        sa0 = askd.get(ba0, 0.0) if ba0 is not None else 0.0
        apply(ob.get("bids") or [], bidd, bidl)
        apply(ob.get("asks") or [], askd, askl)
        o["frames"] += 1
        if snap or bb0 is None or ba0 is None:
            prev_recv = r
            continue
        lo = np.searchsorted(trec, prev_recv, "right")
        hi = np.searchsorted(trec, r, "right")
        if hi - lo > 1:
            o["batch_multi"] += 1
        nb = bidd.get(bb0, 0.0)
        if nb < sb0:
            o["dec_b"] += sb0 - nb
            o["n_dec_b"] += 1
            mk = (~tmk[lo:hi]) & (np.abs(tpx[lo:hi] - bb0) < 1e-9)
            o["exp_b"] += float(tsz[lo:hi][mk].sum())
        na = askd.get(ba0, 0.0)
        if na < sa0:
            o["dec_a"] += sa0 - na
            o["n_dec_a"] += 1
            mk = (tmk[lo:hi]) & (np.abs(tpx[lo:hi] - ba0) < 1e-9)
            o["exp_a"] += float(tsz[lo:hi][mk].sum())
        prev_recv = r
    o["explained_b"] = o["exp_b"] / o["dec_b"] if o["dec_b"] else float("nan")
    o["explained_a"] = o["exp_a"] / o["dec_a"] if o["dec_a"] else float("nan")
    o["broken_gz"] = meta["broken_gz"]
    o["sym"] = sym
    o["day"] = day
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--d0", default=D0)
    ap.add_argument("--d1", default=D1)
    ap.add_argument("--audit", default="", help="板の監査を行う銘柄(カンマ区切り)")
    ap.add_argument("--day", default="", help="監査する日")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.audit:
        res = []
        for sym in a.audit.split(","):
            t0 = time.time()
            r = audit_book(sym, a.day)
            r["sec"] = round(time.time() - t0, 1)
            res.append(r)
            print(f"{sym} {a.day}: フレーム {r['frames']:,} / nonce 切れ "
                  f"{r['nonce_gap']} / snapshot {r['snap']} / "
                  f"約定で説明できた減少 買 {100*r['explained_b']:.1f}% "
                  f"売 {100*r['explained_a']:.1f}% / "
                  f"1 フレーム複数約定 {r['batch_multi']:,} ({r['sec']}s)",
                  flush=True)
        p = OUT / f"audit_book_{a.day}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("書き出し", p)
        return
    syms = a.symbols.split(",") if a.symbols else SYMS
    metas = []
    for sym in syms:
        t0 = time.time()
        T, m = parse_trades(sym, a.d0, a.d1)
        if not T.height:
            print(f"{sym}: 約定なし", flush=True)
            continue
        T.write_parquet(OUT / f"trades_{sym}.parquet")
        m["sec"] = round(time.time() - t0, 1)
        metas.append(m)
        print(f"{sym}: 約定 {m['rows']:,} 行 / 重複 {m['dup']} / "
              f"逆転 {m['inv_server_order']:,} ({100*m['inv_frac']:.3f}%) / "
              f"受信遅れ 中央 {m['lag_ms_p50']:.0f}ms "
              f"[{m['lag_ms_p10']:.0f}, {m['lag_ms_p90']:.0f}] / "
              f"壊れ gz {m['broken_gz']} ({m['sec']}s)", flush=True)
    if metas:
        pl.DataFrame(metas).write_csv(OUT / "audit_trades.csv")
        print("書き出し", OUT / "audit_trades.csv")


if __name__ == "__main__":
    main()
