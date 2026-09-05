"""l1(注文イベント)だけから最良気配の表 bbo を組み直す。

    uv run python scripts/build_bbo_l1.py --coin xyz:INTC [--days N]
    uv run python scripts/build_bbo_l1.py --coin xyz:MU --days 10 --suffix _l1

出力: data/bbo_l1_<coin>/dt=*.parquet と、束ねた data/bbo_<coin>.parquet
      (--suffix を付けると data/bbo_<coin><suffix>.parquet。検証用)

★なぜ要るか
------------
`fetch_bbo.py` が読む `l2/bbo` は **DEEP_ARCHIVE へ移行済み**で、
新しい銘柄については取得できない(2026-09-05 実測。復元は 12〜48 時間)。

    xyz:INTC  l2/bbo  STANDARD 99 / DEEP_ARCHIVE 135 (0.49 GiB)

一方 `l1` は GLACIER_IR なので直接読める。板の再構成は
`build_obi_levels.day_features` が既に l1 だけでやっているので、
同じ規則で **最良気配だけ**を取り出せば bbo の代わりになる。

板に載せる規則は `build_obi_levels` と 1 行ずつ同じにしてある
(RESTING の tif のみ / トリガー注文を除く / Rejected 系を除く /
数量は 0.001 のロット整数 / 同一 ns は open(0) → filled(2) → 終端(3))。

★この表は l2/bbo と同じものではない
------------------------------------
約定の一部は `filled` イベントを出さず、後日の取消イベントの
`orig_sz − remaining_sz` にしか現れない(hl-l4-pipeline の実測で
**取引数量の 29.17%**)。その間その注文は板に残ったままなので、

  * 最良の**数量が過大**になる
  * 既に約定した注文が最良に居座り、**買い ≥ 売り(クロス)に見える**

という偏りが出る。どちらも `l2` の状態機械が fills を統合して直している
ものなので、l1 単独では原理的に消せない。大きさは
`--validate` で xyz:MU の実物 `bbo_xyz_MU.parquet` と突き合わせて測る。

x が確定する時刻 / y の期間: 該当なし(板の再構成であって予測ではない)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import CANCELS, PX_UNIT, RESTING, SZ_LOT, TERMINAL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000
MARGIN = 160                    # 価格窓の余白(0.01 単位)


def day_events(fp: Path, carry: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """1 日分の l1 を「板の差分」に直す。carry は前日から残っている注文。"""
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status",
                                     "remaining_sz", "tif", "is_trigger"])
    t0 = (int(d["ts"].cast(pl.Int64).min()) // DAY_NS) * DAY_NS
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int64),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS)))
    del d
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("sz", "gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"]).with_columns(
        prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
        last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after"))
    # ★同一 ns の途中経過は sweep 側で捨てるので、ts だけで並べれば足りる
    #   (その ns の差分を全部当てたあとの板は、当てる順に依らない)
    delta = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
             .filter((~pl.col("synth")) & (pl.col("delta") != 0))
             .select("ts", "pidx", "is_bid", "delta").sort("ts"))
    return delta, nxt, t0


def sweep(delta: pl.DataFrame, carry: pl.DataFrame):
    """差分列を時刻順に当てて、最良気配が動いた瞬間だけを書き出す。

    最良の更新は 2 通りしかない。
      * 今の最良より内側に数量が入った  → その場で更新 (O(1))
      * 今の最良の数量が 0 になった      → 1 つずつ外へ探す (通常 数ティック)

    ★アンクロス(older-loses)
    ------------------------
    l1 だけで組んだ板は放っておくと **100% クロスする**(2026-09-05 実測。
    xyz:MU 05-05 12:00 で 買い 603.99 に対し 売り 549.52 が 0.029 だけ残る)。
    黙って約定した注文が終端イベントを出さずに残り続けるためで、残りかすは
    ほぼ必ず極小の数量である。

    そこで買い ≥ 売り になったら、**最後に動いたのが古いほうの価格水準**を
    板から落とす。hl-l4-pipeline のゲート3 で newer-loses が 82.4 万件の
    暴走を起こし、older-loses が 798 件で収束したのと同じ規則である
    (`CLAUDE.md` の L2 確定版 v5b)。板に残る注文が本当に約定していれば
    それは古い側なので、この向きでなければならない。
    """
    pi_a = delta["pidx"].to_numpy()
    cpx = carry["pidx"].to_numpy() if carry.height else pi_a[:1]
    p_min = int(min(pi_a.min(), cpx.min())) - MARGIN
    p_max = int(max(pi_a.max(), cpx.max())) + MARGIN
    W = p_max - p_min + 1

    depb = [0] * W
    depa = [0] * W
    tb = [0] * W                # その価格水準が最後に動いた時刻(古い側が負ける)
    ta = [0] * W
    if carry.height:
        cb = carry["is_bid"].to_numpy()
        cp = carry["pidx"].to_numpy() - p_min
        cs = carry["size_after"].to_numpy()
        for k in range(cp.size):                       # 繰越は日に数万本
            if cb[k]:
                depb[cp[k]] += int(cs[k])
            else:
                depa[cp[k]] += int(cs[k])
    bb = -1
    for p in range(W - 1, -1, -1):
        if depb[p]:
            bb = p
            break
    ba = W
    for p in range(W):
        if depa[p]:
            ba = p
            break

    ts_l = delta["ts"].to_numpy().tolist()
    px_l = (pi_a - p_min).tolist()
    bd_l = delta["is_bid"].to_numpy().tolist()
    dv_l = delta["delta"].to_numpy().tolist()

    o_ts: list[int] = []
    o_bb: list[int] = []
    o_ba: list[int] = []
    o_sb: list[int] = []
    o_sa: list[int] = []
    pb = pa = -2
    psb = psa = -1
    n_evict = 0
    v_evict = 0
    n_neg = 0
    for i in range(len(ts_l)):
        p = px_l[i]
        t = ts_l[i]
        if bd_l[i]:
            v = depb[p] + dv_l[i]
            if v < 0:                          # ★退去させた注文の取消が後から来る
                v = 0                          #   分。深さは負にならない
                n_neg += 1
            depb[p] = v
            tb[p] = t
            if v > 0:
                if p > bb:
                    bb = p
            elif p == bb:
                while bb >= 0 and not depb[bb]:
                    bb -= 1
        else:
            v = depa[p] + dv_l[i]
            if v < 0:
                v = 0
                n_neg += 1
            depa[p] = v
            ta[p] = t
            if v > 0:
                if p < ba:
                    ba = p
            elif p == ba:
                while ba < W and not depa[ba]:
                    ba += 1
        while 0 <= ba <= bb and bb < W:            # ★クロス/ロックを解く
            if ta[ba] <= tb[bb]:                   # 売りのほうが古い → 売りが負ける
                v_evict += depa[ba]
                depa[ba] = 0
                while ba < W and not depa[ba]:
                    ba += 1
            else:
                v_evict += depb[bb]
                depb[bb] = 0
                while bb >= 0 and not depb[bb]:
                    bb -= 1
            n_evict += 1
        sb = depb[bb] if bb >= 0 else 0
        sa = depa[ba] if ba < W else 0
        if bb != pb or ba != pa or sb != psb or sa != psa:
            o_ts.append(ts_l[i])
            o_bb.append(bb)
            o_ba.append(ba)
            o_sb.append(sb)
            o_sa.append(sa)
            pb, pa, psb, psa = bb, ba, sb, sa

    ts = np.asarray(o_ts, dtype=np.int64)
    bbi = np.asarray(o_bb, dtype=np.int64)
    bai = np.asarray(o_ba, dtype=np.int64)
    szb = np.asarray(o_sb, dtype=np.int64)
    sza = np.asarray(o_sa, dtype=np.int64)
    # 同一 ns の途中経過は捨て、その ns を処理し終えた状態だけを残す
    keep = np.ones(ts.size, bool)
    if ts.size > 1:
        keep[:-1] = ts[:-1] != ts[1:]
    ok = keep & (bbi >= 0) & (bai < W)
    return (ts[ok],
            (bbi[ok] + p_min) * PX_UNIT,
            (bai[ok] + p_min) * PX_UNIT,
            szb[ok] * SZ_LOT,
            sza[ok] * SZ_LOT,
            int(keep.sum()), int((~ok & keep).sum()),
            n_evict, v_evict * SZ_LOT, n_neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--suffix", default="", help="出力名の接尾辞(検証用)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = DATA / f"l1_{tag}"
    files = sorted(src.glob("dt=*.parquet"))
    if not files:
        sys.exit(f"{src} が空。先に fetch_l1.py を回すこと")
    if a.days:
        files = files[: a.days]
    out = DATA / f"bbo_l1_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int64, "size_after": pl.Int64})
    n_cross_all = 0
    n_row_all = 0
    n_ev_all = 0
    for k, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        fo = out / f"dt={dt}.parquet"
        t1 = time.time()
        # ★sweep には「その日の初めに残っていた板」を渡す。nxt(日末)を渡すと
        #   初期の最良が翌日の状態で始まってしまう
        head = carry
        delta, nxt, _ = day_events(fp, carry)
        carry = nxt
        if fo.exists():                        # 再開: 板は通すが書き直さない
            print(f"  [{k+1}/{len(files)}] {dt} すでに有り(板だけ通した)", flush=True)
            continue
        if delta.height == 0:
            print(f"  [{k+1}/{len(files)}] {dt} 差分なし。飛ばす", flush=True)
            continue
        ts, pb, pa, qb, qa, n_kept, n_cross, n_ev, v_ev, n_neg = sweep(delta, head)
        pl.DataFrame({"ts": ts, "best_bid": pb, "best_ask": pa,
                      "bid_sz": qb, "ask_sz": qa}).write_parquet(fo)
        n_cross_all += n_cross
        n_row_all += ts.size
        n_ev_all += n_ev
        print(f"  [{k+1}/{len(files)}] {dt} 差分 {delta.height:,} → "
              f"bbo {ts.size:,} 行 (残クロス {n_cross:,} 除外) "
              f"退去 {n_ev:,} 回 / {v_ev:,.3f} 枚 / 負丸め {n_neg:,}  "
              f"繰越 {carry.height:,} 本  {time.time()-t1:.1f}s", flush=True)

    fs = sorted(out.glob("dt=*.parquet"))
    big = pl.concat([pl.read_parquet(f).with_columns(
        dt=pl.lit(f.stem.split("=")[1])) for f in fs])
    dst = DATA / f"bbo_{tag}{a.suffix}.parquet"
    big.write_parquet(dst)
    print(f"書き出し {dst}  {big.height:,} 行 / {len(fs)} 日 "
          f"(残クロス除外 {n_cross_all:,} / 退去 {n_ev_all:,} 回)")


if __name__ == "__main__":
    main()
