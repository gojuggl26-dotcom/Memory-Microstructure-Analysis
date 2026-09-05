"""指値 1 本ごとの「置いた場所・前に居た量・寿命・結末」を作る(L4 でしか作れない層)。

    uv run python scripts/build_orderlife.py --coin xyz:INTC [--days N]

出力: data/orderlife_<coin>/dt=*.parquet(その日に**置かれた**注文 1 本 1 行)

列
--
| 列 | 中身 | いつ確定するか |
|---|---|---|
| `oid` `ts_open` `is_bid` `pidx` `orig_lots` | 発注そのもの | 発注時 |
| `dist_tick` | 発注時の**同じ側の最良**から何ティック奥か(0 = 最良) | 発注時 |
| `ahead_lots` | 同じ価格に**既に積まれていた**数量 | 発注時 |
| `spread_tick` | 発注時のスプレッド(ティック) | 発注時 |
| `ts_close` `status` `rest_lots` | 終端イベント | 終端時 |
| `filled_lots` | `orig − rest`(★`reduce_only` は約定を意味しない) | 終端時 |
| `life_ns` | `ts_close − ts_open` | 終端時 |
| `reduce_only` `tif` | 注文の属性 | 発注時 |

`dist_tick` と `ahead_lots` は **その注文を板に入れる直前**の板から測る。
自分自身を数えないためで、これを怠るとキュー位置が必ず 1 本ぶん大きく出る。

★`filled_lots` の注意
---------------------
`reduce_only` 注文は、ポジションが減ると取引所が残量を自動的に縮める。
残量が減っても約定とは限らないので、約定量として数えると過大になる
(hl-l4-pipeline で filled_sz を 22.8% 過大計上した罠。誤計上の 100% が reduce_only)。
集計側では `reduce_only` を必ず分けること。

x が確定する時刻 / y の期間: 発注時に判る列と終端時に判る列を上の表で分けてある。
発注時に判る列だけを説明変数に使えば先読みにならない。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bbo_l1 import MARGIN  # noqa: E402
from build_obi_levels import CANCELS, PX_UNIT, RESTING, SZ_LOT, TERMINAL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000


def day_stream(fp: Path, carry: pl.DataFrame, l1u: Path | None = None):
    """1 日分を「板の差分 + その行の素性」に直す。build_bbo_l1 と同じ規則。

    `l1u` を渡すと `user` 列を貼る。l1u は fetch_l1_user.py が同じ元ファイルから
    **並べ替えずに**取ったもので、あちら側で `status` の並びが l1 と行単位で
    一致することを確かめてから書いている(合わない日は書かずに止まる)。
    ここでは行数だけを念のため確認する。
    """
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status",
                                     "orig_sz", "remaining_sz", "tif",
                                     "reduce_only", "is_trigger"])
    if l1u is not None and l1u.exists():
        u = pl.read_parquet(l1u)          # 列は `user` だけ。行順は l1 と同じ
        if u.height != d.height:
            raise SystemExit(f"{l1u.name}: 行数が違う {u.height:,} vs {d.height:,}")
        d = d.with_columns(user=u["user"])
        del u
    else:
        d = d.with_columns(user=pl.lit(None, pl.String))
    t0 = (int(d["ts"].cast(pl.Int64).min()) // DAY_NS) * DAY_NS
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int64),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  orig=(pl.col("orig_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS),
                  status=pl.col("status"),
                  ro=pl.col("reduce_only"), tif=pl.col("tif"),
                  user=pl.col("user")))
    del d
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          sz=pl.col("size_after"), orig=pl.col("size_after"),
                          status=pl.lit("carry"), ro=pl.lit(False),
                          tif=pl.lit("carry"), user=pl.lit(None, pl.String),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"]).with_columns(
        prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
        last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after"))
    # 終端(その oid の最後の行で size_after == 0)
    term = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") == 0)
                      & (~pl.col("synth")))
            .select("oid", ts_close="ts", status="status", rest_lots="sz"))
    st = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
          .filter((~pl.col("synth")) & (pl.col("delta") != 0))
          .select("ts", "oid", "pidx", "is_bid", "delta", "rk", "orig", "ro",
                  "tif", "user").sort("ts"))
    return st, nxt, term, t0


def sweep(st: pl.DataFrame, carry: pl.DataFrame):
    """板を進めながら、`open` の行だけ「距離」と「前に居た量」を書き留める。"""
    pi_a = st["pidx"].to_numpy()
    cpx = carry["pidx"].to_numpy() if carry.height else pi_a[:1]
    p_min = int(min(pi_a.min(), cpx.min())) - MARGIN
    p_max = int(max(pi_a.max(), cpx.max())) + MARGIN
    W = p_max - p_min + 1
    depb, depa = [0] * W, [0] * W
    tb, ta = [0] * W, [0] * W
    if carry.height:
        cb = carry["is_bid"].to_numpy()
        cp = carry["pidx"].to_numpy() - p_min
        cs = carry["size_after"].to_numpy()
        for k in range(cp.size):
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

    ts_l = st["ts"].to_numpy().tolist()
    px_l = (pi_a - p_min).tolist()
    bd_l = st["is_bid"].to_numpy().tolist()
    dv_l = st["delta"].to_numpy().tolist()
    rk_l = st["rk"].to_numpy().tolist()
    o_i, o_dist, o_ahead, o_spr = [], [], [], []
    for i in range(len(ts_l)):
        p = px_l[i]
        t = ts_l[i]
        bid = bd_l[i]
        if rk_l[i] == 0:                       # ★入れる前の板から測る
            if bid:
                o_dist.append(bb - p if bb >= 0 else -999)
                o_ahead.append(depb[p])
            else:
                o_dist.append(p - ba if ba < W else -999)
                o_ahead.append(depa[p])
            o_spr.append(ba - bb if (bb >= 0 and ba < W) else -999)
            o_i.append(i)
        if bid:
            v = depb[p] + dv_l[i]
            if v < 0:
                v = 0
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
            depa[p] = v
            ta[p] = t
            if v > 0:
                if p < ba:
                    ba = p
            elif p == ba:
                while ba < W and not depa[ba]:
                    ba += 1
        while 0 <= ba <= bb and bb < W:        # older-loses でアンクロス
            if ta[ba] <= tb[bb]:
                depa[ba] = 0
                while ba < W and not depa[ba]:
                    ba += 1
            else:
                depb[bb] = 0
                while bb >= 0 and not depb[bb]:
                    bb -= 1
    return (np.asarray(o_i, dtype=np.int64), np.asarray(o_dist, dtype=np.int32),
            np.asarray(o_ahead, dtype=np.int64), np.asarray(o_spr, dtype=np.int32),
            p_min)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((DATA / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    out = DATA / f"orderlife_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int64, "size_after": pl.Int64})
    # 終端が翌日以降に来る注文のために、未決の発注を持ち越す
    pend = pl.DataFrame(schema={"oid": pl.Int64, "ts_open": pl.Int64,
                                "is_bid": pl.Boolean, "pidx": pl.Int64,
                                "orig_lots": pl.Int64, "dist_tick": pl.Int32,
                                "ahead_lots": pl.Int64, "spread_tick": pl.Int32,
                                "reduce_only": pl.Boolean, "tif": pl.String,
                                "user": pl.String, "dt_open": pl.String})
    for k, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        t1 = time.time()
        head = carry
        st, nxt, term, _ = day_stream(fp, carry, DATA / f"l1u_{tag}" / fp.name)
        carry = nxt
        if st.height == 0:
            continue
        idx, dist, ahead, spr, _pm = sweep(st, head)
        op = st[idx].select("oid", ts_open="ts", is_bid="is_bid", pidx="pidx",
                            orig_lots="orig", reduce_only="ro", tif="tif",
                            user="user")
        op = op.with_columns(dist_tick=pl.Series(dist), ahead_lots=pl.Series(ahead),
                             spread_tick=pl.Series(spr), dt_open=pl.lit(dt))
        pend = pl.concat([pend, op.select(pend.columns)])
        j = pend.join(term, on="oid", how="left")
        done = j.filter(pl.col("ts_close").is_not_null())
        pend = j.filter(pl.col("ts_close").is_null()).select(pend.columns)
        if done.height:
            done = done.with_columns(
                life_ns=pl.col("ts_close") - pl.col("ts_open"),
                filled_lots=pl.col("orig_lots") - pl.col("rest_lots"))
            done.write_parquet(out / f"dt={dt}.parquet")
        print(f"  [{k+1}/{len(files)}] {dt} 発注 {op.height:,} / 決着 {done.height:,} "
              f"/ 未決 {pend.height:,}  {time.time()-t1:.1f}s", flush=True)
    if pend.height:
        # 窓の終わりまで残った注文は右打ち切り。混ぜないよう別に置く
        pend.write_parquet(DATA / f"orderlife_{tag}_censored.parquet")
        print(f"右打ち切り {pend.height:,} 本 → orderlife_{tag}_censored.parquet")


if __name__ == "__main__":
    main()
