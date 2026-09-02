"""口座ごとの約定のされ方を 11 指標で測る。

【何を測るか】
指値を置いた口座から見て「置いた注文がどれだけ約定に至ったか」「約定したとき
何が起きていたか」を 11 通りに測る。母集団は板に留まる指値
(`Alo` / `Gtc`、トリガーとテイカーと reduce_only を除く)。

  出し方に対する約定の割合
    fill rate                  約定した注文の**本数** / 置いた本数
    fill probability           **数量で重みづけた**約定確率
                               Σ(orig_sz × 1[約定あり]) / Σ orig_sz
    fill-to-order ratio        **全量約定**した本数 / 置いた本数
    fill-to-cancel ratio       約定した本数 / 約定せず取り消した本数
    volume executed/quoted     約定した数量 / 置いた数量
    maker execution share      その口座のメイカー約定数量 / 市場全体

  約定したときの中身
    average time to fill       置いてから最初に約定するまでの時間
    partial-fill frequency     約定したうち**全量に至らなかった**本数の割合
    fill size                  約定 1 件あたりの数量
    queue-position-at-fill     約定した瞬間、自分より前に並んでいた数量
    adverse selection after fill  約定後 Δ 秒の mid 変化を**メイカーに不利な向きを正**
                               として符号づけたもの(bp)

★ **fill rate / fill probability / volume executed÷quoted / fill-to-order ratio は
  互いに近い量である。** 別々の 4 つの発見があるかのように並べると誤読されるので、
  分子と分母を上の表で明示し、実測した相関をレポートに出す。違いは次のとおり。

    fill rate                本数。1 枚でも約定すれば「約定した」と数える
    fill probability         数量で重みづけた同じ確率。大口の寄与が大きい
    volume executed/quoted   実際に約定した枚数だけを数える(部分約定は部分だけ)
    fill-to-order ratio      全量約定だけを「約定」と数える(最も厳しい)

  4 つの比は「約定の完全さ」を測っている。たとえば fill probability ÷
  volume executed÷quoted は「約定した注文が平均してどれだけ埋まったか」の逆数になる。

【x が確定する時刻 / y の期間】
本レポートは予測ではなく**記述**である。ただし adverse selection だけは
約定時刻 t の後 [t, t+Δ) を見るので、時刻の扱いを明示する。

    queue-position-at-place  置いた瞬間(その時点で板にある自分より前の数量)
    queue-position-at-fill   約定した瞬間
    adverse selection        約定時刻 t の mid と t+Δ の mid の差。
                             mid は**その時刻以前の最後の bbo**(backward asof)

【★踏んだ落とし穴】
- **★`status == "filled"` だけを数えると約定量を 3 割取りこぼす。**
  `filled` イベントを持たないまま約定している注文があり、その分は
  **取消イベントの残量**にしか現れない。よって約定量は「イベントごとに
  報告された残量の減少分」として数える(open の当初数量 → 各事象の残量)。
  2026-05-04 の実測: `filled` だけだと 48,903 契約(node_fills の 56.7%)、
  取消イベントの残量も使うと 81,617 契約(**94.7%**)になる。
- **reduce_only は母集団から外す。** 取引所がポジション減少に合わせて数量を
  自動で縮めるため、残量が減っても約定とは限らない。上の数え方と組み合わせると
  影響が大きく、同じ日で 157,663 契約(**node_fills の 183%**)まで膨らむ。
  除外が正しいことはこの過大計上で裏づけられる。
- **ティック幅は 0.01 固定ではない**(1000 以上は 0.1)。本 script は価格を
  刻み幅で割らないので影響を受けないが、価格帯の指標を足すときは注意する。
- **node_fills の時刻はミリ秒精度で、板イベント(ns)より最大 ~1ms 早い。**
  ns で結合してはいけない。node_fills は**合計値の照合**にだけ使う。
- FIFO なので**置いた瞬間は必ず最後尾**である。「前にいる数量」は
  その価格水準の現在の合計そのもので、「後ろ」は定義上 0 になる。

【出力】
    data/wfill_days_<coin>/dt=*.parquet       日 × 口座の集計(再開の単位)
    data/wfill_days_<coin>/fill=*.parquet     約定 1 件ごとの記録
    data/wfill_wallet_<coin>.parquet          口座ごとの集計(全期間)
    data/wfill_fills_<coin>.parquet           約定 1 件ごとの記録(全期間)
    data/wfill_meta_<coin>.csv                日ごとの検算

実行例:
    uv run python scripts/build_wallet_fill.py --coin xyz:MU --start 0 --n 49
    uv run python scripts/build_wallet_fill.py --coin xyz:MU --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, PX_UNIT, RESTING,  # noqa: E402
                              SZ_LOT, TERMINAL, clean_bbo)

ROOT = Path(__file__).resolve().parents[1]
ADV_H = (1, 10, 60)              # adverse selection を測る先(秒)
WARMUP_DAYS = 5                  # 分割実行で板を温める日数
QMAX_SCAN = 4096                 # キューを遡る上限(これを超えたら打ち切り記録)

# 口座ごとの集計に使う列(日 × 口座で 1 行)
ACC = ("n_ord", "n_fill", "n_full", "n_part", "n_cxl",
       "v_ord", "v_fill", "n_fillev", "sum_qa_place", "sum_ttf_ns",
       # ★1 枚でも約定した注文の「当初数量」の和。fill probability を
       #   数量で重みづけて出すのに要る(本数ベースの fill rate と分けるため)
       "v_ord_filled")


def load_day(fp: Path, widdf: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """1 日分の L1 を、キュー再生に必要な形へ整える。"""
    lf = pl.scan_parquet(fp)
    n_raw = int(lf.select(pl.len()).collect().item())
    ev = (lf.filter((~pl.col("is_trigger")) & (~pl.col("reduce_only"))
                    & pl.col("tif").is_in(RESTING)
                    & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int64),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  # 0=置いた 1=部分約定 2=全量約定 3=取消
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when((pl.col("status") == "filled")
                           & (pl.col("remaining_sz") > 0)).then(1)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8))
          .collect()
          .join(widdf, on="oid", how="left")
          .with_columns(pl.col("wid").fill_null(-1))
          .sort(["ts", "rk"]))
    return ev, n_raw


def day_run(ev: pl.DataFrame, bb_day: pl.DataFrame, live: dict, nw: int):
    """1 日を回して、口座別の集計と約定 1 件ごとの記録を返す。"""
    oid = ev["oid"].to_numpy()
    ts = ev["ts"].to_numpy()
    isb = ev["is_bid"].to_numpy()
    pid = ev["pidx"].to_numpy()
    szv = ev["sz"].to_numpy()
    rkv = ev["rk"].to_numpy()
    widv = ev["wid"].to_numpy()

    q = live["q"]        # (is_bid, pidx) -> {oid: size}  ★挿入順 = FIFO
    tot = live["tot"]    # (is_bid, pidx) -> 数量合計
    own = live["own"]    # oid -> [is_bid, pidx, size, open_ts, wid, orig, filled]

    acc = np.zeros((nw + 1, len(ACC)), np.float64)   # 末尾 = wid 不明(-1)
    F = {k: [] for k in ("ts", "wid", "is_bid", "pidx", "fill_sz", "qa_fill",
                         "qa_place", "ttf_ns", "is_first", "is_full", "orig")}
    n_orph = n_trunc = 0

    for i in range(len(oid)):
        o = int(oid[i]); b = bool(isb[i]); p = int(pid[i]); k = (b, p)
        r = int(rkv[i]); w = int(widv[i]); wi = w if w >= 0 else nw

        if r == 0:                                   # ---------- 置いた ----------
            sz = int(szv[i])
            if sz <= 0:
                continue
            qd = q.get(k)
            if qd is None:
                qd = q[k] = {}
                tot[k] = 0
            va = tot[k]                              # ★FIFO: 現在の全部が前にいる
            qd[o] = sz
            tot[k] = va + sz
            own[o] = [b, p, sz, int(ts[i]), w, sz, 0, va]
            acc[wi, 0] += 1                          # n_ord
            acc[wi, 5] += sz                         # v_ord
            acc[wi, 8] += va                         # sum_qa_place
            continue

        st = own.pop(o, None) if r >= 2 else own.get(o)
        if st is None:                               # 前の窓から続く注文(孤児)
            n_orph += 1
            continue
        b0, p0, cur, open_ts, w0, orig, filled, qa_place = st
        k0 = (b0, p0)
        qd = q.get(k0)
        wi0 = w0 if w0 >= 0 else nw
        # ★報告された残量から約定量を出す。取消イベントの残量も使う。
        #   `filled` イベントを持たずに約定している注文があり、そこを落とすと
        #   約定量を 3 割ほど取りこぼす(§検証)。
        rep = int(szv[i])
        dec = cur - rep if cur > rep else 0           # この事象で約定した数量
        leave = dec if r == 1 else cur                # 板から消える数量

        if r in (1, 2, 3):                           # ------ 約定(部分/全量) ------
            if dec > 0:
                # 約定した瞬間の「前にいる数量」= 自分より先に入った生存注文の和
                qa = 0
                if qd is not None:
                    for j, (oo, ss) in enumerate(qd.items()):
                        if oo == o:
                            break
                        qa += ss
                        if j >= QMAX_SCAN:
                            qa = -1
                            n_trunc += 1
                            break
                else:
                    qa = -1
                F["ts"].append(int(ts[i])); F["wid"].append(w0)
                F["is_bid"].append(b0); F["pidx"].append(p0)
                F["fill_sz"].append(dec); F["qa_fill"].append(qa)
                F["qa_place"].append(qa_place)
                F["ttf_ns"].append(int(ts[i]) - open_ts)
                F["is_first"].append(filled == 0)
                F["is_full"].append(r == 2)
                F["orig"].append(orig)
                acc[wi0, 6] += dec                   # v_fill
                acc[wi0, 7] += 1                     # n_fillev
                if filled == 0:                      # 最初の約定でだけ数える
                    acc[wi0, 1] += 1                 # n_fill
                    acc[wi0, 9] += int(ts[i]) - open_ts   # sum_ttf_ns
                    acc[wi0, 10] += orig             # v_ord_filled
        if r == 2:                                   # 全量約定で終わった
            acc[wi0, 2] += 1                         # n_full
        elif r == 3:                                 # 取消で終わった
            if filled + dec > 0:
                acc[wi0, 3] += 1                     # n_part(約定を含む取消)
            else:
                acc[wi0, 4] += 1                     # n_cxl(1 枚も約定せず取消)

        # --- キューの更新 ---
        if qd is not None:
            if r == 1:
                qd[o] = rep
            else:
                qd.pop(o, None)
            tot[k0] = tot.get(k0, 0) - leave
            if not qd:
                q.pop(k0, None)
                tot.pop(k0, None)
        if r == 1:
            st[2] = rep
            st[6] = filled + dec

    # ---- adverse selection: 約定時刻の mid と Δ 秒後の mid(backward asof)----
    out = pl.DataFrame({k: v for k, v in F.items()}) if F["ts"] else None
    if out is not None and bb_day.height:
        bts = bb_day["ts"].cast(pl.Int64).to_numpy()
        bmid = ((bb_day["best_bid"].to_numpy() + bb_day["best_ask"].to_numpy())
                / 2.0)
        ft = out["ts"].to_numpy()

        def mid_at(t):
            j = np.searchsorted(bts, t, side="left") - 1   # ★厳密に t 以前
            m = np.where(j >= 0, bmid[np.clip(j, 0, len(bmid) - 1)], np.nan)
            return np.where(j >= 0, m, np.nan)

        m0 = mid_at(ft)
        cols = {"mid": m0}
        sgn = np.where(out["is_bid"].to_numpy(), -1.0, 1.0)   # 買い方は下落が不利
        for h in ADV_H:
            mh = mid_at(ft + h * 1_000_000_000)
            cols[f"adv{h}"] = sgn * (mh - m0) / m0 * 1e4
        out = out.with_columns([pl.Series(k, v) for k, v in cols.items()])

    meta = {"n_orph": n_orph, "n_trunc": n_trunc,
            "n_fillev": 0 if out is None else out.height,
            "n_live_end": len(own)}
    return acc, out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=WARMUP_DAYS)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    daydir = ROOT / "data" / f"wfill_days_{tag}"

    wal = pl.read_parquet(ROOT / "data" / f"l1user_{tag}" / "_wallets.parquet")
    nw = int(wal["wid"].max()) + 1

    if a.merge:
        af = sorted(daydir.glob("dt=*.parquet"))
        ff = sorted(daydir.glob("fill=*.parquet"))
        if not af:
            sys.exit(f"{daydir} に日別ファイルがない")
        acc = (pl.concat([pl.read_parquet(f) for f in af])
               .group_by("wid").agg([pl.col(c).sum() for c in ACC])
               .sort("wid"))
        acc = acc.join(wal, on="wid", how="left")
        acc.write_parquet(ROOT / "data" / f"wfill_wallet_{tag}.parquet",
                          compression="zstd")
        fills = pl.concat([pl.read_parquet(f) for f in ff])
        fills.write_parquet(ROOT / "data" / f"wfill_fills_{tag}.parquet",
                            compression="zstd")
        ms = sorted(daydir.glob("meta=*.json"))
        pl.DataFrame([json.loads(m.read_text(encoding="utf-8")) for m in ms]) \
          .write_csv(ROOT / "data" / f"wfill_meta_{tag}.csv")
        print(f"[merge] 口座 {acc.height:,} / 約定 {fills.height:,} 件",
              file=sys.stderr)
        return

    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    have = set(pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
               .select("dt").unique().collect()["dt"].to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]
    lo = max(0, a.start - a.warmup)
    hi = len(files) if not a.n else min(len(files), a.start + a.n)
    files = files[lo:hi]
    my_days = [f.stem.split("=")[1] for f in files]
    bb_all, n_drop = clean_bbo(
        pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
        .filter(pl.col("dt").is_in(my_days)).collect())
    daydir.mkdir(parents=True, exist_ok=True)
    print(f"[load] 処理 {len(files)} 日(温め {a.start - lo} 日)/ 口座 {nw:,} / "
          f"bbo 除外 {n_drop:,} 行", file=sys.stderr)

    live = {"q": {}, "tot": defaultdict(int), "own": {}}
    for kday, fp in enumerate(files, start=lo):
        dt = fp.stem.split("=")[1]
        widdf = pl.read_parquet(ROOT / "data" / f"l1user_{tag}" / fp.name)
        ev, n_raw = load_day(fp, widdf)
        if not ev.height:
            continue
        bb_day = bb_all.filter(pl.col("dt") == dt).sort("ts")
        acc, fills, meta = day_run(ev, bb_day, live, nw)
        if kday < a.start:
            print(f"  {dt} 温め(書き出さない)", file=sys.stderr)
            continue
        nz = np.flatnonzero(acc.any(axis=1))
        pl.DataFrame({"wid": np.where(nz == nw, -1, nz).astype(np.int64),
                      **{c: acc[nz, j] for j, c in enumerate(ACC)}}) \
          .write_parquet(daydir / f"dt={dt}.parquet", compression="zstd")
        if fills is not None:
            fills.with_columns(pl.lit(dt).alias("dt")).write_parquet(
                daydir / f"fill={dt}.parquet", compression="zstd")
        meta.update(dt=dt, n_raw=n_raw, n_ev=ev.height,
                    n_wallet=int(len(nz)), warmup=a.start - lo)
        (daydir / f"meta={dt}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"  {dt} 事象 {ev.height:,} 約定 {meta['n_fillev']:,} "
              f"口座 {len(nz):,} 孤児 {meta['n_orph']:,} "
              f"生存 {meta['n_live_end']:,}", file=sys.stderr)
    print(f"[done] -> {daydir}", file=sys.stderr)


if __name__ == "__main__":
    main()
