"""ティック水準別・キュー位置別の約定率を出し、取引高で変わるかを検証する。

【何を測るか】
指値を board に置いたとき、それが**最終的に約定する確率**を

    ティック水準 … 置いた価格が同じ側の最良気配から何ティック離れているか
    キュー位置   … 置いた瞬間、同じ価格に既に何本(何枚)並んでいたか

の 2 つで層別する。どちらも**発注した瞬間に確定する量**なので、
「置く前に判る条件」で約定確率を語れる。

【板の再構成】
l1 の注文イベント(open / 各種 cancel / filled)を時刻順に処理し、

    live[oid]           … 生きている注文(側・価格・残量)
    depth[(側, 価格)]   … その価格に並んでいる合計数量と本数

を保つ。最良気配は遅延削除つきヒープで持つ(価格帯が数千あるので毎回
最大最小を取り直すと間に合わない)。open の瞬間に

    tick_level  = (同じ側の最良気配 − 自分の価格) / 0.01     ※ ask は符号を反転
    queue_ahead = その価格に既にある数量と本数

を記録する。tick_level が負なら**気配を改善した**(スプレッドの内側に置いた)。

【★板に入れてはいけない注文(実際に踏んだ)】
- **トリガー注文**(`is_trigger`)は発火するまで板に載らない。これを入れると
  買い $991 と売り $96 が同時に生存し、板が 99.95% の時間クロスした。除外する。
- **テイカー**(`tif` が Ioc / FrontendMarket / LiquidationMarket)は板に留まらない。
  「置いた指値が約定するか」を測る母集団ではないので除外し、`Alo` と `Gtc` だけを見る。

【★約定の判定】
`filled` は終端で、1 つの oid に 2 回現れることはない(実データで確認済み)。
ただし部分約定があるので、約定枚数は終端イベントの `orig_sz - remaining_sz` で取る。
**reduce_only の注文は除外する。** 取引所がポジション減少に合わせて注文を自動縮小
するため、残量が減っても約定とは限らないため(hl-l4-pipeline で filled_sz を
22.8% 過大計上した既知の罠)。

【x が確定する時刻 / y の期間】
    x = ティック水準・キュー位置 … 発注の瞬間に確定
    y = その注文が最終的に約定したか … 発注より後
先読みは無い。

【取引高との関係】
日ごとに約定率と出来高を出し、出来高の三分位で層別する。
★出来高が増えれば約定が増えるのは半ば自明なので、
「水準ごとの約定率の**形**が変わるか」と「比例以上か以下か」を見る。

    uv run python scripts/build_fill_rate.py --coin xyz:MU
出力: data/fill_rate_cells_<coin>.parquet … 日 × ティック水準 × キュー位置の計数
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.01
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}

# 水準の帯(ティック)。負 = 気配を改善して置いた
LV_EDGES = [0, 1, 2, 3, 5, 10, 20, 50]
LV_LAB = ["改善(<0)", "0(最良)", "1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
# キューの帯(自分より前に並んでいる本数)
Q_EDGES = [1, 2, 3, 5, 10, 20]
Q_LAB = ["0(先頭)", "1", "2", "3-4", "5-9", "10-19", "20+"]


def lv_bucket(t: int) -> int:
    if t < 0:
        return 0
    return int(np.searchsorted(LV_EDGES, t, side="right"))


def q_bucket(n: int) -> int:
    return int(np.searchsorted(Q_EDGES, n, side="right"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0, help="先頭 N 日だけ(0 = 全日)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    srcdir = ROOT / "data" / f"l1_{tag}"
    files = sorted(srcdir.glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"{srcdir} が空。先に fetch_l1.py を実行すること")
    print(f"[load] {len(files)} 日分", file=sys.stderr)

    live: dict[int, tuple] = {}                 # oid -> (isbid, px, lv, qn, dt)
    dep_s: dict[tuple, float] = {}              # (isbid, px) -> 合計数量
    dep_n: dict[tuple, int] = {}                # (isbid, px) -> 本数
    # 最良気配は遅延削除つきヒープで持つ。★同じ価格を二重に積まないよう集合で管理する。
    #   これを怠ると、価格水準が作られ直すたびに積み上がって数千万要素になり
    #   メモリを使い切る(実際に落ちた)。集合で押さえるとヒープは活きている
    #   価格水準の数(千程度)で頭打ちになる。
    hb: list[int] = []                          # bid 価格の最大ヒープ(負で持つ)
    ha: list[int] = []                          # ask 価格の最小ヒープ
    in_hb: set[int] = set()
    in_ha: set[int] = set()
    cells: dict[tuple, list] = {}               # (dt, lv, qb) -> [n, n_fill, sz, fill_sz]
    n_orph = 0
    n_ro = 0

    def best_bid():
        while hb:
            p = -hb[0]
            if dep_n.get((True, p), 0) > 0:
                return p
            heapq.heappop(hb); in_hb.discard(p)
        return None

    def best_ask():
        while ha:
            p = ha[0]
            if dep_n.get((False, p), 0) > 0:
                return p
            heapq.heappop(ha); in_ha.discard(p)
        return None

    RESTING = {"Alo", "Gtc"}
    n_cross = n_chk = 0
    for fp in files:
        d = pl.read_parquet(fp).sort("ts")
        oid = d["oid"].to_numpy()
        isbid = (d["side"].to_numpy() == "B")
        px = np.rint(d["px"].to_numpy() / TICK).astype(np.int64)   # 整数ティックで持つ
        st = d["status"].to_numpy()
        osz = d["orig_sz"].to_numpy()
        rsz = d["remaining_sz"].to_numpy()
        ro = d["reduce_only"].to_numpy()
        trg = d["is_trigger"].to_numpy()
        tif = d["tif"].to_numpy()
        dd = fp.stem.split("=")[1]

        for i in range(len(st)):
            s = st[i]
            if s == "open":
                # トリガー注文は発火まで板に載らない。テイカーは板に留まらない。
                if trg[i] or tif[i] not in RESTING:
                    continue
                b, p = bool(isbid[i]), int(px[i])
                key = (b, p)
                qn = dep_n.get(key, 0)
                bb, ba = best_bid(), best_ask()
                if bb is not None and ba is not None:
                    n_chk += 1
                    if bb >= ba:
                        n_cross += 1
                ref = bb if b else ba
                # 片側しか無い / 板が空なら水準を決められないので記録しない
                lv = None if ref is None else ((ref - p) if b else (p - ref))
                live[int(oid[i])] = (b, p, lv, qn, dd)
                if qn == 0:
                    if b:
                        if p not in in_hb:
                            heapq.heappush(hb, -p); in_hb.add(p)
                    elif p not in in_ha:
                        heapq.heappush(ha, p); in_ha.add(p)
                dep_n[key] = qn + 1
                dep_s[key] = dep_s.get(key, 0.0) + float(osz[i])
            elif s == "filled" or s in CANCEL:
                o = live.pop(int(oid[i]), None)
                if o is None:
                    n_orph += 1
                    continue
                b, p, lv, qn, d0 = o
                key = (b, p)
                dep_n[key] = dep_n.get(key, 1) - 1
                dep_s[key] = max(dep_s.get(key, 0.0) - float(osz[i]), 0.0)
                if dep_n[key] <= 0:
                    dep_n.pop(key, None); dep_s.pop(key, None)
                if lv is None:
                    continue
                if ro[i]:                   # 自動縮小が混じるので約定判定に使えない
                    n_ro += 1
                    continue
                fsz = float(osz[i]) - float(rsz[i])
                filled = 1 if (s == "filled" or fsz > 0) else 0
                k = (d0, lv_bucket(lv), q_bucket(qn))
                c = cells.get(k)
                if c is None:
                    cells[k] = [1, filled, float(osz[i]), max(fsz, 0.0)]
                else:
                    c[0] += 1; c[1] += filled; c[2] += float(osz[i]); c[3] += max(fsz, 0.0)
        print(f"  {dd} live {len(live):,} 水準 {len(dep_n):,}", file=sys.stderr)
    print(f"[検算] open 時に両側そろっていた {n_chk:,} 中 クロス {n_cross:,} "
          f"({n_cross/max(n_chk,1):.3%})", file=sys.stderr)

    rows = [{"dt": k[0], "lv": k[1], "qb": k[2], "n": v[0], "n_fill": v[1],
             "sz": v[2], "fill_sz": v[3]} for k, v in cells.items()]
    C = pl.DataFrame(rows)
    C = C.with_columns(lv_lab=pl.col("lv").map_elements(lambda i: LV_LAB[i], return_dtype=pl.String),
                       q_lab=pl.col("qb").map_elements(lambda i: Q_LAB[i], return_dtype=pl.String))
    C.write_parquet(ROOT / "data" / f"fill_rate_cells_{tag}.parquet")
    tot = C["n"].sum()
    print(f"\n[集計] 注文 {tot:,} / 孤児 {n_orph:,} / reduce_only 除外 {n_ro:,}",
          file=sys.stderr)
    print(f"  全体の約定率 {C['n_fill'].sum()/tot:.3%}", file=sys.stderr)

    print("\n=== ティック水準別 ===", file=sys.stderr)
    g = C.group_by("lv").agg(n=pl.col("n").sum(), f=pl.col("n_fill").sum()).sort("lv")
    for r in g.iter_rows(named=True):
        print(f"  {LV_LAB[r['lv']]:<10}{r['n']:>12,}  約定率 {r['f']/r['n']:>7.3%}", file=sys.stderr)
    print("\n=== キュー位置別(自分より前の本数)===", file=sys.stderr)
    g = C.group_by("qb").agg(n=pl.col("n").sum(), f=pl.col("n_fill").sum()).sort("qb")
    for r in g.iter_rows(named=True):
        print(f"  {Q_LAB[r['qb']]:<10}{r['n']:>12,}  約定率 {r['f']/r['n']:>7.3%}", file=sys.stderr)
    print(f"\n-> data/fill_rate_cells_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
