"""指値の待ち行列(キュー)を 21 種類で測る。

価格水準ごとの待ち行列を FIFO で追いかけ、11.1 キューの大きさ / 11.2 各注文の
並び位置 / 11.3 キューの動きを出す。

【11.1 キューの大きさ】その注文が並んだ瞬間の、その価格水準の状態
    total queue size        並んでいる数量の合計
    queue length            並んでいる注文の本数
    wallet count            並んでいる注文を出した口座の数(重複を除く)
    average queue order size  合計 / 本数

【11.2 並び位置】★測る時点を 2 つに分ける
FIFO なので**並んだ瞬間はどの注文も最後尾**である。したがって
「後ろに何本いるか」は定義上 0 にしかならず、百分位も常に 1 になる。
そこで 2 つの時点で測る。

    置いた瞬間   orders ahead / volume ahead / notional ahead
                 (behind は構造上 0。だから出さない)
    消えた瞬間   orders ahead / volume ahead / orders behind / volume behind /
                 relative queue position / queue percentile / normalized queue rank

    V^ahead_i = Σ_{j≺i} q_j          自分より前に並んでいる生きた注文の数量
    relative queue position = V^ahead / V^total                 自分を含む数量で見る
    queue percentile        = V^ahead / (V^ahead + V^behind)    自分を除く数量で見る
    normalized queue rank   = orders_ahead / (orders_ahead + orders_behind)  本数で見る

★この 3 つは似ているが別物である。1 つ目だけ自分の数量が分母に入るので、
  キューに自分しかいないとき 0 になる(残り 2 つは未定義)。

★消えた瞬間の「前/後ろ」は水準の生存注文を走査するので O(キュー長)かかる。
全注文でやると遅すぎるので **SAMPLE 本に 1 本**だけ厳密に測る。置いた瞬間の
3 つは running total から O(1) で出るので**全注文**で測る。

【11.3 キューの動き】最良気配のキューについて 100ms 窓ごと
    queue growth       その窓に入ってきた数量
    queue depletion    出ていった数量
    queue turnover     (入 + 出) / 窓の始めのキュー数量
    queue arrival rate 入ってきた本数 / 秒
    queue removal rate 出ていった本数 / 秒
    queue net flow     入 − 出(数量)
    queue imbalance    (入 − 出) / (入 + 出)
    queue churn        入 + 出(数量)
    queue replacement rate  min(入, 出) / 窓の始めのキュー数量
                            = 出た分がどれだけ埋め直されたか

【板の再構成】
build_obi_levels.py と同じ規則。tif ∈ {Alo, Gtc} かつ is_trigger でない注文の
open と終端だけを見る。数量は 0.001 の整数ロットで持つ(浮動小数だと
空の水準に 1e-13 が残る)。同一 ns の行順は論理順と逆なので
open(0) → filled(2) → 終端(3) で並べ直す。部分約定は remaining_sz に出る。

★日を跨いで生きている注文はキューに残したまま翌日へ持ち越す。

【x が確定する時刻】
すべて実測量であり、目的変数を持たない。並び位置は「その時点で観測できる量」
だけから作る(未来の注文は入らない)。先読みは構造的に起こり得ない。

    uv run python scripts/build_queue.py --coin xyz:MU
出力: data/queue_place_<coin>.parquet   … 置いた瞬間の集計(日 × 帯)
      data/queue_term_<coin>.parquet    … 消えた瞬間の標本(SAMPLE 本に 1 本)
      data/queue_dyn_<coin>/dt=*.parquet … 100ms 窓のキューの動き 9 種
      data/queue_meta_<coin>.csv        … 日ごとの規模と検算
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, GRID_NS, PX_UNIT,  # noqa: E402
                              RESTING, SZ_LOT, TERMINAL, clean_bbo, grid_mid)

ROOT = Path(__file__).resolve().parents[1]
NG = DAY_NS // GRID_NS
SAMPLE = 40                    # 消えた瞬間を厳密に測る間隔(何本に 1 本か)
# 置いた瞬間の集計に使う「前にいる数量」の帯(ロット)
AH_EDGES = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def day_run(fp: Path, widdf: pl.DataFrame, bb_day: pl.DataFrame, live: dict,
            meta: dict):
    """1 日分を回す。live は前日から持ち越したキュー(その場で更新する)。"""
    d = pl.read_parquet(fp)
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
                  gone=pl.col("status").is_in(CANCELS))
          # ★口座は 540 万件の Python dict にすると 500MB 近く食う。結合で持つ
          .join(widdf, on="oid", how="left")
          .with_columns(wid=pl.col("wid").fill_null(-1))
          .sort(["ts", "rk"]))
    del d
    oid = ev["oid"].to_numpy(); ts = ev["ts"].to_numpy()
    isb = ev["is_bid"].to_numpy(); pid = ev["pidx"].to_numpy()
    szv = ev["sz"].to_numpy(); rkv = ev["rk"].to_numpy()
    gonev = ev["gone"].to_numpy(); widv = ev["wid"].to_numpy()
    del ev

    # ---- 最良気配(100ms 格子。動きの集計を最良のキューに限るために使う)----
    mid, bbi, bai, good, _, _ = grid_mid(bb_day, t0, NG)

    q = live["q"]          # (is_bid, pidx) -> {oid: size}  ★挿入順 = FIFO
    tot = live["tot"]      # (is_bid, pidx) -> 数量合計
    wct = live["wct"]      # (is_bid, pidx) -> {wid: 本数}
    own = live["own"]      # oid -> (is_bid, pidx, size, seq)
    seqc = live["seq"]

    # 置いた瞬間の集計(全注文)。帯ごとの件数と和
    nb = len(AH_EDGES)
    pl_cnt = np.zeros(nb, np.int64)
    pl_sum = np.zeros((nb, 5), np.float64)  # qsize, qlen, wallets, avg_sz, notional
    # 分位を出すための対数ヒストグラム(前にいる数量)
    LH = 80
    pl_hist = np.zeros(LH, np.int64)
    term_rows = []

    # 動き(100ms 窓 × 側)。入/出 の 数量と本数
    # ★float64 の (864000, 2, 4) は 55MB、frame 化で 166MB になり、
    #   並行して 6GB 級の処理が走っている環境では OOM した(実際に踏んだ)。
    dyn = np.zeros((NG, 2, 4), np.float32)     # [win, side, (add_v, rem_v, add_n, rem_n)]
    q0 = np.full((NG, 2), np.nan, np.float32)  # 窓の始めの最良キュー数量

    n_orph = n_open = n_term = 0
    last_g = -1
    gi_all = np.clip((ts - t0) // GRID_NS, 0, NG - 1)
    for i in range(len(oid)):
        o = int(oid[i]); b = bool(isb[i]); p = int(pid[i]); k = (b, p)
        g = int(gi_all[i])
        if g != last_g:
            # ★窓の始めのキュー数量。ループの外で取ると 1 日の最後の状態になる
            if good[g]:
                q0[g, 0] = tot.get((True, int(bbi[g])), 0)
                q0[g, 1] = tot.get((False, int(bai[g])), 0)
            last_g = g
        best = bbi[g] if b else bai[g]
        at_best = bool(good[g]) and (p == int(best))
        s = 0 if b else 1
        if rkv[i] == 0:                                   # ---- 置いた ----
            sz = int(szv[i])
            if sz <= 0:
                continue
            n_open += 1
            qd = q.get(k)
            if qd is None:
                qd = q[k] = {}; tot[k] = 0; wct[k] = defaultdict(int)
            va = tot[k]; la = len(qd)                     # ★FIFO なので前 = 現在の全部
            w = int(widv[i])
            # --- 11.1 + 11.2(置いた瞬間)を全注文で集計 ---
            j = int(np.searchsorted(AH_EDGES, va, side="right") - 1)
            pl_cnt[j] += 1
            # notional ahead = 前にいる数量 × その価格
            pl_sum[j] += (va, la, len(wct[k]), (va / la if la else 0.0),
                          va * SZ_LOT * p * PX_UNIT)
            pl_hist[min(int(np.log1p(va) / np.log(10) * 10), LH - 1)] += 1
            qd[o] = sz; tot[k] = va + sz; wct[k][w] += 1
            own[o] = (b, p, sz, seqc[0]); seqc[0] += 1
            if at_best:
                dyn[g, s, 0] += sz; dyn[g, s, 2] += 1
        else:                                             # ---- 消えた / 部分約定 ----
            prev = own.get(o)
            if prev is None:
                n_orph += 1
                continue
            pb, pp, psz, pseq = prev
            kk = (pb, pp)
            qd = q.get(kk)
            rem = int(szv[i])
            if rkv[i] == 2 and not gonev[i] and rem > 0:   # 部分約定: 残量を減らす
                if qd is not None and o in qd:
                    tot[kk] += rem - qd[o]
                    qd[o] = rem
                    own[o] = (pb, pp, rem, pseq)
                    if at_best:
                        dyn[g, s, 1] += psz - rem
                continue
            n_term += 1
            if qd is not None and o in qd:
                # --- 11.2(消えた瞬間)。走査するので SAMPLE 本に 1 本だけ ---
                if pseq % SAMPLE == 0:
                    va = vb = 0; oa = ob = 0; seen = False
                    for oo, ss in qd.items():
                        if oo == o:
                            seen = True; continue
                        if seen:
                            vb += ss; ob += 1
                        else:
                            va += ss; oa += 1
                    vt = va + vb + qd[o]
                    vo = va + vb
                    term_rows.append((
                        ts[i], pb, pp, oa, ob, float(va), float(vb),
                        # relative queue position … 自分を含めた全体に対する前の割合
                        float(va) / vt if vt else np.nan,
                        # queue percentile … 自分を除いた数量に対する前の割合
                        float(va) / vo if vo else np.nan,
                        # normalized queue rank … 本数で見た前の割合
                        oa / (oa + ob) if (oa + ob) else np.nan,
                        float(qd[o]), int(gonev[i])))
                tot[kk] -= qd[o]
                if at_best:
                    dyn[g, s, 1] += qd[o]; dyn[g, s, 3] += 1
                del qd[o]
                cw = wct[kk]; ww = int(widv[i])
                cw[ww] -= 1
                if cw[ww] <= 0:
                    del cw[ww]
                if not qd:
                    del q[kk]; del tot[kk]; del wct[kk]
            del own[o]

    meta.update(n_open=n_open, n_term=n_term, n_orphan=n_orph,
                n_live_end=len(own), n_levels_end=len(q),
                n_term_sampled=len(term_rows))
    return (pl_cnt, pl_sum, pl_hist, term_rows, dyn, q0, good)


def dyn_frame(dyn, q0, good) -> pl.DataFrame:
    """100ms 窓 × 側 の 9 種にまとめる。

    ★動きのあった窓だけ書く。全 864,000 窓 × 2 側を出すと 1 日 1.7M 行になり、
    98 日で 1GB を超えるうえ frame 化で 166MB のピークが立って OOM する。
    実測で動いた窓は 4.4% しかないので、残りは書かない(全窓数は
    n_good_day 列に入れておき、割合はそこから復元できる)。
    """
    out = []
    ng_good = int(good.sum())
    for si, nm in ((0, "bid"), (1, "ask")):
        av, rv, an, rn = (dyn[:, si, j].astype(np.float64) for j in range(4))
        base = q0[:, si].astype(np.float64)
        m = np.flatnonzero(good & ((av + rv) > 0))
        if not len(m):
            continue
        av, rv, an, rn, base = av[m], rv[m], an[m], rn[m], base[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            turn = np.where(base > 0, (av + rv) / base, np.nan)
            repl = np.where(base > 0, np.minimum(av, rv) / base, np.nan)
            imb = np.where(av + rv > 0, (av - rv) / (av + rv), np.nan)
        out.append(pl.DataFrame({
            "win": m.astype(np.int32),
            "side": np.full(len(m), nm),
            "queue_growth": av * SZ_LOT,
            "queue_depletion": rv * SZ_LOT,
            "queue_turnover": turn,
            "queue_arrival_rate": an / (GRID_NS / 1e9),
            "queue_removal_rate": rn / (GRID_NS / 1e9),
            "queue_net_flow": (av - rv) * SZ_LOT,
            "queue_imbalance": imb,
            "queue_churn": (av + rv) * SZ_LOT,
            "queue_replacement_rate": repl,
            "q_start": base * SZ_LOT,
            "n_good_day": np.full(len(m), ng_good, np.int32),
        }))
    return (pl.concat(out) if out else
            pl.DataFrame(schema={"win": pl.Int32, "side": pl.String}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    bb, ndrop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 除いた行 {ndrop:,} -> {bb.height:,}", file=sys.stderr)
    have = set(bb["dt"].unique().to_list())
    days = sorted(p.name.split("=")[1].removesuffix(".parquet")
                  for p in (ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        days = days[: a.days]

    dyndir = ROOT / "data" / f"queue_dyn_{tag}"
    dyndir.mkdir(parents=True, exist_ok=True)
    live = {"q": {}, "tot": {}, "wct": {}, "own": {}, "seq": [0]}
    P_cnt = np.zeros(len(AH_EDGES), np.int64)
    P_sum = np.zeros((len(AH_EDGES), 5), np.float64)
    P_hist = np.zeros(80, np.int64)
    T_rows, metas = [], []
    for i, day in enumerate(days, 1):
        if day not in have:
            print(f"  {day} bbo 無し。飛ばす", file=sys.stderr)
            continue
        t = time.time()
        wf = ROOT / "data" / f"l1user_{tag}" / f"dt={day}.parquet"
        widdf = (pl.read_parquet(wf) if wf.exists()
                 else pl.DataFrame(schema={"oid": pl.Int64, "wid": pl.Int32}))
        m = {"dt": day, "n_wid": widdf.height}
        pc, ps, ph, tr, dyn, q0, good = day_run(
            ROOT / "data" / f"l1_{tag}" / f"dt={day}.parquet", widdf,
            bb.filter(pl.col("dt") == day), live, m)
        P_cnt += pc; P_sum += ps; P_hist += ph
        T_rows.extend((day, *r) for r in tr)
        dyn_frame(dyn, q0, good).write_parquet(dyndir / f"dt={day}.parquet")
        m["sec"] = round(time.time() - t, 1)
        metas.append(m)
        print(f"  {i}/{len(days)} {day} 置 {m['n_open']:,} 消 {m['n_term']:,} "
              f"標本 {m['n_term_sampled']:,} 生存 {m['n_live_end']:,} {m['sec']}s",
              file=sys.stderr, flush=True)

    pl.DataFrame({"band_lo": AH_EDGES, "n": P_cnt,
                  "sum_qsize": P_sum[:, 0], "sum_qlen": P_sum[:, 1],
                  "sum_wallets": P_sum[:, 2], "sum_avgsz": P_sum[:, 3],
                  "sum_notional": P_sum[:, 4],
                  "hist": [P_hist.tolist()] * len(AH_EDGES)}
                 ).write_parquet(ROOT / "data" / f"queue_place_{tag}.parquet")
    pl.DataFrame(T_rows, schema=["dt", "ts", "is_bid", "pidx", "orders_ahead",
                                 "orders_behind", "volume_ahead", "volume_behind",
                                 "relative_queue_position", "queue_percentile",
                                 "normalized_queue_rank", "own_size", "canceled"],
                 orient="row").write_parquet(ROOT / "data" / f"queue_term_{tag}.parquet")
    pl.DataFrame(metas).write_csv(ROOT / "data" / f"queue_meta_{tag}.csv")
    print(f"-> data/queue_place_{tag}.parquet / queue_term_{tag}.parquet / "
          f"{dyndir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
