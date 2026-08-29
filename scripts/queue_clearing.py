"""自分の前の行列が「約定で消えたか / 取消で消えたか」と逆選択・期待損益の関係。

【定義】
  自分の注文 O が時刻 t0 に発注され、t_f に約定したとする。
  t0 の時点で O より前に並んでいた(同じ側・同じ価格・先に到着し、まだ生きている)
  注文の集合を A とする。t0 から t_f の間に A から消えた数量を 2 つに分ける。

    exec_vol   = Σ_{a∈A} (a が t0 < ts ≤ t_f に約定した数量)
    cancel_vol = Σ_{a∈A, a が t_f までに消滅} (t0 時点の残量 − 上の約定分)

    clear_ratio = (exec_vol − cancel_vol) / (exec_vol + cancel_vol)   ∈ [−1, +1]

      +1 … 前の行列は**全部が約定して**消えた(大口が板を食い破った)
      −1 … 前の行列は**全部が取消で**消えた(メイカーが引いた)

【仮説】
  clear_ratio ≈ +1 は「自分は大口の掃きの一部として約定した」を意味するので、
  強く逆選択されるはずである。clear_ratio ≈ −1 は前が自発的に引いただけなので
  相対的に無害なはず。
  → **clear_ratio と逆選択の間に負の関係**が出れば、
    queue_position_report.md の非単調性の機構を説明できる。

【整合性の検査】
  O が約定するには前がほぼ全部消えている必要があるので、
  exec_vol + cancel_vol ≈ q_ahead_open(queue_qi の独立計算)になるはず。
  この一致を毎日測って記録する。ずれが大きければ実装が間違っている。

【時間契約】
  clear_ratio は t0 から t_f までの情報で決まるので、**発注時点では判らない**。
  したがってこれは**予測に使える特徴量ではなく、機構を説明するための事後分解**である。
  逆選択・損益は約定時刻から前向きに測る(結果)。混同しないこと。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
OUT = D / "queue_clearing"
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
HORIZONS = {"100ms": 100_000_000, "1s": 1_000_000_000,
            "10s": 10_000_000_000, "60s": 60_000_000_000}
# clear_ratio の区分(−1 = 全部取消、+1 = 全部約定)
CB = [(-1.01, -0.8, "-1.0〜-0.8 (ほぼ全部取消)"), (-0.8, -0.4, "-0.8〜-0.4"),
      (-0.4, 0.0, "-0.4〜0.0"), (0.0, 0.4, "0.0〜0.4"),
      (0.4, 0.8, "0.4〜0.8"), (0.8, 1.01, "0.8〜1.0 (ほぼ全部約定)")]


def run_day(dt: str) -> dict | None:
    f = OUT / f"{dt}.json"
    fn = OUT / f"{dt}.npy"
    if f.exists() and fn.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    lf = LIFE / f"dt={dt}" / "part-000.parquet"
    if not lf.exists():
        return None
    life = (pl.read_parquet(lf, columns=["oid", "side", "px", "orig_sz", "ts_open",
                                         "ts_close", "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null()
                    & pl.col("ts_close").is_not_null()))
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fl = pl.concat([pl.read_parquet(x, columns=["oid", "ts", "px", "sz", "crossed"])
                    for x in fs], how="diagonal_relaxed").with_columns(
        pl.col("ts").cast(pl.Int64).alias("t")).sort(["oid", "t"])
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "mid", "best_bid", "best_ask", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    mts = mp["ts"].to_numpy(); mid = mp["mid"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    qi = pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                         columns=["oid", "q_ahead_open", "open_unknown"]).filter(
        (~pl.col("open_unknown")) & pl.col("q_ahead_open").is_not_null())

    # ---- 注文 oid ごとの約定履歴(時刻昇順 + 累積数量)-------------------
    f_oid = fl["oid"].to_numpy(); f_t = fl["t"].to_numpy(); f_sz = fl["sz"].to_numpy()
    bnd = np.flatnonzero(np.diff(f_oid, prepend=f_oid[0] - 1, append=f_oid[-1] + 1))
    fill_idx = {}
    for k in range(len(bnd) - 1):
        s, e = bnd[k], bnd[k + 1]
        fill_idx[int(f_oid[s])] = (f_t[s:e], np.concatenate([[0.0], np.cumsum(f_sz[s:e])]))

    def executed(oid: int, a: int, b: int) -> float:
        """oid が (a, b] の間に約定した数量。"""
        v = fill_idx.get(oid)
        if v is None:
            return 0.0
        t_, cs = v
        return float(cs[np.searchsorted(t_, b, side="right")]
                     - cs[np.searchsorted(t_, a, side="right")])

    # ---- 対象 = メイカーとして約定した注文(最初の約定時刻)---------------
    mk = (fl.filter(~pl.col("crossed")).group_by("oid")
            .agg(pl.col("t").min().alias("t_fill"), pl.col("px").first().alias("fill_px")))
    tgt = life.join(mk, on="oid", how="inner").join(qi, on="oid", how="inner")
    if tgt.height < 500:
        return None
    tset = {int(o): (int(tf), float(qa))
            for o, tf, qa in zip(tgt["oid"], tgt["t_fill"], tgt["q_ahead_open"])}

    # ---- 価格レベルごとの生存集合を時間順に掃く --------------------------
    o_arr = life["ts_open"].to_numpy(); c_arr = life["ts_close"].to_numpy()
    oid_arr = life["oid"].to_numpy(); sz_arr = life["orig_sz"].to_numpy()
    lvl = (np.round(life["px"].to_numpy() * 1e6).astype(np.int64) * 2
           + (life["side"].to_numpy() == "B"))
    ev_t = np.concatenate([o_arr, c_arr])
    ev_k = np.concatenate([np.zeros(len(o_arr), np.int8), np.ones(len(c_arr), np.int8)])
    ev_i = np.concatenate([np.arange(len(o_arr)), np.arange(len(c_arr))])
    # ★同時刻は「到着を先に」処理する(build_queue_qi と同じ規約)。
    #   この市場は ts_open == ts_close(同一 ns で発注即取消)の注文が大量にあり、
    #   消滅を先に処理すると pop が空振りして生存集合に残り続ける。
    #   実際それで前の行列が 8 倍に膨らみ、整合性検査が 46.7 倍で落ちた。
    order = np.lexsort((ev_k, ev_t))
    ev_t, ev_k, ev_i = ev_t[order], ev_k[order], ev_i[order]

    # numpy スカラ添字は遅いので、掃く前に Python リストへ落とす(3 倍ほど速い)
    lvl_l = lvl.tolist(); oid_l = oid_arr.tolist()
    alive: dict[int, dict] = {}
    ahead: dict[int, list] = {}
    for k, i in zip(ev_k.tolist(), ev_i.tolist()):
        L = lvl_l[i]
        a = alive.get(L)
        if k == 1:
            if a is not None:
                a.pop(oid_l[i], None)
            continue
        if a is None:
            a = alive[L] = {}
        oid = oid_l[i]
        if oid in tset:
            ahead[oid] = list(a)                      # 到着時点で前に居た注文
        a[oid] = i

    # ---- 前の行列がどう消えたかを分解 ------------------------------------
    idx_of = {int(o): i for i, o in enumerate(oid_arr)}
    rows = []
    for oid, (t_f, q_open) in tset.items():
        A = ahead.get(oid)
        if not A:
            continue
        i0 = idx_of[oid]
        t0 = int(o_arr[i0])
        ex = cn = 0.0
        for a in A:
            ja = idx_of[a]
            rem = float(sz_arr[ja]) - executed(a, -1, t0)      # t0 時点の残量
            if rem <= 0:
                continue
            e = executed(a, t0, t_f)
            ex += e
            if c_arr[ja] <= t_f:                                # t_f までに消滅した
                cn += max(rem - e, 0.0)
        tot = ex + cn
        if tot <= 0:
            continue
        rows.append((oid, i0, t_f, ex, cn, tot, (ex - cn) / tot, q_open))
    if len(rows) < 200:
        return None

    oi = np.array([r[1] for r in rows])
    tf = np.array([r[2] for r in rows], dtype=np.int64)
    ex = np.array([r[3] for r in rows]); cn = np.array([r[4] for r in rows])
    tot = np.array([r[5] for r in rows]); cr = np.array([r[6] for r in rows])
    qo = np.array([r[7] for r in rows])
    px_of = dict(zip(tgt["oid"].to_list(), tgt["fill_px"].to_list()))
    fpx = np.array([px_of[r[0]] for r in rows])
    sd = np.where(life["side"].to_numpy()[oi] == "B", 1.0, -1.0)

    # 整合性: 消えた総量 ≈ q_ahead_open か
    ok = qo > 0
    ratio = np.median(tot[ok] / qo[ok]) if ok.sum() else np.nan

    j0 = np.searchsorted(mts, tf, side="right") - 1
    good = j0 >= 0
    m0 = mid[np.clip(j0, 0, len(mid) - 1)]
    gross = sd * (m0 - fpx) / m0 * 1e4
    # 発注時に最良気配だったかで層別(前報告と揃える)
    jo = np.searchsorted(mts, o_arr[oi], side="right") - 1
    jo = np.clip(jo, 0, len(mid) - 1)
    ref = np.where(sd > 0, bb[jo], ba[jo])
    dist = sd * (ref - life["px"].to_numpy()[oi]) / mid[jo] * 1e4
    at_best = np.abs(dist) < 1e-9

    adv = {}
    for hn, dn in HORIZONS.items():
        j1 = np.searchsorted(mts, tf + dn, side="right") - 1
        v = np.where(j1 >= 0, sd * (mid[np.clip(j1, 0, len(mid) - 1)] - m0) / m0 * 1e4, np.nan)
        v[tf + dn > mts[-1]] = np.nan
        adv[hn] = v

    out_rows = []
    for lo, hi, lab in CB:
        for scope, base in (("全体", good), ("最良気配のみ", good & at_best)):
            m = base & (cr > lo) & (cr <= hi)
            if m.sum() < 100:
                continue
            r = {"bucket": lab, "scope": scope, "n": int(m.sum()),
                 "clear_ratio_mean": float(cr[m].mean()),
                 "gross_bp": float(gross[m].mean()),
                 "q_ahead_median": float(np.median(qo[m]))}
            for hn in HORIZONS:
                v = adv[hn][m]
                v = v[np.isfinite(v)]
                if v.size < 50:
                    continue
                r[f"adv_{hn}"] = float(v.mean())
                r[f"net_{hn}"] = float(gross[m][np.isfinite(adv[hn][m])].mean() + v.mean())
                if hn == "1s":
                    r["q_adv"] = {str(p): float(np.percentile(v, p))
                                  for p in (5, 25, 50, 75, 95)}
                    r["sd_1s"] = float(v.std())
                    r["neg_1s"] = float(np.mean(v < 0))
            out_rows.append(r)

    out = {"dt": dt, "n_targets": int(len(rows)),
           "consistency_tot_over_qahead": float(ratio),
           "share_all_cancel": float(np.mean(cr < -0.8)),
           "share_all_exec": float(np.mean(cr > 0.8)),
           "clear_ratio_mean": float(cr.mean()), "clear_ratio_median": float(np.median(cr)),
           "exec_share_overall": float(ex.sum() / (ex.sum() + cn.sum())),
           "buckets": out_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    # 注文単位の生データ。交絡の切り分け(clear_ratio × q_ahead など)を
    # 掃き直しなしで行えるようにする
    np.save(fn, np.stack([cr, qo, ex, cn, gross, at_best.astype(float),
                          adv["100ms"], adv["1s"], adv["10s"], adv["60s"],
                          good.astype(float)]).astype(np.float32))
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    for dt in sys.argv[1:]:
        r = run_day(dt)
        print(f"{dt}: 対象 {r['n_targets']:,} / 整合性 {r['consistency_tot_over_qahead']:.3f} "
              f"/ 約定で消えた割合 {r['exec_share_overall']:.3f} "
              f"/ 全部取消 {r['share_all_cancel']:.3f}" if r else f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
