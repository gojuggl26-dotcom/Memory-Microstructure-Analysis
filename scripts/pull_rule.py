"""段階①: リアルタイム代理指標「前方取消フロー」でのオフライン検証(日次処理)。

【戦略】
  自分の指値を出した後も、**自分より前に並ぶ注文**を追い続ける。
  直近 N ミリ秒でそのうち一定割合が**取消**で消えたら、自分も引く。

【シグナル(時刻 t で計算可能なことが必須)】
    flow(t, N) = Σ_{a ∈ A} (a が (t−N, t] に取消で消えた数量)
  A = 発注時 t0 に自分より前に並んでいた注文の集合。**未来の情報は一切使わない。**

  閾値の当て方を 2 通り持つ。
    割合 … flow / q_ahead_open > 閾値   「前の行列の何割が蒸発したか」
    絶対 … flow > 閾値(単位)           「何単位が蒸発したか」

  ★割合だけだと行列の深さで発火率が両極に振れる
    (pull_rule_strata_report.md §4-1: q≤20 で 91%、q>400 で 6%)。
    分母 q_ahead が浅いと必ず届き、深いと届かないため。
    絶対量ならこの副作用が無い。両方を出して比較する。

【反応時間 δ】
  時刻 t に発火を検知しても、取消が板に届くのは t + δ。
  したがって **t + δ < t_fill** のときだけ約定を回避できたことにする。
  margin = t_fill − t_trigger を保存しておけば、δ は事後に何通りでも当てられる。

【対照(これが検定の要)】
  同じ枠組みで 3 種のフローを比較する。
    cancel … 取消で消えた数量(本命)
    exec   … 約定で消えた数量
    total  … 両方の合計
  機構が「取消は情報」なら **cancel だけが効く**はず。
  「前で何か起きたら危ない」という一般論なら 3 種とも効くはず。
  この差が、queue_clearing_report.md の解釈を検定にかけることになる。

【評価の母集団】
  実際に起きたメイカー約定(前に注文があったもの)。
  引かなければ約定していたので、引いた分の損益がそのまま差分になる。
  **先頭の注文(前に何も無い)は対象外** — 監視する対象が存在しないため。

  層別に評価できるよう、**売買の別(side)と発注時の最良気配からの距離(dist_bp)**、
  および行列の深さ(q_ahead)を注文ごとに保存する。
  queue_clearing_report.md §4 では**深い行列(100 超)はどの消え方でも黒字**だったので、
  最良気配・浅い行列だけを見ていた前版とは結論が変わりうる。

【短期価格予測との組合せ(2026-08-16 追加)】
  book_slope(25bp 帯・1 秒グリッド)から向きの予測を作り、2 つの使い道を測る。
      align = s × (−(S^Bid − S^Ask))
      s = +1(自分が買い) / −1(自分が売り)
    slope_regression_report.md より予測リターンは −slope_diff に比例するので、
    align > 0 は「この約定は自分に有利な向き」を意味する。
  ① 参入ゲート … align_open(発注時点)が不利なら**そもそも置かない**
  ② 退出ゲート … **保有中に一度でも**不利へ転じたら引く

  ★②は align_min(t0 から t_fill−δ までの align の最小値)で判定する。
    「約定 δ 前の値」だけで見ると、途中で不利になっても直前に回復していれば
    約定を残してしまい、**実装不可能な後知恵**になる。
    実際の退出規則は「閾値を下回った瞬間に引く。その後回復しても戻れない」である。

  ★グリッドが 1 秒なので、δ=100ms を当てても実際の情報は最大 1.1 秒古い。
    前方フロー(ns 精度)より不利な条件での評価になる。

【時間契約】
  A は発注時点で確定。フローは (t−N, t] の後ろ向き窓。
  判定は t + δ < t_fill。未来は構造的に入らない。
  align も **backward asof のみ**(その時刻以前の最後のグリッド点)。
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
OUT = D / "pull_rule"
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
SIGNALS = ["cancel", "exec", "total"]
WINDOWS_MS = [10, 50, 100, 500]
THRESH_FRAC = [0.25, 0.50, 0.75]                    # q_ahead に対する割合
THRESH_ABS = [5.0, 20.0, 50.0, 150.0, 400.0]        # 絶対量(単位)
THRESHOLDS = ([f"frac{t}" for t in THRESH_FRAC]
              + [f"abs{t:g}" for t in THRESH_ABS])   # 表示用のラベル
NTHR = len(THRESHOLDS)
H1S = 1_000_000_000


def run_day(dt: str) -> dict | None:
    fj = OUT / f"{dt}.json"
    fn = OUT / f"{dt}.npz"
    if fj.exists() and fn.exists():
        return json.loads(fj.read_text(encoding="utf-8"))
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
    fl = (pl.concat([pl.read_parquet(x, columns=["oid", "ts", "px", "sz", "crossed"])
                     for x in fs], how="diagonal_relaxed")
          .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort(["oid", "t"]))
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "mid", "best_bid", "best_ask", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    mts = mp["ts"].to_numpy(); mid = mp["mid"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    sl = D / "slope25" / f"{dt}.parquet"
    slope = None
    if sl.exists():
        _s = pl.read_parquet(sl).sort("ts")
        slope = (_s["ts"].to_numpy(), (_s["s_bid"] - _s["s_ask"]).to_numpy())
    qi = (pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                          columns=["oid", "q_ahead_open", "open_unknown"])
          .filter((~pl.col("open_unknown")) & pl.col("q_ahead_open").is_not_null()))

    # ---- oid ごとの約定履歴(累積)。取消量は fills を真値として orig_sz から引く ----
    f_oid = fl["oid"].to_numpy(); f_t = fl["t"].to_numpy(); f_sz = fl["sz"].to_numpy()
    bnd = np.flatnonzero(np.diff(f_oid, prepend=f_oid[0] - 1, append=f_oid[-1] + 1))
    fill_idx = {}
    tot_filled = {}
    for k in range(len(bnd) - 1):
        s, e = bnd[k], bnd[k + 1]
        cs = np.concatenate([[0.0], np.cumsum(f_sz[s:e])])
        fill_idx[int(f_oid[s])] = (f_t[s:e], cs)
        tot_filled[int(f_oid[s])] = float(cs[-1])

    def executed(oid: int, a: int, b: int) -> float:
        v = fill_idx.get(oid)
        if v is None:
            return 0.0
        t_, cs = v
        return float(cs[np.searchsorted(t_, b, side="right")]
                     - cs[np.searchsorted(t_, a, side="right")])

    mk = (fl.filter(~pl.col("crossed")).group_by("oid")
            .agg(pl.col("t").min().alias("t_fill"), pl.col("px").first().alias("fill_px")))
    tgt = life.join(mk, on="oid", how="inner").join(qi, on="oid", how="inner")
    if tgt.height < 300:
        return None
    tset = {int(o): (int(tf), float(qa))
            for o, tf, qa in zip(tgt["oid"], tgt["t_fill"], tgt["q_ahead_open"])}

    # ---- 価格レベルごとの生存集合を掃く(到着を先に処理する規約)----------
    o_arr = life["ts_open"].to_numpy(); c_arr = life["ts_close"].to_numpy()
    oid_arr = life["oid"].to_numpy(); sz_arr = life["orig_sz"].to_numpy()
    lvl = (np.round(life["px"].to_numpy() * 1e6).astype(np.int64) * 2
           + (life["side"].to_numpy() == "B"))
    ev_t = np.concatenate([o_arr, c_arr])
    ev_k = np.concatenate([np.zeros(len(o_arr), np.int8), np.ones(len(c_arr), np.int8)])
    ev_i = np.concatenate([np.arange(len(o_arr)), np.arange(len(c_arr))])
    order = np.lexsort((ev_k, ev_t))
    ev_k, ev_i = ev_k[order], ev_i[order]
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
            ahead[oid] = list(a)
        a[oid] = i

    # ---- 各対象注文について、発火時刻の余裕 margin を全組合せで算出 --------
    idx_of = {int(o): i for i, o in enumerate(oid_arr)}
    W = [w * 1_000_000 for w in WINDOWS_MS]
    keys, margins = [], []
    for oid, (t_f, q_open) in tset.items():
        A = ahead.get(oid)
        if not A or q_open <= 0:
            continue
        i0 = idx_of[oid]
        t0 = int(o_arr[i0])
        # 前の注文が「いつ・どれだけ・どの理由で」消えたか(t0 より後の消滅のみ)
        cs, cv, ev = [], [], []
        for a in A:
            ja = idx_of[a]
            tc = int(c_arr[ja])
            if tc <= t0 or tc > t_f:
                continue
            rem = float(sz_arr[ja]) - executed(a, -1, t0)      # t0 時点の残量
            if rem <= 0:
                continue
            e = executed(a, t0, tc)                            # 消えるまでに約定した分
            cs.append(tc); cv.append(max(rem - e, 0.0)); ev.append(e)
        if not cs:
            continue
        o_ = np.argsort(np.array(cs))
        ct = np.array(cs, dtype=np.int64)[o_]
        vec = {"cancel": np.array(cv)[o_], "exec": np.array(ev)[o_]}
        vec["total"] = vec["cancel"] + vec["exec"]
        m = np.full((len(SIGNALS), len(W), NTHR), np.nan, dtype=np.float32)
        for si, sn in enumerate(SIGNALS):
            cum = np.concatenate([[0.0], np.cumsum(vec[sn])])
            for wi, w in enumerate(W):
                lo = np.searchsorted(ct, ct - w, side="left")   # (t−N, t] の左端
                flow_abs = cum[1:] - cum[lo]                    # 絶対量
                run_abs = np.maximum.accumulate(flow_abs)       # そこまでの最大
                run_frac = np.maximum.accumulate(flow_abs / q_open)
                for ti, thr in enumerate(THRESH_FRAC):
                    j = np.searchsorted(run_frac, thr, side="left")
                    if j < len(ct):
                        m[si, wi, ti] = float(t_f - ct[j])      # 余裕 ns(δ と比較)
                for ti, thr in enumerate(THRESH_ABS):
                    j = np.searchsorted(run_abs, thr, side="left")
                    if j < len(ct):
                        m[si, wi, len(THRESH_FRAC) + ti] = float(t_f - ct[j])
        keys.append(oid)
        margins.append(m)
    if len(keys) < 200:
        return None

    # ---- 各対象の損益(粗利 + 1 秒後の逆選択)-----------------------------
    ks = np.array(keys)
    oi = np.array([idx_of[int(k)] for k in ks])
    tf = np.array([tset[int(k)][0] for k in ks], dtype=np.int64)
    qo = np.array([tset[int(k)][1] for k in ks])
    px_of = dict(zip(tgt["oid"].to_list(), tgt["fill_px"].to_list()))
    fpx = np.array([px_of[int(k)] for k in ks])
    sd = np.where(life["side"].to_numpy()[oi] == "B", 1.0, -1.0)
    j0 = np.searchsorted(mts, tf, side="right") - 1
    good = j0 >= 0
    m0 = mid[np.clip(j0, 0, len(mid) - 1)]
    gross = sd * (m0 - fpx) / m0 * 1e4
    j1 = np.searchsorted(mts, tf + H1S, side="right") - 1
    adv = np.where(j1 >= 0, sd * (mid[np.clip(j1, 0, len(mid) - 1)] - m0) / m0 * 1e4, np.nan)
    adv[tf + H1S > mts[-1]] = np.nan
    net = gross + adv
    jo = np.clip(np.searchsorted(mts, o_arr[oi], side="right") - 1, 0, len(mid) - 1)
    ref = np.where(sd > 0, bb[jo], ba[jo])
    # 自分の側の最良気配からの距離(内側 = 負、外側 = 正)
    dist_bp = sd * (ref - life["px"].to_numpy()[oi]) / mid[jo] * 1e4
    at_best = np.abs(dist_bp) < 1e-9

    # ---- 短期価格予測 align(backward asof のみ)-------------------------
    def align_at(t_ns, sdir):
        if slope is None:
            return np.full(len(t_ns), np.nan)
        k = np.searchsorted(slope[0], t_ns, side="right") - 1
        v = np.where(k >= 0, slope[1][np.clip(k, 0, len(slope[1]) - 1)], np.nan)
        return sdir * (-v)

    t_open = o_arr[oi]
    align_open = align_at(t_open, sd)                       # ① 参入ゲート用
    align_fill = align_at(tf - 100_000_000, sd)             # 参考(δ 前の値)

    # ★② 退出ゲート用: 保有中 (t0, t_fill−δ] の align の最小値。
    #    一度でも閾値を下回ったら引く、という実装可能な規則に対応する
    align_min = np.full(len(oi), np.nan)
    if slope is not None:
        st_, sv_ = slope
        lo_ = np.searchsorted(st_, t_open, side="right")     # t0 より後の最初の点
        hi_ = np.searchsorted(st_, tf - 100_000_000, side="right")   # δ 前まで
        for k in range(len(oi)):
            a_, b_ = int(lo_[k]), int(hi_[k])
            if b_ > a_:
                align_min[k] = float(np.min(sd[k] * (-sv_[a_:b_])))
            else:
                align_min[k] = align_open[k]                 # 窓に点が無ければ発注時の値

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fn, margin=np.stack(margins).astype(np.float32),
                        align_open=align_open.astype(np.float32),
                        align_fill=align_fill.astype(np.float32),
                        align_min=align_min.astype(np.float32),
                        ts_open=t_open.astype(np.int64), t_fill=tf.astype(np.int64),
                        oid=ks.astype(np.int64),
                        net=net.astype(np.float32), gross=gross.astype(np.float32),
                        adv=adv.astype(np.float32), qo=qo.astype(np.float32),
                        at_best=at_best, is_bid=(sd > 0),
                        dist_bp=dist_bp.astype(np.float32),
                        good=(good & np.isfinite(net)))
    # ★データ品質: queue_qi が壊れた日を機械的に弾く。
    #   2026-08-10 は lifecycle の ts_close に null があり build_queue_qi の
    #   マージが破綻して q_ahead_open が 1,135 倍に膨らんでいた。
    #   日付をハードコードせず、指標そのもので検出する
    q_med = float(np.median(qo))
    out = {"dt": dt, "n_targets": int(len(ks)),
           "n_valid": int((good & np.isfinite(net)).sum()),
           "q_ahead_median": q_med, "suspect": bool(q_med > 1000),
           "signals": SIGNALS, "windows_ms": WINDOWS_MS, "thresholds": THRESHOLDS}
    fj.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    for dt in sys.argv[1:]:
        r = run_day(dt)
        if r:
            flag = "  ★q_ahead 異常" if r.get("suspect") else ""
            print(f"{dt}: 対象 {r['n_targets']:,} / 有効 {r['n_valid']:,} "
                  f"/ q_ahead中央 {r.get('q_ahead_median', 0):.0f}{flag}", flush=True)
        else:
            print(f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
