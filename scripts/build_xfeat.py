"""bbo + fills だけから、銘柄横断で比べられる特徴量パネルを 100ms 格子で作る。

    uv run python scripts/build_xfeat.py --coin xyz:AMD

なぜ別に作るか
--------------
[INTC の l4feat](../reports/hyperliquid/INTC/intc_l4feat_report.md) は l1(注文
一本ごと)から 20 水準の板を組むが、l1 はローカルに無い銘柄がある。いっぽう
**bbo(最良気配 1 段)と fills(約定)はミラーに全銘柄ぶんある**ので、
egress ゼロで作れる範囲を切り出したのがこのパネル。l4feat との違いは
「板は最良 1 段だけ」で、板の形・空白・多水準 OBI・キュー・寿命は作れない。
作れるのは価格 / マイクロプライス / L1 OBI / L1 OFI / ボラティリティ /
最良気配の入れ替わり / 約定フロー、とその派生。

MU/INTC と同じ物差しで**全所有銘柄を横並び**にするのが狙い。

時間契約(★破ったら結果は捨てる)
--------------------------------
  * 気配は searchsorted(..., "left") - 1 で厳密に T 未満の最後の行
  * フロー窓は [T-w, T)。格子点 T の 100ms 区間そのものを含めない
  * 目的変数(label_)だけが未来。説明変数に負のシフトを書かない
  * クロス(best_ask < best_bid)の行は除外
  * 分位・レジームの基準は前日から作る(このパネルでは派生の z/ dev は
    後ろ向き窓なので全標本パラメータは使っていない)

出力: E:/Memory-xfeat/<coin>/dt=YYYY-MM-DD.parquet  1 秒おきの行
目的変数は 100ms 格子で作るので 100ms・500ms も正確。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTROOT = Path("E:/Memory-xfeat")
GRID_NS = 100_000_000
DAY_NS = 86_400_000_000_000
NG = DAY_NS // GRID_NS
OUT_STEP = 10
WIN = [1, 5, 10, 50, 100, 300, 600]            # 100ms 単位
WLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]
FWD = [1, 5, 10, 50, 100, 300, 600]
FLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]

FAM: dict[str, str] = {}


def reg(n, f):
    FAM[n] = f
    return n


def cum0(x):
    c = np.empty(x.size + 1, np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    return c


def bsum(c, idx, w):
    return c[idx] - c[np.maximum(idx - w, 0)]


def safe(a, b, fill=np.nan):
    b = np.asarray(b, np.float64)
    ok = np.abs(b) > 1e-12
    return np.where(ok, np.asarray(a, np.float64) / np.where(ok, b, 1.0), fill)


def tsince(flag, dt):
    n = flag.size
    idx = np.where(flag, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, (np.arange(n) - last) * dt, 1e6)


def clean_bbo(b: pl.DataFrame):
    """クロスと外れ値を落とす(l4feat/obi_levels と同じ規則)。"""
    n0 = b.height
    b = (b.filter(pl.col("best_ask") > pl.col("best_bid")).sort("ts")
         .with_columns(mid=(pl.col("best_bid") + pl.col("best_ask")) / 2))
    b = b.with_columns(
        rs=(pl.col("best_ask") - pl.col("best_bid")) / pl.col("mid") * 1e4,
        bmed=pl.col("mid").rolling_median(21, min_samples=5).shift(1))
    b = b.filter((pl.col("rs") <= 100)
                 & (pl.col("bmed").is_null()
                    | ((pl.col("mid") / pl.col("bmed")).log().abs() <= 0.05)))
    return b.drop("mid", "rs", "bmed"), n0 - b.height


def grid_state(b, t0):
    """100ms 格子へ後ろ向きに最良気配を載せる。"""
    ts = b["ts"].to_numpy()
    tg = t0 + np.arange(NG, dtype=np.int64) * GRID_NS
    j = np.searchsorted(ts, tg, side="left") - 1
    ok = j >= 0
    j = np.where(ok, j, 0)
    pb = b["best_bid"].to_numpy()[j]
    pa = b["best_ask"].to_numpy()[j]
    qb = b["bid_sz"].to_numpy()[j]
    qa = b["ask_sz"].to_numpy()[j]
    good = ok & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)
    return pb, pa, qb, qa, good, j


def tick_of(px):
    """価格から刻みを推定(<1000 は 0.01、以上は 0.1)。"""
    return np.where(px >= 1000, 0.1, 0.01)


def build_day(bb_day, fl_day, t0):
    pb, pa, qb, qa, good, jrow = grid_state(bb_day, t0)
    mid = np.where(good, (pb + pa) / 2, np.nan)
    s = qb + qa
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = np.where(good & (s > 0), (pb * qa + pa * qb) / np.where(s > 0, s, 1), np.nan)
    lm = np.log(np.where(good, mid, np.nan))
    lmi = np.log(np.where(np.isfinite(micro) & (micro > 0), micro, np.nan))
    o = np.arange(0, NG, OUT_STEP)
    F: dict[str, np.ndarray] = {}
    tick = tick_of(pb[o])
    sp = (pa - pb)

    # --- 1.1 最良気配 -----------------------------------------------------
    F[reg("best_bid", "1.1 最良気配")] = pb[o]
    F[reg("best_ask", "1.1 最良気配")] = pa[o]
    F[reg("bid_sz", "1.1 最良気配")] = qb[o]
    F[reg("ask_sz", "1.1 最良気配")] = qa[o]
    F[reg("mid", "1.2 中値")] = mid[o]
    F[reg("log_mid", "1.2 中値")] = lm[o]
    F[reg("spread_abs", "1.3 スプレッド")] = sp[o]
    F[reg("spread_tick", "1.3 スプレッド")] = sp[o] / tick
    F[reg("spread_bp", "1.3 スプレッド")] = safe(sp[o], mid[o]) * 1e4
    F[reg("spread_rel", "1.3 スプレッド")] = safe(sp[o], mid[o])
    F[reg("is_1tick", "1.3 スプレッド")] = (np.abs(sp[o] / tick - 1) < 0.01).astype(float)

    # --- 1.2 リターン(後ろ向き)-----------------------------------------
    for k, lab in zip(FWD, FLAB):
        F[reg(f"ret_{lab}", "1.2 中値")] = (lm[o] - lm[np.maximum(o - k, 0)]) * 1e4
        F[reg(f"mret_{lab}", "2.1 マイクロプライス")] = (lmi[o] - lmi[np.maximum(o - k, 0)]) * 1e4
    F[reg("d_mid", "1.2 中値")] = F["ret_100ms"]
    F[reg("abs_d_mid", "1.2 中値")] = np.abs(F["ret_100ms"])
    F[reg("sgn_d_mid", "1.2 中値")] = np.sign(F["ret_100ms"])

    # --- 2.1 マイクロプライス --------------------------------------------
    F[reg("micro", "2.1 マイクロプライス")] = micro[o]
    F[reg("delta_bp", "2.1 マイクロプライス")] = safe(micro[o] - mid[o], mid[o]) * 1e4
    F[reg("delta_norm", "2.1 マイクロプライス")] = safe(micro[o] - mid[o], sp[o] / 2)
    F[reg("micro_bid", "2.1 マイクロプライス")] = safe(micro[o] - pb[o], mid[o]) * 1e4
    F[reg("ask_micro", "2.1 マイクロプライス")] = safe(pa[o] - micro[o], mid[o]) * 1e4
    F[reg("mp_mom", "2.1 マイクロプライス")] = F["delta_bp"] - np.concatenate(
        [[np.nan] * 10, F["delta_bp"][:-10]])

    # --- 4 L1 OBI --------------------------------------------------------
    obi1 = safe(qb - qa, qb + qa, 0.0)
    F[reg("obi1", "4 板の偏り OBI")] = obi1[o]
    F[reg("obi_abs", "4 板の偏り OBI")] = np.abs(obi1[o])
    F[reg("obi_sign", "4 板の偏り OBI")] = np.sign(obi1[o])
    nb = qb * pb
    na = qa * pa
    F[reg("obi_not1", "4 板の偏り OBI")] = safe(nb - na, nb + na, 0.0)[o]
    F[reg("log_depth_ratio", "4 板の偏り OBI")] = np.log(safe(qb, qa, np.nan))[o]
    F[reg("d_obi", "4 板の偏り OBI")] = obi1[o] - np.concatenate([[np.nan], obi1[o][:-1]])
    # 慣性: 同符号が続いた長さ
    sgn = np.sign(obi1[o])
    same = np.concatenate([[False], sgn[1:] == sgn[:-1]])
    run = np.zeros(o.size)
    cur = 0.0
    for k in range(o.size):
        cur = cur + 1 if same[k] else 1.0
        run[k] = cur
    F[reg("obi_streak", "62 慣性")] = run * sgn

    # --- 10 L1 OFI(Cont, Kukanov, Stoikov)-------------------------------
    # 気配イベント列で e = Δ(bid depth at best) − Δ(ask depth) を作り、格子へ集計
    ofi_e, tsofi = l1_ofi(bb_day)
    gi = ((tsofi - t0) // GRID_NS).clip(0, NG - 1)
    ofi_grid = np.bincount(gi, weights=ofi_e, minlength=NG).astype(np.float64)
    absflow = np.bincount(gi, weights=np.abs(ofi_e), minlength=NG).astype(np.float64)
    cofi = cum0(ofi_grid)
    cabs = cum0(absflow)
    for w, lab in zip(WIN, WLAB):
        F[reg(f"ofi_{lab}", "10 注文フローの偏り OFI")] = bsum(cofi, o, w)
        F[reg(f"ofi_norm_{lab}", "10 注文フローの偏り OFI")] = safe(
            bsum(cofi, o, w), bsum(cabs, o, w), 0.0)
    F[reg("ofi_fast_slow", "67 多尺度")] = F["ofi_1s"] - F["ofi_30s"] / 30.0

    # --- 39 最良気配の入れ替わり / 43 経過時間 / 44 群発 ------------------
    bch = np.concatenate([[True], (np.diff(pb) != 0) | (np.diff(pa) != 0)])
    mch = np.concatenate([[True], np.diff(mid) != 0])
    sch = np.concatenate([[True], np.diff(sp) != 0])
    cbch = cum0(bch.astype(float))
    for w, lab in zip(WIN, WLAB):
        F[reg(f"bbo_chg_{lab}", "39 最良気配の入れ替わり")] = bsum(cbch, o, w)
    F[reg("t_since_bbo", "43 経過時間")] = tsince(bch, GRID_NS / 1e9)[o]
    F[reg("t_since_mid", "43 経過時間")] = tsince(mch, GRID_NS / 1e9)[o]
    F[reg("t_since_sp", "43 経過時間")] = tsince(sch, GRID_NS / 1e9)[o]
    # 気配イベントの群発性(格子ごとの気配更新数の Fano・CV)
    ev_ts = bb_day["ts"].to_numpy()
    egi = ((ev_ts - t0) // GRID_NS)
    egi = egi[(egi >= 0) & (egi < NG)].astype(np.int64)
    ecnt = np.bincount(egi, minlength=NG).astype(np.float64)
    cec = cum0(ecnt)
    cec2 = cum0(ecnt * ecnt)
    for w, lab in zip((100, 600), ("10s", "60s")):
        m1 = bsum(cec, o, w) / w
        m2 = bsum(cec2, o, w) / w
        F[reg(f"msg_rate_{lab}", "28 メッセージの流量")] = bsum(cec, o, w) / (w * GRID_NS / 1e9)
        F[reg(f"fano_{lab}", "44 群発性")] = safe(m2 - m1 * m1, m1, np.nan)

    # --- 47 ボラティリティ ----------------------------------------------
    r = np.diff(lm, prepend=np.nan) * 1e4
    r2 = np.nan_to_num(r * r)
    cr2 = cum0(r2)
    ar = np.nan_to_num(np.abs(r))
    cbp = cum0(ar * np.concatenate([[0.0], ar[:-1]]))
    cup = cum0(np.where(np.nan_to_num(r) > 0, r2, 0))
    cdn = cum0(np.where(np.nan_to_num(r) < 0, r2, 0))
    rm = np.diff(lmi, prepend=np.nan) * 1e4
    cm2 = cum0(np.nan_to_num(rm * rm))
    csp = cum0(np.nan_to_num(np.diff(np.where(good, sp, np.nan), prepend=np.nan) ** 2))
    for w, lab in zip(WIN[2:], WLAB[2:]):
        F[reg(f"rv_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(cr2, o, w))
        F[reg(f"bpv_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(cbp, o, w) * np.pi / 2)
        F[reg(f"rsv_up_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(cup, o, w))
        F[reg(f"rsv_dn_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(cdn, o, w))
        F[reg(f"jump_{lab}", "47 ボラティリティ")] = safe(
            F[f"rv_{lab}"] - F[f"bpv_{lab}"], F[f"rv_{lab}"])
    for w, lab in zip((100, 600), ("10s", "60s")):
        F[reg(f"rv_micro_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(cm2, o, w))
        F[reg(f"rv_spread_{lab}", "47 ボラティリティ")] = np.sqrt(bsum(csp, o, w))

    # --- 51/52 約定フロー(fills)----------------------------------------
    trade_feats(fl_day, t0, o, mid, qb, qa, F)

    # --- 派生(59 レジーム簡易・60 z・63 dev・61 加速・70 交互作用)------
    derived(F, o.size)

    return F, o, lm, lmi, dict(n_bbo=bb_day.height)


def l1_ofi(b):
    """最良気配 1 段の Order Flow Imbalance の増分列を返す。

    Cont, Kukanov, Stoikov (2014):
      e_n = ( 1{P^b_n >= P^b_{n-1}} q^b_n − 1{P^b_n <= P^b_{n-1}} q^b_{n-1} )
          − ( 1{P^a_n <= P^a_{n-1}} q^a_n − 1{P^a_n >= P^a_{n-1}} q^a_{n-1} )
    価格が上がれば買い圧、下がれば売り圧。数量は枚数。
    """
    pb = b["best_bid"].to_numpy()
    pa = b["best_ask"].to_numpy()
    qb = b["bid_sz"].to_numpy()
    qa = b["ask_sz"].to_numpy()
    ts = b["ts"].to_numpy()
    pbp = np.concatenate([[pb[0]], pb[:-1]])
    pap = np.concatenate([[pa[0]], pa[:-1]])
    qbp = np.concatenate([[qb[0]], qb[:-1]])
    qap = np.concatenate([[qa[0]], qa[:-1]])
    eb = np.where(pb >= pbp, qb, 0.0) - np.where(pb <= pbp, qbp, 0.0)
    ea = np.where(pa <= pap, qa, 0.0) - np.where(pa >= pap, qap, 0.0)
    e = eb - ea
    e[0] = 0.0
    return e, ts


def trade_feats(fl, t0, o, mid, qb, qa, F):
    if fl is None or not fl.height:
        z = np.zeros(NG)
        cb = ca = cbv = cav = z
    else:
        f = fl.filter(pl.col("crossed"))
        g = ((f["ts"].cast(pl.Int64).to_numpy() - t0) // GRID_NS)
        m = (g >= 0) & (g < NG)
        g = g[m].astype(np.int64)
        sz = f["sz"].to_numpy()[m].astype(np.float64)
        buy = (f["side"].to_numpy()[m] == "B")
        cb = cum0(np.bincount(g[buy], minlength=NG).astype(float))
        ca = cum0(np.bincount(g[~buy], minlength=NG).astype(float))
        cbv = cum0(np.bincount(g[buy], weights=sz[buy], minlength=NG))
        cav = cum0(np.bincount(g[~buy], weights=sz[~buy], minlength=NG))
    dep = (qb + qa)
    for w, lab in zip(WIN[2:], WLAB[2:]):
        nb, na = bsum(cb, o, w), bsum(ca, o, w)
        vb, va = bsum(cbv, o, w), bsum(cav, o, w)
        dtw = w * GRID_NS / 1e9
        F[reg(f"trd_rate_{lab}", "51 約定の符号")] = (nb + na) / dtw
        F[reg(f"trd_imb_{lab}", "51 約定の符号")] = safe(nb - na, nb + na, 0.0)
        F[reg(f"vol_imb_{lab}", "51 約定の符号")] = safe(vb - va, vb + va, 0.0)
        F[reg(f"sgn_vol_{lab}", "51 約定の符号")] = vb - va
        F[reg(f"vol_{lab}", "51 約定の符号")] = vb + va
        F[reg(f"vpin_{lab}", "49 毒性の代理")] = safe(np.abs(vb - va), vb + va, np.nan)
        F[reg(f"trd_sz_{lab}", "51 約定の符号")] = safe(vb + va, nb + na, np.nan)
    F[reg("cvd_60s", "51 約定の符号")] = F["sgn_vol_60s"]
    for lab in ("1s", "10s", "60s"):
        F[reg(f"vol_over_depth_{lab}", "52 板と約定")] = safe(F[f"vol_{lab}"], dep[o], np.nan)


CORE = ["obi1", "delta_bp", "spread_bp", "ofi_1s", "ofi_10s", "rv_10s",
        "vol_imb_10s", "bbo_chg_10s", "micro_bid", "trd_imb_10s"]


def derived(F, n):
    W = 300
    for nm in CORE:
        if nm not in F:
            continue
        x = np.asarray(F[nm], np.float64)
        d1 = np.concatenate([[np.nan], np.diff(x)])
        F[reg(f"{nm}__d", "61 加速と高次の動き")] = d1
        F[reg(f"{nm}__dd", "61 加速と高次の動き")] = np.concatenate([[np.nan], np.diff(d1)])
        xx = np.nan_to_num(x)
        ok = np.isfinite(x).astype(float)
        c1, c2, cn = cum0(xx), cum0(xx * xx), cum0(ok)
        i = np.arange(n)
        cnt = np.maximum(cn[i] - cn[np.maximum(i - W, 0)], 1)
        m = (c1[i] - c1[np.maximum(i - W, 0)]) / cnt
        v = np.maximum((c2[i] - c2[np.maximum(i - W, 0)]) / cnt - m * m, 0)
        F[reg(f"{nm}__z", "60 衝撃(z 得点)")] = safe(x - m, np.sqrt(v), np.nan)
        F[reg(f"{nm}__dev", "63 平均回帰")] = x - m
    for a, b in [("obi1", "spread_bp"), ("ofi_10s", "rv_10s"), ("delta_bp", "spread_bp"),
                 ("obi1", "rv_10s"), ("vol_imb_10s", "rv_10s")]:
        if a in F and b in F:
            F[reg(f"{a}__x__{b}", "70 交互作用")] = np.asarray(F[a]) * np.asarray(F[b])


def labels(F, o, lm, lmi):
    n = lm.size
    for k, lab in zip(FWD, FLAB):
        j = np.minimum(o + k, n - 1)
        F[f"label_micro_{lab}"] = (lmi[j] - lmi[o]) * 1e4
        F[f"label_mid_{lab}"] = (lm[j] - lm[o]) * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=999)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    bb = pl.read_parquet(DATA / f"bbo_{tag}.parquet")
    fp_fl = DATA / f"fills_{tag}.parquet"
    fl = pl.read_parquet(fp_fl) if fp_fl.exists() else None
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(bb["dt"].unique().to_list())
    sel = days[a.start:a.start + a.n]
    for dt in sel:
        t = time.time()
        bd, _ = clean_bbo(bb.filter(pl.col("dt") == dt))
        if not bd.height:
            print(f"  {dt} 空", flush=True)
            continue
        t0 = (int(bd["ts"].min()) // DAY_NS) * DAY_NS
        fd = fl.filter(pl.col("dt") == dt) if fl is not None else None
        F, o, lm, lmi, meta = build_day(bd, fd, t0)
        labels(F, o, lm, lmi)
        df = pl.DataFrame({"dt": np.full(o.size, dt),
                           "sec": (o * GRID_NS // 1_000_000_000).astype(np.int32)}
                          | {k: np.asarray(v, np.float32) for k, v in F.items()})
        df.write_parquet(out / f"dt={dt}.parquet", compression="zstd", compression_level=3)
        print(f"  {dt}  {df.width} 列 {df.height} 行  {time.time()-t:.1f}s", flush=True)
    if FAM:
        pl.DataFrame({"col": list(FAM), "family": list(FAM.values())}) \
          .write_csv(DATA / f"xfeat_cols_{tag}.csv")


if __name__ == "__main__":
    main()
