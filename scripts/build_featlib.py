"""L4 板データから、δ = microprice − mid を分解するための特徴量ライブラリを作る。

    uv run python scripts/build_featlib.py --coin xyz:MU [--days N]

狙い
----
新しい alpha を増やすのではなく、**δ が何を代理しているのかを分解できる**
特徴量を一式そろえる。12 分類・約 180 個。

  1 Price / BBO 基礎状態      2 Order Book Imbalance
  3 Depth / Liquidity         4 Microprice 系
  5 Book Shape                6 Liquidity Gap
  7 Liquidity Elasticity      8 Order Flow Imbalance
  9 Limit Order Flow         10 Cancellation
 11 Limit-order Arrival      12 Aggressive Trade Flow

δ ≈ (Spread/2) × OBI という恒等式に近い関係があるので、
**OBI と Spread と OBI×Spread を必ず別々に持つ**。

板の再構成
----------
多水準の板は build_obi_levels.day_features を **keep つき**で呼び、
水準ごとの数量 Qb / Qa を 1 秒格子で取り出す。水準は最良気配から
10 ティックまで(NLV=10)。刻みは価格で変わる(<1000 は 0.01、>=1000 は 0.1)。

時間契約
--------
すべて**その時刻までの情報だけ**で作る。窓は後ろ向き、差分は過去との差。
将来の値は目的変数(fwd_*)としてのみ持ち、説明変数には混ぜない。

★2026-09-05 修正 (1): 9〜12 分類(指値フロー / 取消 / 到着 / 攻撃的な約定)は
イベントを「その秒」の格子点に入れていた。格子点 T の値が [T, T+1) を集計する
一方、目的変数は (T, T+h] なので **1 秒ぶん重なっていた**(h=1s では丸ごと同時点)。
集計先を 1 つ後ろへずらし、格子点 T には [T−1s, T) のイベントだけを入れる。

★2026-09-05 修正 (2): `ofi_pct` は「その日全体での順位」だったため、格子点 T の値を
T より後の値まで含めた分布で順位づけていた。**分位の基準を全標本から作らない**
という規則(CLAUDE.md)に反するので、**前日の分布に対する順位**へ直した。
初日は基準が無いので NaN になる。xyz:MU の 21 日で測った限り
`fwd_micro_10s` との相関は 0.0568(旧)で、同じく単調変換の `ofi_sign` の
0.0572 とほぼ同じだった — つまり効果は小さいが、小さいことは残す理由にならない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, GRID_NS, NLV, PX_UNIT,  # noqa: E402
                              RESTING, SZ_LOT, TERMINAL, clean_bbo,
                              day_features as ladder_day)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000
GRID_S = 1                     # 特徴量を作る格子 (秒)
NG = 86_400
KEEP_EVERY = 5                 # 書き出しは 5 秒おき (分布・関係の分析には十分)
OFI_WIN = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
FWD = [1.0, 10.0, 60.0]        # 目的変数の前向きホライズン (秒)
LAM = 0.5                      # 指数重み OBI/microprice の減衰


def tsince(mask, dtsec=GRID_S):
    """直近で True になってからの経過秒。一度も無ければ大きな値。"""
    n = mask.size
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, (np.arange(n) - last) * dtsec, 1e6)


def rsum(x, w):
    """後ろ向き w 点の移動和 (t を含む)。"""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size) + 1
    return c[i] - c[np.maximum(i - w, 0)]


def rstd(x, w):
    m = rsum(x, w) / w
    m2 = rsum(x * x, w) / w
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def zsc(x, w):
    m = rsum(x, w) / w
    s = rstd(x, w)
    return np.where(s > 1e-12, (x - m) / np.where(s > 1e-12, s, 1.0), 0.0)


def streak(sign):
    """同符号が続いている長さ (点数)。"""
    n = sign.size
    out = np.zeros(n)
    same = np.concatenate([[False], sign[1:] == sign[:-1]])
    run = 0
    for i in range(n):                       # 走査 1 回で足りる
        run = run + 1 if same[i] else 1
        out[i] = run
    return out


def safe(a, b, fill=0.0):
    return np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1.0),
                    fill)


def ladder_feats(Qb, Qa, tick_bp, mid, spread_bp):
    """水準ごとの数量から 2〜7 分類を作る。Qb/Qa は (n, NLV) のロット数。"""
    F = {}
    L = np.arange(1, NLV + 1)
    cb = np.cumsum(Qb, axis=1)
    ca = np.cumsum(Qa, axis=1)

    # --- 2. Order Book Imbalance -----------------------------------------
    for K in (1, 2, 3, 5, 10):
        F[f"obi{K}"] = safe(cb[:, K - 1] - ca[:, K - 1], cb[:, K - 1] + ca[:, K - 1])
    wd = 1.0 / L
    we = np.exp(-LAM * (L - 1))
    for nm, w in (("dw", wd), ("ew", we)):
        b = Qb @ w
        a = Qa @ w
        F[f"obi_{nm}"] = safe(b - a, b + a)
    pb_lv = mid[:, None] - (L[None, :] - 0.5) * tick_bp[:, None] * mid[:, None] / 1e4
    pa_lv = mid[:, None] + (L[None, :] - 0.5) * tick_bp[:, None] * mid[:, None] / 1e4
    nb = (Qb * SZ_LOT * pb_lv).sum(1)
    na = (Qa * SZ_LOT * pa_lv).sum(1)
    F["obi_notional"] = safe(nb - na, nb + na)
    F["depth_diff"] = (Qb[:, 0] - Qa[:, 0]) * SZ_LOT
    F["depth_ratio"] = safe(Qb[:, 0], Qa[:, 0], np.nan)
    F["log_depth_ratio"] = np.log(np.maximum(F["depth_ratio"], 1e-9))
    F["obi_abs"] = np.abs(F["obi1"])
    F["obi_sign"] = np.sign(F["obi1"])
    F["obi_near_deep"] = F["obi1"] - F["obi10"]
    d1 = np.diff(F["obi1"], prepend=F["obi1"][0])
    F["d_obi"] = d1
    F["obi_vel"] = d1 / GRID_S
    F["obi_acc"] = np.diff(F["obi_vel"], prepend=F["obi_vel"][0])
    F["obi_persist"] = tsince(np.concatenate(
        [[True], np.sign(F["obi1"][1:]) != np.sign(F["obi1"][:-1])]))
    F["obi_streak"] = streak(np.sign(F["obi1"]))

    # --- 3. Depth / Liquidity --------------------------------------------
    for K in (1, 2, 3, 5, 10):
        F[f"qb{K}"] = Qb[:, K - 1] * SZ_LOT
        F[f"qa{K}"] = Qa[:, K - 1] * SZ_LOT
        F[f"cum_b{K}"] = cb[:, K - 1] * SZ_LOT
        F[f"cum_a{K}"] = ca[:, K - 1] * SZ_LOT
    F["depth_bbo"] = (Qb[:, 0] + Qa[:, 0]) * SZ_LOT
    F["depth_cum10"] = (cb[:, 9] + ca[:, 9]) * SZ_LOT
    F["depth_asym"] = safe(cb[:, 9] - ca[:, 9], cb[:, 9] + ca[:, 9])
    near = (cb[:, 2] + ca[:, 2]) * SZ_LOT
    deep = (cb[:, 9] - cb[:, 2] + ca[:, 9] - ca[:, 2]) * SZ_LOT
    F["near_depth"] = near
    F["deep_depth"] = deep
    F["near_deep_ratio"] = safe(near, deep, np.nan)
    F["dollar_depth"] = (nb + na)
    F["depth_per_tick"] = F["depth_cum10"] / NLV
    F["depth_per_bp"] = safe(F["depth_cum10"], NLV * tick_bp, np.nan)
    F["liq_density"] = safe(F["depth_cum10"], spread_bp, np.nan)

    # --- 4. Microprice ---------------------------------------------------
    hs = spread_bp / 2.0
    for K in (1, 2, 3, 5, 10):
        o = F[f"obi{K}"]
        F[f"micro{K}"] = mid * (1.0 + hs * o / 1e4)
        F[f"delta{K}"] = mid * hs * o / 1e4
    for nm in ("dw", "ew", "notional"):
        F[f"micro_{nm}"] = mid * (1.0 + hs * F[f"obi_{nm}"] / 1e4)
    F["delta1_bp"] = hs * F["obi1"]
    F["delta5_bp"] = hs * F["obi5"]
    F["delta10_bp"] = hs * F["obi10"]
    F["delta_disagree"] = F["delta1_bp"] - F["delta5_bp"]
    F["micro_slope"] = (F["obi10"] - F["obi1"]) / 9.0
    F["obi_x_spread"] = F["obi1"] * spread_bp        # ★δ の分解に必須

    # --- 5. Book Shape ---------------------------------------------------
    x = L.astype(float)
    xc = x - x.mean()
    den = float((xc * xc).sum())
    F["slope_b"] = (Qb * SZ_LOT) @ xc / den
    F["slope_a"] = (Qa * SZ_LOT) @ xc / den
    F["slope_asym"] = F["slope_b"] - F["slope_a"]
    F["local_slope_b"] = (Qb[:, 2] - Qb[:, 0]) * SZ_LOT / 2.0
    F["local_slope_a"] = (Qa[:, 2] - Qa[:, 0]) * SZ_LOT / 2.0
    F["deep_slope_b"] = (Qb[:, 9] - Qb[:, 4]) * SZ_LOT / 5.0
    F["deep_slope_a"] = (Qa[:, 9] - Qa[:, 4]) * SZ_LOT / 5.0
    F["curv_b"] = (Qb[:, 0] - 2 * Qb[:, 4] + Qb[:, 9]) * SZ_LOT
    F["curv_a"] = (Qa[:, 0] - 2 * Qa[:, 4] + Qa[:, 9]) * SZ_LOT
    F["shape_asym"] = F["curv_b"] - F["curv_a"]
    for nm, Q in (("b", Qb), ("a", Qa)):
        tot = Q.sum(1)
        w = safe(Q, tot[:, None], 0.0)
        F[f"hhi_{nm}"] = (w * w).sum(1)
        ws = np.sort(w, axis=1)
        F[f"gini_{nm}"] = ((2 * np.arange(1, NLV + 1) - NLV - 1) * ws).sum(1) / NLV
        F[f"entropy_{nm}"] = -(w * np.log(np.maximum(w, 1e-12))).sum(1)
    F["liq_conc"] = 0.5 * (F["hhi_b"] + F["hhi_a"])

    # --- 6. Liquidity Gap -------------------------------------------------
    for nm, Q in (("b", Qb), ("a", Qa)):
        occ = Q > 0
        first = np.where(occ[:, 1:].any(1), occ[:, 1:].argmax(1) + 1, NLV)
        F[f"gap12_{nm}"] = first.astype(float)
        pos = [np.where(occ[:, i], i + 1.0, np.nan) for i in range(NLV)]
        P = np.column_stack(pos)
        d = np.diff(np.where(np.isnan(P), np.nan, P), axis=1)
        F[f"max_gap_{nm}"] = np.nan_to_num(np.nanmax(np.where(np.isnan(d), 0, d),
                                                     axis=1))
        F[f"mean_gap_{nm}"] = safe(np.nan_to_num(np.where(np.isnan(d), 0, d)).sum(1),
                                   np.maximum(occ.sum(1) - 1, 1))
        F[f"empty_lv_{nm}"] = (~occ).sum(1).astype(float)
        big = Q > (Q.mean(1, keepdims=True) * 3.0)
        F[f"wall_dist_{nm}"] = np.where(big.any(1), big.argmax(1) + 1.0, NLV + 1.0)
        F[f"cliff_{nm}"] = np.where(occ.any(1), (~occ).argmax(1) + 1.0, NLV + 1.0)
    F["gap_asym"] = F["gap12_b"] - F["gap12_a"]
    F["med_gap"] = 0.5 * (F["mean_gap_b"] + F["mean_gap_a"])
    F["cum_gap"] = F["empty_lv_b"] + F["empty_lv_a"]
    F["wgt_gap"] = (F["gap12_b"] * Qb[:, 0] + F["gap12_a"] * Qa[:, 0]) \
        / np.maximum(Qb[:, 0] + Qa[:, 0], 1e-9)

    # --- 7. Liquidity Elasticity -----------------------------------------
    #   %Δ深さ / %Δ距離。距離は水準(ティック)で測る。
    for nm, C in (("b", cb), ("a", ca)):
        e_lo = safe(np.log(np.maximum(C[:, 2], 1e-9)) - np.log(np.maximum(C[:, 0], 1e-9)),
                    np.log(3.0))
        e_hi = safe(np.log(np.maximum(C[:, 9], 1e-9)) - np.log(np.maximum(C[:, 4], 1e-9)),
                    np.log(10.0 / 5.0))
        F[f"elast_{nm}"] = safe(np.log(np.maximum(C[:, 9], 1e-9))
                                - np.log(np.maximum(C[:, 0], 1e-9)), np.log(10.0))
        F[f"local_elast_{nm}"] = e_lo
        F[f"deep_elast_{nm}"] = e_hi
        F[f"marg_elast_{nm}"] = safe(np.log(np.maximum(C[:, 1], 1e-9))
                                     - np.log(np.maximum(C[:, 0], 1e-9)), np.log(2.0))
    F["elast_asym"] = F["elast_b"] - F["elast_a"]
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="何日おきに処理するか")
    # ★l2/bbo が DEEP_ARCHIVE で読めない銘柄は build_bbo_l1.py で組み直した板を使う。
    #   既定は空文字なので、これまでの呼び出しは何も変わらない。
    ap.add_argument("--bbo-suffix", default="", help="bbo_<coin><suffix>.parquet を読む")
    ap.add_argument("--out-suffix", default="", help="featlib_<coin><suffix>/ へ書く")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = DATA / f"l1_{tag}"
    files = sorted(src.glob("dt=*.parquet"))
    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}{a.bbo_suffix}.parquet"))
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "px", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    bdays = set(bb["dt"].unique().to_list())
    files = [f for f in files if f.stem.split("=")[1] in bdays][::a.stride]
    if a.days:
        files = files[: a.days]
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 除外) / L1 {len(files)} 日", flush=True)

    out = DATA / f"featlib_{tag}{a.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64})
    prev_ofi = None                # 前日の OFI の分布(ofi_pct の基準)
    for kf, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        if d.height < 1000:
            continue
        keep = {"step": GRID_NS and int(10 ** 9 // GRID_NS), "gi": [], "qb": [],
                "qa": [], "ok": []}
        X, xb, l1ok, midg, good, nxt, meta = ladder_day(fp, d, carry, keep=keep)
        carry = nxt
        gi = np.concatenate(keep["gi"])
        Qb = np.concatenate(keep["qb"])
        Qa = np.concatenate(keep["qa"])
        okl = np.concatenate(keep["ok"])
        sec = np.clip(gi // (10 ** 9 // GRID_NS), 0, NG - 1)
        QB = np.full((NG, NLV), np.nan, np.float32)
        QA = np.full((NG, NLV), np.nan, np.float32)
        QB[sec] = np.where(okl[:, None], Qb, np.nan)
        QA[sec] = np.where(okl[:, None], Qa, np.nan)
        del keep, gi, Qb, Qa

        ts = d["ts"].cast(pl.Int64).to_numpy()
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        tg = d0 + np.arange(NG, dtype=np.int64) * 10 ** 9
        pb = d["best_bid"].to_numpy()
        pa = d["best_ask"].to_numpy()
        qb1 = d["bid_sz"].to_numpy()
        qa1 = d["ask_sz"].to_numpy()
        gidx = np.searchsorted(ts, tg, side="right") - 1
        okg = gidx >= 0
        gi_s = np.maximum(gidx, 0)
        F = {}
        # --- 1. Price / BBO -------------------------------------------------
        F["best_bid"] = np.where(okg, pb[gi_s], np.nan)
        F["best_ask"] = np.where(okg, pa[gi_s], np.nan)
        mid = 0.5 * (F["best_bid"] + F["best_ask"])
        F["mid"] = mid
        F["log_mid"] = np.log(mid)
        F["spread_abs"] = F["best_ask"] - F["best_bid"]
        tick = np.where(mid >= 1000.0, 0.1, 0.01)
        F["tick"] = tick
        tick_bp = tick / mid * 1e4
        F["spread_tick"] = F["spread_abs"] / tick
        F["spread_bp"] = F["spread_abs"] / mid * 1e4
        F["spread_rel"] = F["spread_abs"] / mid
        for nm, v in (("bid", F["best_bid"]), ("ask", F["best_ask"]),
                      ("mid", mid), ("spread", F["spread_abs"])):
            F[f"d_{nm}"] = np.diff(v, prepend=v[0])
        F["t_since_mid_move"] = tsince(np.abs(F["d_mid"]) > 1e-12)
        F["t_since_bbo_chg"] = tsince((np.abs(F["d_bid"]) > 1e-12)
                                      | (np.abs(F["d_ask"]) > 1e-12))
        F["bbo_age"] = F["t_since_bbo_chg"]
        F["spread_duration"] = tsince(np.abs(F["d_spread"]) > 1e-12)

        # --- 2〜7 (板の再構成から) -------------------------------------------
        F.update(ladder_feats(QB, QA, tick_bp, mid, F["spread_bp"]))

        # --- 8. Order Flow Imbalance ---------------------------------------
        e = np.concatenate([[0.0], (
            (pb[1:] >= pb[:-1]) * qb1[1:] - (pb[1:] <= pb[:-1]) * qb1[:-1]
            - ((pa[1:] <= pa[:-1]) * qa1[1:] - (pa[1:] >= pa[:-1]) * qa1[:-1]))])
        cofi = np.cumsum(e)
        cofi_g = np.where(okg, cofi[gi_s], np.nan)
        for w in OFI_WIN:
            j = np.searchsorted(ts, tg - int(w * 1e9), side="right") - 1
            F[f"ofi_{w:g}s"] = cofi_g - np.where(j >= 0, cofi[np.maximum(j, 0)],
                                                 np.nan)
        of1 = F["ofi_1s"]
        F["ofi_dollar"] = of1 * mid
        F["ofi_norm"] = safe(of1, F["depth_bbo"], np.nan)
        cnt = np.concatenate([[0], np.ones(ts.size - 1)]).cumsum()
        cg = np.where(okg, cnt[gi_s], np.nan)
        j1 = np.searchsorted(ts, tg - 10 ** 9, side="right") - 1
        nev = cg - np.where(j1 >= 0, cnt[np.maximum(j1, 0)], np.nan)
        F["ofi_per_event"] = safe(of1, nev, np.nan)
        F["ofi_per_time"] = of1 / 1.0
        F["ofi_cum"] = cofi_g
        al = 2.0 / (10.0 + 1.0)
        ew = np.zeros(NG)
        v = np.nan_to_num(of1)
        for i in range(1, NG):
            ew[i] = al * v[i] + (1 - al) * ew[i - 1]
        F["ofi_ewma"] = ew
        F["d_ofi"] = np.diff(of1, prepend=of1[0])
        F["ofi_vel"] = F["d_ofi"] / GRID_S
        F["ofi_acc"] = np.diff(F["ofi_vel"], prepend=F["ofi_vel"][0])
        F["ofi_sign"] = np.sign(of1)
        F["ofi_persist"] = tsince(np.concatenate(
            [[True], np.sign(of1[1:]) != np.sign(of1[:-1])]))
        F["ofi_streak"] = streak(np.sign(np.nan_to_num(of1)))
        F["ofi_reversal"] = (np.sign(of1) != np.sign(
            np.concatenate([[of1[0]], of1[:-1]]))).astype(float)
        F["ofi_z"] = zsc(np.nan_to_num(of1), 300)
        F["ofi_shock"] = (np.abs(F["ofi_z"]) > 3).astype(float)
        # ★ここは 2026-09-05 まで「その日全体での順位」だった。
        #   格子点 T の値を、T より後の値まで含めた分布で順位づけていたので
        #   時間契約(x が確定する時刻 <= y の期間の開始時刻)に反する。
        #   前日の分布に対する順位に直した。初日は分布が無いので NaN。
        if prev_ofi is None:
            F["ofi_pct"] = np.full(NG, np.nan)
        else:
            F["ofi_pct"] = (np.searchsorted(prev_ofi, np.nan_to_num(of1),
                                            side="right") / prev_ofi.size)
        prev_ofi = np.sort(np.nan_to_num(of1))
        # 多水準 OFI: 累積深さの変化を符号つきで
        for K in (2, 3, 5, 10):
            db = np.diff(F[f"cum_b{K}"], prepend=np.nan)
            da = np.diff(F[f"cum_a{K}"], prepend=np.nan)
            F[f"ofi_lv{K}"] = db - da

        # --- 9〜11. Limit order flow / cancel / arrival ---------------------
        ev = pl.read_parquet(fp, columns=["ts", "side", "px", "status",
                                          "orig_sz", "remaining_sz", "tif",
                                          "is_trigger"]).filter(
            (~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
            & pl.col("status").is_in(["open"] + TERMINAL))
        et = ev["ts"].cast(pl.Int64).to_numpy()
        # ★2026-09-05 修正: 以前は「その秒に起きたイベント」を格子点 T へ入れて
        #   いたので x が [T, T+1)、y が (T, T+h] となり 1 秒ぶん重なっていた
        #   (h=1s では丸ごと同時点)。格子点 T には [T-1s, T) だけを入れる。
        #   `es_raw` は「イベントが起きた秒」で、最良気配との突合にだけ使う。
        es_raw = np.clip((et - d0) // 10 ** 9, 0, NG - 1)
        es = np.clip(es_raw + 1, 0, NG - 1)
        eb = (ev["side"].to_numpy() == "B")
        epx = ev["px"].to_numpy()
        est = ev["status"].to_numpy()
        eo = ev["orig_sz"].to_numpy()
        er = ev["remaining_sz"].to_numpy()
        is_add = est == "open"
        is_can = np.isin(est, CANCELS)
        is_exe = est == "filled"
        at_bbo = (eb & (np.abs(epx - F["best_bid"][es_raw]) < 1e-9)) \
            | ((~eb) & (np.abs(epx - F["best_ask"][es_raw]) < 1e-9))
        vol = np.where(is_add, eo, er)
        for nm, m in (("add", is_add), ("cancel", is_can), ("exec", is_exe)):
            for sn, sm in (("b", eb), ("a", ~eb)):
                v_ = np.zeros(NG)
                c_ = np.zeros(NG)
                sel = m & sm
                np.add.at(v_, es[sel], vol[sel])
                np.add.at(c_, es[sel], 1.0)
                F[f"{nm}_vol_{sn}"] = v_
                F[f"{nm}_cnt_{sn}"] = c_
            b_, a_ = F[f"{nm}_vol_b"], F[f"{nm}_vol_a"]
            F[f"{nm}_imb"] = safe(b_ - a_, b_ + a_)
        for sn, sm in (("b", eb), ("a", ~eb)):
            for tag2, mm in (("bbo", at_bbo), ("deep", ~at_bbo)):
                v_ = np.zeros(NG)
                np.add.at(v_, es[is_can & sm & mm], vol[is_can & sm & mm])
                F[f"cancel_{tag2}_{sn}"] = v_
                v2 = np.zeros(NG)
                np.add.at(v2, es[is_add & sm & mm], vol[is_add & sm & mm])
                F[f"add_{tag2}_{sn}"] = v2
        F["net_lof"] = (F["add_vol_b"] - F["cancel_vol_b"]) \
            - (F["add_vol_a"] - F["cancel_vol_a"])
        F["repl_b"] = F["add_vol_b"]
        F["repl_a"] = F["add_vol_a"]
        F["depl_b"] = F["cancel_vol_b"] + F["exec_vol_b"]
        F["depl_a"] = F["cancel_vol_a"] + F["exec_vol_a"]
        F["repl_depl_ratio"] = safe(F["repl_b"] + F["repl_a"],
                                    F["depl_b"] + F["depl_a"], np.nan)
        F["add_cancel_ratio"] = safe(F["add_vol_b"] + F["add_vol_a"],
                                     F["cancel_vol_b"] + F["cancel_vol_a"], np.nan)
        F["cancel_add_ratio"] = safe(F["cancel_vol_b"] + F["cancel_vol_a"],
                                     F["add_vol_b"] + F["add_vol_a"], np.nan)
        cv = F["cancel_vol_b"] + F["cancel_vol_a"]
        cc = F["cancel_cnt_b"] + F["cancel_cnt_a"]
        F["cancel_intensity"] = rsum(cc, 10) / 10.0
        F["cancel_int_b"] = rsum(F["cancel_cnt_b"], 10) / 10.0
        F["cancel_int_a"] = rsum(F["cancel_cnt_a"], 10) / 10.0
        F["cancel_rate"] = safe(cv, F["depth_cum10"], np.nan)
        F["cancel_to_depth"] = safe(cv, F["depth_bbo"], np.nan)
        F["cancel_z"] = zsc(cv, 300)
        F["cancel_burst"] = (F["cancel_z"] > 3).astype(float)
        F["cancel_shock_b"] = zsc(F["cancel_vol_b"], 300)
        F["cancel_shock_a"] = zsc(F["cancel_vol_a"], 300)
        F["bbo_cancel_rate"] = safe(F["cancel_bbo_b"] + F["cancel_bbo_a"], cv, np.nan)
        F["deep_cancel_rate"] = safe(F["cancel_deep_b"] + F["cancel_deep_a"], cv,
                                     np.nan)
        F["cancel_persist"] = tsince(cv > 0)
        F["t_since_cancel"] = F["cancel_persist"]
        av = F["add_vol_b"] + F["add_vol_a"]
        ac = F["add_cnt_b"] + F["add_cnt_a"]
        F["add_intensity"] = rsum(ac, 10) / 10.0
        F["bbo_add_int"] = rsum(F["add_bbo_b"] + F["add_bbo_a"], 10) / 10.0
        F["deep_add_int"] = rsum(F["add_deep_b"] + F["add_deep_a"], 10) / 10.0
        F["repl_intensity"] = rsum(av, 10) / 10.0
        F["arrival_acc"] = np.diff(F["add_intensity"], prepend=F["add_intensity"][0])
        F["arrival_burst"] = (zsc(ac, 300) > 3).astype(float)

        # --- 12. Aggressive Trade Flow --------------------------------------
        fd = FID.get(dt)
        if fd is not None:
            ft = fd["ts"].cast(pl.Int64).to_numpy()
            fsz = fd["sz"].to_numpy()
            fpx = fd["px"].to_numpy()
            fbuy = fd["side"].to_numpy() == "B"
            fs_ = np.clip((ft - d0) // 10 ** 9 + 1, 0, NG - 1)   # ★同上
            for nm, m in (("buy", fbuy), ("sell", ~fbuy)):
                v_ = np.zeros(NG)
                c_ = np.zeros(NG)
                np.add.at(v_, fs_[m], fsz[m])
                np.add.at(c_, fs_[m], 1.0)
                F[f"trade_{nm}_vol"] = v_
                F[f"trade_{nm}_cnt"] = c_
            dv = np.zeros(NG)
            np.add.at(dv, fs_, fsz * fpx * np.where(fbuy, 1.0, -1.0))
            F["trade_dollar_imb"] = dv
        else:
            for c in ("trade_buy_vol", "trade_sell_vol", "trade_buy_cnt",
                      "trade_sell_cnt", "trade_dollar_imb"):
                F[c] = np.zeros(NG)
        bvl, svl = F["trade_buy_vol"], F["trade_sell_vol"]
        bct, sct = F["trade_buy_cnt"], F["trade_sell_cnt"]
        F["trade_imb"] = safe(bvl - svl, bvl + svl)
        F["signed_vol"] = bvl - svl
        F["signed_cnt"] = bct - sct
        F["aggr_buy_int"] = rsum(bct, 10) / 10.0
        F["aggr_sell_int"] = rsum(sct, 10) / 10.0
        F["aggr_imb"] = safe(rsum(bvl, 10) - rsum(svl, 10),
                             rsum(bvl, 10) + rsum(svl, 10))
        F["aggr_mom"] = rsum(F["signed_vol"], 10)
        F["aggr_acc"] = np.diff(F["aggr_mom"], prepend=F["aggr_mom"][0])
        F["aggr_shock"] = (np.abs(zsc(F["signed_vol"], 300)) > 3).astype(float)
        F["size_wgt_imb"] = safe(bvl * bvl - svl * svl, bvl * bvl + svl * svl)

        # --- 深さの動き (3 の残り) ------------------------------------------
        F["d_depth"] = np.diff(F["depth_bbo"], prepend=F["depth_bbo"][0])
        F["depth_vol"] = rstd(np.nan_to_num(F["depth_bbo"]), 60)
        F["depl_rate"] = safe(F["depl_b"] + F["depl_a"], F["depth_cum10"], np.nan)
        F["repl_rate"] = safe(F["repl_b"] + F["repl_a"], F["depth_cum10"], np.nan)

        # --- microprice の動き (4 の残り) ------------------------------------
        m1 = F["micro1"]
        F["d_micro"] = np.diff(m1, prepend=m1[0])
        F["micro_ret"] = np.diff(np.log(m1), prepend=0.0) * 1e4
        F["micro_mom"] = rsum(np.nan_to_num(F["micro_ret"]), 10)
        F["micro_persist"] = tsince(np.concatenate(
            [[True], np.sign(F["micro_ret"][1:]) != np.sign(F["micro_ret"][:-1])]))
        F["micro_minus_bid"] = (m1 - F["best_bid"]) / mid * 1e4
        F["micro_minus_ask"] = (m1 - F["best_ask"]) / mid * 1e4

        # --- 目的変数 (説明変数には混ぜない) --------------------------------
        for h in FWD:
            j = np.minimum(np.arange(NG) + int(h), NG - 1)
            F[f"fwd_mid_{h:g}s"] = (np.log(mid[j]) - np.log(mid)) * 1e4
            F[f"fwd_micro_{h:g}s"] = (np.log(m1[j]) - np.log(m1)) * 1e4
            F[f"fwd_mid_{h:g}s"][-int(h):] = np.nan
            F[f"fwd_micro_{h:g}s"][-int(h):] = np.nan

        sl = slice(0, NG, KEEP_EVERY)
        T = pl.DataFrame({"dt": [dt] * len(range(0, NG, KEEP_EVERY)),
                          "sec": np.arange(0, NG, KEEP_EVERY, dtype=np.int32),
                          **{k: np.asarray(v, dtype=np.float32)[sl]
                             for k, v in F.items()}})
        T.write_parquet(out / f"dt={dt}.parquet")
        if (kf + 1) % 5 == 0 or kf == 0:
            print(f"  [{kf+1}/{len(files)}] {dt}  特徴量 {len(F)} 個 "
                  f"→ {T.height:,} 行", flush=True)
    print(f"書き出し {out}  ({len(list(out.glob('dt=*.parquet')))} 日)")


if __name__ == "__main__":
    main()
