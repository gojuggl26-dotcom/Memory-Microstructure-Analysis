"""L4 板データから、分類表に沿った特徴量パネルを 100ms 格子で作る。

    uv run python scripts/build_l4feat.py --coin xyz:INTC [--start 0 --n 99]

既存の `build_featlib.py`(1 秒格子・229 本)との違い
--------------------------------------------------
  * 格子を **100ms** にした。100ms / 500ms のホライズンは 1 秒格子では
    そもそも表現できない。目的変数は 100ms・500ms・1s・5s・10s・30s・60s の 7 本
  * 板を **数量だけでなく本数・サイズの二乗和・対数和・発注時刻の和**でも
    再構成する。これで水準ごとの「平均サイズ・ばらつき・年齢」が出る
  * 水準を 10 → **20** に伸ばした。板の形・空白・仮想的な成行の衝撃に要る

板の再構成 — なぜ 5 本の階段を同時に作れるのか
---------------------------------------------
注文 1 本の属性を `f(size)` の形に書ければ、その水準での合計は

    Σ_orders f(size)

であり、イベントごとの差分は `f(size_after) - f(size_before)` になる。
つまり**数量と同じ仕掛け(価格 × 格子の 2 次元での累積和)がそのまま使える**。

    f(x) = x                → 数量        L
    f(x) = 1{x>0}           → 本数        C
    f(x) = x^2              → サイズ二乗和 S2   → 平均・分散・変動係数
    f(x) = ln(x)·1{x>0}     → 対数和      SL   → 対数サイズの平均
    f(x) = t_open·1{x>0}    → 発注時刻の和 T1   → 本数重みの平均年齢
    f(x) = t_open^2·1{x>0}  → 同上の二乗和 T2   → 年齢のばらつき
    f(x) = x·t_open         → 数量重みの和 TL   → 数量重みの平均年齢

`t_open` は**その日の 00:00 UTC からの秒**で持つ。ns のまま二乗すると
3e36 になって float64 の桁が飛び、年齢の分散が桁ごと壊れる。

時間契約(★これを壊したら結果は全部捨てる)
------------------------------------------
格子点 T の説明変数は **ts < T のイベントだけ**で作る。

  * 板の階段は `cumsum(...)[:-1]` で 1 つずらしてあるので、格子 g の深さに
    g 番目の区間 [T, T+100ms) のイベントは入らない
  * 気配は `searchsorted(..., "left") - 1` で厳密に T 未満の最後の行
  * フローの窓は `[T-w, T)` — **区間 g そのものを含めない**。
    含めると h=100ms の目的変数と丸ごと重なる(`build_featlib` で実際に踏んだ)
  * 前日基準の分位以外、標準化・分位のパラメータを全標本から作らない
  * `label_` で始まる列だけが未来を使う

出力
----
  E:/Memory-l4feat/<coin>/dt=YYYY-MM-DD.parquet   … 1 秒おきの行(86,400 行/日)
  data/l4feat_cols_<coin>.csv                     … 列 → 分類表の対応

100ms 格子で計算し、書き出しだけ 1 秒おきに間引く。目的変数は間引く前の
100ms 格子で作るので、100ms・500ms のホライズンは正確に出る。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (BIG_PX, CANCELS, GRID_NS, MARGIN,  # noqa: E402
                              PX_UNIT, RESTING, SZ_LOT, TERMINAL,
                              clean_bbo, grid_mid)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTROOT = Path("E:/Memory-l4feat")

DAY_NS = 86_400_000_000_000
NG = DAY_NS // GRID_NS                 # 864,000 点/日
NLV = 20                               # 板の水準(最良から 20 ティック)
OUT_STEP = 10                          # 書き出しは 1 秒おき
MAX_G = 4_000                          # かたまりの格子点上限
CELL_CAP = 2.5e6                       # かたまりの (格子点 × 価格帯) 上限

# 後ろ向き窓(100ms 単位)
WIN = [1, 5, 10, 20, 50, 100, 300, 600]
WLAB = ["100ms", "500ms", "1s", "2s", "5s", "10s", "30s", "60s"]
# 予測ホライズン(100ms 単位)
FWD = [1, 5, 10, 50, 100, 300, 600]
FLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]

NFEAT_WIN = ["1s", "10s", "60s"]       # フローのうち全指標を作る窓(それ以外は主要だけ)

# 列 -> 分類表の節。レポートで棚卸しするために持つ
FAM: dict[str, str] = {}


def reg(name: str, fam: str) -> str:
    FAM[name] = fam
    return name


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------
def cum0(x: np.ndarray) -> np.ndarray:
    """先頭に 0 を足した累積和。区間和を引き算で出すため。"""
    c = np.empty(x.size + 1, np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    return c


def back_sum(c: np.ndarray, idx: np.ndarray, w: int) -> np.ndarray:
    """格子点 idx について、区間 [idx-w, idx) の和。★idx 自身は入れない。"""
    return c[idx] - c[np.maximum(idx - w, 0)]


def safe(a, b, fill=np.nan):
    b = np.asarray(b, np.float64)
    ok = np.abs(b) > 1e-12
    return np.where(ok, np.asarray(a, np.float64) / np.where(ok, b, 1.0), fill)


def tsince(flag: np.ndarray, dt: float) -> np.ndarray:
    """直近で True になってからの経過秒(その点自身は含めない)。"""
    n = flag.size
    idx = np.where(flag, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, (np.arange(n) - last) * dt, 1e6)


def chunks(bbi, bai, good, ng):
    """価格が動く日でも (格子点 × 価格帯) が上限を超えないように区切る。"""
    out, g = [0], 0
    while g < ng:
        e = min(g + MAX_G, ng)
        while e - g > 100:
            gg = good[g:e]
            if not gg.any():
                break
            w = (max(bbi[g:e][gg].max(), bai[g:e][gg].max())
                 - min(bbi[g:e][gg].min(), bai[g:e][gg].min()) + 2 * MARGIN + 1)
            if w * (e - g) <= CELL_CAP:
                break
            e = g + (e - g) // 2
        out.append(e)
        g = e
    return np.array(out, np.int64)


# --------------------------------------------------------------------------
# 1 日分のイベントを「7 本の階段の差分」に直す
# --------------------------------------------------------------------------
NACC = 7                     # L, C, S2, SL, T1, T2, TL


def day_events(fp: Path, carry: pl.DataFrame, t0: int):
    """(gi, pidx, is_bid, delta[NACC]) と、翌日へ渡す carry を返す。"""
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status",
                                     "remaining_sz", "tif", "is_trigger"])
    n_raw = d.height
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS)))
    del d
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")),
                          synth=pl.lit(False),
                          topen=pl.when(pl.col("rk") == 0)
                          .then((pl.col("ts") - t0) / 1e9)
                          .otherwise(None).cast(pl.Float64))
          .drop("sz", "gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx", "size_after", "topen",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(
        prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
        last=pl.col("oid") != pl.col("oid").shift(-1),
        # 発注時刻は oid ごとに前方へ埋める。前日から来た注文は carry の値。
        topen=pl.col("topen").forward_fill().over("oid"))
    # 窓の外で置かれ carry も無い注文(孤児)。年齢は「日の始まり」に丸める
    n_orph = int(ev["topen"].is_null().sum())
    ev = ev.with_columns(topen=pl.col("topen").fill_null(0.0))

    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after",
                   topen=pl.col("topen") - 86400.0))     # 翌日の原点へ直す

    ev = (ev.filter((~pl.col("synth"))
                    & (pl.col("size_after") != pl.col("prev")))
          .with_columns(gi=((pl.col("ts") - t0) // GRID_NS)
                        .clip(0, NG - 1).cast(pl.Int32))
          .select("gi", "pidx", "is_bid", "size_after", "prev", "topen", "rk",
                  "ts"))
    return ev, nxt, n_raw, n_orph


def deltas(ev: pl.DataFrame) -> np.ndarray:
    """(n, NACC) の差分行列。列は L, C, S2, SL, T1, T2, TL。"""
    a = ev["size_after"].to_numpy().astype(np.float64)
    p = ev["prev"].to_numpy().astype(np.float64)
    to = ev["topen"].to_numpy()
    oa, op = (a > 0), (p > 0)
    D = np.empty((a.size, NACC), np.float64)
    D[:, 0] = a - p
    D[:, 1] = oa.astype(np.float64) - op
    D[:, 2] = a * a - p * p
    D[:, 3] = np.where(oa, np.log(np.maximum(a, 1.0)), 0.0) \
        - np.where(op, np.log(np.maximum(p, 1.0)), 0.0)
    D[:, 4] = to * D[:, 1]
    D[:, 5] = to * to * D[:, 1]
    D[:, 6] = to * D[:, 0]
    return D


# --------------------------------------------------------------------------
# 板の階段を組んで、出力格子点だけ水準別に取り出す
# --------------------------------------------------------------------------
def ladder(ev: pl.DataFrame, D: np.ndarray, carry: pl.DataFrame,
           bbi, bai, good, oidx, t0):
    """出力格子点 oidx における (2, NLV, NACC) の水準別集計を返す。

    戻り値 Z[side, level, acc] の形は (nout, 2, NLV, NACC)。side 0=買い 1=売り。
    """
    nout = oidx.size
    Z = np.full((nout, 2, NLV, NACC), np.nan, np.float64)
    if not ev.height:
        return Z, 0
    gi = ev["gi"].to_numpy().astype(np.int64)
    pi = ev["pidx"].to_numpy().astype(np.int64)
    bm = ev["is_bid"].to_numpy()
    o = np.argsort(gi, kind="stable")
    gi, pi, bm, D = gi[o], pi[o], bm[o], D[o]

    pmin = int(pi.min()) - MARGIN
    pmax = int(pi.max()) + MARGIN
    if carry.height:
        pmin = min(pmin, int(carry["pidx"].min()) - MARGIN)
        pmax = max(pmax, int(carry["pidx"].max()) + MARGIN)
    # ★価格の窓 [lo, hi] は**最良気配**から決まるので、注文が一度も出ていない
    #   価格帯まで伸びうる。base をそこまで確保しないと足し込みの形が合わない
    #   (2026-07-10 で実際に (500,4057) += (2557,) の破断が出た)
    if good.any():
        pmin = min(pmin, int(min(bbi[good].min(), bai[good].min())) - MARGIN)
        pmax = max(pmax, int(max(bbi[good].max(), bai[good].max())) + MARGIN)
    P = pmax - pmin + 1
    base = np.zeros((2, P, NACC), np.float64)
    if carry.height:
        cb = carry["is_bid"].to_numpy()
        cp = carry["pidx"].to_numpy().astype(np.int64) - pmin
        cs = carry["size_after"].to_numpy().astype(np.float64)
        ct = carry["topen"].to_numpy()
        CD = np.stack([cs, np.ones_like(cs), cs * cs, np.log(np.maximum(cs, 1.0)),
                       ct, ct * ct, ct * cs], axis=1)
        for s, m in ((0, cb), (1, ~cb)):
            np.add.at(base[s], cp[m], CD[m])

    bnd = chunks(bbi, bai, good, NG)
    sidx = np.searchsorted(gi, bnd)
    n_neg = 0
    for ci in range(len(bnd) - 1):
        g0, g1 = int(bnd[ci]), int(bnd[ci + 1])
        G = g1 - g0
        s0, s1 = int(sidx[ci]), int(sidx[ci + 1])
        cg, cp, cb, cd = gi[s0:s1], pi[s0:s1], bm[s0:s1], D[s0:s1]
        gg = good[g0:g1]
        # このかたまりに入る出力格子点
        lo_o = np.searchsorted(oidx, g0)
        hi_o = np.searchsorted(oidx, g1)
        if gg.any() and hi_o > lo_o:
            lo = int(min(bbi[g0:g1][gg].min(), bai[g0:g1][gg].min())) - MARGIN
            hi = int(max(bbi[g0:g1][gg].max(), bai[g0:g1][gg].max())) + MARGIN
            W = hi - lo + 1
            inw = (cp >= lo) & (cp <= hi)
            fl = (cg[inw] - g0) * W + (cp[inw] - lo)
            ib, dw = cb[inw], cd[inw]
            rows = oidx[lo_o:hi_o] - g0            # かたまり内の相対位置
            okc = gg[rows].copy()
            lb = np.where(bbi[oidx[lo_o:hi_o]] >= BIG_PX, 10, 1)
            la = np.where(bai[oidx[lo_o:hi_o]] >= BIG_PX, 10, 1)
            jb0 = bbi[oidx[lo_o:hi_o]] - lo
            ja0 = bai[oidx[lo_o:hi_o]] - lo
            for s, msk in ((0, ib), (1, ~ib)):
                M = np.empty((G, W, NACC), np.float64)
                for k in range(NACC):
                    h = np.bincount(fl[msk], weights=dw[msk, k],
                                    minlength=G * W).reshape(G, W)
                    np.cumsum(h, axis=0, out=h)
                    M[1:, :, k] = h[:-1]
                    M[0, :, k] = 0.0
                    M[:, :, k] += base[s, lo - pmin: hi - pmin + 1, k]
                n_neg += int((M[:, :, 0] < 0).sum())
                np.maximum(M[:, :, 0], 0.0, out=M[:, :, 0])
                np.maximum(M[:, :, 1], 0.0, out=M[:, :, 1])
                r = np.arange(rows.size)
                for l in range(NLV):
                    j = (jb0 - l * lb) if s == 0 else (ja0 + l * la)
                    o2 = okc & (j >= 0) & (j < W)
                    if o2.any():
                        Z[lo_o + r[o2], s, l] = M[rows[o2], j[o2]]
                del M
        if s1 > s0:
            for s, msk in ((0, cb), (1, ~cb)):
                np.add.at(base[s], cp[msk] - pmin, cd[msk])
    return Z, n_neg


# --------------------------------------------------------------------------
# 板の状態から特徴量を作る(分類 1〜6・18〜21・24)
# --------------------------------------------------------------------------
def book_feats(Z, mid, bidp, askp, tickpx, T_sec, F):
    """Z: (nout, 2, NLV, NACC)。F に列を足す。

    ★水準 l の価格は**最良気配から** l ティック外側。中値から数えると
    スプレッドのぶんだけずれて、板の形がスプレッドの影になる。
    """
    L = Z[:, :, :, 0] * SZ_LOT                     # 数量(枚)
    C = Z[:, :, :, 1]                              # 本数
    S2 = Z[:, :, :, 2] * SZ_LOT ** 2
    SL = Z[:, :, :, 3]
    T1, T2, TL = Z[:, :, :, 4], Z[:, :, :, 5], Z[:, :, :, 6] * SZ_LOT
    lv = np.arange(NLV)
    px_b = bidp[:, None] - lv[None, :] * tickpx[:, None]
    px_a = askp[:, None] + lv[None, :] * tickpx[:, None]
    # 中値からの距離(bp)。水準 1 の距離はスプレッドの半分になる
    dist_b = (mid[:, None] - px_b) / mid[:, None] * 1e4
    dist_a = (px_a - mid[:, None]) / mid[:, None] * 1e4
    dist = np.stack([dist_b, dist_a], axis=1)      # (nout, 2, NLV)
    dist = np.maximum(dist, 1e-6)

    # --- 3.1 水準ごと -----------------------------------------------------
    for l in (0, 1, 2, 4):
        F[reg(f"qb_l{l+1}", "3.1 水準別の厚み")] = L[:, 0, l]
        F[reg(f"qa_l{l+1}", "3.1 水準別の厚み")] = L[:, 1, l]
        F[reg(f"nb_l{l+1}", "3.1 水準別の厚み")] = C[:, 0, l]
        F[reg(f"na_l{l+1}", "3.1 水準別の厚み")] = C[:, 1, l]
    for s, tag in ((0, "b"), (1, "a")):
        m1 = safe(L[:, s, 0], C[:, s, 0])
        m2 = safe(S2[:, s, 0], C[:, s, 0])
        F[reg(f"avgsz_{tag}1", "3.1 水準別の厚み")] = m1
        F[reg(f"sdsz_{tag}1", "3.1 水準別の厚み")] = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
        F[reg(f"cvsz_{tag}1", "3.1 水準別の厚み")] = safe(
            np.sqrt(np.maximum(m2 - m1 ** 2, 0)), m1)
        F[reg(f"logsz_{tag}1", "3.1 水準別の厚み")] = safe(SL[:, s, 0], C[:, s, 0])

    # --- 3.2 累積 / 3.3 想定元本 -----------------------------------------
    cL = np.cumsum(L, axis=2)
    cC = np.cumsum(C, axis=2)
    px = np.stack([px_b, px_a], axis=1)
    cN = np.cumsum(L * px, axis=2)
    for K in (1, 2, 3, 5, 10, 20):
        k = K - 1
        F[reg(f"cdb{K}", "3.2 累積の厚み")] = cL[:, 0, k]
        F[reg(f"cda{K}", "3.2 累積の厚み")] = cL[:, 1, k]
        F[reg(f"cdtot{K}", "3.2 累積の厚み")] = cL[:, 0, k] + cL[:, 1, k]
        # --- 4 板の偏り OBI ---
        F[reg(f"obi{K}", "4 板の偏り OBI")] = safe(
            cL[:, 0, k] - cL[:, 1, k], cL[:, 0, k] + cL[:, 1, k])
        F[reg(f"obi_cnt{K}", "4 板の偏り OBI")] = safe(
            cC[:, 0, k] - cC[:, 1, k], cC[:, 0, k] + cC[:, 1, k])
        F[reg(f"obi_not{K}", "4 板の偏り OBI")] = safe(
            cN[:, 0, k] - cN[:, 1, k], cN[:, 0, k] + cN[:, 1, k])
    F[reg("ndb", "3.3 想定元本の厚み")] = cN[:, 0, -1]
    F[reg("nda", "3.3 想定元本の厚み")] = cN[:, 1, -1]
    F[reg("ndtot", "3.3 想定元本の厚み")] = cN[:, 0, -1] + cN[:, 1, -1]

    # --- 4.3 重みつき OBI -------------------------------------------------
    for nm, w in (("dw", 1.0 / dist), ("iw", 1.0 / dist ** 2),
                  ("ew", np.exp(-dist / 5.0))):
        wb = (L[:, 0] * w[:, 0]).sum(1)
        wa = (L[:, 1] * w[:, 1]).sum(1)
        F[reg(f"obi_{nm}", "4 板の偏り OBI")] = safe(wb - wa, wb + wa)
    ab = safe(L[:, 0], C[:, 0], 0.0).mean(1)
    aa = safe(L[:, 1], C[:, 1], 0.0).mean(1)
    F[reg("obi_avgsz", "4 板の偏り OBI")] = safe(ab - aa, ab + aa)
    # 年齢重み(古い流動性ほど重い)
    age_b = np.maximum(T_sec[:, None] - safe(T1[:, 0], C[:, 0], 0.0), 0.0)
    age_a = np.maximum(T_sec[:, None] - safe(T1[:, 1], C[:, 1], 0.0), 0.0)
    wb = (L[:, 0] * np.log1p(age_b)).sum(1)
    wa = (L[:, 1] * np.log1p(age_a)).sum(1)
    F[reg("obi_age", "4 板の偏り OBI")] = safe(wb - wa, wb + wa)

    # --- 13 注文の年齢 ----------------------------------------------------
    F[reg("age_b1", "13 注文の年齢")] = age_b[:, 0]
    F[reg("age_a1", "13 注文の年齢")] = age_a[:, 0]
    F[reg("age_asym", "13 注文の年齢")] = age_b[:, 0] - age_a[:, 0]
    agl_b = np.maximum(T_sec - safe(TL[:, 0, 0], L[:, 0, 0], 0.0), 0.0)
    agl_a = np.maximum(T_sec - safe(TL[:, 1, 0], L[:, 1, 0], 0.0), 0.0)
    F[reg("agelot_b1", "13 注文の年齢")] = agl_b
    F[reg("agelot_a1", "13 注文の年齢")] = agl_a
    v_b = np.maximum(safe(T2[:, 0, 0], C[:, 0, 0])
                     - safe(T1[:, 0, 0], C[:, 0, 0]) ** 2, 0.0)
    v_a = np.maximum(safe(T2[:, 1, 0], C[:, 1, 0])
                     - safe(T1[:, 1, 0], C[:, 1, 0]) ** 2, 0.0)
    F[reg("agesd_b1", "13 注文の年齢")] = np.sqrt(v_b)
    F[reg("agesd_a1", "13 注文の年齢")] = np.sqrt(v_a)
    wgt = np.where(np.isfinite(L), L, 0.0)
    F[reg("age_book_b", "13 注文の年齢")] = safe(
        (wgt[:, 0] * np.nan_to_num(age_b)).sum(1), wgt[:, 0].sum(1))
    F[reg("age_book_a", "13 注文の年齢")] = safe(
        (wgt[:, 1] * np.nan_to_num(age_a)).sum(1), wgt[:, 1].sum(1))

    # --- 5 板の形 ---------------------------------------------------------
    for s, tag in ((0, "b"), (1, "a")):
        d = dist[:, s]
        cd = cL[:, s]
        x, y = np.log(d), np.log(np.maximum(cd, 1e-9))
        for nm, xx, yy, k0, k1 in (("slope", d, cd, 0, NLV),
                                   ("lslope", x, y, 0, NLV)):
            X = xx[:, k0:k1]
            Y = yy[:, k0:k1]
            mx = X.mean(1, keepdims=True)
            my = Y.mean(1, keepdims=True)
            F[reg(f"{nm}_{tag}", "5 板の形")] = safe(
                ((X - mx) * (Y - my)).sum(1), ((X - mx) ** 2).sum(1))
        # 曲率 = 2 階差分の平均
        F[reg(f"curv_{tag}", "5 板の形")] = np.diff(cd, n=2, axis=1).mean(1)
        # 集中度
        tot = cd[:, -1]
        F[reg(f"top1_{tag}", "5 板の形")] = safe(L[:, s, 0], tot)
        F[reg(f"top3_{tag}", "5 板の形")] = safe(cd[:, 2], tot)
        F[reg(f"top5_{tag}", "5 板の形")] = safe(cd[:, 4], tot)
        p = safe(L[:, s], tot[:, None], 0.0)
        F[reg(f"hhi_{tag}", "5 板の形")] = (p ** 2).sum(1)
        F[reg(f"ent_{tag}", "5 板の形")] = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
        F[reg(f"com_{tag}", "5 板の形")] = safe((L[:, s] * d).sum(1), tot)
        F[reg(f"disp_{tag}", "5 板の形")] = np.sqrt(np.maximum(
            safe((L[:, s] * d ** 2).sum(1), tot)
            - safe((L[:, s] * d).sum(1), tot) ** 2, 0))
        # --- 19 板の空白 ---
        emp = (L[:, s] <= 0)
        F[reg(f"nempty_{tag}", "19 板の空白")] = emp.sum(1)
        first = np.where(emp.any(1), emp.argmax(1), NLV)
        F[reg(f"gap1_{tag}", "19 板の空白")] = first.astype(np.float64)
        # 連続する空白の最大長
        run = np.zeros(emp.shape[0])
        cur = np.zeros(emp.shape[0])
        for l in range(NLV):
            cur = np.where(emp[:, l], cur + 1, 0)
            run = np.maximum(run, cur)
        F[reg(f"gapmax_{tag}", "19 板の空白")] = run
        # --- 18 板の圧力 ---
        for a in (0.5, 1.0, 2.0):
            F[reg(f"prs{a:g}_{tag}", "18 板の圧力")] = (L[:, s] / d ** a).sum(1)
        # --- 24 弾力性 = d ln(累積) / d ln(距離) ---
        F[reg(f"elast_{tag}", "24 板の弾力性")] = safe(
            np.log(np.maximum(cd[:, 9], 1e-9)) - np.log(np.maximum(cd[:, 1], 1e-9)),
            np.log(d[:, 9]) - np.log(d[:, 1]))
    for nm in ("slope", "lslope", "curv", "hhi", "ent", "com", "disp"):
        F[reg(f"{nm}_asym", "5 板の形")] = F[f"{nm}_b"] - F[f"{nm}_a"]
    F[reg("prs_net", "18 板の圧力")] = F["prs1_b"] - F["prs1_a"]
    F[reg("prs_imb", "18 板の圧力")] = safe(F["prs1_b"] - F["prs1_a"],
                                            F["prs1_b"] + F["prs1_a"])
    F[reg("elast_asym", "24 板の弾力性")] = F["elast_b"] - F["elast_a"]
    F[reg("gap_asym", "19 板の空白")] = F["gap1_b"] - F["gap1_a"]

    # --- 20/21 仮想的な成行の衝撃 -----------------------------------------
    for s, tag in ((0, "b"), (1, "a")):
        sh = np.nan_to_num(L[:, s])                    # 各水準の枚数
        csh = np.cumsum(sh, axis=1)
        cno = np.cumsum(sh * px[:, s], axis=1)
        for N in (10_000.0, 50_000.0, 200_000.0):
            k = (cno < N).sum(1)                       # 使い切る水準の数
            full = k >= NLV
            k = np.minimum(k, NLV - 1)
            prev_n = np.where(k > 0, cno[np.arange(k.size), k - 1], 0.0)
            prev_s = np.where(k > 0, csh[np.arange(k.size), k - 1], 0.0)
            pk = px[np.arange(k.size), s, k]
            need = np.maximum(N - prev_n, 0.0)
            sh_tot = prev_s + safe(need, pk, 0.0)
            vwap = safe(N, sh_tot)
            slip = (vwap - mid) / mid * 1e4 * (1 if s == 1 else -1)
            lab = f"{int(N/1000)}k"
            F[reg(f"slip{lab}_{tag}", "20/21 仮想的な成行の衝撃")] = np.where(full, np.nan, slip)
            F[reg(f"lvl{lab}_{tag}", "20/21 仮想的な成行の衝撃")] = np.where(full, np.nan, k + 1.0)
    for lab in ("10k", "50k", "200k"):
        F[reg(f"slip{lab}_rt", "20/21 仮想的な成行の衝撃")] = \
            F[f"slip{lab}_b"] + F[f"slip{lab}_a"]
        F[reg(f"slip{lab}_asym", "20/21 仮想的な成行の衝撃")] = \
            F[f"slip{lab}_a"] - F[f"slip{lab}_b"]
    F[reg("impact_convex", "20/21 仮想的な成行の衝撃")] = safe(
        F["slip200k_rt"], F["slip50k_rt"]) - safe(F["slip50k_rt"], F["slip10k_rt"])
    return L, C, dist, px_b, px_a


# --------------------------------------------------------------------------
# 1 日分
# --------------------------------------------------------------------------
def build_day(fp: Path, bb: pl.DataFrame, fills: pl.DataFrame,
              carry: pl.DataFrame, qprev: dict | None):
    t0 = int(pl.read_parquet(fp, columns=["ts"])["ts"].cast(pl.Int64).min()
             // DAY_NS) * DAY_NS
    ev, nxt, n_raw, n_orph = day_events(fp, carry, t0)
    D = deltas(ev)
    bb2, n_drop = clean_bbo(bb)
    mid_g, bbi, bai, good, qb1, qa1 = grid_mid(bb2, t0, NG)
    oidx = np.arange(0, NG, OUT_STEP)
    nout = oidx.size
    T_sec = oidx * (GRID_NS / 1e9)

    # ---- 目的変数は 100ms 格子で作る ------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = qb1 + qa1
        micro_g = np.where(good & (sb > 0),
                           (bbi * PX_UNIT * qa1 + bai * PX_UNIT * qb1)
                           / np.where(sb > 0, sb, 1.0), np.nan)
    lm_g = np.log(np.where(good, mid_g, np.nan))
    lmi_g = np.log(np.where(np.isfinite(micro_g) & (micro_g > 0), micro_g, np.nan))

    F: dict[str, np.ndarray] = {}
    tickpx = np.where(bbi[oidx] >= BIG_PX, 0.1, 0.01)

    # ---- 1 価格と最良気配 ------------------------------------------------
    bidp = np.where(good, bbi * PX_UNIT, np.nan)[oidx]
    askp = np.where(good, bai * PX_UNIT, np.nan)[oidx]
    mid = mid_g[oidx]
    F[reg("best_bid", "1.1 最良気配")] = bidp
    F[reg("best_ask", "1.1 最良気配")] = askp
    F[reg("bbo_qb", "1.1 最良気配")] = qb1[oidx]
    F[reg("bbo_qa", "1.1 最良気配")] = qa1[oidx]
    F[reg("mid", "1.2 中値")] = mid
    F[reg("log_mid", "1.2 中値")] = lm_g[oidx]
    sp = askp - bidp
    F[reg("spread", "1.3 スプレッド")] = sp
    F[reg("spread_tick", "1.3 スプレッド")] = sp / tickpx
    F[reg("spread_bp", "1.3 スプレッド")] = sp / mid * 1e4
    F[reg("log_spread", "1.3 スプレッド")] = np.log(np.maximum(sp, 1e-9))
    F[reg("is_1tick", "1.3 スプレッド")] = (np.abs(sp / tickpx - 1) < 0.01).astype(float)

    # 変化(後ろ向き差分)
    for k, lab in zip(FWD, FLAB):
        F[reg(f"ret_{lab}", "1.2 中値")] = (lm_g[oidx] - lm_g[np.maximum(oidx - k, 0)]) * 1e4
        F[reg(f"mret_{lab}", "2.1 マイクロプライス")] = \
            (lmi_g[oidx] - lmi_g[np.maximum(oidx - k, 0)]) * 1e4
    F[reg("d_mid", "1.2 中値")] = F["ret_100ms"]
    F[reg("abs_d_mid", "1.2 中値")] = np.abs(F["ret_100ms"])
    F[reg("sgn_d_mid", "1.2 中値")] = np.sign(F["ret_100ms"])
    dsp_g = np.concatenate([[np.nan], np.diff(np.where(good, (bai - bbi) * PX_UNIT, np.nan))])
    F[reg("d_spread", "1.3 スプレッド")] = dsp_g[oidx]
    F[reg("sp_widen", "1.3 スプレッド")] = (dsp_g[oidx] > 0).astype(float)
    F[reg("sp_narrow", "1.3 スプレッド")] = (dsp_g[oidx] < 0).astype(float)
    ch_sp = np.concatenate([[True], np.diff(bai - bbi) != 0])
    ch_bbo = np.concatenate([[True], (np.diff(bbi) != 0) | (np.diff(bai) != 0)])
    ch_mid = np.concatenate([[True], np.diff(mid_g) != 0])
    F[reg("t_since_sp", "1.3 スプレッド")] = tsince(ch_sp, GRID_NS / 1e9)[oidx]
    F[reg("t_since_bbo", "43 経過時間")] = tsince(ch_bbo, GRID_NS / 1e9)[oidx]
    F[reg("t_since_mid", "43 経過時間")] = tsince(ch_mid, GRID_NS / 1e9)[oidx]
    F[reg("spread_dur", "1.3 スプレッド")] = F["t_since_sp"]

    # ---- 2 マイクロプライス ---------------------------------------------
    micro = micro_g[oidx]
    F[reg("micro", "2.1 マイクロプライス")] = micro
    F[reg("delta_bp", "2.1 マイクロプライス")] = (micro - mid) / mid * 1e4
    F[reg("delta_norm", "2.1 マイクロプライス")] = safe(micro - mid, sp / 2.0)
    F[reg("micro_bid", "2.1 マイクロプライス")] = (micro - bidp) / mid * 1e4
    F[reg("ask_micro", "2.1 マイクロプライス")] = (askp - micro) / mid * 1e4

    # ---- 板の階段 --------------------------------------------------------
    Z, n_neg = ladder(ev, D, carry, bbi, bai, good, oidx, t0)
    L, C, dist, px_b, px_a = book_feats(Z, mid, bidp, askp, tickpx, T_sec, F)

    # 2.2 / 2.3 多水準・キューを見たマイクロプライス
    def mp(wb, wa, name, fam):
        sb_ = np.nansum(wb, 1)
        sa_ = np.nansum(wa, 1)
        pb_ = safe(np.nansum(wb * px_b, 1), sb_)
        pa_ = safe(np.nansum(wa * px_a, 1), sa_)
        v = safe(pa_ * sb_ + pb_ * sa_, sb_ + sa_)
        F[reg(name, fam)] = (v - mid) / mid * 1e4
    for K in (2, 5, 20):
        w = np.zeros(NLV); w[:K] = 1.0
        mp(L[:, 0] * w, L[:, 1] * w, f"mp_lv{K}", "2.2 多水準のマイクロプライス")
    mp(L[:, 0] / dist[:, 0], L[:, 1] / dist[:, 1], "mp_invd",
       "2.2 多水準のマイクロプライス")
    mp(L[:, 0] * np.exp(-dist[:, 0] / 5), L[:, 1] * np.exp(-dist[:, 1] / 5),
       "mp_exp", "2.2 多水準のマイクロプライス")
    mp(C[:, 0], C[:, 1], "mp_cnt", "2.3 キューを見たマイクロプライス")
    age_b = np.maximum(T_sec[:, None] - safe(Z[:, 0, :, 4], Z[:, 0, :, 1], 0.0), 0)
    age_a = np.maximum(T_sec[:, None] - safe(Z[:, 1, :, 4], Z[:, 1, :, 1], 0.0), 0)
    mp(L[:, 0] * np.log1p(age_b), L[:, 1] * np.log1p(age_a), "mp_age",
       "2.3 キューを見たマイクロプライス")
    mp(L[:, 0] * np.exp(-age_b / 10.0), L[:, 1] * np.exp(-age_a / 10.0),
       "mp_fresh", "2.3 キューを見たマイクロプライス")
    mp(L[:, 0] * (1 - np.exp(-age_b / 10.0)), L[:, 1] * (1 - np.exp(-age_a / 10.0)),
       "mp_persist", "2.3 キューを見たマイクロプライス")
    F[reg("mp_mom", "2.1 マイクロプライス")] = F["delta_bp"] - np.concatenate(
        [[np.nan] * 10, F["delta_bp"][:-10]])
    del Z

    # ---- フロー(分類 7・8・10・15・28・42・44) -------------------------
    flow_feats(ev, bbi, bai, good, oidx, F, ch_mid)

    # ---- 約定(分類 51・52・22・23・49) ---------------------------------
    trade_feats(fills, t0, oidx, mid_g, F)

    # ---- 47 ボラティリティ ----------------------------------------------
    r = np.diff(lm_g, prepend=np.nan) * 1e4
    r2 = np.nan_to_num(r ** 2)
    cr2 = cum0(r2)
    ar = np.nan_to_num(np.abs(r))
    cbp = cum0(ar * np.concatenate([[0.0], ar[:-1]]))
    cup = cum0(np.where(np.nan_to_num(r) > 0, r2, 0.0))
    cdn = cum0(np.where(np.nan_to_num(r) < 0, r2, 0.0))
    cq = cum0(r2 ** 2)
    for w, lab in zip(WIN[2:], WLAB[2:]):
        F[reg(f"rv_{lab}", "47 ボラティリティ")] = np.sqrt(back_sum(cr2, oidx, w))
        F[reg(f"bpv_{lab}", "47 ボラティリティ")] = np.sqrt(
            back_sum(cbp, oidx, w) * np.pi / 2)
        F[reg(f"rsv_up_{lab}", "47 ボラティリティ")] = np.sqrt(back_sum(cup, oidx, w))
        F[reg(f"rsv_dn_{lab}", "47 ボラティリティ")] = np.sqrt(back_sum(cdn, oidx, w))
        F[reg(f"rq_{lab}", "47 ボラティリティ")] = back_sum(cq, oidx, w) ** 0.25
        F[reg(f"jump_{lab}", "47 ボラティリティ")] = safe(
            F[f"rv_{lab}"] - F[f"bpv_{lab}"], F[f"rv_{lab}"])
    rm = np.diff(lmi_g, prepend=np.nan) * 1e4
    cm2 = cum0(np.nan_to_num(rm ** 2))
    csp = cum0(np.nan_to_num(np.diff(np.where(good, (bai - bbi) * PX_UNIT, np.nan),
                                     prepend=np.nan) ** 2))
    for w, lab in zip((100, 600), ("10s", "60s")):
        F[reg(f"rv_micro_{lab}", "47 ボラティリティ")] = np.sqrt(back_sum(cm2, oidx, w))
        F[reg(f"rv_spread_{lab}", "47 ボラティリティ")] = np.sqrt(back_sum(csp, oidx, w))

    return F, nxt, mid_g, micro_g, dict(
        n_raw=n_raw, n_orphan=n_orph, n_neg=n_neg,
        n_bbo_drop=n_drop, n_grid_ok=int(good.sum()))


def flow_feats(ev, bbi, bai, good, oidx, F, ch_mid):
    """分類 7 到着 / 8 除去 / 10 OFI / 15 回転 / 28 メッセージ / 42 事象時間 / 44 群発。"""
    gi = ev["gi"].to_numpy().astype(np.int64)
    pidx = ev["pidx"].to_numpy().astype(np.int64)
    isb = ev["is_bid"].to_numpy()
    rk = ev["rk"].to_numpy()
    a = ev["size_after"].to_numpy().astype(np.float64)
    p = ev["prev"].to_numpy().astype(np.float64)
    ts = ev["ts"].to_numpy()
    lots = (a - p)                                  # 正 = 増える、負 = 減る
    isnew = rk == 0
    isfil = rk == 2
    iscxl = rk == 3
    # 事象の時点での最良気配(格子は 100ms、板と同じく「その区間の直前」)
    bb_e = bbi[gi]
    ba_e = bai[gi]
    ok_e = good[gi]
    tickr = np.where(bb_e >= BIG_PX, 10, 1)
    d_tick = np.where(isb, (bb_e - pidx), (pidx - ba_e)) / np.maximum(tickr, 1)
    d_tick = np.where(ok_e, d_tick, np.nan)

    def bin_(mask, val=None):
        if not mask.any():
            return np.zeros(NG)
        w = np.ones(mask.sum()) if val is None else val[mask]
        return np.bincount(gi[mask], weights=w, minlength=NG).astype(np.float64)

    add = np.maximum(lots, 0) * SZ_LOT
    rem = np.maximum(-lots, 0) * SZ_LOT
    band = np.nan_to_num(d_tick, nan=1e9) <= 10      # 最良から 10 ティック以内
    touch = np.nan_to_num(d_tick, nan=1e9) <= 0      # 最良かその内側
    series = {
        "new_b": bin_(isnew & isb), "new_a": bin_(isnew & ~isb),
        "newsz_b": bin_(isnew & isb, add), "newsz_a": bin_(isnew & ~isb, add),
        "cxl_b": bin_(iscxl & isb), "cxl_a": bin_(iscxl & ~isb),
        "cxlsz_b": bin_(iscxl & isb, rem), "cxlsz_a": bin_(iscxl & ~isb, rem),
        "fil_b": bin_(isfil & isb), "fil_a": bin_(isfil & ~isb),
        "filsz_b": bin_(isfil & isb, rem), "filsz_a": bin_(isfil & ~isb, rem),
        "newt_b": bin_(isnew & isb & touch), "newt_a": bin_(isnew & ~isb & touch),
        "cxlt_b": bin_(iscxl & isb & touch), "cxlt_a": bin_(iscxl & ~isb & touch),
        "newd_b": bin_(isnew & isb & ~band), "newd_a": bin_(isnew & ~isb & ~band),
        "addsz_b": bin_(isb & band, add), "addsz_a": bin_(~isb & band, add),
        "remsz_b": bin_(isb & band, rem), "remsz_a": bin_(~isb & band, rem),
        "dist_sum": bin_(isnew & band, np.nan_to_num(d_tick)),
        "dist_n": bin_(isnew & band),
        "dist_sq": bin_(isnew & band, np.nan_to_num(d_tick) ** 2),
        "msz1": bin_(isnew, add), "msz2": bin_(isnew, add ** 2),
        "msz3": bin_(isnew, add ** 3), "msz4": bin_(isnew, add ** 4),
        "mszl": bin_(isnew, np.log(np.maximum(add, 1e-6))),
        "msz_ent": bin_(isnew, -add * np.log(np.maximum(add, 1e-6))),
        "nev": bin_(np.ones(gi.size, bool)),
    }
    cum = {k: cum0(v) for k, v in series.items()}
    ns = 0
    for w, lab in zip(WIN, WLAB):
        g = {k: back_sum(c, oidx, w) for k, c in cum.items()}
        dtw = w * GRID_NS / 1e9
        # 7.1 到着の強度
        F[reg(f"new_rate_{lab}", "7 指値の到着")] = (g["new_b"] + g["new_a"]) / dtw
        F[reg(f"new_imb_{lab}", "7 指値の到着")] = safe(g["new_b"] - g["new_a"],
                                                       g["new_b"] + g["new_a"], 0.0)
        F[reg(f"newsz_imb_{lab}", "7 指値の到着")] = safe(
            g["newsz_b"] - g["newsz_a"], g["newsz_b"] + g["newsz_a"], 0.0)
        # 8 除去(取消)
        F[reg(f"cxl_rate_{lab}", "8 取消と除去")] = (g["cxl_b"] + g["cxl_a"]) / dtw
        F[reg(f"cxl_imb_{lab}", "8 取消と除去")] = safe(g["cxl_b"] - g["cxl_a"],
                                                       g["cxl_b"] + g["cxl_a"], 0.0)
        F[reg(f"cxlsz_imb_{lab}", "8 取消と除去")] = safe(
            g["cxlsz_b"] - g["cxlsz_a"], g["cxlsz_b"] + g["cxlsz_a"], 0.0)
        F[reg(f"car_{lab}", "8 取消と除去")] = safe(
            g["cxl_b"] + g["cxl_a"], g["new_b"] + g["new_a"], np.nan)
        # 10 OFI(帯域内の増減の差)
        ofi = (g["addsz_b"] - g["remsz_b"]) - (g["addsz_a"] - g["remsz_a"])
        F[reg(f"ofi_{lab}", "10 注文フローの偏り OFI")] = ofi
        F[reg(f"ofi_norm_{lab}", "10 注文フローの偏り OFI")] = safe(
            ofi, g["addsz_b"] + g["remsz_b"] + g["addsz_a"] + g["remsz_a"], 0.0)
        F[reg(f"ofi_new_{lab}", "10 注文フローの偏り OFI")] = g["newsz_b"] - g["newsz_a"]
        F[reg(f"ofi_cxl_{lab}", "10 注文フローの偏り OFI")] = g["cxlsz_a"] - g["cxlsz_b"]
        F[reg(f"ofi_fil_{lab}", "10 注文フローの偏り OFI")] = g["filsz_a"] - g["filsz_b"]
        # 15 流動性の回転
        F[reg(f"churn_{lab}", "15 流動性の回転")] = (
            g["addsz_b"] + g["addsz_a"] + g["remsz_b"] + g["remsz_a"])
        F[reg(f"churn_imb_{lab}", "15 流動性の回転")] = safe(
            (g["addsz_b"] + g["remsz_b"]) - (g["addsz_a"] + g["remsz_a"]),
            g["addsz_b"] + g["remsz_b"] + g["addsz_a"] + g["remsz_a"], 0.0)
        # 28 メッセージ
        F[reg(f"msg_rate_{lab}", "28 メッセージの流量")] = g["nev"] / dtw
        if lab in NFEAT_WIN:
            # 7.3 発注の距離
            n = g["dist_n"]
            m1 = safe(g["dist_sum"], n)
            F[reg(f"pdist_mean_{lab}", "7 指値の到着")] = m1
            F[reg(f"pdist_sd_{lab}", "7 指値の到着")] = np.sqrt(np.maximum(
                safe(g["dist_sq"], n) - m1 ** 2, 0))
            F[reg(f"touch_share_{lab}", "7 指値の到着")] = safe(
                g["newt_b"] + g["newt_a"], g["new_b"] + g["new_a"], np.nan)
            F[reg(f"deep_share_{lab}", "7 指値の到着")] = safe(
                g["newd_b"] + g["newd_a"], g["new_b"] + g["new_a"], np.nan)
            F[reg(f"cxl_touch_{lab}", "8 取消と除去")] = safe(
                g["cxlt_b"] + g["cxlt_a"], g["cxl_b"] + g["cxl_a"], np.nan)
            # 6 サイズの分布(到着したサイズのモーメント)
            nn = g["new_b"] + g["new_a"]
            s1 = safe(g["msz1"], nn)
            s2 = safe(g["msz2"], nn)
            s3 = safe(g["msz3"], nn)
            s4 = safe(g["msz4"], nn)
            var = np.maximum(s2 - s1 ** 2, 1e-12)
            F[reg(f"osz_mean_{lab}", "6 注文サイズの分布")] = s1
            F[reg(f"osz_cv_{lab}", "6 注文サイズの分布")] = np.sqrt(var) / np.maximum(s1, 1e-9)
            F[reg(f"osz_skew_{lab}", "6 注文サイズの分布")] = (
                s3 - 3 * s1 * var - s1 ** 3) / var ** 1.5
            F[reg(f"osz_kurt_{lab}", "6 注文サイズの分布")] = (
                s4 - 4 * s1 * s3 + 6 * s1 ** 2 * s2 - 3 * s1 ** 4) / var ** 2
            F[reg(f"osz_logmean_{lab}", "6 注文サイズの分布")] = safe(g["mszl"], nn)
            F[reg(f"osz_hhi_{lab}", "6 注文サイズの分布")] = safe(
                g["msz2"], g["msz1"] ** 2)
            F[reg(f"osz_ent_{lab}", "6 注文サイズの分布")] = safe(
                g["msz_ent"], g["msz1"]) + np.log(np.maximum(g["msz1"], 1e-9))
        ns += 1
    # 多尺度(速い − 遅い)
    F[reg("ofi_fast_slow", "67 多尺度")] = F["ofi_1s"] - F["ofi_30s"] / 30.0
    F[reg("new_fast_slow", "67 多尺度")] = F["new_rate_1s"] - F["new_rate_30s"]
    F[reg("cxl_fast_slow", "67 多尺度")] = F["cxl_rate_1s"] - F["cxl_rate_30s"]

    # 42 事象時間 — 直近の中値の変化からいくつ事象が起きたか
    #    ★「直近の変化」は過去の情報だけで決まる(その点自身は含めない)
    ii = np.where(ch_mid, np.arange(NG), -1)
    lastm = np.maximum.accumulate(ii)
    lastm = np.maximum(lastm, 0)
    lm_o = lastm[oidx]
    for nm, key in (("ev", "nev"), ("new", "new_b"), ("cxl", "cxl_b"),
                    ("fil", "fil_b")):
        if nm == "ev":
            v = cum["nev"][oidx] - cum["nev"][lm_o]
        else:
            v = ((cum[f"{key}"][oidx] - cum[f"{key}"][lm_o])
                 + (cum[key.replace('_b', '_a')][oidx]
                    - cum[key.replace('_b', '_a')][lm_o]))
        F[reg(f"since_mid_{nm}", "42 事象時間")] = v
    F[reg("since_mid_ofi", "42 事象時間")] = (
        (cum["addsz_b"][oidx] - cum["addsz_b"][lm_o])
        - (cum["remsz_b"][oidx] - cum["remsz_b"][lm_o])
        - (cum["addsz_a"][oidx] - cum["addsz_a"][lm_o])
        + (cum["remsz_a"][oidx] - cum["remsz_a"][lm_o]))
    F[reg("since_mid_add", "42 事象時間")] = (
        cum["addsz_b"][oidx] - cum["addsz_b"][lm_o]
        + cum["addsz_a"][oidx] - cum["addsz_a"][lm_o])
    F[reg("since_mid_rem", "42 事象時間")] = (
        cum["remsz_b"][oidx] - cum["remsz_b"][lm_o]
        + cum["remsz_a"][oidx] - cum["remsz_a"][lm_o])

    # 43 経過時間
    for nm, m in (("new", isnew), ("cxl", iscxl), ("fil", isfil)):
        flag = np.zeros(NG, bool)
        if m.any():
            flag[np.unique(gi[m])] = True
        F[reg(f"t_since_{nm}", "43 経過時間")] = tsince(flag, GRID_NS / 1e9)[oidx]

    # 44 群発性(事象間隔の変動係数)
    dts = np.diff(ts, prepend=ts[0] if ts.size else 0) / 1e9
    cdt = cum0(np.nan_to_num(dts))
    cdt2 = cum0(np.nan_to_num(dts ** 2))
    cn = cum0(np.ones(gi.size))
    # 事象の並びを格子に写すため、格子ごとの事象数で割る
    for w, lab in zip((100, 600), ("10s", "60s")):
        gn = back_sum(cum["nev"], oidx, w)
        # 事象の通し番号での窓
        e_end = np.searchsorted(gi, oidx, "left")
        e_beg = np.searchsorted(gi, np.maximum(oidx - w, 0), "left")
        n = np.maximum(e_end - e_beg, 1)
        m1 = (cdt[e_end] - cdt[e_beg]) / n
        m2 = (cdt2[e_end] - cdt2[e_beg]) / n
        sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
        F[reg(f"burst_{lab}", "44 群発性")] = np.where(
            e_end - e_beg >= 5, safe(sd - m1, sd + m1, np.nan), np.nan)
        F[reg(f"idur_{lab}", "44 群発性")] = m1
        F[reg(f"idur_cv_{lab}", "44 群発性")] = safe(sd, m1, np.nan)
        F[reg(f"nev_{lab}", "28 メッセージの流量")] = gn
    _ = cn


def trade_feats(fills: pl.DataFrame, t0: int, oidx, mid_g, F):
    """分類 51 約定の符号 / 52 板と約定 / 23 Kyle / 49 毒性。"""
    zero = np.zeros(NG)
    if fills is None or not fills.height:
        b = a = bv = av = nsq = zero
    else:
        f = fills.filter(pl.col("crossed"))          # テイカー側 = 1 取引 1 行
        g = ((f["ts"].cast(pl.Int64).to_numpy() - t0) // GRID_NS)
        m = (g >= 0) & (g < NG)
        g = g[m].astype(np.int64)
        sz = f["sz"].to_numpy()[m].astype(np.float64)
        px = f["px"].to_numpy()[m].astype(np.float64)
        buy = (f["side"].to_numpy()[m] == "B")
        b = np.bincount(g[buy], minlength=NG).astype(float)
        a = np.bincount(g[~buy], minlength=NG).astype(float)
        bv = np.bincount(g[buy], weights=sz[buy], minlength=NG)
        av = np.bincount(g[~buy], weights=sz[~buy], minlength=NG)
        nsq = np.bincount(g, weights=(sz * px) ** 2, minlength=NG)
    cb, ca, cbv, cav, cq = (cum0(b), cum0(a), cum0(bv), cum0(av), cum0(nsq))
    r = np.diff(np.log(np.where(np.isfinite(mid_g) & (mid_g > 0), mid_g, np.nan)),
                prepend=np.nan) * 1e4
    sv = bv - av
    csv2 = cum0(sv ** 2)
    csvr = cum0(np.nan_to_num(sv * r))
    car = cum0(np.nan_to_num(np.abs(r)))
    for w, lab in zip(WIN[2:], WLAB[2:]):
        nb, na = back_sum(cb, oidx, w), back_sum(ca, oidx, w)
        vb, va = back_sum(cbv, oidx, w), back_sum(cav, oidx, w)
        dtw = w * GRID_NS / 1e9
        F[reg(f"trd_rate_{lab}", "51 約定の符号")] = (nb + na) / dtw
        F[reg(f"trd_imb_{lab}", "51 約定の符号")] = safe(nb - na, nb + na, 0.0)
        F[reg(f"vol_imb_{lab}", "51 約定の符号")] = safe(vb - va, vb + va, 0.0)
        F[reg(f"sgn_vol_{lab}", "51 約定の符号")] = vb - va
        F[reg(f"vol_{lab}", "51 約定の符号")] = vb + va
        F[reg(f"vpin_{lab}", "49 毒性の代理")] = safe(np.abs(vb - va), vb + va, np.nan)
        F[reg(f"trd_sz_{lab}", "51 約定の符号")] = safe(vb + va, nb + na, np.nan)
        F[reg(f"trd_sz2_{lab}", "51 約定の符号")] = np.sqrt(
            safe(back_sum(cq, oidx, w), nb + na, np.nan))
    for w, lab in zip((100, 600), ("10s", "60s")):
        xx = back_sum(csv2, oidx, w)
        xy = back_sum(csvr, oidx, w)
        F[reg(f"kyle_{lab}", "23 Kyle / Amihud")] = safe(xy, xx, np.nan)
        F[reg(f"amihud_{lab}", "23 Kyle / Amihud")] = safe(
            back_sum(car, oidx, w), back_sum(cbv, oidx, w) + back_sum(cav, oidx, w),
            np.nan)
    # 52 板と約定
    for lab in ("1s", "10s", "60s"):
        F[reg(f"vol_over_depth_{lab}", "52 板と約定")] = safe(
            F[f"vol_{lab}"], F["cdtot10"], np.nan)
        F[reg(f"vol_over_bbo_{lab}", "52 板と約定")] = safe(
            F[f"vol_{lab}"], F["bbo_qb"] + F["bbo_qa"], np.nan)
    F[reg("cvd_60s", "51 約定の符号")] = F["sgn_vol_60s"]


# --------------------------------------------------------------------------
# 派生(分類 60 衝撃 / 61 加速 / 62 慣性 / 63 平均回帰 / 70 交互作用)
# --------------------------------------------------------------------------
CORE = ["obi1", "obi5", "obi10", "delta_bp", "spread_bp", "cdtot10", "prs_imb",
        "ofi_1s", "ofi_10s", "cxl_imb_10s", "new_imb_10s", "churn_10s",
        "slope_asym", "elast_asym", "gap_asym", "age_asym", "rv_10s",
        "vol_imb_10s", "top1_b", "hhi_b"]


def derived(F, nout):
    """核となる 20 本に、変化・加速・z 得点・偏差を足す。窓は全て後ろ向き。"""
    W = 300                                        # 5 分 = 300 点(1 秒格子)
    for nm in CORE:
        x = np.asarray(F[nm], np.float64)
        d1 = np.concatenate([[np.nan], np.diff(x)])
        d2 = np.concatenate([[np.nan], np.diff(d1)])
        F[reg(f"{nm}__d", "61 加速と高次の動き")] = d1
        F[reg(f"{nm}__dd", "61 加速と高次の動き")] = d2
        xx = np.nan_to_num(x)
        ok = np.isfinite(x).astype(float)
        c1, c2, cn = cum0(xx), cum0(xx ** 2), cum0(ok)
        i = np.arange(nout)
        n = np.maximum(cn[i] - cn[np.maximum(i - W, 0)], 1)
        m = (c1[i] - c1[np.maximum(i - W, 0)]) / n
        v = np.maximum((c2[i] - c2[np.maximum(i - W, 0)]) / n - m ** 2, 0)
        F[reg(f"{nm}__z", "60 衝撃(z 得点)")] = safe(x - m, np.sqrt(v), np.nan)
        F[reg(f"{nm}__dev", "63 平均回帰")] = x - m
        # 62 慣性: 同符号が続いた長さ
        s = np.sign(np.nan_to_num(d1))
        same = np.concatenate([[False], s[1:] == s[:-1]])
        run = np.zeros(nout)
        cur = 0.0
        for k in range(nout):
            cur = cur + 1 if same[k] else 1.0
            run[k] = cur
        F[reg(f"{nm}__streak", "62 慣性")] = run * s
    # 70 交互作用
    pairs = [("obi5", "spread_bp"), ("obi5", "rv_10s"), ("ofi_10s", "rv_10s"),
             ("ofi_10s", "cdtot10"), ("cxl_imb_10s", "cdtot10"),
             ("delta_bp", "spread_bp"), ("obi5", "gap_asym"),
             ("hhi_b", "ofi_10s"), ("elast_asym", "ofi_10s")]
    for x, y in pairs:
        F[reg(f"{x}__x__{y}", "70 交互作用")] = np.asarray(F[x]) * np.asarray(F[y])
    F[reg("ofi_over_depth", "70 交互作用")] = safe(F["ofi_10s"], F["cdtot10"], np.nan)


# --------------------------------------------------------------------------
def labels(F, oidx, mid_g, micro_g):
    lm = np.log(np.where(np.isfinite(mid_g) & (mid_g > 0), mid_g, np.nan))
    lmi = np.log(np.where(np.isfinite(micro_g) & (micro_g > 0), micro_g, np.nan))
    n = lm.size
    for k, lab in zip(FWD, FLAB):
        j = np.minimum(oidx + k, n - 1)
        F[f"label_micro_{lab}"] = (lmi[j] - lmi[oidx]) * 1e4
        F[f"label_mid_{lab}"] = (lm[j] - lm[oidx]) * 1e4


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=999)
    ap.add_argument("--out", default=str(OUTROOT))
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = DATA / f"l1_{tag}"
    bbo = DATA / f"bbo_l1_{tag}"
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[3:13] for p in src.glob("dt=*.parquet"))
    sel = days[a.start:a.start + a.n]
    fills_all = None
    fp_fills = DATA / f"fills_{tag}.parquet"
    if fp_fills.exists():
        fills_all = pl.read_parquet(fp_fills,
                                    columns=["ts", "px", "sz", "side", "crossed", "dt"])

    # ★状態を跨ぐので、指定した区間の 1 日前から助走する(carry を作るため)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64,
                                 "topen": pl.Float64})
    warm = days[max(0, a.start - 1):a.start]
    for dt in warm + sel:
        t = time.time()
        fp = src / f"dt={dt}.parquet"
        bb = pl.read_parquet(bbo / f"dt={dt}.parquet")
        fdt = fills_all.filter(pl.col("dt") == dt) if fills_all is not None else None
        fo = out / f"dt={dt}.parquet"
        F, nxt, mid_g, micro_g, meta = build_day(fp, bb, fdt, carry, None)
        carry = nxt
        if dt not in sel:
            print(f"  warm {dt}  {time.time()-t:.0f}s", flush=True)
            continue
        oidx = np.arange(0, NG, OUT_STEP)
        derived(F, oidx.size)
        labels(F, oidx, mid_g, micro_g)
        df = pl.DataFrame({"dt": np.full(oidx.size, dt),
                          "sec": (oidx * GRID_NS // 1_000_000_000).astype(np.int32)}
                          | {k: np.asarray(v, np.float32) for k, v in F.items()})
        df.write_parquet(fo, compression="zstd", compression_level=3)
        print(f"  {dt}  {df.width} 列 {df.height} 行  {time.time()-t:.0f}s  "
              f"孤児 {meta['n_orphan']} 負 {meta['n_neg']}", flush=True)
    # 列 -> 分類 の対応表
    if FAM:
        pl.DataFrame({"col": list(FAM), "family": list(FAM.values())}) \
          .write_csv(DATA / f"l4feat_cols_{tag}.csv")


if __name__ == "__main__":
    main()
