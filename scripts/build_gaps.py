"""板の「隙間」13 種を 100ms 格子で作り、OBI / OFI / 板の回復力との関係を測る。

【なぜ隙間を測るか】
板の厚み(depth)は「どこにいくら積まれているか」しか見ない。実際の執行では
**積まれていない価格がどこにどれだけ続くか**が効く。空の価格帯を跨ぐ注文は
その分だけ滑るからである。ここでは占有されている価格水準の並びから
隙間の統計量を作る。

【板の再構成】
build_obi_levels.py の手順をそのまま使う(同じ関数を import している)。
要点だけ再掲する。

    ・ティックは価格で変わる。< 1000 は 0.01 刻み、>= 1000 は 0.1 刻み
    ・数量は 0.001 が最小単位。整数のロット数で足し引きする
      (浮動小数だと空の水準に 1e-13 が残り「深さあり」と読まれる)
    ・部分約定は remaining_sz にそのまま出る
    ・同一 ns の行順は論理順と逆。open(0) → filled(2) → 終端(3) で並べ直す
    ・日を跨いで残る注文は carry として持ち越す(途中の日を飛ばすと板が壊れる)

【13 種の定義】
最良気配から外側へ NOFF ティックぶんを見る。k = 0 が最良気配、k が 1 増えると
その側のティック 1 つぶん外側。占有 = その価格に数量が 1 ロット以上ある。

    (1) bid first gap    最良買いから次に占有された水準までのティック数
    (2) ask first gap    売り側の同じもの
    (3) maximum gap      両側を通して最大の隙間(ティック)
    (4) mean gap         隙間の平均(ティック)
    (5) median gap       隙間の中央値(ティック)
    (6) gap variance     隙間の分散
    (7) gap asymmetry    (買いの平均隙間 − 売りの平均隙間)/(和)   ∈ [−1, +1]
    (8) distance to next occupied bid   (1) を **mid からの bp** で測り直したもの
    (9) distance to next occupied ask   (2) の bp 版
   (10) empty-level count 窓の中で空だった水準の数(両側の合計)
   (11) liquidity void size    最も長く連続した空き(ティック)= (3) − 1
   (12) liquidity void asymmetry (買いの空き − 売りの空き)/(和)
   (13) depth discontinuity    隣り合う占有水準の数量比の |log| の最大

★(1)/(2) と (8)/(9) は同じ現象を別の物差しで測る。前者は**ティック**(板の
機構そのもの)、後者は **mid からの bp**(値段の水準に依らない)。刻みが
価格で変わるので、この 2 つは 1000 を跨ぐ日には別物になる。

【OBI / OFI】
どちらも l2/bbo から作る(板の再構成ではなく、他のレポートと定義を揃える)。

    OBI = (q_bid − q_ask)/(q_bid + q_ask)         … 格子点 T での状態
    OFI = Cont–Kukanov–Stoikov の板流量           … (T−100ms, T] の流量

【板の回復力(order book resilience)】
スプレッドが広がった後、どれだけ元に戻るかで測る。格子点 T で

    s_T   … T のスプレッド(bp)
    s̄_T   … **T より前** 60 秒の中央値(先読みしない)
    ショック e_T = s_T − s̄_T > 0 のときだけ定義し、

    R_k = (s_T − s_{T+k}) / e_T

R = 1 で完全復元、0 で戻らない。**R は (T, T+k] の未来を使う目的変数**であり、
説明変数には決して使わない。

【x が確定する時刻 / y の期間】
    x  = 隙間 13 種 / OBI / OFI  … いずれも格子点 T までの情報だけで決まる
    y  = R_k                      … 期間は (T, T+k]

OBI と OFI との関係は**同時点**(どちらも T)である。予測ではない。
回復力との関係だけが前向きになる。この 2 つを混同しない。

    uv run python scripts/build_gaps.py --coin xyz:MU
出力: data/gaps_<coin>/dt=*.parquet   … 100ms 格子の 13 種 + OBI/OFI/スプレッド
      data/gaps_meta_<coin>.csv       … 日ごとの再構成の質
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (BIG_PX, CANCELS, DAY_NS, GRID_NS,  # noqa: E402
                              PX_UNIT, RESTING, SZ_LOT, TERMINAL, clean_bbo,
                              grid_mid)

ROOT = Path(__file__).resolve().parents[1]

NOFF = 40                        # 外側へ見るティック数
MARGIN_G = NOFF * 10 + 20        # 0.01 単位の余白。0.1 刻みでも NOFF を覆う
MAX_G = 6_000
CELL_CAP = 2.0e7
RESIL_LOOKBACK = 600             # 60 秒 = 600 格子点(スプレッドの平常値)

FEATS = ["bid_first_gap", "ask_first_gap", "max_gap", "mean_gap", "median_gap",
         "gap_var", "gap_asym", "dist_next_bid_bp", "dist_next_ask_bp",
         "empty_count", "void_size", "void_asym", "depth_disc"]


def chunk_bounds(bbi, bai, good, ng):
    """(格子点 × 価格帯) が上限を超えないように区切る(余白が広い版)。"""
    out, g = [0], 0
    while g < ng:
        e = min(g + MAX_G, ng)
        while e - g > 100:
            gg = good[g:e]
            if not gg.any():
                break
            w = (max(bbi[g:e][gg].max(), bai[g:e][gg].max())
                 - min(bbi[g:e][gg].min(), bai[g:e][gg].min()) + 2 * MARGIN_G + 1)
            if w * (e - g) <= CELL_CAP:
                break
            e = g + (e - g) // 2
        out.append(e)
        g = e
    return np.array(out, dtype=np.int64)


def side_stats(Q: np.ndarray, valid: np.ndarray):
    """占有の並びから隙間の統計量を出す。Q は (G, NOFF)、k=0 が最良気配。

    返すのは (first_gap, n_gap, sum_gap, sum_gap2, max_gap, hist, disc)。
    hist[g, d] = 隙間の長さ d が何回あったか(中央値を出すため)。
    """
    G = Q.shape[0]
    occ = (Q > 0) & valid
    rows = np.arange(G)
    first = np.full(G, -1, np.int64)          # 最良より外で最初に占有された k
    run = np.zeros(G, np.int64)               # 直前の占有からの距離
    ngap = np.zeros(G, np.int64)
    sg = np.zeros(G, np.float64)
    sg2 = np.zeros(G, np.float64)
    mx = np.zeros(G, np.int64)
    hist = np.zeros((G, NOFF + 1), np.int32)
    disc = np.zeros(G, np.float64)
    qprev = np.where(occ[:, 0], Q[:, 0], np.nan)
    seen = occ[:, 0].copy()                   # 最良が占有されている行だけ数える
    for k in range(1, NOFF):
        run += 1
        hit = occ[:, k] & seen & valid[:, k]
        if hit.any():
            r = run[hit]
            ngap[hit] += 1
            sg[hit] += r
            sg2[hit] += r.astype(np.float64) ** 2
            mx[hit] = np.maximum(mx[hit], r)
            np.add.at(hist, (rows[hit], r), 1)
            f = hit & (first < 0)
            first[f] = k
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.abs(np.log(Q[hit, k] / qprev[hit]))
            disc[hit] = np.maximum(disc[hit], np.where(np.isfinite(d), d, 0.0))
            qprev[hit] = Q[hit, k]
            run[hit] = 0
    return first, ngap, sg, sg2, mx, hist, disc


def weighted_median(hist: np.ndarray) -> np.ndarray:
    """行ごとの度数分布から中央値を出す。度数 0 の行は nan。"""
    tot = hist.sum(1)
    c = np.cumsum(hist, axis=1)
    half = (tot + 1) // 2
    idx = (c >= half[:, None]).argmax(1).astype(np.float64)
    return np.where(tot > 0, idx, np.nan)


def day_features(fp: Path, bb_day: pl.DataFrame, carry: pl.DataFrame):
    """1 日分の 100ms 格子の特徴量と、翌日へ渡す carry を返す。"""
    d = pl.read_parquet(fp)
    t0 = (int(d["ts"].cast(pl.Int64).min()) // DAY_NS) * DAY_NS
    ng = DAY_NS // GRID_NS

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
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("sz", "gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
                         last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after"))
    ev = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
          .filter((~pl.col("synth")) & (pl.col("delta") != 0))
          .with_columns(gi=((pl.col("ts") - t0) // GRID_NS).clip(0, ng - 1).cast(pl.Int32))
          .select("gi", "pidx", "is_bid", "delta").sort("gi"))
    gi = ev["gi"].to_numpy().astype(np.int64)
    pi = ev["pidx"].to_numpy()
    bmask = ev["is_bid"].to_numpy()
    dv = ev["delta"].to_numpy().astype(np.float64)
    del ev, d

    cpx_all = nxt["pidx"].to_numpy().astype(np.int64) if nxt.height else pi[:1]
    p_min = int(min(pi.min(), cpx_all.min(),
                    carry["pidx"].min() if carry.height else pi.min())) - MARGIN_G
    p_max = int(max(pi.max(), cpx_all.max(),
                    carry["pidx"].max() if carry.height else pi.max())) + MARGIN_G
    dep_b = np.zeros(p_max - p_min + 1, np.float64)
    dep_a = np.zeros(p_max - p_min + 1, np.float64)
    if carry.height:
        cb0 = carry["is_bid"].to_numpy()
        cpx = carry["pidx"].to_numpy().astype(np.int64) - p_min
        csz = carry["size_after"].to_numpy().astype(np.float64)
        np.add.at(dep_b, cpx[cb0], csz[cb0])
        np.add.at(dep_a, cpx[~cb0], csz[~cb0])

    mid, bbi, bai, good, qb1, qa1 = grid_mid(bb_day, t0, ng)
    out = {f: np.full(ng, np.nan) for f in FEATS}
    l0_occ = np.zeros(ng, bool)                 # 最良が再構成でも占有されていたか
    bnd = chunk_bounds(bbi, bai, good, ng)
    idx = np.searchsorted(gi, bnd)

    for ci in range(len(bnd) - 1):
        g0, g1 = int(bnd[ci]), int(bnd[ci + 1])
        G = g1 - g0
        s0, s1 = int(idx[ci]), int(idx[ci + 1])
        cp, cg, cb, cd = pi[s0:s1], gi[s0:s1], bmask[s0:s1], dv[s0:s1]
        gg = good[g0:g1]
        if gg.any():
            lo = int(min(bbi[g0:g1][gg].min(), bai[g0:g1][gg].min())) - MARGIN_G
            hi = int(max(bbi[g0:g1][gg].max(), bai[g0:g1][gg].max())) + MARGIN_G
            W = hi - lo + 1
            inw = (cp >= lo) & (cp <= hi)
            fl = (cg[inw] - g0) * W + (cp[inw] - lo)
            ib, dw = cb[inw], cd[inw]
            zr = np.zeros((1, W))
            Db = dep_b[lo - p_min: hi - p_min + 1] + np.concatenate([zr, np.cumsum(
                np.bincount(fl[ib], weights=dw[ib], minlength=G * W).reshape(G, W),
                axis=0)[:-1]])
            Da = dep_a[lo - p_min: hi - p_min + 1] + np.concatenate([zr, np.cumsum(
                np.bincount(fl[~ib], weights=dw[~ib], minlength=G * W).reshape(G, W),
                axis=0)[:-1]])
            np.maximum(Db, 0.0, out=Db)
            np.maximum(Da, 0.0, out=Da)

            rows = np.arange(G)
            lb = np.where(bbi[g0:g1] >= BIG_PX, 10, 1)
            la = np.where(bai[g0:g1] >= BIG_PX, 10, 1)
            jb0, ja0 = bbi[g0:g1] - lo, bai[g0:g1] - lo
            Qb = np.zeros((G, NOFF)); Qa = np.zeros((G, NOFF))
            vb = np.zeros((G, NOFF), bool); va = np.zeros((G, NOFF), bool)
            for k in range(NOFF):
                jb, ja = jb0 - k * lb, ja0 + k * la
                ob = gg & (jb >= 0); oa = gg & (ja < W)
                Qb[ob, k] = Db[rows[ob], jb[ob]]; vb[ob, k] = True
                Qa[oa, k] = Da[rows[oa], ja[oa]]; va[oa, k] = True
            del Db, Da

            fb, nb, sb, sb2, mb_, hb, db_ = side_stats(Qb, vb)
            fa, na, sa, sa2, ma_, ha, da_ = side_stats(Qa, va)
            l0_occ[g0:g1] = (Qb[:, 0] > 0) & (Qa[:, 0] > 0)

            with np.errstate(invalid="ignore", divide="ignore"):
                mgb = np.where(nb > 0, sb / nb, np.nan)
                mga = np.where(na > 0, sa / na, np.nan)
                n2 = nb + na
                mg = np.where(n2 > 0, (sb + sa) / n2, np.nan)
                gv = np.where(n2 > 1, (sb2 + sa2) / n2 - mg ** 2, np.nan)
                asym = np.where(np.isfinite(mgb + mga) & (mgb + mga > 0),
                                (mgb - mga) / (mgb + mga), np.nan)
                # ★窓の中に「次の占有水準」が 1 つも無い側は隙間が右打ち切りに
                #   なる。0 のまま −1 して void = −1 にしてはいけない(実際に
                #   void_asym が [−1,+1] を飛び出して ±3 になった)。nan にする。
                vb_ = np.where(nb > 0, mb_ - 1.0, np.nan)
                va_ = np.where(na > 0, ma_ - 1.0, np.nan)
                vs = np.fmax(vb_, va_)              # 片側だけ定義でも取れる
                vas = np.where(np.isfinite(vb_ + va_) & (vb_ + va_ > 0),
                               (vb_ - va_) / (vb_ + va_), np.nan)
                md = weighted_median(hb + ha)
                # 空いていた水準の数(最良より外、両側)
                ec = ((~(Qb[:, 1:] > 0)) & vb[:, 1:]).sum(1) + \
                     ((~(Qa[:, 1:] > 0)) & va[:, 1:]).sum(1)
                m_ = mid[g0:g1]
                pxb = (bbi[g0:g1] - fb * lb) * PX_UNIT
                pxa = (bai[g0:g1] + fa * la) * PX_UNIT
                dnb = np.where(fb > 0, (m_ - pxb) / m_ * 1e4, np.nan)
                dna = np.where(fa > 0, (pxa - m_) / m_ * 1e4, np.nan)

            ok = gg & (Qb[:, 0] > 0) & (Qa[:, 0] > 0)
            put = lambda name, v: out[name].__setitem__(
                slice(g0, g1), np.where(ok, v, np.nan))
            put("bid_first_gap", np.where(fb > 0, fb, np.nan))
            put("ask_first_gap", np.where(fa > 0, fa, np.nan))
            put("max_gap", np.fmax(np.where(nb > 0, mb_, np.nan),
                                   np.where(na > 0, ma_, np.nan)))
            put("mean_gap", mg); put("median_gap", md); put("gap_var", gv)
            put("gap_asym", asym); put("dist_next_bid_bp", dnb)
            put("dist_next_ask_bp", dna); put("empty_count", ec.astype(float))
            put("void_size", vs); put("void_asym", vas)
            put("depth_disc", np.maximum(db_, da_))
            del Qb, Qa, vb, va

    # ---- OBI / OFI / スプレッド(l2/bbo から。他のレポートと定義を揃える)----
    with np.errstate(invalid="ignore", divide="ignore"):
        s = qb1 + qa1
        obi = np.where(good & (s > 0), (qb1 - qa1) / s, np.nan)
    pb = np.where(good, bbi * PX_UNIT, np.nan)
    pa = np.where(good, bai * PX_UNIT, np.nan)
    pbp, pap = np.roll(pb, 1), np.roll(pa, 1)
    qbp, qap = np.roll(qb1, 1), np.roll(qa1, 1)
    ofi = (np.where(pb >= pbp, qb1, 0.0) - np.where(pb <= pbp, qbp, 0.0)
           - np.where(pa <= pap, qa1, 0.0) + np.where(pa >= pap, qap, 0.0))
    ofi[0] = np.nan
    ofi = np.where(good & np.roll(good, 1), ofi, np.nan)
    spread_bp = np.where(good, (pa - pb) / mid * 1e4, np.nan)

    df = pl.DataFrame({**{f: out[f] for f in FEATS},
                       "obi": obi, "ofi": ofi, "spread_bp": spread_bp,
                       "logmid": np.log(np.where(mid > 0, mid, np.nan)),
                       "good": good, "l0_occ": l0_occ})
    meta = {"n_grid_good": int(good.sum()), "n_l0_occ": int(l0_occ.sum()),
            "n_feat_ok": int(np.isfinite(out["mean_gap"]).sum()),
            "n_carry_out": nxt.height}
    return df, nxt, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb, ndrop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 信じられない行を {ndrop:,} 除いた -> {bb.height:,} 行", file=sys.stderr)
    days = sorted(p.name.split("=")[1].removesuffix(".parquet")
                  for p in (ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    have = set(bb["dt"].unique().to_list())
    if a.days:
        days = days[: a.days]

    outdir = ROOT / "data" / f"gaps_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64})
    metas = []
    import time
    for i, day in enumerate(days, 1):
        t = time.time()
        fp = ROOT / "data" / f"l1_{tag}" / f"dt={day}.parquet"
        if day not in have:
            print(f"  {day} bbo 無し。飛ばす", file=sys.stderr)
            continue
        df, carry, meta = day_features(fp, bb.filter(pl.col("dt") == day), carry)
        # ★板は前の日から続いているので、書き出しを飛ばしても計算は飛ばせない
        df.write_parquet(outdir / f"dt={day}.parquet")
        metas.append({"dt": day, **meta, "sec": round(time.time() - t, 1)})
        print(f"  {i}/{len(days)} {day} 使える格子 {meta['n_feat_ok']:,} "
              f"繰越 {meta['n_carry_out']:,} {metas[-1]['sec']}s", file=sys.stderr, flush=True)

    pl.DataFrame(metas).write_csv(ROOT / "data" / f"gaps_meta_{tag}.csv")
    print(f"-> {outdir}/ と data/gaps_meta_{tag}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
