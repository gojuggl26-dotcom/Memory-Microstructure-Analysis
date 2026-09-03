"""仮想的な成行注文 Q に対する板の応答を 11 指標で測る(5 秒格子)。

【問い】
「いま Q 枚の成行を出したら、いくらで約定し、板をどこまで削り、そのうち
どれだけが**実際にはもう消えている**か」。前半 6 つは板の形そのもの、後半 5 つは
その板がどれだけ脆いか(あてにできないか)を測る。

【前半 — 板の形から決まる量】b = 売り成行で bid を削る / a = 買い成行で ask を削る

    sweep levels     Q を満たすまでに削る**占有された価格水準の数**
    price impact     Impact(Q) = P_VWAP(Q) − m           …… mid 基準、bp
    slippage         Slip(Q)   = P_VWAP(Q) − p_best      …… 最良気配基準、bp
                     Impact = Slip + ハーフスプレッド の関係にある
    marginal depth   削り終えた価格からさらに 1bp 先までにある数量(契約/bp)
    marginal impact  dImpact/dQ = (p_k − P_VWAP(Q)) / Q  …… bp/契約
                     P_VWAP = C(Q)/Q を Q で微分すると閉じた形になる

【後半 — 板の脆さ】
「見えている板」と「あてにできる板」は別である。板は約定より取消で消えるほうが
桁違いに多い([指値注文の生存率](../reports/MU/mu_hazard_report.md): 最終的に
約定するのは 0.915% だけ、寿命の中央値は 565ms)。そこで消える速さを直接測る。

    fragility        直前 30 秒に mid ±25bp で**取り消された**数量
                     ÷ (30 秒 × 同区間の平均の深さ)                …… 毎秒
                     「あてにしている板が 1 秒あたり何割入れ替わるか」
    fragility imbalance  (bid − ask) / (bid + ask)  ∈ [−1, +1]。正 = 買い板が脆い
    depth-at-risk    D25 × (1 − e^{−f Δ})  …… 契約
    gap-adjusted fragility
                     脆い分が一様に消えたとき、Q を約定させるのに余分に払う額。
                     一様に θ=e^{−fΔ} 倍へ薄くなるのは、元の板で Q/θ を削るのと
                     同じなので
                         GapFrag(Q) = Impact(Q e^{fΔ}) − Impact(Q)   …… bp
                     板に**隙間**があるとここが跳ねる。だから gap-adjusted。

    ★ Δ は 1 秒。f を測る窓(30 秒)とは別物である。e^{−fΔ} は
      「誰も板を足さなかったら」という減衰模型なので、実際には補充が続く
      30 秒に当てると意味を失う(f の中央値 0.22/秒では 30 秒で 403 倍の Q に
      なり、板の窓を突き抜けて 6 割が算出不能になった)。1 秒だけ補充が
      止まったらどうなるか、という**応力試験**として読むこと。

【x が確定する時刻 / y の期間】
すべて格子時刻 T **以前**の情報だけで決まる。板の状態は T 未満のイベントまで
(格子 g の行は g 番目の区間に入るイベントを**適用する前**の状態)、最良気配は
`ts < T` の最後の bbo、脆さは [T−30秒, T) に実際に起きた取消。
将来のリターンとの関係は `build_impact_signal.py` で別に測る。

【板の再構成】
`build_obi_levels.py` の手順をそのまま使う(定数と `clean_bbo` を import)。
要点: ティックは価格で変わる(<1000 は 0.01、>=1000 は 0.1)/ 数量は 0.001 の
整数ロットで足し引きする(浮動小数だと空の水準に残りかすが出て板がクロスする)/
部分約定は remaining_sz に出る / 同一 ns の行順は論理順と逆なので rk で並べ直す /
日を跨ぐ注文は carry で持ち越す。

    uv run python scripts/build_impact.py --coin xyz:MU --days 3
    uv run python scripts/build_impact.py --coin xyz:MU
出力: data/impact_<coin>.parquet   5 秒格子 × 各指標
      data/impact_meta_<coin>.csv  日ごとの検算
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, PX_UNIT, RESTING,  # noqa: E402
                              SZ_LOT, TERMINAL, clean_bbo)

ROOT = Path(__file__).resolve().parents[1]

GRID_NS = 5_000_000_000          # 5 秒
MARGIN_I = 1_200                 # 窓の余白(0.01 単位 = $12。価格 924 で 130bp)
MAX_G = 4_000
CELL_CAP = 5.0e6                 # 1 かたまりの (格子点 × 価格帯) の上限
QS = [0.5, 5.0, 50.0]            # 仮想成行の数量(契約)。実測の成行 1 本の分布の
QTAG = ["q05", "q5", "q50"]      # 中央値 0.511 / p85 / p98 に対応する丸めた値
REGION_BP = 25.0                 # 脆さと depth-at-risk を測る帯(mid から ±bp)
FRAG_S = 30.0                    # 脆さ f を測る後ろ向きの窓(秒)
FRAG_L = int(FRAG_S * 1e9 // GRID_NS)
STRESS_S = 1.0                   # depth-at-risk / gap 調整の減衰時間(秒)
MDEPTH_BP = 1.0                  # marginal depth を測る先の幅(bp)
NAMES = ("lv", "imp", "slp", "mdep", "mimp", "gfr")


def grid_mid(bb: pl.DataFrame, t0: int, ng: int):
    """bbo を格子へ**後ろ向きに**載せる。厳密に t 未満の最後の行を使う。

    `build_obi_levels.grid_mid` と同じ処理だが、あちらは 100ms 固定なので
    こちらの格子幅で作り直している。
    """
    ts = bb["ts"].cast(pl.Int64).to_numpy()
    tg = t0 + np.arange(ng, dtype=np.int64) * GRID_NS
    j = np.searchsorted(ts, tg, side="left") - 1        # ★ t 未満(先読みしない)
    ok = j >= 0
    j = np.where(ok, j, 0)
    pb = bb["best_bid"].to_numpy()[j]
    pa = bb["best_ask"].to_numpy()[j]
    good = ok & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)
    mid = np.where(good, (pb + pa) / 2.0, np.nan)
    bi = np.where(good, np.rint(pb / PX_UNIT), 0).astype(np.int64)
    ai = np.where(good, np.rint(pa / PX_UNIT), 0).astype(np.int64)
    return mid, bi, ai, good, pb, pa


def chunk_bounds(bbi, bai, good, ng):
    """価格が動く日でも (格子点 × 価格帯) が上限を超えないように区切る。"""
    out, g = [0], 0
    while g < ng:
        e = min(g + MAX_G, ng)
        while e - g > 50:
            gg = good[g:e]
            if not gg.any():
                break
            w = (max(bbi[g:e][gg].max(), bai[g:e][gg].max())
                 - min(bbi[g:e][gg].min(), bai[g:e][gg].min()) + 2 * MARGIN_I + 1)
            if w * (e - g) <= CELL_CAP:
                break
            e = g + (e - g) // 2
        out.append(e)
        g = e
    return np.array(out, dtype=np.int64)


def frag_upto(canc, d25, g0, g1):
    """[T−FRAG_S, T) の取消数量 ÷ (FRAG_S × 同区間の平均の深さ)。単位は毎秒。

    格子 g の行に入っている取消は [T_g, T_g+5秒) に起きたものなので、T_g の時点で
    使えるのは g−1 まで。窓は [g−FRAG_L, g−1] を取る。
    """
    c = np.nan_to_num(canc[:g1], nan=0.0)
    cc = np.concatenate([[0.0], np.cumsum(c)])
    d = d25[:g1]
    dc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(d, nan=0.0))])
    nc = np.concatenate([[0.0], np.cumsum(np.isfinite(d).astype(float))])
    g = np.arange(g0, g1)
    lo = g - FRAG_L
    out = np.full(g1 - g0, np.nan)
    m = lo >= 0
    if not m.any():
        return out
    gm, lm = g[m], lo[m]
    den_n = nc[gm] - nc[lm]
    with np.errstate(invalid="ignore", divide="ignore"):
        dbar = np.where(den_n > 0, (dc[gm] - dc[lm]) / np.maximum(den_n, 1), np.nan)
        v = (cc[gm] - cc[lm]) / (FRAG_S * dbar)
    out[m] = np.where(np.isfinite(v) & (dbar > 0), v, np.nan)
    return out


def sweep(D, pxs, sgn, j0, mid, rows):
    """片側のラダーを削る。D は index が増えるほど気配から遠い向きに並べてある。

    D    : (G, W) 深さ(ロット)
    pxs  : (W,)   sgn を掛けて**増加**になるよう向きを揃えた価格
    sgn  : +1 = ask 側(価格は上へ) / −1 = bid 側(価格は下へ)
    j0   : (G,)   各行の最良気配の index
    戻り: (metrics dict, cs, cp)  cs / cp は gap 調整で使い回す
    """
    W = D.shape[1]
    cs = np.cumsum(D, axis=1)
    cp = np.cumsum(D * (sgn * pxs)[None, :], axis=1)
    nc = np.cumsum(D > 0, axis=1).astype(np.float64)
    jm = np.maximum(j0 - 1, 0)
    zb = (j0 > 0).astype(np.float64)
    # 最良気配より内側(本来は空)の分を基準として引く
    cs -= (cs[rows, jm] * zb)[:, None]
    cp -= (cp[rows, jm] * zb)[:, None]
    nc -= (nc[rows, jm] * zb)[:, None]

    out: dict[str, np.ndarray] = {}
    jr = np.clip(np.searchsorted(pxs, sgn * mid * (1 + sgn * REGION_BP * 1e-4)),
                 0, W - 1)
    out["d25"] = cs[rows, jr] * SZ_LOT
    out["_jr"] = jr
    total = cs[:, -1]
    for q, tg in zip(QS, QTAG):
        Q = q / SZ_LOT
        inwin = total >= Q
        k = np.argmax(cs >= Q, axis=1)
        km = np.maximum(k - 1, 0)
        zk = (k > 0).astype(np.float64)
        prev_s = cs[rows, km] * zk
        prev_p = cp[rows, km] * zk
        pk = sgn * pxs[k]
        vwap = (prev_p + pk * (Q - prev_s)) / Q
        j1 = np.clip(np.searchsorted(pxs, sgn * pk * (1 + sgn * MDEPTH_BP * 1e-4)),
                     0, W - 1)
        vals = {
            "lv": nc[rows, k],
            "imp": sgn * (vwap - mid) / mid * 1e4,
            "slp": sgn * (vwap - sgn * pxs[j0]) / mid * 1e4,
            "mdep": (cs[rows, j1] - cs[rows, k]) * SZ_LOT / MDEPTH_BP,
            "mimp": sgn * (pk - vwap) / mid * 1e4 / q,
        }
        for nm, v in vals.items():
            out[f"{nm}_{tg}"] = np.where(inwin, v, np.nan)
        out[f"_ok_{tg}"] = inwin
    return out, cs, cp


def gap_adjust(cs, cp, pxs, sgn, j0, mid, rows, frag, out):
    """脆い分が消えた板で Q を削り直し、余分に払う額を bp で返す。"""
    theta = np.exp(-np.clip(np.nan_to_num(frag, nan=0.0), 0, 10) * STRESS_S)
    for q, tg in zip(QS, QTAG):
        Qp = (q / SZ_LOT) / np.maximum(theta, 1e-3)
        inwin = cs[:, -1] >= Qp
        k = np.argmax(cs >= Qp[:, None], axis=1)
        km = np.maximum(k - 1, 0)
        zk = (k > 0).astype(np.float64)
        pk = sgn * pxs[k]
        vwap = ((cp[rows, km] * zk) + pk * (Qp - cs[rows, km] * zk)) / Qp
        imp2 = sgn * (vwap - mid) / mid * 1e4
        base = out[f"imp_{tg}"]
        out[f"gfr_{tg}"] = np.where(inwin & np.isfinite(base) & np.isfinite(frag),
                                    imp2 - base, np.nan)


def day_features(fp: Path, bb_day: pl.DataFrame, carry: pl.DataFrame):
    """1 日分の 5 秒格子の指標と、翌日へ渡す carry を返す。"""
    d = pl.read_parquet(fp)
    n_raw = d.height
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
    n_open = int((ev["rk"] == 0).sum())
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")),
                          synth=pl.lit(False), canc=pl.col("gone"))
          .drop("sz", "gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True),
                          canc=pl.lit(False))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
                         last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after"))
    n_orph = int((ev["prev"].eq(0) & ev["rk"].gt(0) & ev["size_after"].eq(0)).sum())
    ev = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
          .filter((~pl.col("synth")) & (pl.col("delta") != 0))
          .with_columns(gi=((pl.col("ts") - t0) // GRID_NS).clip(0, ng - 1)
                        .cast(pl.Int32),
                        # 取消で消えた数量(正)。約定で減った分は含めない
                        cq=pl.when(pl.col("canc")).then(-pl.col("delta"))
                           .otherwise(0))
          .select("gi", "pidx", "is_bid", "delta", "cq").sort("gi"))
    gi = ev["gi"].to_numpy().astype(np.int64)
    pi = ev["pidx"].to_numpy()
    bmask = ev["is_bid"].to_numpy()
    dv = ev["delta"].to_numpy().astype(np.float64)
    cqv = ev["cq"].to_numpy().astype(np.float64)
    del ev, d

    cpx_all = nxt["pidx"].to_numpy().astype(np.int64) if nxt.height else pi[:1]
    p_min = int(min(pi.min(), cpx_all.min(),
                    carry["pidx"].min() if carry.height else pi.min())) - MARGIN_I
    p_max = int(max(pi.max(), cpx_all.max(),
                    carry["pidx"].max() if carry.height else pi.max())) + MARGIN_I
    dep = {"b": np.zeros(p_max - p_min + 1, np.float64),
           "a": np.zeros(p_max - p_min + 1, np.float64)}
    if carry.height:
        cb0 = carry["is_bid"].to_numpy()
        cpx = carry["pidx"].to_numpy().astype(np.int64) - p_min
        csz = carry["size_after"].to_numpy().astype(np.float64)
        np.add.at(dep["b"], cpx[cb0], csz[cb0])
        np.add.at(dep["a"], cpx[~cb0], csz[~cb0])

    mid, bbi, bai, good, pb, pa = grid_mid(bb_day, t0, ng)
    cols: dict[str, np.ndarray] = {
        "ts": t0 + np.arange(ng, dtype=np.int64) * GRID_NS, "mid": mid}
    with np.errstate(invalid="ignore"):
        cols["spread_bp"] = np.where(good, (pa - pb) / mid * 1e4, np.nan)
    for s in ("b", "a"):
        for nm in ("d25", "canc", "frag", "dar"):
            cols[f"{nm}_{s}"] = np.full(ng, np.nan)
        for tg in QTAG:
            for nm in NAMES:
                cols[f"{nm}_{s}_{tg}"] = np.full(ng, np.nan)

    n_neg = n_edge = 0
    n_out = {tg: 0 for tg in QTAG}
    bnd = chunk_bounds(bbi, bai, good, ng)
    idx = np.searchsorted(gi, bnd)
    for ci in range(len(bnd) - 1):
        g0, g1 = int(bnd[ci]), int(bnd[ci + 1])
        G = g1 - g0
        s0, s1 = int(idx[ci]), int(idx[ci + 1])
        cpx_, cg, cb, cd, cc = (pi[s0:s1], gi[s0:s1], bmask[s0:s1],
                                dv[s0:s1], cqv[s0:s1])
        gg = good[g0:g1]
        if gg.any():
            lo = int(min(bbi[g0:g1][gg].min(), bai[g0:g1][gg].min())) - MARGIN_I
            hi = int(max(bbi[g0:g1][gg].max(), bai[g0:g1][gg].max())) + MARGIN_I
            W = hi - lo + 1
            inw = (cpx_ >= lo) & (cpx_ <= hi)
            fl = (cg[inw] - g0) * W + (cpx_[inw] - lo)
            ib, dw, cw = cb[inw], cd[inw], cc[inw]
            zr = np.zeros((1, W))
            rows = np.arange(G)
            mm = np.where(gg, mid[g0:g1], 1.0)
            pxv = (lo + np.arange(W)) * PX_UNIT

            for side, sgn, msk in (("a", +1.0, ~ib), ("b", -1.0, ib)):
                D = dep[side][lo - p_min: hi - p_min + 1] + np.concatenate(
                    [zr, np.cumsum(np.bincount(fl[msk], weights=dw[msk],
                                               minlength=G * W).reshape(G, W),
                                   axis=0)[:-1]])
                n_neg += int((D < 0).sum())
                np.maximum(D, 0.0, out=D)
                C = np.bincount(fl[msk], weights=cw[msk],
                                minlength=G * W).reshape(G, W)
                if sgn > 0:
                    Do, Co, pxs, j0 = D, C, pxv, bai[g0:g1] - lo
                else:
                    Do, Co, pxs = D[:, ::-1], C[:, ::-1], -pxv[::-1]
                    j0 = (W - 1) - (bbi[g0:g1] - lo)
                okc = gg & (j0 >= 0) & (j0 < W)
                n_edge += int((gg & ~okc).sum())
                j0c = np.clip(j0, 0, W - 1)
                sub, cs, cpm = sweep(Do, pxs, sgn, j0c, mm, rows)

                # 帯内で取り消された数量(その格子点ぶん)
                ccum = np.cumsum(Co, axis=1)
                jm = np.maximum(j0c - 1, 0)
                canc = (ccum[rows, sub["_jr"]]
                        - ccum[rows, jm] * (j0c > 0)) * SZ_LOT
                cols[f"canc_{side}"][g0:g1] = np.where(okc, canc, np.nan)
                cols[f"d25_{side}"][g0:g1] = np.where(okc, sub["d25"], np.nan)
                # 脆さは直前の格子点までしか使わないので、この時点で確定できる
                fr = frag_upto(cols[f"canc_{side}"], cols[f"d25_{side}"], g0, g1)
                cols[f"frag_{side}"][g0:g1] = np.where(okc, fr, np.nan)
                gap_adjust(cs, cpm, pxs, sgn, j0c, mm, rows, fr, sub)
                for tg in QTAG:
                    n_out[tg] += int((okc & ~sub[f"_ok_{tg}"]).sum())
                    for nm in NAMES:
                        cols[f"{nm}_{side}_{tg}"][g0:g1] = np.where(
                            okc, sub[f"{nm}_{tg}"], np.nan)
                del D, C, Do, Co, cs, cpm, ccum, sub
        if s1 > s0:
            np.add.at(dep["b"], cpx_[cb] - p_min, cd[cb])
            np.add.at(dep["a"], cpx_[~cb] - p_min, cd[~cb])

    for s in ("b", "a"):
        with np.errstate(invalid="ignore", over="ignore"):
            cols[f"dar_{s}"] = cols[f"d25_{s}"] * (
                1 - np.exp(-cols[f"frag_{s}"] * STRESS_S))
    fb, fa = cols["frag_b"], cols["frag_a"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["frag_imb"] = np.where(fb + fa > 0, (fb - fa) / (fb + fa), np.nan)

    meta = {"n_raw": n_raw, "n_open": n_open, "n_orphan": n_orph,
            "n_neg_depth": n_neg, "n_edge": n_edge, "n_grid_ok": int(good.sum()),
            "n_carry_in": carry.height, "n_carry_out": nxt.height}
    for tg in QTAG:
        meta[f"n_outside_{tg}"] = n_out[tg]
    return cols, nxt, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    bb_all, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    have = set(bb_all["dt"].unique().to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]
    print(f"[load] {len(files)} 日 / bbo の除外行 {n_drop:,}", file=sys.stderr)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64})
    parts, metas = [], []
    for fp in files:
        dt = fp.stem.split("=")[1]
        bb_day = bb_all.filter(pl.col("dt") == dt).sort("ts")
        cols, carry, meta = day_features(fp, bb_day, carry)
        meta["dt"] = dt
        metas.append(meta)
        df = (pl.DataFrame({k: v for k, v in cols.items() if not k.startswith("_")})
              .with_columns(pl.lit(dt).alias("dt"))
              .filter(pl.col("mid").is_not_null()))
        parts.append(df)
        print(f"  {dt} 格子 {meta['n_grid_ok']:,} 孤児 {meta['n_orphan']:,} "
              f"負の深さ {meta['n_neg_depth']:,} 窓外(Q=50) {meta['n_outside_q50']:,}",
              file=sys.stderr)
    out = pl.concat(parts)
    num = [c for c in out.columns if c not in ("ts", "dt")]
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in num])
    out.write_parquet(ROOT / "data" / f"impact_{tag}.parquet", compression="zstd")
    pl.DataFrame(metas).write_csv(ROOT / "data" / f"impact_meta_{tag}.csv")
    print(f"\n[out] {out.height:,} 行 × {len(out.columns)} 列 "
          f"-> data/impact_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
