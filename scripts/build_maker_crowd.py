"""最良気配の近くにいるメイカーの混み具合と競争(maker crowd / competition)を測る。

【何を測るか】
1 秒ごとに板を復元し、**同じ側の最良気配から K ティック以内**に数量を置いている
口座(メイカー)だけを取り出して、その集団の姿を 9 つの指標で測る。

    number of makers near touch  近傍に数量を置いている口座の数
    effective maker count        N_eff = 1 / HHI(1 者独占で 1、n 者均等で n)
    maker competition index      MCI = N_eff / N(1 に近いほど互角。均等度)
    maker concentration          HHI = Σ w_i^2(近傍の数量シェアで測る)
    maker turnover               Δ 秒後との顔ぶれの入れ替わり(Jaccard 距離)
    maker entry rate             Δ 秒後に新しく現れる口座の割合
    maker exit rate              Δ 秒後に居なくなる口座の割合
    maker quote overlap          埋まっている 1 価格あたりの口座数(1 なら独占)
    maker clustering             1 口座が自分の数量を何価格に散らしているか

**maker concentration(HHI)と maker competition index(N_eff/N)は別物である。**
前者は「上位がどれだけ取っているか」、後者は「居る口座どうしが互角か」を測る。
口座が 2 者しか居なくても均等なら MCI は 1 になり、HHI は 0.5 になる。
どちらか一方だけでは「混んでいるのに 1 者が支配している」状態を書けない。

**maker quote overlap と maker clustering は向きが逆である。**
overlap は「同じ価格に何者が居るか」(口座どうしの重なり)、
clustering は「1 者が自分の数量を何価格に散らしているか」
(その口座自身の価格ヘルフィンダール。1 なら 1 価格に集中)。

【近傍の定義】
K = 10 ティックを主とし、0(最良ちょうど)/ 5 / 25 も併せて出す。
ティックは価格で変わる(1000 未満 0.01、1000 以上 0.1)。
K = 0 の口座数は [口座の集中度](mu_wallet_conc_report.md)の BBO wallet count
(平均 3.17 者)と一致するはずで、これが再構成の検算になる。

【板の再構成】
[口座の集中度](mu_wallet_conc_report.md)とまったく同じ規則を使う。

  - 注文ごとに「板に置かれている数量」の階段関数を作る(部分約定も追う)
  - 数量は 0.001 を 1 とする整数。同一 ns の行順はゲート2 D4 の論理順で並べ直す
  - トリガー注文とテイカー(Ioc など)は除く。reduce_only は板に載るので含める
  - **24 時間まったく動きの無い注文は落とす。** 終端イベントが最後まで来ない
    注文があり(約定の一部は filled を出さない)、放置すると板が単調に膨らむ。
    実測で寿命が 24 時間を超えた注文は 1 本も無い

【x が確定する時刻 / y の期間】
記述統計であって予測ではない。格子点 g の板はバケット g−1 までの累積、
最良気配も t_g 未満の最後の bbo 行を使う。将来の情報は入っていない。

    uv run python scripts/build_maker_crowd.py --coin xyz:MU
出力: data/maker_crowd_daily_<coin>.parquet / .csv … 日 × 指標
      data/maker_crowd_curves_<coin>.parquet       … 日 × (帯 K / 時間差 Δ)の曲線
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GRID_NS = 1_000_000_000
DAY_NS = 86_400_000_000_000
CHUNK_G = 600                      # 10 分。価格帯を狭く保って展開量を抑える
PX_UNIT = 0.01
SZ_LOT = 0.001
BIG_PX = 100_000                   # = 1000.00。これ以上はティックが 0.1
MAX_REST_NS = 86_400_000_000_000   # 24 時間動きが無ければ消えたものとみなす
RESTING = ["Alo", "Gtc"]
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]
TERMINAL = ["filled"] + CANCELS
BANDS = [0, 5, 10, 25]             # 近傍の幅[ティック]
KMAIN = 10
DELTAS = [1, 5, 10, 60, 300]       # 顔ぶれを比べる時間差[秒]
QS = [10, 25, 50, 75, 90]
MET = ["n_maker", "n_maker_bid", "n_maker_ask", "hhi", "eff_n", "mci",
       "top1", "overlap", "clustering", "depth_near", "near_share"]


def day_segments(fp: Path, wal: Path, carry: pl.DataFrame, t0: int):
    """1 日分の注文を (口座, 価格, 側, 開始格子, 終了格子, 数量) の線分に直す。"""
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status",
                                     "remaining_sz", "tif", "is_trigger"])
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  isbid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS)))
    del d
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("sz", "gone"))
    u = pl.read_parquet(wal).select("oid", "wid")
    if carry.height:
        u = pl.concat([u, carry.select("oid", "wid")])
    u = u.unique(subset=["oid"])
    ev = ev.join(u, on="oid", how="left").filter(pl.col("wid").is_not_null())
    if carry.height:
        cr = carry.select("oid", "isbid", "pidx", "wid",
                          ts=pl.col("ts_last"), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "isbid", "pidx", "wid", "size_after",
                   ts_last=pl.col("ts")))
    # 線分 = 「あるイベントの直後から、次のイベントの直前まで」その数量が板にある
    seg = (ev.with_columns(t_end=pl.col("ts").shift(-1).over("oid"))
           .filter((pl.col("size_after") > 0) & pl.col("t_end").is_not_null())
           .select("wid", "pidx", "isbid", "size_after",
                   g0=((pl.col("ts") - t0) // GRID_NS + 1).cast(pl.Int64),
                   g1=((pl.col("t_end") - t0 - 1) // GRID_NS + 1).cast(pl.Int64)))
    # 日末まで残る線分(次のイベントが無いもの)は翌日へ carry するので、
    # この日については格子の終わりまで生きているものとして足す
    tail = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
            .select("wid", "pidx", "isbid", "size_after",
                    g0=((pl.col("ts") - t0) // GRID_NS + 1).cast(pl.Int64),
                    g1=pl.lit(DAY_NS // GRID_NS, pl.Int64)))
    ng = DAY_NS // GRID_NS
    seg = (pl.concat([seg, tail])
           .with_columns(g0=pl.col("g0").clip(0, ng), g1=pl.col("g1").clip(0, ng))
           .filter(pl.col("g1") > pl.col("g0")).sort("g0"))
    return seg, nxt


def seg_stats(g, w, p, q, ng):
    """(格子, 口座, 価格, 数量) から格子ごとの指標を出す。"""
    out = {}
    o = np.lexsort((p, w, g))
    g, w, p, q = g[o], w[o], p[o], q[o]
    # --- (格子, 口座) ごとの合計
    kw = (g != np.roll(g, 1)) | (w != np.roll(w, 1))
    kw[0] = True
    st = np.flatnonzero(kw)
    gw_g = g[st]
    gw_q = np.add.reduceat(q, st)
    gw_q2 = np.add.reduceat(q * q, st)          # 口座内の価格ヘルフィンダール用
    # --- 格子ごと
    n = np.bincount(gw_g, minlength=ng).astype(np.float64)
    s1 = np.bincount(gw_g, weights=gw_q, minlength=ng)
    s2 = np.bincount(gw_g, weights=gw_q * gw_q, minlength=ng)
    mx = np.zeros(ng)
    np.maximum.at(mx, gw_g, gw_q)
    with np.errstate(invalid="ignore", divide="ignore"):
        hhi = np.where(s1 > 0, s2 / (s1 * s1), np.nan)
        out["n_maker"] = np.where(n > 0, n, np.nan)
        out["hhi"] = hhi
        out["eff_n"] = 1.0 / hhi
        out["mci"] = np.where(n > 0, (1.0 / hhi) / n, np.nan)
        out["top1"] = np.where(s1 > 0, mx / s1, np.nan)
        out["depth_near"] = np.where(s1 > 0, s1 * SZ_LOT, np.nan)
        # maker clustering = 口座ごとの価格ヘルフィンダールの平均
        cl = np.where(gw_q > 0, gw_q2 / (gw_q * gw_q), np.nan)
        cs = np.bincount(gw_g, weights=np.nan_to_num(cl), minlength=ng)
        out["clustering"] = np.where(n > 0, cs / n, np.nan)
    # --- maker quote overlap = 埋まっている 1 価格あたりの口座数
    o2 = np.lexsort((w, p, g))
    g2, p2 = g[o2], p[o2]
    kp = (g2 != np.roll(g2, 1)) | (p2 != np.roll(p2, 1))
    kp[0] = True
    lv = np.bincount(g2[kp], minlength=ng).astype(np.float64)   # 埋まった価格の数
    npair = np.bincount(g, minlength=ng).astype(np.float64)     # (価格, 口座) の数
    with np.errstate(invalid="ignore", divide="ignore"):
        out["overlap"] = np.where(lv > 0, npair / lv, np.nan)
    return out, gw_g, w[st]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"data/l1_{tag} が空。先に fetch_l1.py を実行すること")
    bbo, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 信じられない行を {n_drop:,} 除いた -> {bbo.height:,} 行", file=sys.stderr)
    wdir = ROOT / "data" / f"l1user_{tag}"
    outd = ROOT / "data" / f"maker_crowd_days_{tag}"
    outd.mkdir(parents=True, exist_ok=True)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "isbid": pl.Boolean,
                                 "pidx": pl.Int32, "wid": pl.Int32,
                                 "size_after": pl.Int64, "ts_last": pl.Int64})
    k0 = 0
    for i, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        if all((outd / f"{w}_{dt}.parquet").exists() for w in ("daily", "carry")):
            k0 = i + 1
        else:
            break
    if k0:
        carry = pl.read_parquet(outd / f"carry_{files[k0-1].stem.split('=')[1]}.parquet")
        print(f"[再開] {k0} 日ぶんを飛ばす(繰越 {carry.height:,} 本)", file=sys.stderr)

    ng = DAY_NS // GRID_NS
    for fp in files[k0:]:
        dt = fp.stem.split("=")[1]
        tw = time.time()
        bd = bbo.filter(pl.col("dt") == dt).sort("ts")
        wf = wdir / f"dt={dt}.parquet"
        if bd.is_empty() or not wf.exists():
            print(f"  {dt} bbo か口座情報が無い。飛ばす", file=sys.stderr)
            continue
        t0 = (int(pl.scan_parquet(fp).select(pl.col("ts").cast(pl.Int64).min())
                  .collect().item()) // DAY_NS) * DAY_NS
        n_stale = 0
        if carry.height:
            keep = carry.filter(t0 - pl.col("ts_last") <= MAX_REST_NS)
            n_stale = carry.height - keep.height
            carry = keep
        seg, carry = day_segments(fp, wf, carry, t0)

        bts = bd["ts"].cast(pl.Int64).to_numpy()
        tg = t0 + np.arange(ng, dtype=np.int64) * GRID_NS
        j = np.searchsorted(bts, tg, side="left") - 1
        gok = j >= 0
        j = np.where(gok, j, 0)
        bbg = np.where(gok, np.rint(bd["best_bid"].to_numpy()[j] / PX_UNIT),
                       -10 ** 9).astype(np.int64)
        bag = np.where(gok, np.rint(bd["best_ask"].to_numpy()[j] / PX_UNIT),
                       10 ** 9).astype(np.int64)
        tkb = np.where(bbg >= BIG_PX, 10, 1)
        tka = np.where(bag >= BIG_PX, 10, 1)

        sg0 = seg["g0"].to_numpy()
        sg1 = seg["g1"].to_numpy()
        sw = seg["wid"].to_numpy().astype(np.int64)
        sp = seg["pidx"].to_numpy().astype(np.int64)
        sb = seg["isbid"].to_numpy()
        sq = seg["size_after"].to_numpy().astype(np.float64)
        del seg
        uw, sw = np.unique(sw, return_inverse=True)
        W = len(uw)

        acc = {k: [] for k in ["g", "w", "p", "q", "d", "b"]}
        for c0 in range(0, ng, CHUNK_G):
            c1 = min(c0 + CHUNK_G, ng)
            gg = gok[c0:c1]
            if not gg.any():
                continue
            # ★このかたまりの価格帯だけに絞ってから展開する。絞らずに全生存注文を
            #   格子へ展開すると 1 日 1 億行を超え、桁違いに遅くなる。
            blo = int(bbg[c0:c1][gg].min()) - BANDS[-1] * 10
            bhi = int(bag[c0:c1][gg].max()) + BANDS[-1] * 10
            m = (sg0 < c1) & (sg1 > c0) & (sp >= blo) & (sp <= bhi)
            if not m.any():
                continue
            a0 = np.maximum(sg0[m], c0)
            a1 = np.minimum(sg1[m], c1)
            ln = (a1 - a0).astype(np.int64)
            ok = ln > 0
            if not ok.any():
                continue
            a0, ln = a0[ok], ln[ok]
            idx = np.repeat(np.flatnonzero(m)[ok], ln)
            off = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
            gv = np.repeat(a0, ln) + off
            pv, wv, qv, bv = sp[idx], sw[idx], sq[idx], sb[idx]
            dist = np.where(bv, (bbg[gv] - pv) // tkb[gv], (pv - bag[gv]) // tka[gv])
            keep = gok[gv] & (dist >= 0) & (dist <= BANDS[-1])
            for k, v in (("g", gv), ("w", wv), ("p", pv), ("q", qv),
                         ("d", dist), ("b", bv)):
                acc[k].append(v[keep])
        if not acc["g"] or sum(len(v) for v in acc["g"]) == 0:
            print(f"  {dt} 近傍に何も無い。飛ばす", file=sys.stderr)
            continue
        G = {k: np.concatenate(v) for k, v in acc.items()}
        del acc

        rec = {"dt": dt, "n_seg": int(len(sg0)), "n_wallet_day": W,
               "n_stale": n_stale, "n_carry": int(carry.height),
               "n_grid_ok": int(gok.sum()), "n_inc": int(len(G["g"]))}
        crows = []
        for K in BANDS:
            s = G["d"] <= K
            if not s.any():
                continue
            st, _, _ = seg_stats(G["g"][s], G["w"][s], G["p"][s], G["q"][s], ng)
            sb_ = G["b"][s]
            for side, nm in ((True, "n_maker_bid"), (False, "n_maker_ask")):
                t = sb_ == side
                o = np.lexsort((G["w"][s][t], G["g"][s][t]))
                gg2, ww2 = G["g"][s][t][o], G["w"][s][t][o]
                kk = (gg2 != np.roll(gg2, 1)) | (ww2 != np.roll(ww2, 1))
                if len(kk):
                    kk[0] = True
                st[nm] = np.where(np.bincount(gg2[kk], minlength=ng) > 0,
                                  np.bincount(gg2[kk], minlength=ng), np.nan)
            for k in MET:
                if k in ("near_share",):
                    continue
                v = st.get(k)
                if v is None:
                    continue
                crows.append({"dt": dt, "kind": "band", "K": K, "metric": k,
                              "mean": float(np.nanmean(v)),
                              "p50": float(np.nanmedian(v))})
                if K == KMAIN:
                    rec[f"{k}_mean"] = float(np.nanmean(v))
                    for qq in QS:
                        rec[f"{k}_p{qq}"] = float(np.nanpercentile(v, qq))
            if K == KMAIN:
                # --- 顔ぶれの入れ替わり(全日の (格子, 口座) の集合で計算)
                key = np.unique(G["g"][s] * W + G["w"][s])
                for D_ in DELTAS:
                    inter = np.intersect1d(key, key - D_ * W, assume_unique=True).size
                    # g と g+Δ の両方が有効な格子点に限る
                    both = gok[: ng - D_] & gok[D_:]
                    nA = np.bincount(G["g"][s], minlength=ng)
                    nA = (nA > 0).astype(np.int64)
                    sz_g = np.bincount(np.unique(G["g"][s] * W + G["w"][s]) // W,
                                       minlength=ng).astype(np.float64)
                    tot = float(sz_g[: ng - D_][both].sum())
                    tot2 = float(sz_g[D_:][both].sum())
                    un = tot + tot2 - inter
                    crows.append({"dt": dt, "kind": "delta", "K": K, "metric": "turnover",
                                  "mean": 1.0 - inter / un if un > 0 else np.nan,
                                  "p50": float(D_)})
                    crows.append({"dt": dt, "kind": "delta", "K": K, "metric": "entry",
                                  "mean": (tot2 - inter) / tot if tot > 0 else np.nan,
                                  "p50": float(D_)})
                    crows.append({"dt": dt, "kind": "delta", "K": K, "metric": "exit",
                                  "mean": (tot - inter) / tot if tot > 0 else np.nan,
                                  "p50": float(D_)})
                    if D_ == 60:
                        rec["turnover60"] = 1.0 - inter / un if un > 0 else np.nan
                        rec["entry60"] = (tot2 - inter) / tot if tot > 0 else np.nan
                        rec["exit60"] = (tot - inter) / tot if tot > 0 else np.nan
        pl.DataFrame([rec]).write_parquet(outd / f"daily_{dt}.parquet")
        pl.DataFrame(crows).write_parquet(outd / f"curves_{dt}.parquet")
        carry.write_parquet(outd / f"carry_{dt}.parquet")
        print(f"  {dt} 近傍 {rec.get('n_maker_mean', float('nan')):.1f} 者 "
              f"HHI {rec.get('hhi_mean', float('nan')):.3f} "
              f"MCI {rec.get('mci_mean', float('nan')):.3f} "
              f"重なり {rec.get('overlap_mean', float('nan')):.2f} "
              f"入替(60s) {rec.get('turnover60', float('nan')):.3f} "
              f"{time.time()-tw:.1f}s", file=sys.stderr, flush=True)
        del G

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    D = cat("daily").sort("dt")
    D.write_parquet(ROOT / "data" / f"maker_crowd_daily_{tag}.parquet")
    D.write_csv(ROOT / "data" / f"maker_crowd_daily_{tag}.csv")
    cat("curves").write_parquet(ROOT / "data" / f"maker_crowd_curves_{tag}.parquet")
    print(f"\n[集計] 日 {D.height}", file=sys.stderr)
    print(f"-> data/maker_crowd_daily_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
