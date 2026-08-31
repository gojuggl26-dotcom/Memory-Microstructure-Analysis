"""最良気配の入れ替わり(BBO turnover)を 9 指標で測る。

【なぜ格子を使わないか】
最良価格が保たれる時間は**中央値 207〜343ms**(実測)で、1 秒格子では短すぎて
測れない。そこで本スクリプトは格子を一切使わず、**区間の演算だけ**で ns 精度の
まま測る。

    run       ある価格が最良であり続けた時間区間(bbo から作る)
    線分      ある注文が板にその数量で載っていた時間区間(l1 から作る)
    交差      その注文が「最良にあった」時間区間 = 線分 ∩ 同じ価格の run

【9 指標】

    bid / ask BBO replacement count  最良価格が変わった回数[回/秒]
    BBO lifetime                     run の長さの分布
    BBO size turnover                同じ価格が最良の間に消えた数量 ÷ 平均数量[回転/秒]
    best-price persistence           同じ最良価格が Δ 続く割合
    BBO order turnover               最良に居る注文が入れ替わる率[本/秒]
    BBO wallet turnover              最良に居る口座が入れ替わる率[者/秒]
    touch persistence                最良にある注文が Δ 後も残っている割合
    touch ownership duration         口座が最良に居続ける連続時間の分布

【★持続の 2 つを同じ形で定義する】
区間の長さ L の集合に対して

    persistence(Δ) = Σ max(0, L_i − Δ) / Σ L_i

とする。best-price は run の長さ、touch は交差区間の長さに当てる。
**価格が同じなら中身も同じとは限らない**ので、必ず
best-price persistence ≥ touch persistence になる。この包含関係が検算になる。

一度離れた価格が戻ってくる場合は「続いた」に数えない(下限として読む)。

【交差の求め方】
同じ (側, 価格) の run は互いに重ならず時間順に並ぶので、開始も終了も単調である。
したがって線分 [t0, t1) と重なる run は

    lo = searchsorted(run の終了, t0, "right")
    hi = searchsorted(run の開始, t1, "left")

で [lo, hi) と一発で決まる。走査もループも要らない。

【x が確定する時刻 / y の期間】
記述統計であって予測ではない。すべて観測された区間の重なりだけを数える。

    uv run python scripts/build_bbo_turnover.py --coin xyz:MU
出力: data/bbo_turnover_daily_<coin>.parquet / .csv … 日 × 指標
      data/bbo_turnover_curves_<coin>.parquet       … 日 × 分布と持続曲線
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

DAY_NS = 86_400_000_000_000
PX_UNIT = 0.01
SZ_LOT = 0.001
MAX_REST_NS = 86_400_000_000_000     # 24 時間動きが無ければ消えたものとみなす
RESTING = ["Alo", "Gtc"]
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]
TERMINAL = ["filled"] + CANCELS
# 持続を測る時間差[ns]
DELTAS = [10_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000,
          1_000_000_000, 2_000_000_000, 5_000_000_000]
DLAB = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s", "2s", "5s"]
# 分布の対数ビン(1ms〜1000s)
HIST_EDGES = np.concatenate([[0.0], 10.0 ** np.arange(6.0, 12.01, 0.5)])


def persistence(L: np.ndarray, deltas) -> list[float]:
    """長さ L の区間が Δ 続く割合。Σ max(0, L−Δ) / Σ L。"""
    tot = float(L.sum())
    if tot <= 0:
        return [np.nan] * len(deltas)
    return [float(np.maximum(L - d, 0).sum() / tot) for d in deltas]


def union_len(a: np.ndarray, b: np.ndarray) -> float:
    """区間 [a, b) の和集合の長さ。重なりを二重に数えない。"""
    if len(a) == 0:
        return 0.0
    o = np.argsort(a, kind="stable")
    a, b = a[o], b[o]
    e = np.maximum.accumulate(b)
    start = np.concatenate([[True], a[1:] > e[:-1]])
    idx = np.flatnonzero(start)
    ends = np.concatenate([e[idx[1:] - 1], [e[-1]]])
    return float((ends - a[idx]).sum())


def union_runs(a: np.ndarray, b: np.ndarray):
    """区間 [a, b) の和集合を連結成分ごとに返す(開始, 終了)。"""
    if len(a) == 0:
        return np.empty(0), np.empty(0)
    o = np.argsort(a, kind="stable")
    a, b = a[o], b[o]
    e = np.maximum.accumulate(b)
    start = np.concatenate([[True], a[1:] > e[:-1]])
    idx = np.flatnonzero(start)
    ends = np.concatenate([e[idx[1:] - 1], [e[-1]]])
    return a[idx], ends


def day_lines(fp: Path, wal: Path, carry: pl.DataFrame, t0: int):
    """1 日分の注文を (口座, 価格, 側, 開始 ns, 終了 ns, 数量) の線分に直す。"""
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
    ev = (ev.join(u.unique(subset=["oid"]), on="oid", how="left")
          .filter(pl.col("wid").is_not_null()))
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
    end = t0 + DAY_NS
    seg = (ev.with_columns(t_end=pl.col("ts").shift(-1).over("oid").fill_null(end))
           .filter(pl.col("size_after") > 0)
           .select("oid", "wid", "pidx", "isbid", "size_after",
                   a=pl.col("ts").clip(t0, end), b=pl.col("t_end").clip(t0, end))
           .filter(pl.col("b") > pl.col("a")))
    return seg, nxt


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
    outd = ROOT / "data" / f"bbo_turnover_days_{tag}"
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
        seg, carry = day_lines(fp, wf, carry, t0)

        bts = bd["ts"].cast(pl.Int64).to_numpy()
        rec = {"dt": dt, "n_bbo": int(len(bts)), "n_seg": int(seg.height),
               "n_stale": n_stale, "n_carry": int(carry.height)}
        crows = []
        span = float(bts[-1] - bts[0])

        sp = seg["pidx"].to_numpy().astype(np.int64)
        sb = seg["isbid"].to_numpy()
        sa = seg["a"].to_numpy()
        sbb = seg["b"].to_numpy()
        sw = seg["wid"].to_numpy().astype(np.int64)
        sq = seg["size_after"].to_numpy().astype(np.float64)
        del seg

        for side, nm in ((True, "bid"), (False, "ask")):
            px = np.rint((bd["best_bid"] if side else bd["best_ask"]).to_numpy()
                         / PX_UNIT).astype(np.int64)
            qz = (bd["bid_sz"] if side else bd["ask_sz"]).to_numpy()
            ch = np.flatnonzero(np.diff(px) != 0) + 1
            rs = np.concatenate([[0], ch])            # run の開始行
            re_ = np.concatenate([ch, [len(px) - 1]])  # run の終了行
            r_s, r_e = bts[rs], bts[re_]
            r_p = px[rs]
            L = (r_e - r_s).astype(np.float64)
            ok = L > 0
            r_s, r_e, r_p, L = r_s[ok], r_e[ok], r_p[ok], L[ok]

            rec[f"repl_{nm}"] = len(ch) / (span / 1e9)
            rec[f"life_{nm}_ms_p50"] = float(np.median(L)) / 1e6
            rec[f"life_{nm}_ms_mean"] = float(L.mean()) / 1e6
            rec[f"life_{nm}_ms_p90"] = float(np.percentile(L, 90)) / 1e6
            # 同じ価格が続く間に消えた数量 ÷ 平均数量
            same = np.diff(px) == 0
            dq = np.diff(qz)
            dec = float(np.where(same & (dq < 0), -dq, 0.0).sum())
            rec[f"size_turn_{nm}"] = dec / max(qz.mean(), 1e-9) / (span / 1e9)
            for i2, v in enumerate(persistence(L, DELTAS)):
                crows.append({"dt": dt, "kind": f"bestpx_{nm}", "x": DLAB[i2],
                              "xv": float(DELTAS[i2]), "v": v})
            h = np.histogram(L, bins=HIST_EDGES)[0]
            for i2 in range(len(h)):
                crows.append({"dt": dt, "kind": f"life_{nm}",
                              "x": f"{HIST_EDGES[i2]/1e6:.4g}ms",
                              "xv": float(HIST_EDGES[i2]), "v": float(h[i2])})

            # ---- 線分 ∩ run(同じ側・同じ価格)-------------------------------
            m = sb == side
            lp, la, lb, lw, lq = sp[m], sa[m], sbb[m], sw[m], sq[m]
            o = np.argsort(lp, kind="stable")
            lp, la, lb, lw, lq = lp[o], la[o], lb[o], lw[o], lq[o]
            ro = np.argsort(r_p, kind="stable")
            rp_s, rs_s, re_s = r_p[ro], r_s[ro], r_e[ro]
            # 価格ごとのまとまりを取り出す
            up, first = np.unique(rp_s, return_index=True)
            last = np.append(first[1:], len(rp_s))
            lf = np.searchsorted(lp, up, side="left")
            ll = np.searchsorted(lp, up, side="right")
            A, B, WD, QD = [], [], [], []
            for k in range(len(up)):
                i0, i1 = lf[k], ll[k]
                if i1 <= i0:
                    continue
                j0, j1 = first[k], last[k]
                # ★同じ価格の run は重ならず時間順なので開始も終了も単調。
                #   したがって重なる run は searchsorted 2 回で範囲が決まる。
                rr_s, rr_e = rs_s[j0:j1], re_s[j0:j1]
                lo = np.searchsorted(rr_e, la[i0:i1], side="right")
                hi = np.searchsorted(rr_s, lb[i0:i1], side="left")
                cnt = np.maximum(hi - lo, 0)
                if cnt.sum() == 0:
                    continue
                sel = cnt > 0
                idx = np.repeat(np.arange(i0, i1)[sel], cnt[sel])
                base = np.repeat(lo[sel], cnt[sel])
                off = np.arange(cnt[sel].sum()) - np.repeat(
                    np.cumsum(cnt[sel]) - cnt[sel], cnt[sel])
                rj = j0 + base + off
                A.append(np.maximum(la[idx], rs_s[rj]))
                B.append(np.minimum(lb[idx], re_s[rj]))
                WD.append(lw[idx])
                QD.append(lq[idx])
            if not A:
                continue
            ca, cb2 = np.concatenate(A), np.concatenate(B)
            cw, cq = np.concatenate(WD), np.concatenate(QD)
            good = cb2 > ca
            ca, cb2, cw, cq = ca[good], cb2[good], cw[good], cq[good]
            cl = (cb2 - ca).astype(np.float64)
            best_span = float(L.sum())           # 最良が定義されていた総時間

            rec[f"order_turn_{nm}"] = len(cl) / (best_span / 1e9)
            rec[f"touch_dwell_{nm}_ms_p50"] = float(np.median(cl)) / 1e6
            rec[f"touch_dwell_{nm}_ms_mean"] = float(cl.mean()) / 1e6
            for i2, v in enumerate(persistence(cl, DELTAS)):
                crows.append({"dt": dt, "kind": f"touch_{nm}", "x": DLAB[i2],
                              "xv": float(DELTAS[i2]), "v": v})
            # ---- 口座が最良に居続ける連続時間 ----------------------------------
            ow = np.argsort(cw, kind="stable")
            cw_s, ca_s, cb_s = cw[ow], ca[ow], cb2[ow]
            bnd = np.flatnonzero(np.concatenate([[True], cw_s[1:] != cw_s[:-1]]))
            bnd2 = np.append(bnd[1:], len(cw_s))
            durs, nsp = [], 0
            for i3 in range(len(bnd)):
                s_, e_ = union_runs(ca_s[bnd[i3]:bnd2[i3]], cb_s[bnd[i3]:bnd2[i3]])
                durs.append((e_ - s_).astype(np.float64))
                nsp += len(s_)
            du = np.concatenate(durs) if durs else np.zeros(0)
            rec[f"own_{nm}_ms_p50"] = float(np.median(du)) / 1e6 if du.size else np.nan
            rec[f"own_{nm}_ms_mean"] = float(du.mean()) / 1e6 if du.size else np.nan
            rec[f"own_{nm}_ms_p90"] = (float(np.percentile(du, 90)) / 1e6
                                       if du.size else np.nan)
            rec[f"wallet_turn_{nm}"] = nsp / (best_span / 1e9)
            rec[f"n_wallet_{nm}"] = int(len(bnd))
            h = np.histogram(du, bins=HIST_EDGES)[0]
            for i2 in range(len(h)):
                crows.append({"dt": dt, "kind": f"own_{nm}",
                              "x": f"{HIST_EDGES[i2]/1e6:.4g}ms",
                              "xv": float(HIST_EDGES[i2]), "v": float(h[i2])})
            # 最良にある数量が交差から復元できるかの検算(bbo の数量と比べる)
            rec[f"touch_cover_{nm}"] = union_len(ca, cb2) / best_span

        pl.DataFrame([rec]).write_parquet(outd / f"daily_{dt}.parquet")
        pl.DataFrame(crows).write_parquet(outd / f"curves_{dt}.parquet")
        carry.write_parquet(outd / f"carry_{dt}.parquet")
        print(f"  {dt} 置換 {rec.get('repl_bid', 0):.2f}/秒 寿命 "
              f"{rec.get('life_bid_ms_p50', 0):.0f}ms 注文回転 "
              f"{rec.get('order_turn_bid', 0):.2f}/秒 所有 "
              f"{rec.get('own_bid_ms_p50', 0):.0f}ms 被覆 "
              f"{rec.get('touch_cover_bid', 0):.3f} {time.time()-tw:.1f}s",
              file=sys.stderr, flush=True)

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    D = cat("daily").sort("dt")
    D.write_parquet(ROOT / "data" / f"bbo_turnover_daily_{tag}.parquet")
    D.write_csv(ROOT / "data" / f"bbo_turnover_daily_{tag}.csv")
    cat("curves").write_parquet(ROOT / "data" / f"bbo_turnover_curves_{tag}.parquet")
    print(f"\n[集計] 日 {D.height}", file=sys.stderr)
    print(f"-> data/bbo_turnover_daily_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
