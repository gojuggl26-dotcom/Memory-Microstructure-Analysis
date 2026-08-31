"""メッセージの流量(quote stuffing / message activity)を 14 指標で測る。

【何を測るか】
板の状態ではなく、**取引所から流れてくるメッセージそのもの**を数える。
板を作らないので、他のレポートと違い**除外を一切しない**。トリガー注文も
テイカーも `...Rejected` も、流れてきた以上はメッセージである。

    updates per second   1 秒あたりのメッセージ数
    NEW/sec              新規発注(open)
    REMOVE/sec           板から消える(取消 / 残量 0 の filled)
    UPDATE/sec           板に残ったまま変わる(部分約定 / triggered)
    messages per block   1 ブロックに入るメッセージ数
    orders changed/block 1 ブロックで動いた注文(oid)の数

    inter-event duration        ブロック間隔
    event-duration variance     その分散
    coefficient of variation    CV = σ/μ
    update burstiness           B = (σ − μ)/(σ + μ)   Goh and Barabasi (2008)
    event clustering            M = corr(τ_i, τ_{i+1})(記憶係数。B の相方)
    Fano factor                 窓長 T の計数の 分散/平均。ポアソンなら 1
    activity entropy            1 秒ごとの計数のシャノンエントロピー
    activity autocorrelation    1 秒ごとの計数の自己相関

【★ブロック時刻であることが効く】
このデータの `ts` は **ブロックの時刻**で、同じブロックの事象は全て同じ ns を持つ。
実測(2026-07-15)で 1 ブロック平均 23.9 メッセージ・中央値 10・最大 767、
ブロック間隔は中央値 72.6ms・平均 99.5ms である。したがって

  - **メッセージ単位の「イベント間隔」は同一ブロック内で 0 になり意味を持たない。**
    間隔の統計はすべて **ブロック間隔**で測る。
  - 窓長がブロック間隔より短い Fano factor は「その窓にブロックが入ったか」を
    測ってしまう。窓長ごとの値を並べて、この効果が見える形で出す。

【Fano factor は 2 通り出す】
1 日を通した計数は日内の繁閑(米国市場の寄り付き等)を含むので、Fano factor は
「日内の不均一」と「微視的な群れ」の両方を拾う。両者を分けるため、
**その日で最も忙しい 1 時間**に限った Fano factor も併記する。

【x が確定する時刻 / y の期間】
記述統計であって予測ではない。すべてその時点までに流れたメッセージだけを数える。

    uv run python scripts/build_msg_activity.py --coin xyz:MU
出力: data/msg_activity_daily_<coin>.parquet / .csv … 日 × 指標
      data/msg_activity_curves_<coin>.parquet      … 日 × 曲線(Fano / ACF / 分布)
      data/msg_activity_burst_<coin>.parquet       … 日ごとの最繁秒とその中身
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

DAY_NS = 86_400_000_000_000
SEC_NS = 1_000_000_000
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]
CLS = ["NEW", "UPDATE", "REMOVE", "REJECTED"]
# Fano factor を測る窓長[ns]。ブロック間隔(中央 72.6ms)の前後を挟むように取る
FANO_W = [10_000_000, 50_000_000, 100_000_000, 500_000_000,
          1_000_000_000, 5_000_000_000, 10_000_000_000, 60_000_000_000]
FANO_LAB = ["10ms", "50ms", "100ms", "500ms", "1s", "5s", "10s", "60s"]
ACF_LAGS = [1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600]
# ブロック間隔の分布(対数ビン、1ms〜100s)
GAP_EDGES = np.concatenate([[0], 10.0 ** np.arange(6.0, 11.01, 0.25)])
MPB_EDGES = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
TOPSEC = 10                      # 最繁の秒を何本記録するか


def acf(x: np.ndarray, lags) -> np.ndarray:
    """1 秒ごとの計数の自己相関。平均を引いてから正規化する。"""
    v = x - x.mean()
    d = float(v @ v)
    if d <= 0:
        return np.full(len(lags), np.nan)
    return np.array([float(v[:-k] @ v[k:]) / d for k in lags])


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

    # 取引数は fills から取る(filled イベントは約定の一部しか出ないため)
    fp_fills = ROOT / "data" / f"fills_{tag}.parquet"
    trades = {}
    if fp_fills.exists():
        t = (pl.read_parquet(fp_fills, columns=["dt", "tid"])
             .group_by("dt").agg(n=pl.col("tid").n_unique()))
        trades = dict(zip(t["dt"].to_list(), t["n"].to_list()))

    wdir = ROOT / "data" / f"l1user_{tag}"
    outd = ROOT / "data" / f"msg_activity_days_{tag}"
    outd.mkdir(parents=True, exist_ok=True)

    for fp in files:
        dt = fp.stem.split("=")[1]
        if (outd / f"daily_{dt}.parquet").exists():
            continue
        t0w = time.time()
        d = pl.read_parquet(fp, columns=["ts", "oid", "status", "remaining_sz"])
        d = d.with_columns(
            ts=pl.col("ts").cast(pl.Int64),
            cls=pl.when(pl.col("status") == "open").then(0)
               .when(pl.col("status") == "triggered").then(1)
               .when((pl.col("status") == "filled")
                     & (pl.col("remaining_sz") > 0)).then(1)
               .when((pl.col("status") == "filled")
                     & (pl.col("remaining_sz") <= 0)).then(2)
               .when(pl.col("status").is_in(CANCELS)).then(2)
               .otherwise(3).cast(pl.Int8))
        ts = d["ts"].to_numpy()
        n_msg = len(ts)
        t0 = (int(ts[0]) // DAY_NS) * DAY_NS
        cls = d["cls"].to_numpy()

        # ---- ブロック(同じ ts をまとめたもの)------------------------------
        g = (d.group_by("ts").agg(m=pl.len(), o=pl.col("oid").n_unique())
             .sort("ts"))
        blk_ts = g["ts"].to_numpy()
        mpb = g["m"].to_numpy()
        opb = g["o"].to_numpy()
        gap = np.diff(blk_ts).astype(np.float64)          # ns
        mu, sd = float(gap.mean()), float(gap.std(ddof=1))
        cv = sd / mu if mu > 0 else np.nan
        burst = (sd - mu) / (sd + mu) if (sd + mu) > 0 else np.nan
        # 記憶係数 M = 連続する間隔の相関(B の相方。群れているかを見る)
        x1, x2 = gap[:-1], gap[1:]
        mem = float(np.corrcoef(x1, x2)[0, 1]) if len(x1) > 2 else np.nan
        # ★ブロック間隔はチェーンのブロック生成に量子化されている。
        #   実測で最頻値が 67ms 付近・その 2 倍 134ms・3 倍 201ms に並ぶ。
        #   つまり「間隔」はこの銘柄に動きがあったブロックの間引きであって、
        #   参加者の到着の速さそのものではない。素の基本周期と、
        #   連続ブロックに動きがあった割合を測って明示する。
        near = gap[gap <= 100_000_000]
        base_ms = float(np.median(near)) / 1e6 if near.size else np.nan
        share_1blk = float((gap <= 100_000_000).mean())
        mpb_cv = float(mpb.std(ddof=1) / mpb.mean()) if mpb.mean() > 0 else np.nan

        # ---- 1 秒ごとの計数 ---------------------------------------------------
        sec = ((ts - t0) // SEC_NS).astype(np.int64)
        np.clip(sec, 0, 86399, out=sec)
        cnt = np.bincount(sec, minlength=86400).astype(np.float64)
        ccnt = [np.bincount(sec[cls == k], minlength=86400).astype(np.float64)
                for k in range(4)]
        act = cnt > 0
        p = cnt / cnt.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        srt = np.sort(cnt)[::-1]
        top1pct = float(srt[:864].sum() / cnt.sum())

        # ---- Fano factor(全日 と 最繁 1 時間)---------------------------------
        hcnt = cnt.reshape(24, 3600).sum(1)
        hbest = int(np.argmax(hcnt))
        lo, hi = t0 + hbest * 3600 * SEC_NS, t0 + (hbest + 1) * 3600 * SEC_NS
        tsh = ts[(ts >= lo) & (ts < hi)]
        fano, fano_h = [], []
        for W in FANO_W:
            c = np.bincount(((ts - t0) // W).astype(np.int64),
                            minlength=int(DAY_NS // W)).astype(np.float64)
            fano.append(float(c.var() / c.mean()) if c.mean() > 0 else np.nan)
            ch = np.bincount(((tsh - lo) // W).astype(np.int64),
                             minlength=max(int(3600 * SEC_NS // W), 1)).astype(np.float64)
            fano_h.append(float(ch.var() / ch.mean()) if ch.mean() > 0 else np.nan)

        r = acf(cnt, ACF_LAGS)

        # ---- 最も忙しい秒の中身(誰が出しているか)-----------------------------
        top = np.argsort(cnt)[::-1][:TOPSEC]
        brows = []
        wf = wdir / f"dt={dt}.parquet"
        u = pl.read_parquet(wf).unique(subset=["oid"]) if wf.exists() else None
        for s in top:
            m = sec == s
            sh = np.nan
            if u is not None and m.sum():
                oo = pl.DataFrame({"oid": d["oid"].to_numpy()[m]})
                w = oo.join(u, on="oid", how="left")["wid"].drop_nulls()
                if w.len():
                    vc = w.value_counts(sort=True)
                    sh = float(vc["count"][0] / w.len())
            brows.append({"dt": dt, "sec": int(s), "n_msg": int(cnt[s]),
                          "n_new": int(ccnt[0][s]), "n_remove": int(ccnt[2][s]),
                          "top_wallet_share": sh})

        rec = {"dt": dt, "n_msg": n_msg, "n_block": len(blk_ts),
               "n_sec_active": int(act.sum()),
               "n_trade": int(trades.get(dt, 0)),
               "msg_per_sec": n_msg / 86400.0,
               "msg_per_sec_active": float(cnt[act].mean()) if act.any() else np.nan,
               "msg_per_sec_p50": float(np.percentile(cnt, 50)),
               "msg_per_sec_p90": float(np.percentile(cnt, 90)),
               "msg_per_sec_p99": float(np.percentile(cnt, 99)),
               "msg_per_sec_max": float(cnt.max()),
               "mpb_mean": float(mpb.mean()), "mpb_p50": float(np.median(mpb)),
               "mpb_p99": float(np.percentile(mpb, 99)), "mpb_max": int(mpb.max()),
               "opb_mean": float(opb.mean()), "opb_p50": float(np.median(opb)),
               "gap_mean_ms": mu / 1e6, "gap_p50_ms": float(np.median(gap)) / 1e6,
               "gap_p99_ms": float(np.percentile(gap, 99)) / 1e6,
               "gap_var_ms2": float(gap.var(ddof=1)) / 1e12,
               "gap_sd_ms": sd / 1e6, "cv": cv, "burstiness": burst, "memory": mem,
               "block_base_ms": base_ms, "share_1blk": share_1blk, "mpb_cv": mpb_cv,
               "entropy": ent, "entropy_norm": ent / np.log(86400.0),
               "top1pct_share": top1pct,
               "fano_1s": fano[FANO_LAB.index("1s")],
               "fano_1s_busy": fano_h[FANO_LAB.index("1s")],
               "acf1": float(r[0]), "acf60": float(r[ACF_LAGS.index(60)]),
               "busy_hour": hbest}
        for k in range(4):
            rec[f"{CLS[k]}_per_sec"] = float(ccnt[k].sum() / 86400.0)
            rec[f"{CLS[k]}_share"] = float(ccnt[k].sum() / n_msg)
        rec["order_to_trade"] = (rec["NEW_per_sec"] * 86400.0 / trades[dt]
                                 if trades.get(dt) else np.nan)
        rec["msg_to_trade"] = (n_msg / trades[dt]) if trades.get(dt) else np.nan

        crows = ([{"dt": dt, "kind": "fano", "x": FANO_LAB[i], "xv": FANO_W[i],
                   "v": fano[i]} for i in range(len(FANO_W))]
                 + [{"dt": dt, "kind": "fano_busy", "x": FANO_LAB[i], "xv": FANO_W[i],
                     "v": fano_h[i]} for i in range(len(FANO_W))]
                 + [{"dt": dt, "kind": "acf", "x": str(ACF_LAGS[i]),
                     "xv": ACF_LAGS[i], "v": float(r[i])} for i in range(len(ACF_LAGS))])
        hg = np.histogram(gap, bins=GAP_EDGES)[0]
        crows += [{"dt": dt, "kind": "gap_hist", "x": f"{GAP_EDGES[i]/1e6:.3g}ms",
                   "xv": float(GAP_EDGES[i]), "v": float(hg[i])}
                  for i in range(len(hg))]
        hm = np.histogram(mpb, bins=MPB_EDGES + [10 ** 9])[0]
        crows += [{"dt": dt, "kind": "mpb_hist", "x": str(MPB_EDGES[i]),
                   "xv": float(MPB_EDGES[i]), "v": float(hm[i])}
                  for i in range(len(hm))]
        crows += [{"dt": dt, "kind": "hourly", "x": str(h), "xv": float(h),
                   "v": float(hcnt[h] / 3600.0)} for h in range(24)]

        pl.DataFrame([rec]).write_parquet(outd / f"daily_{dt}.parquet")
        pl.DataFrame(crows).write_parquet(outd / f"curves_{dt}.parquet")
        pl.DataFrame(brows).write_parquet(outd / f"burst_{dt}.parquet")
        print(f"  {dt} {n_msg:,} 通 / {len(blk_ts):,} ブロック / "
              f"{rec['msg_per_sec']:.0f} 通/秒 間隔 {rec['gap_p50_ms']:.1f}ms "
              f"B {burst:+.3f} M {mem:+.3f} Fano(1s) {rec['fano_1s']:.0f} "
              f"{time.time()-t0w:.1f}s", file=sys.stderr, flush=True)
        del d, ts, cls, sec, cnt, ccnt

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    D = cat("daily").sort("dt")
    D.write_parquet(ROOT / "data" / f"msg_activity_daily_{tag}.parquet")
    D.write_csv(ROOT / "data" / f"msg_activity_daily_{tag}.csv")
    cat("curves").write_parquet(ROOT / "data" / f"msg_activity_curves_{tag}.parquet")
    cat("burst").write_parquet(ROOT / "data" / f"msg_activity_burst_{tag}.parquet")
    print(f"\n[集計] 日 {D.height} / メッセージ {int(D['n_msg'].sum()):,}",
          file=sys.stderr)
    print(f"-> data/msg_activity_daily_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
