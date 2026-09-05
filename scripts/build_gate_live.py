"""公開フィードだけで計算できる説明変数に絞った門(live gate)。

    uv run python scripts/build_gate_live.py --coin xyz:MU

なぜ要るか
----------
入口の門(`build_entrygate.py`)は 36 変数で、そのうち

    cancel_10s, add_10s, fragility, resilience, touch_cancel_rate,
    queue_ahead_est_cnt

の 6 個は **L4(注文 1 本ごとの追加・取消イベント)からしか作れない**。
通常の WS クライアント(l2Book + trades)では計算できないので、
**probe が `EV_visible` を実時間で計算するには、この 6 個を落とした門が要る**。

ここでは落とした版(live gate)を同じ手順で学習し、

  - 標本外の往復 EV がどれだけ落ちるか
  - 元の門とどれだけ一致するか(P(live 版も通す | 完全版が通す))

を測る。係数は json に書き出して probe から読ませる。

時間契約は `build_entrygate.py` と同一(学習 59 日 / 評価 39 日、標準化も学習期間だけ)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entrygate import COLS  # noqa: E402
from build_fillpnl import (BULK, DATA, NAMES, SUB_FIT, TRAIN_FRAC,  # noqa: E402
                           design)
from plot_latency import fit_logit  # noqa: E402

# L4 の注文イベントが無いと作れないもの
L4_ONLY = ["log_cancel_10s", "log_add_10s", "log_fragility", "log_resilience",
           "log_touch_cancel_rate", "log_queue_ahead_est_cnt"]
KEEP = [i for i, c in enumerate(NAMES) if c not in L4_ONLY]
KEEP_NAMES = [NAMES[i] for i in KEEP]


def nw(v, q=14):
    n = v.size
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, q + 1):
        g += 2.0 * (1 - lg / (q + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sfx", default="_q1")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    P = pl.read_parquet(DATA / f"inv_posts_{tag}{a.sfx}.parquet")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    print(f"完全版 {len(NAMES)} 変数 → live 版 {len(KEEP)} 変数 "
          f"(落とした: {', '.join(L4_ONLY)})", flush=True)

    def day(f):
        dt = f.stem.split("=")[1]
        G = P.filter(pl.col("dt") == dt)
        if not G.height:
            return None
        T = pl.read_parquet(f, columns=COLS)
        X = design(T)
        ts = T["ts"].cast(pl.Int64).to_numpy()
        sdv = T["side"].to_numpy()
        xs, fs, ps, tt_, ss_ = [], [], [], [], []
        for sg in (1, -1):
            m = np.flatnonzero(sdv == sg)
            tt = ts[m]
            g = G.filter(pl.col("side") == sg).sort("t")
            gt = g["t"].to_numpy()
            j = np.searchsorted(tt, gt)
            ok = (j < tt.size) & (tt[np.minimum(j, tt.size - 1)] == gt)
            idx = m[np.minimum(j, tt.size - 1)][ok]
            xs.append(X[idx]); fs.append(g["filled"].to_numpy()[ok])
            ps.append(g["rt_pnl"].to_numpy()[ok]); tt_.append(gt[ok])
            ss_.append(np.full(int(ok.sum()), sg))
        return (np.concatenate(xs), np.concatenate(fs), np.concatenate(ps),
                np.concatenate(tt_), np.concatenate(ss_), dt)

    Xs, ys, ps, ms = [], [], [], []
    for k, f in enumerate(files[:ntr]):
        d = day(f)
        if d is None:
            continue
        X, fl, pn, _, _, _ = d
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32)); ys.append(fl[sl].astype(float))
        ms.append(np.zeros(X[sl].shape[0], bool))
        kp = (fl > 0) & np.isfinite(pn)
        Xs.append(X[kp].astype(np.float32))
        ys.append(np.full(int(kp.sum()), np.nan))
        ps.append(pn[kp].astype(float)); ms.append(np.ones(int(kp.sum()), bool))
        if (k + 1) % 20 == 0:
            print(f"  学習 {k+1}/{ntr}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)[:, KEEP]
    yA = np.concatenate(ys); isf = np.concatenate(ms); PN = np.concatenate(ps)
    del Xs, ys, ps, ms
    base = XA[~isf]
    mu = base.mean(axis=0); sd_ = base.std(axis=0); sd_[sd_ < 1e-12] = 1.0
    w_fill = fit_logit((XA[~isf] - mu) / sd_, yA[~isf])
    Z = np.column_stack([np.ones(int(isf.sum())), (XA[isf] - mu) / sd_])
    w_pnl, *_ = np.linalg.lstsq(Z, PN, rcond=None)
    json.dump({"names": KEEP_NAMES, "keep_idx": KEEP, "mu": mu.tolist(),
               "sd": sd_.tolist(), "w_fill": w_fill.tolist(),
               "w_pnl": w_pnl.tolist(),
               "note": "公開フィードだけで作れる変数に絞った門。probe が "
                       "EV_visible を実時間で計算するために使う"},
              open(DATA / f"entrygate_live_model_{tag}{a.sfx}.json", "w"))
    del XA, yA, isf, PN, base

    M = json.load(open(DATA / f"entrygate_model_{tag}{a.sfx}.json"))
    fmu = np.array(M["mu"]); fsd = np.array(M["sd"])
    fwf = np.array(M["w_fill"]); fwp = np.array(M["w_pnl"])

    def ev_full(X):
        Z = np.column_stack([np.ones(X.shape[0]), (X - fmu) / fsd])
        return (1 / (1 + np.exp(-(Z @ fwf)))) * (Z @ fwp)

    def ev_live(X):
        Xl = X[:, KEEP]
        Z = np.column_stack([np.ones(Xl.shape[0]), (Xl - mu) / sd_])
        return (1 / (1 + np.exp(-(Z @ w_fill)))) * (Z @ w_pnl)

    rows, agree = [], {"nf": 0, "nl": 0, "both": 0}
    for k, f in enumerate(files[ntr:]):
        d = day(f)
        if d is None:
            continue
        X, fl, pn, gt, sg, dt = d
        ef, el = ev_full(X), ev_live(X)
        gf, gl = ef > 0, el > 0
        agree["nf"] += int(gf.sum()); agree["nl"] += int(gl.sum())
        agree["both"] += int((gf & gl).sum())
        r = {"dt": dt}
        for nm, g in (("full", gf), ("live", gl)):
            v = pn[g & (fl > 0)]
            v = v[np.isfinite(v)]
            r[f"n_{nm}"] = int(g.sum())
            r[f"lots_{nm}"] = int(v.size)
            r[f"ev_{nm}"] = float(v.mean()) if v.size else np.nan
        rows.append(r)
        if (k + 1) % 10 == 0:
            print(f"  評価 {k+1}/{len(files)-ntr}", flush=True)
    D = pl.DataFrame(rows)
    D.write_csv(DATA / f"gate_live_days_{tag}.csv")
    print()
    for nm in ("full", "live"):
        v = D[f"ev_{nm}"].to_numpy()
        v = v[np.isfinite(v)]
        m, se = nw(v)
        print(f"{nm:5s}: 通した発注 {int(D['n_'+nm].sum()):>8,}  "
              f"建玉 {int(D['lots_'+nm].sum()):>7,} "
              f"({int(D['lots_'+nm].sum())/D.height:5.0f}/日)  "
              f"1 組 {m:+.4f} ± {se:.4f} bp (t={m/se:+5.2f})")
    print(f"\n完全版が通した {agree['nf']:,} 件のうち live 版も通すのは "
          f"{agree['both']:,} 件 = {100*agree['both']/max(agree['nf'],1):.1f}%")
    print(f"live 版が通した {agree['nl']:,} 件のうち完全版も通すのは "
          f"{100*agree['both']/max(agree['nl'],1):.1f}%")
    print(f"\n書き出し {DATA}/entrygate_live_model_{tag}{a.sfx}.json")


if __name__ == "__main__":
    main()
