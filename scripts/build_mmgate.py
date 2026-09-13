"""入口の門を往復損益で学習する(最良気配と約定だけで作る版)。

    uv run python scripts/build_mmgate.py --coin xyz:INTC

出力: data/entrygate_<coin>_mmg.parquet   (dt, t, side, ev, pass)
      data/mmgate_model_<coin>.json       係数と標準化のパラメータ
      data/mmgate_fit_<coin>.csv          学習・評価それぞれの当てはまり

`build_entrygate.py` と同じ考え方だが、**入力を 45 列の候補テーブルではなく
bbo と fills だけにした**ものである。候補テーブルは 1 銘柄 6.1GB あり、
作ってあるのは `xyz:MU` だけなので、他の 6 銘柄を同じ土俵に載せられない。

## 何を学習するか

`build_inventory.py --posts` が出した**発注 1 件ずつ**の記録

    (dt, t, side, filled, rt_pnl)

に対して、**その発注時点 t までの板と約定だけ**から作った説明変数 X_t で

    EV(X_t) = P(約定 | X_t) × E[往復損益 | 約定, X_t]

を推定し、**在庫を増やす側は EV > 0 のときだけ出す**。
P はロジット、E[·] は最小二乗。どちらも**学習期間の平均と標準偏差だけ**で
標準化する(`build_entrygate.py` と同じ)。

## 時間契約

- 発注時刻 t は bbo が動いた時刻そのもの(遅延 0 の設定)なので、
  後ろ向き添字 `searchsorted(ts, t, "right") − 1` は **t の気配そのもの**を指す。
  未来は 1 点も入らない。
- 目的変数 `filled` / `rt_pnl` だけが t 以降を使う。
- 学習は前 60% の日、評価は後 40% の日。**標準化・係数はすべて学習期間だけから**。
- 分位・クリップは使わない(使うなら学習期間だけから作ること)。

## 説明変数(20 本、すべて t で確定)

スプレッド / 自分側と反対側の待ち行列 / OBI(自分側に揃える)/ microprice のずれ /
OFI 1s・5s・10s(自分側に揃える)/ 攻撃的な符号つき数量 1s・10s(同)/
10 秒の約定件数 / mid の実現ボラ 10s・60s / 最後に mid が動いてからの秒数 /
mid のリターン 1s・10s(自分側に揃える)/ ティック幅 / 時刻の 1 次調和 2 本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN_FRAC = 0.60
SUB_FIT = 20                # ロジットの学習は 20 行に 1 行(全部だと重い)
DAY_NS = 86_400_000_000_000

NAMES = ["spread_bp", "log_qown", "log_qopp", "obi_own", "delta_own",
         "ofi1_own", "ofi5_own", "ofi10_own", "av1_own", "av10_own",
         "ntrade10", "vol10", "vol60", "log_tsince", "ret1_own", "ret10_own",
         "tick_bp", "sin_t", "cos_t", "qratio"]


def design(ts, pb, pa, qb, qa, ft, fpx, fsz, fbuy, tq, sd):
    """発注時刻 tq・側 sd に対する説明変数。すべて後ろ向き。"""
    mid = 0.5 * (pb + pa)
    j = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, ts.size - 1)

    # OFI(Cont–Kukanov–Stoikov の増分の累積)
    e = np.concatenate([[0.0], (
        (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
        - ((pa[1:] <= pa[:-1]) * qa[1:] - (pa[1:] >= pa[:-1]) * qa[:-1]))])
    cofi = np.cumsum(e)

    def back(w):
        k = np.clip(np.searchsorted(ts, tq - int(w * 1e9), side="right") - 1,
                    0, ts.size - 1)
        return cofi[j] - cofi[k], k

    o1, _ = back(1.0)
    o5, _ = back(5.0)
    o10, k10 = back(10.0)

    # 攻撃的な符号つき数量(買いテイカー − 売りテイカー)
    sv = np.concatenate([[0.0], np.cumsum(np.where(fbuy, fsz, -fsz))])
    cn = np.concatenate([[0.0], np.cumsum(np.ones(ft.size))])

    def agg(w):
        hi = np.searchsorted(ft, tq, side="right")
        lo = np.searchsorted(ft, tq - int(w * 1e9), side="right")
        return sv[hi] - sv[lo], cn[hi] - cn[lo]

    a1, _ = agg(1.0)
    a10, n10 = agg(10.0)

    # mid のリターンと実現ボラ
    lm = np.log(mid)
    r1 = lm[j] - lm[np.clip(np.searchsorted(ts, tq - 1_000_000_000,
                                            side="right") - 1, 0, ts.size - 1)]
    r10 = lm[j] - lm[k10]
    d = np.diff(lm, prepend=lm[0])
    c2 = np.concatenate([[0.0], np.cumsum(d * d)])

    def vol(w):
        k = np.clip(np.searchsorted(ts, tq - int(w * 1e9), side="right") - 1,
                    0, ts.size - 1)
        return np.sqrt(np.maximum(c2[j + 1] - c2[k + 1], 0.0)) * 1e4

    # 最後に mid が動いてからの秒数
    mv = np.flatnonzero(np.concatenate([[True], mid[1:] != mid[:-1]]))
    last = ts[mv[np.clip(np.searchsorted(mv, j, side="right") - 1, 0,
                         mv.size - 1)]]
    tsince = np.maximum(tq - last, 0) / 1e9

    m = mid[j]
    sp = (pa[j] - pb[j]) / m * 1e4
    qo = np.where(sd > 0, qb[j], qa[j])          # 自分側の待ち行列
    qp = np.where(sd > 0, qa[j], qb[j])
    obi = (qb[j] - qa[j]) / np.maximum(qb[j] + qa[j], 1e-12)
    tick = np.where(m >= 1000.0, 0.1, 0.01)
    tod = (tq % DAY_NS) / DAY_NS * 2 * np.pi

    X = np.column_stack([
        sp,
        np.log1p(qo), np.log1p(qp),
        sd * obi,                                 # 自分側が厚いほど正
        sd * sp / 2 * obi,                        # microprice のずれ(bp)
        sd * o1, sd * o5, sd * o10,
        sd * a1, sd * a10,
        np.log1p(n10),
        vol(10.0), vol(60.0),
        np.log1p(tsince),
        sd * r1 * 1e4, sd * r10 * 1e4,
        tick / m * 1e4,
        np.sin(tod), np.cos(tod),
        np.log1p(qo) - np.log1p(qp),
    ])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def day_design(tag, dt, G):
    d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                  .filter(pl.col("dt") == dt).collect())[0].sort("ts")
    if d.height < 100 or not G.height:
        return None
    ts = d["ts"].cast(pl.Int64).to_numpy()
    F = (pl.scan_parquet(DATA / f"fills_{tag}.parquet")
         .filter(pl.col("crossed") & (pl.col("dt") == dt))
         .select("ts", "px", "sz", "side").collect().sort("ts"))
    X = design(ts, d["best_bid"].to_numpy(), d["best_ask"].to_numpy(),
               d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy(),
               F["ts"].cast(pl.Int64).to_numpy(), F["px"].to_numpy(),
               F["sz"].to_numpy(), (F["side"] == "B").to_numpy(),
               G["t"].to_numpy(), G["side"].to_numpy().astype(np.float64))
    return X, G["filled"].to_numpy().astype(float), G["rt_pnl"].to_numpy()


def fit_logit(X, y, iters=40, l2=1e-3):
    Z = np.column_stack([np.ones(X.shape[0]), X])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + l2 * w
        s = p * (1 - p) + 1e-9
        H = (Z * s[:, None]).T @ Z + l2 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    ap.add_argument("--sfx", default="_q1", help="読み込む inv_posts の接尾辞")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    P = pl.read_parquet(DATA / f"inv_posts_{tag}{a.sfx}.parquet")
    days = P["dt"].unique(maintain_order=False).sort().to_list()
    ntr = int(round(len(days) * TRAIN_FRAC))
    print(f"{a.coin}: 発注 {P.height:,} / {len(days)} 日 "
          f"(学習 {ntr} 日 / 評価 {len(days)-ntr} 日)", flush=True)

    # ---- 学習 -----------------------------------------------------------
    Xs, yf, Xp, yp = [], [], [], []
    for k, dt in enumerate(days[:ntr]):
        r = day_design(tag, dt, P.filter(pl.col("dt") == dt).sort("t"))
        if r is None:
            continue
        X, fl, pn = r
        Xs.append(X[::SUB_FIT].astype(np.float32))
        yf.append(fl[::SUB_FIT])
        kp = (fl > 0) & np.isfinite(pn)
        Xp.append(X[kp].astype(np.float32))
        yp.append(pn[kp].astype(float))
        if (k + 1) % 20 == 0:
            print(f"  学習 {k+1}/{ntr}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)
    yA = np.concatenate(yf)
    XB = np.concatenate(Xp).astype(np.float64)
    yB = np.concatenate(yp)
    mu, sd_ = XA.mean(0), XA.std(0)
    sd_[sd_ < 1e-12] = 1.0
    w_fill = fit_logit((XA - mu) / sd_, yA)
    Z = np.column_stack([np.ones(XB.shape[0]), (XB - mu) / sd_])
    w_pnl, *_ = np.linalg.lstsq(Z, yB, rcond=None)
    print(f"ロジット {XA.shape[0]:,} 行(約定率 {yA.mean()*100:.2f}%) / "
          f"損益モデル {XB.shape[0]:,} 行(平均 {yB.mean():+.4f} bp)", flush=True)
    json.dump({"names": NAMES, "mu": mu.tolist(), "sd": sd_.tolist(),
               "w_fill": w_fill.tolist(), "w_pnl": w_pnl.tolist(),
               "n_train_days": ntr},
              open(DATA / f"mmgate_model_{tag}.json", "w"))

    # ---- 適用(全日)----------------------------------------------------
    out, fit = [], []
    for k, dt in enumerate(days):
        G = P.filter(pl.col("dt") == dt).sort("t")
        r = day_design(tag, dt, G)
        if r is None:
            continue
        X, fl, pn = r
        Z = np.column_stack([np.ones(X.shape[0]), (X - mu) / sd_])
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w_fill, -30, 30)))
        ev = p * (Z @ w_pnl)
        out.append(pl.DataFrame({
            "dt": [dt] * ev.size, "t": G["t"].to_numpy(),
            "side": G["side"].to_numpy(),
            "ev": ev.astype(np.float32),
            "pass": (ev > 0).astype(np.int8)}))
        y = np.where(fl > 0, np.nan_to_num(pn), 0.0)     # 出さなければ 0
        fit.append({"dt": dt, "train": int(k < ntr), "n": int(ev.size),
                    "p_pass": float(np.mean(ev > 0)),
                    "ev_mean": float(ev.mean()),
                    "real_all": float(y.mean()),
                    "real_pass": float(y[ev > 0].mean()) if (ev > 0).any() else np.nan})
        if (k + 1) % 20 == 0:
            print(f"  適用 {k+1}/{len(days)}", flush=True)
    GA = pl.concat(out)
    GA.write_parquet(DATA / f"entrygate_{tag}_mmg.parquet", compression="zstd")
    # 帰無対照(同じ本数を無作為に通す門)は build_mmgate_null.py が作る
    Fd = pl.DataFrame(fit)
    Fd.write_csv(DATA / f"mmgate_fit_{tag}.csv")
    ev = Fd.filter(pl.col("train") == 0)
    print(f"\n門を通す割合 学習 {Fd.filter(pl.col('train')==1)['p_pass'].mean()*100:.1f}%"
          f" / 評価 {ev['p_pass'].mean()*100:.1f}%")
    print(f"1 発注あたり実現損益(評価期間) 門なし {ev['real_all'].mean():+.5f} bp"
          f" / 門あり {ev['real_pass'].mean():+.5f} bp")
    print(f"書き出し data/entrygate_{tag}_mmg.parquet")


if __name__ == "__main__":
    main()
