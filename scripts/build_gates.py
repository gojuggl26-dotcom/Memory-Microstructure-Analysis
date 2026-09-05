"""陽性領域の中だけで OBI / OFI / スプレッドの門を順に足す。

    uv run python scripts/build_gates.py --coin xyz:MU

やり方(重要)
--------------
100 セル全体から改めて最適な 5 次元セルを探すと、それは探索そのものになる。
そこで

  1. **候補領域を学習期間だけで固定する**。領域は予測 EV の符号、すなわち
     `Qhat > 0`(= 予測損益が正)。これは学習期間で当てはめたモデルの出力
     だけで決まり、評価期間を一切見ない
  2. その領域の中で `OBI*` → `OFI*` → `Spread` の順に門を足す。
     閾値は**学習期間の五分位のうち学習期間の EV を最大にするもの**を選ぶ
     (門を掛けない選択肢も含める。発火率の下限 5% を課す)
  3. 選んだ門を**評価期間へ一度だけ**当てて ΔEV を見る

目的変数は 2 本並べる:

  - markout のみ(手仕舞いの費用なし。既報と接続するため)
  - **往復**(Hybrid・T_max=10s。手仕舞いの費用込み。これが本命)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fillpnl import (BULK, DATA, H, NW_LAGS, SUB_FIT, TRAIN_FRAC,  # noqa
                           design, labels, load)
from plot_latency import fit_logit  # noqa: E402

NG_Q = 5              # 門の分位数
MIN_RATE = 0.05       # 発火率の下限(一度も発火しない門を「最良」にしないため)
GATES = [("obi1", "OBI"), ("ofi_1s", "OFI 1s"), ("spread_bp", "スプレッド")]
Y_MK = "label_pnl_1s_bp"
Y_RT = "rt_hyb_10s_bp"


def nwse(v):
    n = v.size
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, NW_LAGS + 1):
        g += 2.0 * (1.0 - lg / (NW_LAGS + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n))


def collect(files, phat, qhat, gcols):
    """日ごとに (領域内の) 候補を集める。領域は Qhat > 0。"""
    out = []
    for f in files:
        T = load(f)
        X = design(T)
        y, P, _ = labels(T)
        q = qhat(X)
        m = q > 0
        if not m.any():
            continue
        d = {"dt": f.stem.split("=")[1], "y": y[m],
             "mk": np.where(y[m] > 0, np.nan_to_num(P[Y_MK][m]), 0.0),
             "rt": np.where(y[m] > 0, np.nan_to_num(P[Y_RT][m]), 0.0),
             "rt_ok": ((y[m] > 0) & np.isfinite(P[Y_RT][m])).astype(float)}
        for c, _ in gcols:
            d[c] = np.nan_to_num(T[c].to_numpy().astype(np.float64))[m]
        out.append(d)
        del T, X
    return out


def daily(rows, mask_fn, key):
    """日ごとの 1 候補あたり損益の系列と、通算の値を返す。"""
    v, ntot, stot, ftot = [], 0, 0.0, 0
    for d in rows:
        k = mask_fn(d)
        n = int(k.sum())
        if n == 0:
            continue
        s = float(d[key][k].sum())
        v.append(s / n)
        ntot += n
        stot += s
        ftot += int(d["y"][k].sum())
    return np.array(v), ntot, stot, ftot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    tr, te = files[:ntr], files[ntr:]
    print(f"学習 {len(tr)} 日 / 評価 {len(te)} 日", flush=True)

    Xs, ys, ps, ms = [], [], [], []
    for k, f in enumerate(tr):
        T = load(f)
        X = design(T)
        y, P, _ = labels(T)
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32)); ys.append(y[sl])
        ms.append(np.zeros(X[sl].shape[0], bool))
        keep = y > 0
        Xs.append(X[keep].astype(np.float32))
        ys.append(np.full(int(keep.sum()), np.nan))
        ps.append(P[Y_MK][keep]); ms.append(np.ones(int(keep.sum()), bool))
        del T, X
        if (k + 1) % 20 == 0:
            print(f"  学習読み込み {k+1}/{len(tr)}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)
    yA = np.concatenate(ys); isf = np.concatenate(ms); PN = np.concatenate(ps)
    del Xs, ys, ps, ms
    base = XA[~isf]
    mu = base.mean(axis=0); sd = base.std(axis=0); sd[sd < 1e-12] = 1.0
    w_fill = fit_logit((XA[~isf] - mu) / sd, yA[~isf])
    Zf = np.column_stack([np.ones(int(isf.sum())), (XA[isf] - mu) / sd])
    w_pnl, *_ = np.linalg.lstsq(Zf, PN, rcond=None)
    del XA, yA, isf, PN, base, Zf

    def phat(X):
        return 1.0 / (1.0 + np.exp(
            -(np.column_stack([np.ones(X.shape[0]), (X - mu) / sd]) @ w_fill)))

    def qhat(X):
        return np.column_stack([np.ones(X.shape[0]), (X - mu) / sd]) @ w_pnl

    print("学習期間を集計中…", flush=True)
    TR = collect(tr, phat, qhat, GATES)
    print("評価期間を集計中…", flush=True)
    TE = collect(te, phat, qhat, GATES)

    # 門の閾値は学習期間の分位から。評価期間は一切見ない
    allv = {c: np.concatenate([d[c] for d in TR]) for c, _ in GATES}
    edges = {c: np.quantile(allv[c], np.linspace(0, 1, NG_Q + 1)[1:-1])
             for c, _ in GATES}
    n_tr = sum(d["y"].size for d in TR)
    n_te = sum(d["y"].size for d in TE)
    print(f"\n領域 Qhat>0: 学習 {n_tr:,} 候補 / 評価 {n_te:,} 候補", flush=True)

    lines = []

    def show(name, fn, rows, tagname):
        for key, lab in ((Y_MK, "markout"), (Y_RT, "往復")):
            k2 = "mk" if key == Y_MK else "rt"
            v, n, s, nf = daily(rows, fn, k2)
            if v.size < 5:
                continue
            m_, se_ = nwse(v)
            lines.append({"stage": name, "set": tagname, "obj": lab, "n": n,
                          "fills": nf, "ev": m_, "se": se_,
                          "per_fill": s / max(nf, 1)})
            print(f"  [{tagname}] {name:38s} {lab:8s} "
                  f"N {n:>10,}  約定 {nf:>8,}  "
                  f"EV {m_:+.5f} ± {se_:.5f} bp  1 約定 {s/max(nf,1):+.3f} bp")

    # 段階 0: 領域そのもの
    print("\n段階 0 — 領域 Qhat>0")
    show("gate なし", lambda d: np.ones(d["y"].size, bool), TR, "学習")
    show("gate なし", lambda d: np.ones(d["y"].size, bool), TE, "評価")

    # 各変数の五分位を領域内で見る(形を確認するため)
    print("\n領域内の五分位(評価期間・往復)")
    qt = []
    for c, nm in GATES:
        for k in range(NG_Q):
            def fn(d, c=c, k=k):
                b = np.searchsorted(edges[c], d[c], side="right")
                return b == k
            v, n, s, nf = daily(TE, fn, "rt")
            v2, _, s2, _ = daily(TE, fn, "mk")
            if v.size < 5:
                continue
            m_, se_ = nwse(v)
            qt.append({"var": nm, "q": k + 1, "n": n, "fills": nf,
                       "ev_rt": m_, "se_rt": se_, "ev_mk": float(v2.mean()),
                       "per_fill_rt": s / max(nf, 1)})
            print(f"  {nm:10s} Q{k+1}  N {n:>9,}  約定 {nf:>7,}  "
                  f"往復 EV {m_:+.5f} ± {se_:.5f}  markout EV {v2.mean():+.5f}")

    # 段階 1〜3: 学習期間で閾値を選び、評価期間へ一度だけ当てる
    chosen = {}
    cur_tr = lambda d: np.ones(d["y"].size, bool)     # noqa: E731
    cur_te = lambda d: np.ones(d["y"].size, bool)     # noqa: E731
    for st, (c, nm) in enumerate(GATES, 1):
        best = (-9e9, None)
        for k in range(NG_Q):           # 「Qk 以上を残す」。k=0 は門を掛けない
            def fn(d, c=c, k=k, prev=cur_tr):
                b = np.searchsorted(edges[c], d[c], side="right")
                return prev(d) & (b >= k)
            v, n, s, nf = daily(TR, fn, "rt")
            if v.size < 5 or n < MIN_RATE * n_tr:
                continue
            if s / n > best[0]:
                best = (s / n, k)
        k = best[1] if best[1] is not None else 0
        chosen[c] = k
        thr = "なし" if k == 0 else f"Q{k+1} 以上 (>{edges[c][k-1]:+.4g})"
        print(f"\n段階 {st} — {nm} の門: 学習期間の選択 = {thr} "
              f"(学習 EV {best[0]:+.5f})")
        cur_tr = (lambda d, c=c, k=k, prev=cur_tr:
                  prev(d) & (np.searchsorted(edges[c], d[c], side="right") >= k))
        cur_te = (lambda d, c=c, k=k, prev=cur_te:
                  prev(d) & (np.searchsorted(edges[c], d[c], side="right") >= k))
        show(f"+{nm}", cur_tr, TR, "学習")
        show(f"+{nm}", cur_te, TE, "評価")

    pl.DataFrame(lines).write_csv(DATA / f"gates_stages_{tag}.csv")
    pl.DataFrame(qt).write_csv(DATA / f"gates_quintiles_{tag}.csv")
    print(f"\n書き出し {DATA}/gates_stages_{tag}.csv, gates_quintiles_{tag}.csv")


if __name__ == "__main__":
    main()
