"""入口の門を**往復損益**で学習し直す(項目 5)。

    uv run python scripts/build_entrygate.py --coin xyz:MU

これまでの目的変数は markout(手仕舞いの費用なし)だった。ここでは
**出口の方策を先に固定**したうえで、その方策のもとで実現した

    Y = PnL_roundtrip   (在庫 FIFO 相殺・上限 |q|=1・BBO 追随・時間切れなし)

を入口の目的変数にする。予測は

    EV_entry = P(Fill_entry | X_t) × E[PnL_roundtrip | Fill, X_t, 出口方策]

で、**EV_entry > 0 のときだけ在庫を増やす側を出す**(在庫を減らす側は常に出す)。

時間契約
--------
- X_t は発注時点までに観測できる値のみ(候補テーブルの 45 列)
- 学習は前 59 日、評価は後 39 日。標準化・係数はすべて学習期間だけから
- 出口の方策は学習の前に固定してあり、目的変数はその方策の実現値
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fillpnl import BULK, DATA, NEED, SUB_FIT, TRAIN_FRAC, design
from plot_latency import fit_logit  # noqa: E402

# NEED には既に "side" が入っているので重複させない
COLS = NEED + [c for c in ("ts", "side") if c not in NEED]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sfx", default="_q1")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    P = pl.read_parquet(DATA / f"inv_posts_{tag}{a.sfx}.parquet")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    tr_days = {f.stem.split("=")[1] for f in files[:ntr]}
    print(f"発注 {P.height:,} 件 / 学習 {len(tr_days)} 日", flush=True)

    def feats(f, G):
        """その日の候補テーブルから、発注時刻・側に対応する行の説明変数を取る。"""
        T = pl.read_parquet(f, columns=COLS)
        X = design(T)
        ts = T["ts"].cast(pl.Int64).to_numpy()
        sd = T["side"].to_numpy()
        out_x, out_i = [], []
        for sg in (1, -1):
            m = sd == sg
            tt = ts[m]
            g = G.filter(pl.col("side") == sg)
            gt = g["t"].to_numpy()
            j = np.searchsorted(tt, gt)
            ok = (j < tt.size) & (tt[np.minimum(j, tt.size - 1)] == gt)
            idx = np.flatnonzero(m)[np.minimum(j, tt.size - 1)]
            out_x.append(X[idx[ok]])
            out_i.append((g["filled"].to_numpy()[ok],
                          g["rt_pnl"].to_numpy()[ok]))
        return (np.concatenate(out_x),
                np.concatenate([o[0] for o in out_i]),
                np.concatenate([o[1] for o in out_i]))

    Xs, ys, ps, ms = [], [], [], []
    for k, f in enumerate(files[:ntr]):
        dt = f.stem.split("=")[1]
        G = P.filter(pl.col("dt") == dt)
        if not G.height:
            continue
        X, fl, pn = feats(f, G)
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32)); ys.append(fl[sl].astype(float))
        ms.append(np.zeros(X[sl].shape[0], bool))
        kp = (fl > 0) & np.isfinite(pn)
        Xs.append(X[kp].astype(np.float32))
        ys.append(np.full(int(kp.sum()), np.nan))
        ps.append(pn[kp].astype(float)); ms.append(np.ones(int(kp.sum()), bool))
        if (k + 1) % 20 == 0:
            print(f"  学習 {k+1}/{ntr}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)
    yA = np.concatenate(ys); isf = np.concatenate(ms); PN = np.concatenate(ps)
    del Xs, ys, ps, ms
    base = XA[~isf]
    mu = base.mean(axis=0); sd_ = base.std(axis=0); sd_[sd_ < 1e-12] = 1.0
    print(f"ロジット {int((~isf).sum()):,} 行 (約定率 "
          f"{100*float(yA[~isf].mean()):.2f}%)", flush=True)
    w_fill = fit_logit((XA[~isf] - mu) / sd_, yA[~isf])
    print(f"往復損益モデル {int(isf.sum()):,} 行 (平均 {PN.mean():+.4f} bp)",
          flush=True)
    Z = np.column_stack([np.ones(int(isf.sum())), (XA[isf] - mu) / sd_])
    w_pnl, *_ = np.linalg.lstsq(Z, PN, rcond=None)
    del XA, yA, isf, PN, base

    def ev(X):
        Z = np.column_stack([np.ones(X.shape[0]), (X - mu) / sd_])
        p = 1.0 / (1.0 + np.exp(-(Z @ w_fill)))
        return p * (Z @ w_pnl), p, Z @ w_pnl

    rows = []
    stat = {"n": 0, "pass": 0}
    for k, f in enumerate(files):
        dt = f.stem.split("=")[1]
        G = P.filter(pl.col("dt") == dt)
        if not G.height:
            continue
        T = pl.read_parquet(f, columns=COLS)
        X = design(T)
        ts = T["ts"].cast(pl.Int64).to_numpy()
        sdv = T["side"].to_numpy()
        for sg in (1, -1):
            m = sdv == sg
            tt = ts[m]
            g = G.filter(pl.col("side") == sg).sort("t")
            gt = g["t"].to_numpy()
            j = np.clip(np.searchsorted(tt, gt), 0, tt.size - 1)
            idx = np.flatnonzero(m)[j]
            e, _, _ = ev(X[idx])
            rows.append(pl.DataFrame({"dt": [dt] * gt.size, "t": gt,
                                      "side": np.full(gt.size, sg, np.int8),
                                      "ev": e.astype(np.float32),
                                      "pass": (e > 0).astype(np.int8)}))
            stat["n"] += gt.size
            stat["pass"] += int((e > 0).sum())
        if (k + 1) % 20 == 0:
            print(f"  予測 {k+1}/{len(files)}", flush=True)
    GA = pl.concat(rows)
    GA.write_parquet(DATA / f"entrygate_{tag}{a.sfx}.parquet")
    print(f"\n門を通る割合 {100*stat['pass']/stat['n']:.2f}% "
          f"({stat['pass']:,}/{stat['n']:,})")
    te = GA.join(P.select("dt", "t", "side", "filled", "rt_pnl"),
                 on=["dt", "t", "side"]).filter(
        ~pl.col("dt").is_in(list(tr_days)))
    for nm, F in (("評価期間 全体", te),
                  ("評価期間 門を通った分", te.filter(pl.col("pass") == 1))):
        f_ = F.filter(pl.col("filled") == 1)
        v = f_["rt_pnl"].drop_nulls().to_numpy()
        print(f"{nm}: 発注 {F.height:,}  約定 {f_.height:,} "
              f"({100*f_.height/max(F.height,1):.2f}%)  "
              f"往復損益 平均 {v.mean():+.4f} bp  中央 {np.median(v):+.4f}  "
              f"正 {100*float((v>0).mean()):.1f}%")
    print(f"\n書き出し {DATA}/entrygate_{tag}{a.sfx}.parquet")


if __name__ == "__main__":
    main()
