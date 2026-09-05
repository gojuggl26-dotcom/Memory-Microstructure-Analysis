"""共通分母構造をほどいた特徴量で、損益モデルを 4 通り比べる。

    uv run python scripts/build_fillpnl_models.py --coin xyz:MU

なぜ
----
`fragility = Cancel/Depth` と `resilience = Add/Depth` は分母が同じで
ほぼ同値なので、全変数版では係数が ±8.8 まで暴れていた(多重共線性)。
これを

    FlowBalance    = log((Add + eps) / (Cancel + eps))     … 正味の補充
    LiquidityScale = log(Depth)                            … 板の水準
    FlowActivity   = log1p(Add + Cancel)                   … 流量の大きさ

へ分解し直す。そのうえで損益モデルを

    OLS / Ridge / ElasticNet / GAM(加法・各変数を訓練期間の十分位で段階関数化)

の 4 通りで当てはめ、**標本外**で島が残るかを比べる。
Ridge で残るなら、多重共線性への懸念はかなり軽くなる。

時間契約は `build_fillpnl.py` と同一(学習 59 日 / 評価 39 日、
標準化・分位・罰則の強さ・十分位の境界はすべて学習期間だけから)。
罰則の強さは**学習期間をさらに前 45 日 / 後 14 日に割った内側の分割**で選ぶ。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fillpnl import (DATA, H, LABS, MIN_N_DAY, MKH, NEED, NEED_MU,  # noqa
                           NQ, NW_LAGS, POS, RAW, SEED, SGN, SUB_FIT,
                           TRAIN_FRAC, BULK, load)
from plot_latency import fit_logit  # noqa: E402

EPS = 1e-3
INNER = 45            # 学習 59 日のうち前 45 日で当てはめ、後 14 日で罰則を選ぶ
NBIN = 10             # GAM の段階数
LAM = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5]
ENET = [1e-4, 1e-3, 1e-2, 1e-1]
CHUNK = 200_000

# 共通分母を持つ 3 つを外し、分解した量に置き換える
DROP = ["fragility", "resilience", "cancel_10s", "add_10s"]
POS2 = [c for c in POS if c not in DROP]
NAMES2 = ([f"log_{c}" for c in POS2] + [f"slog_{c}" for c in SGN] + RAW
          + ["obi_l2", "obi_l2_na", "inv_tau", "hour_sin", "hour_cos",
             "flow_balance_own", "flow_balance_all", "flow_activity"])


def design2(T: pl.DataFrame) -> np.ndarray:
    """v2 の説明変数。t 以降の情報は使わない。"""
    cols = []
    for c in POS2:
        v = T[c].to_numpy().astype(np.float64)
        cols.append(np.log1p(np.maximum(np.nan_to_num(v, nan=0.0), 0.0)))
    for c in SGN:
        v = np.nan_to_num(T[c].to_numpy().astype(np.float64), nan=0.0)
        cols.append(np.sign(v) * np.log1p(np.abs(v)))
    for c in RAW:
        cols.append(np.nan_to_num(T[c].to_numpy().astype(np.float64), nan=0.0))
    l2 = T["obi_l2"].to_numpy().astype(np.float64)
    cols.append(np.nan_to_num(l2, nan=0.0))
    cols.append(np.isnan(l2).astype(np.float64))
    tau = T["tau_hat_s"].to_numpy().astype(np.float64)
    cols.append(1.0 / (1.0 + np.where(np.isfinite(tau), np.maximum(tau, 0.0),
                                      np.inf)))
    hh = T["hour_utc"].to_numpy().astype(np.float64)
    cols.append(np.sin(2 * np.pi * hh / 24.0))
    cols.append(np.cos(2 * np.pi * hh / 24.0))
    # ★ 共通分母をほどく。fragility/resilience は分母が同じなので比を取ると消える
    fr = np.nan_to_num(T["fragility"].to_numpy().astype(np.float64))
    rs = np.nan_to_num(T["resilience"].to_numpy().astype(np.float64))
    ca = np.nan_to_num(T["cancel_10s"].to_numpy().astype(np.float64))
    ad = np.nan_to_num(T["add_10s"].to_numpy().astype(np.float64))
    cols.append(np.log((rs + EPS) / (fr + EPS)))            # 自分側の正味補充
    cols.append(np.log((ad + EPS) / (ca + EPS)))            # 板全体の正味補充
    cols.append(np.log1p(np.maximum(ad + ca, 0.0)))         # 流量の大きさ
    X = np.column_stack(cols)
    assert X.shape[1] == len(NAMES2), (X.shape, len(NAMES2))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def ridge(Z, y, lam):
    """切片は罰しない閉形式。"""
    n, p = Z.shape
    A = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    for i in range(0, n, CHUNK):
        W = np.column_stack([np.ones(min(CHUNK, n - i)), Z[i:i + CHUNK]])
        A += W.T @ W
        b += W.T @ y[i:i + CHUNK]
    A[1:, 1:] += lam * np.eye(p)
    return np.linalg.solve(A, b)


def enet(Z, y, alpha, l1=0.5):
    """座標降下。標準化済みなので列ノルムは揃っている。"""
    n, p = Z.shape
    G = np.zeros((p, p))
    c = np.zeros(p)
    ym = y.mean()
    for i in range(0, n, CHUNK):
        W = Z[i:i + CHUNK]
        G += W.T @ W
        c += W.T @ (y[i:i + CHUNK] - ym)
    G /= n
    c /= n
    d = np.diag(G).copy()
    w = np.zeros(p)
    for _ in range(300):
        w0 = w.copy()
        for j in range(p):
            r = c[j] - G[j] @ w + d[j] * w[j]
            s = max(abs(r) - alpha * l1, 0.0) * np.sign(r)
            w[j] = s / (d[j] + alpha * (1 - l1))
        if np.max(np.abs(w - w0)) < 1e-7:
            break
    return np.concatenate([[ym - 0.0], w])       # Z は中心化済み前提


def bin_basis(X, edges):
    """各変数を訓練期間の分位で NBIN 段に離散化し、段差を並べる(加法モデル)。"""
    cols = []
    for j in range(X.shape[1]):
        b = np.searchsorted(edges[j], X[:, j], side="right")
        for k in range(1, NBIN):
            cols.append((b >= k).astype(np.float64))
    return np.column_stack(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    tr, te = files[:ntr], files[ntr:]
    print(f"学習 {len(tr)} 日 / 評価 {len(te)} 日  変数 {len(NAMES2)} 個", flush=True)

    Xs, ys, ps, ms, ds = [], [], [], [], []
    for k, f in enumerate(tr):
        T = load(f)
        X = design2(T)
        lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
        pnl = T["label_pnl_1s_bp"].to_numpy().astype(np.float64)
        y = (np.isfinite(lat) & (lat <= H) & np.isfinite(pnl)).astype(np.float64)
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32)); ys.append(y[sl])
        ms.append(np.zeros(X[sl].shape[0], bool)); ds.append(np.full(X[sl].shape[0], k))
        keep = y > 0
        Xs.append(X[keep].astype(np.float32)); ys.append(np.full(int(keep.sum()), np.nan))
        ps.append(pnl[keep]); ms.append(np.ones(int(keep.sum()), bool))
        ds.append(np.full(int(keep.sum()), k))
        del T, X
        if (k + 1) % 20 == 0:
            print(f"  学習読み込み {k+1}/{len(tr)}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)
    yA = np.concatenate(ys)
    isf = np.concatenate(ms)
    dayi = np.concatenate(ds)
    PN = np.concatenate(ps)
    del Xs, ys, ps, ms, ds

    base = XA[~isf]
    mu = base.mean(axis=0); sdv = base.std(axis=0); sdv[sdv < 1e-12] = 1.0
    Zh = (XA[~isf] - mu) / sdv
    print(f"ロジット学習 {Zh.shape[0]:,} 行 "
          f"(250ms 約定 {int(yA[~isf].sum()):,})", flush=True)
    w_fill = fit_logit(Zh, yA[~isf])

    Zf = (XA[isf] - mu) / sdv
    dfi = dayi[isf]
    tr_in, va_in = dfi < INNER, dfi >= INNER
    print(f"損益モデル学習 {Zf.shape[0]:,} 行 "
          f"(内側 学習 {int(tr_in.sum()):,} / 検証 {int(va_in.sum()):,})", flush=True)
    edges = [np.quantile(Zf[tr_in, j], np.linspace(0, 1, NBIN + 1)[1:-1])
             for j in range(Zf.shape[1])]
    edges = [np.unique(e) for e in edges]

    def r2(pred, y):
        return 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    fits = {}
    # OLS
    w = ridge(Zf[tr_in], PN[tr_in], 0.0)
    fits["OLS"] = ("lin", ridge(Zf, PN, 0.0), r2(
        np.column_stack([np.ones(int(va_in.sum())), Zf[va_in]]) @ w, PN[va_in]))
    # Ridge
    best = (-9e9, None)
    for lam in LAM:
        w = ridge(Zf[tr_in], PN[tr_in], lam)
        s = r2(np.column_stack([np.ones(int(va_in.sum())), Zf[va_in]]) @ w,
               PN[va_in])
        if s > best[0]:
            best = (s, lam)
    fits["Ridge"] = ("lin", ridge(Zf, PN, best[1]), best[0])
    print(f"  Ridge の罰則 lambda={best[1]:g} (内側 R2 {best[0]:+.5f})", flush=True)
    # ElasticNet
    best = (-9e9, None)
    for al in ENET:
        w = enet(Zf[tr_in], PN[tr_in], al)
        s = r2(np.column_stack([np.ones(int(va_in.sum())), Zf[va_in]]) @ w,
               PN[va_in])
        if s > best[0]:
            best = (s, al)
    fits["ElasticNet"] = ("lin", enet(Zf, PN, best[1]), best[0])
    print(f"  ElasticNet の alpha={best[1]:g} (内側 R2 {best[0]:+.5f})", flush=True)
    # GAM (加法・段階関数)
    Bt = bin_basis(Zf[tr_in], edges)
    best = (-9e9, None)
    Bv = bin_basis(Zf[va_in], edges)
    for lam in LAM[1:]:
        w = ridge(Bt, PN[tr_in], lam)
        s = r2(np.column_stack([np.ones(Bv.shape[0]), Bv]) @ w, PN[va_in])
        if s > best[0]:
            best = (s, lam)
    del Bt, Bv
    fits["GAM"] = ("bin", ridge(bin_basis(Zf, edges), PN, best[1]), best[0])
    print(f"  GAM の罰則 lambda={best[1]:g} (内側 R2 {best[0]:+.5f})", flush=True)

    def phat(X):
        return 1.0 / (1.0 + np.exp(
            -(np.column_stack([np.ones(X.shape[0]), (X - mu) / sdv]) @ w_fill)))

    def qhat(X, kind, w):
        Z = (X - mu) / sdv
        B = bin_basis(Z, edges) if kind == "bin" else Z
        return np.column_stack([np.ones(B.shape[0]), B]) @ w

    eh = np.linspace(0, 1, NQ + 1)[1:-1]
    e_p = np.quantile(phat(XA[~isf]), eh)
    e_q = {k: np.quantile(qhat(XA[~isf], v[0], v[1]), eh) for k, v in fits.items()}
    del XA, yA, isf, PN, Zf, Zh, base

    rng = np.random.default_rng(SEED)
    rows = []
    for k, f in enumerate(te):
        T = load(f)
        X = design2(T)
        lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
        P = {c: T[c].to_numpy().astype(np.float64) for c in LABS}
        y = (np.isfinite(lat) & (lat <= H)
             & np.isfinite(P["label_pnl_1s_bp"])).astype(np.float64)
        W = {c: np.where(y > 0, np.nan_to_num(P[c]), 0.0) for c in LABS}
        pf = phat(X)
        ip = np.searchsorted(e_p, pf, side="right")
        dt = f.stem.split("=")[1]
        L_ = NQ * NQ
        for nm, (kind, w, _) in fits.items():
            iq = np.searchsorted(e_q[nm], qhat(X, kind, w), side="right")
            cc = ip * NQ + iq
            n = np.bincount(cc, minlength=L_)
            D = {"dt": [dt] * L_, "kind": [nm] * L_,
                 "hd": np.repeat(np.arange(NQ), NQ).astype(np.int8),
                 "qd": np.tile(np.arange(NQ), NQ).astype(np.int8),
                 "n": n.astype(np.int64),
                 "n_fill": np.bincount(cc, weights=y, minlength=L_)}
            for c in LABS:
                D[f"s_{c}"] = np.bincount(cc, weights=W[c], minlength=L_)
            rows.append(pl.DataFrame(D))
        del T, X
        if (k + 1) % 10 == 0:
            print(f"  評価 {k+1}/{len(te)}", flush=True)
    D = pl.concat(rows)
    D.write_parquet(DATA / f"fillpnl_models_days_{tag}.parquet")

    zb = 3.481
    res = []
    for nm in fits:
        S = D.filter(pl.col("kind") == nm).sort("dt")
        P = (S.group_by("hd", "qd").agg(
            n=pl.col("n").sum(), n_fill=pl.col("n_fill").sum(),
            **{f"s_{c}": pl.col(f"s_{c}").sum() for c in LABS}).sort("hd", "qd"))
        ev, se = [], []
        for hd, qd in zip(P["hd"], P["qd"]):
            s = S.filter((pl.col("hd") == hd) & (pl.col("qd") == qd))
            m = s["n"].to_numpy() >= MIN_N_DAY
            v = (s["s_label_pnl_1s_bp"].to_numpy()
                 / np.maximum(s["n"].to_numpy(), 1))[m]
            if v.size < 5:
                ev.append(np.nan); se.append(np.nan); continue
            e = v - v.mean()
            g = float((e * e).sum() / v.size)
            for lg in range(1, NW_LAGS + 1):
                g += 2.0 * (1 - lg / (NW_LAGS + 1.0)) * float(
                    (e[lg:] * e[:-lg]).sum() / v.size)
            ev.append(float(v.mean())); se.append(float(np.sqrt(max(g, 0) / v.size)))
        ev = np.array(ev); se = np.array(se)
        ok = np.isfinite(ev) & np.isfinite(se)
        t = np.where(ok, ev / se, 0.0)
        win = ok & (ev > 0) & (t > zb)
        nn = P["n"].to_numpy().astype(float)
        sm = P["s_label_pnl_1s_bp"].to_numpy()
        su = P["s_label_pnlmu_1s_bp"].to_numpy()
        res.append({"model": nm, "inner_r2": fits[nm][2],
                    "pos": int(np.sum(ok & (ev > 0))), "pos_bonf": int(win.sum()),
                    "share_n": float(nn[win].sum() / nn.sum()) if win.any() else 0.0,
                    "ev_pos_mid": float(sm[win].sum() / nn[win].sum()) if win.any() else np.nan,
                    "ev_pos_micro": float(su[win].sum() / nn[win].sum()) if win.any() else np.nan,
                    "ev_all_mid": float(sm.sum() / nn.sum()),
                    "best_ev": float(np.nanmax(np.where(ok, ev, np.nan)))})
        P.with_columns(ev_daily=pl.Series(ev), se_daily=pl.Series(se)).write_parquet(
            DATA / f"fillpnl_models_{tag}_{nm}.parquet")
    R = pl.DataFrame(res)
    R.write_csv(DATA / f"fillpnl_models_{tag}.csv")
    print()
    print(R)
    json.dump({"names": NAMES2, "coef_fill": dict(zip(["const"] + NAMES2,
                                                      map(float, w_fill)))},
              open(DATA / f"fillpnl_models_meta_{tag}.json", "w"), indent=1)
    print(f"\n書き出し {DATA}/fillpnl_models_{tag}.csv ほか")


if __name__ == "__main__":
    main()
