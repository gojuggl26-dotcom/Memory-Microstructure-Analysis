"""xfeat パネルの予測力を 7 ホライズンで測る(bbo+fills 版、単層)。

    uv run python scripts/fit_xfeat.py --coin xyz:AMD [--stage pool|daily]

測り方は [l4feat](fit_l4feat.py) と同じ:
  * 順位相関は **midrank**(同順位を平均)。これを省くと 100ms の
    ゼロだらけの目的変数で偽の相関が出る([spearman-tie トラップ])
  * r_fwd 前向き / r_bwd 後ろ向き(同時性)/ r_plc 帰無対照(日内 1 時間ずらし)
    / r_1st,r_2nd(日で前半後半)を別欄で出す
  * 重なりを避け、短い h は 20 秒おき(標本 A)、長い h は 60 秒おき(標本 B)
  * クロス行は build 時に除外済み

出力: data/xfeat_pred_<tag>.csv / data/xfeat_daily_<tag>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = Path("E:/Memory-xfeat")
NS = 86_400
FLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]
STRIDE = {"100ms": 20, "500ms": 20, "1s": 20, "5s": 20,
          "10s": 60, "30s": 60, "60s": 60}
SETS = {"A": 20, "B": 60}
HSET = {h: next(k for k, v in SETS.items() if v == STRIDE[h]) for h in FLAB}
PLACEBO = 3600
BATCH = 80


def rank(x):
    o = np.argsort(x, kind="stable")
    xs = x[o]
    b = np.empty(x.size + 1, bool)
    b[0] = b[-1] = True
    np.not_equal(xs[1:], xs[:-1], out=b[1:-1])
    idx = np.flatnonzero(b)
    avg = (idx[:-1] + idx[1:] - 1) / 2.0
    r = np.empty(x.size, np.float64)
    r[o] = np.repeat(avg, np.diff(idx))
    return r


def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 200:
        return np.nan, n
    a, b = rank(x[m]), rank(y[m])
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / d) if d > 0 else np.nan), n


def crank(x):
    """中心化した順位ベクトル(全行が有限であることを前提)。"""
    r = rank(x)
    return r - r.mean()


def dot(a, b):
    """中心化済み順位どうしの Spearman。"""
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def tval(r, n):
    if not (isinstance(r, float) or np.isfinite(r)) or n < 3:
        return np.nan
    return r * np.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))


def day_list(tag):
    return sorted(p.name[3:13] for p in (SRC / tag).glob("dt=*.parquet"))


def cols_of(tag, days):
    fam = {}
    f = DATA / f"xfeat_cols_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        fam = dict(zip(d["col"].to_list(), d["family"].to_list()))
    n = pl.scan_parquet(SRC / tag / f"dt={days[0]}.parquet").collect_schema().names()
    cols = [c for c in n if not c.startswith("label_") and c not in ("dt", "sec")]
    return cols, fam


def positions():
    pos, plc = {}, {}
    for k, st in SETS.items():
        p = np.arange(0, NS, st)
        pos[k] = p
        plc[k] = (p + PLACEBO) % NS
    return pos, plc


def read_block(tag, days, cols, pos, plc=None):
    out = {k: {c: [] for c in cols} for k in SETS}
    outp = {k: {c: [] for c in cols} for k in SETS} if plc is not None else None
    for d in days:
        t = pl.read_parquet(SRC / tag / f"dt={d}.parquet", columns=cols)
        for c in cols:
            v = t[c].to_numpy()
            for k in SETS:
                out[k][c].append(v[pos[k]])
                if plc is not None:
                    outp[k][c].append(v[plc[k]])
    out = {k: {c: np.concatenate(v) for c, v in dd.items()} for k, dd in out.items()}
    if plc is not None:
        outp = {k: {c: np.concatenate(v) for c, v in dd.items()} for k, dd in outp.items()}
    return out, outp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--stage", default="pool")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    days = day_list(tag)
    nday = len(days)
    cols, fam = cols_of(tag, days)
    print(f"{tag}: {nday} 日 / 特徴量 {len(cols)} 本 × {len(FLAB)} = "
          f"{len(cols)*len(FLAB)} セル", flush=True)
    pos, plc = positions()
    half = nday // 2
    dayv = {k: np.repeat(np.arange(nday), pos[k].size) for k in SETS}
    lb = [f"label_micro_{h}" for h in FLAB] + [f"label_mid_{h}" for h in FLAB] \
        + [f"mret_{h}" for h in FLAB]
    Y, _ = read_block(tag, days, lb, pos)

    # ★目的変数に 1 つでも非有限がある行は先に落とす。こうすると、説明変数が
    #   全て有限な列(大多数)では突合の行集合が列によらず同じになり、
    #   **目的変数の順位を 1 度だけ計算して使い回せる**。
    #   同じ順位相関を、再ランキングを避けて厳密に出すための整理。
    keep = {}
    for k in SETS:
        m = np.ones(Y[k][lb[0]].size, bool)
        for c in lb:
            m &= np.isfinite(Y[k][c])
        keep[k] = m
        Y[k] = {c: v[m] for c, v in Y[k].items()}
        dayv[k] = dayv[k][m]
        print(f"  標本 {k}: {int(m.sum()):,} 行(目的変数の欠損 {int((~m).sum()):,} 行を除外)",
              flush=True)
    RY = {k: {c: crank(Y[k][c]) for c in lb} for k in SETS}

    if a.stage == "pool":
        rows = []
        for k0 in range(0, len(cols), BATCH):
            sub = cols[k0:k0 + BATCH]
            X, XP = read_block(tag, days, sub, pos, plc)
            for c in sub:
                for h in FLAB:
                    k = HSET[h]
                    x, xp = X[k][c][keep[k]], XP[k][c][keep[k]]
                    y = Y[k][f"label_micro_{h}"]
                    n = int(np.isfinite(x).sum())
                    if np.isfinite(x).all():
                        rx = crank(x)
                        r = dot(rx, RY[k][f"label_micro_{h}"])
                        rm = dot(rx, RY[k][f"label_mid_{h}"])
                        rb = dot(rx, RY[k][f"mret_{h}"])
                    else:                       # 欠損のある列だけ従来どおり
                        r, n = sp(x, y)
                        rm, _ = sp(x, Y[k][f"label_mid_{h}"])
                        rb, _ = sp(x, Y[k][f"mret_{h}"])
                    rp, _ = sp(xp, y)
                    if h in ("1s", "10s"):      # 前半後半は報告する 2 地平だけ
                        d1 = dayv[k] < half
                        r1, _ = sp(x[d1], y[d1])
                        r2, _ = sp(x[~d1], y[~d1])
                    else:
                        r1 = r2 = np.nan
                    rows.append(dict(col=c, family=fam.get(c, "?"), h=h, n=n,
                                     r_fwd=r, r_mid=rm, r_bwd=rb, r_plc=rp,
                                     r_1st=r1, r_2nd=r2, t=tval(r, n)))
            print(f"  {min(k0+BATCH,len(cols))}/{len(cols)}", flush=True)
        pl.DataFrame(rows).write_csv(DATA / f"xfeat_pred_{tag}.csv")
        d = pl.DataFrame(rows).filter(pl.col("h") == "1s").with_columns(
            ab=pl.col("r_fwd").abs()).sort("ab", descending=True)
        print(d.head(10).select("col", "family", "r_fwd", "r_bwd", "r_plc"))

    elif a.stage == "daily":
        pred = pl.read_csv(DATA / f"xfeat_pred_{tag}.csv")
        sel = []
        for h in ("1s", "10s"):
            sel += (pred.filter(pl.col("h") == h).with_columns(ab=pl.col("r_fwd").abs())
                    .sort("ab", descending=True).head(a.top)["col"].to_list())
        sel = list(dict.fromkeys(sel))
        X, _ = read_block(tag, days, sel, pos)
        rows = []
        for c in sel:
            for h in ("1s", "10s"):
                k = HSET[h]
                x, y = X[k][c][keep[k]], Y[k][f"label_micro_{h}"]
                rs = []
                for dd in range(nday):
                    m = dayv[k] == dd
                    r, _ = sp(x[m], y[m])
                    if np.isfinite(r):
                        rs.append(r)
                rs = np.array(rs)
                if rs.size:
                    rows.append(dict(col=c, family=fam.get(c, "?"), h=h,
                                     nday=rs.size, mean=float(rs.mean()),
                                     med=float(np.median(rs)),
                                     pos=int((rs > 0).sum()),
                                     worst=float(rs.min()), best=float(rs.max())))
        pl.DataFrame(rows).write_csv(DATA / f"xfeat_daily_{tag}.csv")
        print("日次を書き出した", flush=True)


if __name__ == "__main__":
    main()
