"""特徴量パネルの予測力を 7 つのホライズンで測る。

    uv run python scripts/fit_l4feat.py --coin xyz:INTC --stage pool
    uv run python scripts/fit_l4feat.py --coin xyz:INTC --stage daily
    uv run python scripts/fit_l4feat.py --coin xyz:INTC --stage corr

測るもの
--------
  r_fwd    前向き   corr( x_T , log micro(T+h) − log micro(T) )   ← 予測
  r_mid    同上を mid 建てで測ったもの(参考)
  r_bwd    後ろ向き corr( x_T , log micro(T) − log micro(T−h) )   ← 同時性
  r_plc    帰無対照 x を **同じ日の中で 1 時間ずらして**同じ計算をする
  r_1st / r_2nd  日で前半・後半に割った値(行のシャッフルはしない)

★ r_fwd と r_bwd は必ず別の欄に書く。混ぜると「もう起きたこと」を
  「これから起きること」の予測力として報告してしまう。

重なりの扱い
------------
1 秒ごとの行で 60 秒先を測ると隣の標本が 59 秒ぶん重なり、自由度が
水増しされる。**ホライズンより広い間隔で間引いた 2 つの標本**を使う。

    標本 A … 20 秒おき(h ≤ 5s 用)  4,320 行/日 → 99 日で 42.8 万行
    標本 B … 60 秒おき(h ≥ 10s 用) 1,440 行/日 → 99 日で 14.3 万行

t 値はこの間引いた行数で計算する。標本 A を最短ホライズン(100ms)に
合わせて 1 秒おきにすると 855 万行になるが、順位づけの計算量が
特徴量 1,000 本 × 7 ホライズンで現実的でなくなる。20 秒おきでも
順位相関の標準誤差は 0.0015 で、Bonferroni 後に主張できる下限
(|r| ≒ 0.012)より十分小さい。

順位相関を使う理由
------------------
板の量は裾が非常に重い。Pearson だと少数の行が係数を支配するので
Spearman を主とする。

多重比較
--------
特徴量 × ホライズンのセル数を必ず報告し、Bonferroni の閾値を併記する。
「有意でない」は「効果がない」ではない(検出力が足りないだけかもしれない)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = Path("E:/Memory-l4feat")
NS = 86_400                       # 1 日の行数(1 秒格子)
FLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]
STRIDE = {"100ms": 20, "500ms": 20, "1s": 20, "5s": 20,
          "10s": 60, "30s": 60, "60s": 60}
SETS = {"A": 20, "B": 60}
HSET = {h: ("A" if STRIDE[h] == 5 else "B") for h in FLAB}
PLACEBO_SHIFT = 3600              # 帰無対照のずらし幅(秒)
BATCH = 60


def rank(x: np.ndarray) -> np.ndarray:
    """★同順位は平均順位にする(midrank)。これを省くと結果が壊れる。

    最初は `argsort` の結果をそのまま順位にしていた。すると**同じ値の行に
    元の並び順どおりの順位**が付く。100ms のリターンは同じ値(とくに 0)が
    大量に並ぶので、順位が実質「その日の中の時刻」になり、ゆっくり動く
    特徴量なら何でも r ≈ 0.65 が出た。**帰無対照(1 時間ずらし)まで
    0.657 になって**初めて気づいた。同順位を平均にすれば消える。
    """
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


def tval(r, n):
    r = float(r) if r == r else np.nan
    if not np.isfinite(r) or n < 3:
        return np.nan
    return r * np.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))


LAYERS = ["", "_life", "_post"]        # パネル / 寿命層 / 派生層


def dirs(tag: str) -> list[Path]:
    return [SRC / f"{tag}{s}" for s in LAYERS if (SRC / f"{tag}{s}").exists()]


def day_list(tag: str) -> list[str]:
    sets = [{p.name[3:13] for p in d.glob("dt=*.parquet")} for d in dirs(tag)]
    out = sets[0]
    for s in sets[1:]:
        if s:
            out &= s
    return sorted(out)


def feat_cols(tag: str, days) -> tuple[dict[str, list[str]], dict]:
    """({層のパス -> その層の列}, 列 -> 分類)。同じ名前は先に見つけた層を使う。"""
    fam = {}
    for nm in ("l4feat", "l4life", "l4post"):
        f = DATA / f"{nm}_cols_{tag}.csv"
        if f.exists():
            d = pl.read_csv(f)
            fam |= dict(zip(d["col"].to_list(), d["family"].to_list()))
    got, seen = {}, set()
    for d in dirs(tag):
        n = pl.scan_parquet(d / f"dt={days[0]}.parquet").collect_schema().names()
        c = [x for x in n if not x.startswith("label_")
             and x not in ("dt", "sec") and x not in seen]
        seen |= set(c)
        got[str(d)] = c
    return got, fam


def positions():
    """標本 A / B の、1 日の中での行位置と、帰無対照用のずらした位置。"""
    pos, plc = {}, {}
    for k, st in SETS.items():
        p = np.arange(0, NS, st)
        pos[k] = p
        plc[k] = (p + PLACEBO_SHIFT) % NS
    return pos, plc


def read_block(tag: str, days, want: dict[str, list[str]], pos, plc=None):
    """指定の列を、標本 A / B の行だけ全日ぶん読む。

    全行を持つとメモリが足りない(700 列 × 855 万行)。**読みながら間引く**。
    want は {層のパス: 列名} 。
    """
    allc = [c for v in want.values() for c in v]
    out = {k: {c: [] for c in allc} for k in SETS}
    outp = {k: {c: [] for c in allc} for k in SETS} if plc is not None else None
    for d in days:
        for dr, cs in want.items():
            if not cs:
                continue
            t = pl.read_parquet(Path(dr) / f"dt={d}.parquet", columns=cs)
            for c in cs:
                v = t[c].to_numpy()
                for k in SETS:
                    out[k][c].append(v[pos[k]])
                    if plc is not None:
                        outp[k][c].append(v[plc[k]])
    out = {k: {c: np.concatenate(v) for c, v in dd.items()} for k, dd in out.items()}
    if plc is not None:
        outp = {k: {c: np.concatenate(v) for c, v in dd.items()}
                for k, dd in outp.items()}
    return out, outp


def pick(got: dict[str, list[str]], sub: list[str]) -> dict[str, list[str]]:
    """列名の集合を、層ごとの読み出し指示に直す。"""
    s = set(sub)
    return {d: [c for c in v if c in s] for d, v in got.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    ap.add_argument("--stage", default="pool")
    ap.add_argument("--top", type=int, default=36)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    days = day_list(tag)
    nday = len(days)
    got, fam = feat_cols(tag, days)
    cols = [c for v in got.values() for c in v]
    print(f"{tag}: {nday} 日 / 特徴量 {len(cols)} 本 "
          + " + ".join(f"{Path(d).name} {len(v)}" for d, v in got.items())
          + f" × {len(FLAB)} ホライズン = {len(cols)*len(FLAB)} セル", flush=True)
    pos, plc = positions()
    half = nday // 2
    dayv = {k: np.repeat(np.arange(nday), pos[k].size) for k in SETS}

    # ---- 目的変数 --------------------------------------------------------
    lb = [f"label_micro_{h}" for h in FLAB] + [f"label_mid_{h}" for h in FLAB] \
        + [f"mret_{h}" for h in FLAB]
    Yall, _ = read_block(tag, days, {str(SRC / tag): lb}, pos)
    print(f"標本 A {Yall['A'][lb[0]].size:,} 行 / 標本 B {Yall['B'][lb[0]].size:,} 行",
          flush=True)

    if a.stage == "pool":
        rows = []
        for k0 in range(0, len(cols), BATCH):
            sub = cols[k0:k0 + BATCH]
            X, XP = read_block(tag, days, pick(got, sub), pos, plc)
            for c in sub:
                for h in FLAB:
                    k = HSET[h]
                    x, xp = X[k][c], XP[k][c]
                    y = Yall[k][f"label_micro_{h}"]
                    r, n = sp(x, y)
                    rm, _ = sp(x, Yall[k][f"label_mid_{h}"])
                    rb, _ = sp(x, Yall[k][f"mret_{h}"])
                    rp, _ = sp(xp, y)
                    d1 = dayv[k] < half
                    r1, _ = sp(x[d1], y[d1])
                    r2, _ = sp(x[~d1], y[~d1])
                    rows.append(dict(col=c, family=fam.get(c, "?"), h=h, n=n,
                                     r_fwd=r, r_mid=rm, r_bwd=rb, r_plc=rp,
                                     r_1st=r1, r_2nd=r2, t=tval(r, n)))
            del X, XP
            print(f"  {min(k0+BATCH, len(cols))}/{len(cols)}", flush=True)
        out = pl.DataFrame(rows)
        out.write_csv(DATA / f"l4feat_pred_{tag}.csv")
        top = (out.filter(pl.col("h") == "10s")
               .with_columns(ab=pl.col("r_fwd").abs())
               .sort("ab", descending=True).head(12))
        print(top.select("col", "family", "r_fwd", "r_bwd", "r_plc"))

    elif a.stage == "daily":
        pred = pl.read_csv(DATA / f"l4feat_pred_{tag}.csv")
        sel = []
        for h in ("1s", "10s", "60s"):
            sel += (pred.filter(pl.col("h") == h)
                    .with_columns(ab=pl.col("r_fwd").abs())
                    .sort("ab", descending=True).head(a.top)["col"].to_list())
        sel = list(dict.fromkeys(sel))
        X, _ = read_block(tag, days, pick(got, sel), pos)
        rows = []
        for c in sel:
            for h in ("1s", "10s", "60s"):
                k = HSET[h]
                x, y = X[k][c], Yall[k][f"label_micro_{h}"]
                rs = []
                for d in range(nday):
                    m = dayv[k] == d
                    r, _ = sp(x[m], y[m])
                    if np.isfinite(r):
                        rs.append(r)
                rs = np.array(rs)
                if not rs.size:
                    continue
                rows.append(dict(col=c, family=fam.get(c, "?"), h=h,
                                 nday=rs.size, mean=float(rs.mean()),
                                 med=float(np.median(rs)),
                                 pos=int((rs > 0).sum()), worst=float(rs.min()),
                                 best=float(rs.max())))
        pl.DataFrame(rows).write_csv(DATA / f"l4feat_daily_{tag}.csv")
        print("日ごとの符号一致を書き出した", flush=True)

    elif a.stage == "corr":
        # 972 × 14 万行を倍精度で持つと 1.1GB になる。相関行列に要る精度は
        # そこまで無いので、標本 B をさらに 5 分の 1 に間引く(2.9 万行)
        thin = 5
        n = pos["B"][::thin].size * nday
        M = np.empty((len(cols), n), np.float64)
        for k0 in range(0, len(cols), BATCH):
            sub = cols[k0:k0 + BATCH]
            X, _ = read_block(tag, days, pick(got, sub), pos)
            for j, c in enumerate(sub):
                M[k0 + j] = X["B"][c].reshape(nday, -1)[:, ::thin].ravel()
            del X
            print(f"  {min(k0+BATCH, len(cols))}/{len(cols)}", flush=True)
        M -= np.nanmean(M, axis=1, keepdims=True)
        M = np.nan_to_num(M)
        M /= np.maximum(np.sqrt((M ** 2).sum(1, keepdims=True)), 1e-12)
        C = (M @ M.T).astype(np.float32)
        np.save(DATA / f"l4feat_corr_{tag}.npy", C)
        (DATA / f"l4feat_corr_{tag}.cols").write_text("\n".join(cols),
                                                      encoding="utf-8")
        iu = np.triu_indices(len(cols), 1)
        v = np.abs(C[iu])
        print(f"|r|>0.99 のペア {int((v>0.99).sum()):,} 組 / {v.size:,} 組",
              flush=True)


if __name__ == "__main__":
    main()
