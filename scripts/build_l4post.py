"""既に書いたパネルから、派生の 4 分類を足す。

    uv run python scripts/build_l4post.py --coin xyz:INTC

  48 流動性 × ボラティリティ … 厚み・幅・偏り・回転をボラで割る/掛ける
  59 レジーム                … 板の状態を**前日の分位**で 3 段に切る
  68 移動分布                … 主要量の移動平均/分散/歪度/最小最大/分位順位
  69 ラグ                    … 主要量の 1・2・5・10 秒前の値と、その差

パネルを作り直さずに済むよう別ファイルにした。入力は
`E:/Memory-l4feat/<coin>/dt=*.parquet` の一部の列だけで、1 日 1 分ほど。

時間契約
--------
  * 移動窓は全て**後ろ向き**(t を含み、t より先は入れない)
  * ★分位・レジームの境目は**前日の分布**から作る。当日全体から作ると
    格子点 T の値を T より後ろの分布で順位づけることになる(CLAUDE.md の規則)
  * 初日は基準が無いので NaN
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = Path("E:/Memory-l4feat")
NS = 86_400

# 48 の材料。(名前, 分子, 演算, 分母)
VOL = "rv_10s"
LIQ = ["cdtot10", "spread_bp", "obi5", "churn_10s", "cxl_rate_10s",
       "new_rate_10s", "gap1_b", "wal_hhi", "slip50k_rt", "prs_imb"]
# 59 / 68 / 69 の対象
CORE = ["obi5", "delta_bp", "spread_bp", "cdtot10", "ofi_10s", "rv_10s",
        "churn_10s", "cxl_imb_10s", "wal_hhi", "n_pers10s_share", "live_imb",
        "n_flr_n01_10s", "slope_asym", "gap1_b", "vol_imb_10s"]
ROLL = [10, 60, 300]                  # 移動窓(秒)
LAGS = [1, 2, 5, 10]                  # ラグ(秒)
FAM: dict[str, str] = {}


def reg(n, f):
    FAM[n] = f
    return n


def rs(x, w):
    """後ろ向き w 点の移動和(t を含む)。"""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    i = np.arange(x.size) + 1
    return c[i] - c[np.maximum(i - w, 0)]


def rn(x, w):
    return np.maximum(rs(np.isfinite(x).astype(float), w), 1)


def safe(a, b, fill=np.nan):
    b = np.asarray(b, np.float64)
    ok = np.abs(b) > 1e-12
    return np.where(ok, np.asarray(a, np.float64) / np.where(ok, b, 1.0), fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    out = SRC / f"{tag}_post"
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[3:13] for p in (SRC / tag).glob("dt=*.parquet"))
    have_p = set(pl.scan_parquet(SRC / tag / f"dt={days[0]}.parquet")
                 .collect_schema().names())
    lf = SRC / f"{tag}_life"
    have_l = (set(pl.scan_parquet(lf / f"dt={days[0]}.parquet")
                  .collect_schema().names()) if lf.exists() else set())
    need = sorted(set(CORE + LIQ + [VOL]))
    cp = [c for c in need if c in have_p]
    cl = [c for c in need if c in have_l and c not in have_p]
    miss = [c for c in need if c not in cp + cl]
    if miss:
        print(f"[注意] パネルに無い列を飛ばす: {miss}", flush=True)
    prev_q: dict[str, np.ndarray] = {}
    for dt in days:
        t = time.time()
        if cl and not (lf / f"dt={dt}.parquet").exists():
            print(f"  {dt} 寿命層がまだ無いので飛ばす", flush=True)
            continue
        d = pl.read_parquet(SRC / tag / f"dt={dt}.parquet", columns=cp)
        if cl:
            d = d.hstack(pl.read_parquet(lf / f"dt={dt}.parquet", columns=cl))
        X = {c: d[c].to_numpy().astype(np.float64) for c in d.columns}
        F: dict[str, np.ndarray] = {}
        v = X.get(VOL)
        # --- 48 流動性 × ボラティリティ ---------------------------------
        if v is not None:
            for c in LIQ:
                if c not in X:
                    continue
                F[reg(f"{c}__xvol", "48 流動性 × ボラティリティ")] = X[c] * v
                F[reg(f"{c}__ivol", "48 流動性 × ボラティリティ")] = safe(X[c], v)
        # --- 68 移動分布 / 69 ラグ / 59 レジーム -------------------------
        for c in CORE:
            if c not in X:
                continue
            x = X[c]
            for w in ROLL:
                n = rn(x, w)
                m = rs(x, w) / n
                s2 = np.maximum(rs(x * x, w) / n - m ** 2, 0.0)
                F[reg(f"{c}__m{w}", "68 移動分布")] = m
                F[reg(f"{c}__sd{w}", "68 移動分布")] = np.sqrt(s2)
                if w == 300:
                    m3 = rs(x ** 3, w) / n
                    F[reg(f"{c}__sk{w}", "68 移動分布")] = safe(
                        m3 - 3 * m * s2 - m ** 3, np.maximum(s2, 1e-12) ** 1.5)
                    # 最小・最大は移動窓の順序統計。累積最大では出せないので
                    # 粗い格子(60 秒ごと)の最大で近似せず、素直に stride 計算
                    F[reg(f"{c}__rng{w}", "68 移動分布")] = roll_range(x, w)
            for L in LAGS:
                lag = np.concatenate([np.full(L, np.nan), x[:-L]])
                F[reg(f"{c}__lag{L}", "69 ラグ")] = lag
                F[reg(f"{c}__dlag{L}", "69 ラグ")] = x - lag
            # --- 59 レジーム: 前日の三分位で 0/1/2 に切る ---
            q = prev_q.get(c)
            fin = np.isfinite(x)
            F[reg(f"{c}__reg", "59 レジーム")] = (
                np.where(fin, np.searchsorted(q, x), np.nan)
                if q is not None else np.full(x.size, np.nan))
            qf = prev_q.get(c + "_full")
            F[reg(f"{c}__pct", "68 移動分布")] = (
                np.where(fin, np.searchsorted(qf, x) / 100.0, np.nan)
                if qf is not None else np.full(x.size, np.nan))
        # 翌日のための基準
        for c in CORE:
            if c in X:
                xx = X[c][np.isfinite(X[c])]
                if xx.size > 1000:
                    prev_q[c] = np.quantile(xx, [1 / 3, 2 / 3])
                    prev_q[c + "_full"] = np.quantile(xx, np.linspace(0, 1, 101))
        df = pl.DataFrame({"dt": np.full(NS, dt),
                           "sec": np.arange(NS, dtype=np.int32)}
                          | {k: np.asarray(x, np.float32) for k, x in F.items()})
        df.write_parquet(out / f"dt={dt}.parquet", compression="zstd",
                         compression_level=3)
        print(f"  {dt}  {df.width} 列  {time.time()-t:.0f}s", flush=True)
    pl.DataFrame({"col": list(FAM), "family": list(FAM.values())}) \
      .write_csv(DATA / f"l4post_cols_{tag}.csv")


def roll_range(x, w):
    """後ろ向き w 点の最大 − 最小。ブロックに割って端だけ丁寧に見る。"""
    n = x.size
    pad = (-n) % w
    y = np.concatenate([x, np.full(pad, np.nan)])
    B = y.reshape(-1, w)
    with np.errstate(invalid="ignore"):
        # 前方累積(ブロック内)と後方累積(ひとつ前のブロック)の組み合わせ
        fmax = np.maximum.accumulate(B, axis=1)
        fmin = np.minimum.accumulate(B, axis=1)
        bmax = np.maximum.accumulate(B[:, ::-1], axis=1)[:, ::-1]
        bmin = np.minimum.accumulate(B[:, ::-1], axis=1)[:, ::-1]
    # 位置 i = b*w + j の窓は「1 つ前のブロックの j+1 以降」と
    # 「このブロックの j まで」の 2 つに割れる。j = w−1 のとき前半は空。
    pmax = np.concatenate([np.full((1, w), np.nan), bmax[:-1]])
    pmin = np.concatenate([np.full((1, w), np.nan), bmin[:-1]])
    nanc = np.full((B.shape[0], 1), np.nan)
    mx = np.fmax(fmax, np.concatenate([pmax[:, 1:], nanc], axis=1))
    mn = np.fmin(fmin, np.concatenate([pmin[:, 1:], nanc], axis=1))
    return (mx - mn).ravel()[:n]


if __name__ == "__main__":
    main()
