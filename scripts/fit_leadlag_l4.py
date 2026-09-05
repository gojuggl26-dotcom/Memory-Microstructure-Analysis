"""銘柄をまたいだ先行・遅行(分類 55)を、100ms 格子のリターンで測る。

    uv run python scripts/fit_leadlag_l4.py --a xyz:INTC --b xyz:MU

問い
----
`xyz:MU` の**過去**の動きは `xyz:INTC` の**将来**の動きを説明するか。
逆向きも同じ手続きで測り、**どちらが先か**を非対称性として出す。

    先行度(B → A) = corr( r_B[T−h, T) , r_A[T, T+h) )
    先行度(A → B) = corr( r_A[T−h, T) , r_B[T, T+h) )

時間契約
--------
説明側は `[T−h, T)`、目的側は `[T, T+h)`。**同じ区間を両側で使わない**。
同時性の大きさも別欄で出す(corr( r_B[T,T+h) , r_A[T,T+h) ))。
標本はホライズンと同じ間隔で間引き、重なりを作らない。
帰無対照は片方を **1 時間ずらす**。

日ごとに計算して、日をまたいだ平均と符号一致(何日中何日が同符号か)を出す。
プールしないのは、1 日の大きな動きが全体を支配するのを避けるため。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import GRID_NS, PX_UNIT, clean_bbo, grid_mid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000
NG = DAY_NS // GRID_NS
FLAB = ["100ms", "500ms", "1s", "5s", "10s", "30s", "60s"]
STEP = [1, 5, 10, 50, 100, 300, 600]      # 100ms 単位
SHIFT = 36_000                            # 帰無対照のずらし(= 1 時間)


def micro_grid(coin: str, dt: str):
    """その日の 100ms 格子のマイクロプライス(対数)。無ければ None。"""
    tag = coin.replace(":", "_")
    fp = DATA / f"bbo_l1_{tag}" / f"dt={dt}.parquet"
    if not fp.exists():
        return None
    b = pl.read_parquet(fp)
    if not b.height:
        return None
    t0 = (int(b["ts"].cast(pl.Int64).min()) // DAY_NS) * DAY_NS
    b2, _ = clean_bbo(b)
    mid, bbi, bai, good, qb, qa = grid_mid(b2, t0, NG)
    s = qb + qa
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(good & (s > 0),
                     (bbi * PX_UNIT * qa + bai * PX_UNIT * qb)
                     / np.where(s > 0, s, 1.0), np.nan)
    return np.log(np.where(np.isfinite(m) & (m > 0), m, np.nan))


def pear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="xyz:INTC")
    ap.add_argument("--b", default="xyz:MU")
    x = ap.parse_args()
    ta, tb = x.a.replace(":", "_"), x.b.replace(":", "_")
    da = {p.name[3:13] for p in (DATA / f"bbo_l1_{ta}").glob("dt=*.parquet")}
    db = {p.name[3:13] for p in (DATA / f"bbo_l1_{tb}").glob("dt=*.parquet")}
    days = sorted(da & db)
    print(f"{x.a} と {x.b} の共通日 {len(days)} 日", flush=True)
    acc = {h: {"ba": [], "ab": [], "sync": [], "plc": []} for h in FLAB}
    for dt in days:
        la, lb = micro_grid(x.a, dt), micro_grid(x.b, dt)
        if la is None or lb is None:
            continue
        for h, k in zip(FLAB, STEP):
            i = np.arange(k, NG - k, k)          # 重ならない標本
            # 過去の区間 [T−h, T) と 未来の区間 [T, T+h)
            pa = la[i] - la[i - k]
            pb = lb[i] - lb[i - k]
            fa = la[i + k] - la[i]
            fb = lb[i + k] - lb[i]
            acc[h]["ba"].append(pear(pb, fa))    # B の過去 → A の未来
            acc[h]["ab"].append(pear(pa, fb))    # A の過去 → B の未来
            acc[h]["sync"].append(pear(fa, fb))  # 同時
            j = (i + SHIFT) % (NG - 2 * k) + k
            acc[h]["plc"].append(pear(lb[j] - lb[j - k], fa))
    rows = []
    for h in FLAB:
        d = {k: np.array([v for v in acc[h][k] if np.isfinite(v)])
             for k in acc[h]}
        if not d["ba"].size:
            continue
        rows.append(dict(
            h=h, nday=d["ba"].size,
            ba=float(d["ba"].mean()), ba_pos=int((d["ba"] > 0).sum()),
            ab=float(d["ab"].mean()), ab_pos=int((d["ab"] > 0).sum()),
            sync=float(d["sync"].mean()), plc=float(d["plc"].mean()),
            plc_abs_max=float(np.abs(d["plc"]).max())))
    out = pl.DataFrame(rows)
    out.write_csv(DATA / f"leadlag_{ta}_{tb}.csv")
    print(out)


if __name__ == "__main__":
    main()
