"""価格リターンの自己相関を 1ms / 10ms / 100ms / 1s / 10s / 60s で算出する。

    r_{t+1} = log Price_{t+1} − log Price_t     (等間隔グリッド上)

価格は 3 種類:
    mid        = (best_bid + best_ask)/2      … 既定
    micro      = マイクロプライス
    trade      = 直近約定価格(node_fills)

計算量の工夫: 1ms グリッドは 1 日 8,640 万点(99 日で 86 億点)になり素直には持てない。
しかし価格は階段関数なので、**リターンが 0 でないグリッド点は価格が動いた回数以下**
(mid で 2,347 万)。そこで
  - 各変化 i の対数価格増分 d_i を、その変化が最初に反映されるグリッド番号
    j_i = ceil((τ_i − t0)/Δ) に足し込む(同一グリッドの複数変化は自然に合算 = telescoping)
  - 非ゼロ位置 P と値 V だけを持ち、自己共分散 Σ_j r_j r_{j−k} は
    「P と P−k の共通部分」を searchsorted で取って計算する
これは間引きでも近似でもなく、ゼロを明示的に持たないだけの厳密計算である。

推定は監査(regression_report.md §9)の作法に従い日次で行い、中央値と四分位で読む。
n = 86,400,000/日 のような標本数で有意性を語っても意味がないため、
Bartlett の 1/√n も参考値として出すにとどめる。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
FILLS = Path("data/fills_v99")
NS_DAY = 86_400_000_000_000
STEPS = {"1ms": 1_000_000, "10ms": 10_000_000, "100ms": 100_000_000,
         "1s": 1_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000}
KMAX = 20
SERIES = ("mid", "micro", "trade")


def acf_sparse(tau: np.ndarray, lp: np.ndarray, t0: int, step: int, kmax: int) -> dict | None:
    """階段関数の対数価格から、歩幅 step のグリッド上の ACF 用十分統計量を返す。"""
    if tau.size < 3:
        return None
    d = np.diff(lp)                                   # 価格変化の対数増分
    j = -((t0 - tau[1:]) // step)                     # ceil((tau - t0)/step)
    n_grid = NS_DAY // step
    j_first = -((t0 - tau[0]) // step)                # 最初に価格が存在するグリッド
    keep = (j > j_first) & (j <= n_grid) & (d != 0.0)
    if keep.sum() < 3:
        return None
    j, d = j[keep], d[keep]
    pos, inv = np.unique(j, return_inverse=True)
    val = np.bincount(inv, weights=d)
    n = int(n_grid - j_first)                         # 有効なリターンの本数(ゼロを含む)
    out = {"n": n, "s1": float(val.sum()), "s2": float((val * val).sum()),
           "nz": int(pos.size), "c": []}
    for k in range(1, kmax + 1):
        # pos に含まれ、かつ pos−k も pos に含まれる位置だけが積に寄与する
        want = pos - k
        idx = np.searchsorted(pos, want)
        ok = (idx < pos.size)
        ok[ok] = pos[idx[ok]] == want[ok]
        if not ok.any():
            out["c"].append(0.0)
            continue
        out["c"].append(float((val[ok] * val[idx[ok]]).sum()))
    return out


def rho_from(acc: dict, kmax: int) -> list[float]:
    n, s1, s2 = acc["n"], acc["s1"], acc["s2"]
    if n < 10 or s2 <= 0:
        return [float("nan")] * kmax
    mu = s1 / n
    var = s2 / n - mu * mu                            # ゼロも含めた全 n 点の分散
    if var <= 0:
        return [float("nan")] * kmax
    out = []
    for k in range(1, kmax + 1):
        cov = acc["c"][k - 1] / (n - k) - mu * mu     # E[r_j r_{j−k}] − μ²
        out.append(cov / var)
    return out


def load_day(dt: str) -> dict:
    t = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                         columns=["ts", "mid", "microprice", "is_crossed"])
         .filter(~pl.col("is_crossed")).sort("ts"))
    res = {"mid": (t["ts"].to_numpy(), np.log(t["mid"].to_numpy())),
           "micro": (t["ts"].to_numpy(), np.log(t["microprice"].to_numpy()))}
    p = FILLS / f"dt={dt}"
    files = sorted(p.rglob("*.parquet")) if p.exists() else []
    if files:
        f = pl.concat([pl.read_parquet(x, columns=["ts", "px", "crossed", "tid"])
                       for x in files], how="diagonal_relaxed")
        f = (f.filter(pl.col("crossed")).unique(subset=["tid"], keep="first")
               .with_columns(pl.col("ts").cast(pl.Int64).alias("ts_ns")).sort("ts_ns"))
        res["trade"] = (f["ts_ns"].to_numpy(), np.log(f["px"].to_numpy()))
    return res


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D / "microprice").iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        days = days[: int(sys.argv[1])]
    pooled = {(s, g): {"n": 0, "s1": 0.0, "s2": 0.0, "nz": 0, "c": [0.0] * KMAX}
              for s in SERIES for g in STEPS}
    daily = {(s, g): [] for s in SERIES for g in STEPS}
    daily_rows: list[dict] = []

    for i, dt in enumerate(days, 1):
        t0 = int(datetime.strptime(dt, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
        src = load_day(dt)
        for s in SERIES:
            if s not in src:
                continue
            tau, lp = src[s]
            for g, step in STEPS.items():
                a = acf_sparse(tau, lp, t0, step, KMAX)
                if a is None:
                    continue
                p = pooled[(s, g)]
                p["n"] += a["n"]
                p["s1"] += a["s1"]
                p["s2"] += a["s2"]
                p["nz"] += a["nz"]
                for k in range(KMAX):
                    p["c"][k] += a["c"][k]
                rr = rho_from(a, KMAX)
                daily[(s, g)].append(rr)
                daily_rows.append({"dt": dt, "series": s, "step": g,
                                   "rho1": rr[0], "rho2": rr[1], "rho3": rr[2],
                                   "zero_share": 1.0 - a["nz"] / a["n"]})
        if i % 20 == 0:
            print(f"{i}/{len(days)}", flush=True)

    res = {}
    for (s, g), p in pooled.items():
        if p["n"] == 0:
            continue
        arr = np.array(daily[(s, g)], dtype=float)
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanquantile(arr, 0.25, axis=0)
        q75 = np.nanquantile(arr, 0.75, axis=0)
        neg = np.nansum(arr < 0, axis=0)
        res[f"{s}|{g}"] = {
            "n_grid_returns": p["n"], "nonzero_returns": p["nz"],
            "zero_share": 1.0 - p["nz"] / p["n"],
            "bartlett_se": 1.0 / np.sqrt(p["n"]),
            "rho_pooled": rho_from(p, KMAX),
            "rho_daily_median": med.tolist(),
            "rho_daily_q25": q25.tolist(), "rho_daily_q75": q75.tolist(),
            "days_negative_lag1": int(neg[0]), "days": int(arr.shape[0]),
        }
    pl.DataFrame(daily_rows).write_csv(D / "return_acf_daily.csv")
    (D / "return_acf.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    hdr = f"{'系列':>6} {'窓':>6} {'点数':>14} {'ゼロ率':>7} {'ρ1(プール)':>11} {'ρ1(日次中央)':>12} {'ρ2':>8} {'ρ3':>8} {'負の日':>7}"
    print(hdr)
    for s in SERIES:
        for g in STEPS:
            k = f"{s}|{g}"
            if k not in res:
                continue
            v = res[k]
            print(f"{s:>6} {g:>6} {v['n_grid_returns']:>14,} {v['zero_share']:>7.4f} "
                  f"{v['rho_pooled'][0]:>11.4f} {v['rho_daily_median'][0]:>12.4f} "
                  f"{v['rho_daily_median'][1]:>8.4f} {v['rho_daily_median'][2]:>8.4f} "
                  f"{v['days_negative_lag1']:>3}/{v['days']}")


if __name__ == "__main__":
    main()
