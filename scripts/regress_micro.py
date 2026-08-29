"""r_{t+i} = α + β m_t + ε の推定(m_t = mid_t − microprice_t)。

仕様:
  x  m_t   = (mid − microprice)/mid × 1e4  [bp]  ※符号は指示どおり mid − micro
  y  r_{t+i} = (log mid_{t+i} − log mid_t) × 1e4 [bp]
  地平 i = 1..20 イベント / 1..20 秒
  spec A: r ~ 1 + m
  spec B: r ~ 1 + m + r_{t-1}(直前 1 イベントのリターン。モメンタムの統制)

標準誤差は 2 通り:
  - Newey-West(重複リターンの MA 誤差用。イベント地平は L=2i、
    時間地平は L=2i×その日の毎秒ティック数、上限 400)
  - 日次クラスター(99 クラスター。日内の任意の系列相関・不均一分散を許容)
どちらも同じ結論なら頑健。t が数百になるのは n が 2,347 万だからであって
経済的な大きさとは無関係 — 判断は R² と bp 換算で行う。

2 パス構成: パス 1 で X'X, X'y を貯めて β を確定、パス 2 で残差から
HAC の meat 行列とクラスター和を貯める。
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/microprice")
OUT = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
EV_H = list(range(1, 21))               # イベント地平
CL_H = [i * 1_000_000_000 for i in range(1, 21)]  # 秒地平(ns)
SPREAD_BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9)]
NW_CAP = 400
KEY_EV = [1, 5, 20]
KEY_CL = [1, 5, 20]


def band_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}bp" if hi < 1e8 else f"{lo:g}bp+"


def load_day(path: Path) -> dict:
    df = pl.read_parquet(
        path, columns=["ts", "mid", "microprice", "spread_bp", "best_bid", "best_ask"]
    ).sort("ts")
    # ★クロス状態(best_ask < best_bid)を除く。全体の 0.045% しかないが mid が意味を失い
    #   |m| が最大 7,349bp に飛ぶため、除かないと回帰のレバレッジの半分をこの行が占める
    df = df.filter(pl.col("best_ask") > pl.col("best_bid"))
    ts = df["ts"].to_numpy()
    mid = df["mid"].to_numpy()
    mp = df["microprice"].to_numpy()
    logm = np.log(mid)
    m = (mid - mp) / mid * 1e4                      # x: mid − micro [bp]
    r_prev = np.empty_like(logm)
    r_prev[0] = 0.0
    r_prev[1:] = (logm[1:] - logm[:-1]) * 1e4       # 直前 1 イベントのリターン
    return {"ts": ts, "logm": logm, "m": m, "r_prev": r_prev,
            "spread": df["spread_bp"].to_numpy(), "n": len(ts)}


def future_ret(d: dict, kind: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(有効な t の添字, r_{t+h}) を返す。"""
    logm, ts, n = d["logm"], d["ts"], d["n"]
    if kind == "ev":
        if n <= h:
            return np.empty(0, dtype=np.int64), np.empty(0)
        idx = np.arange(n - h)
        return idx, (logm[h:] - logm[:-h]) * 1e4
    # 時間地平: t+h ns 時点で観測されている直近の mid(その日の中で完結する t のみ)
    tgt = ts + h
    ok = tgt <= ts[-1]
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return idx, np.empty(0)
    j = np.searchsorted(ts, tgt[idx], side="right") - 1
    return idx, (logm[j] - logm[idx]) * 1e4


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    if len(sys.argv) > 1:      # 検証用に日数を絞る
        days = days[: int(sys.argv[1])]
    specs = [("ev", h) for h in EV_H] + [("cl", h) for h in CL_H]

    # ---- パス 1: X'X, X'y の蓄積 ----
    A = {s: np.zeros((3, 3)) for s in specs}   # [1, m, r_prev]
    B = {s: np.zeros(3) for s in specs}
    SY = {s: np.zeros(2) for s in specs}       # Σy, Σy²
    NN = {s: 0 for s in specs}
    band_acc: dict = {}
    daily: dict = {}
    tick_rate = {}

    for di, dt in enumerate(days, 1):
        d = load_day(SRC / f"dt={dt}" / "part-000.parquet")
        span = max(1e-9, (d["ts"][-1] - d["ts"][0]) / 1e9)
        tick_rate[dt] = d["n"] / span
        for s in specs:
            kind, h = s
            idx, y = future_ret(d, kind, h)
            if idx.size < 100:
                continue
            x = d["m"][idx]
            rp = d["r_prev"][idx]
            X = np.column_stack([np.ones_like(x), x, rp])
            A[s] += X.T @ X
            B[s] += X.T @ y
            SY[s] += (y.sum(), (y * y).sum())
            NN[s] += idx.size
            sp = d["spread"][idx]
            for lo, hi in SPREAD_BANDS:
                sel = (sp >= lo) & (sp < hi)
                if sel.sum() < 50:
                    continue
                xb, yb = x[sel], y[sel]
                k = (s, band_name(lo, hi))
                a = band_acc.setdefault(k, np.zeros(6))
                a += (sel.sum(), xb.sum(), yb.sum(), (xb * xb).sum(), (xb * yb).sum(),
                      (yb * yb).sum())
            # 日次の十分統計量(Fama-MacBeth とレバレッジ集中度の算出用)
            nn = idx.size
            sx, sy2 = x.sum(), y.sum()
            vx = (x * x).sum() / nn - (sx / nn) ** 2
            vy = (y * y).sum() / nn - (sy2 / nn) ** 2
            cxy = (x * y).sum() / nn - (sx / nn) * (sy2 / nn)
            daily.setdefault(s, []).append({
                "dt": dt, "n": int(nn),
                "beta": cxy / vx if vx > 0 else None,
                "corr": cxy / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None,
                "leverage": nn * vx,          # Σ(x−x̄)² = プール推定への寄与度
                "std_x": vx ** 0.5, "std_y": vy ** 0.5,
            })
        if di % 20 == 0:
            print(f"pass1 {di}/{len(days)}", flush=True)

    est = {}
    for s in specs:
        # spec A(定数 + m)は 3x3 の左上 2x2 を使う
        Aa, Ba = A[s][:2, :2], B[s][:2]
        beta_a = np.linalg.solve(Aa, Ba)
        beta_b = np.linalg.solve(A[s], B[s])
        est[s] = {"A_a": Aa, "b_a": beta_a, "A_b": A[s], "b_b": beta_b}

    # ---- パス 2: HAC の meat とクラスター和 ----
    meat = {s: np.zeros((2, 2)) for s in specs}
    meat_b = {s: np.zeros((3, 3)) for s in specs}
    clus = {s: np.zeros((2, 2)) for s in specs}
    clus_b = {s: np.zeros((3, 3)) for s in specs}
    rss = {s: 0.0 for s in specs}
    rss_b = {s: 0.0 for s in specs}
    econ: dict = {}

    for di, dt in enumerate(days, 1):
        d = load_day(SRC / f"dt={dt}" / "part-000.parquet")
        rate = tick_rate[dt]
        for s in specs:
            kind, h = s
            idx, y = future_ret(d, kind, h)
            if idx.size < 100:
                continue
            x = d["m"][idx]
            rp = d["r_prev"][idx]
            Xa = np.column_stack([np.ones_like(x), x])
            Xb = np.column_stack([np.ones_like(x), x, rp])
            ua = y - Xa @ est[s]["b_a"]
            ub = y - Xb @ est[s]["b_b"]
            rss[s] += float(ua @ ua)
            rss_b[s] += float(ub @ ub)
            # 経済的有意性: 予測幅 |β·m| がハーフスプレッドを超える割合
            if (kind == "ev" and h in KEY_EV) or (kind == "cl" and h // 10**9 in KEY_CL):
                pred = np.abs(est[s]["b_a"][1] * x)
                half = d["spread"][idx] / 2.0
                e = econ.setdefault(s, np.zeros(4))
                e += (idx.size, np.abs(x).sum(), half.sum(), (pred > half).sum())
            ga = Xa * ua[:, None]
            gb = Xb * ub[:, None]
            sa, sb = ga.sum(0), gb.sum(0)
            clus[s] += np.outer(sa, sa)
            clus_b[s] += np.outer(sb, sb)
            # Newey-West: 重複の次数に合わせてラグを取る
            if kind == "ev":
                L = min(NW_CAP, 2 * h)
            else:
                L = int(min(NW_CAP, max(4, 2 * (h / 1e9) * rate)))
            meat[s] += ga.T @ ga
            meat_b[s] += gb.T @ gb
            for j in range(1, L + 1):
                if j >= ga.shape[0]:
                    break
                w = 1.0 - j / (L + 1.0)
                G = ga[j:].T @ ga[:-j]
                meat[s] += w * (G + G.T)
                Gb = gb[j:].T @ gb[:-j]
                meat_b[s] += w * (Gb + Gb.T)
        if di % 20 == 0:
            print(f"pass2 {di}/{len(days)}", flush=True)

    def p_two(t: float) -> float:
        return math.erfc(abs(t) / math.sqrt(2.0))

    res = {"n_days": len(days), "specs": {}}
    for s in specs:
        kind, h = s
        n = NN[s]
        Aa = est[s]["A_a"]
        ba = est[s]["b_a"]
        Ainv = np.linalg.inv(Aa)
        V_nw = Ainv @ meat[s] @ Ainv
        V_cl = Ainv @ clus[s] @ Ainv
        se_nw, se_cl = math.sqrt(V_nw[1, 1]), math.sqrt(V_cl[1, 1])
        sy, syy = SY[s]
        tss = syy - sy * sy / n
        r2 = 1.0 - rss[s] / tss
        mx = Aa[0, 1] / n
        vx = Aa[1, 1] / n - mx * mx
        Ainv_b = np.linalg.inv(est[s]["A_b"])
        Vb_cl = Ainv_b @ clus_b[s] @ Ainv_b
        Vb_nw = Ainv_b @ meat_b[s] @ Ainv_b
        name = f"{'event' if kind == 'ev' else 'clock'}_{h if kind == 'ev' else h // 10**9}"
        res["specs"][name] = {
            "kind": "event" if kind == "ev" else "clock",
            "horizon": h if kind == "ev" else h // 10**9,
            "n": int(n), "alpha_bp": ba[0], "beta": ba[1],
            "se_newey_west": se_nw, "t_newey_west": ba[1] / se_nw,
            "p_newey_west": p_two(ba[1] / se_nw),
            "se_cluster_day": se_cl, "t_cluster_day": ba[1] / se_cl,
            "p_cluster_day": p_two(ba[1] / se_cl),
            "r2": r2, "std_x_bp": math.sqrt(vx), "std_y_bp": math.sqrt(tss / n),
            "spec_b": {
                "beta_m": est[s]["b_b"][1], "gamma_rprev": est[s]["b_b"][2],
                "se_m_cluster": math.sqrt(Vb_cl[1, 1]),
                "t_m_cluster": est[s]["b_b"][1] / math.sqrt(Vb_cl[1, 1]),
                "se_m_nw": math.sqrt(Vb_nw[1, 1]),
                "r2": 1.0 - rss_b[s] / tss,
            },
        }
    res["by_spread_band"] = {}
    for (s, band), a in band_acc.items():
        kind, h = s
        n, sx, sy, sxx, sxy, syy = a
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        cxy = sxy / n - (sx / n) * (sy / n)
        name = f"{'event' if kind == 'ev' else 'clock'}_{h if kind == 'ev' else h // 10**9}|{band}"
        res["by_spread_band"][name] = {
            "n": int(n), "beta": cxy / vx if vx > 0 else None,
            "corr": cxy / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None,
        }

    # ---- Fama-MacBeth(日ごとに推定して平均)とレバレッジ集中度 ----
    res["fama_macbeth"] = {}
    for s, rows in daily.items():
        kind, h = s
        name = f"{'event' if kind == 'ev' else 'clock'}_{h if kind == 'ev' else h // 10**9}"
        b = np.array([r["beta"] for r in rows if r["beta"] is not None], dtype=float)
        lev = np.array([r["leverage"] for r in rows], dtype=float)
        if b.size < 5:
            continue
        se = b.std(ddof=1) / math.sqrt(b.size)
        share = np.sort(lev)[::-1]
        share = share / share.sum()
        res["fama_macbeth"][name] = {
            "n_days": int(b.size), "beta_mean": float(b.mean()), "se": float(se),
            "t": float(b.mean() / se), "p": p_two(b.mean() / se),
            "beta_median": float(np.median(b)),
            "beta_q25": float(np.quantile(b, 0.25)), "beta_q75": float(np.quantile(b, 0.75)),
            "leverage_share_top1": float(share[0]),
            "leverage_share_top5": float(share[:5].sum()),
            "effective_days": float(1.0 / (share ** 2).sum()),   # Herfindahl の逆数
        }
    res["economics"] = {
        f"{'event' if k[0] == 'ev' else 'clock'}_{k[1] if k[0] == 'ev' else k[1] // 10**9}": {
            "n": int(v[0]), "mean_abs_m_bp": v[1] / v[0], "mean_half_spread_bp": v[2] / v[0],
            "mean_pred_move_bp": abs(res["specs"][
                f"{'event' if k[0] == 'ev' else 'clock'}_{k[1] if k[0] == 'ev' else k[1] // 10**9}"
            ]["beta"]) * v[1] / v[0],
            "frac_pred_gt_half_spread": v[3] / v[0],
        } for k, v in econ.items()
    }
    res["daily_beta"] = {
        f"{'event' if k[0] == 'ev' else 'clock'}_{k[1] if k[0] == 'ev' else k[1] // 10**9}": v
        for k, v in daily.items()
    }
    (OUT / "regression.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                    default=float), encoding="utf-8")
    print("saved", OUT / "regression.json")


if __name__ == "__main__":
    main()
