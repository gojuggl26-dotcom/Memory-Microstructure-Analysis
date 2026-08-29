"""OBI をレベル別に分解し、各レベルがどれだけ情報を持つかを OLS で測る。

【2 つの量を区別する】
  累積 OBI(L) = (Σ_{i≤L} q_b^i − Σ_{i≤L} q_a^i) / (Σ_{i≤L} q_b^i + Σ_{i≤L} q_a^i)
      … 1 段目から L 段目までをまとめた不均衡。既存の obi テーブルはこれ
  レベル別  I_L = (q_b^L − q_a^L) / (q_b^L + q_a^L)
      … **L 段目だけ**の不均衡。「そのレベル自体が持つ情報」を見るにはこちらが要る

  既存の obi テーブルからは累積しか復元できない(各段の数量が無い)ので、
  book_px を 1 秒グリッドで再生して各段の数量を取り出す。

【情報量の指標】
  単変量 OLS  y = α + β·x + ε   の **R²** を「情報量」とし、
  1 段目を 1 とした比 R²_L / R²_1 で報告する。
  併せて β の符号一貫性(何日中何日)も出す。R² は符号を持たないため。

  さらに **増分 R²**(1..L−1 の重回帰に L を足したときの R² の増加)も出す。
  これが「そのレベルが**新たに**持ち込む情報」である。

【時間契約】
  x は時刻 T の板状態(backward)。y = (log mid(T+1s) − log mid(T)) × 10⁴。
  重ならない。標準化は行わず生値で回す(R² は尺度不変)。
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
BOOKPX = Path("data/l2_v99/book_px")
PANEL = D / "panel_v2"
OUT = D / "obi_levels"
STEP = 1_000_000_000
NS_DAY = 86_400_000_000_000
NL = 10


def run_day(dt: str) -> dict | None:
    f = OUT / f"{dt}.json"
    if f.exists() and (OUT / f"{dt}.npz").exists():
        return json.loads(f.read_text(encoding="utf-8"))
    bp = BOOKPX / f"dt={dt}" / "part-000.parquet"
    pn = PANEL / f"dt={dt}" / "part-000.parquet"
    if not bp.exists() or not pn.exists():
        return None
    t0 = int(datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             .timestamp()) * 1_000_000_000
    grid = np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)
    df = (pl.read_parquet(bp, columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list(); sd = df["side"].to_list()
    px = df["px"].to_list(); sz = df["total_sz"].to_list()
    n = grid.size
    keys: dict[str, list] = {"B": [], "A": []}
    vals: dict[str, list] = {"B": [], "A": []}
    # 各段の数量(最良から NL 段)
    qb = np.zeros((n, NL)); qa = np.zeros((n, NL))
    gi = i = 0
    N = len(ts)
    while gi < n:
        while i < N and ts[i] <= grid[gi]:
            s = sd[i]
            k = -px[i] if s == "B" else px[i]      # 最良が先頭に来るよう符号を反転
            arr, va = keys[s], vals[s]
            j = bisect_left(arr, k)
            hit = j < len(arr) and arr[j] == k
            if sz[i] > 0:
                if hit:
                    va[j] = sz[i]
                else:
                    arr.insert(j, k); va.insert(j, sz[i])
            elif hit:
                del arr[j]; del va[j]
            i += 1
        for s, out in (("B", qb), ("A", qa)):
            v = vals[s][:NL]
            if v:
                out[gi, :len(v)] = v
        gi += 1

    pan = pl.read_parquet(pn, columns=["ts", "y", "stale_ns"]).sort("ts")
    m = pl.DataFrame({"ts": grid}).join(pan, on="ts", how="inner")
    if m.height < 5000:
        return None
    idx = np.searchsorted(grid, m["ts"].to_numpy())
    y = m["y"].to_numpy()
    ok = np.isfinite(y) & (m["stale_ns"].to_numpy() < 5e9)
    idx, y = idx[ok], y[ok]
    B, A = qb[idx], qa[idx]
    if len(y) < 3000:
        return None

    # レベル別と累積
    lvl = np.where(B + A > 0, (B - A) / np.maximum(B + A, 1e-9), 0.0)
    cB, cA = np.cumsum(B, 1), np.cumsum(A, 1)
    cum = np.where(cB + cA > 0, (cB - cA) / np.maximum(cB + cA, 1e-9), 0.0)

    def r2b(x):
        """単変量 OLS の R² と β。x が定数なら 0 を返す。"""
        v = x.var()
        if v <= 1e-18:
            return 0.0, 0.0
        b = np.cov(x, y, bias=True)[0, 1] / v
        r = np.corrcoef(x, y)[0, 1]
        return float(r * r), float(b)

    res = {"dt": dt, "n": int(len(y)), "level": [], "cumulative": [], "incremental": []}
    for L in range(NL):
        r, b = r2b(lvl[:, L]); res["level"].append({"L": L + 1, "r2": r, "beta": b,
                                                    "nonzero": float((B[:, L] + A[:, L] > 0).mean())})
        r, b = r2b(cum[:, L]); res["cumulative"].append({"L": L + 1, "r2": r, "beta": b})
    # 増分 R²(累積 1..L の重回帰)
    prev = 0.0
    for L in range(NL):
        X = np.column_stack([np.ones(len(y)), lvl[:, :L + 1]])
        try:
            bb = np.linalg.lstsq(X, y, rcond=None)[0]
            p = X @ bb
            r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        except np.linalg.LinAlgError:
            r2 = prev
        res["incremental"].append({"L": L + 1, "r2_cumfit": float(r2),
                                   "delta": float(r2 - prev)})
        prev = r2
    OUT.mkdir(parents=True, exist_ok=True)
    # ★MLOBI の重み推定に使う生系列も保存する。
    #   日次の R²/β だけでは Σ w_L·I_L の w を当てはめられない。
    #   数量も残しておけば、数量加重型の MLOBI も後から作れる。
    np.savez_compressed(OUT / f"{dt}.npz",
                        I=lvl.astype(np.float32),        # レベル別不均衡 (n, 10)
                        C=cum.astype(np.float32),        # 累積 OBI(L)   (n, 10)
                        qb=B.astype(np.float32), qa=A.astype(np.float32),
                        y=y.astype(np.float32))
    f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def main() -> None:
    for dt in sys.argv[1:]:
        r = run_day(dt)
        print(f"{dt}: {r['n']:,} 点 / L1 の R² {r['level'][0]['r2']:.5f}"
              if r else f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
