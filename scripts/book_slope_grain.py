"""book_slope の最適粒度を測る。

粒度は 2 つある。

  (a) 時間粒度  … どの間隔で状態を取り、どの地平のリターンを説明するか
                  100ms グリッドで状態を取り、地平 h ∈ {100ms, 1s, 5s, 10s, 60s} で評価
  (b) 深さ粒度  … 最良気配から何 bp 以内のレベルまで含めるか
                  5bp / 10bp / 25bp / 100bp / 全レベル(上限 60 段)

    book_slope_side(帯) = Σ_{i∈帯} (距離_i[bp] × 数量_i) / Σ_{i∈帯} 数量_i
    diff = bid − ask

★時間契約: 状態は T 時点(backward)、目的変数は [T, T+h) のリターン。重ならない。
成熟期 10 日で測る(粒度の比較なので全期間は要らない。採否は別途全期間で確認)。
"""

from __future__ import annotations

import json
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
BOOKPX = Path("data/l2_v99/book_px")
STEP = 100_000_000                      # 100ms グリッドで状態を取る
NS_DAY = 86_400_000_000_000
BANDS = [("5bp", 5.0), ("10bp", 10.0), ("25bp", 25.0), ("100bp", 100.0),
         ("all", float("inf"))]
HORIZONS = {"100ms": 1, "1s": 10, "5s": 50, "10s": 100, "60s": 600}   # グリッド歩数
DAYS = ["2026-07-16", "2026-07-19", "2026-07-22", "2026-07-25", "2026-07-28",
        "2026-07-31", "2026-08-02", "2026-08-04", "2026-08-06", "2026-08-09"]
MAXLV = 60


def slopes(dt: str, grid: np.ndarray) -> dict:
    df = (pl.read_parquet(BOOKPX / f"dt={dt}" / "part-000.parquet",
                          columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list(); sd = df["side"].to_list()
    px = df["px"].to_list(); sz = df["total_sz"].to_list()
    n = grid.size
    keys = {"B": [], "A": []}
    vals = {"B": [], "A": []}
    out = {f"{tag}_{b}": np.zeros(n) for tag in ("bid", "ask") for b, _ in BANDS}
    gi = i = 0
    N = len(ts)
    while gi < n:
        while i < N and ts[i] <= grid[gi]:
            s = sd[i]
            k = -px[i] if s == "B" else px[i]
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
        for s, tag in (("B", "bid"), ("A", "ask")):
            arr, va = keys[s], vals[s]
            if not arr:
                for b, _ in BANDS:
                    out[f"{tag}_{b}"][gi] = out[f"{tag}_{b}"][gi - 1] if gi else 0.0
                continue
            best = -arr[0] if s == "B" else arr[0]
            num = np.zeros(len(BANDS)); den = np.zeros(len(BANDS))
            for kk, vv in zip(arr[:MAXLV], va[:MAXLV]):
                p = -kk if s == "B" else kk
                dd = abs(p - best) / best * 1e4
                for bi, (_, lim) in enumerate(BANDS):
                    if dd <= lim:
                        num[bi] += dd * vv; den[bi] += vv
                if dd > 100.0:
                    break                     # 100bp を超えたら以降は全帯に効かない(all を除く)
            for bi, (b, _) in enumerate(BANDS):
                out[f"{tag}_{b}"][gi] = num[bi] / den[bi] if den[bi] > 0 else 0.0
        gi += 1
        if i >= N and gi < n:
            for key in out:
                out[key][gi:] = out[key][gi - 1]
            break
    return out


CACHE = D / "book_slope_grain"


def build(days: list[str]) -> None:
    """日ごとに 100ms グリッドの傾きを算出してキャッシュする(分割実行できるように)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    for dt in days:
        f = CACHE / f"{dt}.parquet"
        if f.exists():
            continue
        t0 = int(datetime.strptime(dt, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
        grid = np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)
        sl = slopes(dt, grid)
        pl.DataFrame({"ts": grid, **{k: v for k, v in sl.items()}}).write_parquet(
            f, compression="zstd")
        print(f"{dt} 算出済", flush=True)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(sys.argv[2:] if len(sys.argv) > 2 else DAYS)
        return
    acc: dict = {}
    for dt in DAYS:
        t0 = int(datetime.strptime(dt, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
        grid = np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)
        mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                              columns=["ts", "mid", "is_crossed"])
              .filter(~pl.col("is_crossed")).sort("ts"))
        mts = mp["ts"].to_numpy(); lm = np.log(mp["mid"].to_numpy())
        j = np.searchsorted(mts, grid, side="right") - 1
        ok = (j >= 0) & (grid >= mts[0]) & (grid <= mts[-1])
        base = np.where(ok, lm[np.clip(j, 0, lm.size - 1)], np.nan)
        f = CACHE / f"{dt}.parquet"
        if not f.exists():
            continue
        c = pl.read_parquet(f)
        sl = {k: c[k].to_numpy() for k in c.columns if k != "ts"}
        for b, _ in BANDS:
            x = sl[f"bid_{b}"] - sl[f"ask_{b}"]
            for hname, k in HORIZONS.items():
                fut = np.full(grid.size, np.nan)
                fut[:-k] = base[k:]
                y = (fut - base) * 1e4
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 5000 or x[m].std() == 0 or y[m].std() == 0:
                    continue
                acc.setdefault((b, hname), []).append(
                    float(np.corrcoef(x[m], y[m])[0, 1]))
        print(f"{dt} 済", flush=True)

    res = {f"{b}|{h}": {"days": len(v), "median": float(np.median(v)),
                        "n_neg": int(sum(1 for z in v if z < 0))}
           for (b, h), v in acc.items()}
    (D / "book_slope_grain.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"\n{'深さ帯':>8} " + " ".join(f"{h:>12}" for h in HORIZONS))
    for b, _ in BANDS:
        row = []
        for h in HORIZONS:
            k = f"{b}|{h}"
            row.append(f"{res[k]['median']:>+8.4f}({res[k]['n_neg']:>2})" if k in res
                       else f"{'-':>12}")
        print(f"{b:>8} " + " ".join(row))
    print("\n※ 括弧内は 10 日中「符号が正だった日数」(負が期待される向き)")


if __name__ == "__main__":
    main()
