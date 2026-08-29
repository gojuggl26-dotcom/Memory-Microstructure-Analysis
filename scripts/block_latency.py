"""Hyperliquid のブロック間隔 = 遅延の床とその裾。

【なぜブロック間隔が「床」なのか】
  発注も取消も**ブロックに取り込まれて初めて効く**。したがって
  「判断してから効くまで」は、どれだけネットワークが速くても
  **次のブロックまでの待ち時間**を下回れない。
  裾(長いブロック間隔)はそのまま「取消が効かない時間」= 逆選択に晒される時間になる。

【測り方】
  lifecycle の `ts_open` / `ts_close` の**相異なる時刻**を集めるとブロック時刻の集合になる
  (同一ブロックの複数イベントは同じ ns を共有する。CLAUDE.md 記載)。
  DRAM に 1 件もイベントが無いブロックは現れないので、
  **これは「観測可能な更新間隔」の分布**であり、真のブロック生成間隔の上界である。
  忙しい時間帯ほど真値に近づくので、活動量で層別して両方を出す。

【裾の危険】
  長い間隔をまたいで価格がどれだけ動いたかを測る。
  「取消できない時間に、どれだけ不利になりうるか」がテールリスクの実体。

【出力】 data/block_latency.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

PIPE = Path("C:/Users/ii562/hl-l4-pipeline")
D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
QS = [0, 1, 10, 25, 50, 75, 90, 99, 99.9, 99.99, 100]
GAPS_MS = [200, 500, 1000, 2000, 5000, 10000]


def day_blocks(dt: str) -> np.ndarray | None:
    fs = glob.glob(str(PIPE / f"data/l2_v99/lifecycle/dt={dt}/*.parquet"))
    if not fs:
        return None
    d = pl.read_parquet(fs[0], columns=["ts_open", "ts_close"])
    t = np.concatenate([d["ts_open"].to_numpy(), d["ts_close"].drop_nulls().to_numpy()])
    t = np.unique(t[np.isfinite(t) & (t > 0)])
    return t if len(t) > 1000 else None


def mid_at(dt: str):
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    return ts, m["mid"].to_numpy()


def main() -> None:
    days = sorted(p.name.split("=")[1]
                  for p in (PIPE / "data/l2_v99/lifecycle").glob("dt=*"))
    all_iv: list[np.ndarray] = []
    gap_cnt = {g: 0 for g in GAPS_MS}
    gap_move: dict[int, list[float]] = {g: [] for g in GAPS_MS}
    n_iv = 0
    used = []
    for i, dt in enumerate(days):
        t = day_blocks(dt)
        if t is None:
            continue
        iv = np.diff(t)
        iv = iv[(iv > 0) & (iv < 3600 * 1_000_000_000)]     # 日跨ぎ・欠測は除く
        if len(iv) < 1000:
            continue
        used.append(dt)
        n_iv += len(iv)
        # 全部は持てないので日ごとに 20 万本まで無作為抽出
        rng = np.random.default_rng(1234 + i)
        all_iv.append(iv[rng.choice(len(iv), min(len(iv), 200_000), replace=False)]
                      if len(iv) > 200_000 else iv)
        mm = mid_at(dt)
        for g in GAPS_MS:
            sel = np.where(iv >= g * 1_000_000)[0]
            gap_cnt[g] += int(len(sel))
            if mm is not None and len(sel):
                s = sel[:2000]                              # 重い日は上限
                a = mm[1][np.clip(np.searchsorted(mm[0], t[s], side="right") - 1,
                                  0, len(mm[1]) - 1)]
                b = mm[1][np.clip(np.searchsorted(mm[0], t[s + 1], side="right") - 1,
                                  0, len(mm[1]) - 1)]
                ok = (a > 0) & (b > 0)
                gap_move[g].extend(np.abs(np.log(b[ok] / a[ok])) * 1e4)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(days)}", flush=True)
    iv = np.concatenate(all_iv) / 1e6                        # ms
    res = {
        "n_days": len(used), "n_intervals_total": n_iv,
        "n_sampled": int(len(iv)),
        "quantiles_ms": {str(q): float(np.percentile(iv, q)) for q in QS},
        "mean_ms": float(iv.mean()),
        "gaps": {str(g): {"count": gap_cnt[g],
                          "per_day": gap_cnt[g] / max(len(used), 1),
                          "share_of_time_pct": None,
                          "abs_move_bp_median":
                              float(np.median(gap_move[g])) if gap_move[g] else None,
                          "abs_move_bp_p90":
                              float(np.percentile(gap_move[g], 90)) if gap_move[g] else None,
                          "abs_move_bp_max":
                              float(np.max(gap_move[g])) if gap_move[g] else None}
                 for g in GAPS_MS},
    }
    # 「取消できない時間」が 1 日に占める割合
    for g in GAPS_MS:
        sel = iv[iv >= g]
        res["gaps"][str(g)]["share_of_time_pct"] = float(sel.sum() / iv.sum() * 100)
    (D / "block_latency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("完了")


if __name__ == "__main__":
    main()
