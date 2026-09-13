"""十分位分析の配管そのものを、合成データで検算する。

    uv run python scripts/check_decile_null.py --coin xyz:MU

出力: data/decile_check_<coin>.csv

なぜ要るか
----------
実データを 1 営業日ずらした帰無対照が Bonferroni を 25% も通ったので、
「検定の設計が悪いのか、フィルタが壊れているのか」を切り分ける必要があった。
**構造を持たない合成特徴量を同じ配管に通す**のがいちばん速い。

    白色雑音            … |t|>1.96 が名目どおり 5% 前後、Bonferroni 通過は 0 本
    日内パターンだけの雑音 … 同上(時間帯だけでは偽の有意は出ない)

これが出れば配管は正しい。実際そうなり、真犯人は **polars の `NaN > 数値` が
true になる**仕様で、十分位が潰れた特徴量の `t = NaN` が全部「有意」に
数えられていたことだった(`.is_finite()` を足したらプラセボは 0.0%)。

x が確定する時刻 / y の期間: 合成特徴量は格子点 T に紐づけるだけ。
目的変数は (T, T+h]。先読みは無い。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_decile import NPD, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(7)

    fw = pl.read_parquet(D / f"fwd_{tag}.parquet").sort("dt", "sec")
    yn = [c for c in fw.columns if c.startswith("fwd_")]
    days = fw["dt"].unique(maintain_order=True).to_list()
    nd = len(days)
    dmap = {d: i for i, d in enumerate(days)}
    day_all = np.array([dmap[d] for d in fw["dt"].to_list()], np.int64)
    Y = np.column_stack([fw[c].to_numpy().astype(np.float64) for c in yn])
    fin = np.all(np.isfinite(Y), axis=1)

    X = {}
    for i in range(a.n):
        X[f"iid{i}"] = rng.standard_normal(fin.size)[fin]
    for i in range(a.n):
        prof = rng.standard_normal(NPD)                  # 日内の形だけ持つ
        X[f"tod{i}"] = (np.tile(prof, nd)[fin]
                        + 0.3 * rng.standard_normal(fin.size)[fin])
    _, S = run(X, Y[fin].T, yn, day_all[fin], nd, tag)
    S.write_csv(D / f"decile_check_{tag}.csv")

    ncell = 229 * len(yn)          # 本番の検定数に合わせた閾値で判定する
    thr = float(stats.t.ppf(1 - 0.05 / ncell / 2, nd - 1))
    print(f"合成特徴量 {a.n*2} 本 × 目的 {len(yn)} 本 = {S.height} 通り / "
          f"Bonferroni の |t| 閾値 {thr:.2f}(本番の {ncell:,} 通りに合わせた)")
    fi = pl.col("t").is_finite()
    for k, lab in (("iid", "白色雑音(構造なし)"), ("tod", "日内パターンだけ持つ雑音")):
        T = S.filter(pl.col("feature").str.starts_with(k) & fi)
        x = T.filter(pl.col("t").abs() > 1.96).height
        y = T.filter(pl.col("t").abs() > thr).height
        print(f"  {lab:<24} n={T.height:>4}  |t|>1.96 {x:>4} ({x/T.height*100:4.1f}%) "
              f"/ |t|>{thr:.2f} {y:>4}  最大 |t| {float(T['t'].abs().max()):.1f}")
    print("  → 名目 5% 前後で Bonferroni 通過が 0 本なら、配管は正しい")


if __name__ == "__main__":
    main()
