"""夜間の価格発見はどこで起きているか — の図。

    uv run python scripts/plot_nightdisc.py --coin xyz:MU

出力: charts/<tag>_nightdisc.png

配色は 2 色(#3b6fd4 = 背景・現物、#c2410c = 市場の刻み・perp)。
`palette_check.validate` で ok=True を確認済み。

x が確定する時刻 / y の期間: 図 D のみラグを持つ(凡例に明示)。ほかは記述。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mk = pl.read_csv(D / f"night_marks_{tag}.csv").sort("sec")
    ex = pl.read_csv(D / f"night_explain_{tag}.csv")
    ld = pl.read_csv(D / f"night_lead_{tag}.csv")

    x = mk["sec"].to_numpy() / 3600
    ismk = np.array([bool(s) for s in mk["mark"]])
    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.8))

    for i, (col, ttl, ylab) in enumerate((
            ("jump1", "A. その瞬間の 1 秒で mid はどれだけ動くか — "
                      "跳ねるのは米国の寄りだけ", "|Δmid| 1 秒(bp・中央値)"),
            ("asym300", "B. 直後 5 分 ÷ 直前 5 分 — "
                        "東京・ソウルの寄りが 48 点中 1 位", "後 5 分 ÷ 前 5 分"))):
        b = ax[0, i]
        v = mk[col].to_numpy()
        b.bar(x[~ismk], v[~ismk], width=0.42, color=C1, label="市場の刻みでない 37 点")
        b.bar(x[ismk], v[ismk], width=0.42, color=C2, label="市場の寄り・引け 11 点")
        b.set_xlim(-0.5, 24)
        b.set_xticks(range(0, 25, 3))
        b.set_xlabel("UTC の時刻")
        b.set_ylabel(ylab)
        b.set_title(ttl, loc="left")
        b.legend(fontsize=7.5, frameon=False, loc="upper left")
        top = mk.with_columns(rk=pl.col(col).rank("min", descending=True)) \
                .filter((pl.col("mark") != "") & (pl.col("rk") <= 5))
        for r in top.iter_rows(named=True):
            b.annotate(f"{r['mark']}\n{int(r['rk'])}位", (r["sec"] / 3600, r[col]),
                       fontsize=6.8, color=C2, ha="center",
                       xytext=(0, 4), textcoords="offset points")
        b.set_ylim(0, float(np.nanmax(v)) * 1.28)

    b = ax[1, 0]
    y = np.arange(ex.height)
    b.barh(y, ex["r2"], color=[C1, C2, CM], height=0.6)
    for i, (v, n) in enumerate(zip(ex["r2"], ex["adj_r2"])):
        b.text(v + 0.012, i, f"{v:.3f}(自由度調整 {n:.3f})", fontsize=8, va="center")
    b.set_yticks(y)
    b.set_yticklabels(ex["model"], fontsize=8.5)
    b.invert_yaxis()
    b.set_xlim(0, 1.05)
    b.set_xlabel("夜の値動き(perp の 00:00→08:00 UTC)の説明力 R²")
    b.set_title(f"C. 夜の値動きを何が説明するか(n={int(ex['n'][0])} 営業日・"
                f"標準偏差 275bp)", loc="left")
    b.text(0.03, 2.42, "指数先物にアジア現物を足す ΔR²=+0.078(p=0.004) / "
                       "アジア現物に指数先物を足す ΔR²=+0.120(p=8e-06)",
           fontsize=7.5, color=CM)

    b = ax[1, 1]
    y = np.arange(ld.height)
    b.barh(y, ld["corr_same"], color=C2, height=0.26, label="同じ 1 時間(同時点)")
    b.barh(y + 0.28, ld["corr_perp_lags"], color=C1, height=0.26,
           label="perp が 1 時間遅れて追う")
    b.barh(y + 0.56, ld["corr_asia_lags"], color=CM, height=0.26,
           label="相手が 1 時間遅れて追う")
    b.axvline(0, color=CM, lw=0.8)
    b.set_yticks(y + 0.28)
    b.set_yticklabels([f"{m}(n={n})" for m, n in zip(ld["market"], ld["n"])],
                      fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("perp の 1 時間リターンとの相関")
    b.set_title("D. 夜の 1 時間ごと — 同時に動くだけで、どちらも先行しない", loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="lower right")
    save(fig, f"{tag}_nightdisc.png",
         f"{a.coin}: 米国が閉まっている夜、値段を動かしているのはどこか"
         f"(2026-05-04 〜 09-10)")


if __name__ == "__main__":
    main()
