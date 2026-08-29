"""spread duration の検出可否を図にする。

対比が主題なので、**検出できない量(spread duration / spread)と
検出できる量(book_slope_diff)を同じ軸で並べる**。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA, CHARTS = ROOT / "data", ROOT / "charts"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})
LAB = {"log_dur": "spread duration  log(1+秒)",
       "spread_pp": "spread  (pp)",
       "slope_diff": "book_slope_diff  (pp)"}
COL = {"log_dur": "#dc2626", "spread_pp": "#f59e0b", "slope_diff": "#2563eb"}


def main() -> int:
    R = json.loads((DATA / "duration_pooled.json").read_text(encoding="utf-8"))["results"]
    hs = sorted({r["horizon"] for r in R})
    xs = ["log_dur", "spread_pp", "slope_diff"]

    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 2.0], hspace=0.40, wspace=0.26)

    # 上段: 当てはめ曲線(地平 3600s)
    for c, xc in enumerate(xs):
        ax = fig.add_subplot(gs[0, c])
        for h, ls, al in zip(hs, ["-", "--"], [0.55, 1.0]):
            r = next(z for z in R if z["x"] == xc and z["horizon"] == h)
            xg = np.array(r["grid_x"])
            ax.fill_between(xg, r["ci_lo"], r["ci_hi"], color=COL[xc],
                            alpha=0.10 * al + 0.05, lw=0)
            ax.plot(xg, r["grid_f"], ls, color=COL[xc], lw=2.0, alpha=al,
                    label=f"{h}s  edf={r['edf']:.1f}  λ={r['lambda']:.3g}")
            bx, by = np.array(r["bin_x"]), np.array(r["bin_y"])
            m = np.isfinite(bx) & np.isfinite(by)
            if h == hs[-1]:
                ax.plot(bx[m], by[m], "o", ms=3.6, color=COL[xc], alpha=0.55, mew=0)
        ax.axhline(0, color="#111827", lw=0.9)
        ax.set_xlabel(LAB[xc])
        if c == 0:
            ax.set_ylabel("将来の mid 変化(因果標準化)")
        r3 = next(z for z in R if z["x"] == xc and z["horizon"] == hs[-1])
        ok = r3["n_pos_nested"] == r3["n_mk"]
        ax.set_title(f"{'★検出' if ok else '検出できず'}  "
                     f"正の市場 {r3['n_pos_nested']}/{r3['n_mk']}",
                     fontsize=11, loc="left",
                     color="#111827" if ok else "#6b7280")
        ax.legend(fontsize=8, frameon=False)

    # 下段左: 市場ごとの相関
    ax = fig.add_subplot(gs[1, :2])
    w = 0.26
    for i, xc in enumerate(xs):
        r = next(z for z in R if z["x"] == xc and z["horizon"] == 3600)
        cs = np.array(r["cors"]); cp = np.array(r["cors_placebo"])
        pos = np.arange(len(cs)) + (i - 1) * w
        ax.bar(pos, cs, w * 0.9, color=COL[xc], label=LAB[xc].split("  ")[0])
        ax.plot(pos, cp, "_", color="#6b7280", ms=9, mew=1.6)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([f"m{m}" for m in [150, 151, 159, 163, 174, 175, 182]])
    ax.set_ylabel("抜いた市場での相関")
    ax.set_title("市場ごとの標本外相関(地平 3600s、1 市場を抜いて学習)"
                 "  —  灰の横棒はプラセボ", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, ncol=3)

    # 下段右: spread duration の分布
    ax = fig.add_subplot(gs[1, 2])
    v = []
    for m in [150, 151, 159, 163, 174, 175, 182]:
        p = DATA / f"event_book_{m}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p).filter(~pl.col("extrapolated")).sort("ts")
        ts = d["ts"].to_numpy().astype(float)
        sp = d["spread_pp"].to_numpy().astype(float)
        chg = np.r_[True, sp[1:] != sp[:-1]]
        v.append(ts - np.maximum.accumulate(np.where(chg, ts, -np.inf)))
    v = np.concatenate(v)
    ax.hist(np.log1p(v), bins=70, color="#fca5a5", edgecolor="none")
    for s, lab in [(1, "1s"), (60, "1分"), (3600, "1時間"), (86400, "1日")]:
        ax.axvline(np.log1p(s), color="#374151", lw=0.8, ls=":")
        ax.text(np.log1p(s), ax.get_ylim()[1] * 0.94, lab, fontsize=7.5,
                ha="center", color="#374151")
    ax.set_xlabel("log(1 + spread duration 秒)")
    ax.set_ylabel("件数")
    ax.set_title(f"spread duration の分布(7 市場 {len(v):,} イベント)\n"
                 f"0 秒 {np.mean(v == 0):.0%} / 1 分未満 {np.mean(v < 60):.0%} / "
                 f"1 時間超 {np.mean(v > 3600):.0%}", fontsize=10, loc="left")

    fig.suptitle("Boros KuCoin 7 市場 — spread duration は検出可能か / "
                 "平滑化スプライン min{Σ(y−f(x))² + λ∫f''(x)²dx}、"
                 "λ は入れ子 LOO で機械選択", fontsize=12.5, y=0.985)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_duration_pooled.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
