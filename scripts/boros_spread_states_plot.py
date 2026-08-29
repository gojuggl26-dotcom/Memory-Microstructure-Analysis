"""spread の継続時間 × 幅 の図。

主題は「**広いスプレッドは短命**」。ただし市場ごとに幅の尺度が違うので、
**市場内で順位化した幅**を主軸にする(素の tick でプールすると関係が洗い流される)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, CHARTS = ROOT / "data", ROOT / "charts"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


def main() -> int:
    d = (pl.read_parquet(DATA / "spread_states.parquet")
           .filter(~pl.col("censored") & ~pl.col("extrapolated")))
    du = d["duration"].to_numpy()
    wr = d["w_rank"].to_numpy()
    wt = d["width_tick"].to_numpy().astype(float)
    ttm = d["ttm_years"].to_numpy() * 365
    mk = d["market"].to_numpy()

    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 2.0], hspace=0.42, wspace=0.30)

    # (1) 市場内の幅の十分位 → 継続時間
    ax = fig.add_subplot(gs[0, 0])
    b = np.clip((wr * 10).astype(int), 0, 9)
    xs = np.arange(10) * 10 + 5
    med = [np.median(du[b == i]) for i in range(10)]
    med1 = [np.median(du[(b == i) & (du >= 1)]) for i in range(10)]
    p90 = [np.percentile(du[b == i], 90) for i in range(10)]
    ax.semilogy(xs, np.maximum(p90, 0.5), "o-", color="#1d4ed8", lw=2, ms=5,
                label="p90")
    ax.semilogy(xs, np.maximum(med1, 0.5), "s-", color="#2563eb", lw=2, ms=5,
                label="中央値(≥1 秒のみ)")
    ax.semilogy(xs, np.maximum(med, 0.5), "^--", color="#93c5fd", lw=1.8, ms=5,
                label="中央値(全状態)")
    ax.set_xlabel("市場内での spread の幅(順位 %)")
    ax.set_ylabel("継続時間(秒、対数)")
    ax.set_title("★広いほど短命 — 全列で単調", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # (2) 0 秒(同一ブロックで解消)の割合
    ax = fig.add_subplot(gs[0, 1])
    z0 = [np.mean(du[b == i] == 0) for i in range(10)]
    h1 = [np.mean(du[b == i] > 3600) for i in range(10)]
    ax.bar(xs, z0, 8, color="#fca5a5", label="0 秒(同一ブロックで解消)")
    ax.plot(xs, h1, "o-", color="#166534", lw=2, ms=5, label="1 時間超まで残る")
    ax.set_xlabel("市場内での spread の幅(順位 %)")
    ax.set_ylabel("割合")
    ax.set_title("広い状態は「一瞬で戻る」が本体\n"
                 f"0 秒率 {z0[0]:.0%} → {z0[-1]:.0%}", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # (3) 生存関数
    ax = fig.add_subplot(gs[0, 2])
    grid = np.logspace(0, 5.5, 90)
    for lo, hi, lab, c in [(0.0, 1 / 3, "狭(下位 1/3)", "#166534"),
                           (1 / 3, 2 / 3, "中", "#f59e0b"),
                           (2 / 3, 1.01, "広(上位 1/3)", "#dc2626")]:
        m = (wr >= lo) & (wr < hi)
        x = du[m]
        s = [(x > t).mean() for t in grid]
        ax.loglog(grid, np.maximum(s, 1e-5), color=c, lw=2, label=lab)
    for t, lab in [(60, "1分"), (3600, "1時間"), (86400, "1日")]:
        ax.axvline(t, color="#6b7280", lw=0.7, ls=":")
        ax.text(t, 0.6, lab, fontsize=7.5, ha="center", color="#374151")
    ax.set_xlabel("経過時間 t(秒)")
    ax.set_ylabel("P(継続時間 > t)")
    ax.set_title("生存関数 — 幅 3 群", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (4) 市場ごとの「狭/広」比
    ax = fig.add_subplot(gs[1, 0])
    r = []
    for m in np.unique(mk):
        s = mk == m
        n1 = s & (wr < 0.3) & (du >= 1)
        w1 = s & (wr > 0.7) & (du >= 1)
        if n1.sum() >= 50 and w1.sum() >= 50:
            r.append(np.median(du[n1]) / np.median(du[w1]))
    r = np.array(r)
    ax.hist(np.log10(r), bins=34, color="#bfdbfe", edgecolor="#2563eb")
    ax.axvline(0, color="#111827", lw=1.2)
    ax.set_xlabel("log₁₀(狭の中央値 / 広の中央値)")
    ax.set_ylabel("市場数")
    ax.set_title(f"★市場ごとの比 — 1 超が {int((r > 1).sum())}/{len(r)}\n"
                 f"中央 {np.median(r):.2f}倍", fontsize=10.5, loc="left")

    # (5) 満期までの残存で層別
    ax = fig.add_subplot(gs[1, 1])
    labs, na, wa = [], [], []
    for lo, hi, lab in [(0, 7, "≤1 週"), (7, 30, "1週–1月"), (30, 1e9, ">1 月")]:
        t = (ttm >= lo) & (ttm < hi)
        if t.sum() < 5000:
            continue
        labs.append(lab)
        na.append(np.median(du[t & (wr < 0.3) & (du >= 1)]))
        wa.append(np.median(du[t & (wr > 0.7) & (du >= 1)]))
    i = np.arange(len(labs))
    ax.bar(i - 0.19, na, 0.36, color="#166534", label="狭(下位 30%)")
    ax.bar(i + 0.19, wa, 0.36, color="#dc2626", label="広(上位 30%)")
    for k in i:
        ax.text(k, max(na[k], wa[k]) * 1.06, f"{na[k]/wa[k]:.1f}x",
                ha="center", fontsize=9, color="#111827")
    ax.set_xticks(i); ax.set_xticklabels(labs)
    ax.set_ylabel("継続時間の中央値(秒、≥1 秒)")
    ax.set_title("★満期が遠いほど差が開く", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (6) 幅の分布(tick)
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.log10(np.maximum(wt, 1)), bins=70, color="#e9d5ff",
            edgecolor="none")
    for t, lab in [(1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
        ax.axvline(np.log10(t), color="#6b7280", lw=0.7, ls=":")
    ax.set_xlabel("log₁₀(spread の幅、tick)")
    ax.set_ylabel("状態数")
    ax.set_title(f"幅の分布 — 中央 {np.median(wt):.0f} tick\n"
                 f"(pp 中央 {np.median(d['width_pp'].to_numpy()):.4f})",
                 fontsize=10.5, loc="left")

    fig.suptitle(f"Boros 全 188 市場 — spread の状態 {len(d):,} 件の継続時間と幅 "
                 f"(1,042 万イベントから抽出)", fontsize=12.5, y=0.985)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_spread_states.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
