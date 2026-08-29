r"""Pressure(最良気配を最も重くした指数減衰)× 将来リターンの図。

主題は「**減衰長 τ をどこに置くと信号が出るか**」。
  τ→0   … 最良 1 レベルのみ
  τ=1〜64 … exp(−d/τ)、d は最良からの tick 距離
  τ→∞   … 片側の総量

出すもの:
  上段 3 枚 = long / short / diff について、τ ごとの符号一貫性(全窓)
  下段 3 枚 = 同じものを**ブロック跨ぎの窓に限定**したもの(執行可能な標本)
  右列      = τ の梯子(z と有効標本)と、帰無対照・格子全体
"""
from __future__ import annotations

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
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 125,
})

HOR = [1, 2, 3, 5, 10, 20, 30]
TAGS = ["bq", "ew1", "ew4", "ew16", "ew64", "depth"]
TAGLAB = {"bq": "τ→0 最良のみ", "ew1": "τ=1", "ew4": "τ=4",
          "ew16": "τ=16 ★", "ew64": "τ=64", "depth": "τ→∞ 総量"}
TAGCOL = {"bq": "#a855f7", "ew1": "#2563eb", "ew4": "#0891b2",
          "ew16": "#16a34a", "ew64": "#ea580c", "depth": "#6b7280"}
SIDES = ["long", "short", "diff"]
SIDEPRED = {"long": +1, "short": -1, "diff": +1}
SIGNALS = [f"{t}_{s}" for t in TAGS for s in SIDES]
BONF_Z = 3.79      # 0.05/252 の両側に対応するおおよその z


def zval(d, sg, k, col="beta"):
    v = d.filter((pl.col("signal") == sg) & (pl.col("k") == k))[col].to_numpy()
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, 0
    return ((v > 0).sum() - len(v) / 2) / np.sqrt(len(v) / 4), len(v)


def main() -> int:
    d = pl.read_parquet(DATA / "pressure_results.parquet")
    tp = DATA / "pressure_results_tradeable.parquet"
    dt = pl.read_parquet(tp) if tp.exists() else None
    x = np.arange(len(HOR))

    fig = plt.figure(figsize=(19.5, 10.4))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.30)

    def panel(ax, dd, side, title, sub):
        for t in TAGS:
            z = [zval(dd, f"{t}_{side}", k)[0] * SIDEPRED[side] for k in HOR]
            ax.plot(x, z, "o-", color=TAGCOL[t], lw=2, ms=4.5, label=TAGLAB[t])
        ax.axhline(0, color="#111827", lw=1.0)
        for s_ in (+1, -1):
            ax.axhline(s_ * BONF_Z, color="#dc2626", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in HOR])
        ax.set_xlim(-0.4, len(HOR) - 0.6); ax.set_ylim(-10, 10)
        ax.set_xlabel("予測ホライズン k(イベント)")
        ax.set_ylabel("符号検定 z(予測どおりの向きを正)")
        ax.set_title(f"{title}\n{sub}", fontsize=10.5, loc="left")

    for j, side in enumerate(SIDES):
        panel(fig.add_subplot(gs[0, j]), d, side,
              f"全窓 — {side}",
              "破線 = Bonferroni(0.05/252)の z=±3.79")
        if dt is not None:
            panel(fig.add_subplot(gs[1, j]), dt, side,
                  f"★ブロック跨ぎ限定 — {side}",
                  "同一ブロック内は執行不能・同一取引の続きを含む")
    fig.axes[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")

    # --- 右上: τ の梯子 ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    src = dt if dt is not None else d
    xt = np.arange(len(TAGS))
    # ★|z| ではなく**予測どおりの向きにそろえた z** の最大を取る。
    #   |z| だと「予測と逆向きに強い」セルが成功のように見えてしまう。
    for side, c, m in [("long", "#2563eb", "o"), ("short", "#dc2626", "s"),
                       ("diff", "#166534", "^")]:
        best = [max((zval(src, f"{t}_{side}", k)[0] * SIDEPRED[side] for k in HOR),
                    default=np.nan) for t in TAGS]
        ax.plot(xt, best, m + "-", color=c, lw=2, ms=6, label=side)
    ax.axhline(BONF_Z, color="#dc2626", lw=0.9, ls="--")
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_xticks(xt); ax.set_xticklabels([TAGLAB[t] for t in TAGS], rotation=30,
                                          ha="right", fontsize=8)
    ax.set_ylabel("全地平での最大 z(予測どおりの向きを正)")
    ax.set_title("★τ の梯子 — 最良をどれだけ重くするか\n"
                 "(ブロック跨ぎ限定。破線 = Bonferroni)", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # --- 右下: 有効標本と帰無対照 -----------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ne, fz = [], []
    for t in TAGS:
        s = d.filter(pl.col("signal") == f"{t}_diff")
        ne.append(float(np.median(s["n_eff"].to_numpy())) if s.height else np.nan)
        fz.append(float(np.median(s["frac_zero"].to_numpy())) if s.height else np.nan)
    ax.bar(xt, ne, 0.6, color="#bfdbfe", edgecolor="#2563eb",
           label="有効標本 n_eff(市場中央値)")
    ax.set_xticks(xt); ax.set_xticklabels([TAGLAB[t] for t in TAGS], rotation=30,
                                          ha="right", fontsize=8)
    ax.set_ylabel("有効標本(x≠0 の行数)")
    a2 = ax.twinx(); a2.grid(False)
    a2.plot(xt, fz, "o-", color="#dc2626", lw=2, ms=6, label="x が厳密に 0 の割合")
    a2.set_ylabel("ゼロ率", color="#dc2626"); a2.set_ylim(0, 1)
    # 帰無対照は**全 126 セル**で取る(diff だけだと本文の数字と食い違う)
    pl_ = [abs(zval(src, sg, k, "beta_placebo")[0])
           for sg in SIGNALS for k in HOR]
    ax.set_title("★τ を上げるとゼロ膨張が解ける\n"
                 f"帰無対照は全 {len(pl_)} セルで最大 |z| = {np.nanmax(pl_):.2f}(偶然の範囲)",
                 fontsize=10.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper center")

    nm = d["market"].n_unique()
    fig.suptitle(f"Boros 全 {nm} 市場 — 最良気配を最も重くした指数減衰 Pressure と将来リターン "
                 "(重み exp(−d/τ)、d は最良からの tick 距離)", fontsize=13, y=0.975)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_pressure_ew.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
