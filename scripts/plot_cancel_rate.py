"""キャンセル率の傾きと将来 log リターンの図。

    uv run python scripts/plot_cancel_rate.py --coin xyz:MU
出力: charts/<coin>_cancel_rate.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cancel_rate import GRID_NS, cancels_for_day, grid_day  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
UP, DOWN, WARN = "#e34948", "#2a78d6", "#f59e0b"
HOR = [100, 300, 500, 1000, 5000, 10000, 30000, 60000]
HLAB = ["100ms", "300ms", "500ms", "1s", "5s", "10s", "30s", "60s"]

# 例に使う窓は結果を見てから選ばない。機械的な規則で決める:
#   最も活況な日の、米国市場が開く 13:30 UTC から 60 秒
EX_DAY, EX_HHMM, EX_SEC = "2026-07-29", (13, 30), 60


def example(coin: str):
    tag = coin.replace(":", "_")
    bb = (pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
          .filter((pl.col("dt") == EX_DAY) & (pl.col("best_ask") > pl.col("best_bid"))
                  & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)).sort("ts"))
    fl = (pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                          columns=["ts", "sz", "side", "crossed", "dt"])
          .filter((pl.col("dt") == EX_DAY) & pl.col("crossed")).sort("ts"))
    c = cancels_for_day(bb, fl)
    g = grid_day(c, 1000, 10)
    ts = c["ts"]
    t0 = (ts[0] // GRID_NS) * GRID_NS
    day_ns = (t0 // 86_400_000_000_000) * 86_400_000_000_000
    start = day_ns + (EX_HHMM[0] * 3600 + EX_HHMM[1] * 60) * 1_000_000_000
    i0 = int((start - t0) // GRID_NS)
    i1 = i0 + EX_SEC * 10
    sl = np.arange(i0, i1)
    mid_idx = np.full(len(g["ci"]), -1, dtype=np.int64)
    gi = np.clip(((ts - t0) // GRID_NS).astype(np.int64), 0, len(g["ci"]) - 1)
    mid_idx[gi] = np.arange(len(ts))
    np.maximum.accumulate(mid_idx, out=mid_idx)
    mid = c["mid"][np.where(mid_idx >= 0, mid_idx, 0)]
    return sl, g["cra"][sl], g["crb"][sl], g["ci"][sl], g["slope"][sl], mid[sl]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"cancel_rate_cells_{tag}.parquet")
    B = pl.read_parquet(ROOT / "data" / f"cancel_rate_bins_{tag}.parquet")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.6, 10.6), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.24,
                          left=0.075, right=0.975, top=0.800, bottom=0.075)

    # ---- (1) 数列を関数化して傾きを取る、の実例 ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    sl, cra, crb, ci, slope, mid = example(a.coin)
    n_show = 300                                    # 30 秒 = 300 点
    x = (sl[:n_show] - sl[0]) / 10.0
    ax.plot(x, ci[:n_show], "-", color="#111827", lw=1.0, alpha=0.75,
            marker="o", ms=2.2, label="CI(100ms 刻み)")
    # 直近 10 点に当てた OLS を、2 秒ごとに実際の線分として重ねる
    m = 10
    for j in range(m - 1, n_show, 20):
        if not np.isfinite(slope[j]):
            continue
        xs = np.arange(j - m + 1, j + 1)
        seg = ci[xs]
        if not np.isfinite(seg).all():
            continue
        xr = xs / 10.0                      # xs は局所インデックス。秒に直すだけ
        cen = xr.mean()
        yy = seg.mean() + slope[j] * (xr - cen) * 10.0
        ax.plot(xr, yy, "-", color=UP if slope[j] > 0 else DOWN, lw=2.4, alpha=0.9)
    ax.axhline(0, color=BASELINE, lw=0.9)
    ax.set_xlabel("経過秒(2026-07-29 13:30 UTC から)", color=INK2, fontsize=10)
    ax.set_ylabel("CI(キャンセル不均衡)", color=INK2, fontsize=10)
    ax.set_title("① 数列を関数化して傾きを取る(実例・30 秒)\n"
                 "太線 = 直近 10 点(1 秒)に当てた OLS。赤 = 傾き正、青 = 傾き負",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower left")

    # ---- (2) 用量反応 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for h, cl in [(1000, DOWN), (10000, "#0891b2"), (60000, "#7c3aed")]:
        s = (B.filter((pl.col("day_type") == "立会日") & (pl.col("h_ms") == h))
             .group_by("dec").agg(n=pl.col("n").sum(), t=pl.col("sum").sum()).sort("dec"))
        v = (s["t"] / s["n"] * 1e4).to_numpy()
        ax.plot(np.arange(1, len(v) + 1), v, "o-", color=cl, lw=1.9, ms=4.5,
                label=HLAB[HOR.index(h)])
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("CI の傾きの十分位(1 = 最も負、10 = 最も正)", color=INK2, fontsize=10)
    ax.set_ylabel("平均 log リターン[bp]", color=INK2, fontsize=10)
    ax.set_title("② 傾きが大きいほどリターンも大きい(用量反応)\n"
                 "中央付近の凹みは傾きがちょうど 0 の 14.5% が入るため",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, title="ホライズン",
              title_fontsize=8.5)

    # ---- (3) 効果とホライズン ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(HOR))
    for dtl, cl in [("立会日", DOWN), ("閉場日", WARN)]:
        s = C.filter(pl.col("day_type") == dtl).sort("h_ms")
        d = s["diff_bp"].to_numpy()
        ax.plot(xs, d, "o-", color=cl, lw=2, ms=5, label=f"{dtl} 傾き>0 − 何もしない")
        ax.fill_between(xs, s["ci_lo"].to_numpy(), s["ci_hi"].to_numpy(),
                        color=cl, alpha=0.16, lw=0)
        ax.plot(xs, s["placebo_diff_bp"].to_numpy(), ":", color=cl, lw=1.5,
                label=f"{dtl} 帰無対照")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels(HLAB, fontsize=9)
    ax.set_xlabel("予測ホライズン", color=INK2, fontsize=10)
    ax.set_ylabel("平均 log リターンの差[bp]", color=INK2, fontsize=10)
    ax.set_title("③ 傾きが正のとき log リターンは正 — ただし 5 秒で頭打ち\n"
                 "帯は日単位ブロックブートストラップの 95% 区間",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    # ---- (4) 費用との比較 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    s = C.filter(pl.col("day_type") == "立会日").sort("h_ms")
    ax.bar(xs, s["diff_bp"].to_numpy(), color=DOWN, width=0.62, label="効果(傾き>0 の上乗せ)")
    half = 0.62
    ax.axhline(half, color=UP, lw=2.0)
    ax.text(len(HOR) - 0.4, half * 1.04, f"スプレッドの半分 {half:.2f} bp(片道の費用)",
            ha="right", va="bottom", color=UP, fontsize=9.5, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(HLAB, fontsize=9)
    ax.set_ylim(0, half * 1.28)
    ax.set_xlabel("予測ホライズン", color=INK2, fontsize=10)
    ax.set_ylabel("bp", color=INK2, fontsize=10)
    ax.set_title("④ ★効果は片道の費用の 1/8 以下\n"
                 "スプレッドを跨いで取りに行くと必ず負けになる",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    for j, v in enumerate(s["diff_bp"].to_numpy()):
        ax.annotate(f"{v:.3f}", (j, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color=INK2)

    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            if ax not in (fig.axes[1],):
                ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(BASELINE); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{a.coin} キャンセル率の傾きは将来の log リターンを説明するか"
                 f"(100ms 格子・立会日 67 日 / 閉場日 31 日)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.075, 0.930,
             "CR = [t−Δ, t] のキャンセル数量 ÷ 同じ窓の平均深さ、"
             "CI = (CR_ask − CR_bid)/(CR_ask + CR_bid + ε)。傾きは CI の直近 10 点(1 秒)の OLS。\n"
             "★キャンセルは bbo の最良気配の減少から約定分を引いて復元したもので、"
             "板の奥は含まない(lifecycle が DEEP_ARCHIVE のため)。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo + node_fills を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_cancel_rate.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
