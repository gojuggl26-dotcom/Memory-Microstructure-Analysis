"""所有銘柄を横並びにした比較図。

    uv run python scripts/plot_xcompare.py
出力: charts/xcompare.png

(a) 市場の姿(価格・スプレッド・厚み・約定頻度・ボラ)を銘柄ごとに
(b) 主要特徴量 × 銘柄の 1 秒先 前向き順位相関のヒートマップ
(c) 同 10 秒先
(d) 銘柄どうしで「効く特徴量ベクトル」がどれだけ似ているか
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap  # noqa: E402
import numpy as np                       # noqa: E402
import polars as pl                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
BLUE, GREEN, PURPLE, RED = "#2a78d6", "#0f8f63", "#4a3aa7", "#e34948"
DIV = LinearSegmentedColormap.from_list("d", [BLUE, "#eef1f4", RED])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "text.parse_math": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})


def short(coin):
    return coin.replace("xyz:", "")


def logfmt(a):
    from matplotlib.ticker import FuncFormatter, LogLocator
    a.yaxis.set_major_locator(LogLocator(base=10))
    a.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("0" if v == 0 else (f"{v:,.0f}" if v >= 1 else f"{v:g}"))))


def main():
    st = pl.read_csv(DATA / "xcompare_stats.csv")
    pr = pl.read_csv(DATA / "xcompare_pred.csv")
    coins = st["coin"].to_list()
    order = sorted(range(len(coins)), key=lambda i: st["mid_med"][i])
    coins = [coins[i] for i in order]
    st = st[order]
    lab = [short(c) for c in coins]

    fig, ax = plt.subplots(2, 3, figsize=(15.6, 9.2))

    # (a) 市場の姿 — 4 つの量を並べる
    metrics = [("spread_bp_med", "スプレッド中央値 (bp)", False),
               ("bid_sz_med", "最良買い数量の中央値 (枚)", True),
               ("trd_rate_10s_med", "約定頻度 (件/秒)", False),
               ("rv_10s_med", "実現ボラ 10s 中央値 (bp)", False)]
    for k, (col, ttl, logy) in enumerate(metrics):
        a = ax.flat[k]
        v = st[col].to_numpy()
        a.bar(range(len(coins)), v, color=BLUE, alpha=0.85)
        if logy:
            a.set_yscale("log"); logfmt(a)
        a.set_xticks(range(len(coins)), lab, rotation=30, fontsize=7.5, ha="right")
        a.set_title(f"({chr(97+k)}) {ttl}", loc="left")

    # (e) 予測力ヒートマップ 1s
    a = ax.flat[4]
    feats = pr.filter(pl.col("h") == "1s")["col"].unique(maintain_order=True).to_list()
    M = np.full((len(feats), len(coins)), np.nan)
    for i, f in enumerate(feats):
        for j, c in enumerate(coins):
            s = pr.filter((pl.col("col") == f) & (pl.col("coin") == c) & (pl.col("h") == "1s"))
            if s.height:
                M[i, j] = s["r_fwd"][0]
    vmax = np.nanmax(np.abs(M))
    im = a.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    a.set_xticks(range(len(coins)), lab, rotation=30, fontsize=7, ha="right")
    a.set_yticks(range(len(feats)), feats, fontsize=6.2)
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.045, pad=0.02)
    a.set_title("(e) 主要特徴量 × 銘柄 の 1 秒先 前向き順位相関", loc="left")

    # (f) 銘柄どうしの「効く特徴量ベクトル」の相関
    a = ax.flat[5]
    # 各銘柄の (特徴量 × 1s の r_fwd) ベクトルの相互相関
    V = np.full((len(coins), len(feats)), np.nan)
    for j, c in enumerate(coins):
        for i, f in enumerate(feats):
            s = pr.filter((pl.col("col") == f) & (pl.col("coin") == c) & (pl.col("h") == "1s"))
            if s.height:
                V[j, i] = s["r_fwd"][0]
    C = np.full((len(coins), len(coins)), np.nan)
    for i in range(len(coins)):
        for j in range(len(coins)):
            m = np.isfinite(V[i]) & np.isfinite(V[j])
            if m.sum() >= 4:
                C[i, j] = np.corrcoef(V[i][m], V[j][m])[0, 1]
    im2 = a.imshow(C, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    a.set_xticks(range(len(coins)), lab, rotation=30, fontsize=7, ha="right")
    a.set_yticks(range(len(coins)), lab, fontsize=7)
    a.grid(False)
    for i in range(len(coins)):
        for j in range(len(coins)):
            if np.isfinite(C[i, j]):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if abs(C[i, j]) > 0.6 else INK)
    fig.colorbar(im2, ax=a, fraction=0.045, pad=0.02)
    a.set_title("(f) 銘柄間で「効き方」はどれだけ似ているか", loc="left")

    fig.suptitle("所有 Hyperliquid 銘柄の横並び — 市場の姿と特徴量の効き方",
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(CH / "xcompare.png", facecolor=SURFACE, bbox_inches="tight")
    print("保存:", CH / "xcompare.png", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
