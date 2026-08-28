"""スプレッド幅・注文量・約定量と OBI / OFI の関係図を描く。

図 1  7 変数のスピアマン順位相関行列(同時点)
図 2  OBI / OFI の帯ごとの スプレッド幅・注文量・約定量・約定の発生率(2 行 4 列)

相関は順位で取る。3 つの量はいずれも裾が重く(第 99 パーセンタイルが中央値の
十数倍)、ピアソン相関では少数の大きい行が値を支配してしまうため。

約定量は 85.3% のイベントで 0 なので、帯ごとの中央値はすべて 0 になり意味を持たない。
**全イベントの平均**と**約定が起きた割合**を別々のパネルに分けて示す。
縦軸を 2 本持つ図は目盛りの合わせ方が恣意的になるので作らない。

    uv run python scripts/plot_spread_flow.py --coin xyz:MU
出力: charts/<coin>_spread_flow_corr.png, charts/<coin>_spread_flow_bins.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_SPREAD, C_QSIZE, C_FILL, C_RATE = "#1baf7a", "#2a78d6", "#eb6834", "#4a3aa7"

VAR_JA = {"spread_bp": "スプレッド幅 (bp)", "qsize": "注文量 (枚)", "fillsz": "約定量 (枚)",
          "obi": "OBI", "abs_obi": "|OBI|", "ofi_z": "OFI (正規化)", "abs_ofi_z": "|OFI|"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    # --- 図 1: 相関行列 -------------------------------------------------------
    C = pl.read_csv(ROOT / "data" / f"spread_flow_corr_{tag}.csv")
    variables = C["var"].to_list()
    R = np.array([[C[v][i] for v in variables] for i in range(len(variables))])

    fig, ax = plt.subplots(figsize=(9.8, 8.2), dpi=170)
    # 発散配色は中間を無彩色にする(0 が「関係なし」と読めるように)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "corr", ["#e34948", "#f0efec", "#2a78d6"])
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
    lab = [VAR_JA[v] for v in variables]
    ax.set_xticks(range(len(variables)), lab, fontsize=9.5, rotation=30, ha="right")
    ax.set_yticks(range(len(variables)), lab, fontsize=9.5)
    for i in range(len(variables)):
        for j in range(len(variables)):
            ax.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                    color=INK if abs(R[i, j]) < 0.6 else SURFACE,
                    fontsize=10, weight="bold" if i != j else "normal")
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{a.coin} イベントごとの 5 つの量の関係(スピアマン順位相関)",
                 loc="left", color=INK, fontsize=13, pad=46, weight="bold")
    ax.text(0, 1.085, "同じイベントで測った量どうしの同時点の関係で、予測力ではない。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.text(0, 1.045, "裾が重いので順位相関。抜き取り 1,054,115 イベント。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.72, label="順位相関")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.835, bottom=0.14)
    out1 = ROOT / "charts" / f"{tag}_spread_flow_corr.png"
    fig.savefig(out1)
    plt.close(fig)
    print(f"[chart] {out1}")

    # --- 図 2: 帯ごとの代表値 --------------------------------------------------
    B = pl.read_csv(ROOT / "data" / f"spread_flow_bins_{tag}.csv")
    specs = [
        ("spread_bp", "スプレッド幅 (bp)", C_SPREAD, "med", "中央値と四分位"),
        ("qsize", "注文量 (枚)", C_QSIZE, "med", "中央値と四分位"),
        ("fillsz", "約定量 (枚)", C_FILL, "mean", "全イベントの平均"),
        ("fill_rate", "約定が起きた割合 (%)", C_RATE, "rate", "全イベントに対する割合"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.8), dpi=170)
    for r, feat in enumerate(("OBI", "OFI_z")):
        g = B.filter(pl.col("feat") == feat).sort("bin_i")
        x = np.arange(g.height)
        labels = g["bin"].to_list()
        for c, (v, ylab, col, stat, sub) in enumerate(specs):
            ax = axes[r][c]
            style(ax)
            if stat == "med":
                ax.fill_between(x, g[f"{v}_q1"].to_numpy(), g[f"{v}_q3"].to_numpy(),
                                color=col, alpha=0.16, lw=0, zorder=2)
                y = g[f"{v}_med"].to_numpy()
            elif stat == "mean":
                y = g[f"{v}_mean_full"].to_numpy()
            else:
                y = g["fill_rate"].to_numpy() * 100
            ax.plot(x, y, color=col, lw=2.2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
            ax.set_xticks(x, labels, fontsize=7.5, rotation=45, ha="right")
            ax.set_ylabel(ylab, color=INK2, fontsize=9)
            ax.set_ylim(bottom=0)
            name = "OBI の帯" if feat == "OBI" else "OFI の帯"
            ax.set_title(f"{name} → {ylab.split(' (')[0]}({sub})", loc="left",
                         color=INK, fontsize=9.5, pad=8, weight="bold")

    fig.suptitle(f"{a.coin} OBI / OFI の帯ごとの スプレッド幅・注文量・約定量",
                 x=0.005, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.005, 0.945,
             "上段が OBI(板の残高の偏り)、下段が OFI(板の変化の偏り、正規化)。"
             "横軸の帯は分布を見る前に機械的に決めた区切り(OBI は等幅 0.25、OFI は ±0.5/±1/±2σ)。"
             "すべて同時点の関係で、予測力ではない。",
             color=INK2, fontsize=9.5)
    fig.text(0.005, 0.008,
             "出所: Hyperliquid L4 (Artemis) l2/bbo と node_fills / 窓 2026-05-04〜08-10 / "
             "35,171,746 イベント(クロス除外後)",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.875, bottom=0.135,
                        hspace=0.78, wspace=0.32)
    out2 = ROOT / "charts" / f"{tag}_spread_flow_bins.png"
    fig.savefig(out2)
    print(f"[chart] {out2}")


if __name__ == "__main__":
    main()
