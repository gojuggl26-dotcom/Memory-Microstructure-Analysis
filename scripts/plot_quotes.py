"""仮想発注候補テーブルの概観を描く。

    uv run python scripts/plot_quotes.py --coin xyz:MU
出力: charts/<coin>_quotes_daily.png  日次の推移(標本が均質でないことを示す)
      charts/<coin>_quotes_dist.png   主な説明変数とラベルの分布

【読み方】
98 日を 1 つの母集団として扱ってはいけない。候補数・約定率・スプレッドが
期間内で 2 倍以上動くので、日次で見るか期間で層別すること。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_vol import (BASELINE, GRID, INK, INK2, MUTED, SURFACE,  # noqa: E402
                      legend, style)

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_A, C_B, C_C, C_D = "#2a78d6", "#e34948", "#1f8a5e", "#c48a12"


def daily(tag: str, coin: str) -> None:
    D = pl.read_parquet(DATA / f"quotes_day_{tag}.parquet").sort("dt")
    x = np.arange(D.height)
    lab = [s[5:] for s in D["dt"].to_list()]
    tk = x[::14]
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 6.6))

    a = ax[0, 0]
    a.bar(x, D["n"].to_numpy() / 1e3, color=C_A, width=0.85, lw=0)
    style(a, "", "候補行数(千行/日)")
    a.set_title("(a) 1 日あたりの候補行数 — 後半で 1.65 倍", color=INK,
                fontsize=10, loc="left")

    a = ax[0, 1]
    a.plot(x, 100 * D["fill"].to_numpy(), color=C_B, lw=1.6)
    a.axhline(100 * float(D["fill"].median()), color=MUTED, lw=1.0, ls="--",
              label=f"中央 {100*float(D['fill'].median()):.1f}%")
    style(a, "", "60 秒以内の約定率 (%)")
    legend(a, loc="upper left")
    a.set_title("(b) 約定率 — 32.2% (8/08) 〜 78.0% (7/29)", color=INK,
                fontsize=10, loc="left")

    a = ax[1, 0]
    a.plot(x, D["spr_med"].to_numpy(), color=C_C, lw=1.6)
    style(a, "", "スプレッド 中央値 (bp)", logy=True)
    a.set_title("(c) スプレッド — 前半 2.34bp → 後半 0.98bp", color=INK,
                fontsize=10, loc="left")

    a = ax[1, 1]
    v = D["pnl1_mean"].to_numpy()
    a.axhline(0, color=BASELINE, lw=1.0)
    a.fill_between(x, 0, v, color=C_D, alpha=0.35, lw=0)
    a.plot(x, v, color=C_D, lw=1.4)
    style(a, "", "1 秒 net PnL 平均 (bp)")
    a.set_title("(d) 手数料込みの 1 秒 PnL — 98 日中 97 日が負", color=INK,
                fontsize=10, loc="left")

    for a in ax.ravel():
        a.set_xticks(tk)
        a.set_xticklabels([lab[i] for i in tk], rotation=0)
        a.set_xlim(-1, D.height)
    fig.suptitle(f"{coin} 仮想発注候補テーブル — 日次の推移(期間内で標本の性質が変わる)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_quotes_daily.png", dpi=170)
    plt.close(fig)
    return int((v < 0).sum()), D


def dist(tag: str, coin: str) -> None:
    T = pl.read_parquet(DATA / f"quotes_sub_{tag}")
    B = T.filter(pl.col("side") == 1)
    fig, ax = plt.subplots(2, 4, figsize=(14.2, 6.4))
    spec = [
        ("obi1", B, (-1, 1), "OBI (最良気配)", "自分側が厚い →", False),
        ("ofi_1s", B, (-400, 400), "OFI 1 秒", "", False),
        ("spread_bp", B, (0, 12), "スプレッド (bp)", "", False),
        ("queue_ahead_qty", B, (0, 40), "自分の前の待ち行列", "", False),
        ("label_fill_lat_s", B, (0, 60), "約定までの秒数", "約定した行のみ", True),
        ("label_edge_bp", B, (-20, 20), "実現エッジ (bp)", "約定した行のみ", True),
        ("label_markout_1s_bp", B, (-15, 15), "1 秒 markout (bp)", "約定した行のみ", True),
        ("label_pnl_1s_bp", B, (-25, 15), "1 秒 net PnL (bp)", "手数料 0.088bp 控除後", True),
    ]
    for a, (c, F, rg, ti, sub, isl) in zip(ax.ravel(), spec):
        v = F[c].drop_nulls().to_numpy()
        a.hist(v, bins=90, range=rg, color=C_B if isl else C_A, lw=0,
               density=True, alpha=0.85)
        md = float(np.median(v))
        a.axvline(md, color=INK, lw=1.2, ls="--")
        if rg[0] < 0 < rg[1]:
            a.axvline(0, color=BASELINE, lw=1.0)
        style(a, ti, "密度")
        a.set_title(f"{ti}  中央 {md:+.3f}" + (f"{chr(10)}{sub}" if sub else ""),
                    color=INK, fontsize=9.5, loc="left")
    fig.suptitle(f"{coin} 候補テーブルの分布 — 上段 説明変数(t 時点で観測可能) / "
                 f"下段 ラベル(未来を使う)", color=INK, fontsize=11.5,
                 x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(CH / f"{tag}_quotes_dist.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    nneg, D = daily(tag, a.coin)
    print("pnl1_mean が負の日", nneg, "/", D.height)
    dist(tag, a.coin)
    print("書き出し", CH / f"{tag}_quotes_daily.png", CH / f"{tag}_quotes_dist.png")
