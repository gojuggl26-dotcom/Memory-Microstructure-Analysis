"""xfeat の予測力を銘柄ごとに 1 枚にまとめる。

    uv run python scripts/plot_xfeat.py --coin xyz:AMD
出力: charts/<tag>_xfeat.png

配色は 3 色まで(青/緑/紫)、符号のある量は発散、帰無対照は輪郭。
対数軸ラベルの生 LaTeX 化を避けるため logfmt を使う。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import polars as pl                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_xfeat import FLAB               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
BLUE, GREEN, PURPLE, RED = "#2a78d6", "#0f8f63", "#4a3aa7", "#e34948"
HSEC = [0.1, 0.5, 1, 5, 10, 30, 60]

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


def famkey(f):
    m = re.match(r"(\d+)(?:\.(\d+))?", f)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (999, 0)


def logfmt(a, axis="y"):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax = a.yaxis if axis == "y" else a.xaxis
    ax.set_major_locator(LogLocator(base=10))
    ax.set_major_formatter(FuncFormatter(
        lambda v, _: ("0" if v == 0 else (f"{v:,.0f}" if v >= 1 else f"{v:g}"))))
    ax.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    p = pl.read_csv(DATA / f"xfeat_pred_{tag}.csv").with_columns(ab=pl.col("r_fwd").abs())
    dp = DATA / f"xfeat_daily_{tag}.csv"
    daily = pl.read_csv(dp) if dp.exists() else None

    fig = plt.figure(figsize=(14.6, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.15, 1, 1],
                          wspace=0.34, hspace=0.34)
    axA = fig.add_subplot(gs[:, 0])

    # (a) family × horizon heatmap
    piv = (p.group_by(["family", "h"]).agg(pl.col("ab").max())
           .pivot(on="h", index="family", values="ab"))
    fams = sorted(piv["family"].to_list(), key=famkey)
    M = np.array([[piv.filter(pl.col("family") == f)[h][0] for h in FLAB] for f in fams], float)
    im = axA.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=float(np.nanmax(M)))
    axA.set_xticks(range(len(FLAB)), FLAB, fontsize=7.5)
    axA.set_yticks(range(len(fams)), [f[:24] for f in fams], fontsize=6.6)
    axA.grid(False)
    fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02)
    for i in range(len(fams)):
        j = int(np.nanargmax(M[i]))
        axA.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=5.6,
                 color="white" if M[i, j] > np.nanmax(M) * 0.55 else INK)
    axA.set_title("(a) 分類ごとの最大 |順位相関|", loc="left")

    # (b) 1s top features
    a1 = fig.add_subplot(gs[0, 1])
    t1 = p.filter(pl.col("h") == "1s").sort("ab", descending=True).head(14)
    y = np.arange(t1.height)[::-1]
    a1.barh(y, t1["r_fwd"], color=BLUE, height=0.66, label="実測(前向き 1s)")
    a1.barh(y, t1["r_plc"], facecolor="none", edgecolor=INK, lw=0.9, height=0.66,
            label="帰無対照")
    a1.set_yticks(y, t1["col"].to_list(), fontsize=6.2)
    a1.axvline(0, color=INK, lw=0.8)
    a1.legend(fontsize=6.6, frameon=False)
    a1.set_title("(b) 1 秒先の上位 14 本", loc="left")

    # (c) horizon dependence of top-6
    a2 = fig.add_subplot(gs[0, 2])
    top6, seen = [], set()
    for c in p.filter(pl.col("h") == "1s").sort("ab", descending=True)["col"].to_list():
        b = c.split("__")[0]
        if b in seen:
            continue
        seen.add(b); top6.append(c)
        if len(top6) == 6:
            break
    cols = [BLUE, GREEN, PURPLE]
    for i, c in enumerate(top6):
        s = p.filter(pl.col("col") == c)
        v = [float(s.filter(pl.col("h") == h)["r_fwd"][0]) for h in FLAB]
        a2.plot(HSEC, v, marker="o", ms=3, lw=1.2, color=cols[i % 3],
                alpha=0.5 + 0.16 * (i // 3), label=c)
    a2.set_xscale("log")
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(HSEC, FLAB, fontsize=7)
    a2.legend(fontsize=6, frameon=False, ncol=2)
    a2.set_title("(c) 上位のホライズン依存", loc="left")

    # (d) predict vs contemporaneous (10s)
    a3 = fig.add_subplot(gs[1, 1])
    q10 = p.filter((pl.col("h") == "10s") & (pl.col("r_bwd").abs() < 0.99))
    a3.scatter(q10["r_bwd"], q10["r_fwd"], s=8, color=BLUE, alpha=0.5, lw=0)
    lim = float(np.nanmax(np.abs(np.concatenate(
        [q10["r_bwd"].to_numpy(), q10["r_fwd"].to_numpy()])))) * 1.05
    a3.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    a3.axhline(0, color=INK, lw=0.7); a3.axvline(0, color=INK, lw=0.7)
    nmore = int((q10["r_fwd"].abs() > q10["r_bwd"].abs()).sum())
    a3.set_xlabel("後ろ向き(同時性)"); a3.set_ylabel("前向き(予測)")
    a3.set_title(f"(d) 予測 vs 同時性 10s — 予測が強い {nmore}/{q10.height}", loc="left")

    # (e) OOS or daily sign
    a4 = fig.add_subplot(gs[1, 2])
    if daily is not None and daily.height:
        d = (daily.filter(pl.col("h") == "1s").with_columns(ab=pl.col("mean").abs())
             .sort("ab", descending=True).head(14))
        mu = d["mean"].to_numpy(); nd = d["nday"].to_numpy()
        agree = np.where(mu > 0, d["pos"].to_numpy(), nd - d["pos"].to_numpy())
        o = np.argsort(agree / nd)
        yy = np.arange(d.height)
        a4.barh(yy, (agree / nd * 100)[o], color=np.where(mu[o] > 0, BLUE, RED), height=0.7)
        a4.axvline(50, color=INK, lw=0.9, ls="--")
        a4.set_yticks(yy, [d["col"].to_list()[i] for i in o], fontsize=6)
        a4.set_xlim(0, 105)
        a4.set_xlabel("符号一致の日 %")
        a4.set_title(f"(e) 日ごとの符号 1s({int(nd[0])} 日中)", loc="left")
    else:
        q = p.filter(pl.col("h") == "1s")
        a4.scatter(q["r_1st"], q["r_2nd"], s=8, color=GREEN, alpha=0.5, lw=0)
        a4.set_title("(e) 前半 vs 後半 1s", loc="left")

    mxplc = float(p["r_plc"].abs().max())
    fig.suptitle(f"{a.coin} bbo+fills 特徴量の予測力(帰無対照の最大 |r|={mxplc:.4f})",
                 fontsize=12.5, y=0.995)
    fig.savefig(CH / f"{tag}_xfeat.png", facecolor=SURFACE, bbox_inches="tight")
    print("保存:", CH / f"{tag}_xfeat.png", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
