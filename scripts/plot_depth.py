"""BBO から 10 ティック奥までの P(Fill) x PnL_fill を描く。

    uv run python scripts/plot_depth.py --coin xyz:MU
出力: charts/<coin>_depth.png / data/depth_summary_<coin>.csv

【読み方】
奥へ置けば名目の優位は k ティック増えるが、**そこまで価格が来ないと約定しない**。
来たということは fair value 自体がそこまで落ちたということなので、
mid(約定直前) − 約定値段 で測った実際の優位は増えない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_maker import MAKER_FEE_BP  # noqa: E402
from plot_heat import DIV, draw  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KS = list(range(11))
HH = [(1.0, "1s", "#b5322f"), (10.0, "10s", "#1f5fa8"), (60.0, "60s", "#7a52c9")]
LAB = {"obi": "OBI", "ofi_10s": "OFI(10s)", "ai_net_10s": "攻撃的数量(10s)"}


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def prep(P):
    return P.with_columns(
        p_fill=pl.col("n_fill") / pl.col("n"),
        edge=pl.col("sum_edge") / pl.col("n_fill"),
        AS=pl.col("sum_as") / pl.col("n_fill"),
        pnl_fill=pl.col("sum_pnl") / pl.col("n_fill"),
        wait_s=pl.col("sum_wait") / pl.col("n_fill")).with_columns(
        EV=pl.col("n_fill") / pl.col("n") * pl.col("sum_pnl") / pl.col("n_fill"))


def series(P, sd, h, col):
    d = P.filter((pl.col("side") == sd) & (pl.col("h") == h)).sort("k")
    return d["k"].to_numpy(), d[col].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    P = prep(pl.read_csv(D / f"depth_pooled_{tag}.csv"))
    G = prep(pl.read_csv(D / f"depth_signal_{tag}.csv"))
    P.write_csv(D / f"depth_summary_{tag}.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{a.coin}  最良気配から k ティック奥に置いたら — "
                 f"P(Fill) × PnL_fill(待ち行列の先頭にいる前提)・97 日",
                 color=INK, fontsize=13, y=0.985)

    ax = axes[0, 0]
    for h, hn, c in HH:
        for sd, ls in ((1, "-"), (-1, "--")):
            k, v = series(P, sd, h, "p_fill")
            ax.plot(k, v * 100, color=c, lw=2.0, ls=ls, marker="o", ms=3.5,
                    label=f"h={hn} {'買' if sd == 1 else '売'}")
    style(ax, "最良気配から何ティック奥に置くか k", "P(Fill) %", logy=True)
    ax.set_xticks(KS)
    ax.set_title("(a) 奥へ置くほど約定しない", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower left", ncol=3)

    ax = axes[0, 1]
    for h, hn, c in HH:
        for sd, ls in ((1, "-"), (-1, "--")):
            k, v = series(P, sd, h, "pnl_fill")
            ax.plot(k, v, color=c, lw=2.0, ls=ls, marker="o", ms=3.5,
                    label=f"h={hn} {'買' if sd == 1 else '売'}")
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    style(ax, "k", "約定したときの損益 PnL_fill (bp)")
    ax.set_xticks(KS)
    ax.set_title("(b) 奥ほど 1 回あたりの損は大きい", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7, loc="lower right", ncol=3)

    ax = axes[0, 2]
    for h, hn, c in HH:
        for sd, ls in ((1, "-"), (-1, "--")):
            k, v = series(P, sd, h, "EV")
            ax.plot(k, v, color=c, lw=2.0, ls=ls, marker="o", ms=3.5,
                    label=f"h={hn} {'買' if sd == 1 else '売'}")
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    style(ax, "k", "EV = P(Fill) × PnL_fill (bp / 1 発注)")
    ax.set_xticks(KS)
    ax.set_title("(c) ★ どの深さでも負", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower right", ncol=3)

    ax = axes[1, 0]
    k, ed = series(P, 1, 10.0, "edge")
    _, As = series(P, 1, 10.0, "AS")
    w = 0.6
    ax.bar(k, ed, width=w, color="#1f5fa8", lw=0, label="実際の優位 mid(約定前) − 値段")
    ax.bar(k, np.full_like(ed, -MAKER_FEE_BP), width=w, bottom=ed,
           color="#9a9890", lw=0, label="手数料")
    ax.bar(k, As, width=w, bottom=ed - MAKER_FEE_BP, color="#b5322f", lw=0,
           label="AS = r(τ→τ+h)")
    ax.plot(k, ed - MAKER_FEE_BP + As, color=INK, lw=1.8, marker="o", ms=4,
            label="合計 PnL_fill")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(KS)
    style(ax, "k", "bp (h = 10 秒・買い指値)")
    ax.set_title("(d) 名目の k ティックは実際の優位にならない", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower left")

    ax = axes[1, 1]
    for h, hn, c in HH:
        k, v = series(P, 1, h, "wait_s")
        ax.plot(k, v, color=c, lw=2.0, marker="o", ms=3.5, label=f"h={hn} 買")
    style(ax, "k", "約定までの平均待ち時間 (秒)")
    ax.set_xticks(KS)
    ax.set_title("(e) 待ち時間は k でほとんど変わらない", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7, loc="upper left")

    ax = axes[1, 2]
    A = np.full((10, 11), np.nan)
    for r in G.filter((pl.col("feat") == "obi") & (pl.col("side") == 1)
                      & (pl.col("h") == 10.0)).iter_rows(named=True):
        A[r["s_dec"], r["k"]] = r["EV"]
    draw(ax, A, "(f) EV — OBI の信号十分位 × k (買い指値・h=10s)", DIV, True,
         "{:+.2f}")
    ax.set_xticks(range(11), [str(i) for i in KS], fontsize=7)

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    out = ROOT / "charts" / f"{tag}_depth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")
    pos = P.filter(pl.col("EV") > 0)
    print(f"EV>0 の (k, 側, h) は {pos.height} / {P.height}")


if __name__ == "__main__":
    main()
