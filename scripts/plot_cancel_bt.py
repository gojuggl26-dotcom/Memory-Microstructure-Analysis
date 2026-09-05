"""遅延を織り込んだ取り消し backtest の結果を描く。

    uv run python scripts/plot_cancel_bt.py --coin xyz:MU
出力: charts/<coin>_cancel_bt.png / data/cancel_bt_summary_<coin>.csv

【読み方】
効果は「無作為に同じ回数だけ取り消した場合」との差で見る。
そして遅延 L を増やすと効果がどれだけ削られるかを見る。
1 発注あたりの期待値は、取り消しを増やすほどゼロに近づく
(取引しなければ損もしない)。**黒字になるわけではない**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cancel_bt import LATENCY_MS, TRIG_RATE  # noqa: E402
from plot_burst import nwse  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LC = {0: "#0f3a6b", 65: "#1f5fa8", 130: "#7fa8dd", 200: "#e08b7f",
      130.0: "#7fa8dd", 200.0: "#e08b7f", 300: "#b5322f", 300.0: "#b5322f",
      0.0: "#0f3a6b", 65.0: "#1f5fa8"}


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def agg(B):
    return (B.group_by("rule", "trig_rate", "lat_ms")
            .agg(**{c: pl.col(c).sum() for c in
                    ("n", "n_cancel", "n_fill", "sum_mk", "sum_edge", "sum_pnl")})
            .with_columns(fill=pl.col("n_fill") / pl.col("n"),
                          canc=pl.col("n_cancel") / pl.col("n"),
                          markout=pl.col("sum_mk") / pl.col("n_fill"),
                          edge=pl.col("sum_edge") / pl.col("n_fill"),
                          pnl=pl.col("sum_pnl") / pl.col("n_fill"),
                          ev=pl.col("sum_pnl") / pl.col("n"))
            .sort("rule", "trig_rate", "lat_ms"))


def daily_se(B, rule, r, L, col, den):
    x = B.filter((pl.col("rule") == rule) & (pl.col("trig_rate") == r)
                 & (pl.col("lat_ms") == L)).group_by("dt").agg(
        v=pl.col(col).sum() / pl.col(den).sum()).sort("dt")
    m, se, _, _ = nwse(x["v"].to_numpy())
    return m, se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    B = pl.read_parquet(D / f"cancel_bt_{tag}.parquet").filter(
        pl.col("split") == "評価")
    G = agg(B)
    G.write_csv(D / f"cancel_bt_summary_{tag}.csv")
    b0 = G.filter(pl.col("trig_rate") == 0.0).row(0, named=True)
    print(f"基準: 約定率 {100*b0['fill']:.2f}%  markout {b0['markout']:+.3f}  "
          f"1 約定 {b0['pnl']:+.3f}  1 発注 {b0['ev']:+.4f} bp")

    p2 = D / f"cancel_bt_{tag}_hq2.parquet"
    G2 = agg(pl.read_parquet(p2).filter(pl.col("split") == "評価")) \
        if p2.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{a.coin}  遅延を織り込んだ取り消しの backtest — "
                 f"評価期間 39 日・クオート寿命 10 秒", color=INK, fontsize=13,
                 y=0.985)
    rr = [r for r in TRIG_RATE if r > 0]
    xr = np.array(rr) * 100

    ax = axes[0, 0]
    for L in LATENCY_MS:
        v = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                            & (pl.col("lat_ms") == L))["markout"][0]) for r in rr]
        ax.plot(xr, v, color=LC[L], lw=2.2, marker="o", ms=4.5,
                label=f"L={L}ms" + (" (遅延ゼロ)" if L == 0 else ""))
    v = [float(G.filter((pl.col("rule") == "無作為") & (pl.col("trig_rate") == r)
                        & (pl.col("lat_ms") == 130))["markout"][0]) for r in rr]
    ax.plot(xr, v, color=MUTED, lw=1.8, ls="--", marker="s", ms=4,
            label="無作為 (L=130ms)")
    ax.axhline(b0["markout"], color=INK, lw=1.4, zorder=3)
    ax.text(xr[0], b0["markout"] + 0.01, " 取り消しなし", color=INK, fontsize=8.5)
    style(ax, "発火率 (板更新の何 % で取り消しを出すか)", "残った約定の markout (bp)")
    ax.set_title("(a) ★ 遅延が効果を削る", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower left")

    ax = axes[0, 1]
    for r in (0.05, 0.10, 0.20, 0.40):
        v = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                            & (pl.col("lat_ms") == L))["markout"][0])
             - b0["markout"] for L in LATENCY_MS]
        ax.plot(LATENCY_MS, v, lw=2.0, marker="o", ms=4.5,
                label=f"発火率 {int(r*100)}%")
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    ax.axvspan(65, 200, color="#b5322f", alpha=0.08, lw=0, zorder=0)
    ax.text(130, 0.005, "実測の現実的な範囲\n(1〜3 ブロック)", color="#b5322f",
            fontsize=8, ha="center", va="bottom")
    style(ax, "取り消しが有効になるまでの遅延 L (ms)", "markout の改善 (bp)")
    ax.set_title("(b) ★ L=300ms で改善はほぼ消える", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="upper right")

    ax = axes[0, 2]
    for L in (0, 130, 300):
        vs = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                             & (pl.col("lat_ms") == L))["markout"][0]) for r in rr]
        vr = [float(G.filter((pl.col("rule") == "無作為") & (pl.col("trig_rate") == r)
                             & (pl.col("lat_ms") == L))["markout"][0]) for r in rr]
        ax.plot(xr, np.array(vs) - np.array(vr), color=LC[L], lw=2.2, marker="o",
                ms=4.5, label=f"L={L}ms")
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    style(ax, "発火率 (%)", "スコア − 無作為 (bp)")
    ax.set_title("(c) 同じ回数を無作為に切った場合との差", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="lower right")

    ax = axes[1, 0]
    for L in (0, 130, 300):
        v = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                            & (pl.col("lat_ms") == L))["fill"][0]) * 100
             for r in rr]
        ax.plot(xr, v, color=LC[L], lw=2.2, marker="o", ms=4.5, label=f"L={L}ms")
    ax.axhline(b0["fill"] * 100, color=INK, lw=1.4, zorder=3)
    ax.text(xr[0], b0["fill"] * 100 + 0.5, " 取り消しなし", color=INK, fontsize=8.5)
    style(ax, "発火率 (%)", "約定率 (%)")
    ax.set_title("(d) 取り消すほど約定しなくなる", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="upper right")

    ax = axes[1, 1]
    for L in (0, 130, 300):
        v = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                            & (pl.col("lat_ms") == L))["pnl"][0]) for r in rr]
        ax.plot(xr, v, color=LC[L], lw=2.2, marker="o", ms=4.5, label=f"L={L}ms")
    ax.axhline(b0["pnl"], color=INK, lw=1.4, zorder=3)
    ax.axhline(0, color="#1f8a5e", lw=1.2, ls="--")
    ax.text(xr[0], 0.05, " ここに届けば黒字", color="#1f8a5e", fontsize=8.5)
    style(ax, "発火率 (%)", "1 約定あたりの損益 (bp)")
    ax.set_title("(e) 1 約定あたりは改善するが黒字には届かない", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="lower right")

    ax = axes[1, 2]
    for L in (0, 130, 300):
        v = [float(G.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                            & (pl.col("lat_ms") == L))["ev"][0]) for r in rr]
        ax.plot(xr, v, color=LC[L], lw=2.2, marker="o", ms=4.5, label=f"L={L}ms")
    if G2 is not None:
        b02 = G2.filter(pl.col("trig_rate") == 0.0).row(0, named=True)
        v = [float(G2.filter((pl.col("rule") == "スコア") & (pl.col("trig_rate") == r)
                             & (pl.col("lat_ms") == 130))["ev"][0]) for r in rr]
        ax.plot(xr, v, color="#7a52c9", lw=1.8, ls="--", marker="^", ms=4.5,
                label="寿命 2 秒・L=130ms")
        ax.axhline(b02["ev"], color="#7a52c9", lw=1.0, ls=":")
    ax.axhline(b0["ev"], color=INK, lw=1.4, zorder=3)
    ax.axhline(0, color="#1f8a5e", lw=1.4, ls="--")
    ax.text(xr[-1], 0.01, "何もしない = 0 ", color="#1f8a5e", fontsize=8.5,
            ha="right")
    style(ax, "発火率 (%)", "1 発注あたりの期待値 (bp)")
    ax.set_title("(f) ★ ゼロに近づくだけで、超えない", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7, loc="lower right")

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    out = ROOT / "charts" / f"{tag}_cancel_bt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")

    print("\n=== 評価期間・日次 Newey-West つき (発火率 20%)")
    for L in LATENCY_MS:
        m1, s1 = daily_se(B, "スコア", 0.20, L, "sum_mk", "n_fill")
        m2, s2 = daily_se(B, "無作為", 0.20, L, "sum_mk", "n_fill")
        m0, s0 = daily_se(B, "スコア", 0.0, L, "sum_mk", "n_fill")
        print(f"  L={L:>3}ms  スコア {m1:+.3f}±{s1:.3f}  無作為 {m2:+.3f}±{s2:.3f}"
              f"  基準 {m0:+.3f}  スコア−基準 {m1-m0:+.3f}"
              f"  スコア−無作為 {m1-m2:+.3f}")


if __name__ == "__main__":
    main()
