"""反実仮想のメイカー注文 — 約定率と、その十分位ごとの将来リターンを描く。

    uv run python scripts/plot_maker.py --coin xyz:MU
出力: charts/<coin>_maker_fill.png   約定率と約定時の損益 (対照つき)
      charts/<coin>_maker_decile.png 待ち行列の十分位ごとの約定率・リターン・損益
      data/maker_summary_<coin>.csv  6 通り x 8 ホライズンの要約
      data/maker_decile_<coin>.csv   6 通り x 8 ホライズン x 10 分位

【読み方】
約定したときの損益は「受け取る半スプレッド」+「約定後の mid の動き」に分かれる。
受動的に置いた注文は**動く方向に食われる**ので後者は負になる。これが逆選択で、
信号の向きに置くとむしろ悪化する。対照 (信号のない Q3) と必ず並べて読むこと。
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
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ["obi", "ofi_10s", "ai_net_10s"]
FC = {"obi": "#b5322f", "ofi_10s": "#1f5fa8", "ai_net_10s": "#7a52c9"}
LAB = {"obi": "OBI", "ofi_10s": "OFI(10s)", "ai_net_10s": "攻撃的数量(10s)"}
CASES = [(1, 4, "Q5 買指値", "-"), (-1, 0, "Q1 売指値", "--")]
DC = ["#1f5fa8", "#7a52c9", "#b5322f"]


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def hx(ax):
    ax.set_xscale("log")
    ax.set_xticks(HS)
    ax.set_xticklabels(["100ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"])
    ax.minorticks_off()


def agg(M, keys):
    cols = ("n", "n_fill", "n_touch", "sum_pnl", "sum_pnl_touch", "sum_ret",
            "sum_ret_fill", "sum_retc", "sum_retc_fill", "sum_mmend_fill",
            "sum_spr", "sum_tau", "sum_q")
    return (M.group_by(*keys).agg(**{c: pl.col(c).sum() for c in cols})
            .with_columns(
                fill_rate=pl.col("n_fill") / pl.col("n"),
                touch_rate=pl.col("n_touch") / pl.col("n"),
                pnl=pl.col("sum_pnl") / pl.col("n_fill") - MAKER_FEE_BP,
                pnl_touch=pl.col("sum_pnl_touch") / pl.col("n_touch") - MAKER_FEE_BP,
                ret=pl.col("sum_ret") / pl.col("n"),
                ret_fill=pl.col("sum_ret_fill") / pl.col("n_fill"),
                retc=pl.col("sum_retc") / pl.col("n"),
                spread=pl.col("sum_spr") / pl.col("n"),
                tau=pl.col("sum_tau") / pl.col("n"),
                q_ahead=pl.col("sum_q") / pl.col("n"))
            .with_columns(
                per_order=pl.col("fill_rate")
                * (pl.col("sum_pnl") / pl.col("n_fill") - MAKER_FEE_BP),
                adverse=pl.col("sum_pnl") / pl.col("n_fill")
                - pl.col("sum_spr") / pl.col("n") / 2.0,
                # microprice 建ての損益 (板が傾いた状態での mid 評価の偏りを確認)
                pnl_micro=pl.col("sum_pnl") / pl.col("n_fill") - MAKER_FEE_BP
                + pl.col("sum_mmend_fill") / pl.col("n_fill"))
            .sort(*keys))


def pick(G, f, q, sd):
    return G.filter((pl.col("feat") == f) & (pl.col("q") == q)
                    & (pl.col("side") == sd)).sort("h")


def fig_fill(G, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{tag}  信号を検出した瞬間に BBO へメイカー注文を置いたら — "
                 f"約定率と逆選択 (97 日)", color=INK, fontsize=13, y=0.985)

    ax = axes[0, 0]
    for f in FEATS:
        for sd, q, nm, ls in CASES:
            r = pick(G, f, q, sd)
            ax.plot(HS, r["fill_rate"].to_numpy() * 100, color=FC[f], lw=2.0,
                    ls=ls, marker="o", ms=3.5, label=f"{LAB[f]} {nm}")
    hx(ax)
    style(ax, "置いてからの経過 h", "約定率 (%)", logy=True)
    ax.set_title("(a) 6 通りの約定率", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="upper left", ncol=2)

    ax = axes[0, 1]
    for f in FEATS:
        for sd, q, nm, ls in CASES:
            a = pick(G, f, q, sd)["fill_rate"].to_numpy()
            b = pick(G, f, 2, sd)["fill_rate"].to_numpy()
            ax.plot(HS, a / b, color=FC[f], lw=2.0, ls=ls, marker="o", ms=3.5,
                    label=f"{LAB[f]} {nm}")
    ax.axhline(1.0, color=INK, lw=1.4, zorder=3)
    ax.text(HS[0], 1.03, " 対照 (同じ特徴量の Q3) と同じ", color=INK, fontsize=8)
    hx(ax)
    style(ax, "置いてからの経過 h", "約定率 ÷ 対照 Q3 の約定率")
    ax.set_title("(b) OBI だけ対照より約定しにくい — 板が厚い側に置くから",
                 color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="upper right", ncol=2)

    ax = axes[0, 2]
    for f in FEATS:
        for sd, q, nm, ls in CASES:
            ax.plot(HS, pick(G, f, q, sd)["pnl"].to_numpy(), color=FC[f],
                    lw=2.0, ls=ls, marker="o", ms=3.5, label=f"{LAB[f]} {nm}")
            ax.plot(HS, pick(G, f, 2, sd)["pnl"].to_numpy(), color=FC[f],
                    lw=1.0, ls=":", alpha=0.7)
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    sp = float(G["spread"].mean())
    ax.axhline(sp / 2, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(HS[0], sp / 2 + 0.1, f" 受け取る半スプレッド {sp/2:.2f} bp",
            color=MUTED, fontsize=8)
    hx(ax)
    style(ax, "置いてからの経過 h", "約定したときの損益 (bp)")
    ax.set_title("(c) 全ホライズンで負。点線 = 対照 Q3", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7, loc="lower left", ncol=2)

    ax = axes[1, 0]
    w = 0.13
    xs = np.arange(len(HS))
    for i, f in enumerate(FEATS):
        for j, (sd, q, nm, ls) in enumerate(CASES):
            r = pick(G, f, q, sd)
            ax.bar(xs + (i * 2 + j - 2.5) * w, r["adverse"].to_numpy(), width=w,
                   color=FC[f], alpha=1.0 if j == 0 else 0.55, lw=0,
                   label=f"{LAB[f]} {nm}")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(xs, ["100ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"],
                  fontsize=8.5)
    style(ax, "置いてからの経過 h", "約定後の mid の動き (bp)")
    ax.set_title("(d) 逆選択 — 約定した後に価格は必ず不利へ動く", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower left", ncol=2)

    ax = axes[1, 1]
    for f in FEATS:
        for sd, q, nm, ls in CASES:
            ax.plot(HS, pick(G, f, q, sd)["per_order"].to_numpy(), color=FC[f],
                    lw=2.0, ls=ls, marker="o", ms=3.5, label=f"{LAB[f]} {nm}")
            ax.plot(HS, pick(G, f, 2, sd)["per_order"].to_numpy(), color=FC[f],
                    lw=1.0, ls=":", alpha=0.7)
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    hx(ax)
    style(ax, "置いてからの経過 h", "1 発注あたりの期待損益 (bp)")
    ax.set_title("(e) 約定率 × 損益。点線 = 対照 Q3", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7, loc="lower left", ncol=2)

    ax = axes[1, 2]
    for f in FEATS:
        r = pick(G, f, 4, 1)
        ax.fill_between(HS, r["fill_rate"].to_numpy() * 100,
                        r["touch_rate"].to_numpy() * 100, color=FC[f],
                        alpha=0.18, lw=0)
        ax.plot(HS, r["fill_rate"].to_numpy() * 100, color=FC[f], lw=2.0,
                label=f"{LAB[f]} 下限 (待ち行列を抜く)")
        ax.plot(HS, r["touch_rate"].to_numpy() * 100, color=FC[f], lw=1.2,
                ls="--", label=f"{LAB[f]} 上限 (先頭にいた場合)")
    hx(ax)
    style(ax, "置いてからの経過 h", "約定率 (%)", logy=True)
    ax.set_title("(f) 真の約定率はこの帯の中 (Q5 買指値)", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=6.5, loc="upper left")

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def fig_decile(Dg, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{tag}  待ち行列がはける推定時間の十分位 — "
                 f"約定率で分けると将来リターンはどう変わるか", color=INK,
                 fontsize=13, y=0.985)
    xs = np.arange(1, 11)
    HH = [(1.0, "1 秒"), (10.0, "10 秒")]

    for col, f in enumerate(FEATS):
        ax = axes[0, col]
        for j, (h, hn) in enumerate(HH):
            for sd, q, nm, ls in CASES:
                r = Dg.filter((pl.col("feat") == f) & (pl.col("q") == q)
                              & (pl.col("side") == sd)
                              & (pl.col("h") == h)).sort("dec")
                ax.plot(xs, r["fill_rate"].to_numpy() * 100, color=DC[j],
                        lw=2.0, ls=ls, marker="o", ms=3.5,
                        label=f"h={hn} {nm}")
        ax.set_xticks(xs)
        style(ax, "十分位 (D1 = はけるのが速い)", "約定率 (%)", logy=True)
        ax.set_title(f"{LAB[f]} — 約定率は十分位で単調", color=INK,
                     fontsize=10, loc="left")
        legend(ax, fs=7, loc="lower left", ncol=2)

    for col, f in enumerate(FEATS):
        ax = axes[1, col]
        for j, (h, hn) in enumerate(HH):
            for sd, q, nm, ls in CASES:
                r = Dg.filter((pl.col("feat") == f) & (pl.col("q") == q)
                              & (pl.col("side") == sd)
                              & (pl.col("h") == h)).sort("dec")
                ax.plot(xs, r["ret"].to_numpy(), color=DC[j], lw=2.0, ls=ls,
                        marker="o", ms=3.5, label=f"h={hn} {nm} mid")
                ax.plot(xs, r["retc"].to_numpy(), color=DC[j], lw=1.0, ls=":",
                        alpha=0.8)
        ax.axhline(0, color=INK, lw=1.2, zorder=3)
        ax.set_xticks(xs)
        style(ax, "十分位 (D1 = はけるのが速い)", "将来リターン (信号方向・bp)")
        ax.set_title(f"{LAB[f]} — 点線 = microprice で測った同じ量", color=INK,
                     fontsize=10, loc="left")
        legend(ax, fs=7, loc="lower right", ncol=2)

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    M = pl.read_parquet(ROOT / "data" / f"maker_cells_{tag}.parquet")
    G = agg(M, ("feat", "q", "side", "h"))
    Dg = agg(M, ("feat", "q", "side", "h", "dec"))
    G.write_csv(ROOT / "data" / f"maker_summary_{tag}.csv")
    Dg.filter(pl.col("q") != 2).write_csv(ROOT / "data" / f"maker_decile_{tag}.csv")
    print(f"{a.coin}: 発注機会 {int(M.filter(pl.col('h') == 1.0)['n'].sum()):,}"
          f" (h=1s, 全五分位・両側)")
    ch = ROOT / "charts"
    fig_fill(G, a.coin, ch / f"{tag}_maker_fill.png")
    fig_decile(Dg, a.coin, ch / f"{tag}_maker_decile.png")


if __name__ == "__main__":
    main()
