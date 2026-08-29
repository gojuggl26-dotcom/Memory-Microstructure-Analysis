"""ウォレットがマーケットメイカーらしいかを 13 基準の採点表で図示する。

    A 採点表   … 5 者 × 13 基準。色は **5 者の中での順位**(絶対評価ではない)
    B 位置取り … 気配の距離と最良気配に居る割合。MM らしさは大きさでは決まらない
    C 日ごとの安定性 … 主要 4 基準の日次推移

★色は「5 者の中で最も MM らしい」を濃く塗っただけで、**絶対的な合格線ではない**。
生の値をセルに書いてあるのでそちらで判断すること。

配色は dataviz の参照パレット slot 1〜5(palette_check 済み)。
採点表は単一色相の順序ランプ。

    uv run python scripts/plot_wallet_mm.py --coin xyz:MU
出力: charts/<coin>_wallet_mm.png
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
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

# (列名, 表示名, 単位, 高いほど MM らしいか, 書式)
CRIT = [
    ("two_sided_ratio", "two-sided quoting", "", True, "{:.1%}"),
    ("continuous_ratio", "continuous quoting", "", True, "{:.1%}"),
    ("bbo_participation", "BBO participation", "", True, "{:.1%}"),
    ("symmetric_placement", "symmetric placement", "", True, "{:.3f}"),
    ("update_freq_per_h", "update frequency", "/h", True, "{:,.0f}"),
    ("cancel_rate", "cancellation rate", "", True, "{:.3f}"),
    ("median_lifetime_s", "median lifetime", "s", False, "{:.2f}"),
    ("replenish_1s_share", "replenishment (1s 以内)", "", True, "{:.1%}"),
    ("inv_skew_corr", "inventory-skew 相関", "", True, "{:+.3f}"),
    ("depth_contribution", "depth contribution", "", True, "{:.1%}"),
    ("market_coverage", "market coverage (±10bp)", "", True, "{:.1%}"),
    ("quote_width_cv", "quote width の変動係数", "", False, "{:.3f}"),
    ("quote_size_cv", "quote size の変動係数", "", False, "{:.3f}"),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_csv(ROOT / "data" / f"wallet_mm_daily_{tag}.csv")
    num = [c for c in D.columns if c not in ("user", "dt")]
    S = (D.group_by("user")
          .agg([pl.col(c).is_finite().sum().alias(f"_n_{c}") for c in num]
               + [pl.col(c).filter(pl.col(c).is_finite()).median().alias(c) for c in num]
               + [pl.len().alias("n_days")])
          .sort("depth_contribution", descending=True, nulls_last=True))
    S = S.sort("market_coverage", descending=True, nulls_last=True)
    S.select(["user", "n_days"] + num).write_csv(ROOT / "data" / f"wallet_mm_{tag}.csv")

    users = S["user"].to_list()
    short = [u[:6] + "…" + u[-4:] for u in users]
    nu = len(users)

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.2, 11.6), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.22,
                          left=0.19, right=0.975, top=0.855, bottom=0.065)

    # ---- A 採点表 ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.grid(False)
    for r, (col, name, unit, hi_good, fmt) in enumerate(CRIT):
        v = np.array([S.filter(pl.col("user") == u)[col][0] for u in users], dtype=float)
        ok = np.isfinite(v)
        rk = np.full(nu, np.nan)
        if ok.sum() >= 2:
            order = np.argsort(np.argsort(v[ok] if hi_good else -v[ok]))
            rk[ok] = order / max(ok.sum() - 1, 1)
        for c in range(nu):
            if not np.isfinite(v[c]):
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, facecolor=SURFACE,
                                           edgecolor=GRID, lw=1.0))
                ax.text(c, r, "—", ha="center", va="center", fontsize=9, color=MUTED)
                continue
            k = int(round(rk[c] * (len(RAMP) - 1)))
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, facecolor=RAMP[k],
                                       edgecolor=SURFACE, lw=1.6))
            ax.text(c, r, fmt.format(v[c]) + (f" {unit}" if unit else ""),
                    ha="center", va="center", fontsize=8.6,
                    color="#ffffff" if k >= 5 else INK)
    ax.set_xlim(-.5, nu - .5)
    ax.set_ylim(len(CRIT) - .5, -.5)
    ax.set_xticks(range(nu))
    ax.set_xticklabels(short, fontsize=8.6, color=MUTED)
    ax.set_yticks(range(len(CRIT)))
    ax.set_yticklabels([c[1] for c in CRIT], fontsize=9.0, color=INK2)
    ax.set_title("A  13 基準の採点表(色は 5 者の中での順位。濃いほど MM らしい)",
                 loc="left", color=INK, fontsize=12, pad=22, weight="bold")
    ax.text(0, 1.025, "★色は相対順位であって合格線ではない。判断はセルの生の値で行うこと。"
                      "— は算出できなかったセル(該当日が足りない)",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    # ---- B 位置取り --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    for i, u in enumerate(users):
        q = S.filter(pl.col("user") == u)
        x = float((q["bid_dist_bp"][0] + q["ask_dist_bp"][0]) / 2)
        y = float(q["bbo_participation"][0]) * 100
        n = float(q["market_coverage"][0]) if q["market_coverage"][0] is not None else 0.0
        ax.scatter([x], [y], s=60 + 900 * n, color=SLOT[i % 5], alpha=0.8,
                   edgecolors=SURFACE, linewidths=1.4, zorder=4)
        ax.annotate(short[i], (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8.4, color=SLOT[i % 5], weight="bold")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 25, 50, 100, 250])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("mid からの平均距離 [bp](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("最良気配に置いた注文の割合 [%]", color=INK2, fontsize=9.5)
    ax.set_title("B  位置取り — 近くに置くほど MM らしい", loc="left", color=INK,
                 fontsize=12, pad=22, weight="bold")
    ax.text(0, 1.03, "丸の大きさ = mid ±10bp に置いた割合(market coverage)",
            transform=ax.transAxes, color=INK2, fontsize=8.8)

    # ---- C 日ごとの安定性 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    days = sorted(D["dt"].unique().to_list())
    xi = {d: i for i, d in enumerate(days)}
    ends = []
    for i, (col, nm) in enumerate((("two_sided_ratio", "two-sided"),
                                   ("continuous_ratio", "continuous"),
                                   ("market_coverage", "coverage ±10bp"),
                                   ("bbo_participation", "BBO participation"))):
        q = D.filter(pl.col("user") == users[0]).sort("dt")
        x = np.array([xi[d] for d in q["dt"].to_list()])
        y = q[col].to_numpy().astype(float) * 100
        ax.plot(x, y, "-o", color=SLOT[i], lw=1.6, ms=2.6, zorder=4)
        ends.append((y[-1], nm, SLOT[i]))
    lo, hi = ax.get_ylim()
    ends.sort(key=lambda e: e[0])
    fr = [(e[0] - lo) / (hi - lo) for e in ends]
    for j in range(1, len(fr)):
        fr[j] = max(fr[j], fr[j - 1] + 0.07)
    for (y, nm, c), f in zip(ends, fr):
        ax.annotate(nm, (len(days) - 1, lo + f * (hi - lo)), textcoords="offset points",
                    xytext=(7, 0), fontsize=8.3, color=c, weight="bold", va="center",
                    annotation_clip=False)
    step = max(len(days) // 4, 1)
    ax.set_xticks(list(range(0, len(days), step)))
    ax.set_xticklabels([days[i][5:] for i in range(0, len(days), step)],
                       fontsize=8.5, color=MUTED)
    ax.set_xlabel("日", color=INK2, fontsize=9.5)
    ax.set_ylabel("割合 [%]", color=INK2, fontsize=9.5)
    ax.set_title(f"C  {short[0]} の日ごとの在席", loc="left", color=INK,
                 fontsize=12, pad=22, weight="bold")
    ax.text(0, 1.03, "採点表で最も MM らしい 1 者の内訳", transform=ax.transAxes,
            color=INK2, fontsize=8.8)

    nd = int(D["dt"].n_unique())
    fig.suptitle(f"{a.coin} このウォレットはマーケットメイカーか — 13 基準",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.19, 0.938,
             f"mid ±10bp の板厚に占める割合が大きい順に選んだ 5 者を、{nd} 日の中央値で比べた。"
             "対象の選び方は結果を見る前に決めてある。\n"
             "★大きさ(板に出している金額)と MM らしさは一致しない。"
             "最大の出し手が最も MM らしいとは限らないことが下段 B に出ている。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) l1 注文イベント + l2/bbo + node_fills",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_wallet_mm.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
