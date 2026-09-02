"""口座ごとの約定 11 指標を 1 枚の図にする。

    uv run python scripts/plot_wallet_fill.py --coin xyz:MU
出力: charts/<coin>_wallet_fill.png

【配色】
指標は順序を持たないので色で区別せずパネルを分ける。口座の大小を表すときだけ
単一色相の濃淡を使い、符号のある量(adverse selection)は発散(青 = 有利、
赤 = 不利)にする。palette_check.py の実測は docstring 末尾に記す。

    #2a78d6 / #e34948 の発散ペア        PASS
    #2a78d6 / #0f8f63                   PASS(対比 4.30 / 3.99)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_wallet_fill import METRICS, JP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
UP, DOWN, WARN, GREEN, PURPLE = "#e34948", "#2a78d6", "#eb6834", "#0f8f63", "#4a3aa7"
DIVERGE = LinearSegmentedColormap.from_list(
    "dv", [DOWN, "#eef1f5", SURFACE, "#f7eeee", UP])
LOGM = ("maker_share", "avg_ttf_s", "fill_size", "queue_at_fill",
        "fill_to_cancel", "fill_rate", "fill_prob", "fill_to_order",
        "vol_exec_quoted")


def logfmt(ax, which="both"):
    r"""text.parse_math=False だと既定の指数表記($\mathdefault{...}$)が
    生の文字列で描画される。対数軸は必ずこの関数で目盛りを明示する。"""
    def f(v, _):
        if v <= 0:
            return ""
        if v >= 1:
            return f"{v:,.0f}" if v < 1e5 else f"1e{int(round(np.log10(v)))}"
        return f"{v:g}"
    for axis in ((ax.xaxis, ax.yaxis) if which == "both"
                 else (ax.xaxis,) if which == "x" else (ax.yaxis,)):
        axis.set_major_formatter(mpl.ticker.FuncFormatter(f))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d = pl.read_csv(ROOT / "data" / f"wfmetrics_{tag}.csv")
    C = pl.read_csv(ROOT / "data" / f"wfmetrics_corr_{tag}.csv")
    s = json.loads((ROOT / "data" / f"wfmetrics_summary_{tag}.json")
                   .read_text(encoding="utf-8"))
    R = np.column_stack([C[m].to_numpy() for m in METRICS])

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(17.6, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.28,
                          left=0.055, right=0.985, top=0.79, bottom=0.085)

    # ---- ① 11 指標の分布(口座ごとのばらつき) --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    show = ["fill_rate", "fill_prob", "fill_to_order", "vol_exec_quoted",
            "partial_freq"]
    for i, m in enumerate(show):
        v = d[m].to_numpy().astype(np.float64)
        v = v[np.isfinite(v) & (v > 0)]
        q = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([i, i], [q[0], q[4]], color=DOWN, lw=1.1, alpha=0.55)
        ax.plot([i, i], [q[1], q[3]], color=DOWN, lw=6.5)
        ax.plot([i], [q[2]], "o", ms=5.5, color=SURFACE, mec=DOWN, mew=1.6, zorder=3)
    ax.set_yscale("log")
    logfmt(ax, "y")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(["fill\nrate", "fill\nprobability", "fill-to-\norder",
                        "vol exec\n/ quoted", "partial-fill\nfrequency"],
                       fontsize=8.2, color=INK2)
    ax.set_ylabel("割合(対数軸)", color=INK2, fontsize=9.5)
    ax.set_title("① 約定の割合 — 定義で桁が変わる\n"
                 "箱 = 四分位、髭 = 5–95%、丸 = 中央値。口座ごとの値",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ② 4 つの約定割合の重なり ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    idx = [METRICS.index(m) for m in
           ("fill_rate", "fill_prob", "fill_to_order", "vol_exec_quoted")]
    lab = ["fill rate", "fill prob", "fill-to-order", "vol exec/quoted"]
    sub = R[np.ix_(idx, idx)]
    ax.imshow(sub, cmap=DIVERGE, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
              aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{sub[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color=INK if abs(sub[i, j]) < 0.55 else SURFACE)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8, color=INK2,
                                                rotation=20, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("② ★4 つは独立な指標ではない\n"
                 "口座間の順位相関。分子と分母の取り方が違うだけの近い量",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")

    # ---- ③ 出した量と約定量(メイカーの集中) ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    x = d["v_ord_c"].to_numpy(); y = d["v_fill_c"].to_numpy()
    ok = (x > 0) & (y > 0)
    ax.scatter(x[ok], y[ok], s=18, color=DOWN, alpha=0.55, edgecolors="none")
    lo = min(x[ok].min(), y[ok].min()); hi = max(x[ok].max(), y[ok].max())
    for r in (1.0, 0.1, 0.01, 0.001):
        ax.plot([lo, hi], [lo * r, hi * r], color=BASELINE, lw=0.8,
                ls=(0, (4, 3)), zorder=0)
        ax.annotate(f"{r:g}", (hi, hi * r), fontsize=7, color=MUTED,
                    ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    logfmt(ax)
    ax.set_xlabel("置いた数量(契約、対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("約定した数量(契約、対数)", color=INK2, fontsize=9.5)
    ax.set_title("③ 出した量と約定した量\n"
                 f"破線 = 約定率。上位 1 者が市場の "
                 f"{100*s['top1_maker_share']:.1f}%、上位 5 者で "
                 f"{100*s['top5_maker_share']:.1f}%",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    style(ax)

    # ---- ④ 約定までの時間と約定 1 件の大きさ ----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    x = d["avg_ttf_s"].to_numpy(); y = d["fill_size"].to_numpy()
    c = d["maker_share"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    sc = ax.scatter(x[ok], y[ok], s=14 + 260 * np.sqrt(np.clip(c[ok], 0, None)),
                    color=DOWN, alpha=0.5, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    logfmt(ax)
    ax.set_xlabel("average time to fill(秒、対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("fill size(契約、対数)", color=INK2, fontsize=9.5)
    ax.set_title("④ 待ち時間と約定の大きさ\n"
                 "丸の大きさ = maker execution share",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    style(ax)

    # ---- ⑤ 約定時のキュー位置 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    v = d["share_q0"].to_numpy().astype(np.float64) * 100
    v = v[np.isfinite(v)]
    ax.hist(v, bins=np.linspace(0, 100, 26), color=DOWN, alpha=0.9)
    ax.axvline(float(np.median(v)), color=WARN, lw=1.4, ls=(0, (3, 2)))
    ax.annotate(f"中央値 {np.median(v):.0f}%", (float(np.median(v)), 0),
                xytext=(6, 12), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set_xlabel("約定した瞬間に自分が先頭だった割合(%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("口座数", color=INK2, fontsize=9.5)
    ax.set_title("⑤ queue-position-at-fill\n"
                 "★ほとんどの約定は「前に誰もいない」状態で起きている",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑥ 約定後の逆選択 ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    y = np.arange(len(d))
    order = np.argsort(-d["maker_share"].to_numpy())[:20]
    adv = d["adverse_10s"].to_numpy()[order]
    ms = d["maker_share"].to_numpy()[order] * 100
    yy = np.arange(len(order))
    cols = [UP if v > 0 else DOWN for v in adv]
    ax.barh(yy, adv, color=cols, height=0.7)
    ax.axvline(0, color=BASELINE, lw=0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{m:.1f}%" for m in ms], fontsize=7, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("約定 10 秒後の不利方向の値動き(bp、正 = 不利)",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("メイカー約定シェアの大きい順", color=INK2, fontsize=9.5)
    ax.set_title("⑥ adverse selection after fill\n"
                 "赤 = 約定後に不利へ動いた(逆選択)、青 = 有利へ動いた",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.grid(axis="y", lw=0)
    style(ax)
    ax.grid(axis="y", lw=0)

    fig.suptitle(f"{a.coin} 口座ごとの約定のされ方 11 指標"
                 f"({d.height} 口座・98 日)",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.055, 0.922,
             f"母集団は板に留まる指値(Alo / Gtc、トリガー・テイカー・reduce_only を除く)。"
             f"置いた本数が {s['min_orders']:,} 本以上の口座だけを載せる"
             f"(注文の {100*s['share_orders_kept']:.1f}%、"
             f"メイカー約定量の {100*s['share_fillvol_kept']:.1f}% を覆う)。\n"
             "約定量は「イベントごとに報告された残量の減少分」で数える。"
             "filled イベントだけを数えると 3 割取りこぼす(検証は本文)。"
             "口座は内部連番で扱う。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses から板とキューを再構成 / "
             "合計値は node_fills で照合",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_wallet_fill.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
