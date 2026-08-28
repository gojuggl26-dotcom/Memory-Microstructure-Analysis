"""churn 7 種と将来 log リターンの回帰を、予測ホライズンごとに描く。

上 7 枚が特徴量ごとの相関のホライズン依存、下 2 枚が符号つきリターンに対する
相関の一覧(前向き / 後ろ向き)。

【1 枚あたり 3 本の線の意味】
    青 実線   前向き(予測)× その特徴量に合う目的変数
    赤 破線   後ろ向き(過去。帰無対照)× 同じ目的変数
    紫 点線   前向き × もう一方の目的変数

「合う目的変数」は、符号を持つ churn_imb だけ **符号つきリターン**、
残りの 6 つは大きさなので **|リターン|** とする。大きさの特徴量が
符号つきリターンを当てられるはずがない、というのが紫の線の役割。

相関 r で描く(β は特徴量ごとに単位が違って比較できない)。標本は
**重ならない部分標本**(sample = sub)。重なる窓の t 値は水増しされるため。

配色は scripts/palette_check.py で検証済み(ok=true、CVD ΔE 最小 20.0、
通常視 25.5、対比 4.30 / 3.85 / 5.44)。

    uv run python scripts/plot_churn.py --coin xyz:MU
出力: charts/<coin>_churn_ols.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
FWD, PAST, ALT = "#2a78d6", "#e34948", "#7a4fb5"

HORS = ["100ms", "200ms", "500ms", "1s", "2s", "5s", "10s", "30s", "1m", "2m", "5m"]
SEC = [0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
FEATS = ["churn_total", "churn_bid", "churn_ask", "churn_top", "churn_deep",
         "churn_imb", "churn_cnt"]
JP = {"churn_total": "total churn\n板の入れ替わり全体",
      "churn_bid": "bid churn\n買い側",
      "churn_ask": "ask churn\n売り側",
      "churn_top": "top-of-book churn\n最良気配以上",
      "churn_deep": "deep-book churn\nそれより外側",
      "churn_imb": "churn imbalance\n(買い−売り)/(買い+売り)",
      "churn_cnt": "order-count churn\n量ではなく本数"}
# 特徴量に「合う」目的変数
NAT = {f: ("符号つき" if f == "churn_imb" else "絶対値") for f in FEATS}
DIV = LinearSegmentedColormap.from_list("d", [PAST, "#f4f2ee", FWD])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def series(R, feat, lay, xform, ykind, direc, col="r"):
    """(ホライズン順に並べた値)。欠けたホライズンは nan。"""
    t = R.filter((pl.col("feat") == f"{feat}|{lay}") & (pl.col("x_form") == xform)
                 & (pl.col("y_kind") == ykind)
                 & (pl.col("hor").str.starts_with(direc)))
    d = {r[0].split("|")[1]: r[1] for r in t.select("hor", col).rows()}
    return np.array([d.get(h, np.nan) for h in HORS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--layer", default="全窓")
    ap.add_argument("--xform", default="生")
    ap.add_argument("--day", default="全日")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    R = (pl.read_parquet(ROOT / "data" / f"churn_ols_{tag}.parquet")
         .filter((pl.col("sample") == "sub") & (pl.col("day_type") == a.day)))

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.4, 12.8), dpi=160)
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.26,
                          left=0.062, right=0.978, top=0.772, bottom=0.062)

    # ---- 7 枚: 特徴量ごとの相関のホライズン依存 ------------------------------
    lo, hi = 0.0, 0.0
    cache = {}
    for f in FEATS:
        nat, alt = NAT[f], ("絶対値" if NAT[f] == "符号つき" else "符号つき")
        cache[f] = (series(R, f, a.layer, a.xform, nat, "前向き"),
                    series(R, f, a.layer, a.xform, nat, "後ろ向き"),
                    series(R, f, a.layer, a.xform, alt, "前向き"))
        for v in cache[f]:
            if np.isfinite(v).any():
                lo, hi = min(lo, np.nanmin(v)), max(hi, np.nanmax(v))
    pad = (hi - lo) * 0.10
    lo, hi = lo - pad, hi + pad

    for i, f in enumerate(FEATS):
        ax = fig.add_subplot(gs[i // 3, i % 3]); style(ax)
        nat, alt = NAT[f], ("絶対値" if NAT[f] == "符号つき" else "符号つき")
        fw, pa, al = cache[f]
        ax.axhline(0, color=BASELINE, lw=1.0, zorder=2)
        ax.plot(SEC, al, ":", color=ALT, lw=1.6, marker="^", ms=3.6, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.7)
        ax.plot(SEC, pa, "--", color=PAST, lw=1.8, marker="s", ms=4.0, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.plot(SEC, fw, "-", color=FWD, lw=2.4, marker="o", ms=4.4, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=0.9)
        ax.set_xscale("log"); ax.set_ylim(lo, hi)
        ax.set_xticks(SEC); ax.set_xticklabels(HORS, rotation=55, ha="right", fontsize=7.5)
        ax.set_title(f"({i+1}) {JP[f]}", loc="left", color=INK, fontsize=10.5,
                     pad=7, weight="bold")
        ax.text(0.98, 0.955, f"合う目的変数: {nat}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK2)
        if i % 3 == 0:
            ax.set_ylabel("相関 r", color=INK2, fontsize=9.5)
        if i >= 4:
            ax.set_xlabel("予測ホライズン", color=INK2, fontsize=9.5)

    # ---- 8, 9 枚目: 符号つきリターンに対する r の一覧 -------------------------
    M = {d: np.vstack([series(R, f, a.layer, a.xform, "符号つき", d) for f in FEATS])
         for d in ("前向き", "後ろ向き")}
    vmax = float(np.nanmax(np.abs(np.concatenate([M["前向き"], M["後ろ向き"]]))))
    for k, (d, ttl) in enumerate((("前向き", "(8) 符号つきリターンとの相関 — 前向き(予測)"),
                                  ("後ろ向き", "(9) 同じもの — 後ろ向き(過去。帰無対照)"))):
        ax = fig.add_subplot(gs[2, 1 + k]); style(ax)
        ax.grid(False)
        im = ax.imshow(M[d], cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(HORS)))
        ax.set_xticklabels(HORS, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(FEATS)))
        ax.set_yticklabels([f.replace("churn_", "") for f in FEATS], fontsize=8.5)
        ax.set_title(ttl, loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
        for r in range(len(FEATS)):
            for c in range(len(HORS)):
                v = M[d][r, c]
                if not np.isfinite(v):
                    continue
                ax.text(c, r, f"{v:+.2f}".replace("+0.", ".").replace("-0.", "-."),
                        ha="center", va="center", fontsize=6.4,
                        color=SURFACE if abs(v) > vmax * 0.55 else INK2)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5, length=2)

    fig.legend(handles=[
        Line2D([], [], color=FWD, lw=2.4, marker="o", ms=5, label="前向き(予測)× 合う目的変数"),
        Line2D([], [], color=PAST, lw=1.8, ls="--", marker="s", ms=4.6,
               label="後ろ向き(過去。帰無対照)× 同じ目的変数"),
        Line2D([], [], color=ALT, lw=1.6, ls=":", marker="^", ms=4.6,
               label="前向き × もう一方の目的変数")],
        loc="upper center", bbox_to_anchor=(0.52, 0.888), frameon=False,
        fontsize=9.5, labelcolor=INK2, ncol=3, handlelength=2.6,
        columnspacing=2.4)

    # 前向きが後ろ向きより小さいセルを数える(決め打ちで書かない)
    nlt = ntot_ = 0
    for f in FEATS:
        fw, pa, _ = cache[f]
        m = np.isfinite(fw) & np.isfinite(pa)
        nlt += int((np.abs(fw[m]) < np.abs(pa[m])).sum()); ntot_ += int(m.sum())
    fig.text(0.062, 0.866,
             f"読み取りどころ: 前向き(青)が後ろ向き(赤)を下回るのは "
             f"{ntot_} セル中 {nlt} セル。"
             "churn は将来のリターンより過去のリターンと強く結びついている。" + "\n"
             "紫(向きの予測)がどの枚でもほぼ 0 なのは、"
             "大きさの特徴量が値動きの向きを含まないため。",
             fontsize=10.5, color=INK, va="top", weight="bold", linespacing=1.7)

    fig.suptitle(f"{a.coin} 板の入れ替わり(churn)7 種と将来 log リターンの OLS",
                 fontsize=13.5, y=0.978, color=INK, weight="bold")
    fig.text(0.062, 0.948,
             "x = 100ms 窓の churn(窓の終わり T で確定)、"
             "y = log mid(T+h) − log mid(T)。前向きは予測、後ろ向きは "
             "y = log mid(T) − log mid(T−h) で同時点・過去の関係。\n"
             "大きさの特徴量(6 つ)は |リターン|、符号を持つ churn imbalance は"
             "符号つきリターンが「合う目的変数」。"
             f"標本は重ならない部分標本 / {a.day} / {a.layer} / x は{a.xform}。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.5, 0.006,
             "出所: Hyperliquid L4 (Artemis) L1 注文イベント + l2/bbo / "
             "窓 2026-05-04〜08-09(98 日)",
             color=MUTED, fontsize=8, ha="center")
    out = ROOT / "charts" / f"{tag}_churn_ols.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
