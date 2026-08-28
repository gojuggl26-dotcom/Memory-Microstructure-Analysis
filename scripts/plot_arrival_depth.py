"""板の深さごとの注文到着率を、時間帯別・買い売り別に描く。

上段は時間帯ごとの **合計** 到着率(件/秒)。大きさの比較はここで読む。
下段の 6 枚は **深さ帯ごとの形**。時間帯によって水準が 20 倍以上違うので、
形を見比べられるように **共通の対数目盛** にしてある(線形だと静かな帯が潰れる)。

配色は買い=青 / 売り=赤。scripts/palette_check.py で検証済み:
CVD ΔE 21.6(目標 8)、通常視 ΔE 32.3(下限 15)、対比 4.30 / 3.85。

    uv run python scripts/plot_arrival_depth.py --coin xyz:MU
出力: charts/<coin>_arrival_depth.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BID, ASK = "#2a78d6", "#e34948"

MAIN = ["開場前 1h", "開場後 1h", "昼 12 時 ET", "閉場前 1h", "夜 12 時 ET",
        "閉場日 昼 12 時 ET"]
SUB = {"開場前 1h": "12:30–13:30 UTC / 立会日",
       "開場後 1h": "13:30–14:30 UTC / 立会日",
       "昼 12 時 ET": "16:00–17:00 UTC / 立会日",
       "閉場前 1h": "19:00–20:00 UTC / 立会日",
       "夜 12 時 ET": "04:00–05:00 UTC / 立会日",
       "閉場日 昼 12 時 ET": "16:00–17:00 UTC / 閉場日"}
BANDS = ["mid より内", "0–1", "1–2", "2–5", "5–10", "10–25", "25–50", "50–100", "100+"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    S = pl.read_csv(ROOT / "data" / f"arrival_depth_{tag}.csv")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.0, 11.2), dpi=160)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1, 1], hspace=0.52, wspace=0.20,
                          left=0.065, right=0.975, top=0.845, bottom=0.065)

    # ---- 上段: 時間帯ごとの合計到着率 -----------------------------------------
    ax = fig.add_subplot(gs[0, :]); style(ax)
    xs = np.arange(len(MAIN)); w = 0.38
    tot = {}
    for off, (side, col) in enumerate(((("買い(bid)"), BID), ("売り(ask)", ASK))):
        v = [float(S.filter((pl.col("window") == m) & (pl.col("side") == side))
                   ["rate_mean"].sum()) for m in MAIN]
        tot[side] = v
        ax.bar(xs + (off - 0.5) * w, v, width=w, color=col, zorder=3, linewidth=0)
        for i, y in enumerate(v):
            ax.annotate(f"{y:.0f}", (xs[i] + (off - 0.5) * w, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m}\n{SUB[m]}" for m in MAIN], fontsize=9)
    ax.set_ylabel("到着率の合計[件/秒]", color=INK2, fontsize=10)
    ax.set_title("時間帯ごとの注文到着率(全ての深さ帯の合計)", loc="left",
                 color=INK, fontsize=12, pad=26, weight="bold")
    hi = max(max(tot["買い(bid)"]), max(tot["売り(ask)"]))
    lo = min(min(tot["買い(bid)"]), min(tot["売り(ask)"]))
    ax.text(0, 1.06, f"最も濃い帯と最も薄い帯で {hi/lo:.0f} 倍の開きがある。"
            "どの時間帯でも買いのほうが多い。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.legend(handles=[Patch(facecolor=BID, edgecolor="none", label="買い(bid)"),
                       Patch(facecolor=ASK, edgecolor="none", label="売り(ask)")],
              loc="upper right", frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
              handlelength=1.6, handleheight=1.0, borderpad=0.2)
    ax.set_ylim(0, hi * 1.22)

    # ---- 下段 6 枚: 深さ帯ごとの形(共通の対数目盛)----------------------------
    vals = S.filter(pl.col("window").is_in(MAIN))["rate_mean"]
    ymin = max(float(vals.filter(vals > 0).min()) * 0.6, 1e-3)
    ymax = float(vals.max()) * 1.6
    for i, m in enumerate(MAIN):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3]); style(ax)
        for side, col in (("買い(bid)", BID), ("売り(ask)", ASK)):
            t = S.filter((pl.col("window") == m) & (pl.col("side") == side)).sort("band_i")
            xs2 = t["band_i"].to_numpy(); ys = t["rate_mean"].to_numpy()
            ax.plot(xs2, ys, "-o", color=col, lw=2.0, ms=4.5, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.1)
        nd = int(S.filter(pl.col("window") == m)["n_days"].max())
        s_tot = sum(tot[s][MAIN.index(m)] for s in ("買い(bid)", "売り(ask)"))
        ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(BANDS, rotation=55, ha="right", fontsize=7.5)
        ax.set_title(f"{m}\n{SUB[m]} / {nd} 日 / 合計 {s_tot:.0f} 件秒⁻¹",
                     loc="left", color=INK, fontsize=10, pad=8, weight="bold")
        if i % 3 == 0:
            ax.set_ylabel("到着率[件/秒](対数)", color=INK2, fontsize=9.5)
        if i >= 3:
            ax.set_xlabel("mid からの距離[bp]", color=INK2, fontsize=9.5)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.get_yaxis().set_minor_formatter(mpl.ticker.NullFormatter())

    fig.suptitle(f"{a.coin} 板の深さごとの注文到着率(1 秒あたり)",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.065, 0.938,
             "到着 = L1 の status が open になった行(その注文が実際に板に載った瞬間)。"
             "深さは到着より厳密に前の BBO の mid からの距離。\n"
             "日ごとに 1 秒あたりの率を出してから日をまたいで平均している"
             "(活発な 1 日に引きずられないため)。下段 6 枚は共通の対数目盛。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.005,
             "出所: Hyperliquid L4 (Artemis) L1 注文イベント + l2/bbo / 窓 2026-05-04〜08-09",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_arrival_depth.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
