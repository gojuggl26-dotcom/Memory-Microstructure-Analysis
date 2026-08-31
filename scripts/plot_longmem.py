"""9 系列の長期記憶と自己相関を図示する。

    ① ACF   ② PACF   ③ 4 推定量から出した d   ④ ★帰無対照で見た推定量の偏り
    ⑤ DFA のスケーリング   ⑥ 生 と 周期除去   ⑦ lag-k の一覧   ⑧ 推定量どうしの一致

    uv run python scripts/plot_longmem.py --coin xyz:MU
出力: charts/<coin>_longmem.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
UP, DOWN, WARN, GREEN, PURPLE = "#e34948", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#dfe9f5", DOWN, "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1a4f96", DOWN, "#dfe9f5", SURFACE,
                                                "#f7dcd8", UP, "#9b2b2a"])
S = ["ofi", "depth", "spread", "queue", "order_size", "arrivals", "cancels",
     "churn", "wallet_act"]
JA = {"ofi": "OFI", "depth": "depth", "spread": "spread", "queue": "queue size",
      "order_size": "order size", "arrivals": "order arrivals",
      "cancels": "cancellations", "churn": "liquidity churn",
      "wallet_act": "wallet activity"}
# 3 つの群で色を分ける(流量 / 板の状態 / 活動量)
COL = {"ofi": PURPLE,
       "depth": "#1baf7a", "spread": "#12855c", "queue": "#0b5c3f",
       "order_size": "#f2a15e", "arrivals": "#eb6834", "cancels": "#c94f22",
       "churn": "#8ab6e8", "wallet_act": "#2a78d6"}
EST = [("d_from_hurst", "Hurst(R/S) − 0.5"), ("d_from_dfa", "DFA α − 0.5"),
       ("gph_d", "GPH d"), ("lw_d", "Local Whittle d")]
LAGS = [1, 10, 60, 300, 3600]


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


def logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_parquet(ROOT / "data" / f"longmem_daily_{tag}.parquet")
    C = pl.read_parquet(ROOT / "data" / f"longmem_curves_{tag}.parquet")
    nday = D["dt"].n_unique()

    def med(kind, series, col):
        t = D.filter((pl.col("kind") == kind) & (pl.col("series") == series))
        return float(np.nanmedian(t[col].to_numpy().astype(float)))

    def cur(series, name):
        g = (C.filter((pl.col("series") == series) & (pl.col("curve") == name))
             .group_by("x").agg(v=pl.col("v").median()).sort("x"))
        return g["x"].to_numpy(), g["v"].to_numpy()

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 10.0), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.30,
                          left=0.05, right=0.985, top=0.785, bottom=0.085)

    # ---- ① ACF --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for s in S:
        xx, vv = cur(s, "acf")
        ax.plot(xx, vv, "-", color=COL[s], lw=1.7, label=JA[s])
    ax.axhline(0, color=INK, lw=1.0)
    logx(ax)
    ax.set_xlabel("ラグ[秒](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("ACF", color=INK2, fontsize=9.5)
    ax.set_title("① ACF — 指数的には減らない\n"
                 "OFI(紫)だけラグ 1 でほぼ 0。他は 1 時間先まで尾を引く",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=6.8, frameon=False, labelcolor=INK2, ncol=2)
    style(ax)

    # ---- ② PACF -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for s in S:
        xx, vv = cur(s, "pacf")
        ax.plot(xx, vv, "-", color=COL[s], lw=1.5)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlim(0, 50)
    ax.set_xlabel("ラグ[秒]", color=INK2, fontsize=9.5)
    ax.set_ylabel("PACF", color=INK2, fontsize=9.5)
    ax.set_title("② PACF — ラグ 1 に集中し、その後は小さく続く\n"
                 "少数の AR 項では説明しきれない(長期記憶の形)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ③ 4 推定量の d ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    M = np.array([[med("生", s, c) for c, _ in EST] for s in S])
    xs = np.arange(len(S))
    w = 0.2
    for j, (c, lab) in enumerate(EST):
        ax.bar(xs + (j - 1.5) * w, M[:, j], width=w, label=lab,
               color=[PURPLE, GREEN, WARN, DOWN][j])
    ax.axhline(0, color=INK, lw=1.0)
    ax.axhline(0.5, color=UP, lw=1.2, ls="--")
    ax.text(len(S) - 0.5, 0.52, "0.5 = 非定常の境", ha="right", color=UP,
            fontsize=8, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([JA[s] for s in S], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("分数階差 d", color=INK2, fontsize=9.5)
    ax.set_title("③ ★4 つの推定量から出した d は揃わない\n"
                 "OFI と depth は揃うが、到着系は GPH だけ 2 倍近く高い",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, ncol=2)
    style(ax)

    # ---- ④ 帰無対照 ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    keys = [("hurst", "Hurst(R/S)", 0.5), ("dfa", "DFA α", 0.5),
            ("gph_d", "GPH d", 0.0), ("lw_d", "Local Whittle d", 0.0)]
    for i, (k, lab, tgt) in enumerate(keys):
        v = D.filter(pl.col("kind") == "帰無対照")[k].to_numpy().astype(float)
        v = v[np.isfinite(v)] - tgt
        bp = ax.boxplot([v], positions=[i], widths=0.55, showfliers=False,
                        patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(NEUTRAL)
            b.set_edgecolor(INK2)
        for w2 in bp["whiskers"] + bp["caps"]:
            w2.set_color(INK2)
        for m2 in bp["medians"]:
            m2.set_color(UP)
            m2.set_linewidth(2)
        ax.annotate(f"{np.median(v):+.3f}", (i, np.median(v)),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=UP, weight="bold")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[1] for k in keys], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("あるべき値からのずれ", color=INK2, fontsize=9.5)
    ax.set_title("④ ★帰無対照 — R/S だけ +0.055 上振れする\n"
                 "系列を並べ替えれば記憶は消えるはず。残る分が推定量自身の偏り",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑤ DFA のスケーリング -----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for s in ["ofi", "depth", "arrivals", "wallet_act"]:
        xx, vv = cur(s, "dfa")
        if len(xx) == 0:
            continue
        ax.plot(xx, vv / vv[0], "-", color=COL[s], lw=1.9,
                label=f"{JA[s]}(α={med('生', s, 'dfa'):.2f})")
    xr = np.array([4, 10000.0])
    for al, ls in ((0.5, ":"), (1.0, "--")):
        ax.plot(xr, (xr / 4) ** al, ls, color=MUTED, lw=1.2)
    ax.text(6000, (6000 / 4) ** 0.5 * 1.2, "α=0.5(記憶なし)", color=MUTED,
            fontsize=7.5, ha="right")
    ax.text(2200, (2200 / 4) ** 1.0 * 0.6, "α=1", color=MUTED, fontsize=7.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    logx(ax)
    # ★text.parse_math=False だと対数の y 軸も既定の目盛が mathtext のまま出る
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: (f"{v:,.0f}" if v >= 1 else f"{v:g}")))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("スケール[秒](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("F(s) / F(4)(対数)", color=INK2, fontsize=9.5)
    ax.set_title("⑤ DFA のスケーリング\n"
                 "直線なら自己相似。傾きが α",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)

    # ---- ⑥ 生 と 周期除去 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    raw = [med("生", s, "lw_d") for s in S]
    des = [med("周期除去", s, "lw_d") for s in S]
    ax.bar(xs - 0.2, raw, width=0.4, color=DOWN, label="生")
    ax.bar(xs + 0.2, des, width=0.4, color=WARN, label="日内周期を除去")
    ax.set_xticks(xs)
    ax.set_xticklabels([JA[s] for s in S], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Local Whittle d", color=INK2, fontsize=9.5)
    dd = float(np.mean(np.array(des) - np.array(raw)))
    ax.set_title(f"⑥ 日内周期を除いても d はほとんど動かない\n"
                 f"平均で {dd:+.3f}。長期記憶は周期の見かけではない",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑦ lag-k の一覧 -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    H = np.array([[med("生", s, f"acf{k}") for k in LAGS] for s in S])
    lim = float(np.nanmax(np.abs(H)))
    im = ax.imshow(H, cmap=SEQ, vmin=0, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(LAGS)))
    ax.set_xticklabels([f"{k}s" for k in LAGS], fontsize=8.5)
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels([JA[s] for s in S], fontsize=8)
    for i in range(len(S)):
        for j in range(len(LAGS)):
            ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if H[i, j] > lim * 0.55 else INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("⑦ lag-1 と lag-k の自己相関\n"
                 "OFI だけ全ラグでほぼ 0。他は 1 時間先まで残る",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=2)
    cb.outline.set_visible(False)

    # ---- ⑧ 推定量どうしの一致 -----------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    lw = np.array([med("生", s, "lw_d") for s in S])
    for c, lab, cl, mk in [("d_from_hurst", "Hurst − 0.5", PURPLE, "o"),
                           ("d_from_dfa", "DFA − 0.5", GREEN, "s"),
                           ("gph_d", "GPH d", WARN, "^")]:
        y = np.array([med("生", s, c) for s in S])
        ax.scatter(lw, y, s=42, color=cl, label=lab, marker=mk,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
    lo, hi = 0.0, 0.72
    ax.plot([lo, hi], [lo, hi], "--", color=INK, lw=1.2, zorder=1)
    ax.text(hi * 0.98, hi * 0.90, "一致線", ha="right", color=INK2, fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Local Whittle d(最も偏りが小さい)", color=INK2, fontsize=9.5)
    ax.set_ylabel("他の推定量の d", color=INK2, fontsize=9.5)
    ax.set_title("⑧ 推定量どうしの一致\n"
                 "GPH(橙)は上、R/S(紫)は下に外れる。d は 1 つに決まらない",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)

    fig.suptitle(f"{a.coin} 9 系列の長期記憶と自己相関"
                 f"({nday} 日・1 秒格子・1 日 86,400 点)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.925,
             "定常な長期記憶過程では d = H − 1/2、DFA α = H = d + 1/2 が成り立つ。"
             "したがって R/S・DFA・GPH・Local Whittle から出した d が揃うかどうかがそのまま検算になる。\n"
             "系列を日内で無作為に並べ替えた帰無対照(④)を必ず併記する。"
             "残ったずれは記憶ではなく推定量自身の偏りであり、R/S は +0.055 上振れするので"
             "補正せずに読んではいけない。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_longmem.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
