"""板の隙間 13 種と OBI / OFI / 板の回復力の関係を描く。

図は 5 枚。

    (1) 同時点の関係(OBI / OFI)  … 13 種 × 2 列のヒートマップ
    (2) 前向きの関係(板の回復力) … 13 種 × 5 ホライズンのヒートマップ
    (3) 用量反応                  … 最も効く 3 指標の十分位ごとの平均回復力
    (4) Pearson と Spearman の食い違い … 91 通りの散布
    (5) 隙間そのものの分布        … 主な 4 指標の分位

★(1) と (2) を並べているが、**(1) は同時点、(2) だけが前向き**である。
色は順位相関(Spearman)。裾の重いこのデータでは Pearson が少数の外れ点に
振り回されることを churn のレポートで実測したため、主表は順位相関にする。

配色は scripts/palette_check.py で検証済み(ok=true、CVD ΔE 最小 20.0)。

    uv run python scripts/plot_gaps.py --coin xyz:MU
出力: charts/<coin>_gaps.png
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
NEG, POS, THIRD = "#e34948", "#2a78d6", "#7a4fb5"
DIV = LinearSegmentedColormap.from_list("d", [NEG, "#f4f2ee", POS])

FEATS = ["bid_first_gap", "ask_first_gap", "max_gap", "mean_gap", "median_gap",
         "gap_var", "gap_asym", "dist_next_bid_bp", "dist_next_ask_bp",
         "empty_count", "void_size", "void_asym", "depth_disc"]
JP = {"bid_first_gap": "bid first gap\n最良買い→次の占有(tick)",
      "ask_first_gap": "ask first gap\n最良売り→次の占有(tick)",
      "max_gap": "maximum gap\n最大の隙間(tick)",
      "mean_gap": "mean gap\n隙間の平均(tick)",
      "median_gap": "median gap\n隙間の中央値(tick)",
      "gap_var": "gap variance\n隙間の分散",
      "gap_asym": "gap asymmetry\n買い−売りの隙間の偏り",
      "dist_next_bid_bp": "dist to next occupied bid\nmid からの距離(bp)",
      "dist_next_ask_bp": "dist to next occupied ask\nmid からの距離(bp)",
      "empty_count": "empty-level count\n空だった水準の数",
      "void_size": "liquidity void size\n最長の連続空き(tick)",
      "void_asym": "liquidity void asymmetry\n空きの偏り",
      "depth_disc": "depth discontinuity\n隣接水準の数量比の |log| 最大"}
SHORT = {f: JP[f].split("\n")[0] for f in FEATS}
RES = ["回復力 1s", "回復力 2s", "回復力 5s", "回復力 10s", "回復力 30s"]


def style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def heat(ax, M, cols, title, vmax, fig, note=None, cbar=True):
    im = ax.imshow(M, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels([SHORT[f] for f in FEATS], fontsize=8.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:+.3f}".replace("+0.", ".").replace("-0.", "-."),
                    ha="center", va="center", fontsize=7.2,
                    color=SURFACE if abs(v) > vmax * 0.58 else INK2)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
    if note:
        ax.text(0, -0.13, note, transform=ax.transAxes, fontsize=8.5, color=INK2,
                va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--day", default="立会日")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    R = (pl.read_csv(ROOT / "data" / f"gaps_rel_{tag}.csv")
         .filter(pl.col("day_type") == a.day))
    S = (pl.read_parquet(ROOT / "data" / f"gaps_sub_{tag}.parquet")
         .filter(pl.col("day_type") == a.day))

    def val(f, t, col="spearman"):
        r = R.filter((pl.col("feat") == f) & (pl.col("target") == t))
        return float(r[col][0]) if r.height else np.nan

    # 大きさには |OBI| / |OFI|、符号つきには OBI / OFI
    SIGNED = {"gap_asym", "void_asym"}
    Mc = np.array([[val(f, "OBI" if f in SIGNED else "|OBI|"),
                    val(f, "OFI" if f in SIGNED else "|OFI|")] for f in FEATS])
    Mr = np.array([[val(f, t) for t in RES] for f in FEATS])

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(16.2, 12.0), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.42,
                          left=0.085, right=0.975, top=0.845, bottom=0.075)

    vmax = float(np.nanmax(np.abs(np.concatenate([Mc.ravel(), Mr.ravel()]))))
    heat(fig.add_subplot(gs[0, 0]), Mc, ["OBI 系", "OFI 系"],
         "(1) 同時点の関係 — OBI / OFI", vmax, fig, cbar=False, note=
         "大きさの 11 種は |OBI| / |OFI|、符号を持つ 2 種は OBI / OFI に当てた。\n"
         "どちらも格子点 T の量であり、予測ではない。")
    ax2 = fig.add_subplot(gs[0, 1:])
    heat(ax2, Mr, [r.replace("回復力 ", "") for r in RES],
         "(2) 前向きの関係 — 板の回復力(この図で唯一の予測)", vmax, fig,
         "回復力 R = (s_T − s_{T+k}) / (s_T − 平常スプレッド)。"
         "スプレッドが平常より広がった格子点でだけ定義する。\n"
         "R = 1 で完全復元、0 で戻らない。x は T で確定、y は (T, T+k]。")

    # ---- (3) 用量反応 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    # ★median gap のように値が数種類しかない指標は十分位が潰れて 3 点になり、
    #   線が壊れて見える。境目が 8 つ未満の指標はこの枚から外す。
    def prep(f):
        d = (S.select(f, "回復力 1s").drop_nulls()
             .filter(pl.col(f).is_not_nan() & pl.col("回復力 1s").is_not_nan()))
        if d.height < 100:
            return None, None, None
        q = d[f].to_numpy()
        return d, q, np.unique(np.quantile(q, np.linspace(0, 1, 11)))

    cand = [f for f in FEATS if (prep(f)[2] is not None and len(prep(f)[2]) >= 8)]
    order = sorted(cand, key=lambda f: -abs(val(f, "回復力 1s", "partial_sp")))[:3]
    for f, c, mk in zip(order, (POS, NEG, THIRD), ("o", "s", "^")):
        d, q, e = prep(f)
        y = d["回復力 1s"].to_numpy()
        b = np.clip(np.searchsorted(e, q, side="right") - 1, 0, len(e) - 2)
        xs = [q[b == i].mean() for i in range(len(e) - 1)]
        ys = [y[b == i].mean() for i in range(len(e) - 1)]
        ax.plot(range(len(xs)), ys, "-", marker=mk, color=c, lw=2.2, ms=5,
                markeredgecolor=SURFACE, markeredgewidth=0.9,
                label=f"{SHORT[f]}(偏ρ={val(f,'回復力 1s','partial_sp'):+.3f})")
    ax.set_xlabel("その指標の十分位(小 → 大)", color=INK2, fontsize=9.5)
    ax.set_ylabel("平均の回復力 R(1s)", color=INK2, fontsize=9.5)
    ax.set_title("(3) 用量反応 — 十分位ごとの平均回復力(1s)", loc="left",
                 color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="best")

    # ---- (4) スプレッドを抜くと何が残るか ------------------------------------
    #  dist_next_*_bp は定義上 半スプレッドを含む(実測 ρ = +0.79)。回復力は
    #  スプレッドの平均回帰そのものなので、素の相関はスプレッドの代理でありうる。
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    Rr = R.filter(pl.col("direction") == "前向き")
    raw = Rr["spearman"].to_numpy(); par = Rr["partial_sp"].to_numpy()
    isbp = np.array([f.endswith("_bp") for f in Rr["feat"].to_list()])
    lim = np.nanmax(np.abs(np.concatenate([raw, par]))) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "-", color=BASELINE, lw=1.2, zorder=2)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.scatter(raw[~isbp], par[~isbp], s=30, color=POS, zorder=4, alpha=0.85,
               edgecolor=SURFACE, linewidth=0.7, label="ティックで測った 11 種")
    ax.scatter(raw[isbp], par[isbp], s=44, color=NEG, marker="D", zorder=5,
               edgecolor=SURFACE, linewidth=0.8,
               label="mid からの bp で測った 2 種")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("素の順位相関", color=INK2, fontsize=9.5)
    ax.set_ylabel("スプレッドを抜いた偏順位相関", color=INK2, fontsize=9.5)
    ax.set_title("(4) スプレッドを抜くと何が残るか", loc="left", color=INK,
                 fontsize=11, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    m = np.isfinite(raw) & np.isfinite(par)
    # ★「縮む」ではなく「符号ごと反転する」が起きているので、そちらを数える
    flip = int((np.sign(raw[m]) != np.sign(par[m])).sum())
    below = int((par[m] < raw[m]).sum())
    ax.text(0.97, 0.06, f"{int(m.sum())} 通り中 {below} 通りが対角線より下、\n"
            f"うち {flip} 通りは符号ごと反転する",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)

    # ---- (5) 隙間そのものの分布 ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    show = ["mean_gap", "max_gap", "bid_first_gap", "empty_count"]
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    xs = np.arange(len(show))
    for i, f in enumerate(show):
        v = S[f].to_numpy(); v = v[np.isfinite(v)]
        if not len(v):
            continue
        q = np.quantile(v, qs)
        ax.plot([i, i], [q[0], q[4]], "-", color=BASELINE, lw=1.4, zorder=2)
        ax.plot([i, i], [q[1], q[3]], "-", color=POS, lw=6, zorder=3,
                solid_capstyle="butt", alpha=0.85)
        ax.plot(i, q[2], "o", color=SURFACE, ms=7, zorder=5,
                markeredgecolor=INK, markeredgewidth=1.4)
        ax.annotate(f"{q[2]:.1f}", (i, q[2]), textcoords="offset points",
                    xytext=(11, -3), fontsize=8.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[f].split(" ")[0] + "\n" + SHORT[f].split(" ", 1)[1]
                        if " " in SHORT[f] else SHORT[f] for f in show], fontsize=8)
    ax.set_ylabel("値(tick / 個)", color=INK2, fontsize=9.5)
    ax.set_title("(5) 隙間そのものの分布", loc="left", color=INK, fontsize=11,
                 pad=8, weight="bold")
    ax.text(0.98, 0.96, "太い帯 = 四分位、細い線 = p10–p90、白丸 = 中央値",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)

    fig.suptitle(f"{a.coin} 板の隙間 13 種と OBI / OFI / 板の回復力",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.085, 0.945,
             "隙間 = 占有された価格水準どうしの距離(ティック)。最良気配から外側へ 40 ティックを見る。"
             "色はすべて順位相関(Spearman)。\n"
             "★(1) は同時点の関係、(2) だけが前向きの関係である。"
             f"標本は {a.day} の 100ms 格子。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.7)
    fig.text(0.5, 0.008,
             "出所: Hyperliquid L4 (Artemis) L1 注文イベントから再構成した板 + l2/bbo / "
             "窓 2026-05-04〜08-09",
             color=MUTED, fontsize=8, ha="center")
    out = ROOT / "charts" / f"{tag}_gaps.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
