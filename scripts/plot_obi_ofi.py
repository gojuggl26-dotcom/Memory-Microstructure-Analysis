"""OBI と OFI から作った確率推移行列を描く。

上段は行列そのもの(行=帯、列=予測ホライズン、セル=P(上昇|変化あり))。
**色は「何もしない場合」からの乖離**で塗る。無条件の確率は 0.5 ちょうどではないので
0.5 を基準に塗ると構造が歪む。左右で色スケールを共有する。

下段左は OBI × OFI の同時分布。片方で層別してももう一方が効くなら、
2 つは別の情報を持っている。

下段右は **費用との比較**。確率が 65% でも、期待される値動きがスプレッドより
小さければ板を取りに行っては捕まえられない。ここを出さないと読み手が誤る。

配色は scripts/palette_check.py で検証済み:
  発散(青↔赤)  CVD ΔE 21.6 / 通常視 32.3 / 対比 4.30・3.85
  下段右(aqua/violet) CVD ΔE 31.1 / 通常視 35.8。aqua は対比 2.74 < 3 なので
  凡例ではなく **直接ラベル** を付ける(dataviz の relief rule)。

    uv run python scripts/plot_obi_ofi.py --coin xyz:MU
出力: charts/<coin>_obi_ofi_matrix.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
DOWN, UP = "#2a78d6", "#e34948"      # 発散: 青=下落寄り / 赤=上昇寄り
C_OBI, C_OFI = "#1baf7a", "#4a3aa7"  # 下段右の 2 系列(直接ラベルを付ける)
HOR = [1, 5, 10, 20, 30, 50, 100]
JOINT_K = 10
SMALL_N = 1000


def heat(ax, s: pl.DataFrame, labels: list[str], cmap, lim: float, title: str):
    M = np.full((len(labels), len(HOR)), np.nan)
    T = np.full((len(labels), len(HOR)), np.nan)
    N = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        for j, k in enumerate(HOR):
            r = s.filter((pl.col("bin") == lab) & (pl.col("k") == k))
            if r.height:
                M[i, j] = (r["p_up_move"][0] - r["base_up_move"][0]) * 100
                T[i, j] = r["p_up_move"][0] * 100
                N[i] = max(N[i], r["n"][0])
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(labels)):
        faint = N[i] < SMALL_N
        for j in range(len(HOR)):
            if np.isnan(T[i, j]):
                continue
            ax.text(j, i, f"{T[i, j]:.0f}", ha="center", va="center", fontsize=9.5,
                    color=INK if abs(M[i, j]) < lim * 0.55 else "white",
                    alpha=0.45 if faint else 1.0, weight="normal" if faint else "bold")
    ax.set_xticks(range(len(HOR))); ax.set_xticklabels(HOR, fontsize=9.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  ({int(n):,})" for l, n in zip(labels, N)], fontsize=9)
    ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")
    ax.grid(False)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"obi_ofi_cells_{tag}.parquet")
    J = pl.read_parquet(ROOT / "data" / f"obi_ofi_joint_{tag}.parquet")
    meta = pl.read_parquet(ROOT / "data" / f"obi_ofi_meta_{tag}.parquet")
    half = float(meta["half_spread_bp_median"][0])
    full = float(meta["spread_bp_median"][0])

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    cmap = LinearSegmentedColormap.from_list("d", [DOWN, NEUTRAL, UP])

    fig = plt.figure(figsize=(16.0, 12.4), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.34,
                          left=0.115, right=0.935, top=0.845, bottom=0.075)

    # ★2 枚で色スケールを共有する。別々に正規化すると同じ色が別の値を指す
    lim = 0.0
    for feat in ("OBI", "OFI_z"):
        s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == "立会日"))
        lim = max(lim, float((s["p_up_move"] - s["base_up_move"]).abs().max()) * 100)

    ims, axes_top = [], []
    for col, (feat, name) in enumerate([("OBI", "OBI(板の残高の偏り)"),
                                        ("OFI_z", "OFI(板の流量の偏り、σ 正規化)")]):
        ax = fig.add_subplot(gs[0, col])
        s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == "立会日"))
        labels = s.sort("bin_i")["bin"].unique(maintain_order=True).to_list()
        base = [s.filter(pl.col("k") == k)["base_up_move"][0] * 100 for k in HOR]
        ims.append(heat(ax, s, labels, cmap, lim,
                        f"{name}\nP(mid が k イベント後に上昇 | 変化あり)[%]  —  "
                        f"無条件 " + " ".join(f"{v:.0f}" for v in base)))
        axes_top.append(ax)
        ax.set_ylabel(("OBI の帯" if feat == "OBI" else "OFI_z の帯") + "  (件数)",
                      color=INK2, fontsize=10)

    cb = fig.colorbar(ims[0], ax=axes_top, fraction=0.028, pad=0.014)
    cb.set_label("何もしない場合(無条件)との差[pt]", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)

    # ---- 下段左: 同時分布 -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    s = J.filter((pl.col("day_type") == "立会日") & (pl.col("k") == JOINT_K))
    olab = s.sort("obi_i")["obi_bin"].unique(maintain_order=True).to_list()
    flab = s.sort("ofi_i")["ofi_bin"].unique(maintain_order=True).to_list()
    base10 = float(C.filter((pl.col("feat") == "OBI") & (pl.col("day_type") == "立会日")
                            & (pl.col("k") == JOINT_K))["base_up_move"][0]) * 100
    M = np.full((len(olab), len(flab)), np.nan)
    for i in range(len(olab)):
        for j in range(len(flab)):
            r = s.filter((pl.col("obi_i") == i) & (pl.col("ofi_i") == j))
            if r.height and r["n"][0] >= 500:
                M[i, j] = r["p_up_move"][0] * 100
    jlim = np.nanmax(np.abs(M - base10))
    ax.imshow(M - base10, cmap=cmap, norm=TwoSlopeNorm(0, -jlim, jlim), aspect="auto")
    for i in range(len(olab)):
        for j in range(len(flab)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color=INK if abs(M[i, j] - base10) < jlim * 0.55 else "white",
                        weight="bold")
    ax.set_xticks(range(len(flab))); ax.set_xticklabels(flab, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(olab))); ax.set_yticklabels(olab, fontsize=8)
    ax.set_xlabel("OFI_z の帯", color=INK2, fontsize=10)
    ax.set_ylabel("OBI の帯", color=INK2, fontsize=10)
    ax.set_title(f"★2 つは別の情報を持っている(立会日 k={JOINT_K})\n"
                 "片方を固定してももう一方の方向に色が変わる。件数 500 未満は空白",
                 loc="left", color=INK, fontsize=11, pad=10, weight="bold")
    ax.grid(False)

    # ---- 下段右: 費用との比較 -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axhspan(-half, half, color=NEUTRAL, lw=0, zorder=1)
    for y, ls, lab in ((half, "-", f"片道の半スプレッド {half:.2f} bp"),
                       (full, (0, (5, 3)), f"往復のスプレッド {full:.2f} bp")):
        for sgn in (1, -1):
            ax.axhline(sgn * y, color=BASELINE, lw=1.2, ls=ls, zorder=2)
        # 系列の直接ラベルは右端に置くので、費用の線のラベルは左端に寄せる
        ax.annotate(lab, xy=(-0.4, y), xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", color=MUTED, fontsize=8.5, zorder=6)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=2)

    kmax = HOR[-1]
    for feat, cl, nm in (("OBI", C_OBI, "OBI"), ("OFI_z", C_OFI, "OFI_z")):
        r = (C.filter((pl.col("feat") == feat) & (pl.col("day_type") == "立会日")
                      & (pl.col("k") == kmax)).sort("bin_i"))
        xs, ys = r["bin_i"].to_numpy(), r["mean_bp"].to_numpy()
        ax.plot(xs, ys, "-o", color=cl, lw=2.2, ms=6, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
        # aqua は対比が 3 未満なので凡例ではなく直接ラベルを置く
        ax.annotate(nm, xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    color=cl, fontsize=11, weight="bold", va="center", zorder=6)
    ax.set_xticks(range(9))
    ax.set_xticklabels(["最も\n売り寄り", "", "", "", "中立", "", "", "", "最も\n買い寄り"],
                       fontsize=8.5)
    ax.set_xlim(-0.5, 9.4)
    ax.set_xlabel("帯(左が売り寄り、右が買い寄り)", color=INK2, fontsize=10)
    ax.set_ylabel(f"k={kmax} イベント後までの期待値動き[bp]", color=INK2, fontsize=10)
    ax.set_title("★確率が大きくても費用を超えない\n"
                 f"最も有利な k={kmax} でも、往復のスプレッドに届く帯は 1 つも無い",
                 loc="left", color=INK, fontsize=11, pad=10, weight="bold")

    for ax_ in fig.axes:
        if ax_.get_label() == "<colorbar>":
            continue
        ax_.set_facecolor(SURFACE)
        is_heat = ax_ is not fig.axes[-1]
        for sp in ("top", "right"):
            ax_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax_.spines[sp].set_visible(not is_heat)
            if not is_heat:
                ax_.spines[sp].set_color(BASELINE); ax_.spines[sp].set_linewidth(0.8)
        ax_.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)

    n_tot = int(C.filter((pl.col("feat") == "OBI") & (pl.col("day_type") == "立会日")
                         & (pl.col("k") == 1))["n"].sum())
    fig.suptitle(f"{a.coin} OBI / OFI から見た将来 mid の上昇確率"
                 f"(立会日 67 日・{n_tot:,} イベント)",
                 fontsize=13.5, y=0.977, color=INK, weight="bold")
    fig.text(0.115, 0.935,
             "OBI = (bid数量 − ask数量)/(数量計) … 板の「残高」の偏り。"
             "OFI = Cont–Kukanov–Stoikov の板の「流量」の偏りを、直前 5,000 イベントの σ で正規化。\n"
             "説明変数は時刻 t で確定し、目的変数の期間は (t, t+k]。先読みは無い。"
             "同値(mid が動かない)は分母から除いてある。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo を再構成 / 窓 2026-05-04〜08-09 (98 日)",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_obi_ofi_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
