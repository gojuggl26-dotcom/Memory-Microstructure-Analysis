"""MicroPrice と midprice の差から作った確率推移行列を描く。

上段は行列そのもの(行=差の帯、列=予測ホライズン、セル=P(mid が上昇))を
立会日・閉場日で並べる。**色は「何もしない場合」からの乖離**で塗る。
無条件の P(上昇) はホライズンごとに違う(同値が減るため k とともに上がる)ので、
0.5 を基準に塗ると全面が同じ色に寄って構造が見えないため。

下段は、上段の生の確率が同値に薄められていることを示す 2 枚。

    uv run python scripts/plot_microprice.py --coin xyz:MU
出力: charts/<coin>_microprice_matrix.png
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
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
DOWN, UP = "#2a78d6", "#e34948"          # 青=下落寄り / 赤=上昇寄り
HOR = [1, 5, 10, 20, 30, 50, 100]
SMALL_N = 1000                            # これ未満の帯は薄く描いて注意を促す


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"microprice_cells_{tag}.parquet")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    cmap = LinearSegmentedColormap.from_list("d", [DOWN, "#f2f1ec", UP])

    fig = plt.figure(figsize=(15.4, 11.6), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.42, wspace=0.42,
                          left=0.10, right=0.945, top=0.825, bottom=0.075)

    # ★左右で色スケールを共有する。別々に正規化すると同じ色が別の値を指し、
    #   立会日と閉場日を見比べられなくなる。
    lim = 0.0
    for dtl in ["立会日", "閉場日"]:
        s = C.filter(pl.col("day_type") == dtl)
        lim = max(lim, float((s["p_up"] - s["base_up"]).abs().max()) * 100)

    # ---- 上段: 確率推移行列 --------------------------------------------------
    ims = []
    for col, dtl in enumerate(["立会日", "閉場日"]):
        ax = fig.add_subplot(gs[0, col])
        s = C.filter(pl.col("day_type") == dtl)
        labels = (s.sort("bin_i")["bin"].unique(maintain_order=True).to_list())
        M = np.full((len(labels), len(HOR)), np.nan)
        T = np.full((len(labels), len(HOR)), np.nan)
        N = np.zeros(len(labels))
        for i, lab in enumerate(labels):
            for j, k in enumerate(HOR):
                r = s.filter((pl.col("bin") == lab) & (pl.col("k") == k))
                if r.height:
                    M[i, j] = (r["p_up"][0] - r["base_up"][0]) * 100
                    T[i, j] = r["p_up"][0] * 100
                    N[i] = max(N[i], r["n"][0])
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        ims.append(im)
        for i in range(len(labels)):
            faint = N[i] < SMALL_N
            for j in range(len(HOR)):
                if np.isnan(T[i, j]):
                    continue
                ax.text(j, i, f"{T[i,j]:.0f}", ha="center", va="center",
                        fontsize=9.5, color=INK if abs(M[i, j]) < lim * 0.55 else "white",
                        alpha=0.45 if faint else 1.0,
                        weight="normal" if faint else "bold")
        ax.set_xticks(range(len(HOR))); ax.set_xticklabels(HOR, fontsize=9.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f"{l}  ({int(n):,})" if n >= SMALL_N else f"{l}  ({int(n):,}) ※"
                            for l, n in zip(labels, N)], fontsize=9)
        ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
        if col == 0:
            ax.set_ylabel("micro − mid(bp)  (件数)", color=INK2, fontsize=10)
        base = [s.filter(pl.col("k") == k)["base_up"][0] * 100 for k in HOR]
        ax.set_title(f"{dtl} — P(mid が k イベント後に上昇)[%]\n"
                     f"無条件: " + " ".join(f"{v:.0f}" for v in base),
                     loc="left", color=INK, fontsize=11.5, pad=10, weight="bold")
        ax.tick_params(colors=MUTED, length=3, width=0.8)
        ax.grid(False)

    # 2 枚で 1 本の色目盛りを共有する(スケールが同一であることを図で示す)
    cb = fig.colorbar(ims[0], ax=[fig.axes[0], fig.axes[1]], fraction=0.030, pad=0.015)
    cb.set_label("何もしない場合(無条件)との差[pt]", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)

    # ---- 下段左: 同値を除いた方向の確率 --------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for dtl, ls in [("立会日", "-"), ("閉場日", "--")]:
        s = C.filter(pl.col("day_type") == dtl)
        for k, cl in [(1, "#9ca3af"), (10, "#0891b2"), (100, "#7c3aed")]:
            t = s.filter((pl.col("k") == k) & (pl.col("n") >= SMALL_N)).sort("bin_i")
            if t.height == 0:
                continue
            ax.plot(t["bin_i"].to_numpy(), t["p_up_move"].to_numpy() * 100,
                    ls, color=cl, lw=1.9, marker="o", ms=4,
                    label=f"{dtl} k={k}", alpha=0.95)
    ax.axhline(50, color=INK, lw=1.1)
    ax.text(0.02, 50.6, "50% = 方向の情報なし", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color=INK2)
    s0 = C.filter(pl.col("day_type") == "立会日").sort("bin_i")
    labs = s0["bin"].unique(maintain_order=True).to_list()
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8.5)
    ax.set_xlabel("micro − mid(bp)", color=INK2, fontsize=10)
    ax.set_ylabel("P(上昇 | 動いた)[%]", color=INK2, fontsize=10)
    ax.set_title("★同値を除くと方向の情報だけが残る\n"
                 "中央の帯(全体の 74%)はどのホライズンでも 50% = コイン投げ",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=7.5, frameon=False, ncol=2, labelcolor=INK2)

    # ---- 下段右: 変化なし率 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for dtl, cl in [("立会日", DOWN), ("閉場日", "#f59e0b")]:
        s = C.filter(pl.col("day_type") == dtl)
        v = []
        for k in HOR:
            t = s.filter(pl.col("k") == k)
            v.append((t["p_flat"] * t["n"]).sum() / t["n"].sum() * 100)
        ax.plot(range(len(HOR)), v, "o-", color=cl, lw=2, ms=5, label=dtl)
        for j, y in enumerate(v):
            ax.annotate(f"{y:.0f}", (j, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=cl)
    ax.set_xticks(range(len(HOR))); ax.set_xticklabels(HOR, fontsize=9.5)
    ax.set_xlabel("予測ホライズン k(イベント)", color=INK2, fontsize=10)
    ax.set_ylabel("mid が動かなかった割合[%]", color=INK2, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title("上段の生の確率が 50% を下回る理由\n"
                 "k=1 では 4〜5 割が「変化なし」で、上昇にも下落にも入らない",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK2)

    # 枠線の扱いを全パネルで揃える(ヒートマップだけ黒枠が残ると印象が変わる)
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        ax.set_facecolor(SURFACE)
        heat = ax in (fig.axes[0], fig.axes[1])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(not heat)
            if not heat:
                ax.spines[sp].set_color(BASELINE); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)

    n_tot = int(C.filter(pl.col("k") == 1)["n"].sum())
    fig.suptitle(f"{a.coin} MicroPrice と midprice の差 → 将来 mid の上昇確率"
                 f"(イベント {n_tot:,} / 立会日 67 日・閉場日 31 日)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.10, 0.930,
             "micro = (bid×ask数量 + ask×bid数量)/(数量計)。"
             "恒等式 micro − mid = (板の偏り − 1/2) × スプレッド。\n"
             "左右の色スケールは共通。※ = 件数 1,000 未満の帯(薄く表示)",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_microprice_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
