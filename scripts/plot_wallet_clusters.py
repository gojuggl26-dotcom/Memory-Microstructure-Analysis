"""ウォレットの行動クラスタを図示する。

    A 埋め込み       … 距離行列の古典的 MDS を 2 次元。色 = クラスタ、大きさ = 注文数
    B 領域ごとの類似度 … 7 つの行動領域それぞれで、同じクラスタ内と別クラスタの類似度
    C クラスタの横顔 … 生存時間と mid からの距離の分布をクラスタごとに重ねる
    D 類似度の分布   … 対ごとの総合類似度。同じクラスタ内と別クラスタで分ける

★クラスタは**記述**であって正解ラベルではない。同じ主体が複数アドレスを使っている
可能性も、1 アドレスが複数戦略を回している可能性も検証していない。

配色は dataviz の参照パレット slot 1〜6(palette_check 済み)。
対比が 3:1 未満のスロットがあるので relief rule に従い直接ラベルを置く。

    uv run python scripts/plot_wallet_clusters.py --coin xyz:MU
出力: charts/<coin>_wallet_clusters.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wallet_clusters import DIST_EDGES, DOMAIN, JP, LIFE_EDGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
        "#4a3aa7", "#e34948"]


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


def edge_labels(edges, unit=""):
    lab = [f"< {edges[0]:g}"]
    lab += [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    lab += [f"≥ {edges[-1]:g}"]
    return [x + unit for x in lab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_csv(ROOT / "data" / f"wallet_clusters_{tag}.csv")
    S = pl.read_parquet(ROOT / "data" / f"wallet_sim_{tag}.parquet")
    P = pl.read_parquet(ROOT / "data" / f"wallet_prof_{tag}.parquet")
    cl = {r["user"]: r["cluster"] for r in C.select("user", "cluster").to_dicts()}
    S = S.with_columns(
        same=pl.col("a").replace_strict(cl) == pl.col("b").replace_strict(cl))
    ks = sorted(C["cluster"].unique().to_list())

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(15.0, 10.8), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24, left=0.065, right=0.965,
                          top=0.845, bottom=0.07)

    # ---- A 埋め込み ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    n = C["n_orders"].to_numpy().astype(float)
    sz = 25 + 320 * (np.log10(n) - np.log10(n.min())) / max(np.log10(n.max() / n.min()), 1e-9)
    for i, k in enumerate(ks):
        m = (C["cluster"] == k).to_numpy()
        ax.scatter(C["emb_x"].to_numpy()[m], C["emb_y"].to_numpy()[m], s=sz[m],
                   color=SLOT[i % len(SLOT)], alpha=0.75, edgecolors=SURFACE,
                   linewidths=1.0, zorder=4)
        cx, cy = C["emb_x"].to_numpy()[m].mean(), C["emb_y"].to_numpy()[m].mean()
        ax.annotate(f"クラスタ {k}({int(m.sum())} 者)", (cx, cy),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=8.8, color=SLOT[i % len(SLOT)], weight="bold")
    ax.set_xlabel("第 1 軸", color=INK2, fontsize=9.5)
    ax.set_ylabel("第 2 軸", color=INK2, fontsize=9.5)
    ax.set_title("A  行動の埋め込み(古典的 MDS)", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, f"丸の大きさ = 注文数(対数)/ {C.height} 者",
            transform=ax.transAxes, color=INK2, fontsize=8.8)

    # ---- B 領域ごとの類似度 --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    y = np.arange(len(DOMAIN))
    for lab, sel, col, off in (("同じクラスタ", True, SLOT[0], -0.16),
                               ("別クラスタ", False, SLOT[1], 0.16)):
        med, lo, hi = [], [], []
        for d in DOMAIN:
            v = S.filter(pl.col("same") == sel)[f"sim_{d}"].to_numpy()
            q = np.percentile(v, [25, 50, 75])
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ax.hlines(y + off, lo, hi, color=col, lw=5.0, zorder=4)
        ax.plot(med, y + off, "o", color=SURFACE, ms=5, markeredgecolor=col,
                markeredgewidth=1.8, zorder=5)
        ax.annotate(lab, (hi[0], y[0] + off), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.5, color=col, weight="bold", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([JP[d] for d in DOMAIN], fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("類似度(1 − Jensen–Shannon 距離)/ 帯 = 25〜75%",
                  color=INK2, fontsize=9.5)
    ax.set_title("B  どの行動でクラスタが分かれているか", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, "同じクラスタと別クラスタの差が大きい領域ほど、分類に効いている",
            transform=ax.transAxes, color=INK2, fontsize=8.8)

    # ---- C クラスタの横顔 ---------------------------------------------------
    for p, (dom, edges, unit, name) in enumerate((
            ("life", LIFE_EDGES, " s", "生存時間"),
            ("dist", DIST_EDGES, " bp", "mid からの距離"))):
        ax = fig.add_subplot(gs[1, p])
        style(ax)
        labs = edge_labels(edges, unit)
        ends = []
        for i, k in enumerate(ks):
            q = (P.filter((pl.col("domain") == dom) & (pl.col("cluster") == k))
                  .group_by("bin_i").agg(pl.col("val").mean()).sort("bin_i"))
            v = q["val"].to_numpy()
            x = np.arange(len(v))
            ax.plot(x, v * 100, "-o", color=SLOT[i % len(SLOT)], lw=2.0, ms=4.0,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4 + i)
            ends.append((v[-1] * 100, f"クラスタ {k}", SLOT[i % len(SLOT)], x[-1]))
        lo_, hi_ = ax.get_ylim()
        ends.sort(key=lambda e: e[0])
        fr = [(e[0] - lo_) / (hi_ - lo_) for e in ends]
        for j in range(1, len(fr)):
            fr[j] = max(fr[j], fr[j - 1] + 0.07)
        for (yv, nm, c, xv), f in zip(ends, fr):
            ax.annotate(nm, (xv, lo_ + f * (hi_ - lo_)), textcoords="offset points",
                        xytext=(7, 0), fontsize=8.3, color=c, weight="bold",
                        va="center", annotation_clip=False)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=50, ha="right", fontsize=7.6, color=MUTED)
        ax.set_ylabel("注文の割合 [%]", color=INK2, fontsize=9.5)
        ax.set_title(f"{'CD'[p]}  クラスタごとの{name}の分布", loc="left", color=INK,
                     fontsize=11.5, pad=20, weight="bold")
        ax.text(0, 1.035, "クラスタ内のウォレットで平均した分布",
                transform=ax.transAxes, color=INK2, fontsize=8.8)

    same_m = float(S.filter(pl.col("same"))["sim_all"].median())
    diff_m = float(S.filter(~pl.col("same"))["sim_all"].median())
    fig.suptitle(f"{a.coin} ウォレットの行動クラスタ", fontsize=13.5, y=0.972,
                 color=INK, weight="bold")
    fig.text(0.065, 0.932,
             f"7 つの行動領域(数量・距離・生存時間・時刻・取消・価格の置き方・最良気配)の"
             f"分布どうしを Jensen–Shannon 距離で比べ、平均連結で {len(ks)} 群に分けた。\n"
             f"同じクラスタ内の類似度は中央 {same_m:.3f}、別クラスタは {diff_m:.3f}。"
             "★クラスタは記述であって正解ラベルではない。同一主体が複数アドレスを"
             "使っている可能性は検証していない。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) l1 注文イベント + l2/bbo / 2026-05-04〜08-10",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_wallet_clusters.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
