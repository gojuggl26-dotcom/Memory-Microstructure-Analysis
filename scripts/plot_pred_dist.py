"""Q1 と Q5 のときの前向きリターンの分布を 6 つ描く (3 指標 x 2 分位)。

    uv run python scripts/plot_pred_dist.py --coin xyz:MU
出力: charts/<coin>_pred_dist.png      6 つの分布 (ホライズン別)
      charts/<coin>_pred_dist_cmp.png  Q1 と Q5 の比較・効果量・頑健性
      data/pred_dist_stats_<coin>.csv  6 x 8 ホライズンの要約統計

【読み方】
平均の差は 0.1〜1.1 bp なのに、リターンの散らばりは 1 秒でも SD 1.8〜2.2 bp ある。
つまり **分布はほとんど重なっている**。効果量 (平均差 / 共通SD) は 1 秒で
OBI 0.275 / OFI 0.110 / 攻撃的数量 0.036 にすぎない。分布を見ないで平均だけ
見ると、この重なりの大きさを見落とす。

ちょうど 0 のリターン (気配が動かなかった区間) は密度曲線から外し、
別途 P(r=0) として数値で示す。密度に混ぜると原点に無限大の棘が立つ。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FixedFormatter, FixedLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ["obi", "ofi_10s", "ai_net_10s"]
LAB = {"obi": "OBI 板の不均衡", "ofi_10s": "OFI 注文流の不均衡 (10 秒)",
       "ai_net_10s": "攻撃的注文の符号つき数量 (10 秒)"}
HCOL = {0.1: "#c9c7bf", 1.0: "#7fa8dd", 10.0: "#2a78d6", 60.0: "#0b3d78"}
QCOL = {0: "#b5322f", 4: "#1f5fa8"}
QNAME = {0: "Q1 (最も売り側)", 4: "Q5 (最も買い側)"}
def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


XT = [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100]
XL = ["−100", "−10", "−1", "−0.1", "0", "0.1", "1", "10", "100"]


def dens(H, f, h, q, per_dec=5):
    """密度曲線 (中心, 1bp あたりの密度) と、ちょうど 0 の割合を返す。

    ★ビンを粗くしてから描くこと。mid は半ティック刻みでしか動かず、この価格帯
    (約 550) では 1 刻み = 0.091 bp である。元の 1 桁 20 本のビンは |r| < 1bp の
    領域で刻みより細かいので、空のビンと詰まったビンが交互に並び、密度曲線が
    実体のないギザギザになる。既定の 1 桁 5 本まで束ねるとこれが消える。
    """
    d = (H.filter((pl.col("feat") == f) & (pl.col("h") == h)
                  & (pl.col("q") == q)).sort("bin"))
    lo = d["lo"].to_numpy()
    hi = d["hi"].to_numpy()
    c = d["count"].to_numpy().astype(float)
    tot = c.sum()
    fin = np.isfinite(lo) & np.isfinite(hi)
    # ちょうど 0 は [0, 0.001) のビンに入る。ここは原点の点質量なので分ける。
    zb = fin & (lo >= 0.0) & (hi <= 1e-3)
    p0 = float(c[zb].sum() / tot)
    k = fin & ~zb & (c > 0)
    ctr = 0.5 * (lo[k] + hi[k])
    cc = c[k]
    # 対数等比のまま束ねる。|r| < 1bp では log10 が負になるので、符号と
    # 桁位置を別々に持つこと。1 つの数に畳むと正のリターンが負側へ飛ぶ。
    step = 1.0 / per_dec
    sgn = np.where(ctr > 0, 1, -1)
    kk = np.floor(np.log10(np.abs(ctr)) / step).astype(np.int64)
    key = sgn * 1000 + kk
    out_x, out_y = [], []
    for u in np.unique(key):
        m = key == u
        sg = 1 if u > 0 else -1
        k = u - 1000 * sg
        a, b = 10.0 ** (k * step), 10.0 ** ((k + 1) * step)
        out_x.append(sg * np.sqrt(a * b))
        out_y.append(cc[m].sum() / (b - a) / tot)
    o = np.argsort(out_x)
    return np.array(out_x)[o], np.array(out_y)[o], p0, tot


def ccdf(H, f, h, q):
    """両側の裾確率 P(r >= x) と P(r <= -x) を x > 0 について返す。

    累積なのでビンの切り方に依存しない。上側と下側の曲線の開きが、
    そのまま向きの偏りである。
    """
    d = (H.filter((pl.col("feat") == f) & (pl.col("h") == h)
                  & (pl.col("q") == q)).sort("bin"))
    lo = d["lo"].to_numpy()
    hi = d["hi"].to_numpy()
    c = d["count"].to_numpy().astype(float)
    tot = c.sum()
    fin = np.isfinite(lo) & np.isfinite(hi)
    up = fin & (lo >= 1e-3)
    dn = fin & (hi <= -1e-3)
    xu = lo[up]
    yu = (c[up][::-1].cumsum())[::-1] / tot
    xd = -hi[dn]
    yd = c[dn].cumsum() / tot
    o = np.argsort(xd)
    return xu, yu, xd[o], yd[o]


def symx(ax):
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.45)
    ax.set_xlim(-200, 200)
    ax.xaxis.set_major_locator(FixedLocator(XT))
    ax.xaxis.set_major_formatter(FixedFormatter(XL))
    ax.minorticks_off()


def moments(C, f, h, q):
    r = (C.filter((pl.col("feat") == f) & (pl.col("h") == h) & (pl.col("q") == q))
         .select(pl.col(c).sum() for c in
                 ("n", "sum_r", "sum_r2", "sum_r3", "sum_r4",
                  "n_pos", "n_neg", "n_zero")).row(0))
    n, s1, s2, s3, s4, npos, nneg, nz = r
    m = s1 / n
    v = s2 / n - m * m
    sd = np.sqrt(v)
    m3 = s3 / n - 3 * m * s2 / n + 2 * m ** 3
    m4 = s4 / n - 4 * m * s3 / n + 6 * m * m * s2 / n - 3 * m ** 4
    return {"n": n, "mean": m, "sd": sd, "skew": m3 / sd ** 3,
            "kurt": m4 / sd ** 4, "p_pos": npos / n, "p_neg": nneg / n,
            "p_zero": nz / n}


def wmean(H, f, h, q, cut):
    d = H.filter((pl.col("feat") == f) & (pl.col("h") == h)
                 & (pl.col("q") == q)).sort("bin")
    lo = d["lo"].to_numpy()
    hi = d["hi"].to_numpy()
    c = d["count"].to_numpy().astype(float)
    ct = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                  np.where(np.isfinite(lo), lo, hi))
    return float((np.clip(ct, -cut, cut) * c).sum() / c.sum())


def fig_dist(H, C, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.suptitle(f"{tag}  Q1 と Q5 のときの前向きリターンの分布 — 6 通り"
                 f"(3 指標 × 2 分位)・97 日", color=INK, fontsize=13, y=0.985)
    for j, f in enumerate(FEATS):
        for i, q in enumerate((0, 4)):
            ax = axes[i, j]
            for h in (0.1, 1.0, 10.0, 60.0):
                x, y, p0, n = dens(H, f, h, q)
                hl = "100ms" if h < 1 else f"{h:g}s"
                ax.plot(x, y, color=HCOL[h], lw=1.7,
                        label=f"h={hl}  P(r=0)={100*p0:.0f}%")
            mm = moments(C, f, 1.0, q)
            ax.axvline(mm["mean"], color=QCOL[q], lw=1.6, ls="--", zorder=4)
            ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
            symx(ax)
            style(ax, "その後のリターン (bp)", "密度 (1 bp あたり)", logy=True)
            ax.set_ylim(1e-7, 3)
            ax.set_title(f"{LAB[f]} — {QNAME[q]}", color=QCOL[q], fontsize=10,
                         loc="left")
            ax.text(0.02, 0.03,
                    f"h=1s:  平均 {mm['mean']:+.3f}  SD {mm['sd']:.2f}  "
                    f"P(r>0) {100*mm['p_pos']:.1f}%  P(r<0) {100*mm['p_neg']:.1f}%",
                    color=INK2, fontsize=8, transform=ax.transAxes)
            legend(ax, fs=7.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def fig_cmp(H, C, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{tag}  Q1 と Q5 はどれだけ違うのか — 裾の非対称・効果量・頑健性",
                 color=INK, fontsize=13, y=0.985)
    for j, f in enumerate(FEATS):
        ax = axes[0, j]
        for q in (0, 4):
            xu, yu, xd, yd = ccdf(H, f, 1.0, q)
            ax.plot(xu, yu, color=QCOL[q], lw=2.0,
                    label=f"{'Q5' if q else 'Q1'}  上へ P(r ≥ x)")
            ax.plot(xd, yd, color=QCOL[q], lw=1.5, ls="--",
                    label=f"{'Q5' if q else 'Q1'}  下へ P(r ≤ −x)")
        style(ax, "しきい値 x (bp)", "1 秒後にそこを超える確率", logx=True,
              logy=True)
        ax.set_xlim(0.08, 100)
        ax.set_ylim(1e-6, 1)
        a0, b0 = moments(C, f, 1.0, 0), moments(C, f, 1.0, 4)
        sp = np.sqrt((a0["sd"] ** 2 + b0["sd"] ** 2) / 2)
        ax.set_title(f"{LAB[f]}   効果量 d = {(b0['mean']-a0['mean'])/sp:.3f}",
                     color=INK, fontsize=10, loc="left")
        legend(ax, fs=7.5, loc="lower left")

    ax = axes[1, 0]
    for f, c in zip(FEATS, ("#b5322f", "#1f5fa8", "#7a52c9")):
        d = []
        for h in HS:
            a, b = moments(C, f, h, 0), moments(C, f, h, 4)
            sp = np.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2)
            d.append((b["mean"] - a["mean"]) / sp)
        ax.plot(HS, d, color=c, lw=2.0, marker="o", ms=4.5, label=LAB[f])
    for v, t in ((0.2, "小 0.2"), (0.5, "中 0.5")):
        ax.axhline(v, color=MUTED, lw=1.0, ls=":", zorder=1)
        ax.text(HS[-1], v, t + " ", color=MUTED, fontsize=8, ha="right",
                va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(HS)
    ax.set_xticklabels(["100ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.55)
    style(ax, "予測ホライズン h", "効果量 d = (平均Q5 − 平均Q1) / 共通SD")
    ax.set_title("(d) 効果量は最大でも 0.28 — 「小」の水準", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper right")

    ax = axes[1, 1]
    for f, c in zip(FEATS, ("#b5322f", "#1f5fa8", "#7a52c9")):
        for q, ls in ((0, ":"), (4, "-")):
            v = [100 * moments(C, f, h, q)["p_pos"]
                 / (1 - moments(C, f, h, q)["p_zero"]) for h in HS]
            ax.plot(HS, v, color=c, lw=1.9, ls=ls,
                    label=f"{LAB[f].split(' ')[0]} {'Q5' if q else 'Q1'}")
    ax.axhline(50, color=INK, lw=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(HS)
    ax.set_xticklabels(["100ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"])
    ax.minorticks_off()
    style(ax, "予測ホライズン h", "動いた区間のうち上がった割合 (%)")
    ax.set_title("(e) 向きの当たり方 (r=0 を除いた条件つき)", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="lower right", ncol=2)

    ax = axes[1, 2]
    cuts = [3.0, 10.0, 30.0, 100.0, np.inf]
    xs = np.arange(len(cuts))
    for f, c in zip(FEATS, ("#b5322f", "#1f5fa8", "#7a52c9")):
        for h, mk, al in ((1.0, "o", 1.0), (10.0, "s", 0.55)):
            v = [wmean(H, f, h, 4, k) - wmean(H, f, h, 0, k) for k in cuts]
            ax.plot(xs, v, color=c, lw=1.8, marker=mk, ms=5, alpha=al,
                    label=f"{LAB[f].split(' ')[0]} h={h:g}s")
    ax.set_xticks(xs, ["±3bp", "±10bp", "±30bp", "±100bp", "切らず"],
                  fontsize=8.5)
    ax.set_ylim(0, 1.02)
    style(ax, "外れ値をこの値で丸めた", "Q5 − Q1 の平均差 (bp)")
    ax.set_title("(f) 裾を切っても差は残る — 外れ値のせいではない", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="lower right", ncol=2)

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"pred_cells_{tag}.parquet")
    H = pl.read_parquet(ROOT / "data" / f"pred_hist_{tag}.parquet")
    rows = []
    for f in FEATS:
        for q in (0, 4):
            for h in HS:
                m = moments(C, f, h, q)
                rows.append({"feat": f, "q": f"Q{q+1}", "h": h, **m,
                             "w30": wmean(H, f, h, q, 30.0),
                             "w10": wmean(H, f, h, q, 10.0)})
    pl.DataFrame(rows).write_csv(ROOT / "data" / f"pred_dist_stats_{tag}.csv")
    print(f"{a.coin}: 6 分布 × {len(HS)} ホライズン")
    ch = ROOT / "charts"
    fig_dist(H, C, a.coin, ch / f"{tag}_pred_dist.png")
    fig_cmp(H, C, a.coin, ch / f"{tag}_pred_dist_cmp.png")


if __name__ == "__main__":
    main()
