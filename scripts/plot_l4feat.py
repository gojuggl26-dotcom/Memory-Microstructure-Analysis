"""特徴量パネルの棚卸しと予測力を図にする。

    uv run python scripts/plot_l4feat.py --coin xyz:INTC

出力
    charts/<coin>_l4feat.png        分類 × ホライズンの到達点・上位・帰無対照
    charts/<coin>_l4feat_h.png      ホライズン依存と、前半/後半・日ごとの安定性
    charts/<coin>_l4feat_dist.png   上位の用量反応と、板の主要量の分布
    charts/<coin>_l4feat_corr.png   冗長性(相関行列)

【配色】
系列の色は 3 色まで(青 #2a78d6 / 緑 #0f8f63 / 紫 #4a3aa7)。
符号のある量は発散(青 ↔ 赤 #e34948)。**帰無対照は色を足さず輪郭で表す**
(低彩度の灰は彩度下限を割って色覚検査に落ちるため)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap  # noqa: E402
import numpy as np                           # noqa: E402
import polars as pl                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_l4feat import (FLAB, SRC, day_list, feat_cols, pick,  # noqa: E402
                        positions, read_block)

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, NEUTRAL = "#e1e0d9", "#f0efec"
BLUE, GREEN, PURPLE, RED = "#2a78d6", "#0f8f63", "#4a3aa7", "#e34948"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#eef1f4", RED])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "text.parse_math": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})
HSEC = [0.1, 0.5, 1, 5, 10, 30, 60]


def save(fig, name):
    fig.savefig(CH / name, facecolor=SURFACE, bbox_inches="tight")
    print("保存:", CH / name, flush=True)
    plt.close(fig)


def fig_overview(pred: pl.DataFrame, tag: str, coin: str):
    fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.0))
    # (a) 分類 × ホライズンの最大 |r|
    p = pred.with_columns(ab=pl.col("r_fwd").abs())
    piv = (p.group_by(["family", "h"]).agg(pl.col("ab").max())
           .pivot(on="h", index="family", values="ab"))
    fams = sorted(piv["family"].to_list())
    M = np.array([[piv.filter(pl.col("family") == f)[h][0] for h in FLAB]
                  for f in fams], float)
    a = ax[0, 0]
    im = a.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0,
                  vmax=np.nanpercentile(M, 98))
    a.set_xticks(range(len(FLAB)), FLAB)
    a.set_yticks(range(len(fams)), [f[:22] for f in fams], fontsize=6.5)
    a.set_title("(a) 分類ごとの到達点 — その分類で最大の |順位相関|", loc="left")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.03)
    for i in range(len(fams)):
        j = int(np.nanargmax(M[i]))
        a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=5.6,
               color="white" if M[i, j] > np.nanpercentile(M, 80) else INK)

    # (b) 10 秒での上位 18 本
    t10 = (p.filter(pl.col("h") == "10s").sort("ab", descending=True).head(18))
    a = ax[0, 1]
    y = np.arange(t10.height)[::-1]
    a.barh(y, t10["r_fwd"].to_numpy(), color=BLUE, height=0.66,
           label="実測(前向き 10 秒)")
    a.barh(y, t10["r_plc"].to_numpy(), height=0.66, facecolor="none",
           edgecolor=INK, linewidth=0.9, label="帰無対照(1 時間ずらし)")
    a.set_yticks(y, t10["col"].to_list(), fontsize=6.5)
    a.axvline(0, color=INK, lw=0.8)
    a.set_xlabel("順位相関")
    a.legend(fontsize=7, frameon=False)
    a.set_title("(b) 10 秒先のマイクロプライスとの相関 — 上位 18 本", loc="left")

    # (c) 前向き vs 後ろ向き
    a = ax[1, 0]
    q = p.filter(pl.col("h") == "10s")
    a.scatter(q["r_bwd"], q["r_fwd"], s=7, color=BLUE, alpha=0.45, lw=0)
    lim = float(np.nanmax(np.abs(np.concatenate(
        [q["r_bwd"].to_numpy(), q["r_fwd"].to_numpy()])))) * 1.05
    a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    a.axhline(0, color=INK, lw=0.7)
    a.axvline(0, color=INK, lw=0.7)
    a.set_xlabel("後ろ向き 10 秒(= 同時性)")
    a.set_ylabel("前向き 10 秒(= 予測)")
    n_more = int((q["r_fwd"].abs() > q["r_bwd"].abs()).sum())
    a.set_title(f"(c) 予測か同時性か — 予測のほうが強い特徴量は "
                f"{n_more}/{q.height} 本", loc="left")

    # (d) 実測と帰無対照の分布
    a = ax[1, 1]
    b = np.linspace(0, float(np.nanpercentile(p["ab"].to_numpy(), 99.5)), 45)
    a.hist(p.filter(pl.col("h") == "10s")["ab"], bins=b, color=BLUE,
           alpha=0.85, label="実測 |r|")
    a.hist(p.filter(pl.col("h") == "10s")["r_plc"].abs(), bins=b,
           histtype="step", color=INK, lw=1.2, label="帰無対照 |r|")
    a.set_yscale("log")
    a.set_xlabel("|順位相関|(10 秒)")
    a.set_ylabel("本数")
    a.legend(fontsize=7, frameon=False)
    mx = float(p.filter(pl.col("h") == "10s")["r_plc"].abs().max())
    a.set_title(f"(d) 帰無対照の最大は |r| = {mx:.4f}", loc="left")
    fig.suptitle(f"{coin} 板と注文フローの特徴量 — 分類ごとの到達点と予測力",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, f"{tag}_l4feat.png")


def fig_horizon(pred: pl.DataFrame, daily: pl.DataFrame | None, tag: str, coin: str):
    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.4))
    p = pred.with_columns(ab=pl.col("r_fwd").abs())
    top = (p.filter(pl.col("h") == "10s").sort("ab", descending=True)
           .head(8)["col"].to_list())
    a = ax[0, 0]
    cols = [BLUE, GREEN, PURPLE]
    for i, c in enumerate(top):
        s = p.filter(pl.col("col") == c)
        v = [float(s.filter(pl.col("h") == h)["r_fwd"][0]) for h in FLAB]
        a.plot(HSEC, v, marker="o", ms=3.4, lw=1.3, color=cols[i % 3],
               alpha=0.45 + 0.18 * (i // 3), label=c)
    a.set_xscale("log")
    a.axhline(0, color=INK, lw=0.8)
    a.set_xticks(HSEC, FLAB)
    a.set_xlabel("予測ホライズン")
    a.set_ylabel("順位相関")
    a.legend(fontsize=6.6, frameon=False, ncol=2)
    a.set_title("(a) 上位 8 本のホライズン依存", loc="left")

    # (b) 分類ごとに最も効くホライズン
    a = ax[0, 1]
    g = (p.group_by(["family", "h"]).agg(pl.col("ab").max())
         .sort("ab", descending=True).unique(subset="family", keep="first"))
    g = g.sort("ab", descending=True).head(16)
    y = np.arange(g.height)[::-1]
    xi = [HSEC[FLAB.index(h)] for h in g["h"].to_list()]
    a.scatter(xi, y, s=42, color=BLUE, zorder=3)
    for k, (xx, yy, v) in enumerate(zip(xi, y, g["ab"].to_list())):
        a.text(xx * 1.35, yy, f"{v:.3f}", va="center", fontsize=6.3, color=INK2)
    a.set_xscale("log")
    a.set_xticks(HSEC, FLAB)
    a.set_yticks(y, [f[:24] for f in g["family"].to_list()], fontsize=6.5)
    a.set_xlim(0.06, 200)
    a.set_title("(b) 分類ごとに最も効くホライズンと、そのときの |r|", loc="left")

    # (c) 前半と後半
    a = ax[1, 0]
    q = p.filter(pl.col("h") == "10s")
    a.scatter(q["r_1st"], q["r_2nd"], s=7, color=GREEN, alpha=0.5, lw=0)
    lim = float(np.nanmax(np.abs(np.concatenate(
        [q["r_1st"].to_numpy(), q["r_2nd"].to_numpy()])))) * 1.05
    a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    r = np.corrcoef(np.nan_to_num(q["r_1st"].to_numpy()),
                    np.nan_to_num(q["r_2nd"].to_numpy()))[0, 1]
    a.set_xlabel("前半の日で測った r")
    a.set_ylabel("後半の日で測った r")
    a.set_title(f"(c) 標本外(日で前半/後半に分割) — 両者の相関 {r:.3f}", loc="left")

    # (d) 日ごとの符号
    a = ax[1, 1]
    if daily is not None and daily.height:
        d = daily.filter(pl.col("h") == "10s").sort("mean")
        y = np.arange(d.height)
        a.barh(y, d["pos"].to_numpy() / d["nday"].to_numpy() * 100,
               color=np.where(d["mean"].to_numpy() > 0, BLUE, RED), height=0.7)
        a.axvline(50, color=INK, lw=0.9, ls="--")
        a.set_yticks(y, d["col"].to_list(), fontsize=6.2)
        a.set_xlabel("r が正だった日の割合(%)")
        a.set_title("(d) 日ごとの符号 — 上位特徴量、10 秒", loc="left")
    else:
        a.axis("off")
    fig.suptitle(f"{coin} ホライズン依存と、標本外・日ごとの安定性",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, f"{tag}_l4feat_h.png")


def fig_dist(pred, tag, coin, days):
    """上位の用量反応と、板の主要量の分布。"""
    p = pred.with_columns(ab=pl.col("r_fwd").abs())
    top = (p.filter(pl.col("h") == "10s").sort("ab", descending=True)
           .head(3)["col"].to_list())
    show = ["spread_bp", "cdtot10", "age_b1"]
    ttl = ["スプレッド(bp)", "最良から 10 水準の厚み(枚)", "最良の買い注文の年齢(秒)"]
    cols = list(dict.fromkeys(top + show))
    got, _ = feat_cols(tag, days)
    pos, _ = positions()
    lb = [f"label_micro_{h}" for h in ("1s", "10s", "60s")]
    want = pick(got, cols)
    want.setdefault(str(SRC / tag), [])
    want[str(SRC / tag)] = want[str(SRC / tag)] + lb
    # ★ 60 秒おきの標本 B を使う。60 秒先まで見ても標本どうしが重ならない
    Xall, _ = read_block(tag, days, want, pos)
    X = Xall["B"]
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for i, c in enumerate(top):
        a = ax[0, i]
        x = X[c]
        for j, (h, col) in enumerate(zip(("1s", "10s", "60s"),
                                         (BLUE, GREEN, PURPLE))):
            xs, ys = x, X[f"label_micro_{h}"]
            m = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[m], ys[m]
            if xs.size < 1000:
                continue
            q = np.quantile(xs, np.linspace(0, 1, 11))
            q[-1] += 1e-9
            b = np.clip(np.searchsorted(q, xs, "right") - 1, 0, 9)
            mu = np.array([ys[b == k].mean() for k in range(10)])
            se = np.array([ys[b == k].std() / max(np.sqrt((b == k).sum()), 1)
                           for k in range(10)])
            a.errorbar(np.arange(10) + 1 + (j - 1) * 0.12, mu, yerr=1.96 * se,
                       marker="o", ms=3.2, lw=1.1, color=col, capsize=1.8,
                       label=f"{h} 先")
        a.axhline(0, color=INK, lw=0.8)
        a.set_xlabel(f"{c} の十分位")
        if i == 0:
            a.set_ylabel("将来のマイクロプライス変化(bp)")
        a.legend(fontsize=6.6, frameon=False)
        a.set_title(f"({chr(97+i)}) {c} の用量反応", loc="left")
    for i, (c, t) in enumerate(zip(show, ttl)):
        a = ax[1, i]
        x = X[c][np.isfinite(X[c])]
        x = x[x > 0]
        if x.size:
            a.hist(x, bins=np.geomspace(max(np.quantile(x, 0.001), 1e-6),
                                        np.quantile(x, 0.999), 60),
                   color=BLUE, alpha=0.85)
            a.set_xscale("log")
            a.axvline(np.median(x), color=RED, lw=1.2,
                      label=f"中央値 {np.median(x):.3g}")
            a.legend(fontsize=6.8, frameon=False)
        a.set_xlabel(t)
        a.set_ylabel("行数")
        a.set_title(f"({chr(100+i)}) {t} の分布", loc="left")
    fig.suptitle(f"{coin} 上位特徴量の用量反応と、板の主要量の分布",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, f"{tag}_l4feat_dist.png")


def fig_corr(tag, coin, fam):
    fp = DATA / f"l4feat_corr_{tag}.npy"
    if not fp.exists():
        return
    C = np.load(fp)
    cols = (DATA / f"l4feat_corr_{tag}.cols").read_text(encoding="utf-8").split("\n")
    order = np.argsort([fam.get(c, "zz") for c in cols], kind="stable")
    C = C[np.ix_(order, order)]
    f2 = [fam.get(cols[i], "?") for i in order]
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.8),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    im = a.imshow(C, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
    b = [0] + [i for i in range(1, len(f2)) if f2[i] != f2[i - 1]] + [len(f2)]
    for x in b[1:-1]:
        a.axhline(x - 0.5, color=INK, lw=0.35)
        a.axvline(x - 0.5, color=INK, lw=0.35)
    mid = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
    a.set_yticks(mid, [f2[b[i]][:18] for i in range(len(b) - 1)], fontsize=5.6)
    a.set_xticks([])
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.035)
    a.set_title("(a) 特徴量どうしの相関(分類の順に並べた)", loc="left")
    a = ax[1]
    iu = np.triu_indices(C.shape[0], 1)
    v = np.abs(C[iu])
    a.hist(v, bins=60, color=BLUE, alpha=0.85)
    a.set_yscale("log")
    a.axvline(0.99, color=RED, lw=1.2)
    n99 = int((v > 0.99).sum())
    a.set_xlabel("|相関|")
    a.set_ylabel("ペア数")
    a.set_title(f"(b) |r| > 0.99 のペアは {n99:,} 組 / {v.size:,} 組", loc="left")
    fig.suptitle(f"{coin} 特徴量の冗長性", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, f"{tag}_l4feat_corr.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    pred = pl.read_csv(DATA / f"l4feat_pred_{tag}.csv")
    fam = dict(zip(pred["col"].to_list(), pred["family"].to_list()))
    dp = DATA / f"l4feat_daily_{tag}.csv"
    daily = pl.read_csv(dp) if dp.exists() else None
    days = day_list(tag)
    fig_overview(pred, tag, a.coin)
    fig_horizon(pred, daily, tag, a.coin)
    fig_dist(pred, tag, a.coin, days)
    fig_corr(tag, a.coin, fam)


if __name__ == "__main__":
    main()
