"""板の弾力性を 3 枚の図にする。

    1 xyz_MU_resilience_curve.png  … 回復曲線と指数当てはめ、指標の分布、日次の κ
    2 xyz_MU_resilience_ols.png    … 指標 × ホライズンの回帰
    3 xyz_MU_resilience_matrix.png … 指標の帯 × ホライズンの確率遷移行列

配色は dataviz の参照パレット。系列の識別は slot 1〜6(palette_check 済み:
隣接ペアの CVD ΔE 最小 9.1 / 通常視 19.6、全 PASS)。aqua・yellow・magenta は
背景との対比が 3:1 未満なので **relief rule に従い全系列に直接ラベル**を置く。
遷移行列は blue↔red の発散、中点は灰色。

    uv run python scripts/plot_resilience.py --coin xyz:MU
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
from build_resilience import METRICS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP_B = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
RAMP_R = ["#f8d5d5", "#f0a9a9", "#e87e7e", "#e34948", "#c33231", "#9c2524"]
NEUTRAL = "#f0efec"
BBO_FLOOR_MS = 67.5          # BBO のイベント間隔の 25% 点。これより速い回復は観測できない

LAB = {"kappa_depth": "depth 回復率 κ", "kappa_spread": "spread 回復率 κ",
       "kappa_px": "BBO 補充率 κ", "t_half_ms": "回復の半減期",
       "latency_ms": "refill latency", "refill_size": "refill size",
       "refill_intensity": "refill intensity", "queue_repl_rate": "queue 補充率",
       "refill_prob": "refill probability", "r_1s": "1 秒後の回復度 R"}
REG = ["kappa_depth", "kappa_spread", "latency_ms", "refill_intensity",
       "refill_prob", "r_1s"]
MAT = ["kappa_depth", "latency_ms", "refill_prob", "r_1s"]


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


def logticks(ax, which, lo, hi):
    t, e = [], int(np.floor(np.log10(lo))) - 1
    while 10.0 ** e <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                t.append(v)
        e += 1
    ax_ = ax.xaxis if which == "x" else ax.yaxis
    ax_.set_ticks(t)
    ax_.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_.set_minor_formatter(mpl.ticker.NullFormatter())


def spread_labels(ax, ends, gap=0.06):
    lo, hi = ax.get_ylim()
    span = hi - lo
    items = sorted(ends, key=lambda e: e[0])
    fr = [(e[0] - lo) / span for e in items]
    for i in range(1, len(fr)):
        if fr[i] - fr[i - 1] < gap:
            fr[i] = fr[i - 1] + gap
    for (y, lb, c, x), f in zip(items, fr):
        ax.annotate(lb, (x, lo + f * span), textcoords="offset points",
                    xytext=(7, 0), fontsize=8.0, color=c, weight="bold",
                    va="center", annotation_clip=False)


def fit_kappa(tau_ms, r):
    """R = 1 − exp(−κ t) を ln(1−R) = −κt の最小二乗で当てる。κ は 1/s。"""
    m = np.isfinite(r) & (r > 0.02) & (r < 0.95)
    if m.sum() < 3:
        return np.nan
    t = tau_ms[m] / 1e3
    return float(-np.polyfit(t, np.log(1.0 - r[m]), 1)[0])


# ------------------------------------------------------------------ 図 1
def fig_curve(tag, coin, P, S, D):
    fig = plt.figure(figsize=(15.0, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.24, left=0.06, right=0.93,
                          top=0.845, bottom=0.07)
    bins = [b for b in P["drop_bin"].unique().sort().to_list()]
    kap = {}
    for i, side in enumerate(("買い", "売り")):
        ax = fig.add_subplot(gs[0, i])
        style(ax)
        ax.axvspan(1, BBO_FLOOR_MS, color=MUTED, alpha=0.12, lw=0, zorder=1)
        ends = []
        for j, b in enumerate(bins):
            t = P.filter((pl.col("side") == side) & (pl.col("drop_bin") == b)).sort("tau_ms")
            if t.height == 0:
                continue
            x, y = t["tau_ms"].to_numpy().astype(float), t["r_mean"].to_numpy()
            c = RAMP_B[j + 1] if side == "買い" else RAMP_R[j + 1]
            ax.plot(x, y, "-o", color=c, lw=2.0, ms=3.5, zorder=4 + j)
            k = fit_kappa(x, y)
            kap[(side, b)] = k
            if np.isfinite(k):
                ax.plot(x, 1 - np.exp(-k * x / 1e3), color=c, lw=1.2,
                        ls=(0, (2, 2)), zorder=3)
            ends.append((y[-1], f"{b}  κ={k:.2f}", c, x[-1]))
        ax.set_xscale("log")
        ax.set_xlim(1, 1.4e4)
        logticks(ax, "x", 1, 5000)
        spread_labels(ax, ends, gap=0.07)
        ax.set_xlabel("ショックからの経過 [ms](対数)", color=INK2, fontsize=9.5)
        ax.set_ylabel("回復度 R = (D(t)−D_shock)/(D0−D_shock)", color=INK2, fontsize=9.5)
        ax.set_title(f"{'AB'[i]}  {side}側の平均回復経路とショックの大きさ", loc="left",
                     color=INK, fontsize=11.5, pad=20, weight="bold")
        ax.text(0, 1.035, "点線 = 1−exp(−κt) の当てはめ / 灰色帯は BBO の更新が届かない領域",
                transform=ax.transAxes, color=INK2, fontsize=8.6)

    # ---- C 指標の分位 ----
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    keys = [k for k in LAB if k in S.columns and k != "refill_prob"]
    for i, k in enumerate(keys):
        v = S[k].to_numpy().astype(float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < 100:
            continue
        q = np.percentile(v, [10, 25, 50, 75, 90])
        ax.plot([q[0], q[4]], [i, i], color=BASELINE, lw=1.4, zorder=3)
        ax.plot([q[1], q[3]], [i, i], color=SLOT[i % 6], lw=5.0, zorder=4,
                solid_capstyle="butt")
        ax.plot([q[2]], [i], "o", color=SURFACE, ms=5, zorder=5,
                markeredgecolor=SLOT[i % 6], markeredgewidth=1.8)
        ax.annotate(f"中央 {q[2]:.3g}", (q[4], i), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.0, color=INK2, va="center")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{LAB[k]}\n[{METRICS[k][1]}]" for k in keys], fontsize=8.0,
                       color=MUTED)
    ax.set_xscale("log")
    # 既定の LogFormatter は text.parse_math=False だと $\mathdefault{...}$ が
    # そのまま出るので、目盛を明示して自前の書式にする
    lo_x, hi_x = ax.get_xlim()
    logticks(ax, "x", max(lo_x, 1e-2), hi_x)
    ax.invert_yaxis()
    ax.set_xlabel("値(対数)/ 太い帯 = 25〜75%、細い線 = 10〜90%", color=INK2, fontsize=9.5)
    ax.set_title("C  指標の分布", loc="left", color=INK, fontsize=11.5, pad=20,
                 weight="bold")
    pr = float(S["refill_prob"].mean())
    ax.text(0, 1.035, f"refill probability(1 秒以内に D0 まで戻る割合)は {pr:.1%}",
            transform=ax.transAxes, color=INK2, fontsize=8.6)

    # ---- D 日ごとの κ ----
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    kb = [c for c in D.columns if c.startswith("kappa") and "買い" in c][0]
    ka = [c for c in D.columns if c.startswith("kappa") and "売り" in c][0]
    x = np.arange(D.height)
    ends = []
    for c, col, lb in ((kb, SLOT[0], "買い κ"), (ka, SLOT[1], "売り κ")):
        ax.plot(x, D[c].to_numpy(), "-o", color=col, lw=1.6, ms=3.0, zorder=4)
        ends.append((D[c].to_numpy()[-1], lb, col, x[-1]))
    spread_labels(ax, ends, gap=0.05)
    d = D["asym"].to_numpy()
    ax.set_xticks([0, D.height // 4, D.height // 2, 3 * D.height // 4, D.height - 1])
    ax.set_xticklabels([D["dt"][i][5:] for i in
                        (0, D.height // 4, D.height // 2, 3 * D.height // 4, D.height - 1)],
                       fontsize=8.5, color=MUTED)
    ax.set_xlabel("日", color=INK2, fontsize=9.5)
    ax.set_ylabel("その日の κ の中央値 [1/s]", color=INK2, fontsize=9.5)
    ax.set_title("D  買い側と売り側の弾力性(日ごと)", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035,
            f"非対称 (κ買−κ売)/(κ買+κ売) は中央 {np.nanmedian(d):+.4f}、"
            f"買いが速い日 {int(np.nansum(d > 0))} / {int(np.isfinite(d).sum())} 日",
            transform=ax.transAxes, color=INK2, fontsize=8.6)

    fig.suptitle(f"{coin} 板の弾力性 — 流動性ショックからの回復", fontsize=13.5,
                 y=0.972, color=INK, weight="bold")
    fig.text(0.06, 0.932,
             "片側の最良気配の数量が直前 1 秒の平均に対して 50% 以上失われた瞬間をショックとし、"
             "そこからの回復を D(t) = D0 + (D_shock − D0)e^{−κt} で測る。\n"
             "κ は 1 件ごとには半減期から、平均経路には最小二乗で求めた。"
             "l2/bbo の更新間隔は中央 131ms なので、それより速い回復は原理的に見えない。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo / 2026-05-04〜08-09 の 98 日",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_resilience_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}")
    return kap


# ------------------------------------------------------------------ 図 2
def fig_ols(tag, coin, O):
    fig = plt.figure(figsize=(15.0, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.26, left=0.06, right=0.90,
                          top=0.845, bottom=0.07)

    def panel(ax, title, sub, ylab, sel, split=None, styles=("-",)):
        style(ax)
        ax.axhline(0, color=BASELINE, lw=1.1, zorder=2)
        ends = []
        for si, ls in enumerate(styles):
            for i, met in enumerate(REG):
                f = dict(sel)
                if split:
                    f[split[0]] = split[1][si]
                q = O.filter((pl.col("metric") == met))
                for k, v in f.items():
                    q = q.filter(pl.col(k) == v)
                q = q.sort("hor_ms")
                if q.height == 0:
                    continue
                x, y = q["hor_ms"].to_numpy(), q["r"].to_numpy()
                ax.plot(x, y, linestyle=ls, color=SLOT[i], lw=2.0 if si == 0 else 1.3,
                        alpha=1.0 if si == 0 else 0.75, zorder=4 + i)
                if si == 0:
                    ends.append((y[-1], LAB[met], SLOT[i], x[-1]))
        ax.set_xscale("log")
        ax.set_xlim(8, 3e5)
        ax.set_xticks([10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(["10ms", "100ms", "1s", "10s", "100s"], fontsize=8.5,
                           color=MUTED)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        spread_labels(ax, ends)
        ax.set_xlabel("予測ホライズン(対数)", color=INK2, fontsize=9.5)
        ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=20, weight="bold")
        ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.6)
        if len(styles) > 1:
            h = [plt.Line2D([], [], color=MUTED, ls=s, lw=2.0) for s in styles]
            ax.legend(h, list(split[1]), loc="upper left", frameon=False,
                      fontsize=8.5, labelcolor=INK2)

    base = dict(day_type="立会日", x_form="log1p")
    panel(fig.axes[0] if fig.axes else fig.add_subplot(gs[0, 0]),
          "A  買い側のショック → 絶対リターン",
          "log(1+x) / 立会日 / 相関(二乗が決定係数)", "相関",
          dict(side="買い", y_kind="絶対値", **base))
    panel(fig.add_subplot(gs[0, 1]), "B  売り側のショック → 絶対リターン",
          "同じ条件で側だけ入れ替えたもの", "相関",
          dict(side="売り", y_kind="絶対値", **base))
    panel(fig.add_subplot(gs[1, 0]), "C  符号つきリターンへ",
          "実線 = 買い側のショック、破線 = 売り側", "相関",
          dict(y_kind="符号つき", **base), split=("side", ("買い", "売り")),
          styles=("-", (0, (4, 2))))
    panel(fig.add_subplot(gs[1, 1]), "D  生の値と log(1+x) の違い",
          "絶対リターン / 買い側 / 実線 = log(1+x)、破線 = 生", "相関",
          dict(side="買い", y_kind="絶対値", day_type="立会日"),
          split=("x_form", ("log1p", "生")), styles=("-", (0, (4, 2))))

    n = int(O.filter((pl.col("metric") == "kappa_depth") & (pl.col("side") == "買い")
                     & (pl.col("day_type") == "立会日"))["n"].max() or 0)
    ncell = O.height
    fig.suptitle(f"{coin} 弾力性の指標は将来のリターンを説明するか", fontsize=13.5,
                 y=0.972, color=INK, weight="bold")
    fig.text(0.06, 0.932,
             f"指標は区間 [t0, t0+1s] の中で決まるので、x の確定時刻を t0+1s、"
             f"y の期間を (t0+1s, t0+1s+h] とした。先読みは無い。"
             f"立会日・買い側で 1 ホライズンあたり最大 {n:,} 件。\n"
             f"報告しているセルは {ncell:,} 個(指標 10 × 側 2 × 日区分 2 × 変換 2 × "
             f"目的変数 3 × ホライズン 12)。Bonferroni の閾値は |t| = 4.7。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo / 2026-05-04〜08-09 の 98 日",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_resilience_ols.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}")


# ------------------------------------------------------------------ 図 3
def bin_labels(edges):
    lab = [f"< {edges[0]:g}"]
    lab += [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    lab += [f"≥ {edges[-1]:g}"]
    return lab


def fig_matrix(tag, coin, T):
    fig = plt.figure(figsize=(15.0, 10.8), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.20, left=0.10, right=0.975,
                          top=0.84, bottom=0.075)
    hs = (T.filter(pl.col("metric") == MAT[0]).select("hor", "hor_ms").unique()
           .sort("hor_ms"))
    hor = hs["hor"].to_list()
    vmax = 6.0
    for p, met in enumerate(MAT):
        ax = fig.add_subplot(gs[p // 2, p % 2])
        style(ax)
        ax.grid(False)
        q = T.filter((pl.col("metric") == met) & (pl.col("side") == "買い")
                     & (pl.col("day_type") == "立会日"))
        labs = bin_labels(METRICS[met][0])
        nb = len(labs)
        M = np.full((nb, len(hor)), np.nan)
        SG = np.zeros((nb, len(hor)), bool)
        for r in q.to_dicts():
            j = hor.index(r["hor"])
            M[r["bin_i"], j] = r["diff"] * 100
            SG[r["bin_i"], j] = not (r["lo"] <= 0 <= r["hi"])
        for i in range(nb):
            for j in range(len(hor)):
                if not np.isfinite(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               facecolor=SURFACE, edgecolor=GRID, lw=1.0))
                    continue
                f = min(abs(M[i, j]) / vmax, 1.0)
                k = int(round(f * (len(RAMP_B) - 1)))
                c = NEUTRAL if k == 0 else (RAMP_B[k] if M[i, j] > 0 else RAMP_R[k])
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c,
                                           edgecolor=SURFACE, lw=1.4))
                ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if SG[i, j] else ""),
                        ha="center", va="center", fontsize=8.2,
                        color="#ffffff" if k >= 4 else INK)
        ax.set_xlim(-.5, len(hor) - .5)
        ax.set_ylim(nb - .5, -.5)
        ax.set_xticks(range(len(hor)))
        ax.set_xticklabels(hor, fontsize=8.0, color=MUTED, rotation=45, ha="right")
        ax.set_yticks(range(nb))
        ax.set_yticklabels(labs, fontsize=8.2, color=MUTED)
        ax.set_ylabel(f"{LAB[met]} [{METRICS[met][1]}]", color=INK2, fontsize=9.5)
        if p >= 2:
            ax.set_xlabel("予測ホライズン", color=INK2, fontsize=9.5)
        base = float(q["base"].median()) if q.height else np.nan
        ax.set_title(f"{'ABCD'[p]}  {LAB[met]}", loc="left", color=INK,
                     fontsize=11.5, pad=18, weight="bold")
        ax.text(0, 1.02, f"無条件の P(上昇|動いた) = {base:.1%} からの差 [pp]。"
                         "* は日単位ブートストラップの 95% 区間が 0 を含まない",
                transform=ax.transAxes, color=INK2, fontsize=8.3)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)

    fig.suptitle(f"{coin} 弾力性の帯ごとの上昇確率(買い側のショック・立会日)",
                 fontsize=13.5, y=0.968, color=INK, weight="bold")
    fig.text(0.10, 0.925,
             "価格は離散なので短いホライズンでは「変化なし」が多数を占める。"
             "生の P(上昇) ではなく **P(上昇 | 動いた)** を、無条件値との差で示す。\n"
             "青が無条件より上がりやすい、赤が下がりやすい。色は ±6pp で頭打ち。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006, "出所: Hyperliquid L4 (Artemis) l2/bbo / 2026-05-04〜08-09 の 98 日",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_resilience_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    P = pl.read_parquet(ROOT / "data" / f"resil_path_{tag}.parquet")
    files = sorted((ROOT / "data" / f"resil_{tag}").glob("dt=*.parquet"))
    S = pl.concat([pl.read_parquet(f, columns=list(LAB) + ["side", "day_type"])
                   for f in files if f.stat().st_size > 900], how="diagonal_relaxed")
    D = pl.read_csv(ROOT / "data" / f"resil_daily_{tag}.csv")
    O = pl.read_parquet(ROOT / "data" / f"resil_ols_{tag}.parquet")
    T = pl.read_parquet(ROOT / "data" / f"resil_trans_{tag}.parquet")
    fig_curve(tag, a.coin, P, S, D)
    fig_ols(tag, a.coin, O)
    fig_matrix(tag, a.coin, T)


if __name__ == "__main__":
    main()
