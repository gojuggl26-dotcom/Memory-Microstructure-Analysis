"""L4 由来のボラティリティ族を描く (item 47)。

    uv run python scripts/plot_vol.py --coin xyz:MU
出力: charts/<coin>_vol_signature.png  刻み幅の選び方 (signature plot)
      charts/<coin>_vol_measures.png   古典的 7 推定量
      charts/<coin>_vol_series.png     5 本の系列のボラティリティ

【読み方】
RV は刻み幅で値が変わる。この市場のリターンは正に系列相関しているので、
細かく刻むほど RV は小さくなる (教科書の「雑音で大きくなる」と逆)。
したがって単一の数値には意味がなく、曲線ごと読むこと。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vol import HOUR_GRID as HG, REF_GRID, SERIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_MID, C_MIC, C_NW, C_WARN = "#2a78d6", "#e34948", "#7a52c9", "#eb6834"
C5 = {"mid": "#2a78d6", "micro": "#e34948", "spread": "#1f8a5e",
      "bp": "#c48a12", "ofi": "#7a52c9"}
LAB5 = {"mid": "mid", "micro": "microprice", "spread": "スプレッド",
        "bp": "板圧力", "ofi": "OFI"}
LF = chr(10)
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "font.size": 9,
                     "text.parse_math": False,
                     "font.family": ["Yu Gothic", "MS Gothic", "sans-serif"]})


def _pow10(v, _p=None):
    if v <= 0:
        return ""
    e = int(round(math.log10(v)))
    if abs(v - 10 ** e) > 1e-9 * max(v, 1):
        return ""
    if e == 0:
        return "1"
    if 1 <= e <= 4:
        return f"{10 ** e:,}"
    return ("0.1" if e == -1 else "0.01" if e == -2
            else "10" + str(e).translate(SUP))


def style(ax, xlab="", ylab="", logx=False, logy=False):
    for lg, axis, setter in ((logx, ax.xaxis, ax.set_xscale),
                             (logy, ax.yaxis, ax.set_yscale)):
        if lg:
            setter("log")
            axis.set_major_locator(LogLocator(base=10.0))
            axis.set_major_formatter(FuncFormatter(_pow10))
            axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel(xlab, color=INK2, fontsize=9)
    ax.set_ylabel(ylab, color=INK2, fontsize=9)


def legend(ax, **kw):
    lg = ax.legend(fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def band(ax, x, m, lo, hi, c, lab, ls="-"):
    ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, m, color=c, lw=2.0, ls=ls, label=lab, zorder=3)


def curve(D, s, col):
    """格子ごとの (中央値, 25%, 75%) を返す。刻み 0 (イベント) は除く。"""
    q = (D.filter((pl.col("series") == s) & (pl.col("grid_s") > 0))
         .group_by("grid_s")
         .agg(m=pl.col(col).median(), lo=pl.col(col).quantile(0.25),
              hi=pl.col(col).quantile(0.75))
         .sort("grid_s"))
    return (q["grid_s"].to_numpy(), q["m"].to_numpy(),
            q["lo"].to_numpy(), q["hi"].to_numpy())


def fig_signature(D, N, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    fig.suptitle(f"{tag}  刻み幅をどう選ぶか — volatility signature plot (98 日)",
                 color=INK, fontsize=13, y=0.985)

    ax = axes[0, 0]
    for s, c, lab in (("mid", C_MID, "mid"), ("micro", C_MIC, "microprice")):
        x, m, lo, hi = curve(D, s, "rvol")
        band(ax, x, m, lo, hi, c, lab)
    x, m, _, _ = curve(D, "mid", "rv_nw5")
    ax.plot(x, np.sqrt(np.maximum(m, 0)), color=C_NW, lw=1.8, ls="--",
            label="mid・系列相関を補正 (NW5)", zorder=4)
    ax.axvline(REF_GRID, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.text(REF_GRID * 1.15, ax.get_ylim()[0], " 基準", color=MUTED, fontsize=8,
            va="bottom")
    style(ax, "刻み幅 (秒)", "1 日の実現ボラティリティ (bp)", logx=True)
    ax.set_title("(a) 細かく刻むほど小さくなる — 教科書と逆", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper left")

    ax = axes[0, 1]
    for s, c in (("mid", C_MID), ("micro", C_MIC)):
        x, m, lo, hi = curve(D, s, "rho1")
        band(ax, x, m, lo, hi, c, LAB5[s])
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
    ax.text(0.12, 0.02, "雑音 (気配の跳ね返り) ならここは負になる",
            color=MUTED, fontsize=8, transform=ax.transAxes)
    style(ax, "刻み幅 (秒)", "リターンの 1 次自己相関", logx=True)
    ax.set_title("(b) 全格子で正 = 一方向に歩く", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="upper left")

    ax = axes[0, 2]
    for s, c in (("mid", C_MID), ("micro", C_MIC)):
        x, m, lo, hi = curve(D, s, "zero_share")
        band(ax, x, m * 100, lo * 100, hi * 100, c, LAB5[s])
    ax.axhline(10, color=C_WARN, lw=1.0, ls="--", zorder=1)
    ax.text(0.03, 0.12, "10% — ここより上では BV と跳び検定が壊れる",
            color=C_WARN, fontsize=8, transform=ax.transAxes)
    ax.axvline(REF_GRID, color=MUTED, lw=1.0, ls=":", zorder=1)
    style(ax, "刻み幅 (秒)", "0 リターンの割合 (%)", logx=True)
    ax.set_title("(c) 気配が動かない区間の割合", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="upper right")

    ax = axes[1, 0]
    x, m, lo, hi = curve(D, "mid", "se_rv")
    _, mr, _, _ = curve(D, "mid", "rv")
    band(ax, x, m / mr * 100, lo / mr * 100, hi / mr * 100, C_MID,
         "RV の測定誤差 SE/RV")
    x2, m2, lo2, hi2 = curve(D, "mid", "max_share")
    band(ax, x2, m2 * 100, lo2 * 100, hi2 * 100, C_WARN,
         "最大 1 本が RV に占める割合")
    ax.axvline(REF_GRID, color=MUTED, lw=1.0, ls=":", zorder=1)
    style(ax, "刻み幅 (秒)", "RV に対する割合 (%)", logx=True, logy=True)
    ax.set_title("(d) 粗くするほど精度が落ち、外れ値に支配される", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper left")

    ax = axes[1, 1]
    om = N["slope"].to_numpy()
    ax.hist(om, bins=30, color=C_MID, alpha=0.75, lw=0)
    ax.axvline(0, color=INK, lw=1.4, zorder=3)
    neg = float((om < 0).mean()) * 100
    ax.text(0.03, 0.93, f"傾き < 0 の日: {neg:.0f}%  (i.i.d. 雑音なら必ず正)",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "RV(n) = IV + 2n·ω² の傾き", "日数")
    ax.set_title("(e) 雑音分散 ω² は正に推定されない", color=INK, fontsize=10,
                 loc="left")

    ax = axes[1, 2]
    R = D.filter((pl.col("series") == "mid") & (pl.col("grid_s") == REF_GRID))
    xv, yv = R["park_bp"].to_numpy() / 1.6651, R["rvol"].to_numpy()
    ax.scatter(xv, yv, s=18, color=C_MID, alpha=0.6, lw=0)
    lim = [0, max(xv.max(), yv.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=1)
    rho = float(np.corrcoef(xv, yv)[0, 1])
    ax.text(0.03, 0.93, f"相関 {rho:.3f}   RVol/Parkinson の中央値 "
            f"{float(np.median(yv / xv)):.2f}", color=INK, fontsize=8.5,
            transform=ax.transAxes, va="top")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    style(ax, "Parkinson の高安推定 (bp)", f"実現ボラ {REF_GRID:g} 秒格子 (bp)")
    ax.set_title("(f) 独立な推定量との突合", color=INK, fontsize=10, loc="left")

    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def fig_measures(R, tag, out):
    """古典的 7 推定量。R は基準格子・mid の日次。"""
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    fig.suptitle(f"{tag}  実現ボラティリティ族 — {REF_GRID:g} 秒格子・mid・98 日",
                 color=INK, fontsize=13, y=0.985)
    x = np.arange(R.height)
    rv = R["rv"].to_numpy()
    bv = R["bv"].to_numpy()
    rj = R["rj"].to_numpy()
    z = R["z"].to_numpy()
    up, dn = R["rs_up"].to_numpy(), R["rs_dn"].to_numpy()
    oc = R["oc_bp"].to_numpy()

    ax = axes[0, 0]
    ax.plot(x, np.sqrt(rv), color=C_MID, lw=1.8, label="RVol = √RV")
    ax.plot(x, np.sqrt(bv), color=C_MIC, lw=1.5, label="√BV (跳びに頑健)")
    ax.plot(x, np.sqrt(R["medrv"].to_numpy()), color=C_NW, lw=1.3, ls="--",
            label="√MedRV")
    style(ax, "標本日 (1 = 2026-05-04)", "1 日の実現ボラティリティ (bp)")
    ax.set_title("(a) 3 つの推定量はほぼ重なる", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="upper right")

    ax = axes[0, 1]
    zn = jump_null(R)
    ax.scatter(zn, z, s=24, color=C_MID, alpha=0.7, lw=0)
    lim = [0, max(float(np.nanmax(z)), float(np.nanmax(zn))) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=1)
    for v, c in ((1.645, C_WARN), (3.28, INK)):
        ax.axhline(v, color=c, lw=1.0, ls=":", zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    over = int((z > zn).sum())
    ax.text(0.03, 0.93, "帰無対照を上回る日 "
            f"{over}/{len(z)}" + LF + f"片側5%超 {int((z > 1.645).sum())} 日 / "
            f"Bonferroni(98検定, 3.28)超 {int((z > 3.28).sum())} 日",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "跳びが無い場合の z (0 リターン割合を揃えた模擬)",
          "観測された跳び統計量 z")
    ax.set_title("(b) 跳び検定 — 0 リターンによる見せかけとの比較", color=INK,
                 fontsize=10, loc="left")

    ax = axes[0, 2]
    cc = np.where(oc >= 0, C_MID, C_MIC)
    ax.scatter(np.sqrt(up), np.sqrt(dn), s=22, c=cc, alpha=0.7, lw=0)
    lim = [0, max(np.sqrt(up).max(), np.sqrt(dn).max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    sh = float((dn > up).mean()) * 100
    ax.text(0.03, 0.93, f"下向きが大きい日 {sh:.0f}%   青 = 上昇日 / 赤 = 下落日",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "√RS+ 上向き半分散 (bp)", "√RS− 下向き半分散 (bp)")
    ax.set_title("(c) 上向きと下向きの半分散", color=INK, fontsize=10,
                 loc="left")

    ax = axes[1, 0]
    asym = (dn - up) / rv
    ax.scatter(oc, asym, s=22, color=C_MID, alpha=0.7, lw=0)
    ax.axhline(0, color=BASELINE, lw=1.0)
    ax.axvline(0, color=BASELINE, lw=1.0)
    rr = float(np.corrcoef(oc, asym)[0, 1])
    b1 = np.polyfit(oc, asym, 1)
    xs = np.linspace(oc.min(), oc.max(), 10)
    ax.plot(xs, np.polyval(b1, xs), color=C_WARN, lw=1.6, zorder=3)
    ax.text(0.03, 0.93, f"相関 {rr:.3f} — 非対称はその日の値動きでほぼ決まる",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "その日の始値→終値 (bp)", "(RS− − RS+) / RV")
    ax.set_title("(d) 半分散の非対称は同語反復に近い", color=INK, fontsize=10,
                 loc="left")

    ax = axes[1, 1]
    se = R["se_rv"].to_numpy() / rv * 100
    ax.hist(se, bins=28, color=C_NW, alpha=0.8, lw=0)
    ax.axvline(float(np.median(se)), color=INK, lw=1.4)
    ax.text(0.55, 0.93, f"中央値 {np.median(se):.1f}%\n最悪 {se.max():.1f}%",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "SE(RV)/RV = √(2·RQ/n) / RV  (%)", "日数")
    ax.set_title("(e) RQ が与える RV 自身の測定誤差", color=INK, fontsize=10,
                 loc="left")

    ax = axes[1, 2]
    rvol = np.sqrt(rv)
    ax.hist(rvol, bins=28, color=C_MID, alpha=0.8, lw=0)
    med = float(np.median(rvol))
    ax.axvline(med, color=INK, lw=1.4)
    ann = med / 1e4 * math.sqrt(365) * 100
    ax.text(0.4, 0.93, f"中央値 {med:.0f} bp/日\n= 年率 {ann:.0f}%\n"
            f"最小 {rvol.min():.0f} / 最大 {rvol.max():.0f} bp",
            color=INK, fontsize=8.5, transform=ax.transAxes, va="top")
    style(ax, "1 日の実現ボラティリティ (bp)", "日数")
    ax.set_title("(f) 水準 — 年率換算は √365 倍", color=INK, fontsize=10,
                 loc="left")

    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def jump_null(R, n_rep=200, seed=7):
    """跳び検定の帰無対照。

    BV は 0 リターンが混ざると下振れするので、跳びが無くても z が大きく出る。
    そこで各日について「跳びの無い連続過程 + その日と同じ 0 リターン割合」を
    生成し、同じ z を計算する。観測 z がこの分布を超えて初めて跳びと言える。
    """
    rng = np.random.default_rng(seed)
    from build_vol import est
    out = []
    for n, zs, rv in zip(R["n"].to_numpy(), R["zero_share"].to_numpy(),
                         R["rv"].to_numpy()):
        n = int(n)
        zz = []
        for _ in range(n_rep):
            r = rng.standard_normal(n)
            r[rng.random(n) < zs] = 0.0
            sc = np.sqrt(rv / max((r * r).sum(), 1e-12))
            e = est(r * sc)
            if e is not None and np.isfinite(e["z"]):
                zz.append(e["z"])
        out.append(np.median(zz) if zz else np.nan)
    return np.array(out)


def fig_series(D, H, S, tag, out):
    """5 本の系列のボラティリティ。"""
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    fig.suptitle(f"{tag}  5 本の系列のボラティリティと予測力 — "
                 f"{REF_GRID:g} 秒格子・98 日", color=INK, fontsize=13, y=0.985)
    W = D.pivot(on="series", index="dt", values="rvol").sort("dt")
    V = {s: np.log(W[s].to_numpy()) for s in SERIES}

    ax = axes[0, 0]
    ks = list(SERIES)
    M = np.array([[float(np.corrcoef(np.argsort(np.argsort(V[a])),
                                     np.argsort(np.argsort(V[b])))[0, 1])
                   for b in ks] for a in ks])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(ks)):
        for j in range(len(ks)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color=INK if abs(M[i, j]) < 0.6 else "#ffffff")
    ax.set_xticks(range(len(ks)), [LAB5[k] for k in ks], rotation=30, ha="right")
    ax.set_yticks(range(len(ks)), [LAB5[k] for k in ks])
    ax.tick_params(colors=INK2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        colors=INK2, labelsize=8)
    ax.set_title("(a) 日次ボラの順位相関", color=INK, fontsize=10, loc="left")

    ax = axes[0, 1]
    for s in SERIES:
        h = H.filter(pl.col("series") == s)
        v = np.array([float(h[f"h{i:02d}"].sum()) for i in range(24)])
        ax.plot(np.arange(24), v / v.sum() * 100, color=C5[s], lw=2.0,
                label=LAB5[s])
    ax.axhline(100 / 24, color=BASELINE, lw=1.0, ls="--", zorder=1)
    ax.axvspan(13.5, 20, color=MUTED, alpha=0.08, lw=0, zorder=0)
    ax.text(16.7, ax.get_ylim()[1] * 0.97, "米国立会", color=MUTED, fontsize=8,
            ha="center", va="top")
    style(ax, "UTC の時刻帯", "その時刻帯が 1 日の RV に占める割合 (%)")
    ax.set_title(f"(b) 日内の形 ({HG:g} 秒格子)", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="upper left", ncol=2)

    ax = axes[0, 2]
    xm = V["mid"]
    for s in ("spread", "bp", "ofi"):
        yv = V[s]
        ax.scatter(np.exp(xm), (yv - yv.mean()) / yv.std(), s=16, color=C5[s],
                   alpha=0.55, lw=0, label=LAB5[s])
    style(ax, "mid の実現ボラティリティ (bp)", "各系列のボラ (標準化した対数)",
          logx=True)
    ax.set_title("(c) mid のボラと他系列のボラ", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="upper left")

    ax = axes[1, 0]
    A = S.filter(pl.col("term") == "追加項")
    nm = A["model"].to_list()
    tv = np.abs(A["t"].to_numpy())
    th = A["placebo_t_hi"].to_numpy()
    pp = A["placebo_p"].to_numpy()
    yy = np.arange(len(nm))
    ax.barh(yy, th, height=0.62, color=GRID, lw=0,
            label="帰無分布の 95% 点 (56 通りのずらし)")
    ax.barh(yy, tv, height=0.34, color=C_MID, lw=0, label="観測された |t|")
    ax.axvline(1.96, color=C_WARN, lw=1.0, ls="--", zorder=3)
    ax.text(1.96, len(nm) - 0.35, " 名目 1.96", color=C_WARN, fontsize=8,
            va="top")
    for i, v in enumerate(pp):
        ax.text(th[i] + 0.15, i, f"p={v:.2f}", color=INK2, fontsize=8,
                va="center")
    ax.set_yticks(yy, nm, fontsize=8.5)
    ax.set_xlim(0, float(th.max()) * 1.28)
    style(ax, "HAR に足したときの |t| (HAC 5 次)", "")
    ax.set_title("(d) どれも自分の帰無分布を超えない", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="lower right")

    ax = axes[1, 1]
    o1 = A["oos_r2_vs_har"].to_numpy() * 100
    ax.barh(yy, o1, height=0.6, color=np.where(o1 > 0, C_MID, C_MIC), lw=0)
    ax.axvline(0, color=INK, lw=1.2)
    hb = float(S.filter(pl.col("term") == "(標本外)")["oos_r2_vs_best_const"][0])
    ax.set_yticks(yy, nm, fontsize=8.5)
    style(ax, "標本外 R² の HAR からの増分 (%)", "")
    ax.set_title(f"(e) HAR 自体は定数予測の理論上限に対して {hb * 100:+.1f}%",
                 color=INK, fontsize=10, loc="left")

    ax = axes[1, 2]
    ax.plot(np.arange(W.height), np.exp(V["mid"]), color=C_MID, lw=1.6,
            label="mid のボラ (bp)")
    ax2 = np.exp(V["ofi"])
    ax.plot(np.arange(W.height), ax2 / np.median(ax2) * np.median(np.exp(V["mid"])),
            color=C5["ofi"], lw=1.3, ls="--",
            label="OFI のボラ (中央値を合わせた)")
    style(ax, "標本日 (1 = 2026-05-04)", "実現ボラティリティ (bp)")
    ax.set_title("(f) 値動きのボラと注文流のボラは一緒に動く", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper left")

    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    ch = ROOT / "charts"
    D = pl.read_csv(ROOT / "data" / f"vol_daily_{tag}.csv")
    N = pl.read_csv(ROOT / "data" / f"vol_noise_{tag}.csv")
    H = pl.read_csv(ROOT / "data" / f"vol_hour_{tag}.csv")
    S = pl.read_csv(ROOT / "data" / f"vol_signal_{tag}.csv")
    R = D.filter((pl.col("series") == "mid")
                 & (pl.col("grid_s") == REF_GRID)).sort("dt")
    print(f"{a.coin}: {R.height} 日 / 基準格子 {REF_GRID:g} 秒")
    fig_signature(D, N, a.coin, ch / f"{tag}_vol_signature.png")
    fig_measures(R, a.coin, ch / f"{tag}_vol_measures.png")
    fig_series(D.filter(pl.col("grid_s") == REF_GRID), H, S, a.coin,
               ch / f"{tag}_vol_series.png")


if __name__ == "__main__":
    main()
