"""仮想成行 Q に対する板の応答と、その脆さ、そしてシグナルとしての強さを描く。

    uv run python scripts/plot_impact.py --coin xyz:MU
出力: charts/<coin>_impact_curve.png    Impact(Q) の形と限界量
      charts/<coin>_impact_fragility.png 脆さ・depth-at-risk・gap 調整
      charts/<coin>_impact_signal.png    前向き / 後ろ向き / 帰無対照 / 偏相関

【配色】
連続量の帯は単一色相の連続階調、符号のある量は発散(青 ↔ 灰 ↔ 赤)。
`palette_check.py` で背景との対比 3.0 以上を確認した階調を使う。
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
from build_impact import QS, QTAG, REGION_BP, STRESS_S  # noqa: E402
from build_impact_signal import FEATS, HLAB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BUY, SELL = "#2a78d6", "#e34948"          # 発散ペア: 青 = 買い / 赤 = 売り
HB, HR = 4.547787, 0.502944               # OKLCH 色相(#2a78d6 / #e34948)
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _ok2hex(L, a, b):
    def f(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    return "#" + "".join(f"{max(0, min(255, round(f(max(x, 0)) * 255))):02x}"
                         for x in rgb)


def seq(n, h=HB, L0=0.66, L1=0.28, C0=0.10, C1=0.16):
    return [_ok2hex(L0 + (L1 - L0) * (i / max(n - 1, 1)),
                    (C0 + (C1 - C0) * (i / max(n - 1, 1))) * math.cos(h),
                    (C0 + (C1 - C0) * (i / max(n - 1, 1))) * math.sin(h))
            for i in range(n)]


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
    if e == -1:
        return "0.1"
    if e == -2:
        return "0.01"
    return "10" + str(e).translate(SUP)


def style(ax, xlab="", ylab="", logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.yaxis.set_minor_formatter(NullFormatter())
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d = pl.read_parquet(ROOT / "data" / f"impact_{tag}.parquet")
    R = pl.read_csv(ROOT / "data" / f"impact_signal_{tag}.csv")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.framealpha": 0.92})
    out = ROOT / "charts"
    hs = float(np.nanmedian(d["spread_bp"].to_numpy()))

    def col(name):
        return d[name].to_numpy().astype(np.float64)

    # ================= 図 1: Impact(Q) の形 =================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.24,
                          left=0.055, right=0.985, top=0.845, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0])
    qs = np.array(QS)
    keep = {}
    for s, cc, ls, lab in (("b", SELL, "-", "売り(bid を削る)"),
                           ("a", BUY, (0, (5, 3)), "買い(ask を削る)")):
        med = [np.nanmedian(col(f"imp_{s}_{t}")) for t in QTAG]
        p25 = [np.nanpercentile(col(f"imp_{s}_{t}"), 25) for t in QTAG]
        p75 = [np.nanpercentile(col(f"imp_{s}_{t}"), 75) for t in QTAG]
        keep[s] = med
        ax.plot(qs, med, "o", ls=ls, color=cc, lw=2.2, ms=6, label=lab, zorder=3)
        ax.fill_between(qs, p25, p75, color=cc, alpha=0.13, lw=0)
    # べき指数(両対数の傾き)。買いと売りは中央値でほぼ重なる
    ex = np.log(keep["a"][2] / keep["a"][0]) / np.log(qs[2] / qs[0])
    ax.text(0.97, 0.06, f"Impact ∝ Q^{ex:.2f}(両対数の傾き)\n"
            f"買いと売りの中央値の差は {abs(keep['a'][1]-keep['b'][1])/keep['a'][1]*100:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=INK2)
    ax.axhline(hs / 2, color=MUTED, ls=(0, (4, 2)), lw=1.2, zorder=2)
    ax.text(qs[0], hs / 2 * 1.06, f"ハーフスプレッド {hs/2:.2f} bp",
            fontsize=8, color=MUTED, va="bottom")
    ax.set_title("① Impact(Q) = P_VWAP(Q) − mid", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "仮想成行の数量 Q(契約、対数)", "インパクト(bp)", logx=True, logy=True)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    for s, cc, lab in (("a", BUY, "買い"), ("b", SELL, "売り")):
        med = [np.nanmedian(col(f"slp_{s}_{t}")) for t in QTAG]
        ax.plot(qs, med, "o-", color=cc, lw=2.0, ms=6, label=lab, zorder=3)
    ax.set_title("② Slippage(Q) = P_VWAP(Q) − 最良気配", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "Q(契約、対数)", "スリッページ(bp)", logx=True)
    ax.text(0.03, 0.95, "Impact = Slippage + ハーフスプレッド\n(浮動小数の精度で厳密に成立)",
            transform=ax.transAxes, va="top", fontsize=8.2, color=INK2)
    legend(ax, loc="lower right")

    ax = fig.add_subplot(gs[0, 2])
    w = 0.36
    for i, (s, cc, lab) in enumerate((("a", BUY, "買い"), ("b", SELL, "売り"))):
        med = [np.nanmedian(col(f"lv_{s}_{t}")) for t in QTAG]
        ax.bar(np.arange(3) + (i - 0.5) * w, med, width=w, color=cc, label=lab,
               zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Q={q:g}" for q in QS])
    ax.set_title("③ 削る価格水準の数(中央値)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "", "占有された価格水準の数")
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    cols = seq(3)
    for i, (q, t) in enumerate(zip(QS, QTAG)):
        v = np.concatenate([col(f"mimp_a_{t}"), col(f"mimp_b_{t}")])
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) > 5000:
            v = v[:: max(len(v) // 200000, 1)]
        v.sort()
        ax.plot(v, 1 - np.arange(len(v)) / len(v), color=cols[i], lw=1.9,
                label=f"Q={q:g}", zorder=3)
    z = [np.nanmean(np.concatenate([col(f"mimp_a_{t}"), col(f"mimp_b_{t}")]) <= 0)
         for t in QTAG]
    ax.set_title("④ marginal impact  dImpact/dQ", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "bp / 契約(対数)", "これ以上である割合", logx=True)
    ax.set_xlim(1e-6, 3e1)                    # 下端は浮動小数の残りかす
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.06, "ちょうど 0(注文が最良気配 1 枚に収まる)の割合\n"
            + " / ".join(f"Q={q:g}: {100*zz:.0f}%" for q, zz in zip(QS, z)),
            transform=ax.transAxes, va="bottom", fontsize=8.2, color=INK2)
    legend(ax, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    zf = []
    for i, (q, t) in enumerate(zip(QS, QTAG)):
        v = np.concatenate([col(f"mdep_a_{t}"), col(f"mdep_b_{t}")])
        v = v[np.isfinite(v)]
        zf.append(float((v <= 0).mean()))
        v = np.sort(v[v > 0])
        ax.plot(v, 1 - np.arange(len(v)) / len(v), color=cols[i], lw=1.9,
                label=f"Q={q:g}", zorder=3)
    ax.set_title("⑤ marginal depth(削り終えた先 1bp にある数量)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "契約 / bp(対数)", "これ以上である割合", logx=True)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.06, "削り終えた先 1bp が空の割合\n"
            + " / ".join(f"Q={q:g}: {100*z:.0f}%" for q, z in zip(QS, zf)),
            transform=ax.transAxes, va="bottom", fontsize=8.2, color=INK2)
    legend(ax, loc="upper right")

    ax = fig.add_subplot(gs[1, 2])
    x = col("imp_a_q5")
    y = col("imp_b_q5")
    m = np.isfinite(x) & np.isfinite(y)
    idx = np.random.default_rng(0).choice(np.flatnonzero(m),
                                          size=min(40000, int(m.sum())),
                                          replace=False)
    ax.plot(x[idx], y[idx], ".", ms=1.2, color=MUTED, alpha=0.35, zorder=2)
    lim = [0.05, np.nanpercentile(x[m], 99.5)]
    ax.plot(lim, lim, "-", color=INK, lw=1.2, zorder=3)
    ax.set_title("⑥ 買いと売りのインパクトの非対称(Q=5)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "買いのインパクト(bp、対数)", "売りのインパクト(bp、対数)",
          logx=True, logy=True)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    fig.text(0.055, 0.972, "仮想成行 Q に対する板の応答", fontsize=17, color=INK,
             va="top")
    fig.text(0.055, 0.932,
             f"5 秒格子 {d.height:,} 点。Q は実測の成行 1 本の分布(中央値 0.511 契約)"
             f"に合わせて 0.5 / 5 / 50 契約。帯は四分位。", fontsize=10, color=INK2,
             va="top")
    p = out / f"{tag}_impact_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 2: 脆さ ===========================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.24,
                          left=0.055, right=0.985, top=0.845, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0])
    for s, cc, lab in (("b", SELL, "買い板(bid)"), ("a", BUY, "売り板(ask)")):
        v = col(f"frag_{s}")
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(v, bins=np.logspace(-4, 1, 80), color=cc, histtype="step", lw=1.9,
                density=True, label=lab, zorder=3)
    ax.set_title("① 脆さ — 板が 1 秒あたり何割入れ替わるか", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "毎秒(対数)", "密度", logx=True)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    v = col("frag_imb")
    v = v[np.isfinite(v)]
    ax.hist(v, bins=np.linspace(-1, 1, 81), color=BASELINE, zorder=3)
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.set_title("② 脆さの偏り(買い板 − 売り板)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "(bid − ask)/(bid + ask)", "格子点の数")
    ax.text(0.03, 0.95, f"中央値 {np.median(v):+.3f}", transform=ax.transAxes,
            va="top", fontsize=8.5, color=INK2)

    ax = fig.add_subplot(gs[0, 2])
    for s, cc, lab in (("b", SELL, "買い板"), ("a", BUY, "売り板")):
        v = col(f"dar_{s}")
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(v, bins=np.logspace(-1, 4, 80), color=cc, histtype="step", lw=1.9,
                density=True, label=lab, zorder=3)
    ax.set_title(f"③ depth-at-risk({STRESS_S:g} 秒で消える見込みの数量)",
                 color=INK, fontsize=10.5, pad=7, loc="left")
    style(ax, "契約(対数)", "密度", logx=True)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    fr = col("frag_a")
    ok = np.isfinite(fr)
    qs_ = np.nanpercentile(fr[ok], np.arange(5, 100, 10))
    bins = np.digitize(fr, qs_)
    for i, (q, t) in enumerate(zip(QS, QTAG)):
        g = col(f"gfr_a_{t}")
        med = [np.nanmedian(g[(bins == b) & np.isfinite(g)]) for b in range(11)]
        xx = [np.nanmedian(fr[bins == b]) for b in range(11)]
        ax.plot(xx, med, "o-", color=cols[i], lw=1.8, ms=5, label=f"Q={q:g}",
                zorder=3)
    ax.set_title("④ gap 調整後の上乗せ(脆さの十分位ごと)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "脆さ(毎秒)", "余分に払う額(bp)", logx=True, logy=True)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    for s, cc, lab in (("b", SELL, "買い板"), ("a", BUY, "売り板")):
        v = col(f"d25_{s}")
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(v, bins=np.logspace(0, 4.5, 80), color=cc, histtype="step", lw=1.9,
                density=True, label=lab, zorder=3)
    ax.set_title(f"⑤ mid ±{REGION_BP:g}bp の深さ", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "契約(対数)", "密度", logx=True)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 2])
    fa, fb = col("frag_a"), col("frag_b")
    sp = col("spread_bp")
    m = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(sp)
    fm = (fa + fb) / 2
    qq = np.nanpercentile(fm[m], np.arange(5, 100, 5))
    bb = np.digitize(fm, qq)
    xx = [np.nanmedian(fm[m & (bb == b)]) for b in range(20)]
    yy = [np.nanmedian(sp[m & (bb == b)]) for b in range(20)]
    ax.plot(xx, yy, "o-", color=INK, lw=1.8, ms=5, zorder=3)
    ax.set_title("⑥ 脆さとスプレッド幅(同時点)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "脆さ(毎秒、対数)", "スプレッド幅(bp)", logx=True)

    fig.text(0.055, 0.972, "板の脆さ — 見えている板とあてにできる板", fontsize=17,
             color=INK, va="top")
    fig.text(0.055, 0.932,
             f"脆さは直前 30 秒に mid ±{REGION_BP:g}bp で取り消された数量 ÷ "
             f"(30 秒 × 同区間の平均の深さ)。depth-at-risk と gap 調整は"
             f"「{STRESS_S:g} 秒だけ誰も板を足さなかったら」という応力試験。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_impact_fragility.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 3: シグナル =======================================
    fig = plt.figure(figsize=(16.4, 11.4), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.52,
                          left=0.175, right=0.885, top=0.875, bottom=0.065)
    feats = list(FEATS)
    for k, (tname, tlab) in enumerate((("dir", "向き(将来の log リターン)"),
                                       ("mag", "大きさ(将来の |log リターン|)"),
                                       ("vol", "将来の実現ボラティリティ"))):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        M = np.full((len(feats), len(HLAB)), np.nan)
        for i, f in enumerate(feats):
            for j, hl in enumerate(HLAB):
                r = R.filter((pl.col("feature") == f) & (pl.col("target") == tname)
                             & (pl.col("horizon") == hl))
                if r.height:
                    M[i, j] = r["corr_fwd"][0]
        v = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(HLAB)))
        ax.set_xticklabels(HLAB, fontsize=8.5)
        ax.set_yticks(range(len(feats)))
        # 右列は左列と同じ並びなので、ラベルは左列だけに置く(重なりを防ぐ)
        ax.set_yticklabels([FEATS[f] for f in feats] if k % 2 == 0 else [],
                           fontsize=7.4)
        for i in range(len(feats)):
            for j in range(len(HLAB)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                            fontsize=6.6,
                            color=INK if abs(M[i, j]) < v * 0.55 else SURFACE)
        ax.set_title(f"前向き相関 — {tlab}", color=INK, fontsize=10.5, pad=7,
                     loc="left")
        ax.tick_params(colors=INK2)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    ax = fig.add_subplot(gs[1, 1])
    sub = R.filter((pl.col("target") == "vol") & (pl.col("horizon") == "5分"))
    y = np.arange(sub.height)
    ax.barh(y - 0.26, sub["corr_fwd"].to_numpy(), height=0.24, color=SELL,
            label="前向き(将来)", zorder=3)
    ax.barh(y, sub["corr_bwd"].to_numpy(), height=0.24, color=BUY,
            label="後ろ向き(過去)", zorder=3)
    ax.barh(y + 0.26, sub["corr_partial"].to_numpy(), height=0.24, color=INK,
            label="偏相関(直前のボラ・スプレッド・深さの偏りを統制)", zorder=3)
    ax.plot(sub["corr_placebo"].to_numpy(), y, "o", ms=4, color=MUTED,
            label="帰無対照(1 日 + 6 時間ずらす)", zorder=4)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATS[f] for f in sub["feature"].to_list()], fontsize=7.4)
    ax.yaxis.tick_right()                     # 左隣の色見本と重ならないように
    ax.invert_yaxis()
    ax.set_title("5 分先の実現ボラティリティ — 4 通りの比較", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "相関係数", "")
    legend(ax, loc="lower right")

    fig.text(0.055, 0.975, "板の応答指標はシグナルになるか", fontsize=17,
             color=INK, va="top")
    fig.text(0.055, 0.942,
             "上段と左下は前向き相関(x は時刻 T まで、y は T 以降)。"
             "右下は同じ量を後ろ向き・帰無対照・偏相関と並べたもの。"
             "重ならない部分標本だけを使っている。", fontsize=10, color=INK2,
             va="top")
    p = out / f"{tag}_impact_signal.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
