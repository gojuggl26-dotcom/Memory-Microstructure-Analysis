"""隠れ数量(iceberg)と注文分割の代理指標 8 つを描く。

    uv run python scripts/plot_iceberg.py --coin xyz:MU
出力: charts/<coin>_ice_refill.png   補充の形(指標 1・3・4)
      charts/<coin>_ice_hidden.png   隠れ数量の代理と回転(指標 2・5・6・7・8)
      data/ice_wallet_summary_<coin>.csv  口座ごとの 8 指標

【読み方】
この市場では出し直しは全員の既定行動なので、**約定の後**と**取消の後**を
必ず並べる。iceberg なら「削られたから足す」= 約定後のほうが速く・多いはず。
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
from build_iceberg import D_LAB, GAP_EDGES, HID_EDGES, NCH_EDGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
FILLC, CANCC, WARN = "#e34948", "#2a78d6", "#eb6834"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MIN_ORD = 10_000        # 口座ごとの指標を出す下限(その口座の総注文数)


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


def hist(H, key):
    v = H.filter(pl.col("hist") == key).sort("bin")["count"].to_numpy()
    return v.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_csv(ROOT / "data" / f"ice_daily_{tag}.csv")
    B = pl.read_csv(ROOT / "data" / f"ice_band_{tag}.csv")
    H = pl.read_csv(ROOT / "data" / f"ice_hist_{tag}.csv")
    W = pl.read_parquet(ROOT / "data" / f"ice_wallet_{tag}.parquet")

    g = W.group_by("wid").agg(
        n_days=pl.col("dt").n_unique(),
        **{c: pl.col(c).sum() for c in
           ("n_ord", "n_slot", "n_fill", "n_canc", "n_rf_fill", "n_rf_canc",
            "n_rf_same", "n_rf_fill_same", "sz_pl", "sz_fl", "szt", "n_ep")},
        n_modal=pl.col("n_modal").sum(),
        ep_child=pl.col("ep_child").mean(),
        ep_child_max=pl.col("ep_child_max").max())
    d = pl.col
    g = g.filter(d("n_ord") >= MIN_ORD).with_columns(
        rf_fill=pl.when(d("n_fill") > 0).then(d("n_rf_fill") / d("n_fill")),
        rf_canc=pl.when(d("n_canc") > 0).then(d("n_rf_canc") / d("n_canc")),
        same_sz_share=pl.when(d("n_rf_fill") + d("n_rf_canc") > 0)
        .then(d("n_rf_same") / (d("n_rf_fill") + d("n_rf_canc"))),
        modal_share=d("n_modal") / d("n_ord"),
        turnover_s=d("sz_pl") / (d("szt") / 1e9),
        ord_per_slot=d("n_ord") / d("n_slot"))
    g = g.with_columns(rf_ratio=d("rf_fill") / d("rf_canc"))
    g.write_csv(ROOT / "data" / f"ice_wallet_summary_{tag}.csv")

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.framealpha": 0.92})
    out = ROOT / "charts"
    x = np.arange(D.height)

    # ================= 図 1: 補充の形 ======================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.26,
                          left=0.055, right=0.975, top=0.845, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, D["rf_after_fill"] * 100, color=FILLC, lw=1.8,
            label="約定の後", zorder=3)
    ax.plot(x, D["rf_after_canc"] * 100, color=CANCC, lw=1.8,
            label="取消の後(対照)", zorder=3)
    ax.set_title("① 1 秒以内に同じ値段へ出し直した割合", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日(2026-05-04 から)", "%")
    legend(ax, loc="best")
    ax.text(0.5, -0.235,
            f"約定後 中央値 {100*D['rf_after_fill'].median():.1f}% / "
            f"取消後 {100*D['rf_after_canc'].median():.1f}% — ほぼ同じ",
            transform=ax.transAxes, fontsize=8.6, color=INK2, ha='center')

    ax = fig.add_subplot(gs[0, 1])
    ctr = np.sqrt(GAP_EDGES[:-1] * GAP_EDGES[1:]) / 1000.0    # 秒
    for k, cc, lab in (("gap_fill", FILLC, "約定の後"),
                       ("gap_canc", CANCC, "取消の後")):
        v = hist(H, k)
        if v.sum() == 0:
            continue
        c = np.cumsum(v) / v.sum()
        ax.plot(ctr, c[1:len(ctr) + 1] * 100, color=cc, lw=2.0, label=lab,
                zorder=3)
    ax.axvline(1.0, color=INK, ls=(0, (4, 2)), lw=1.2, zorder=4)
    ax.set_title("② 同じ値段へ出し直すまでの時間(累積)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "秒(対数)", "累積%", logx=True)
    ax.set_ylim(0, 100)
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[0, 2])
    bb = B.group_by("band").agg(pl.col(["n", "n_fill", "n_rf_fill", "n_canc",
                                        "n_rf_canc"]).sum()).sort("band")
    bb = bb.filter(pl.col("band") >= 0)
    yy = np.arange(bb.height)
    rf = bb["n_rf_fill"].to_numpy() / np.maximum(bb["n_fill"].to_numpy(), 1)
    rc = bb["n_rf_canc"].to_numpy() / np.maximum(bb["n_canc"].to_numpy(), 1)
    ax.barh(yy - 0.2, rf * 100, height=0.38, color=FILLC, label="約定の後",
            zorder=3)
    ax.barh(yy + 0.2, rc * 100, height=0.38, color=CANCC, label="取消の後",
            zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([D_LAB[int(b)] for b in bb["band"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("③ 距離帯を揃えても向きは変わらない", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "1 秒以内に出し直した割合(%)", "最良からの距離(ティック)")
    legend(ax, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    v = hist(H, "nchild")
    lab = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51-100", "101+"]
    ax.bar(np.arange(len(v)), v / v.sum() * 100, color=BASELINE, zorder=3)
    ax.set_xticks(np.arange(len(v)))
    ax.set_xticklabels(lab[: len(v)], fontsize=8)
    ax.set_title("④ 一続きの補充に含まれる子注文の数", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "子注文の数", "%")
    ax.text(0.55, 0.9, f"1 本で終わる {v[0]/v.sum():.0%}\n"
            f"3 本以上 {v[2:].sum()/v.sum():.1%}",
            transform=ax.transAxes, va="top", fontsize=8.6, color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, D["ep_child_mean"], color=INK, lw=1.8, zorder=3)
    ax.set_title("⑤ 1 つの補充あたりの平均子注文数(日次)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日", "本")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, D["rf_fill_same_sz"] * 100, color=FILLC, lw=1.8,
            label="約定の後", zorder=3)
    ax.plot(x, D["rf_canc_same_sz"] * 100, color=CANCC, lw=1.8,
            label="取消の後", zorder=3)
    ax.set_title("⑥ 出し直しの数量が前と**同じ**だった割合", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日", "%")
    legend(ax, loc="best")

    fig.text(0.055, 0.972, "補充の形 — 削られたから足すのか、値段を付け直すのか",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             f"{D.height} 日 / 指値 {int(D['n_ord'].sum()):,} 本。iceberg なら"
             "「約定の後」のほうが速く・多く出し直すはず。取消の後を対照に置く。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_ice_refill.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 2: 隠れ数量と回転 =================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.26,
                          left=0.055, right=0.975, top=0.845, bottom=0.085)
    hctr = np.sqrt(HID_EDGES[:-1] * HID_EDGES[1:])

    ax = fig.add_subplot(gs[0, 0])
    for k, cc, lab in (("hid_max", INK, "すべての補充"),
                       ("hid_max_multi", WARN, "子が 3 本以上のものだけ")):
        v = hist(H, k)
        if v.sum() == 0:
            continue
        c = np.cumsum(v) / v.sum()
        ax.plot(hctr, c[1:len(hctr) + 1] * 100, color=cc, lw=2.0, label=lab,
                zorder=3)
    ax.axvline(1.0, color=FILLC, ls=(0, (4, 2)), lw=1.4, zorder=4)
    ax.set_title("① 隠れ数量の代理 — 約定量 ÷ 見せた最大の数量", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "倍(対数)", "累積%", logx=True)
    ax.set_ylim(0, 100)
    legend(ax, loc="upper left")
    ax.text(0.55, 0.12, "1 を超える = 見せた以上を約定させた",
            transform=ax.transAxes, fontsize=8.4, color=INK2)

    ax = fig.add_subplot(gs[0, 1])
    v = hist(H, "replen")
    if v.sum():
        ax.plot(hctr, (np.cumsum(v) / v.sum())[1:len(hctr) + 1] * 100,
                color=INK, lw=2.0, zorder=3)
    ax.axvline(1.0, color=FILLC, ls=(0, (4, 2)), lw=1.4, zorder=4)
    ax.set_title("② 補充した数量 ÷ 見せた最大の数量", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "倍(対数)", "累積%", logx=True)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[0, 2])
    to = (bb["n"].to_numpy() * 0 + 1)  # placeholder, 実値は下で入れ替える
    bt = B.group_by("band").agg(sz_pl=pl.col("sz_pl").sum(),
                                szt=pl.col("szt").sum()).sort("band")
    bt = bt.filter(pl.col("band") >= 0)
    tv = bt["sz_pl"].to_numpy() / np.maximum(bt["szt"].to_numpy() / 1e9, 1)
    yy = np.arange(bt.height)
    ax.barh(yy, tv, color=BASELINE, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([D_LAB[int(b)] for b in bt["band"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("③ 価格水準の回転(毎秒)", color=INK, fontsize=10.5, pad=7,
                 loc="left")
    style(ax, "毎秒(= 平均表示時間の逆数、対数)", "", logx=True)
    for i, t_ in enumerate(tv):
        if np.isfinite(t_) and t_ > 0:
            ax.text(t_ * 1.1, i, f"{1/t_:.1f} 秒", va="center", fontsize=7.4,
                    color=INK2)

    ax = fig.add_subplot(gs[1, 0])
    v = g["modal_share"].to_numpy().astype(float)
    v = np.sort(v[np.isfinite(v)])
    ax.plot(v, np.arange(len(v)) / max(len(v), 1) * 100, color=INK, lw=2.0,
            zorder=3)
    ax.set_title("④ 最頻の数量が占める割合(口座ごと)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "その口座の注文のうち最頻数量の割合", "累積%(口座)")
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[1, 1])
    v = g["turnover_s"].to_numpy().astype(float)
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    ax.plot(v, np.arange(len(v)) / max(len(v), 1) * 100, color=INK, lw=2.0,
            zorder=3)
    ax.set_title("⑤ 口座ごとの回転(毎秒)", color=INK, fontsize=10.5, pad=7,
                 loc="left")
    style(ax, "毎秒(対数)", "累積%(口座)", logx=True)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[1, 2])
    rf = g["rf_fill"].to_numpy().astype(float)
    rc = g["rf_canc"].to_numpy().astype(float)
    m = np.isfinite(rf) & np.isfinite(rc)
    ax.plot(rc[m] * 100, rf[m] * 100, "o", ms=5, color=MUTED, alpha=0.7,
            zorder=3)
    lim = [0, max(np.nanmax(rc[m]), np.nanmax(rf[m])) * 100 * 1.05]
    ax.plot(lim, lim, "-", color=INK, lw=1.2, zorder=4)
    ax.set_title("⑥ 口座ごと: 約定後 対 取消後(対角線より上なら iceberg 寄り)",
                 color=INK, fontsize=10.5, pad=7, loc="left")
    style(ax, "取消の後 1 秒以内に出し直し(%)", "約定の後(%)")
    n_ice = int((rf[m] > rc[m]).sum())
    ax.text(0.03, 0.95, f"対角線より上 {n_ice} / {int(m.sum())} 口座",
            transform=ax.transAxes, va="top", fontsize=8.6, color=INK2)

    fig.text(0.055, 0.972, "隠れ数量の代理と回転 — 見せた以上を約定させているか",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             f"口座は総注文 {MIN_ORD:,} 本以上の {g.height} 者。"
             "回転は Σ(出した数量) ÷ Σ(数量 × 表示時間)で、"
             "数量加重の平均表示時間の逆数にあたる。", fontsize=10, color=INK2,
             va="top")
    p = out / f"{tag}_ice_hidden.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)
    print(f"-> data/ice_wallet_summary_{tag}.csv({g.height} 口座)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
