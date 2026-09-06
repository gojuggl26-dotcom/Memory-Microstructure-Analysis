"""指値の待ち行列 21 種を描く。

図は 6 枚。11.1 が 1 枚、11.2 が 3 枚、11.3 が 2 枚。

    (1) 11.1 前にいる数量の帯ごとに、キューの本数 / 口座数 / 1 本あたり数量
    (2) 11.2 前と後ろ(本数・数量)の分布
    (3) 11.2 正規化した並び位置 3 種の分布
    (4) 11.2 ★約定した注文と取り消された注文で位置が違うか
    (5) 11.3 キューの動き 9 種の分位(活動のあった窓に限る)
    (6) 11.3 net flow と imbalance の分布

配色は scripts/palette_check.py で検証済み(ok=true、CVD ΔE 最小 20.0)。

    uv run python scripts/plot_queue.py --coin xyz:MU
出力: charts/<coin>_queue.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
A, B, C = "#2a78d6", "#e34948", "#7a4fb5"
SZ_LOT = 0.001
PX_UNIT = 0.01

DYN = ["queue_growth", "queue_depletion", "queue_turnover", "queue_arrival_rate",
       "queue_removal_rate", "queue_net_flow", "queue_imbalance", "queue_churn",
       "queue_replacement_rate"]
DYN_JP = {"queue_growth": "growth\n入った数量", "queue_depletion": "depletion\n出た数量",
          "queue_turnover": "turnover\n(入+出)/開始", "queue_arrival_rate": "arrival\n入 本/秒",
          "queue_removal_rate": "removal\n出 本/秒", "queue_net_flow": "net flow\n入−出",
          "queue_imbalance": "imbalance\n(入−出)/(入+出)", "queue_churn": "churn\n入+出",
          "queue_replacement_rate": "replacement\nmin(入,出)/開始"}


def plainlog(ax, which="both"):
    r"""★text.parse_math=False のとき、対数軸の目盛は $\mathdefault{...}$ と
    そのまま表示されてしまう。素の数字に差し替える。"""
    fm = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in ((ax.xaxis, ax.yaxis) if which == "both"
                 else (ax.xaxis,) if which == "x" else (ax.yaxis,)):
        axis.set_major_formatter(fm)
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


def style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)


def loghist(ax, v, color, label, bins=60):
    """0 を別扱いにした対数ヒストグラム。0 の割合は凡例に入れる。"""
    v = v[np.isfinite(v)]
    z = float((v <= 0).mean()) if len(v) else np.nan
    pos = v[v > 0]
    if not len(pos):
        return z
    e = np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins)
    ax.hist(pos, bins=e, histtype="step", color=color, lw=2.0,
            label=f"{label}(0 が {z*100:.0f}%)", density=True)
    ax.set_xscale("log")
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    P = pl.read_parquet(ROOT / "data" / f"queue_place_{tag}.parquet")
    T = pl.read_parquet(ROOT / "data" / f"queue_term_{tag}.parquet")
    # 動きは日が多いので systematic に間引いて読む
    fs = sorted(glob.glob(str(ROOT / "data" / f"queue_dyn_{tag}" / "dt=*.parquet")))
    # ★動きのあった窓だけが書かれている。good な窓の総数は n_good_day から復元する
    parts = [pl.read_parquet(f) for f in fs[::7]]
    D = pl.concat([d for d in parts if d.height]).filter(pl.col("q_start") > 0)
    n_all = sum(int(d["n_good_day"][0]) * 2 for d in parts if d.height)

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(16.4, 11.2), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30,
                          left=0.062, right=0.978, top=0.828, bottom=0.125)

    # ---- (1) 11.1 帯ごとのキューの姿 ----------------------------------------
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    Q = P.filter(pl.col("n") > 0)
    lo = Q["band_lo"].to_numpy(); n = Q["n"].to_numpy()
    lab = [("0" if x == 0 else f"{x:g}+") for x in lo * SZ_LOT]
    xs = np.arange(len(lo))
    ax.plot(xs, Q["sum_qlen"].to_numpy() / n, "-o", color=A, lw=2.2, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, label="キューの本数")
    ax.plot(xs, Q["sum_wallets"].to_numpy() / n, "-s", color=B, lw=2.2, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, label="口座数")
    ax.plot(xs, Q["sum_avgsz"].to_numpy() / n * SZ_LOT, "-^", color=C, lw=2.2, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, label="1 本あたり数量")
    ax.set_yscale("log"); plainlog(ax, "y")
    ax.set_xticks(xs); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("並んだ瞬間に前にいた数量(契約)", color=INK2, fontsize=9.5)
    ax.set_title("(1) 11.1 キューの大きさ — 前にいる量で見た姿", loc="left",
                 color=INK, fontsize=10.5, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    ax.text(0.98, 0.04, f"全 {n.sum():,} 本の発注", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)

    # ---- (2) 11.2 前と後ろ ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    loghist(ax, T["orders_ahead"].to_numpy().astype(float), A, "前の本数")
    loghist(ax, T["orders_behind"].to_numpy().astype(float), B, "後ろの本数")
    plainlog(ax, "x")
    ax.set_xlabel("本数(対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("密度", color=INK2, fontsize=9.5)
    ax.set_title("(2) 11.2 消えた瞬間の前と後ろ(本数)", loc="left", color=INK,
                 fontsize=10.5, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

    # ---- (3) 11.2 正規化した位置 --------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); style(ax)
    for c, col, nm in (("relative_queue_position", A, "relative position\n(自分を含む数量)"),
                       ("queue_percentile", B, "percentile\n(自分を除く数量)"),
                       ("normalized_queue_rank", C, "normalized rank\n(本数)")):
        v = T[c].to_numpy(); v = v[np.isfinite(v)]
        ax.hist(v, bins=np.linspace(0, 1, 41), histtype="step", color=col, lw=2.0,
                density=True, label=f"{nm}  n={len(v):,}")
    ax.set_xlabel("0 = 先頭、1 = 最後尾", color=INK2, fontsize=9.5)
    ax.set_ylabel("密度", color=INK2, fontsize=9.5)
    ax.set_title("(3) 11.2 正規化した並び位置 3 種", loc="left", color=INK,
                 fontsize=10.5, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper center")

    # ---- (4) 11.2 約定 と 取消 で位置が違うか --------------------------------
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    cols = ["relative_queue_position", "queue_percentile", "normalized_queue_rank"]
    w = 0.34
    for off, (flag, col, nm) in enumerate(((0, A, "約定して消えた"), (1, B, "取り消された"))):
        sub = T.filter(pl.col("canceled") == flag)
        for i, c in enumerate(cols):
            v = sub[c].to_numpy(); v = v[np.isfinite(v)]
            if len(v) < 50:
                continue
            q = np.quantile(v, qs)
            x = i + (off - 0.5) * w
            ax.plot([x, x], [q[0], q[4]], "-", color=BASELINE, lw=1.4, zorder=2)
            ax.plot([x, x], [q[1], q[3]], "-", color=col, lw=7, zorder=3,
                    solid_capstyle="butt", alpha=0.85)
            ax.plot(x, q[2], "o", color=SURFACE, ms=6.5, zorder=5,
                    markeredgecolor=INK, markeredgewidth=1.3)
            if i == 0:
                ax.plot([], [], "-", color=col, lw=7, alpha=0.85, label=nm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["relative\nposition", "percentile", "normalized\nrank"],
                       fontsize=8.5)
    ax.set_ylabel("並び位置(0 = 先頭)", color=INK2, fontsize=9.5)
    ax.set_title("(4) 11.2 ★約定した注文は前にいたか", loc="left", color=INK,
                 fontsize=10.5, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
              ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    ax.set_ylim(-0.06, 1.16)
    ax.text(0.02, 0.99, "太い帯 = 四分位、細い線 = p10–p90、白丸 = 中央値",
            transform=ax.transAxes, va="top", fontsize=8, color=INK2)

    # ---- (5) 11.3 動き 9 種の分位 -------------------------------------------
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    act = D          # 書かれている窓はすべて動いた窓
    xs = np.arange(len(DYN))
    for i, c in enumerate(DYN):
        v = act[c].to_numpy(); v = v[np.isfinite(v)]
        if not len(v):
            continue
        q = np.quantile(v, qs)
        ax.plot([i, i], [q[0], q[4]], "-", color=BASELINE, lw=1.4, zorder=2)
        ax.plot([i, i], [q[1], q[3]], "-", color=A, lw=7, zorder=3,
                solid_capstyle="butt", alpha=0.85)
        ax.plot(i, q[2], "o", color=SURFACE, ms=6.5, zorder=5,
                markeredgecolor=INK, markeredgewidth=1.3)
    ax.set_yscale("symlog", linthresh=0.01); plainlog(ax, "y")
    ax.set_xticks(xs)
    ax.set_xticklabels([DYN_JP[c] for c in DYN], rotation=38, ha="right",
                       fontsize=7, linespacing=1.35)
    ax.set_title("(5) 11.3 キューの動き 9 種(動いた窓のみ)", loc="left",
                 color=INK, fontsize=10.5, pad=8, weight="bold")
    ax.text(0.98, 0.04,
            f"動いた窓 {act.height:,}(気配のある窓の {act.height/n_all*100:.1f}%)",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)

    # ---- (6) 11.3 imbalance と replacement ----------------------------------
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    v = act["queue_imbalance"].to_numpy(); v = v[np.isfinite(v)]
    ax.hist(v, bins=np.linspace(-1, 1, 61), histtype="step", color=A, lw=2.0,
            density=True, label=f"imbalance  n={len(v):,}")
    both = act.filter((pl.col("queue_growth") > 0) & (pl.col("queue_depletion") > 0))
    vb = both["queue_imbalance"].to_numpy(); vb = vb[np.isfinite(vb)]
    if len(vb):
        ax.hist(vb, bins=np.linspace(-1, 1, 61), histtype="step", color=B, lw=2.0,
                density=True, label=f"入と出が両方あった窓のみ  n={len(vb):,}")
    ax.set_xlabel("queue imbalance", color=INK2, fontsize=9.5)
    ax.set_ylabel("密度", color=INK2, fontsize=9.5)
    ax.set_title("(6) 11.3 imbalance は ±1 に張り付く", loc="left", color=INK,
                 fontsize=10.5, pad=8, weight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center")
    pm1 = float((np.abs(v) > 0.999).mean()) * 100
    ax.text(0.5, 0.60, f"|imbalance| = 1 が {pm1:.0f}%\n"
            "(その窓で入か出の片方しか起きていない)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)

    fig.suptitle(f"{a.coin} 指値の待ち行列 21 種 — 大きさ・並び位置・動き",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.062, 0.938,
             "価格水準ごとの待ち行列を FIFO で追いかけた。"
             "11.1 と 11.2 の「前」は注文が並んだ瞬間、「後ろ」は消えた瞬間に測る"
             "(FIFO なので並んだ瞬間は必ず最後尾で、後ろは定義上 0)。\n"
             "11.3 は最良気配のキューについて 100ms 窓ごと。"
             "消えた瞬間の値は走査が要るため 40 本に 1 本の標本。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.7)
    fig.text(0.5, 0.004,
             "出所: Hyperliquid L4 (Artemis) L1 注文イベント + l2/bbo / "
             "窓 2026-05-04〜08-09(98 日)",
             color=MUTED, fontsize=8, ha="center")
    out = ROOT / "charts" / f"{tag}_queue.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
