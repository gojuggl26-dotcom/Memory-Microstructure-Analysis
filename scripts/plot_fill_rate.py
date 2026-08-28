"""ティック水準別・キュー位置別の約定率を図示する。

図は 5 枚。上段の 3 枚は「約定率の形」、下段は「2 条件の同時の効き」と
「出来高との比例性」。

    A 水準別      置いた価格が最良気配から何ティック離れているか
    B キュー位置別 置いた瞬間に同じ価格へ既に何本並んでいたか
    C 出来高三分位 ★形が変わるか(水準ごとの約定率を三分位で重ねる)
    D 同時分布    A×B のヒートマップ。片方だけで説明できるかを見る
    E 比例性      日次の約定率と日次出来高の両対数。傾き 1 が比例

【x が確定する時刻 / y の期間】
    x = ティック水準・キュー位置 … 発注の瞬間に確定する
    y = その注文が最終的に約定したか … 発注より後
先読みは無い。ただし E と C は**同時点**の関係であって予測ではない。
出来高三分位の境目も全標本から作っているので、これは記述であって
売買規則ではない(規則にするなら直近だけで三分位を作り直す必要がある)。

配色は dataviz の参照パレット。単一系列は series-1、順序のある 3 本は
sequential blue の 250 / 450 / 650。scripts/palette_check.py で検証済み。

    uv run python scripts/plot_fill_rate.py --coin xyz:MU
出力: charts/<coin>_fill_rate.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
TERCILE = ["#86b6ef", "#2a78d6", "#104281"]          # 順序ランプ 250 / 450 / 650
TER_LAB = ["出来高 低位 1/3", "出来高 中位 1/3", "出来高 高位 1/3"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

LV_LAB = ["改善(<0)", "0(最良)", "1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
Q_LAB = ["0(先頭)", "1", "2", "3-4", "5-9", "10-19", "20+"]
NB = 400
RNG = np.random.default_rng(20260828)


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


def comma(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def boot_rate(N: np.ndarray, F: np.ndarray):
    """日を単位に再標本して率の 95% 区間を出す。N/F は 日 × 帯 の行列。

    日ごとに独立とみなす(同じ日の注文どうしは相関するので、注文単位の
    二項区間では狭すぎる)。
    """
    nd = N.shape[0]
    out = np.empty((NB, N.shape[1]))
    for b in range(NB):
        i = RNG.integers(0, nd, nd)
        n, f = N[i].sum(0), F[i].sum(0)
        out[b] = np.where(n > 0, f / np.maximum(n, 1), np.nan)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def marginal(C: pl.DataFrame, key: str, k: int):
    """帯ごとの率・件数・区間を返す。"""
    g = C.group_by(key).agg(pl.col("n").sum(), pl.col("n_fill").sum()).sort(key)
    n = np.zeros(k)
    f = np.zeros(k)
    n[g[key].to_numpy()] = g["n"].to_numpy()
    f[g[key].to_numpy()] = g["n_fill"].to_numpy()

    days = sorted(C["dt"].unique().to_list())
    di = {d: i for i, d in enumerate(days)}
    N = np.zeros((len(days), k))
    F = np.zeros((len(days), k))
    for r in C.group_by("dt", key).agg(pl.col("n").sum(), pl.col("n_fill").sum()).to_dicts():
        N[di[r["dt"]], r[key]] = r["n"]
        F[di[r["dt"]], r[key]] = r["n_fill"]
    lo, hi = boot_rate(N, F)
    return np.where(n > 0, f / np.maximum(n, 1), np.nan), n, lo, hi


def log_ticks(vmin: float, vmax: float) -> list[float]:
    """1-2-5 の梯子から、範囲に入る目盛を選ぶ。

    対数軸の既定は 10 の冪しか置かないので、1 桁ちょっとの範囲だと
    目盛が 1 本か 2 本しか出ず読めなくなる(実際そうなった)。
    """
    t = []
    e = int(np.floor(np.log10(vmin))) - 1
    while 10.0 ** e <= vmax * 10:
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if vmin <= v <= vmax:
                t.append(v)
        e += 1
    return t


def logfmt(ax, which="y"):
    for axis, (lo, hi) in zip(
            (ax.yaxis,) if which == "y" else (ax.xaxis, ax.yaxis),
            (ax.get_ylim(),) if which == "y" else (ax.get_xlim(), ax.get_ylim())):
        axis.set_ticks(log_ticks(lo, hi))
        axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


def line_panel(ax, labels, rate, n, lo, hi, xlab, title, sub):
    style(ax)
    x = np.arange(len(labels))
    pos = (n > 0) & (rate > 0)
    zero = (n > 0) & (rate == 0)
    ax.fill_between(x[pos], lo[pos] * 100, hi[pos] * 100, color=BLUE, alpha=0.16,
                    linewidth=0, zorder=2)
    ax.plot(x[pos], rate[pos] * 100, "-o", color=BLUE, lw=2.0, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    for i in np.where(pos)[0]:
        ax.annotate(f"{rate[i] * 100:.2f}", (x[i], rate[i] * 100),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(rate[pos].min() * 100 * 0.45, rate[pos].max() * 100 * 1.9)
    # 率がちょうど 0 の帯は対数軸に載らないので、床に白抜きで置いて 0 と書く
    for i in np.where(zero)[0]:
        ax.plot([x[i]], [ax.get_ylim()[0] * 1.12], "o", mfc=SURFACE, mec=BLUE,
                mew=1.6, ms=8, zorder=4)
        ax.annotate("0", (x[i], ax.get_ylim()[0] * 1.12), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n{comma(v)}" for lb, v in zip(labels, n)],
                       fontsize=7.8, color=MUTED)
    ax.set_xlabel(xlab, color=INK2, fontsize=9.5)
    ax.set_ylabel("約定率 [%](対数)", color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=20, weight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.8)
    logfmt(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"fill_rate_cells_{tag}.parquet")
    days = sorted(C["dt"].unique().to_list())

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(16.2, 10.6), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.22], hspace=0.50, wspace=0.26,
                          left=0.055, right=0.985, top=0.850, bottom=0.065)

    # ---- A 水準別 ------------------------------------------------------------
    r_lv, n_lv, lo_lv, hi_lv = marginal(C, "lv", 9)
    line_panel(fig.add_subplot(gs[0, 0]), LV_LAB, r_lv, n_lv, lo_lv, hi_lv,
               "最良気配からの距離 [ティック] / 下は本数",
               "A  置いた価格の水準",
               "帯は日を単位に再標本した 95% 区間")

    # ---- B キュー位置別 ------------------------------------------------------
    r_q, n_q, lo_q, hi_q = marginal(C, "qb", 7)
    line_panel(fig.add_subplot(gs[0, 1]), Q_LAB, r_q, n_q, lo_q, hi_q,
               "自分より前に並んでいた本数 / 下は本数",
               "B  置いた瞬間のキュー位置",
               "同じ価格に先客が 1 本いるだけでどれだけ落ちるか")

    # ---- C 出来高三分位ごとの形 ----------------------------------------------
    V = (pl.read_csv(ROOT / "data" / f"daily_oi_volume_{tag}.csv")
           .select(dt=pl.col("d").cast(pl.String), vol=pl.col("volume"), op=pl.col("open_day")))
    D = (C.group_by("dt").agg(pl.col("n").sum(), pl.col("n_fill").sum())
          .join(V, on="dt", how="inner").sort("vol"))
    # ★三分位は立会日の中だけで作る。休場日を混ぜると「出来高が低い」が
    #   ほぼ「週末」と同義になり、出来高の効果ではなく曜日の効果を見てしまう。
    Do = D.filter(pl.col("op"))
    q1, q2 = Do["vol"].quantile(1 / 3), Do["vol"].quantile(2 / 3)
    ter = {r["dt"]: (0 if r["vol"] <= q1 else 1 if r["vol"] <= q2 else 2)
           for r in Do.to_dicts()}
    Ct = C.with_columns(ter=pl.col("dt").replace_strict(ter, default=None)).drop_nulls("ter")

    ax = fig.add_subplot(gs[0, 2])
    style(ax)
    x = np.arange(9)
    ends = []
    for t in (0, 1, 2):
        g = (Ct.filter(pl.col("ter") == t).group_by("lv")
               .agg(pl.col("n").sum(), pl.col("n_fill").sum()).sort("lv"))
        y = np.full(9, np.nan)
        y[g["lv"].to_numpy()] = g["n_fill"].to_numpy() / g["n"].to_numpy()
        nd = sum(1 for v in ter.values() if v == t)
        ax.plot(x, y * 100, "-o", color=TERCILE[t], lw=2.0, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4 + t)
        j = int(np.where(np.isfinite(y))[0][-1])
        ends.append((y[j] * 100, f"{TER_LAB[t]}({nd} 日)", TERCILE[t], j))
    # 凡例ではなく直接ラベル。薄い側は背景との対比が 2.06 と低いため
    # (dataviz の relief rule)、色だけに同定を負わせない
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 12.2)
    for yv, lb, c, j in ends:
        ax.annotate(lb, (j, yv), textcoords="offset points", xytext=(9, -3),
                    fontsize=8.3, color=c, weight="bold", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(LV_LAB, rotation=50, ha="right", fontsize=7.8, color=MUTED)
    ax.set_xlabel("最良気配からの距離 [ティック]", color=INK2, fontsize=9.5)
    ax.set_ylabel("約定率 [%](対数)", color=INK2, fontsize=9.5)
    ax.set_title("C  出来高で層別しても形は変わるか(立会日のみ)", loc="left",
                 color=INK, fontsize=11, pad=20, weight="bold")
    ax.text(0, 1.035, "休場日を混ぜると「出来高が低い日」が週末と同義になるため除く",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    logfmt(ax)

    # ---- D 同時分布(ヒートマップ)-------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    style(ax)
    ax.grid(False)
    J = C.group_by("lv", "qb").agg(pl.col("n").sum(), pl.col("n_fill").sum())
    M = np.full((9, 7), np.nan)
    NN = np.zeros((9, 7))
    for r in J.to_dicts():
        NN[r["lv"], r["qb"]] = r["n"]
        if r["n"] >= 500:
            M[r["lv"], r["qb"]] = r["n_fill"] / r["n"]
    v = M[np.isfinite(M) & (M > 0)]
    lo_e, hi_e = np.log10(v.min()), np.log10(v.max())
    for i in range(9):
        for j in range(7):
            if not np.isfinite(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=SURFACE,
                                           edgecolor=GRID, lw=1.0))
                ax.text(j, i, "—" if NN[i, j] else "", ha="center", va="center",
                        fontsize=9, color=MUTED)
                continue
            # 率がちょうど 0 のセルは「起こりえない組み合わせ(空白)」と
            # 区別がつくよう、最も薄い色を塗って 0.00 と書く
            f = 0.0 if M[i, j] <= 0 else (np.log10(M[i, j]) - lo_e) / (hi_e - lo_e)
            k = int(round(f * (len(RAMP) - 1)))
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=RAMP[k],
                                       edgecolor=SURFACE, lw=1.6))
            ax.text(j, i, f"{M[i, j] * 100:.2f}", ha="center", va="center",
                    fontsize=8.8, color="#ffffff" if k >= 8 else INK)
    ax.set_xlim(-.5, 6.5)
    ax.set_ylim(8.5, -.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(Q_LAB, fontsize=8.8, color=MUTED)
    ax.set_yticks(range(9))
    ax.set_yticklabels(LV_LAB, fontsize=8.8, color=MUTED)
    ax.set_xlabel("置いた瞬間のキュー位置 [自分より前の本数]", color=INK2, fontsize=9.5)
    ax.set_ylabel("最良気配からの距離 [ティック]", color=INK2, fontsize=9.5)
    ax.set_title("D  2 つの条件の同時の効き(セルの数字は約定率 %)", loc="left",
                 color=INK, fontsize=11, pad=20, weight="bold")
    ax.text(0, 1.02, "色は対数目盛の濃淡。件数 500 未満のセルは伏せた(—)。"
                     "空白は起こりえない組み合わせ"
                     "(気配を改善した注文は必ずキューの先頭になり、"
                     "最良気配に置く注文の前には必ず先客がいる)",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    # ---- E 出来高との比例性 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    def elas(sub):
        lx = np.log(sub["vol"].to_numpy().astype(float))
        n = sub["n"].to_numpy().astype(float)
        f = sub["n_fill"].to_numpy().astype(float)
        return (np.polyfit(lx, np.log(n), 1)[0], np.polyfit(lx, np.log(f), 1)[0],
                np.polyfit(lx, np.log(f / n), 1))

    op = D["op"].to_numpy().astype(bool)
    dv = D["vol"].to_numpy().astype(float)
    dr = (D["n_fill"].to_numpy() / D["n"].to_numpy()).astype(float)
    e_all = elas(D)
    e_op = elas(Do)
    b1, b0 = e_op[2]
    lx = np.log(Do["vol"].to_numpy().astype(float))
    ly = np.log((Do["n_fill"] / Do["n"]).to_numpy().astype(float))
    bs = np.empty(NB)
    for b in range(NB):
        i = RNG.integers(0, len(lx), len(lx))
        bs[b] = np.polyfit(lx[i], ly[i], 1)[0]
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    # 立会日は塗り、休場日は白抜き。色だけに区別を負わせない
    ax.scatter(dv[op] / 1e3, dr[op] * 100, s=34, color=BLUE, linewidths=0.9,
               edgecolors=SURFACE, zorder=5, label=f"立会日({int(op.sum())} 日)")
    ax.scatter(dv[~op] / 1e3, dr[~op] * 100, s=34, facecolors=SURFACE,
               edgecolors=BLUE, linewidths=1.6, zorder=5,
               label=f"休場日({int((~op).sum())} 日)")
    xs = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(np.exp(xs) / 1e3, np.exp(b0 + b1 * xs) * 100, color=INK2, lw=1.8,
            ls=(0, (5, 3)), zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("その日の出来高 [千枚]", color=INK2, fontsize=9.5)
    ax.set_ylabel("その日の約定率 [%]", color=INK2, fontsize=9.5)
    ax.set_title("E  出来高が増えると約定率も上がるか", loc="left", color=INK,
                 fontsize=11, pad=20, weight="bold")
    ax.text(0, 1.02,
            f"立会日だけの傾き {b1:+.2f}(95% 区間 {blo:+.2f}〜{bhi:+.2f})/ "
            f"休場日を混ぜると {e_all[2][0]:+.2f} と符号が変わる",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    ax.legend(loc="upper right", frameon=False, fontsize=8.3, labelcolor=INK2,
              handletextpad=0.3, borderpad=0.2)
    ax.text(0, -0.30,
            f"率ではなく総数で見ると、立会日は 発注 {e_op[0]:+.2f} / 約定 {e_op[1]:+.2f}"
            "(出来高 1% 増に対する増加率)。傾き 1 なら比例。\n"
            "同時点の関係であって予測ではない。休場日が 2 日しかないので、"
            "この節だけは 10 日では判定できない",
            transform=ax.transAxes, color=MUTED, fontsize=8.0, linespacing=1.6,
            va="top")
    logfmt(ax, "both")

    tot_n, tot_f = int(C["n"].sum()), int(C["n_fill"].sum())
    fig.suptitle(f"{a.coin} 置いた指値が約定する確率 — ティック水準とキュー位置",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.055, 0.936,
             f"母集団 = 滞留する指値(Alo / Gtc。トリガー注文・テイカー・reduce_only は除く)"
             f"として置かれ、水準が決まり、観測期間内に終端した {tot_n:,} 本。"
             f"うち約定 {tot_f:,} 本 = {tot_f / tot_n:.3%}。\n"
             "どちらの条件も発注した瞬間に確定するので、置く前に判る条件で約定確率を語れる"
             "(先読みは無い)。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.005,
             f"出所: Hyperliquid L4 (Artemis) L1 注文イベント / "
             f"{days[0]}〜{days[-1]} の {len(days)} 日",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_fill_rate.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
