"""200ms 格子で測った OBI / OFI の自己相関を図示する。

    A OBI の自己相関(ラグ 200ms〜5 分)
    B OFI の自己相関(同じラグ。★縦軸の目盛は A の 20 分の 1)
    C ★ A の正体 — 空の格子点の割合と 1 ラグ自己相関がほぼ一直線に乗る
    D OFI の 1 ラグ自己相関は日によって符号が変わる

A と B は量が 30 倍違うので**縦軸を共有していない**。同じ軸に載せると B が
つぶれ、別軸を 1 枚に重ねるのは禁じ手(二重軸)なので、2 枚に分けて
目盛の違いを見出しに書いてある。

配色は dataviz の参照パレット slot 1 / 2(青・橙)。palette_check で検証済み:
CVD ΔE 24.7(目標 8)、通常視 ΔE 33.6(下限 15)、対比 4.30 / 3.12。

    uv run python scripts/plot_acf_200ms.py --coin xyz:MU
出力: charts/<coin>_acf_200ms.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COL = {"立会日": "#2a78d6", "閉場日": "#eb6834"}


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


def log_ticks(vmin, vmax):
    t, e = [], int(np.floor(np.log10(vmin))) - 1
    while 10.0 ** e <= vmax * 10:
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if vmin <= v <= vmax:
                t.append(v)
        e += 1
    return t


def acf_panel(ax, A, var, title, sub, ylim, legend=False):
    style(ax)
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    for dty, c in COL.items():
        s = A.filter((pl.col("var") == var) & (pl.col("day_type") == dty)).sort("lag")
        if s.height == 0:
            continue
        x = s["lag_ms"].to_numpy() / 1000.0
        ax.fill_between(x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=c,
                        alpha=0.18, linewidth=0, zorder=3)
        ax.plot(x, s["rho"].to_numpy(), color=c, lw=2.0, zorder=4,
                label=f"{dty}({s['n_days'][0]} 日)")
        j = -1
        ax.annotate(f"{dty}", (x[j], s["rho"].to_numpy()[j]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8.5,
                    color=c, weight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.16, 900)
    ax.set_ylim(*ylim)
    ax.set_xticks(log_ticks(0.2, 300))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("ラグ [秒](対数)", color=INK2, fontsize=9.5)
    ax.set_ylabel("自己相関", color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=8.8)
    if legend:
        ax.legend(loc="upper right", frameon=False, fontsize=8.8, labelcolor=INK2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    A = pl.read_parquet(ROOT / "data" / f"acf_200ms_{tag}.parquet")
    L = pl.read_csv(ROOT / "data" / f"acf_200ms_daily_lag1_{tag}.csv").sort("dt")
    days = L["dt"].to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    L = L.with_columns(day_type=pl.col("dt").map_elements(
        lambda d: "立会日" if d in sess else "閉場日", return_dtype=pl.String))

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(14.4, 10.2), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.22,
                          left=0.062, right=0.975, top=0.845, bottom=0.075)

    n_ofi = float(A.filter(pl.col("var") == "OFI")["rho_null"].abs().max())
    n_obi = float(A.filter(pl.col("var") == "OBI")["rho_null"].abs().max())

    acf_panel(fig.add_subplot(gs[0, 0]), A, "OBI",
              "A  OBI(板の残高の偏り)の自己相関",
              f"帯は日単位ブートストラップの 95% 区間 / 帰無対照(日内並べ替え)は最大 {n_obi:.4f}",
              (-0.03, 0.98), legend=True)
    acf_panel(fig.add_subplot(gs[0, 1]), A, "OFI",
              "B  OFI(板の流量の偏り)の自己相関",
              f"★縦軸の目盛は A の 20 分の 1 / 帰無対照は最大 {n_ofi:.4f}",
              (-0.006, 0.062))

    # ---- C 空格子点割合 vs OBI の 1 ラグ自己相関 ------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    e = L["empty"].to_numpy() * 100
    b = L["OBI"].to_numpy()
    r = float(np.corrcoef(e, b)[0, 1])
    for dty, c in COL.items():
        m = (L["day_type"] == dty).to_numpy()
        ax.scatter(e[m], b[m], s=34, color=c, alpha=0.8, linewidths=0.8,
                   edgecolors=SURFACE, zorder=4, label=dty)
        ax.annotate(dty, (e[m].mean(), b[m].mean()), textcoords="offset points",
                    xytext=(0, -22), ha="center", fontsize=9, color=c, weight="bold")
    p = np.polyfit(e, b, 1)
    xs = np.linspace(e.min(), e.max(), 20)
    ax.plot(xs, np.polyval(p, xs), color=INK2, lw=1.6, ls=(0, (5, 3)), zorder=3)
    ax.set_xlabel("その日の 200ms 格子のうち最良気配が動かなかった割合 [%]",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("OBI の 1 ラグ(200ms)自己相関", color=INK2, fontsize=9.5)
    ax.set_title("C  A の高さは板の粘りではなく「何も起きない頻度」", loc="left",
                 color=INK, fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035, f"1 点 = 1 日 × {len(e)} 日 / ピアソン相関 {r:+.3f}",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    ax.text(0.03, 0.93, "空の格子点では OBI が定義上そのまま持ち越されるため、\n"
                        "更新の少ない日ほど自動的に自己相関が上がる",
            transform=ax.transAxes, color=MUTED, fontsize=8.3, va="top",
            linespacing=1.5)

    # ---- D OFI の 1 ラグ自己相関の日次推移 -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    o = L["OFI"].to_numpy()
    x = np.arange(len(o))
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
    for dty, c in COL.items():
        m = (L["day_type"] == dty).to_numpy()
        ax.scatter(x[m], o[m], s=30, color=c, alpha=0.85, linewidths=0.8,
                   edgecolors=SURFACE, zorder=4, label=dty)
    ax.axvspan(-0.5, 2.5, color=MUTED, alpha=0.13, zorder=1, linewidth=0)
    ax.text(0.055, 0.10, "動作確認に使った先頭 3 日\n(上場直後で代表性が無い)",
            transform=ax.transAxes, fontsize=8.3, color=INK2, va="bottom",
            linespacing=1.5)
    # ★日数の決め打ち(98 日想定の 0..97)だと 47 日の KIOXIA で落ちる。
    #   実際の日数から等間隔に 6 点を取る
    tick_i = sorted({int(round(x)) for x in
                     np.linspace(0, len(days) - 1, min(6, len(days)))})
    ax.set_xticks(tick_i)
    ax.set_xticklabels([days[i][5:] for i in tick_i], fontsize=8.5, color=MUTED)
    ax.set_xlabel(f"日({days[0]} 〜 {days[-1]})", color=INK2, fontsize=9.5)
    ax.set_ylabel("OFI の 1 ラグ(200ms)自己相関", color=INK2, fontsize=9.5)
    ax.set_title("D  OFI は日によって符号が変わる", loc="left", color=INK,
                 fontsize=11.5, pad=20, weight="bold")
    ax.text(0, 1.035,
            f"負の日 {int((o < 0).sum())} / {len(o)} 日 / 中央 {np.median(o):+.3f} / "
            f"範囲 {o.min():+.3f}〜{o.max():+.3f}",
            transform=ax.transAxes, color=INK2, fontsize=8.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.8, labelcolor=INK2, ncol=2)

    fig.suptitle(f"{a.coin} OBI と OFI の自己相関(200ms 格子・98 日)",
                 fontsize=13.5, y=0.972, color=INK, weight="bold")
    fig.text(0.062, 0.930,
             "イベント時刻に定義された 2 つの量を 200ms の時計に載せ直して測った。"
             "OBI は状態なので区間の最後の値、OFI は流量なので区間内の合計。\n"
             "どちらも区間の終端で確定するので、そのまま説明変数に使える。"
             "クロスした板の行は除外している。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             f"出所: Hyperliquid L4 (Artemis) l2/bbo / {days[0]}〜{days[-1]} の {len(days)} 日"
             "(立会日 67 / 閉場日 31)",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_acf_200ms.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}")


if __name__ == "__main__":
    main()
