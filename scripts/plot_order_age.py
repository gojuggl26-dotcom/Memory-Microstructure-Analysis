"""5 つの時間帯の注文年齢の分布を描く。

年齢は 1 ナノ秒から 20 日以上まで 10 桁以上に広がるので、横軸は必ず対数にする。
左が密度(対数階級のヒストグラム)、右が「その年齢以上まで生き残る割合」
(補相補累積分布、両対数)。長生きする注文の割合は右の図で読む。

**寿命 0 の注文(同一ナノ秒で開いて閉じた注文)は対数軸に置けない。**
その割合は右の図の左端の高さ(1 − 割合)として現れ、図中にも数字で出す。

    uv run python scripts/plot_order_age.py --coin xyz:MU
    uv run python scripts/plot_order_age.py --by-coin
出力: charts/<coin>_order_age_dist.png, charts/order_age_by_coin.png
"""

from __future__ import annotations

import argparse
import datetime as dtm
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from build_order_age import lifecycle_dir, sessions

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# 検証済みパレットの 1〜5 番。palette_check.validate で隣接ペア全通過を確認済み
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
WINDOWS = ["開場前 1 時間", "開場後 1 時間", "昼 12 時から 1 時間",
           "閉場前 1 時間", "夜 12 時から 1 時間"]
CLOCK = {
    "開場前 1 時間": "12:30–13:30 UTC / 8:30–9:30 ET",
    "開場後 1 時間": "13:30–14:30 UTC / 9:30–10:30 ET",
    "昼 12 時から 1 時間": "16:00–17:00 UTC / 12:00–13:00 ET",
    "閉場前 1 時間": "19:00–20:00 UTC / 15:00–16:00 ET",
    "夜 12 時から 1 時間": "04:00–05:00 UTC / 0:00–1:00 ET",
}
COINS = ["xyz:MU", "xyz:SNDK", "xyz:SKHX", "xyz:SMSN",
         "xyz:KIOXIA", "xyz:INTC", "xyz:AMD", "xyz:DRAM"]
LO, BPD = -9, 200                      # build_order_age.py と揃える
# 表示は 1 桁 5 階級まで粗くする。これより細かいと、下の「ブロックの刻み」で
# 述べる 66ms ごとの山が棘になって 5 本の比較が読めなくなる
COARSE = 40
MARKS = [(60, "1 分"), (3600, "1 時間"), (86400, "1 日")]


def setup():
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)


def plain(v, _):
    return f"{v:g}"


def pct(v, _):
    return f"{v * 100:g}%"


def n_sessions(coin: str) -> tuple[int, str, str]:
    """注文が 1 行でもある立会日の数。

    「区画がある」ではなく「行がある」で数える。xyz:KIOXIA は上場が
    2026-06-25 で、それ以前の区画は空のまま置いてある。区画の数で数えると
    32 日の銘柄を 68 日と書いてしまう。
    """
    d = lifecycle_dir(coin)
    days = sorted(p.name[3:] for p in d.glob("dt=*") if (p / "_SUCCESS").exists())
    keep, _ = sessions(days)
    live = [x.name[3:] for x in sorted(d.glob("dt=*"))
            if dtm.date.fromisoformat(x.name[3:]) in keep and any(
                pq.ParquetFile(str(f)).metadata.num_rows
                for f in x.glob("part-*.parquet"))]
    return len(live), live[0], live[-1]


def load(tag: str):
    H = pl.read_csv(ROOT / "data" / f"order_age_hist_{tag}.csv")
    S = pl.read_csv(ROOT / "data" / f"order_age_stats_{tag}.csv")
    return H, S


def series(H: pl.DataFrame, w: str, n: int, n_zero: int):
    """疎な度数表を密な配列に戻し、密度と補相補累積分布を作る。"""
    g = H.filter(pl.col("window") == w)
    idx = np.rint((np.log10(g["lo"].to_numpy()) - LO) * BPD).astype(int)
    nb = idx.max() + 1
    nb += (-nb) % COARSE                       # 粗くする単位で割り切れるように
    full = np.zeros(nb, dtype=np.int64)
    full[idx] = g["count"].to_numpy()
    edges = 10.0 ** (LO + np.arange(nb + 1) / BPD)

    # 密度: 表示用に COARSE 個ずつまとめる。桁あたりの割合にするので階級幅で割る
    c = full.reshape(-1, COARSE).sum(axis=1).astype(float)
    e = edges[::COARSE]
    mid = np.sqrt(e[:-1] * e[1:])
    dens = c / n / (np.log10(e[1:]) - np.log10(e[:-1]))
    # 補相補累積分布は細かいまま。寿命 0 の塊は左端の下駄として効く
    ccdf = 1.0 - (n_zero + np.cumsum(full)) / n
    return mid, dens, edges[1:], ccdf


def dist_chart(coin: str) -> None:
    tag = coin.replace(":", "_")
    H, S = load(tag)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.0, 6.6), dpi=170)

    total = 0
    for w, col in zip(WINDOWS, COLORS):
        r = S.filter(pl.col("window") == w).to_dicts()[0]
        n, nz = int(r["n"]), int(round(r["share_zero"] * r["n"]))
        total += n
        mid, dens, hi, ccdf = series(H, w, n, nz)
        lab = f"{w}({CLOCK[w].split(' / ')[0]})"
        a1.plot(mid, dens, color=col, lw=2.0, zorder=3, label=lab,
                solid_capstyle="round")
        a2.plot(hi, np.clip(ccdf, 1e-7, None), color=col, lw=2.0, zorder=3, label=lab)
        a1.axvline(r["p50"], color=col, lw=1.0, ls=(0, (3, 3)), alpha=0.7, zorder=2)

    for ax in (a1, a2):
        style(ax)
        ax.set_xscale("log")
        ax.set_xlim(3e-3, 3e6)
        ax.set_xticks([1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("注文年齢(秒)", color=INK2, fontsize=10.5)
        for v, name in MARKS:
            ax.axvline(v, color=BASELINE, lw=0.9, zorder=1)
            ax.annotate(name, (v, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -11), textcoords="offset points",
                        color=MUTED, fontsize=8.5)

    a1.set_ylabel("密度(桁あたりの割合)", color=INK2, fontsize=10.5)
    a1.set_ylim(bottom=0)
    a2.set_yscale("log")
    a2.set_ylabel("その年齢以上まで生き残る割合", color=INK2, fontsize=10.5)
    a2.set_ylim(1e-6, 1.2)
    a2.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    a2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(pct))
    a2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    a1.set_title("分布のかたち", loc="left", color=INK, fontsize=11.5, pad=8,
                 weight="bold")
    a2.set_title("長生きする割合(両対数)", loc="left", color=INK, fontsize=11.5,
                 pad=8, weight="bold")
    a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), frameon=False,
              fontsize=9, labelcolor=INK2)

    z = S.filter(pl.col("window").is_in(WINDOWS))
    zmin, zmax = z["share_zero"].min(), z["share_zero"].max()
    ns, d0, d1 = n_sessions(coin)
    fig.suptitle(f"{coin} 時間帯ごとの注文年齢の分布(立会日 {ns} 日)",
                 x=0.005, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.005, 0.925,
             "板に置かれた指値注文(Alo / Gtc)が消えるまでの時間。破線は各時間帯の中央値。"
             f"寿命 0(同一ナノ秒で開いて閉じた注文)が {zmin * 100:.1f}〜{zmax * 100:.1f}% あり、"
             "対数軸に置けないぶん右図の左端が 100% から下がっている。",
             color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses → L2 lifecycle / "
             f"窓 2026-05-04〜08-10 のうち NYSE 立会日 {n_sessions(coin)} 日 /  5 つの時間帯の注文 {total:,} 本",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.855, bottom=0.115, wspace=0.20)
    out = ROOT / "charts" / f"{tag}_order_age_dist.png"
    fig.savefig(out)
    print(f"[chart] {out}")


def by_coin_chart() -> None:
    """銘柄ごとに、5 つの時間帯の中央値を 1 本の行に並べる。"""
    rows = []
    for c in COINS:
        p = ROOT / "data" / f"order_age_stats_{c.replace(':', '_')}.csv"
        if not p.exists():
            continue
        S = pl.read_csv(p)
        for w in WINDOWS:
            g = S.filter(pl.col("window") == w)
            if g.height:
                r = g.to_dicts()[0]
                rows.append({"coin": c, "window": w, "p50": r["p50"],
                             "p25": r["p25"], "p75": r["p75"], "n": r["n"]})
    D = pl.DataFrame(rows)
    coins = [c for c in COINS if c in set(D["coin"])]
    # 昼の中央値が長い順に並べる。並べ替えの基準を結果から選ばないよう軸は固定
    order = (D.filter(pl.col("window") == "開場後 1 時間")
             .sort("p50", descending=True)["coin"].to_list())
    coins = order + [c for c in coins if c not in order]

    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=170)
    style(ax)
    for yi, c in enumerate(coins):
        g = D.filter(pl.col("coin") == c)
        lo = g["p25"].min(); hi = g["p75"].max()
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=6, zorder=1,
                solid_capstyle="round")
        for w, col in zip(WINDOWS, COLORS):
            h = g.filter(pl.col("window") == w)
            if h.height:
                ax.plot([h["p50"][0]], [yi], "o", color=col, ms=9, zorder=3,
                        mec=SURFACE, mew=2.0,
                        label=f"{w}({CLOCK[w].split(' / ')[0]})" if yi == 0 else None)
    ax.set_yticks(range(len(coins)))
    ax.set_yticklabels(coins, color=INK2, fontsize=10)
    ax.set_ylim(-0.6, len(coins) - 0.4)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.08, 12)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("注文年齢の中央値(秒)", color=INK2, fontsize=10.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK2, handletextpad=0.2, columnspacing=1.6)

    NS = {c: n_sessions(c)[0] for c in coins}
    ns = sorted(set(NS.values()))
    nstxt = str(ns[0]) if len(ns) == 1 else f"{ns[0]}〜{ns[-1]}"
    short = [f"{c} は {NS[c]} 日" for c in coins if NS[c] < max(ns)]
    fig.suptitle(f"銘柄ごと・時間帯ごとの注文年齢の中央値(立会日 {nstxt} 日)",
                 x=0.005, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.005, 0.925,
             "点は中央値。灰色の帯はその銘柄で 5 つの時間帯が動く範囲"
             "(最小の第 1 四分位から最大の第 3 四分位まで)。横軸は対数。"
             + ("  標本の短い銘柄: " + " / ".join(short) if short else ""),
             color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) → L2 lifecycle / 窓 2026-05-04〜08-10 / "
             f"板に置かれた指値注文(Alo / Gtc) {int(D['n'].sum()):,} 本",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.815, bottom=0.115)
    out = ROOT / "charts" / "order_age_by_coin.png"
    fig.savefig(out)
    print(f"[chart] {out}")


# くし型の当てはめ(Rayleigh 統計)で実測した刻み。MU / SMSN / DRAM の
# 2026-07-15 で 3 銘柄とも 67.25 ms、集中度 R=0.64〜0.67(無作為なら 0.002)。
# 銘柄によらず同じ値なので、板の作法ではなく連鎖側のブロック生成間隔である
TAU = 0.06725


def block_chart(coin: str) -> None:
    """注文年齢が 67ms ごとの山になることを、横軸を線形にして見せる。"""
    tag = coin.replace(":", "_")
    H, S = load(tag)
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=170)
    style(ax)

    for w, col in zip(WINDOWS, COLORS):
        r = S.filter(pl.col("window") == w).to_dicts()[0]
        n = int(r["n"])
        g = H.filter(pl.col("window") == w).sort("lo")
        lo = g["lo"].to_numpy(); hi = g["hi"].to_numpy()
        c = g["count"].to_numpy().astype(float)
        m = (lo > 0.0) & (hi < 0.62)
        # 線形軸なので、階級の秒幅で割って「秒あたりの割合」にする
        ax.plot((lo[m] + hi[m]) / 2, c[m] / n / (hi[m] - lo[m]),
                color=col, lw=1.6, zorder=3,
                label=f"{w}({CLOCK[w].split(' / ')[0]})")

    for k in range(1, 10):
        ax.axvline(k * TAU, color=BASELINE, lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.annotate(f"{k}", (k * TAU, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                    color=MUTED, fontsize=8.5, ha="center")

    ax.set_xlim(0, 0.62)
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v * 1000:.0f}"))
    ax.set_xlabel("注文年齢(ミリ秒)", color=INK2, fontsize=10.5)
    ax.set_ylabel("密度(秒あたりの割合)", color=INK2, fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK2)

    fig.suptitle(f"{coin} 注文年齢はブロックの刻みに乗る(立会日 {n_sessions(coin)[0]} 日)",
                 x=0.005, ha="left", color=INK, fontsize=14, weight="bold")
    fig.text(0.005, 0.925,
             f"縦の点線は {TAU * 1000:.2f} ms の整数倍(上の数字が何ブロックぶんか)。"
             "山がそこに乗るのは、注文の生死が連鎖のブロック境界で決まるため。"
             "刻みは 3 銘柄で同一値なので銘柄固有の作法ではない。",
             color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) → L2 lifecycle / 窓 2026-05-04〜08-10 / "
             "板に置かれた指値注文(Alo / Gtc)のうち年齢 0〜620ms のもの",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.115)
    out = ROOT / "charts" / f"{tag}_order_age_blocks.png"
    fig.savefig(out)
    print(f"[chart] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin")
    ap.add_argument("--by-coin", action="store_true")
    ap.add_argument("--blocks", action="store_true")
    a = ap.parse_args()
    setup()
    if a.coin:
        dist_chart(a.coin)
        if a.blocks:
            block_chart(a.coin)
    if a.by_coin:
        by_coin_chart()
    if not a.coin and not a.by_coin:
        raise SystemExit("--coin か --by-coin を指定すること")


if __name__ == "__main__":
    main()
