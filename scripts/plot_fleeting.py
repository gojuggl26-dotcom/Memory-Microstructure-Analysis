"""束の間の注文(fleeting order)の 8 つの指標を図示する。

要求された 8 指標を 1 枚に 1 つずつ並べる。

    ① fleeting order count      日次の本数
    ② fleeting volume           日次の数量
    ③ fleeting-liquidity ratio  FLR(τ) の曲線
    ④ bid / ask FLR             側別
    ⑤ touch FLR                 最良気配からの距離帯別
    ⑥ large-order FLR           注文数量の帯別
    ⑦ wallet FLR                口座ごとの分布と二項の帰無対照
    ⑧ 口座の集中                束の間の数量が誰から出ているか

    uv run python scripts/plot_fleeting.py --coin xyz:MU
出力: charts/<coin>_fleeting.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
UP, DOWN, WARN, GREEN, PURPLE = "#e34948", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
TAU_LAB = ["同一ブロック", "100ms", "200ms", "500ms", "1s", "2s", "5s", "10s"]
TAU_X = [0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]      # 対数軸に置く位置(秒)
HEAD = "2s"
D_LAB = ["改善 <0", "0 最良", "1-2", "3-5", "6-10", "11-25", "26-50", "51-100",
         "101-300", "301-1000", "1001+"]
S_LAB = ["-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", "100+"]
WMIN = 1000                                             # 口座別の図に載せる下限本数
SEED = 20260829


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"fleeting_cells_{tag}.parquet")
    D = pl.read_parquet(ROOT / "data" / f"fleeting_daily_{tag}.parquet")
    W = pl.read_parquet(ROOT / "data" / f"fleeting_wallet_{tag}.parquet")
    # 「大口」の閾値は build 側が先頭 5 日から測って meta に書いている。
    # 図に焼き付けず必ずここから読む(閾値を変えたときに本文とずれるため)。
    big = float(pl.read_csv(ROOT / "data" / f"fleeting_meta_{tag}.csv")["big_threshold"][0])
    days = sorted(D["dt"].unique().to_list())
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    D = D.with_columns(open_day=pl.col("dt").is_in(sorted(sess)))

    tot = D.group_by("tau", "tau_ns").agg(
        n_all=pl.col("n_all").sum(), v_all=pl.col("v_all").sum(),
        n_f=pl.col("n_f").sum(), v_f=pl.col("v_f").sum(),
        n_big=pl.col("n_big").sum(), v_big=pl.col("v_big").sum(),
        n_big_f=pl.col("n_big_f").sum(), v_big_f=pl.col("v_big_f").sum()).sort("tau_ns")
    N_ALL = float(tot["n_all"][0])
    V_ALL = float(tot["v_all"][0])

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(17.6, 9.6), dpi=160)
    gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.30,
                          left=0.05, right=0.985, top=0.795, bottom=0.085)

    dh = D.filter(pl.col("tau") == HEAD).sort("dt")
    x = np.arange(dh.height)
    op = dh["open_day"].to_numpy()

    # ---- ① 本数 / ② 数量 ------------------------------------------------------
    for k, (col, ttl, unit, scale) in enumerate([
            ("n_f", "① fleeting order count", "本 / 日", 1e-6),
            ("v_f", "② fleeting volume", "契約 / 日", 1e-6)]):
        ax = fig.add_subplot(gs[0, k])
        v = dh[col].to_numpy() * scale
        ax.bar(x[op], v[op], color=DOWN, width=0.9, label="立会日")
        ax.bar(x[~op], v[~op], color=WARN, width=0.9, label="閉場日")
        ax.set_xticks([0, dh.height // 2, dh.height - 1])
        ax.set_xticklabels([days[0][5:], days[dh.height // 2][5:], days[-1][5:]])
        ax.set_ylabel(f"百万{unit}", color=INK2, fontsize=9.5)
        tt = float(dh[col].sum())
        ax.set_title(f"{ttl}\n2 秒以内に取り消された指値。合計 {tt/1e6:,.1f} 百万{unit[:1]}",
                     loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
        style(ax)

    # ---- ③ FLR(τ) --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    fv = (tot["v_f"] / tot["v_all"]).to_numpy() * 100
    fn = (tot["n_f"] / tot["n_all"]).to_numpy() * 100
    ax.plot(TAU_X, fv, "o-", color=DOWN, lw=2.2, ms=5, label="数量ベース",
            markeredgecolor=SURFACE, markeredgewidth=1.1)
    ax.plot(TAU_X, fn, "o-", color=GREEN, lw=2.2, ms=5, label="本数ベース",
            markeredgecolor=SURFACE, markeredgewidth=1.1)
    ax.axvline(2.0, color=UP, lw=1.4, ls="--")
    ax.text(2.3, 92, "見出しに使う 2 秒", color=UP, fontsize=8.5, weight="bold")
    ax.set_xscale("log")
    ax.set_xticks(TAU_X)
    ax.set_xticklabels(TAU_LAB, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("FLR[%]", color=INK2, fontsize=9.5)
    i2 = TAU_LAB.index(HEAD)
    ax.set_title("③ fleeting-liquidity ratio\n"
                 f"2 秒で数量の {fv[i2]:.1f}% / 本数の {fn[i2]:.1f}%。分母は消えた指値の全量",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    # ---- ④ bid / ask -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    sd = (C.group_by("tau", "tau_ns", "side")
          .agg(v_all=pl.col("v_all").sum(), v_f=pl.col("v_f").sum())
          .with_columns(flr=pl.col("v_f") / pl.col("v_all") * 100).sort("tau_ns"))
    fb = sd.filter(pl.col("side") == "買い").sort("tau_ns")["flr"].to_numpy()
    fa = sd.filter(pl.col("side") == "売り").sort("tau_ns")["flr"].to_numpy()
    ax.plot(TAU_X, fb, "-", color=DOWN, lw=3.4, label="買い(bid)")
    ax.plot(TAU_X, fa, "--", color=UP, lw=2.0, label="売り(ask)")
    axd = ax.twinx()                       # ★2 本はほぼ重なる。差を別軸で出す
    axd.bar(TAU_X, fb - fa, color=NEUTRAL, width=np.array(TAU_X) * 0.5, zorder=0,
            edgecolor=BASELINE, lw=0.6)
    axd.set_ylabel("買い − 売り[ポイント]", color=MUTED, fontsize=8.5)
    axd.set_ylim(-1.5, 1.5)
    axd.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        axd.spines[sp].set_visible(False)
    axd.spines["right"].set_color(BASELINE)
    ax.set_zorder(axd.get_zorder() + 1)
    ax.patch.set_visible(False)
    i2h = TAU_LAB.index(HEAD)
    b2, a2 = float(fb[i2h]), float(fa[i2h])
    ax.set_xscale("log")
    ax.set_xticks(TAU_X)
    ax.set_xticklabels(TAU_LAB, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("FLR[%]", color=INK2, fontsize=9.5)
    ax.set_title(f"④ bid FLR / ask FLR — ほぼ左右対称\n"
                 f"2 秒で買い {b2:.1f}% / 売り {a2:.1f}%(差 {b2-a2:+.2f} ポイント。灰色の棒)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    # ---- ⑤ touch FLR(距離帯別)-------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    dd = (C.filter(pl.col("tau") == HEAD).group_by("dist")
          .agg(v_all=pl.col("v_all").sum(), v_f=pl.col("v_f").sum()))
    lab = [d for d in D_LAB if d in dd["dist"].to_list()]
    vv = [float(dd.filter(pl.col("dist") == d)["v_f"][0]
                / dd.filter(pl.col("dist") == d)["v_all"][0]) * 100 for d in lab]
    sh = [float(dd.filter(pl.col("dist") == d)["v_all"][0]) / V_ALL * 100 for d in lab]
    xs = np.arange(len(lab))
    cols = [PURPLE if d in ("改善 <0", "0 最良") else DOWN for d in lab]
    ax.bar(xs, vv, color=cols, width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("FLR[%]", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(xs, sh, "o--", color=MUTED, lw=1.2, ms=3.5)
    ax2.set_ylabel("その帯が占める数量の割合[%]", color=MUTED, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    tv = sum(float(dd.filter(pl.col("dist") == d)["v_f"][0]) for d in lab
             if d in ("改善 <0", "0 最良"))
    ta = sum(float(dd.filter(pl.col("dist") == d)["v_all"][0]) for d in lab
             if d in ("改善 <0", "0 最良"))
    ax.set_title(f"⑤ touch FLR\n紫 = 最良かそれより内側。まとめて {tv/ta*100:.1f}%"
                 f"(数量の {ta/V_ALL*100:.1f}% を占める)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑥ large-order FLR(数量帯別)------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ss = (C.filter(pl.col("tau") == HEAD).group_by("size")
          .agg(v_all=pl.col("v_all").sum(), v_f=pl.col("v_f").sum()))
    lab = [d for d in S_LAB if d in ss["size"].to_list()]
    vv = [float(ss.filter(pl.col("size") == d)["v_f"][0]
                / ss.filter(pl.col("size") == d)["v_all"][0]) * 100 for d in lab]
    sh = [float(ss.filter(pl.col("size") == d)["v_all"][0]) / V_ALL * 100 for d in lab]
    xs = np.arange(len(lab))
    ax.bar(xs, vv, color=[PURPLE if d == "100+" else DOWN for d in lab], width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("FLR[%]", color=INK2, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(xs, sh, "o--", color=MUTED, lw=1.2, ms=3.5)
    ax2.set_ylabel("その帯が占める数量の割合[%]", color=MUTED, fontsize=8.5)
    ax2.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    r = tot.filter(pl.col("tau") == HEAD)
    lg = float(r["v_big_f"][0] / r["v_big"][0]) * 100
    ax.set_title(f"⑥ large-order FLR\n上位 1%({big:.1f} 契約以上)は {lg:.1f}%。"
                 "紫 = 100 契約以上の帯",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑦ wallet FLR ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    w = W.filter(pl.col("n_all") >= WMIN)
    # ★帰無対照は本数の二項分布なので、実測も本数ベースで測る。数量ベースの比率に
    #   本数の二項を当てると、大きさの違う注文を 1 本と数えたことになり比較にならない。
    p = (w["n_f"] / w["n_all"]).to_numpy() * 100
    base = float(W["n_f"].sum() / W["n_all"].sum())
    rng = np.random.default_rng(SEED)
    nn = w["n_all"].to_numpy()
    null = rng.binomial(nn.astype(np.int64), base) / nn * 100
    bins = np.linspace(0, 100, 41)
    ax.hist(p, bins=bins, color=DOWN, alpha=0.85, label=f"実測({w.height} 口座)")
    ax.hist(null, bins=bins, color=MUTED, alpha=0.55, label="帰無対照(二項)")
    ax.axvline(base * 100, color=UP, lw=1.6, ls="--")
    ax.text(base * 100 - 3, 300, f"全体 {base*100:.1f}%", ha="right",
            color=UP, fontsize=8.5, weight="bold")
    ax.set_yscale("log")
    # ★text.parse_math=False だと対数軸の既定の目盛が mathtext のまま出る
    ax.set_yticks([1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("その口座の FLR[%](本数ベース)", color=INK2, fontsize=9.5)
    ax.set_ylabel("口座数(対数)", color=INK2, fontsize=9.5)
    ax.set_title(f"⑦ wallet FLR\n{WMIN:,} 本以上出した口座。四分位 "
                 f"{np.percentile(p,25):.0f} / {np.percentile(p,50):.0f} / "
                 f"{np.percentile(p,75):.0f}%",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    style(ax)

    # ---- ⑧ 口座の集中 -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    for col, cl, lab2 in [("v_f", DOWN, "束の間の数量"), ("v_all", MUTED, "出した数量すべて")]:
        v = np.sort(W[col].to_numpy())[::-1]
        c = np.cumsum(v) / v.sum() * 100
        ax.plot(np.arange(1, len(c) + 1), c, "-", color=cl, lw=2.0, label=lab2)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("口座の順位(数量の多い順)", color=INK2, fontsize=9.5)
    ax.set_ylabel("累積シェア[%]", color=INK2, fontsize=9.5)
    v = np.sort(W["v_f"].to_numpy())[::-1]
    c = np.cumsum(v) / v.sum() * 100
    top10 = c[min(9, len(c) - 1)]
    ax.axhline(top10, color=UP, lw=1.2, ls=":")
    ax.set_title(f"⑧ 束の間の数量は誰から出ているか\n上位 10 口座で {top10:.1f}%"
                 f"(口座は全部で {W.height} 者)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    style(ax)

    fig.suptitle(f"{a.coin} 束の間の注文(fleeting order)は板の何割を占めるか"
                 f"({len(days)} 日・指値 {N_ALL/1e6:,.0f} 百万本)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.05, 0.925,
             "束の間の注文 = 板に置かれてから τ 以内に、約定せずに取り消された指値"
             "(Hasbrouck and Saar 2009 は 2 秒)。分母はいずれも「その期間に板から消えた指値」の全量で、"
             "板にある残高ではない。\n"
             "母集団はトリガー注文・テイカー(Ioc など)・reduce_only を除いた Alo / Gtc の指値。"
             "最良気配からの距離は発注時刻より前の最後の bbo で測る。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_fleeting.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
