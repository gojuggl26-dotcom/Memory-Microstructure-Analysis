"""見せかけの板を疑う 18 指標 — ベースライン、口座別の分布、そして検証を描く。

    uv run python scripts/plot_manip.py --coin xyz:MU
出力: charts/<coin>_manip_base.png    母集団のベースライン(日次)
      charts/<coin>_manip_wallets.png 口座別の分布と 2 つの合成スコア
      charts/<coin>_manip_check.png   スコアは何かに対応しているかの検証
      data/manip_scores_<coin>.csv    口座ごとの指標とスコア

【スコアの立場】
spoof score / layering score は**母集団の中での相対順位**を平均したものである。
定義上、必ず誰かが最上位に来る。したがって「スコアが高い = 相場操縦」ではない。
スコアが意味を持つのは、**それが何か別の観測量と対応しているとき**だけなので、
第 3 図で「大口を出した直後に値段がその向きへ余分に動いたか」を検証する。
対照は **活動量の層内でスコアを入れ替えた並べ替え検定**。
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

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
UP, DOWN, WARN = "#e34948", "#2a78d6", "#eb6834"
HB, HR = 4.547787, 0.502944
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

MIN_ORD = 1_000        # スコアを出す口座の下限(注文数)
MIN_LARGE = 20         # 大口を分母にする率を出す下限

# スコアの構成要素。順位を平均するので単位は揃っていなくてよい
SPOOF = ["short_large_rate", "large_near_rate", "cbt_rate",
         "retreat_large_rate", "asym_abs"]
LAYER = ["cycle_rate", "layer_rate", "simul_rate", "asym_abs", "bigrep_rate"]
LABEL = {
    "large_rate": "大口の割合(1)",
    "short_large_rate": "大口が 1 秒未満で消える割合(2)",
    "large_near_rate": "大口を最良の近くに出す割合(3)",
    "bigrep_rate": "1 分に大口 3 本以上(4)",
    "cancel_rate": "取消率(5)",
    "canc_per_s_max": "1 秒あたり取消数の最大(5)",
    "cycle_rate": "同一価格へ 1 分 5 本以上(6)",
    "layer_rate": "3 価格以上へ 1 秒以内(7)",
    "simul_rate": "3 価格以上へ同一ブロック(8)",
    "asym_abs": "使う価格数の左右差(9)",
    "fake_rate": "最良に届いても引く割合(10)",
    "cbt_rate": "取消直後にその値で約定(11)",
    "retreat_rate": "遠くへ出し直す(12)",
    "chase_rate": "近くへ出し直す(13)",
    "retreat_large_rate": "大口を遠くへ出し直す(14)",
    "reentry_rate": "同じ価格へ出し直す(15)",
    "fleet_ratio": "1 秒未満が占める表示時間(18)",
}


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
    for lg, ax_ in ((logx, ax.xaxis), (logy, ax.yaxis)):
        if lg:
            (ax.set_xscale if ax_ is ax.xaxis else ax.set_yscale)("log")
            ax_.set_major_locator(LogLocator(base=10.0))
            ax_.set_major_formatter(FuncFormatter(_pow10))
            ax_.set_minor_formatter(NullFormatter())
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


def rankpct(v: np.ndarray) -> np.ndarray:
    """順位を [0,1] に直す。欠測は中央(0.5)に置く。"""
    out = np.full(len(v), 0.5)
    m = np.isfinite(v)
    if m.sum() > 1:
        r = np.argsort(np.argsort(v[m]))
        out[m] = r / (m.sum() - 1)
    return out


def build_scores(W: pl.DataFrame) -> pl.DataFrame:
    """口座 × 日 を口座へまとめ、率にしてから順位で合成する。"""
    g = W.group_by("wid").agg(
        n_days=pl.col("dt").n_unique(),
        **{c: pl.col(c).sum() for c in
           ("n_ord", "n_large", "n_cancel", "n_fill", "n_short_large",
            "n_large_near", "n_reached", "n_fake", "n_cbt", "n_cbt_large",
            "sz_sum", "szt", "szt_fleet", "n_bid", "reentry", "retreat",
            "chase", "retreat_large", "layer_ep", "simul_ep", "cycle_ep",
            "bigrep_ep", "n_push", "push_sum", "push_pre_sum", "n_base",
            "base_sum", "base_pre_sum")},
        canc_per_s_max=pl.col("canc_per_s_max").max(),
        asym_abs=pl.col("asym_absmean").mean())
    # ★episode 系が無い口座は null ではなく 0。null のままだと順位付けで
    #   「真ん中」に置かれ、一度も layering しない口座が中位に来てしまう
    g = g.with_columns([pl.col(c).fill_null(0) for c in
                        ("layer_ep", "simul_ep", "cycle_ep", "bigrep_ep",
                         "reentry", "retreat", "chase", "retreat_large")])
    d = pl.col
    g = g.with_columns(
        large_rate=d("n_large") / d("n_ord"),
        cancel_rate=d("n_cancel") / d("n_ord"),
        fleet_ratio=d("szt_fleet") / d("szt"),
        reentry_rate=d("reentry") / d("n_ord"),
        retreat_rate=d("retreat") / d("n_ord"),
        chase_rate=d("chase") / d("n_ord"),
        cycle_rate=d("cycle_ep") / d("n_days"),
        layer_rate=d("layer_ep") / d("n_days"),
        simul_rate=d("simul_ep") / d("n_days"),
        bigrep_rate=d("bigrep_ep") / d("n_days"),
        bid_share=d("n_bid") / d("n_ord"),
        cbt_rate=pl.when(d("n_cancel") > 0).then(d("n_cbt") / d("n_cancel")),
        fake_rate=pl.when(d("n_reached") > 0).then(d("n_fake") / d("n_reached")),
        short_large_rate=pl.when(d("n_large") >= MIN_LARGE)
        .then(d("n_short_large") / d("n_large")),
        large_near_rate=pl.when(d("n_large") >= MIN_LARGE)
        .then(d("n_large_near") / d("n_large")),
        retreat_large_rate=pl.when(d("n_large") >= MIN_LARGE)
        .then(d("retreat_large") / d("n_large")),
        push_mean=pl.when(d("n_push") > 0).then(d("push_sum") / d("n_push")),
        push_pre_mean=pl.when(d("n_push") > 0)
        .then(d("push_pre_sum") / d("n_push")),
        base_mean=pl.when(d("n_base") > 0).then(d("base_sum") / d("n_base")),
        base_pre_mean=pl.when(d("n_base") > 0)
        .then(d("base_pre_sum") / d("n_base")))
    g = g.with_columns(
        # 「押した分」= 出したあと − 出す前。追いかけただけなら 0 に近い
        excess=d("push_mean") - d("push_pre_mean"),
        excess_base=d("base_mean") - d("base_pre_mean"))
    g = g.filter(d("n_ord") >= MIN_ORD)
    for name, comp in (("spoof_score", SPOOF), ("layer_score", LAYER)):
        M = np.column_stack([rankpct(g[c].to_numpy().astype(float))
                             for c in comp])
        g = g.with_columns(pl.Series(name, M.mean(axis=1)))
    return g.sort("spoof_score", descending=True)


def pooled(num, den, b, n_dec=10):
    """十分位ごとに Σ(分子)/Σ(分母)。口座ごとの平均を平均すると、
    episode が数件しかない口座の雑音が支配する(実際に 376bp が出た)。"""
    o = np.full(n_dec, np.nan)
    for j in range(n_dec):
        m = (b == j) & np.isfinite(num) & np.isfinite(den) & (den > 0)
        if m.sum() >= 3 and den[m].sum() > 0:
            o[j] = num[m].sum() / den[m].sum()
    return o


def perm_null(score, act, num, den, n_dec=10, n_rep=300, seed=0):
    """活動量の層の中でスコアを入れ替える並べ替え検定(プール平均で)。"""
    rng = np.random.default_rng(seed)
    strat = np.digitize(act, np.percentile(act, np.arange(10, 100, 10)))
    out = np.full((n_rep, n_dec), np.nan)
    for r in range(n_rep):
        s = score.copy()
        for k in np.unique(strat):
            m = strat == k
            s[m] = rng.permutation(s[m])
        b = np.digitize(s, np.percentile(s, np.arange(10, 100, 10)))
        out[r] = pooled(num, den, b, n_dec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    W = pl.read_parquet(ROOT / "data" / f"manip_wallet_{tag}.parquet")
    E = pl.read_parquet(ROOT / "data" / f"manip_episodes_{tag}.parquet")
    D = pl.read_csv(ROOT / "data" / f"manip_daily_{tag}.csv")
    G = build_scores(W)
    G.write_csv(ROOT / "data" / f"manip_scores_{tag}.csv")
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.framealpha": 0.92})
    out = ROOT / "charts"
    x = np.arange(D.height)

    # ================= 図 1: ベースライン ==================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.24,
                          left=0.055, right=0.985, top=0.845, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, D["share_short"] * 100, color=UP, lw=1.8, label="本数で数える", zorder=3)
    ax.plot(x, D["fleeting_ratio"] * 100, color=DOWN, lw=1.8,
            label="数量 × 表示時間で数える", zorder=3)
    ax.set_title("① fleeting liquidity — 1 秒未満で消える板", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日(2026-05-04 から)", "%")
    legend(ax, loc="center right")
    ax.text(0.03, 0.55, "本数では 6 割。しかし見えている板の\n量としては 2% しかない",
            transform=ax.transAxes, fontsize=8.4, color=INK2, va="top")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, D["fake_share"] * 100, color=INK, lw=1.8, label="本数", zorder=3)
    ax.plot(x, D["fake_share_sz"] * 100, color=WARN, lw=1.8, label="数量加重",
            zorder=3)
    ax.set_title("② fake depth persistence — 最良に届いても引く割合", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日", "%")
    ax.set_ylim(0, 100)
    legend(ax, loc="lower right")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x, D["cbt_share"] * 100, color=UP, lw=1.8, label="全注文", zorder=3)
    ax.plot(x, D["cbt_share_large"] * 100, color=DOWN, lw=1.8, label="大口だけ",
            zorder=3)
    ax.set_title("③ cancel-before-touch — 引いた直後にその値で約定", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日", "%")
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, D["large_lots"] / 1000, color=INK, lw=1.8, zorder=3)
    ax.set_title("④ 大口の閾値(前日の p99)", color=INK, fontsize=10.5, pad=7,
                 loc="left")
    style(ax, "日", "契約", logy=True)
    ax2 = ax.twiny()
    ax2.set_xticks([])

    ax = fig.add_subplot(gs[1, 1])
    n = G["n_ord"].to_numpy().astype(float)
    o = np.sort(n)[::-1]
    ax.plot(np.arange(1, len(o) + 1) / len(o) * 100, np.cumsum(o) / o.sum() * 100,
            color=INK, lw=2.0, zorder=3)
    ax.plot([0, 100], [0, 100], "-", color=BASELINE, lw=1.0, zorder=2)
    ax.set_title(f"⑤ 注文の集中({len(o)} 口座、多い順)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "口座の割合(%)", "注文の累積割合(%)")
    top1 = np.cumsum(o)[max(int(len(o) * 0.01) - 1, 0)] / o.sum() * 100
    ax.text(0.5, 0.1, f"上位 1% の口座が全注文の {top1:.0f}%",
            transform=ax.transAxes, fontsize=8.6, color=INK2)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, D["push_all"], color=UP, lw=1.8, label="出したあと 10 秒", zorder=3)
    ax.plot(x, D["push_pre_all"], color=DOWN, lw=1.8, label="出す前 10 秒", zorder=3)
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.set_title("⑥ 注文を出す前後の mid の動き(全注文)", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "日", "その注文の側への動き(bp)")
    legend(ax, loc="upper left")

    fig.text(0.055, 0.972, "母集団のベースライン — 「すぐ消す」は全員がやっている",
             fontsize=17, color=INK, va="top")
    fig.text(0.055, 0.930,
             f"{D.height} 日 / 注文 {int(D['n_ord'].sum()):,} 本。"
             "閾値判定の前に、母集団がどれだけ「怪しく」見えるかを置く。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_manip_base.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 2: 口座別 =========================================
    keys = ["short_large_rate", "large_near_rate", "cbt_rate",
            "retreat_large_rate", "asym_abs", "cycle_rate", "layer_rate",
            "simul_rate", "fake_rate", "fleet_ratio", "reentry_rate",
            "cancel_rate"]
    fig = plt.figure(figsize=(16.4, 10.6), dpi=150)
    gs = fig.add_gridspec(3, 5, hspace=0.55, wspace=0.30,
                          left=0.055, right=0.985, top=0.870, bottom=0.065)
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        v = G[k].to_numpy().astype(float)
        v = np.sort(v[np.isfinite(v)])
        if not len(v):
            continue
        lg = v.min() > 0 and v.max() / max(v.min(), 1e-12) > 100
        ax.plot(v, np.arange(len(v)) / len(v) * 100, color=INK, lw=1.8, zorder=3)
        q9 = np.percentile(v, 90)
        ax.axvline(q9, color=UP, ls=(0, (4, 2)), lw=1.2, zorder=4)
        ax.set_title(LABEL.get(k, k), color=INK, fontsize=8.8, pad=5, loc="left")
        style(ax, "", "累積%(口座)" if i % 5 == 0 else "", logx=lg)
        ax.set_ylim(0, 100)
    ax = fig.add_subplot(gs[2, 3:5])
    sc = ax.scatter(G["layer_score"], G["spoof_score"],
                    s=np.clip(G["n_ord"].to_numpy() / 3e4, 3, 120),
                    c=np.log10(np.maximum(G["n_ord"].to_numpy(), 1)),
                    cmap="viridis", alpha=0.75, lw=0)
    ax.set_title("2 つのスコア(点の大きさ・色 = 注文数)", color=INK,
                 fontsize=9.5, pad=6, loc="left")
    style(ax, "layering score", "spoof score")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log10(注文数)", color=INK2, fontsize=8)
    r = np.corrcoef(G["layer_score"], G["spoof_score"])[0, 1]
    ax.text(0.04, 0.94, f"相関 {r:+.3f}", transform=ax.transAxes, fontsize=8.6,
            color=INK2, va="top")

    fig.text(0.055, 0.975, "口座別の分布 — 破線は上位 10% の境目", fontsize=17,
             color=INK, va="top")
    fig.text(0.055, 0.935,
             f"注文 {MIN_ORD:,} 本以上の {G.height} 口座。横軸は指標の値、"
             "縦軸はその値以下の口座の割合。裾がどれだけ伸びているかを見る。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_manip_wallets.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)

    # ================= 図 3: 検証 ===========================================
    fig = plt.figure(figsize=(16.4, 9.4), dpi=155)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                          left=0.055, right=0.985, top=0.845, bottom=0.085)
    act = G["n_ord"].to_numpy().astype(float)
    num = (G["push_sum"].to_numpy().astype(float)
           - G["push_pre_sum"].to_numpy().astype(float))
    den = G["n_push"].to_numpy().astype(float)
    for j, (sc_name, lab) in enumerate((("spoof_score", "spoof score"),
                                        ("layer_score", "layering score"))):
        ax = fig.add_subplot(gs[0, j])
        s = G[sc_name].to_numpy().astype(float)
        b = np.digitize(s, np.percentile(s, np.arange(10, 100, 10)))
        obs = pooled(num, den, b)
        nul = perm_null(s, act, num, den)
        lo, hi = np.nanpercentile(nul, [2.5, 97.5], axis=0)
        ax.fill_between(np.arange(10), lo, hi, color=BASELINE, alpha=0.5, lw=0,
                        label="帰無(活動量の層内で入替、95%)")
        ax.plot(np.arange(10), obs, "o-", color=UP, lw=2.0, ms=6, label="実測",
                zorder=3)
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        ax.set_title(f"① {lab} の十分位 × 「押した分」", color=INK, fontsize=10.5,
                     pad=7, loc="left")
        style(ax, f"{lab} の十分位",
              "出したあと − 出す前(bp)" if j == 0 else "")
        # 帰無の帯は十分位によっては 100bp まで伸びる。実測が見える範囲に切る
        ax.set_ylim(min(-12, np.nanmin(obs) * 1.3), max(35, np.nanmax(obs) * 1.3))
        ax.text(0.03, 0.04, "帯は上下に切れている(十分位によっては ±100bp)",
                transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom")
        legend(ax, loc="best")

    # ---- 決定的な検証: 大口を出したあと、自分は何をしたか -------------------
    ax = fig.add_subplot(gs[0, 2])
    ex = (E["push"].to_numpy().astype(float)
          - E["push_pre"].to_numpy().astype(float))
    opp = E["tk_opp"].to_numpy().astype(float)
    sam = E["tk_same"].to_numpy().astype(float)
    grp = [("反対側で\n成行を出した", (opp > 0), UP),
           ("同じ側で\n成行を出した", (opp == 0) & (sam > 0), DOWN),
           ("自分は成行を\n出さなかった", (opp == 0) & (sam == 0), MUTED)]
    xs, ys, es, ns = [], [], [], []
    for i, (lab, m, cc) in enumerate(grp):
        v = ex[m & np.isfinite(ex)]
        xs.append(i)
        ys.append(v.mean())
        es.append(1.96 * v.std() / np.sqrt(max(len(v), 1)))
        ns.append(len(v))
        ax.bar(i, v.mean(), color=cc, zorder=3, width=0.62)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=1.2,
                capsize=4, zorder=4)
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in grp], fontsize=8)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + (0.4 if y >= 0 else -0.4), f"n={n:,}", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=8, color=INK2)
    ax.set_title("② 大口を出したあと、自分は何をしたか", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "", "押した分(bp)")
    ax.text(0.5, -0.34,
            "見せかけの買い板なら「値段が上がってから自分は売る」はず。\n"
            "実際は反対側で売った層ほど値段は逆に動いている。",
            transform=ax.transAxes, fontsize=8.4, color=INK2, ha="center",
            va="top")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.sort(G["spoof_score"]), np.arange(G.height) / G.height * 100,
            color=INK, lw=1.9, label="spoof", zorder=3)
    ax.plot(np.sort(G["layer_score"]), np.arange(G.height) / G.height * 100,
            color=WARN, lw=1.9, label="layering", zorder=3)
    ax.set_title("③ スコアの分布(相対順位の平均なので必ず誰かが上位)",
                 color=INK, fontsize=10.5, pad=7, loc="left")
    style(ax, "スコア", "累積%(口座)")
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(act, G["spoof_score"], s=8, color=MUTED, alpha=0.6, lw=0)
    r = np.corrcoef(np.log10(np.maximum(act, 1)),
                    G["spoof_score"].to_numpy())[0, 1]
    ax.set_title(f"④ スコアは活動量の言い換えか(順位相関 {r:+.3f})", color=INK,
                 fontsize=10.5, pad=7, loc="left")
    style(ax, "注文数(対数)", "spoof score", logx=True)

    ax = fig.add_subplot(gs[1, 2])
    top = G.head(max(int(G.height * 0.1), 1))
    rest = G.tail(G.height - top.height)
    names = ["short_large_rate", "large_near_rate", "cbt_rate",
             "retreat_large_rate", "asym_abs", "cycle_rate", "layer_rate"]
    y = np.arange(len(names))
    tv = [float(np.nanmedian(top[c].to_numpy().astype(float))) for c in names]
    rv = [float(np.nanmedian(rest[c].to_numpy().astype(float))) for c in names]
    rat = [t / r if r and np.isfinite(r) and r > 0 else np.nan
           for t, r in zip(tv, rv)]
    ax.barh(y, rat, color=UP, zorder=3)
    ax.axvline(1, color=INK, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(c, c) for c in names], fontsize=7.6)
    ax.yaxis.tick_right()                   # 左隣の図と重ならないように
    ax.invert_yaxis()
    ax.set_title("⑤ 上位 10% ÷ 残り(中央値の比、対数)", color=INK, fontsize=10.5,
                 pad=7, loc="left")
    style(ax, "倍(対数)", "", logx=True)
    for yy, r in zip(y, rat):
        if np.isfinite(r):
            big = r > 500                       # 端に貼り付くので内側へ入れる
            ax.text(r / 1.6 if big else r * 1.18, yy,
                    f"{r:,.0f}×" if r >= 10 else f"{r:.2f}×", va="center",
                    ha="right" if big else "left", fontsize=7.4,
                    color=SURFACE if big else INK2)

    fig.text(0.055, 0.972, "検証 — スコアは何かに対応しているか", fontsize=17,
             color=INK, va="top")
    fig.text(0.055, 0.930,
             "「押した分」= 大口を最良の近くに出したあと 10 秒の mid の動き "
             "− 出す前 10 秒の動き。追いかけただけなら 0 に近い。"
             "帯は活動量の層内でスコアを入れ替えた並べ替え検定(300 回)。",
             fontsize=10, color=INK2, va="top")
    p = out / f"{tag}_manip_check.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}", file=sys.stderr)
    print(f"-> data/manip_scores_{tag}.csv({G.height} 口座)", file=sys.stderr)


if __name__ == "__main__":
    main()
