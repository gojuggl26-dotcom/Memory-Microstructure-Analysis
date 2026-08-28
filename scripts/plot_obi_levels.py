"""ティック水準別 OBI の回帰係数を水準ごとに図示する。

★9 ホライズン × 10 水準を 1 枚の折れ線に全部載せると、順序ランプの 9 色が
背景との対比か隣接の識別かのどちらかを必ず割る(palette_check で確認)。
そこで折れ線は 4 ホライズンに絞り、全格子はヒートマップで見せる。

    uv run python scripts/plot_obi_levels.py --coin xyz:MU
出力: charts/<coin>_obi_levels.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
UP, DOWN, WARN, GREEN = "#e34948", "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#5f9ade", "#2a70c4", "#1f5aa4", "#0d366b"]      # 順序ランプ(ホライズン)
SHOW = ["100ms", "1s", "5s", "50s"]                      # 折れ線に出すホライズン
HLAB = ["100ms", "200ms", "500ms", "1s", "2s", "5s", "10s", "20s", "50s"]
DIV = LinearSegmentedColormap.from_list("div", ["#1a4f96", DOWN, "#dfe9f5", SURFACE,
                                                "#f7dcd8", UP, "#9b2b2a"])
NLV = 10


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def by_level(F, defn, h, scope="全日", kind="実測"):
    return F.filter((pl.col("scope") == scope) & (pl.col("kind") == kind)
                    & (pl.col("defn") == defn) & (pl.col("h") == h)).sort("lv")


def heat(F, defn, kind="実測"):
    M = np.full((len(HLAB), NLV), np.nan)
    for i, h in enumerate(HLAB):
        s = by_level(F, defn, h, kind=kind)
        for r in s.iter_rows(named=True):
            M[i, r["lv"] - 1] = r["slope"]
    return M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    F = pl.read_parquet(ROOT / "data" / f"obi_levels_fit_{tag}.parquet")
    G = pl.read_parquet(ROOT / "data" / f"obi_levels_dose_{tag}.parquet")
    M = pl.concat([pl.read_parquet(f) for f in
                   sorted((ROOT / "data" / "obi_levels_meta" / tag).glob("dt=*.parquet"))])
    n_used = float(M["n_used"].sum())
    # ★分母をはっきりさせる。pm1 は「OBI が定義できた点」のうち片側が空の割合、
    #   nodef は「10 水準とも取り出せた点」のうち両側とも空で定義できない割合。
    nval = np.array([float(M[f"n_valid_l{l+1}"].sum()) for l in range(NLV)])
    pm1 = np.array([float(M[f"n_pm1_l{l+1}"].sum()) for l in range(NLV)]) / np.maximum(nval, 1)
    nodef = 1.0 - nval / n_used
    n_day = int(F["n_day"].max())

    # 片道の費用 = 半スプレッドの中央値[bp]。回帰と同じ掃除をした bbo から出す
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_obi_levels import clean_bbo                       # noqa: E402
    bb, _ = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    bb = bb.filter(pl.col("dt").is_in(M["dt"].to_list()))
    pb, pa = bb["best_bid"].to_numpy(), bb["best_ask"].to_numpy()
    half = float(np.median((pa - pb) / (pa + pb) * 1e4))

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(16.4, 10.6), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.28,
                          left=0.055, right=0.985, top=0.800, bottom=0.075)
    xs = np.arange(1, NLV + 1)

    # ---- ① 主図: 傾きを水準ごとに ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for i, h in enumerate(SHOW):
        s = by_level(F, "水準ごと", h)
        if s.is_empty():
            continue
        v = s["slope"].to_numpy()
        sig = (s["bonf_lo"].to_numpy() > 0) | (s["bonf_hi"].to_numpy() < 0)
        ax.fill_between(xs[:len(v)], s["ci_lo"].to_numpy(), s["ci_hi"].to_numpy(),
                        color=RAMP[i], alpha=0.16, lw=0)
        ax.plot(xs[:len(v)], v, "-", color=RAMP[i], lw=2.0, zorder=3)
        ax.plot(xs[:len(v)][sig], v[sig], "o", color=RAMP[i], ms=5.0, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.1)
        ax.plot(xs[:len(v)][~sig], v[~sig], "o", color=SURFACE, ms=5.0, zorder=4,
                markeredgecolor=RAMP[i], markeredgewidth=1.3)
        ax.annotate(h, (NLV + 0.15, v[-1]), fontsize=8.5, color=RAMP[i], va="center",
                    weight="bold")
        p = by_level(F, "水準ごと", h, kind="帰無対照")
        if not p.is_empty():
            ax.plot(xs[:p.height], p["slope"].to_numpy(), ":", color=MUTED, lw=1.0,
                    zorder=2)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.set_xticks(xs)
    ax.set_xlim(0.6, NLV + 1.4)
    ax.set_xlabel("最良気配からの距離[ティック](1 = 最良気配)", color=INK2, fontsize=10)
    ax.set_ylabel("傾き[bp / OBI 1 単位]", color=INK2, fontsize=10)
    ax.set_title("① ★傾きは最良気配で最大、奥へ行くほど単調に減る\n"
                 "帯 = 95% 区間、点線 = 帰無対照、塗りつぶし丸 = Bonferroni 後も有意",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    style(ax)

    # ---- ②③ 全格子のヒートマップ ------------------------------------------------
    Hm = heat(F, "水準ごと")
    Hc = heat(F, "累積")
    lim = float(np.nanmax(np.abs(np.concatenate([Hm.ravel(), Hc.ravel()]))))
    nrm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for k, (Hh, ttl, sub) in enumerate([
        (Hm, "② 水準ごとの OBI(90 格子すべて)",
         "その 1 ティックにある数量だけを使う。色の目盛は ③ と共通"),
        (Hc, "③ 累積の OBI — 奥を足すと係数は増える",
         "1〜l ティックを合計してから比を取る。全 279 格子が有意で正")]):
        ax = fig.add_subplot(gs[0, k + 1])
        im = ax.imshow(Hh, cmap=DIV, norm=nrm, aspect="auto", origin="upper")
        ax.set_xticks(np.arange(NLV))
        ax.set_xticklabels(xs, fontsize=8.5)
        ax.set_yticks(np.arange(len(HLAB)))
        ax.set_yticklabels(HLAB, fontsize=8.5)
        for i in range(len(HLAB)):
            for j in range(NLV):
                if np.isfinite(Hh[i, j]):
                    ax.text(j, i, f"{Hh[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.4,
                            color=SURFACE if abs(Hh[i, j]) > lim * 0.55 else INK2)
        ax.set_xlabel("最良気配からの距離[ティック]", color=INK2, fontsize=10)
        ax.set_ylabel("予測ホライズン", color=INK2, fontsize=10)
        ax.set_title(f"{ttl}\n{sub}", loc="left", color=INK, fontsize=11, pad=8,
                     weight="bold")
        ax.tick_params(colors=MUTED, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
            cb.set_label("傾き[bp / OBI 1 単位]", color=INK2, fontsize=8.5)
            cb.ax.tick_params(colors=MUTED, labelsize=8, length=2)
            cb.outline.set_visible(False)

    # ---- ④ 1σ 当たりの効果と、その水準が空の割合 --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax2 = ax.twinx()
    ax2.bar(xs, pm1 * 100, color=NEUTRAL, width=0.68, zorder=0,
            edgecolor=BASELINE, lw=0.6)
    ax2.bar(xs, nodef * 100, color="#e6e5df", width=0.68, zorder=0, bottom=pm1 * 100,
            edgecolor=BASELINE, lw=0.6, hatch="///")
    ax2.set_ylabel("OBI が ±1 / 定義できない割合[%]", color=MUTED, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(BASELINE)
    for i, h in enumerate(SHOW):
        s = by_level(F, "水準ごと", h)
        if s.is_empty():
            continue
        ax.plot(xs[:s.height], s["slope_per_sd"].to_numpy(), "o-", color=RAMP[i],
                lw=2.0, ms=4.8, label=h, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.axhline(half, color=UP, lw=1.8)
    ax.text(0.7, half, f"片道の費用(半スプレッド中央値){half:.2f} bp", ha="left",
            va="bottom", color=UP, fontsize=8.8, weight="bold")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("最良気配からの距離[ティック]", color=INK2, fontsize=10)
    ax.set_ylabel("OBI が 1σ 動いたときの bp", color=INK2, fontsize=10)
    ax.set_title("④ ★1σ で測ると 2 ティック目が最大 — それでも費用に届かない\n"
                 "灰色の棒 = OBI がちょうど ±1(片側が空)、斜線 = 両側とも空で定義できない",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    style(ax)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, title="ホライズン",
              title_fontsize=8.5, loc="center right")

    # ---- ⑤ 用量反応 -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    edges = np.linspace(-1, 1, 11)
    ctr = (edges[:-1] + edges[1:]) / 2
    for lv, cl in [(1, DOWN), (2, GREEN), (5, WARN), (10, "#4a3aa7")]:
        s = (G.filter((pl.col("scope") == "全日") & (pl.col("defn") == "水準ごと")
                      & (pl.col("lv") == lv) & (pl.col("h") == "1s")).sort("bin"))
        if s.is_empty():
            continue
        ax.plot(ctr[s["bin"].to_numpy()], s["mean_bp"].to_numpy(), "o-", color=cl,
                lw=1.8, ms=4.4, label=f"{lv} ティック目",
                markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.axhline(0, color=INK, lw=1.0)
    ax.axvline(0, color=BASELINE, lw=0.9)
    ax.set_xlabel("OBI(−1 = 売り一色、+1 = 買い一色)", color=INK2, fontsize=10)
    ax.set_ylabel("1 秒後までの平均 log リターン[bp]", color=INK2, fontsize=10)
    ax.set_title("⑤ 用量反応 — 直線を当てて良いかの確認\n"
                 "帯は [−1, +1] を等幅 0.2 で 10 分割(ホライズン 1 秒)",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, title="水準",
              title_fontsize=8.5)
    style(ax)

    # ---- ⑥ 検算: 板の再構成の欠陥が結論を動かすか ------------------------------
    ax = fig.add_subplot(gs[1, 2])
    xh = np.arange(len(HLAB))
    for defn, cl, lab in [("水準ごと", DOWN, "再構成した板の最良気配"),
                          ("bbo 最良", WARN, "bbo の最良数量(fills 統合済み)"),
                          ("水準ごと(L1一致)", GREEN, "両者が厳密に一致した格子点だけ")]:
        s = (F.filter((pl.col("scope") == "全日") & (pl.col("kind") == "実測")
                      & (pl.col("defn") == defn) & (pl.col("lv") == 1)).sort("h_ms"))
        if s.is_empty():
            continue
        ax.plot(xh[:s.height], s["slope"].to_numpy(), "o-", color=cl, lw=2.0, ms=4.8,
                label=lab, markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.fill_between(xh[:s.height], s["ci_lo"].to_numpy(), s["ci_hi"].to_numpy(),
                        color=cl, alpha=0.15, lw=0)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xh)
    ax.set_xticklabels(HLAB, fontsize=8.5)
    ax.set_xlabel("予測ホライズン", color=INK2, fontsize=10)
    ax.set_ylabel("最良気配の OBI の傾き[bp]", color=INK2, fontsize=10)
    ax.set_title("⑥ 検算 — 再構成の弱点は係数を 1〜2 割小さく見せるだけ\n"
                 "帯は日単位ブロックブートストラップの 95% 区間",
                 loc="left", color=INK, fontsize=11, pad=8, weight="bold")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    style(ax)

    fig.suptitle(f"{a.coin} ティック水準別 OBI は将来 log リターンを説明するか"
                 f"(100ms 格子・{n_day} 日)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.055, 0.932,
             "OBI_l = (水準 l の買い数量 − 売り数量) ÷ 合計。水準 l = 最良気配から l−1 ティック離れた価格"
             "(ティックは価格 1000 未満で 0.01、以上で 0.1)。板は l1 の注文イベントから再構成。\n"
             "回帰は y = ln(mid_{t+h} / mid_t) × 10^4 を x = OBI_l(t) へ単回帰。"
             "x は時刻 t より前のイベントだけで決まる(先読み無し)。有意性は Bonferroni 補正(279 通り)後で判定。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.005, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses + l2/bbo を再構成",
             color=MUTED, fontsize=8)
    out = ROOT / "charts" / f"{tag}_obi_levels.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
