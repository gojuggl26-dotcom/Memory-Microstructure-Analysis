"""エントロピー 13 種の分布・冗長性・予測力を 2 枚の図にする。

    uv run python scripts/plot_entropy.py --coin xyz:MU
出力: charts/<coin>_entropy.png          13 種が何を捉えているか(分布・実効数・冗長性)
      charts/<coin>_entropy_predict.png  予測力(前向き 対 後ろ向き・帰無対照・統制)

【配色】
順序のない 13 特徴量は色で区別せず、**3 つの系統でパネルを分ける**
(板 = 青 / イベント = 緑 / 動的 = 紫)。符号のある量(相関・z)は発散(青 ↔ 赤)。
系列が 5 本以上になると全対の色覚検査は通らないため、色を増やさずパネルを割る。

palette_check.py の実測(surface #fcfcfb、全対):
    系統 3 色 #2a78d6 / #0f8f63 / #4a3aa7      PASS(対比 4.30 / 3.99 / 8.33)
    発散ペア  #2a78d6 / #e34948                PASS
    素 / 偏   #2a78d6 / #0f8f63                PASS
帰無対照は**色を足さずに塗り分け**(実測 = 塗りつぶし、対照 = 輪郭)で表す。
低彩度の灰は彩度下限 0.10 を割って検査に落ちるため、系列色には使わない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entropy import FEATURES, HORIZONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
UP, DOWN, WARN, GREEN, PURPLE = "#e34948", "#2a78d6", "#eb6834", "#0f8f63", "#4a3aa7"

BOOK = ["h_depth", "h_size", "h_plevel", "h_wallet", "h_age", "h_queue"]
EVENT = ["h_action", "h_side", "h_dir"]
DYN = ["h_chg", "h_rate", "h_cond", "h_trans"]
GROUPS = [("板の状態", BOOK, DOWN), ("イベントの流れ", EVENT, GREEN),
          ("動的", DYN, PURPLE)]

JP = {"h_depth": "depth\n水準別の数量", "h_size": "order-size\n数量の階級",
      "h_plevel": "price-level\n水準別の本数", "h_wallet": "wallet\n口座別の数量",
      "h_age": "order-age\n滞留時間の階級", "h_queue": "queue\n最良の待ち行列",
      "h_action": "NEW/REMOVE\n/UPDATE", "h_side": "side\nbid / ask",
      "h_dir": "event-direction\n押し上げ / 押し下げ",
      "h_chg": "entropy change\nΔh_depth", "h_rate": "entropy rate\nH₂/2",
      "h_cond": "conditional\nH(Xt|Xt−1)", "h_trans": "transition\n行平均"}
SHORT = {k: v.split("\n")[0] for k, v in JP.items()}
# 理論上限(ビット)。水準数に依存するものは None
CAP = {"h_size": np.log2(17), "h_age": np.log2(11), "h_action": np.log2(3),
       "h_side": 1.0, "h_dir": 1.0, "h_rate": np.log2(36) / 2,
       "h_cond": np.log2(6), "h_trans": np.log2(6)}

DIVERGE = LinearSegmentedColormap.from_list("dv", [DOWN, "#eef1f5", SURFACE,
                                                   "#f7eeee", UP])


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


def heat(ax, M, rows, cols, vmax, title, sub, fmt="{:+.3f}"):
    """発散カラーマップの行列。値をセルに焼き込む。"""
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.imshow(M, cmap=DIVERGE, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5, color=INK2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7.2,
                        color=INK if abs(M[i, j]) < vmax * 0.55 else SURFACE)
    ax.set_title(f"{title}\n{sub}", loc="left", color=INK, fontsize=10.5,
                 pad=7, weight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_features(d, R, cols_corr, daily, tag, coin, out):
    """図 1 — 13 種が何を捉えているか。"""
    fig = plt.figure(figsize=(17.6, 10.2), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.26,
                          left=0.055, right=0.985, top=0.80, bottom=0.075)

    # ---- ①②③ 3 系統の分布 --------------------------------------------------
    for gi, (gname, feats, base) in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, gi])
        pos = np.arange(len(feats))
        for i, f in enumerate(feats):
            v = d[f].to_numpy().astype(np.float64)
            v = v[np.isfinite(v)]
            q = np.percentile(v, [5, 25, 50, 75, 95])
            ax.plot([i, i], [q[0], q[4]], color=base, lw=1.1, alpha=0.55,
                    solid_capstyle="round")
            ax.plot([i, i], [q[1], q[3]], color=base, lw=6.5,
                    solid_capstyle="butt")
            ax.plot([i], [q[2]], marker="o", ms=5.5, color=SURFACE,
                    mec=base, mew=1.6, zorder=3)
            c = CAP.get(f)
            if c is not None:
                ax.plot([i - 0.34, i + 0.34], [c, c], color=WARN, lw=1.4,
                        ls=(0, (3, 2)), zorder=4)
        ax.set_xticks(pos)
        ax.set_xticklabels([JP[f] for f in feats], fontsize=8.2, color=INK2)
        ax.set_ylabel("エントロピー(ビット)", color=INK2, fontsize=9.5)
        ax.set_ylim(bottom=-0.15)
        note = ("橙の破線 = 理論上限 log₂k。Δh_depth は 0 のまわりに集中していて箱が潰れる"
                if gi == 2 else "橙の破線 = 理論上限 log₂k")
        ax.set_title(f"{'①②③'[gi]} {gname}の散らばり\n"
                     f"箱 = 四分位、髭 = 5–95%、丸 = 中央値。{note}",
                     loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
        style(ax)

    # ---- ④ 実効的な個数 2^H ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    eff = [("h_plevel", "価格水準(本数)"), ("h_depth", "価格水準(数量)"),
           ("h_wallet", "口座(数量)"), ("h_size", "数量の階級"),
           ("h_age", "滞留の階級")]
    y = np.arange(len(eff))
    vals = [2 ** np.nanmedian(d[f].to_numpy().astype(np.float64)) for f, _ in eff]
    ax.barh(y, vals, color=DOWN, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v * 1.06, i, f"{v:,.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in eff], fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xscale("log")
    # text.parse_math=False なので既定の指数表記が壊れる。目盛りを明示する
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(1, max(vals) * 2.2)
    ax.set_xlabel("実効的な個数 2^H(中央値)", color=INK2, fontsize=9.5)
    ax.set_title("④ エントロピーを「実効的な個数」に直す\n"
                 "本数で数えると水準は多いが、数量は少数の水準に寄っている",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(axis="y", lw=0)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    style(ax)
    ax.grid(axis="y", lw=0)

    # ---- ⑤ 日ごとの平均(安定しているか) --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    dd = daily.sort("dt")
    x = np.arange(dd.height)
    for f, c, lab in [("h_depth", DOWN, "h_depth"), ("h_plevel", GREEN, "h_plevel"),
                      ("h_wallet", PURPLE, "h_wallet"), ("h_age", WARN, "h_age")]:
        ax.plot(x, dd[f].to_numpy(), color=c, lw=1.3, label=lab)
    ax.set_xticks([0, dd.height // 2, dd.height - 1])
    ax.set_xticklabels([dd["dt"][0][5:], dd["dt"][dd.height // 2][5:],
                        dd["dt"][-1][5:]])
    ax.set_ylabel("日平均(ビット)", color=INK2, fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, ncol=2)
    ax.set_title("⑤ 日ごとの平均は安定しているか\n"
                 "★どれも上昇し続ける。期間をまたぐ比較には標準化が要る",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    style(ax)

    # ---- ⑥ 冗長性(相関行列) --------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    lab = [SHORT.get(c, c) for c in cols_corr]
    heat(ax, R, lab, lab, 1.0, "⑥ 13 種 + 統制 2 の順位相関",
         "★ h_rate / h_cond / h_trans は同じ 2 記号分布から作るので独立ではない",
         fmt="{:+.2f}")
    ax.set_xticklabels(lab, fontsize=6.4, color=INK2, rotation=90)
    ax.set_yticklabels(lab, fontsize=6.4, color=INK2)

    fig.suptitle(f"{coin} 板と注文イベントのエントロピー 13 種 — 何を捉えているか",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.055, 0.925,
             "エントロピー H = −Σ p log₂ p。1 点に集中すれば 0、k 個に均等なら log₂k で最大。"
             "単位はビット。1 秒格子・98 日。\n"
             "板の 6 種は mid ±100bp の帯にある指値が母集団(発注の 90.8% を覆い、"
             "1000bp 付近に駐機している別群を除く)。イベントの 3 種は直前 1 秒、"
             "動的の 3 種は直前 10 秒の窓で測る。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses から板を再構成",
             color=MUTED, fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


def fig_predict(fit, summ, tag, coin, out):
    """図 2 — 予測力。前向き / 後ろ向き / 帰無対照 / 統制。"""
    feats = list(FEATURES)
    zthr = summ["bonferroni_z"]

    def mat(target, col):
        M = np.full((len(feats), len(HORIZONS)), np.nan)
        for i, f in enumerate(feats):
            for j, h in enumerate(HORIZONS):
                r = fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == h)
                               & (pl.col("target") == target))
                if r.height:
                    M[i, j] = r[col][0]
        return M

    fig = plt.figure(figsize=(17.6, 10.4), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.075, right=0.985, top=0.795, bottom=0.075)
    hl = [f"{h}s" for h in HORIZONS]
    rl = [SHORT[f] for f in feats]

    # ---- ① ボラティリティ(|r|)への符号検定 z ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    Mv = mat("vol", "z")
    heat(ax, Mv, rl, hl, np.nanmax(np.abs(Mv)) or 1.0,
         "① 将来のボラティリティ |r| — 符号検定 z",
         f"日ごとの ρ の符号を 98 日で検定。|z| > {zthr:.2f} が Bonferroni 通過",
         fmt="{:+.1f}")

    # ---- ② 向き(r)への符号検定 z ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Md = mat("dir", "z")
    heat(ax, Md, rl, hl, np.nanmax(np.abs(Mv)) or 1.0,
         "② 将来の向き r — 符号検定 z",
         "①と同じ色の尺度。向きには効かないのが期待どおりの姿",
         fmt="{:+.1f}")

    # ---- ③ 前向き 対 後ろ向き(同時性の検査) ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    hpick = 10
    fw = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                              & (pl.col("target") == "vol"))["rho_day_med"][0]
                   for f in feats])
    bw = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                              & (pl.col("target") == "vol"))["backward_rho_med"][0]
                   for f in feats])
    lim = np.nanmax(np.abs(np.concatenate([fw, bw]))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.axvline(0, color=BASELINE, lw=0.8)
    # 点が密集するのでラベルは上下交互にずらし、引き出し線を付ける
    order = np.argsort(bw)
    for rank, i in enumerate(order):
        f = feats[i]
        c = DOWN if f in BOOK else (GREEN if f in EVENT else PURPLE)
        ax.plot(bw[i], fw[i], "o", ms=6.0, color=c, mec=SURFACE, mew=0.9,
                zorder=3)
        dy = 15 if rank % 2 else -15
        dx = 7 if rank % 4 < 2 else -7
        ax.annotate(SHORT[f], (bw[i], fw[i]), fontsize=6.6, color=INK2,
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.6,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"後ろ向き — 過去 {hpick} 秒の |r| との ρ", color=INK2, fontsize=9.5)
    ax.set_ylabel(f"前向き — 将来 {hpick} 秒の |r| との ρ", color=INK2, fontsize=9.5)
    ax.set_title("③ 予測か、後始末か\n"
                 "破線より下 = 過去との関係のほうが強い(= 前触れではない)",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    style(ax)

    # ---- ④ 帰無対照 ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(feats))
    zr = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                              & (pl.col("target") == "vol"))["z"][0] for f in feats])
    zp = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                              & (pl.col("target") == "vol"))["placebo_z"][0]
                   for f in feats])
    ax.barh(y - 0.19, zr, height=0.36, color=DOWN, label="実測")
    ax.barh(y + 0.19, zp, height=0.36, facecolor=SURFACE, edgecolor=DOWN,
            linewidth=1.0, hatch="////", label="帰無対照(12 時間巡回シフト)")
    for s in (zthr, -zthr):
        ax.axvline(s, color=WARN, lw=1.2, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(rl, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel(f"符号検定 z(|r| への予測、{hpick} 秒)", color=INK2, fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    ax.set_title("④ 帰無対照 — 時間をずらしても残るか\n"
                 "橙の破線 = Bonferroni 閾値。対照が閾値を越えるなら日内の形が作った相関",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.grid(axis="y", lw=0)
    style(ax)
    ax.grid(axis="y", lw=0)

    # ---- ⑤ 活動量を統制すると残るか -------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    raw = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                               & (pl.col("target") == "vol"))["rho_pooled"][0]
                    for f in feats])
    par = np.array([fit.filter((pl.col("feature") == f) & (pl.col("horizon_s") == hpick)
                               & (pl.col("target") == "vol"))["rho_partial"][0]
                    for f in feats])
    ax.barh(y - 0.19, raw, height=0.36, color=DOWN, label="素の ρ")
    ax.barh(y + 0.19, par, height=0.36, color=GREEN,
            label="偏 ρ(板の本数とイベント数を統制)")
    ax.axvline(0, color=BASELINE, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rl, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel(f"順位相関 ρ(|r| への予測、{hpick} 秒)", color=INK2, fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="lower right")
    ax.set_title("⑤ 「注文が多いと荒れる」の言い換えではないか\n"
                 "統制後も残る分だけが、散らばり方そのものの寄与",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.grid(axis="y", lw=0)
    style(ax)
    ax.grid(axis="y", lw=0)

    # ---- ⑥ Δエントロピーと将来ボラの関係は U 字 -----------------------------
    ax = fig.add_subplot(gs[1, 2])
    dec = pl.read_csv(ROOT / "data" / f"entropy_decile_{tag}.csv")
    act = dec.filter(pl.col("kind") == "actual").sort("decile")
    pla = dec.filter(pl.col("kind") == "placebo").sort("decile")
    xx = np.arange(1, 11)
    ax.plot(xx, act["absr_med_bp"].to_numpy(), "-o", color=DOWN, lw=1.6, ms=5,
            label="実測")
    ax.plot(xx, pla["absr_med_bp"].to_numpy(), "--s", color=DOWN, lw=1.2, ms=4,
            mfc=SURFACE, label="帰無対照(12 時間巡回シフト)")
    ax.set_xticks(xx)
    ax.set_xlabel("Δh_depth の十分位(左 = 最も下がった)", color=INK2, fontsize=9.5)
    ax.set_ylabel("次の 1 秒の |リターン| 中央値(bp)", color=INK2, fontsize=9.5)
    ax.legend(fontsize=8.0, frameon=False, labelcolor=INK2, loc="upper left")
    r1 = summ["decile_ratio"]
    ab = summ["abschg_1s"]
    ax.set_title("⑥ ★関係は単調ではなく U 字\n"
                 f"両端で荒れ、中央で凪ぐ(第 1 / 第 10 十分位で {r1:.2f} 倍)。"
                 "順位相関は左右差しか拾えない",
                 loc="left", color=INK, fontsize=10.5, pad=7, weight="bold")
    ax.annotate(f"|Δh_depth| で測り直すと ρ={ab['rho_day_med']:.3f}\n"
                f"(後ろ向き {ab['backward_rho_med']:.3f} のほうが大きく、\n"
                f" 統制後は {ab['rho_partial']:.3f} まで落ちる)",
                xy=(0.985, 0.97), xycoords="axes fraction", ha="right",
                va="top", fontsize=7.8, color=INK2, linespacing=1.5)
    style(ax)

    fig.suptitle(f"{coin} エントロピー 13 種は将来の値動きを予測するか"
                 f"({summ['n_cells']} セル全件・98 日)",
                 fontsize=13.5, y=0.975, color=INK, weight="bold")
    fig.text(0.075, 0.925,
             "x が確定する時刻 ≤ y の期間の開始時刻。板の状態は「厳密に T 未満」の"
             "イベントだけで作り、asof は backward のみ。地平 h の窓が重ならないよう "
             "stride = h で間引く。\n"
             "判定は日ごとの ρ の符号検定(98 日)で行う。1 秒格子の隣接点は強く相関するため、"
             "プールした標本数から出した z は使わない。",
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)
    fig.text(0.004, 0.006,
             "出所: Hyperliquid L4 (Artemis) node_order_statuses から板を再構成",
             color=MUTED, fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    print(f"[chart] {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = pl.read_parquet(ROOT / "data" / f"entropy_{tag}.parquet").filter(pl.col("ok"))
    fit = pl.read_csv(ROOT / "data" / f"entropy_fit_{tag}.csv")
    daily = pl.read_csv(ROOT / "data" / f"entropy_daily_{tag}.csv")
    corr = pl.read_csv(ROOT / "data" / f"entropy_corr_{tag}.csv")
    summ = json.loads((ROOT / "data" / f"entropy_summary_{tag}.json")
                      .read_text(encoding="utf-8"))
    cols_corr = corr["feature"].to_list()
    R = np.column_stack([corr[c].to_numpy() for c in cols_corr])

    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig_features(d, R, cols_corr, daily, tag, a.coin,
                 ROOT / "charts" / f"{tag}_entropy.png")
    fig_predict(fit, summ, tag, a.coin,
                ROOT / "charts" / f"{tag}_entropy_predict.png")


if __name__ == "__main__":
    main()
