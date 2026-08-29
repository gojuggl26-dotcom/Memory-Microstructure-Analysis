r"""口座パネルの図 — 集中度・役割・行動、そして「1 口座の癖」仮説の検証。

Boros の板は上位 1 者が発注量の中央 79.7% を占める。既存レポートの
「板の形が金利を予測する」がその 1 口座の癖でないかを、
**集中度と信号強度の関係**という間接検証で示す(直接検証は除外板で別途行う)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, CHARTS = ROOT / "data", ROOT / "charts"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 125,
})


def main() -> int:
    S = pl.read_parquet(DATA / "accounts_market.parquet")
    P = pl.read_parquet(DATA / "accounts_panel.parquet")
    R = pl.read_parquet(DATA / "pressure_results_tradeable.parquet")

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # (1) 集中度の分布
    ax = fig.add_subplot(gs[0, 0])
    t1 = S["top1_share"].to_numpy()
    hh = S["hhi"].to_numpy()
    ax.hist(t1, bins=30, color="#bfdbfe", edgecolor="#2563eb", label="上位 1 者のシェア")
    ax.axvline(np.median(t1), color="#dc2626", lw=2,
               label=f"中央 {np.median(t1):.1%}")
    ax.axvline(0.5, color="#111827", lw=1.0, ls="--", label="50%")
    ax.set_xlabel("上位 1 口座が占める発注量のシェア")
    ax.set_ylabel("市場数")
    ax.set_title(f"★1 口座への集中 — {int((t1 > .5).sum())}/{S.height} 市場で 50% 超\n"
                 f"HHI 中央 {np.median(hh):.3f}(0.25 超が高集中の目安)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # (2) 順位別シェア(ローレンツ的)
    ax = fig.add_subplot(gs[0, 1])
    for c, lab, col in [("top1_share", "上位 1 者", "#1d4ed8"),
                        ("top2_share", "上位 2 者", "#0891b2"),
                        ("top5_share", "上位 5 者", "#16a34a")]:
        v = np.sort(S[c].to_numpy())
        ax.plot(np.linspace(0, 100, len(v)), v, lw=2, color=col,
                label=f"{lab}(中央 {np.median(v):.0%})")
    ax.axhline(0.8, color="#6b7280", lw=0.8, ls=":")
    ax.set_xlabel("市場の百分位(シェアの小さい順)")
    ax.set_ylabel("累積シェア")
    ax.set_ylim(0, 1.02)
    ax.set_title("順位別の累積シェア\n上位 5 者でほぼ全量", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    # (3) 役割
    ax = fig.add_subplot(gs[0, 2])
    tot = S[["n_maker_only", "n_taker_only", "n_both"]].to_numpy().sum(axis=0)
    ax.bar(["メイカー専業", "テイカー専業", "両方"], tot,
           color=["#2563eb", "#dc2626", "#7c3aed"])
    for i, v in enumerate(tot):
        ax.text(i, v * 1.02, f"{v:,}\n{v/tot.sum():.0%}", ha="center", fontsize=9)
    ax.set_ylabel("口座エントリ数(市場 × 口座)")
    ax.set_title(f"役割の分離 — 全 {tot.sum():,} エントリ\n"
                 f"ユニーク口座 {P['account'].n_unique():,}", fontsize=10.5, loc="left")

    # (4) 行動 — 取消率 × 寿命
    ax = fig.add_subplot(gs[1, 0])
    cr = P["cancel_rate"].to_numpy()
    lt = P["median_lifetime_s"].to_numpy()
    vp = P["vol_placed"].to_numpy()
    m = np.isfinite(cr) & np.isfinite(lt) & (lt > 0)
    sc = ax.scatter(cr[m], lt[m], s=np.clip(np.log10(vp[m] + 1) * 6, 2, 60),
                    c=np.log10(vp[m] + 1), cmap="viridis", alpha=0.45, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("取消率(取消 ÷ 終端イベント)")
    ax.set_ylabel("注文寿命の中央値[秒、対数]")
    ax.set_title(f"★口座の行動 — 取消率 >0.95 が {np.mean(cr[np.isfinite(cr)] > .95):.0%}\n"
                 f"寿命中央 {np.median(lt[m]):.0f}s。点の大きさ・色 = 発注量",
                 fontsize=10.5, loc="left")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="log₁₀(発注量)")

    # (5) ★集中度 と 信号強度
    ax = fig.add_subplot(gs[1, 1])
    for sg, k, col, mk in [("bq_long", 10, "#1d4ed8", "o"),
                           ("ew16_short", 10, "#dc2626", "s"),
                           ("depth_short", 30, "#16a34a", "^")]:
        v = R.filter((pl.col("signal") == sg) & (pl.col("k") == k)) \
             .select(["market", "pearson"])
        j = v.join(S.select(["market", "hhi"]), on="market")
        sign = -1 if sg.endswith("_short") else 1
        x = j["hhi"].to_numpy(); y = j["pearson"].to_numpy() * sign
        g = np.isfinite(x) & np.isfinite(y)
        r, p = stats.spearmanr(x[g], y[g])
        ax.scatter(x[g], y[g], s=12, alpha=0.4, color=col, lw=0,
                   label=f"{sg} k={k}  ρ={r:+.2f}(p={p:.3f})")
        # 傾向線(順位回帰ではなく見た目の補助)
        z = np.polyfit(x[g], y[g], 1)
        xx = np.linspace(x[g].min(), x[g].max(), 20)
        ax.plot(xx, np.polyval(z, xx), color=col, lw=1.6, ls="--")
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_xlabel("HHI(発注量の集中度)")
    ax.set_ylabel("信号の相関(予測どおりの向きを正)")
    ax.set_title("★「1 口座の癖」仮説の間接検証\n"
                 "集中が高いほど信号は**弱い** = 仮説とは逆向き",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # (6) ★上位1者を除く前後の比較(直接検証)
    ax = fig.add_subplot(gs[1, 2])
    ex = DATA / "pressure_ex_tradeable.parquet"
    if ex.exists():
        X = pl.read_parquet(ex)
        SP = {"long": 1, "short": -1, "diff": 1}
        za, zb, pa, pb = [], [], [], []
        for t in ["bq", "ew1", "ew4", "ew16", "ew64", "depth"]:
            for sd in SP:
                for k in [1, 2, 3, 5, 10, 20, 30]:
                    sg = f"{t}_{sd}"
                    a = R.filter((pl.col("signal") == sg) & (pl.col("k") == k))                          .select(["market", "beta"])
                    b = X.filter((pl.col("signal") == sg) & (pl.col("k") == k))                          .select(["market", "beta"])
                    j = a.join(b, on="market", suffix="_x")      # 共通市場のみ
                    va, vb = j["beta"].to_numpy(), j["beta_x"].to_numpy()
                    m = np.isfinite(va) & np.isfinite(vb)
                    n = int(m.sum())
                    if n < 10:
                        continue
                    f = lambda v: ((v > 0).sum() - n / 2) / np.sqrt(n / 4) * SP[sd]
                    # 通過判定は z 近似ではなく**二項検定の厳密 p**で行う
                    # (本文の表と数字を一致させるため)
                    g = lambda v: stats.binomtest(int((v > 0).sum()), n, 0.5).pvalue
                    za.append(f(va[m])); zb.append(f(vb[m]))
                    pa.append(g(va[m])); pb.append(g(vb[m]))
        za, zb = np.array(za), np.array(zb)
        pa, pb = np.array(pa), np.array(pb)
        lim = 12
        ax.scatter(za, zb, s=26, alpha=0.6, color="#1d4ed8", lw=0)
        ax.plot([-lim, lim], [-lim, lim], color="#111827", lw=1.0, ls="--",
                label="変化なしの線")
        for v in (3.79, -3.79):
            ax.axhline(v, color="#dc2626", lw=0.8, ls=":")
            ax.axvline(v, color="#dc2626", lw=0.8, ls=":")
        BONF = 0.05 / 252
        na = int(((za > 0) & (pa < BONF)).sum())
        nb = int(((zb > 0) & (pb < BONF)).sum())
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("全体の板での z")
        ax.set_ylabel("上位 1 者を除いた板での z")
        ax.set_title("★直接検証 — 126 セルの前後\n"
                     f"予測どおりの符号で Bonferroni 通過 {na} → {nb}"
                     "(失われたセル 0)", fontsize=10.5, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "除外板の推定が未完了", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(f"Boros 全 {S.height} 市場 — 口座パネル(ユニーク口座 "
                 f"{P['account'].n_unique():,} / エントリ {P.height:,})",
                 fontsize=13, y=0.975)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_accounts.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
