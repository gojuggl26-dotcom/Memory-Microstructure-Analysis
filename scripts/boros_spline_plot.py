"""平滑化スプラインの結果を図にする。

【何を見せるか】
  1. 当てはめた f̂(x) と**日ブロック・ブートストラップの 95% 帯**、
     ノンパラのビン平均、x の分布(下段のヒストグラム)
  2. λ の選択曲線(時間ブロック CV)。格子の端で止まっていないかを目視できるようにする
  3. 実測とプラセボの標本外相関の比較 — **帰無の幅**が一目で判るように

【設計方針】
  ・信頼帯がゼロ線をまたぐかどうかが結論なので、**ゼロ線を必ず引く**
  ・「有意でない」を「効果がない」と読ませないため、**プラセボの散らばりを併記**する
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
# ★font.family にフォント名の列を直接入れると解決に失敗し、
#   軸ラベル・凡例・目盛りが**丸ごと消える**(最初これで気づかず出力した)。
#   generic family + sans-serif の優先列で指定するのが正しい。
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 130,
})
C_FIT, C_CI, C_BIN, C_PL = "#2563eb", "#93c5fd", "#f59e0b", "#9ca3af"


def main(market: int = 163) -> int:
    res = json.loads((DATA / f"spline_{market}.json").read_text(encoding="utf-8"))
    ev = pl.read_parquet(DATA / f"event_book_{market}.parquet") \
           .filter(~pl.col("extrapolated"))
    R = res["results"]
    specs = [(r["x"], r["horizon"], r["yspec"]) for r in R]
    xcols = ["slope_diff", "spread_pp"]
    hs = sorted({h for _, h, _ in specs})
    ylab = "因果標準化"                                    # 体制差を除いた版を主図にする

    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 2.2], hspace=0.42, wspace=0.24)

    for c, xc in enumerate(xcols):
        ax = fig.add_subplot(gs[0, c])
        for h, col, ls in zip(hs, ["#2563eb", "#dc2626"], ["-", "--"]):
            r = next(z for z in R if z["x"] == xc and z["horizon"] == h
                     and z["yspec"] == ylab)
            xg = np.array(r["grid_x"]); fg = np.array(r["grid_f"])
            ax.fill_between(xg, r["ci_lo"], r["ci_hi"], color=col, alpha=0.13, lw=0)
            ax.plot(xg, fg, color=col, ls=ls, lw=2.0,
                    label=f"{h}s  λ={r['lambda_cv']:.3g}  edf={r['edf']:.1f}")
            bx, by = np.array(r["bin_x"]), np.array(r["bin_y"])
            m = np.isfinite(bx) & np.isfinite(by)
            ax.plot(bx[m], by[m], "o", ms=3.4, color=col, alpha=0.5, mew=0)
        ax.axhline(0, color="#111827", lw=0.9)
        nm = "book_slope_diff  (LONG − SHORT の平均距離, pp)" if xc == "slope_diff" \
             else "spread  (pp)"
        ax.set_xlabel(nm); ax.set_ylabel("将来の mid 変化(因果標準化)")
        ax.set_title(f"f̂ = argmin Σ(y−f(x))² + λ∫f''²   —   {xc}",
                     fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False)
        # 分布
        axh = fig.add_subplot(gs[1, c], sharex=ax)
        v = ev[xc].to_numpy().astype(float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        axh.hist(np.clip(v, lo, hi), bins=90, color="#cbd5e1", edgecolor="none")
        axh.set_ylabel("件数", fontsize=8); axh.set_yticks([])
        axh.set_xlabel("")
        axh.tick_params(labelbottom=False)

    # λ 選択曲線
    ax = fig.add_subplot(gs[2, 0])
    for xc, col in zip(xcols, ["#2563eb", "#dc2626"]):
        for h, ls in zip(hs, ["-", "--"]):
            r = next(z for z in R if z["x"] == xc and z["horizon"] == h
                     and z["yspec"] == ylab)
            lam = np.array(r["lambdas"]); cvv = np.array(r["cv_curve"])
            ax.semilogx(lam, cvv, ls, color=col, lw=1.5, alpha=0.85,
                        label=f"{xc} {h}s")
            ax.plot([r["lambda_cv"]], [max(cvv[np.isfinite(cvv)])], "o",
                    color=col, ms=6, mew=0)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xlabel("λ(粗さ罰則)"); ax.set_ylabel("標本外 R²(時間ブロック CV)")
    ax.set_title("λ の機械的選択 — ● が採用点。格子の端で止まっていないことを確認する",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # 実測 vs プラセボ
    ax = fig.add_subplot(gs[2, 1])
    lab, real, plac = [], [], []
    for r in R:
        lab.append(f"{r['x'].replace('_pp','').replace('slope_diff','slope')}"
                   f"\n{r['horizon']}s {'z' if r['yspec']=='因果標準化' else '生'}")
        real.append(r["cor_spline"]); plac.append(r["cor_placebo"])
    i = np.arange(len(lab))
    pb = np.array([p for p in plac if np.isfinite(p)])
    ax.axhspan(pb.min(), pb.max(), color=C_PL, alpha=0.28, lw=0,
               label=f"プラセボの範囲 [{pb.min():+.3f}, {pb.max():+.3f}]")
    ax.bar(i - 0.2, real, 0.38, color=C_FIT, label="実測(スプライン)")
    ax.bar(i + 0.2, plac, 0.38, color=C_PL, label="プラセボ(x を日内シフト)")
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(i); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("標本外相関(フォールド中央値)")
    ax.set_title("★実測はプラセボの幅の中に収まっている = 検出できていない",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle(
        f"Boros market {market}(KUCOIN-BTCUSDT-31JUL2026)— "
        f"板の形から将来の金利変化を予測できるか / 平滑化スプライン "
        f"min{{Σ(y−f(x))² + λ∫f''(x)²dx}}",
        fontsize=12.5, y=0.985)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / f"boros_spline_{market}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
