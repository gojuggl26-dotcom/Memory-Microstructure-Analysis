"""翌日の建玉を当日の出来高に回帰する(OLS と AR(1) 誤差の GLS)。あわせて回転率を出す。

## 時間契約(先読みをしていないことの明示)

- 説明変数 Volume_t は **t 日の終了時点で確定** する。
- 目的変数 OI_(t+1) は **t+1 日の期間平均** であり、その期間は t 日の終了後に始まる。
- したがって「説明変数が確定する時刻 <= 目的変数の期間の開始時刻」を満たす。

比較のため、同時点の関係(OI_t を Volume_t に回帰)も併せて出力する。
予測力と同時性は別物なので、報告では必ず区別すること。

## 単位

出来高と建玉はどちらも枚数で取る。名目ドルで取ると、期間中に価格が 1.5 倍に
なった影響が両辺に共通して入り、見かけの相関が水増しされるため。

    uv run python scripts/regress_oi_volume.py --coin xyz:MU
出力: charts/<coin>_scatter_oi_volume.png, reports 用の数値を標準出力へ
"""

from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
OPEN_C, CLOSED_C = "#2a78d6", "#eb6834"     # 立会日 / 休場日(検証済みパレットの 1 番と 2 番)


def fit_all(y: np.ndarray, x: np.ndarray) -> dict:
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    glsar = sm.GLSAR(y, X, rho=1)          # rho はモデル側が保持する
    gls = glsar.iterative_fit(maxiter=50)
    return {"ols": ols, "hac": hac, "gls": gls, "rho": float(np.asarray(glsar.rho).ravel()[0]),
            "dw": float(sm.stats.stattools.durbin_watson(ols.resid))}


def show(tag: str, f: dict) -> None:
    o, h, g = f["ols"], f["hac"], f["gls"]
    print(f"\n--- {tag} ---")
    print(f"  OLS   切片 {o.params[0]:12,.0f}  傾き {o.params[1]:8.4f}  "
          f"標準誤差 {o.bse[1]:.4f}  t {o.tvalues[1]:6.2f}  p {o.pvalues[1]:.3g}  R2 {o.rsquared:.3f}")
    print(f"  HAC   (Newey-West, 5 lag) 傾きの標準誤差 {h.bse[1]:.4f}  t {h.tvalues[1]:6.2f}  p {h.pvalues[1]:.3g}")
    print(f"  GLS   切片 {g.params[0]:12,.0f}  傾き {g.params[1]:8.4f}  "
          f"標準誤差 {g.bse[1]:.4f}  t {g.tvalues[1]:6.2f}  p {g.pvalues[1]:.3g}")
    print(f"  残差の自己相関 rho {f['rho']:+.3f} / Durbin-Watson {f['dw']:.3f}  (2 に近いほど無相関)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = pl.read_parquet(ROOT / "data" / f"daily_oi_volume_{tag}.parquet").sort("d")
    cal = xc.get_calendar("XNYS")
    sess = {x.date() for x in cal.sessions_in_range(str(d["d"][0]), str(d["d"][-1]))}
    d = d.with_columns(open_day=pl.col("d").is_in(list(sess)))

    print("=== 回転率 Turnover_t = Volume_t / OI_t(ともに枚数、無次元)===")
    t = d["turnover"]
    print(f"  平均 {t.mean():.3f} / 中央値 {t.median():.3f} / 最小 {t.min():.3f}"
          f"({d.filter(pl.col('turnover') == t.min())['d'][0]}) / "
          f"最大 {t.max():.3f}({d.filter(pl.col('turnover') == t.max())['d'][0]})")
    g = d.group_by("open_day").agg(n=pl.len(), mean=pl.col("turnover").mean(),
                                   med=pl.col("turnover").median()).sort("open_day", descending=True)
    for r in g.iter_rows(named=True):
        lab = "立会日" if r["open_day"] else "休場日"
        print(f"  {lab}({r['n']:2d} 日): 平均 {r['mean']:.3f} / 中央値 {r['med']:.3f}")

    # --- 回帰用の並び。y は 1 日先へずらす ---------------------------------
    vol = d["volume"].to_numpy()
    oi = d["oi_mean"].to_numpy()
    x, y = vol[:-1], oi[1:]                    # Volume_t と OI_(t+1)
    day_t = d["d"].to_list()[:-1]
    open_t = np.array(d["open_day"].to_list()[:-1])
    print(f"\n観測数 {len(x)}(99 日から翌日の無い最終日を除く)")

    main_fit = fit_all(y, x)
    show("OI_(t+1) ~ Volume_t(予測の関係)", main_fit)
    show("OI_t ~ Volume_t(同時点の関係。予測ではない)", fit_all(oi[:-1], vol[:-1]))

    # 建玉の水準を対照に置く。当日の建玉で説明できる分を除いても出来高が効くか
    dy = oi[1:] - oi[:-1]
    show("OI_(t+1) - OI_t ~ Volume_t(前日の水準を対照にした場合)", fit_all(dy, x))

    # プラセボ: 説明変数を時間方向にずらす
    print("\n=== プラセボ検定(説明変数を k 日ずらす。関係が消えなければ疑わしい)===")
    for k in (-5, -3, -1, 0, 1, 3, 5):
        xs = np.roll(vol[:-1], k)
        f = sm.OLS(y, sm.add_constant(xs)).fit()
        mark = "  <- 本来の対応" if k == 0 else ""
        print(f"  k={k:+d}: 傾き {f.params[1]:8.4f}  t {f.tvalues[1]:6.2f}  R2 {f.rsquared:.3f}{mark}")

    # 立会日と休場日で出来高の水準も性質も違うので、群ごとに別々に当てはめる。
    # ここでの AR(1) は「その群の中で連続する観測」の間の相関を指す。立会日なら
    # 連続する立会日どうし、休場日なら土・日・土… という並びになる点に注意。
    print()
    print("=== 群ごとの回帰(図に描く 4 本の線)===")
    grp = {}
    for lab, m in (("立会日", open_t), ("休場日", ~open_t)):
        grp[lab] = fit_all(y[m], x[m])
        show(f"{lab}のみ(n={int(m.sum())})", grp[lab])

    # --- 散布図 -------------------------------------------------------------
    mpl.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
                         "axes.unicode_minus": False, "text.parse_math": False,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(10.5, 7.4), dpi=170)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.scatter(x[open_t] / 1e3, y[open_t] / 1e3, s=42, color=OPEN_C, alpha=0.85,
               edgecolors=SURFACE, linewidths=1.0, zorder=3)
    ax.scatter(x[~open_t] / 1e3, y[~open_t] / 1e3, s=42, color=CLOSED_C, alpha=0.85,
               edgecolors=SURFACE, linewidths=1.0, zorder=3)

    # 線は、その群が実際に取った出来高の範囲だけに引く。範囲外へ伸ばすと
    # 観測の無いところを当てはめているように見えてしまうため。
    DASH = (0, (5, 3))
    handles = [
        Line2D([], [], marker="o", ls="", color=OPEN_C, markersize=8, label="t 日が立会日"),
        Line2D([], [], marker="o", ls="", color=CLOSED_C, markersize=8, label="t 日が休場日"),
    ]
    for lab, m, col in (("立会日", open_t, OPEN_C), ("休場日", ~open_t, CLOSED_C)):
        xs = np.linspace(x[m].min(), x[m].max(), 100)
        for key, ls in (("ols", "-"), ("gls", DASH)):
            f = grp[lab][key]
            ax.plot(xs / 1e3, (f.params[0] + f.params[1] * xs) / 1e3, ls=ls, color=col,
                    lw=2.2, zorder=4, solid_capstyle="round")
            handles.append(Line2D([], [], color=col, lw=2.2, ls=ls,
                                  label=f"{lab} {key.upper()}  傾き {f.params[1]:.3f}"))

    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)

    ax.set_xlabel("当日の出来高 Volume_t(千枚)", color=INK2, fontsize=10.5)
    ax.set_ylabel("翌日の平均建玉 OI_(t+1)(千枚)", color=INK2, fontsize=10.5)
    ax.set_title(f"{a.coin} 当日の出来高と翌日の建玉", loc="left", color=INK, fontsize=13,
                 pad=44, weight="bold")
    ax.text(0, 1.085,
            f"n = {len(x)}(立会日 {int(open_t.sum())} / 休場日 {int((~open_t).sum())})。"
            "説明変数は t 日の終了時点で確定し、目的変数は t+1 日の平均なので先読みは無い。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.text(0, 1.040,
            f"GLS は群ごとの AR(1) 誤差(rho = 立会日 {grp['立会日']['rho']:+.2f} / "
            f"休場日 {grp['休場日']['rho']:+.2f})を仮定。線はその群が実際に取った出来高の範囲だけに引いている。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    fig.text(0.005, 0.008, "出所: Hyperliquid L4 (Artemis) node_fills を再構成 / 窓 2026-05-04〜08-10",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.865, bottom=0.085)
    out = ROOT / "charts" / f"{tag}_scatter_oi_volume.png"
    fig.savefig(out)
    print(f"\n[chart] {out}")

    d.select("d", "open_day", "oi_mean", "volume", "turnover").write_csv(
        ROOT / "data" / f"daily_turnover_{tag}.csv")


if __name__ == "__main__":
    main()
