r"""特徴量 34 の図 — 注文の生存時間(競合リスク)と、板の年齢の予測力。"""
from __future__ import annotations

import glob
import re
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
HOR = [1, 2, 3, 5, 10, 20, 30]
FEATS = ["age_diff", "age_best_diff", "age_level", "n_ord_diff"]
FLAB = {"age_diff": "板全体の年齢の差", "age_best_diff": "★最良レベルの年齢の差",
        "age_level": "板全体の古さ(水準)", "n_ord_diff": "本数の偏り(既存の言い換え)"}


def marks(ax):
    for t, lab in [(60, "1分"), (3600, "1時間"), (86400, "1日"), (7 * 86400, "1週")]:
        ax.axvline(t, color="#9ca3af", lw=0.7, ls=":")
        ax.text(t, ax.get_ylim()[1] * 0.96, lab, fontsize=7, ha="center", color="#4b5563")


def main() -> int:
    C = pl.read_parquet(DATA / "lifetime_curves.parquet")
    S = pl.read_parquet(DATA / "lifetime_summary.parquet")
    T = pl.read_parquet(DATA / "age_test.parquet")

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.30)

    # (1) 生存関数
    ax = fig.add_subplot(gs[0, 0])
    for lab, col in [("全体", "#111827"), ("支配的口座", "#dc2626"),
                     ("その他の口座", "#2563eb")]:
        d = C.filter(pl.col("stratum") == lab)
        ax.plot(np.maximum(d["t"].to_numpy(), 0.5), d["S"].to_numpy(), lw=2.2, color=col,
                label=f"{lab}(中央 "
                      f"{S.filter(pl.col('stratum')==lab)['median_life_s'][0]:.0f}s)")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("経過時間[秒、対数]"); ax.set_ylabel("まだ板にある確率 S(t)")
    ax.set_title("★生存関数 — 支配的口座は倍の速さで消える\n"
                 "Kaplan–Meier(右打ち切り 0.3%)", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False); marks(ax)

    # (2) 競合リスク
    ax = fig.add_subplot(gs[0, 1])
    d = C.filter(pl.col("stratum") == "全体")
    t = np.maximum(d["t"].to_numpy(), 0.5)
    ax.plot(t, d["cif_cancel"].to_numpy(), lw=2.2, color="#dc2626", label="取消される")
    ax.plot(t, d["cif_fill"].to_numpy(), lw=2.2, color="#16a34a", label="約定する")
    ax.plot(t, d["S"].to_numpy(), lw=1.6, color="#6b7280", ls="--", label="まだ板にある")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("経過時間[秒、対数]"); ax.set_ylabel("累積確率")
    fin = S.filter(pl.col("stratum") == "全体")["cif_fill_final"][0]
    ax.set_title(f"★競合リスク — 約定は最終 {fin:.1%} で頭打ち\n"
                 "「約定までの時間の中央値」は存在しない", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="center left"); marks(ax)

    # (3) 距離帯別の約定確率
    ax = fig.add_subplot(gs[0, 2])
    labs = ["距離 最も近い 25%", "距離 25-50%", "距離 50-75%", "距離 最も遠い 25%"]
    cols = ["#16a34a", "#0891b2", "#f59e0b", "#dc2626"]
    for lab, col in zip(labs, cols):
        d = C.filter(pl.col("stratum") == lab)
        ax.plot(np.maximum(d["t"].to_numpy(), 0.5), d["cif_fill"].to_numpy(),
                lw=2, color=col,
                label=f"{lab.replace('距離 ','')}"
                      f"({S.filter(pl.col('stratum')==lab)['cif_fill_final'][0]:.2%})")
    ax.set_xscale("log")
    ax.set_xlabel("経過時間[秒、対数]"); ax.set_ylabel("累積約定確率")
    near = S.filter(pl.col("stratum") == labs[0])["cif_fill_final"][0]
    far = S.filter(pl.col("stratum") == labs[-1])["cif_fill_final"][0]
    ax.set_title(f"mid からの距離別の約定確率\n"
                 f"最も近い群と最も遠い群で {near/far:.1f} 倍",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False); marks(ax)

    # (4) ハザード(古い注文ほど消えにくいか)
    ax = fig.add_subplot(gs[1, 0])
    for lab, col in [("全体", "#111827"), ("支配的口座", "#dc2626"),
                     ("その他の口座", "#2563eb")]:
        d = C.filter(pl.col("stratum") == lab).sort("t")
        tt = d["t"].to_numpy(); Sv = d["S"].to_numpy(); nr = d["n_risk"].to_numpy()
        # 区間ハザード率 = 1 − S(t_i)/S(t_{i-1}) を区間幅で割る
        with np.errstate(divide="ignore", invalid="ignore"):
            h = 1.0 - Sv[1:] / np.maximum(Sv[:-1], 1e-12)
            w = np.diff(tt)
            rate = np.where(w > 0, h / w, np.nan)
        ok = np.isfinite(rate) & (nr[1:] > 500) & (tt[1:] > 1)
        ax.loglog(tt[1:][ok], np.maximum(rate[ok], 1e-9), lw=2, color=col, label=lab)
    ax.set_xlabel("経過時間[秒、対数]"); ax.set_ylabel("瞬間ハザード[1/秒、対数]")
    ax.set_title("★数秒で山を打った後は 5 桁にわたり減少\n"
                 "古い注文ほど消えにくい(置きっぱなしの層が残る)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (5) ★予測力
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(HOR))
    cols2 = {"age_diff": "#2563eb", "age_best_diff": "#16a34a",
             "age_level": "#7c3aed", "n_ord_diff": "#9ca3af"}
    for f_ in FEATS:
        z = []
        for k in HOR:
            v = T.filter((pl.col("feature") == f_) & (pl.col("k") == k))["beta"].to_numpy()
            v = v[np.isfinite(v)]
            z.append(((v > 0).sum() - len(v) / 2) / np.sqrt(len(v) / 4) if len(v) >= 10 else np.nan)
        ax.plot(x, z, "o-", color=cols2[f_], lw=2, ms=5, label=FLAB[f_],
                alpha=0.5 if f_ == "n_ord_diff" else 1.0)
    ax.axhline(0, color="#111827", lw=1.0)
    for v in (3.29, -3.29):
        ax.axhline(v, color="#dc2626", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in HOR])
    ax.set_xlabel("予測ホライズン k(イベント)")
    ax.set_ylabel("符号検定 z(★事前予測なし・両側)")
    ax.set_title("★板の年齢は将来の金利を語る\n"
                 "点線 = Bonferroni(0.05/28)。ブロック跨ぎ限定",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # (6) 直交性
    ax = fig.add_subplot(gs[1, 2])
    mids = sorted(int(re.search(r"age_book_(\d+)", f).group(1))
                  for f in glob.glob(str(DATA / "age_book_*.parquet")))[:60]
    refs = ["bq_diff", "depth_diff", "nlv_diff", "slope_diff", "spread"]
    acc = {(a, b): [] for a in ["age_diff", "age_best_diff", "n_ord_diff"] for b in refs}
    for mid in mids:
        A = pl.read_parquet(DATA / f"age_book_{mid}.parquet")
        E = pl.read_parquet(DATA / f"event_book_{mid}.parquet")
        if A.height != E.height:
            continue
        J = E.join(A.select(["ev_i", "age_w_long", "age_w_short", "age_best_long",
                             "age_best_short", "n_ord_long", "n_ord_short"]),
                   on="ev_i").filter(~pl.col("extrapolated"))
        if J.height < 2000:
            continue
        l1 = lambda c: np.log1p(np.maximum(J[c].to_numpy().astype(float), 0))
        F = {"age_diff": l1("age_w_long") - l1("age_w_short"),
             "age_best_diff": l1("age_best_long") - l1("age_best_short"),
             "n_ord_diff": l1("n_ord_long") - l1("n_ord_short")}
        R = {"depth_diff": l1("depth_long") - l1("depth_short"),
             "bq_diff": l1("bq_long") - l1("bq_short"),
             "slope_diff": J["slope_diff"].to_numpy(),
             "nlv_diff": l1("n_lv_long") - l1("n_lv_short"),
             "spread": J["spread_pp"].to_numpy()}
        for a, va in F.items():
            for b in refs:
                vb = R[b]
                m = np.isfinite(va) & np.isfinite(vb)
                if m.sum() > 1000:
                    acc[(a, b)].append(stats.spearmanr(va[m], vb[m]).statistic)
    F3 = ["age_best_diff", "age_diff", "n_ord_diff"]
    M = np.array([[np.median(acc[(a, b)]) if acc[(a, b)] else np.nan for b in refs] for a in F3])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(refs))); ax.set_xticklabels(refs, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(F3))); ax.set_yticklabels(F3, fontsize=8.5)
    for i in range(len(F3)):
        for j in range(len(refs)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M[i, j]) > 0.6 else "#111827",
                    fontweight="bold" if abs(M[i, j]) > 0.7 else "normal")
    ax.grid(False)
    ax.set_title("★既存の特徴量との相関(60 市場の中央値)\n"
                 "n_ord_diff は n_lv_diff と ρ=0.99 = 言い換え", fontsize=10.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    n = S.filter(pl.col("stratum") == "全体")["n"][0]
    fig.suptitle(f"Boros — 注文の生存時間(特徴量 34)。全 188 市場 / 注文 {n:,} 件",
                 fontsize=13, y=0.975)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_lifetime.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
