r"""Lighter 特徴量解析の図(市場概観・予測力総覧・相関行列・十分位・散布図)。

【集計の単位】銘柄ごとに推定 → 銘柄をまたいで中央値。検定は銘柄単位の符号検定
  (n=13 なので検出力は弱い — 表に日数・銘柄数を必ず併記する)。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
★見出しの断定は実測値から組み立てる(結果を見る前に書かない)。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
SRC = Path("E:/Memory-lighter")
ANA = SRC / "ana"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_fam import COLORS, NAME, family  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
SYMS = ["DRAM", "MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD",
        "AAPL", "AMZN", "H100", "MSFT", "NVDA", "TSLA", "XAG", "XAU"]


def sgn(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 5:
        return np.nan, np.nan, 0, n
    p = int((v > 0).sum())
    return float(np.median(v)), float((p - n / 2) / np.sqrt(n / 4)), p, n


def load_rho():
    return pl.concat([pl.read_parquet(f) for f in
                      sorted(glob.glob(str(ANA / "rho_*.parquet")))],
                     how="diagonal_relaxed")


def summarize(R, h):
    rows = []
    for (f,), g in R.filter(pl.col("h") == h).group_by(["feature"]):
        m, z, p, n = sgn(g["spearman"].to_numpy())
        mp, _, _, _ = sgn(g["spearman_pl"].to_numpy())
        mb, _, _, _ = sgn(g["beta_iqr_bp"].to_numpy())
        if n < 8:
            continue
        rows.append({"feature": f, "family": family(f), "rho": m, "z": z,
                     "pos": p, "n": n, "rho_pl": mp, "beta": mb})
    return pl.DataFrame(rows).sort("rho", descending=True)


# ---------------------------------------------------------------- 市場概観
def fig_mktpanel():
    rows = []
    for s in SYMS:
        p = SRC / f"grid_{s}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p, columns=["ts_s", "bb", "ba", "usd_vol",
                                        "W10_b", "W10_a", "n_bbo",
                                        "tb_cnt", "ts_cnt", "suspect"])
        bb = d["bb"].to_numpy(); ba = d["ba"].to_numpy()
        mid = (bb + ba) / 2
        spr = (ba - bb) / mid * 1e4
        days = (d["ts_s"].max() - d["ts_s"].min()) / 86400
        rows.append({
            "sym": s,
            "usd_day": float(d["usd_vol"].sum() / max(days, 1e-9)),
            "spr_med": float(np.nanmedian(spr)),
            "depth_usd": float(np.nanmedian((d["W10_b"].to_numpy()
                                             + d["W10_a"].to_numpy()) * mid)),
            "bbo_s": float(d["n_bbo"].mean()),
            "trades_day": float((d["tb_cnt"].sum() + d["ts_cnt"].sum())
                                / max(days, 1e-9)),
            "cover": float(d.height / (days * 86400)),
            "suspect": float(np.nanmean(d["suspect"].to_numpy())),
        })
    T = pl.DataFrame(rows).sort("usd_day", descending=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6))
    panels = [("usd_day", "日次出来高 [USD]", True),
              ("spr_med", "spread 中央値 [bp]", False),
              ("depth_usd", "mid±10bp の板 [USD]", True),
              ("bbo_s", "BBO 更新 [回/秒]", False),
              ("trades_day", "約定 [件/日]", True),
              ("cover", "グリッド被覆率(1=完全)", False)]
    order = T["sym"].to_list()
    for ax, (c, lab, logy) in zip(axes.ravel(), panels):
        v = T[c].to_numpy()
        ax.bar(range(len(order)), v, 0.65,
               color=["#2563eb" if s not in ("H100",) else "#9ca3af"
                      for s in order])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=60, fontsize=7.5, ha="right")
        ax.set_title(lab, fontsize=10.5, loc="left")
        if logy:
            ax.set_yscale("log")
    fig.suptitle("Lighter 13 銘柄の概観(2026-08-18〜09-05、追加 8 銘柄は 08-22〜)"
                 " — 出来高降順", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = CHARTS / "lighter_mktpanel.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    T.write_parquet(ANA / "mktpanel.parquet")


# ---------------------------------------------------------------- 予測力総覧
def fig_rank(S, h):
    fig = plt.figure(figsize=(17.4, 11.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.27,
                          height_ratios=[1.3, 1.0])
    ax = fig.add_subplot(gs[0, :])
    T = S.sort("rho")
    v = T["rho"].to_numpy()
    fam = T["family"].to_numpy()
    ax.bar(np.arange(len(v)), v, 1.0, color=[COLORS[f] for f in fam],
           linewidth=0)
    plab = float(np.nanpercentile(np.abs(S["rho_pl"].to_numpy()), 95))
    ax.axhspan(-plab, plab, color="#9ca3af", alpha=0.30, zorder=0)
    ax.axhline(0, color="#111827", lw=1.0)
    n_out = int((np.abs(v) > plab).sum())
    ax.set_title(f"★① 全特徴量の予測力(h={h:.0f}s)— 灰色帯 = 帰無対照(巡回シフト)"
                 f"の |ρ| 95%(±{plab:.4f})\n帯の外 {n_out}/{len(v)} 個。"
                 "色は系統(下の凡例)", fontsize=11.5, loc="left")
    ax.set_ylabel("Spearman ρ(銘柄中央値)")
    ax.set_xlabel(f"特徴量 {len(v)} 個(ρ の昇順)")
    hs = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f])
          for f in sorted(set(fam.tolist()))]
    ax.legend(hs, [NAME[f] for f in sorted(set(fam.tolist()))],
              fontsize=7.2, frameon=False, ncol=4, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    fams = sorted(set(S["family"].to_list()))
    data = [S.filter(pl.col("family") == f)["rho"].to_numpy() for f in fams]
    bp = ax.boxplot(data, positions=range(len(fams)), widths=0.62,
                    orientation="horizontal", patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", ms=3, alpha=0.5),
                    medianprops=dict(color="#111827", lw=1.8))
    for p_, f in zip(bp["boxes"], fams):
        p_.set_facecolor(COLORS[f]); p_.set_alpha(0.6)
    ax.axvline(0, color="#111827", lw=1.2)
    ax.axvspan(-plab, plab, color="#9ca3af", alpha=0.28, zorder=0)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([NAME[f] for f in fams], fontsize=8)
    ax.set_xlabel("ρ(銘柄中央値)")
    ax.set_title("② 系統ごとの分布", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    R = load_rho()
    hh = sorted(R["h"].unique().to_list())
    for f in (S.head(5)["feature"].to_list()
              + S.sort("rho").head(3)["feature"].to_list()):
        ys = []
        for h2 in hh:
            g = R.filter((pl.col("feature") == f) & (pl.col("h") == h2))
            ys.append(sgn(g["spearman"].to_numpy())[0])
        ax.plot(range(len(hh)), ys, "o-", lw=1.6, ms=4,
                color=COLORS[family(f)], label=f[:22])
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_xticks(range(len(hh)))
    ax.set_xticklabels([f"{int(x)}s" for x in hh])
    ax.set_xlabel("ホライズン")
    ax.set_ylabel("ρ(銘柄中央値)")
    ax.set_title("③ 上位・下位のホライズン依存", fontsize=11, loc="left")
    ax.legend(fontsize=6.4, frameon=False, ncol=1)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = [f"【ρ 上位 14(h={h:.0f}s)】"]
    for r in S.head(14).iter_rows(named=True):
        txt.append(f"  {r['rho']:+.4f}  {r['pos']:>2}/{r['n']:<2} {r['feature'][:30]}")
    txt.append("")
    txt.append("【ρ 下位 10】")
    for r in S.sort("rho").head(10).iter_rows(named=True):
        txt.append(f"  {r['rho']:+.4f}  {r['pos']:>2}/{r['n']:<2} {r['feature'][:30]}")
    ax.text(0, 1, "\n".join(txt), va="top", ha="left", fontsize=7.4,
            family="monospace", transform=ax.transAxes, linespacing=1.45)
    ax.set_title("④ 順位表(値 / 予測どおりの銘柄数)", fontsize=11, loc="left")
    fig.suptitle(f"Lighter — 特徴量 {S.height} 個の予測力総覧"
                 f"(h={h:.0f}s、銘柄中央値、13 銘柄)", fontsize=14, y=0.965)
    out = CHARTS / "lighter_rank.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------- 相関行列
def fig_corr():
    mats, feats = [], None
    for s in SYMS:
        p = ANA / f"corr_{s}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        if feats is None:
            feats = list(z["features"])
            idx = {f: i for i, f in enumerate(feats)}
            acc = []
        M = np.full((len(feats), len(feats)), np.nan, dtype=np.float32)
        fs = list(z["features"])
        loc = [idx.get(f, -1) for f in fs]
        C = z["corr"]
        for i, li in enumerate(loc):
            if li < 0:
                continue
            for j, lj in enumerate(loc):
                if lj >= 0:
                    M[li, lj] = C[i, j]
        acc.append(M)
    A = np.nanmedian(np.stack(acc), axis=0)
    ns = np.stack(acc).shape[0]
    fam = np.array([family(f) for f in feats])
    order = np.lexsort((feats, fam))
    A2 = A[np.ix_(order, order)]
    fam2 = fam[order]
    fig = plt.figure(figsize=(16.6, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(A2, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    b = np.flatnonzero(np.diff(fam2)) + 0.5
    for x in b:
        ax.axhline(x, color="#111827", lw=0.5, alpha=0.5)
        ax.axvline(x, color="#111827", lw=0.5, alpha=0.5)
    mids = []
    for f in sorted(set(fam2.tolist())):
        w = np.flatnonzero(fam2 == f)
        mids.append((w.mean(), NAME[f]))
    ax.set_yticks([m for m, _ in mids])
    ax.set_yticklabels([n for _, n in mids], fontsize=7.4)
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ(銘柄中央値)")
    ax.set_title(f"★① 特徴量×特徴量の全ペア相関({len(feats)} 列、"
                 f"{ns} 銘柄の中央値)\n黒線 = 系統の境界", fontsize=11.5,
                 loc="left")
    ax.grid(False)

    ax = fig.add_subplot(gs[0, 1])
    iu = np.triu_indices(len(feats), 1)
    v = A[iu]
    v = v[np.isfinite(v)]
    ax.hist(np.abs(v), bins=60, color="#7c3aed", alpha=0.8)
    for thr in (0.5, 0.9, 0.99):
        ax.axvline(thr, color="#dc2626", ls="--", lw=1.0)
        ax.text(thr, ax.get_ylim()[1] * 0.9,
                f" |ρ|>{thr}: {int((np.abs(v) > thr).sum()):,}",
                fontsize=8, rotation=90, va="top")
    ax.set_xlabel("|ρ|(全ペア)")
    ax.set_ylabel("ペア数")
    ax.set_title(f"② 全 {len(v):,} ペアの |ρ| 分布", fontsize=11.5, loc="left")
    fig.suptitle("Lighter — 特徴量どうしの関係(全通り)", fontsize=14, y=0.99)
    out = CHARTS / "lighter_corr.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    np.savez_compressed(ANA / "corr_median.npz", corr=A,
                        features=np.array(feats))


# ---------------------------------------------------------------- 十分位(系統別)
def fig_family(S, K):
    D = pl.concat([pl.read_parquet(f) for f in
                   sorted(glob.glob(str(ANA / "dec_*.parquet")))],
                  how="diagonal_relaxed")
    feats = S.filter(pl.col("family") == K)["feature"].to_list()
    if not feats:
        return
    smap = {r["feature"]: r for r in S.iter_rows(named=True)}
    feats.sort(key=lambda f: -abs(smap[f]["rho"]))
    nc = 6
    nr = int(np.ceil(len(feats) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.45 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(feats):]:
        ax.axis("off")
    col = COLORS[K]
    for ax, f in zip(axes, feats):
        g = D.filter(pl.col("feature") == f)
        ys, lo, hi = [], [], []
        for k in range(1, 11):
            v = g.filter(pl.col("dec") == k)["y_mean"].to_numpy()
            v = v[np.isfinite(v)]
            if len(v) < 8:
                ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                ys.append(np.median(v))
                lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        xs = np.arange(1, 11)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18)
        ax.plot(xs, ys, "o-", color=col, lw=1.7, ms=3.2)
        ax.axhline(0, color="#111827", lw=0.9)
        r = smap[f]
        star = "★" if abs(r["z"]) >= 3.0 else ""
        ax.set_title(f"{star}{f}\nρ={r['rho']:+.4f} ({r['pos']}/{r['n']})",
                     fontsize=7.4, loc="left")
        ax.tick_params(labelsize=6.4)
        ax.set_xticks([1, 5, 10])
    fig.suptitle(f"Lighter {NAME[K]} — 十分位ごとの 60 秒後リターン[bp]"
                 f"(銘柄中央値・帯は四分位)", fontsize=12.5, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = CHARTS / f"lighter_fam{K:02d}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out} ({len(feats)})")


# ---------------------------------------------------------------- 散布図
def fig_scatter(S):
    A = pl.concat([pl.read_parquet(f) for f in
                   sorted(glob.glob(str(ANA / "samp_*.parquet")))],
                  how="diagonal_relaxed")
    mk = A["symbol"].to_numpy()
    y = A["y60"].to_numpy().astype(np.float64)
    ys = np.full(len(y), np.nan)
    for m in np.unique(mk):
        s = mk == m
        v = y[s]
        f = np.isfinite(v)
        sd = v[f].std() if f.sum() > 30 else 0
        if sd > 0:
            ys[s] = (v - v[f].mean()) / sd

    def xrank(c):
        x = A[c].to_numpy().astype(np.float64)
        out = np.full(len(x), np.nan)
        for m in np.unique(mk):
            s = np.flatnonzero(mk == m)
            v = x[s]
            f = np.isfinite(v)
            if f.sum() < 30:
                continue
            o = np.argsort(v[f], kind="stable")
            r = np.empty(f.sum()); r[o] = np.arange(f.sum())
            out[s[f]] = r / max(f.sum() - 1, 1)
        return out

    sel = (S.head(12)["feature"].to_list()
           + S.sort("rho").head(12)["feature"].to_list())
    sel = [c for c in sel if c in A.columns][:24]
    smap = {r["feature"]: r for r in S.iter_rows(named=True)}
    fig, axes = plt.subplots(4, 6, figsize=(18.6, 12.2))
    for ax, f in zip(axes.ravel(), sel):
        X = xrank(f)
        ok = np.isfinite(X) & np.isfinite(ys)
        r = smap[f]
        ax.scatter(X[ok], ys[ok], s=2.5, alpha=0.10,
                   color=COLORS[r["family"]], linewidths=0, rasterized=True)
        e = np.linspace(0, 1, 21)
        k = np.clip(np.searchsorted(e, X[ok], side="right") - 1, 0, 19)
        bx, by = [], []
        for i in range(20):
            m2 = k == i
            if m2.sum() > 30:
                bx.append(X[ok][m2].mean()); by.append(ys[ok][m2].mean())
        ax.plot(bx, by, "o-", color="#111827", lw=1.5, ms=3)
        ax.axhline(0, color="#dc2626", lw=0.9, ls="--")
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f"{f}\nρ={r['rho']:+.4f} ({r['pos']}/{r['n']})",
                     fontsize=7.6, loc="left")
        ax.tick_params(labelsize=6.4)
    fig.suptitle("Lighter — 予測力の上位 12 と下位 12(x=銘柄内順位、"
                 "y=銘柄の sd で割った 60 秒後リターン、黒=20 ビン平均)",
                 fontsize=13.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = CHARTS / "lighter_scatter.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    CHARTS.mkdir(exist_ok=True)
    R = load_rho()
    S = summarize(R, 60.0)
    S.write_parquet(ANA / "summary60.parquet")
    todo = a.only.split(",") if a.only else \
        ["mkt", "rank", "corr", "fam", "scatter"]
    if "mkt" in todo:
        fig_mktpanel()
    if "rank" in todo:
        fig_rank(S, 60.0)
    if "corr" in todo:
        fig_corr()
    if "fam" in todo:
        for K in sorted(set(S["family"].to_list())):
            fig_family(S, K)
    if "scatter" in todo:
        fig_scatter(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
