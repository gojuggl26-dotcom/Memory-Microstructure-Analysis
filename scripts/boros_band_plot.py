r"""報酬帯(特徴量 55)の図。

出すもの:
  上段 = 帯そのものの性質と、「帯は効いているか」の検証(結果は null)
  下段 = 帯を境にした内外の分解が、信号にどう効くか(結果は強い構造あり)
"""
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
SP = {"long": 1, "short": -1, "diff": 1}


def main() -> int:
    inc = pl.read_parquet(DATA / "incentive_range.parquet").filter(pl.col("ok"))
    B = pl.read_parquet(DATA / "pressure_band_tradeable.parquet")

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.30)

    # (1) 帯の幅の分布
    ax = fig.add_subplot(gs[0, 0])
    rl = inc["range_long"].to_numpy() * 100
    rs = inc["range_short"].to_numpy() * 100
    v = np.concatenate([rl[np.isfinite(rl)], rs[np.isfinite(rs)]])
    ax.hist(np.log10(v), bins=45, color="#bfdbfe", edgecolor="#2563eb")
    ax.axvline(np.log10(np.median(v)), color="#dc2626", lw=2,
               label=f"中央 {np.median(v):.3f} pp")
    for t, lab in [(0.1, "0.1"), (1, "1"), (10, "10")]:
        ax.axvline(np.log10(t), color="#6b7280", lw=0.7, ls=":")
    ax.set_xlabel("log₁₀(報酬帯の半幅 [pp])")
    ax.set_ylabel("市場 × 側")
    ax.set_title("報酬帯の幅 — mid からの片側半幅\n"
                 f"中央 {np.median(v):.2f} pp ≒ {np.median(v)/0.01:.0f} tick",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (2) 発注距離の分布と帯の縁(バンチングは出るか)
    ax = fig.add_subplot(gs[0, 1])
    band = {r["market"]: (r["range_long"], r["range_short"])
            for r in inc.iter_rows(named=True)}
    rel = []
    for f in sorted(glob.glob(str(DATA / "band_place_*.parquet")))[:80]:
        m = int(re.search(r"band_place_(\d+)", f).group(1))
        if m not in band or band[m][0] is None:
            continue
        P = pl.read_parquet(f, columns=["side", "dist_pp"])
        for sd, key in ((0, 0), (1, 1)):
            d = P.filter(pl.col("side") == sd)["dist_pp"].to_numpy()
            d = d[np.isfinite(d) & (d > 0)]
            R = band[m][key] * 100
            if len(d) > 200 and R > 0:
                rel.append(d / R)          # 帯の縁を 1 に正規化
    rel = np.concatenate(rel)
    b = np.geomspace(0.01, 20, 70)
    ax.hist(rel, bins=b, color="#e9d5ff", edgecolor="#7c3aed", lw=0.5)
    ax.axvline(1.0, color="#dc2626", lw=2.2, label="帯の縁(=1)")
    ax.set_xscale("log")
    ax.set_xlabel("発注距離 ÷ 帯の半幅(1 が帯の縁)")
    ax.set_ylabel("発注件数")
    ax.set_title("★帯の縁に不連続は出ない\n"
                 f"縁の内側 {np.mean(rel <= 1):.0%}。山は縁より内で滑らか",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (3) 同一市場内の左右比較(規模を完全に統制した検証)
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for f in sorted(glob.glob(str(DATA / "band_place_*.parquet"))):
        m = int(re.search(r"band_place_(\d+)", f).group(1))
        if m not in band or band[m][0] is None:
            continue
        P = pl.read_parquet(f, columns=["side", "dist_pp"])
        dd = {}
        for sd, key, lab in ((0, 0, "long"), (1, 1, "short")):
            d = P.filter(pl.col("side") == sd)["dist_pp"].to_numpy()
            d = d[np.isfinite(d) & (d > 0)]
            if len(d) < 300:
                dd = {}
                break
            dd[lab] = (band[m][key] * 100, float(np.median(d)))
        if dd:
            rows.append((np.log(dd["long"][0] / dd["short"][0]),
                         np.log(dd["long"][1] / dd["short"][1])))
    R = np.array(rows)
    r_, p_ = stats.spearmanr(R[:, 0], R[:, 1])
    ax.scatter(R[:, 0], R[:, 1], s=16, alpha=0.5, color="#1d4ed8", lw=0)
    ax.axhline(0, color="#111827", lw=0.9); ax.axvline(0, color="#111827", lw=0.9)
    lim = np.percentile(np.abs(R), 98)
    ax.plot([-lim, lim], [-lim, lim], color="#dc2626", lw=1.2, ls="--",
            label="帯が効いていれば この向き")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("log(帯 long ÷ 帯 short)")
    ax.set_ylabel("log(発注距離 long ÷ short)")
    ax.set_title(f"★同一市場内の左右比較({len(R)} 市場)\n"
                 f"ρ = {r_:+.3f}(p={p_:.2f})— **関係なし**",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # (4) 帯内シェア
    ax = fig.add_subplot(gs[1, 0])
    sh = []
    for f in sorted(glob.glob(str(DATA / "band_book_*.parquet"))):
        d = pl.read_parquet(f, columns=["ib_share_long", "ib_share_short",
                                        "extrapolated"]).filter(~pl.col("extrapolated"))
        if d.height < 500:
            continue
        for c in ("ib_share_long", "ib_share_short"):
            v = d[c].to_numpy()
            v = v[np.isfinite(v)]
            if len(v):
                sh.append(float(np.median(v)))
    sh = np.array(sh)
    ax.hist(sh, bins=40, color="#bbf7d0", edgecolor="#16a34a")
    ax.axvline(np.median(sh), color="#dc2626", lw=2, label=f"中央 {np.median(sh):.0%}")
    ax.set_xlabel("帯内の残量 ÷ 片側総量(市場 × 側の中央値)")
    ax.set_ylabel("市場 × 側")
    ax.set_title("★算出した特徴量 — 帯内シェア\n"
                 f"中央 {np.median(sh):.0%}。板の大半は帯の内側にある",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # (5) ★信号の分解
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(HOR))
    for tag, lab, col in [("ib", "帯内", "#16a34a"), ("oob", "帯外", "#dc2626"),
                          ("depth", "片側総量(参照)", "#6b7280")]:
        for s, ls, mk in [("short", "-", "o"), ("long", "--", "s")]:
            z = []
            for k in HOR:
                v = B.filter((pl.col("signal") == f"{tag}_{s}") & (pl.col("k") == k))["beta"].to_numpy()
                v = v[np.isfinite(v)]
                z.append(((v > 0).sum() - len(v) / 2) / np.sqrt(len(v) / 4) * SP[s]
                         if len(v) >= 10 else np.nan)
            ax.plot(x, z, ls, marker=mk, color=col, lw=2, ms=4.5,
                    alpha=1.0 if tag != "depth" else 0.55,
                    label=f"{lab} {s}")
    ax.axhline(0, color="#111827", lw=1.0)
    for v in (3.55, -3.55):
        ax.axhline(v, color="#dc2626", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in HOR])
    ax.set_xlabel("予測ホライズン k(イベント)")
    ax.set_ylabel("符号検定 z(予測どおりの向きを正)")
    ax.set_title("★信号は帯の内側にある — 外側は符号が逆\n"
                 "点線 = Bonferroni(0.05/168)", fontsize=10.5, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")

    # (6) 帯内 / 帯外 の注文の性質
    ax = fig.add_subplot(gs[1, 2])
    fi, fo, li, lo, ti, to = [], [], [], [], [], []
    for f in sorted(glob.glob(str(DATA / "band_place_*.parquet"))):
        P = pl.read_parquet(f)
        if P.height < 500:
            continue
        ib = P["in_band"].to_numpy()
        ok = np.array([v is not None for v in ib])
        ibb = np.array([bool(v) if v is not None else False for v in ib])
        term = P["terminal"].to_numpy(); lt = P["lifetime_s"].to_numpy()
        t1 = P["is_top1"].to_numpy()
        a, b_ = ok & ibb, ok & ~ibb
        if a.sum() < 50 or b_.sum() < 50:
            continue
        fi.append((term[a] == "filled").mean()); fo.append((term[b_] == "filled").mean())
        li.append(np.nanmedian(lt[a])); lo.append(np.nanmedian(lt[b_]))
        ti.append(t1[a].mean()); to.append(t1[b_].mean())
    lab = ["約定率", "上位1者の割合"]
    ain = [np.median(fi), np.median(ti)]
    aout = [np.median(fo), np.median(to)]
    i = np.arange(2)
    ax.bar(i - 0.19, ain, 0.36, color="#16a34a", label="帯内")
    ax.bar(i + 0.19, aout, 0.36, color="#dc2626", label="帯外")
    for j in i:
        ax.text(j, max(ain[j], aout[j]) * 1.04,
                f"{ain[j]/max(aout[j],1e-9):.1f}x", ha="center", fontsize=9)
    ax.set_xticks(i); ax.set_xticklabels(lab)
    ax.set_ylabel("市場ごとの中央値")
    pv = stats.binomtest(int((np.array(fi) > np.array(fo)).sum()), len(fi), 0.5).pvalue
    pt = stats.binomtest(int((np.array(ti) > np.array(to)).sum()), len(ti), 0.5).pvalue
    ax.set_title(f"帯の内外での注文の違い({len(fi)} 市場)\n"
                 f"約定率 p={pv:.1e} / 上位1者 p={pt:.2f}(有意でない)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    fig.suptitle("Boros — メイカー報酬帯(特徴量 55)の算出と検証", fontsize=13, y=0.975)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_reward_band.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
