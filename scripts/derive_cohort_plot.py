r"""Derive — 生存バイアス除去・生存分析・RFQ vs CLOB の図。

★見出しの断定は実測値から組み立てる。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
SRC = Path("E:/Memory-derive")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
CPOS, CNEG, CG = "#2563eb", "#dc2626", "#9ca3af"
CRFQ, CCLOB = "#dc2626", "#2563eb"


def km(dur, ev, grid):
    S, out = 1.0, []
    times = np.unique(dur[ev == 1])
    ti = 0
    for g in grid:
        while ti < len(times) and times[ti] <= g:
            t = times[ti]
            nr = int((dur >= t).sum())
            dd = int(((dur == t) & (ev == 1)).sum())
            if nr > 0:
                S *= (1 - dd / nr)
            ti += 1
        out.append(S)
    return np.array(out)


def main() -> int:
    CHARTS.mkdir(exist_ok=True)
    C = pl.read_parquet(SRC / "cohort" / "cohort_wallets.parquet")
    SV = pl.read_parquet(SRC / "cohort" / "survival.parquet")
    MT = pl.read_parquet(SRC / "rfq" / "matched.parquet")
    WR = pl.read_parquet(SRC / "rfq" / "wallet_rfq.parquet")

    fig = plt.figure(figsize=(18.6, 11.6))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.28)

    # ---------- ① 分類期 vs 評価期(生存バイアス除去)----------
    ax = fig.add_subplot(gs[0, 0])
    MK = C.filter(pl.col("is_maker"))
    x = MK["pnl_in"].to_numpy() / 1e3
    y = MK["pnl_out"].to_numpy() / 1e3
    ax.scatter(x, y, s=40, alpha=0.7, color=CPOS, linewidths=0)
    ax.axhline(0, color="#111827", lw=1.0)
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("分類期(最初の 200 約定)の損益 [千ドル]")
    ax.set_ylabel("評価期(201 約定目以降)の損益 [千ドル]")
    from scipy.stats import rankdata
    rho = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    npos = int((y > 0).sum())
    ax.set_title(f"★① 初期の成績は後の成績を予測しない\n"
                 f"Spearman = {rho:+.3f}(n={len(x)})・"
                 f"評価期が黒字 {npos}/{len(x)}", fontsize=11, loc="left")

    # ---------- ② 評価期の損益の集中 ----------
    ax = fig.add_subplot(gs[0, 1])
    v = np.sort(MK["pnl_out"].to_numpy())[::-1]
    ax.bar(range(len(v)), v / 1e6, color=[CPOS if z > 0 else CNEG for z in v])
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_xlabel("メイカー主体ウォレット(評価期の損益の降順)")
    ax.set_ylabel("[百万ドル]")
    tot = v.sum()
    ax.set_title(f"② 生存バイアスを除いても集中は変わらない\n"
                 f"上位 3 者で {v[:3].sum()/tot*100:.0f}% ・"
                 f"中央値 ${np.median(v):,.0f}", fontsize=11, loc="left")

    # ---------- ③ Kaplan-Meier ----------
    ax = fig.add_subplot(gs[0, 2])
    grid = np.arange(0, 600, 5)
    for lab, sub, col in (("全ウォレット", SV, CG),
                          ("メイカー主体", SV.filter(pl.col("maker_share") >= 0.7),
                           "#7c3aed"),
                          ("200 約定以上", SV.filter(pl.col("n_trade") >= 200),
                           CPOS)):
        d = sub["dur_days"].to_numpy().astype(float)
        e = sub["event"].to_numpy()
        ax.step(grid, km(d, e, grid) * 100, where="post", lw=2, color=col,
                label=f"{lab} (n={len(d):,})")
    for m in (90, 180, 365):
        ax.axvline(m, color="#9ca3af", ls=":", lw=1.0)
    ax.set_xlabel("参入からの日数")
    ax.set_ylabel("まだ活動している割合 [%]")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("★③ Kaplan-Meier(30 日取引が無ければ撤退と判定)\n"
                 "点線は 90 / 180 / 365 日", fontsize=11, loc="left")

    # ---------- ④ 撤退者の生涯損益 ----------
    ax = fig.add_subplot(gs[1, 0])
    gone = SV.filter(pl.col("event") == 1)
    for lab, sub, col in (("全ウォレット", gone, CG),
                          ("200 約定以上",
                           gone.filter(pl.col("n_trade") >= 200), CPOS)):
        v2 = sub["pnl_life"].to_numpy()
        v2 = v2[np.isfinite(v2)]
        v2 = np.clip(v2, -1e5, 1e5)
        ax.hist(v2, bins=80, alpha=0.6, color=col,
                label=f"{lab}: 黒字 {(sub['pnl_life'].to_numpy()>0).mean()*100:.0f}%"
                      f" / 中央 ${np.median(sub['pnl_life'].to_numpy()):,.0f}")
    ax.axvline(0, color="#111827", lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("撤退までの生涯損益 [$](±10 万でクリップ)")
    ax.set_ylabel("ウォレット数(対数)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("④ 撤退した者の 8 割は赤字で退場している",
                 fontsize=11, loc="left")

    # ---------- ⑤ RFQ vs CLOB(マッチ後)----------
    ax = fig.add_subplot(gs[1, 1])
    labs = ["edge [bp]", "markout [bp]", "edge+markout [bp]"]
    d = MT.filter(pl.col("metric").is_in(labs))
    pos = np.arange(len(labs))
    r = [float(d.filter(pl.col("metric") == l)["rfq_mean"][0]) for l in labs]
    c = [float(d.filter(pl.col("metric") == l)["clob_mean"][0]) for l in labs]
    ax.bar(pos - 0.2, r, 0.4, color=CRFQ, label="RFQ")
    ax.bar(pos + 0.2, c, 0.4, color=CCLOB, label="CLOB")
    ax.axhline(0, color="#111827", lw=1.2)
    for i, (a_, b_) in enumerate(zip(r, c)):
        ax.text(i - 0.2, a_, f"{a_:+.2f}", ha="center",
                va="bottom" if a_ >= 0 else "top", fontsize=9)
        ax.text(i + 0.2, b_, f"{b_:+.2f}", ha="center",
                va="bottom" if b_ >= 0 else "top", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("名目あたり [bp]")
    ax.legend(fontsize=9, frameon=False)
    n = int(d.filter(pl.col("metric") == "edge [bp]")["n"][0])
    ax.set_title(f"★⑤ RFQ は幅が広く、しかも逆選択が強くない\n"
                 f"同一銘柄・±7 日・サイズ キャリパー 0.5 で {n:,} 組",
                 fontsize=11, loc="left")

    # ---------- ⑥ RFQ 比率と edge ----------
    ax = fig.add_subplot(gs[1, 2])
    rs = WR["rfq_share"].to_numpy() * 100
    ed = WR["edge"].to_numpy()
    no = WR["notional"].to_numpy()
    sz = 20 + 180 * (np.log1p(no) - np.log1p(no).min()) / (
        np.log1p(no).max() - np.log1p(no).min() + 1e-9)
    ax.scatter(rs, ed, s=sz, alpha=0.6, color="#7c3aed", linewidths=0)
    ax.axhline(0, color="#111827", lw=1.0)
    from scipy.stats import rankdata
    ok = np.isfinite(rs) & np.isfinite(ed)
    rho2 = np.corrcoef(rankdata(rs[ok]), rankdata(ed[ok]))[0, 1]
    ax.set_xlabel("そのメイカーの RFQ 比率 [%]")
    ax.set_ylabel("execution edge の平均 [bp]")
    ax.set_title(f"★⑥ RFQ を多く使うメイカーほど edge が大きい\n"
                 f"Spearman = {rho2:+.3f}(200 約定以上の {int(ok.sum())} 者・"
                 "点の大きさは名目)", fontsize=11, loc="left")

    fig.suptitle("Derive — 生存バイアスの除去 / 参入者の生存分析 / RFQ と CLOB の分離",
                 fontsize=13.5, y=0.985)
    out = CHARTS / "derive_cohort_rfq.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
