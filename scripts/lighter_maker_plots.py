r"""Lighter maker 検証の図(第 4 段)。

    uv run python scripts/lighter_maker_plots.py

入力: data/lighter_maker_*.csv(lighter_maker_report.py の出力)
出力: charts/lighter_maker_grid.png / _pnl.png / _micro.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_vol import BASELINE, INK, INK2, MUTED, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_A, C_B, C_C, C_D = "#2a78d6", "#e34948", "#1f8a5e", "#c48a12"
POSN = {-1: "内側1tick", 0: "BBO", 1: "外側1tick"}
LIFE = [0.25, 1.0, 3.0, 10.0]


def gridval(S: pl.DataFrame, key: str, mdl: str = "Q1"):
    """置き場所 × 寿命 の行列を作る。"""
    M = np.full((3, len(LIFE)), np.nan)
    N = np.zeros((3, len(LIFE)), int)
    for a, pos in enumerate((-1, 0, 1)):
        for b, lf in enumerate(LIFE):
            tag = "main_Standard" if (pos == 0 and lf == 1.0) else f"g_p{pos}_l{lf:g}"
            r = S.filter((pl.col("label") == tag) & (pl.col("model") == mdl))
            if r.height:
                M[a, b] = float(r[key][0])
                N[a, b] = int(r["fills"][0])
    return M, N


def heat(ax, M, N, title, fmt="{:+.2f}", cmap="RdYlGn", center=True, unit=""):
    v = M[np.isfinite(M)]
    if v.size == 0:
        ax.text(.5, .5, "データなし", ha="center", transform=ax.transAxes)
        return
    lim = np.nanmax(np.abs(v)) if center else None
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   vmin=-lim if center else np.nanmin(v),
                   vmax=lim if center else np.nanmax(v))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            ax.text(j, i, fmt.format(M[i, j]) + f"\n({N[i, j]})", ha="center",
                    va="center", fontsize=7.5,
                    color=INK if abs(M[i, j]) < (lim or 1) * .6 else "#ffffff")
    ax.set_xticks(range(len(LIFE)), [f"{x:g}s" for x in LIFE], fontsize=8.5)
    ax.set_yticks(range(3), [POSN[p] for p in (-1, 0, 1)], fontsize=8.5)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_xlabel("quote 寿命", fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main() -> None:
    S = pl.read_csv(DATA / "lighter_maker_summary.csv")
    P = pl.read_csv(DATA / "lighter_maker_by_symbol.csv")
    D = pl.read_csv(DATA / "lighter_maker_daily.csv")
    CH.mkdir(exist_ok=True)

    # ---------- 図 1: 置き場所 × 寿命 の格子 ----------
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.4))
    for b, (key, ttl, fmt) in enumerate((
            ("fill_rate", "(a) 約定率(括弧は約定数)", "{:.3%}"),
            ("imp_bp", "(b) 約定時点の入口価格改善 (bp)", "{:+.2f}"),
            ("mk300", "(c) 約定後 300 秒の markout (bp)", "{:+.2f}"),
            ("net_bp", "★(d) maker→taker の純 EV (bp)", "{:+.2f}"))):
        a_ = ax[b // 2, b % 2]
        M, N = gridval(S, key)
        heat(a_, M, N, ttl, fmt=fmt, center=(key != "fill_rate"),
             cmap="RdYlGn" if key != "fill_rate" else "YlGnBu")
    fig.suptitle("Lighter maker — 指値の置き場所 × 寿命(Q1 strict-trade、"
                 "探索期間 20 日・12 銘柄・暫定)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(CH / "lighter_maker_grid.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / "lighter_maker_grid.png")

    # ---------- 図 2: 損益の分解 ----------
    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    # (a) 待ち行列モデル別
    b = ax[0, 0]
    M = S.filter(pl.col("label") == "main_Standard")
    x = np.arange(M.height)
    v = M["net_bp"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(x, v, width=.55, lw=0, color=[C_C if y > 0 else C_B for y in v])
    for i, y in enumerate(v):
        b.text(i, y - .18, f"{y:+.2f}", ha="center", fontsize=9, color=INK)
    b.set_xticks(x, [f"{m}\n約定 {n:,}" for m, n in
                     zip(M["model"].to_list(), M["fills"].to_list())], fontsize=8.5)
    style(b, "", "純 EV (bp / 約定名目)")
    b.set_title("★(a) 待ち行列 3 モデルすべてで負", color=INK, fontsize=9.5, loc="left")

    # (b) 構成別の純 bp
    b = ax[0, 1]
    want = [("主 Standard", "main_Standard", C_A),
            ("Standard maker0ms", "main_Std_mk0", C_A),
            ("Premium", "main_Premium", C_D),
            ("符号反転", "pl_flip", MUTED),
            ("時刻シフト", "pl_shift", MUTED),
            ("定期quote(無信号)", "pl_cadence", MUTED)]
    lb, vv, cc = [], [], []
    for nm, tg, col in want:
        r = S.filter((pl.col("label") == tg) & (pl.col("model") == "Q1"))
        if r.height:
            lb.append(nm)
            vv.append(float(r["net_bp"][0]))
            cc.append(col)
    y = np.arange(len(vv))
    b.axvline(0, color=BASELINE, lw=1.2)
    b.barh(y, vv, height=.6, lw=0, color=cc)
    for i, val in enumerate(vv):
        # 値札は必ず 0 側(棒の内側)へ。棒の先端に置くと軸外へ出る
        b.text(-0.08, i, f"{val:+.2f}", va="center", ha="right", fontsize=8,
               color=INK)
    b.set_xlim(min(vv) * 1.15, max(0.0, max(vv)) + abs(min(vv)) * 0.18)
    b.set_yticks(y, lb, fontsize=8)
    b.invert_yaxis()
    style(b, "純 EV (bp)", "")
    b.set_title("(b) 口座と帰無対照", color=INK, fontsize=9.5, loc="left")

    # (c) markout のホライズン
    b = ax[0, 2]
    hs = [0.25, 0.5, 1, 2, 5, 10, 30, 60, 300]
    for mdl, col in (("Q1", C_A), ("Q3", C_C)):
        r = S.filter((pl.col("label") == "main_Standard") & (pl.col("model") == mdl))
        if not r.height:
            continue
        b.plot(hs, [float(r[f"mk{h:g}"][0]) for h in hs], marker="o", ms=4,
               lw=2.0, color=col, label=mdl)
    b.axhline(0, color=BASELINE, lw=1.0)
    b.set_xscale("log")
    b.set_xticks(hs, [f"{h:g}" for h in hs], fontsize=7.5)
    b.set_xticks([], minor=True)
    style(b, "約定からの秒数(対数)", "markout (bp)")
    legend(b, loc="lower right")
    b.set_title("(c) 約定後の markout — どの時点でも負", color=INK,
                fontsize=9.5, loc="left")

    # (d) 銘柄別
    b = ax[1, 0]
    Q = P.filter(pl.col("label") == "main_Standard").sort("net_bp")
    y = np.arange(Q.height)
    v = Q["net_bp"].to_numpy()
    b.axvline(0, color=BASELINE, lw=1.2)
    b.barh(y, v, height=.65, lw=0, color=[C_C if x > 0 else C_B for x in v])
    b.set_yticks(y, [f"{s} ({n})" for s, n in
                     zip(Q["sym"].to_list(), Q["fills"].to_list())], fontsize=7.5)
    style(b, "純 EV (bp)", "")
    b.set_title("(d) 銘柄別(括弧は約定数)", color=INK, fontsize=9.5, loc="left")

    # (e) 日次
    b = ax[1, 1]
    E = D.filter(pl.col("label") == "main_Standard").sort("day")
    v = E["net_usd"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(range(len(v)), v, width=.7, lw=0,
          color=[C_C if x > 0 else C_B for x in v])
    b.plot(range(len(v)), np.cumsum(v), color=INK, lw=1.8, label="累積")
    b.set_xticks(range(0, len(v), 3), [s[5:] for s in E["day"].to_list()][::3],
                 fontsize=7.5, rotation=45, ha="right")
    style(b, "", "日次 純 PnL ($)")
    legend(b, loc="lower left")
    b.set_title("(e) 日次と累積(1 建玉 100 USD)", color=INK, fontsize=9.5, loc="left")

    # (f) 日単位ブートストラップ
    b = ax[1, 2]
    v = E["net_usd"].to_numpy()
    if v.size > 1:
        rng = np.random.default_rng(11)
        bs = v[rng.integers(0, v.size, (10000, v.size))].mean(axis=1)
        b.hist(bs, bins=60, color=C_A, alpha=.75, lw=0)
        b.axvline(0, color=C_B, lw=1.8)
        b.axvline(np.quantile(bs, .025), color=INK2, lw=1.2, ls="--")
        b.axvline(np.quantile(bs, .975), color=INK2, lw=1.2, ls="--")
        b.text(np.quantile(bs, .025), b.get_ylim()[1] * .95,
               f"95% [{np.quantile(bs,.025):+.2f}, {np.quantile(bs,.975):+.2f}]",
               fontsize=8, color=INK2, ha="left")
    style(b, "日次 純 PnL の平均 ($/日)", "回数")
    b.set_title("★(f) 日単位 bootstrap — 0 は区間の右側", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle("Lighter maker 主構成 — 損益の分解(Q1 strict-trade、"
                 "探索期間 20 日・12 銘柄・1 建玉 100 USD・暫定)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(CH / "lighter_maker_pnl.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / "lighter_maker_pnl.png")


if __name__ == "__main__":
    main()
