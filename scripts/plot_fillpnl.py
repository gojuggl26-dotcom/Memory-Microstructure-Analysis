"""250ms 約定ハザード × 条件つき損益の 10x10 を描く。

    uv run python scripts/plot_fillpnl.py --coin xyz:MU
出力: charts/<coin>_fillpnl.png     4 枚組 (P(Fill) / PnL per fill / EV / N)
      charts/<coin>_fillpnl_rb.png  頑健性 (遅延・前後半・3 変数版・無作為対照)

【読み方】
縦軸は**予測**約定確率の十分位、横軸は**予測**損益の十分位。どちらも学習期間
(前 59 日)だけで作った境界で、図の数値は評価期間(後 39 日)の実測値。
EV は 1 候補あたりで、メイカー手数料 0.088 bp 控除後。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_heat import DIV, SEQ  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
NB = 10
ZB = 3.481          # 両側 5% を 100 セルで Bonferroni 補正
LAGS = [("", 0), ("_lag65", 65), ("_lag130", 130)]


def cells(tag, sfx=""):
    return pl.read_parquet(DATA / f"fillpnl_cells_{tag}{sfx}.parquet")


def g(M, c):
    return M[c].to_numpy().astype(float).reshape(NB, NB)


def draw(ax, A, title, cmap, diverging, fmt="{:.2f}", robust=98.0, mark=None):
    v = A[np.isfinite(A)]
    if diverging:
        m = float(np.percentile(np.abs(v), robust)) if v.size else 1.0
        vmin, vmax = -m, m
    else:
        vmin = float(np.percentile(v, 100 - robust)) if v.size else 0.0
        vmax = float(np.percentile(v, robust)) if v.size else 1.0
    ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    for i in range(NB):
        for j in range(NB):
            if not np.isfinite(A[i, j]):
                continue
            rel = abs(A[i, j]) / max(abs(vmin), abs(vmax), 1e-12)
            ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center",
                    fontsize=5.4, color="#ffffff" if rel > 0.62 else INK)
            if mark is not None and mark[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           ec="#1a7f37", lw=1.6, zorder=5))
    ax.set_xticks(range(NB), [f"Q{i+1}" for i in range(NB)], fontsize=7)
    ax.set_yticks(range(NB), [f"H{i+1}" for i in range(NB)], fontsize=7)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    M = cells(tag)
    ev, se, n = g(M, "ev_daily"), g(M, "se_daily"), g(M, "n")
    t = ev / se
    win = (ev > 0) & (t > ZB)
    tot = float(np.nansum(g(M, "sum_pnl")))
    ntot = float(n.sum())

    fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.4))
    draw(ax[0, 0], 100 * g(M, "p_fill"), "(a) P(Fill 250ms) 実測 (%)",
         SEQ, False, "{:.2f}")
    draw(ax[0, 1], g(M, "pnl_per_fill"),
         "(b) PnL per fill (bp・手数料控除後)", DIV, True, "{:+.2f}")
    draw(ax[1, 0], ev,
         f"(c) EV per candidate (bp) — 緑枠は Bonferroni 陽性 {int(win.sum())} セル",
         DIV, True, "{:+.3f}", mark=win)
    draw(ax[1, 1], n / 1e3, "(d) N (千候補・評価 39 日)", SEQ, False, "{:.0f}")
    for r in ax:
        for b in r:
            b.set_xlabel("予測 PnL 十分位 (Q)", color=INK2, fontsize=8.5)
            b.set_ylabel("予測 P(Fill) 十分位 (H)", color=INK2, fontsize=8.5)
    fig.suptitle(
        f"{a.coin} 250ms 約定ハザード × 条件つき損益 — 学習 59 日 / 評価 39 日"
        f"(標本外)。全体の EV {tot/ntot:+.4f} bp/候補",
        color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_fillpnl.png", dpi=170)
    plt.close(fig)

    # ---- 頑健性 ----
    fig, ax = plt.subplots(2, 3, figsize=(14.4, 8.2))
    for k, (sfx, ms) in enumerate(LAGS):
        L = cells(tag, sfx)
        e = g(L, "ev_daily")
        w = (e > 0) & (e / g(L, "se_daily") > ZB)
        draw(ax[0, k], e, f"({chr(97+k)}) 遅延 {ms} ms — 陽性 {int(w.sum())} セル",
             DIV, True, "{:+.3f}", mark=w)
        ax[0, k].set_xlabel("予測 PnL 十分位 (Q)", color=INK2, fontsize=8.5)
        ax[0, k].set_ylabel("予測 P(Fill) 十分位 (H)", color=INK2, fontsize=8.5)

    b = ax[1, 0]
    e1, e2 = g(M, "ev_h1").ravel(), g(M, "ev_h2").ravel()
    m = np.isfinite(e1) & np.isfinite(e2)
    b.axhline(0, color=BASELINE, lw=1.0)
    b.axvline(0, color=BASELINE, lw=1.0)
    lim = float(np.nanpercentile(np.abs(np.r_[e1[m], e2[m]]), 97))
    b.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.0, ls="--")
    b.scatter(e1[m], e2[m], s=16, color="#2a78d6", alpha=0.75, lw=0)
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    style(b, "評価期間 前半 19 日 の EV (bp)", "後半 20 日 の EV (bp)")
    b.set_title(f"(d) 前後半の一致 — 相関 {np.corrcoef(e1[m], e2[m])[0,1]:+.3f}、"
                f"両方正 {int(((e1>0)&(e2>0))[m].sum())}/{int(m.sum())}",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    S = cells(tag, "_simple")
    es = g(S, "ev_daily")
    draw(b, es, "(e) 3 変数版(取消/追加比・待ち行列・スプレッド)— 陽性 0 セル",
         DIV, True, "{:+.3f}")
    b.set_xlabel("予測 PnL 十分位 (Q)", color=INK2, fontsize=8.5)
    b.set_ylabel("予測 P(Fill) 十分位 (H)", color=INK2, fontsize=8.5)

    b = ax[1, 2]
    R = cells(tag, "")
    sp = g(M, "mean_spread").ravel()
    b.axhline(0, color=BASELINE, lw=1.0)
    b.scatter(sp[~win.ravel()], ev.ravel()[~win.ravel()], s=16, color=MUTED,
              alpha=0.7, lw=0, label="それ以外")
    b.scatter(sp[win.ravel()], ev.ravel()[win.ravel()], s=22, color="#1a7f37",
              alpha=0.9, lw=0, label="Bonferroni 陽性")
    style(b, "セルの平均スプレッド (bp)", "EV per candidate (bp)")
    legend(b, loc="lower right")
    b.set_title("(f) 島はスプレッドの広い側に寄る "
                f"(陽性 {float((sp[win.ravel()]*n.ravel()[win.ravel()]).sum()/n[win].sum()):.2f}"
                f" 対 それ以外 {float((sp[~win.ravel()]*n.ravel()[~win.ravel()]).sum()/n[~win].sum()):.2f} bp)",
                color=INK, fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} EV>0 の島は本物か — 遅延・期間・モデルの単純化・"
                 f"スプレッドとの関係", color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_fillpnl_rb.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_fillpnl.png", CH / f"{tag}_fillpnl_rb.png")

    # ---- 数値の要約 (レポート用) ----
    out = []
    for sfx, ms in LAGS:
        L = cells(tag, sfx)
        e, s2, nn = g(L, "ev_daily"), g(L, "se_daily"), g(L, "n")
        nf, spl = g(L, "n_fill"), g(L, "sum_pnl")
        w = (e > 0) & (e / s2 > ZB)
        out.append({"lag_ms": ms, "pos": int((e > 0).sum()),
                    "pos_bonf": int(w.sum()), "neg_bonf": int(((e < 0) &
                                                               (e / s2 < -ZB)).sum()),
                    "n_pos": float(nn[w].sum()), "share_n": float(nn[w].sum() / nn.sum()),
                    "fills_pos": float(nf[w].sum()), "pnl_pos": float(spl[w].sum()),
                    "ev_pos": float(spl[w].sum() / max(nn[w].sum(), 1)),
                    "pnl_per_fill": float(spl[w].sum() / max(nf[w].sum(), 1)),
                    "spread_pos": float((g(L, "mean_spread")[w] * nn[w]).sum()
                                        / max(nn[w].sum(), 1)),
                    "ev_all": float(spl.sum() / nn.sum())})
    pl.DataFrame(out).write_csv(DATA / f"fillpnl_summary_{tag}.csv")
    print(pl.DataFrame(out))


if __name__ == "__main__":
    main()
