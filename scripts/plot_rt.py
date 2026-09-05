"""microprice 検証と往復 backtest、モデル比較と門の重ね合わせを描く。

    uv run python scripts/plot_rt.py --coin xyz:MU
出力: charts/<coin>_rt_micro.png  microprice 再検証と手仕舞い 3 通り
      charts/<coin>_rt_gates.png  モデル 4 種の比較と門を順に足したときの ΔEV

【読み方】
markout(手仕舞いの費用なし)と往復(費用込み)を必ず並べてある。
markout だけを見ると広いスプレッドが有利に見えるが、往復では消える。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fillpnl import ZB, draw  # noqa: E402
from plot_heat import DIV  # noqa: E402
from plot_vol import BASELINE, INK, INK2, MUTED, SURFACE, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_MID, C_MU, C_RT, C_W = "#2a78d6", "#e34948", "#c48a12", "#1f8a5e"
MAKER_FEE, TAKER_FEE = 0.088, 0.846


def g(M, c):
    return M[c].to_numpy().astype(float).reshape(10, 10)


def fig_micro(tag: str, coin: str) -> None:
    M = pl.read_parquet(DATA / f"fillpnl_cells_{tag}.parquet")
    ev, se, n, nf = g(M, "ev_daily"), g(M, "se_daily"), g(M, "n"), g(M, "n_fill")
    win = (ev > 0) & (ev / se > ZB)
    fig, ax = plt.subplots(2, 3, figsize=(15.2, 8.6))

    e_mu = g(M, "evmu_1s")
    w_mu = (e_mu > 0) & (e_mu / g(M, "se_evmu_1s") > ZB)
    draw(ax[0, 0], e_mu, f"(a) microprice 基準の EV (h=1s) — 陽性 {int(w_mu.sum())} セル",
         DIV, True, "{:+.3f}", mark=w_mu)
    ax[0, 0].set_xlabel("予測 PnL 十分位 (Q)", color=INK2, fontsize=8.5)
    ax[0, 0].set_ylabel("予測 P(Fill) 十分位 (H)", color=INK2, fontsize=8.5)

    b = ax[0, 1]
    b.axhline(0, color=BASELINE, lw=1.0)
    b.axvline(0, color=BASELINE, lw=1.0)
    lim = 0.0
    for h, c in (("0.1", "#7a52c9"), ("1", C_MID), ("10", C_W)):
        x, y = g(M, f"ev_{h}s").ravel(), g(M, f"evmu_{h}s").ravel()
        m = np.isfinite(x) & np.isfinite(y)
        lim = max(lim, float(np.nanpercentile(np.abs(np.r_[x[m], y[m]]), 96)))
        b.scatter(x[m], y[m], s=14, color=c, alpha=0.7, lw=0,
                  label=f"h={h}s  相関 {np.corrcoef(x[m], y[m])[0,1]:+.3f}")
    b.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.0, ls="--")
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    style(b, "EV  mid 基準 (bp/候補)", "EV  microprice 基準 (bp/候補)")
    legend(b, loc="upper left")
    b.set_title("(b) 島は microprice でも消えない — 対角より上が微価格で有利",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 2]
    pol = [("markout のみ", "s_label_pnl_1s_bp", C_MID),
           ("Taker exit 1s", "s_rt_taker_1s_bp", C_RT),
           ("Hybrid 10s", "s_rt_hyb_10s_bp", "#7a52c9"),
           ("Hybrid 60s", "s_rt_hyb_60s_bp", "#e34948")]
    v = [float(g(M, c)[win].sum() / nf[win].sum()) for _, c, _ in pol]
    b.axhline(0, color=BASELINE, lw=1.0)
    b.bar(range(len(pol)), v, color=[c for _, _, c in pol], width=0.62, lw=0)
    for i, x in enumerate(v):
        b.text(i, x + (0.08 if x > 0 else -0.16), f"{x:+.3f}", ha="center",
               fontsize=8.5, color=INK)
    b.set_xticks(range(len(pol)), [p[0] for p in pol], fontsize=8)
    style(b, "", "1 約定あたり損益 (bp)")
    b.set_title("(c) 手仕舞いを入れると符号が変わる(mid 基準の陽性 29 セル)",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 0]
    R = pl.read_parquet(DATA / f"quotes_rt_sub_{tag}")
    # T_max=10s の曲線は 60s の曲線の左端に完全に重なるので、60s を 1 本だけ描く
    v2 = R["rt_pass_lat_60s"].drop_nulls().to_numpy()
    tot = R["rt_hold_60s"].drop_nulls().len()
    x = np.sort(v2)
    cum = np.arange(1, x.size + 1) / max(tot, 1)
    b.plot(x, cum, color=C_RT, lw=1.8)
    for T_, c in ((10, C_W), (60, MUTED)):
        r_ = float((x <= T_).sum()) / max(tot, 1)
        b.axvline(T_, color=c, lw=1.0, ls="--")
        b.plot([T_], [r_], marker="o", ms=6, color=c)
        b.text(T_ * 0.92, r_ + 0.03, f"T_max={T_}s  {100*r_:.1f}%", ha="right",
               fontsize=8.5, color=INK2)
    b.set_ylim(0, max(0.65, float(cum[-1]) * 1.08))
    style(b, "手仕舞いが約定するまでの秒数", "閉じた累積割合", logx=True)
    b.set_title("(d) 受動的な手仕舞いは 10 秒で 26%、60 秒でも 57% しか閉じない",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    mk = float(g(M, "s_label_pnl_1s_bp")[win].sum() / nf[win].sum())
    tk = float(g(M, "s_rt_taker_1s_bp")[win].sum() / nf[win].sum())
    spr = float((g(M, "mean_spread") * n)[win].sum() / n[win].sum())
    steps = [("markout (1s)", mk), ("− 半スプレッド", -spr / 2),
             ("− テイカー手数料", -TAKER_FEE), ("残差", tk - mk + spr / 2 + TAKER_FEE)]
    cum = 0.0
    b.axhline(0, color=BASELINE, lw=1.0)
    for i, (lab, dv) in enumerate(steps):
        b.bar(i, dv, bottom=cum if i else 0.0,
              color=C_MID if dv > 0 else C_RT, width=0.62, lw=0)
        cum = (cum + dv) if i else dv
        b.text(i, cum + (0.10 if dv > 0 else 0.06), f"{dv:+.2f}", ha="center",
               va="bottom", fontsize=8, color=INK)
    b.bar(len(steps), cum, color=INK2, width=0.62, lw=0)
    b.text(len(steps), 0.06, f"{cum:+.2f}", ha="center", va="bottom",
           fontsize=8.5, color=INK)
    b.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["Taker exit"],
                 fontsize=7.5, rotation=16, ha="right")
    style(b, "", "1 約定あたり (bp)")
    b.set_title("(e) Taker exit の内訳 — 手数料 0.846 bp だけで足りない",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 2]
    rt = g(M, "evrt_hyb_10s")
    draw(b, rt, "(f) 往復 (Hybrid 10s) の EV — 陽性セルなし", DIV, True, "{:+.3f}")
    b.set_xlabel("予測 PnL 十分位 (Q)", color=INK2, fontsize=8.5)
    b.set_ylabel("予測 P(Fill) 十分位 (H)", color=INK2, fontsize=8.5)

    fig.suptitle(f"{coin} microprice での再検証と、手仕舞いを入れた往復損益",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_rt_micro.png", dpi=170)
    plt.close(fig)


def fig_gates(tag: str, coin: str) -> None:
    Q = pl.read_parquet(DATA / f"fillpnl_models_{tag}.csv".replace(".csv", ".csv")) \
        if False else pl.read_csv(DATA / f"fillpnl_models_{tag}.csv")
    G = pl.read_csv(DATA / f"gates_quintiles_{tag}.csv")
    S = pl.read_csv(DATA / f"gates_stages_{tag}.csv")
    fig, ax = plt.subplots(2, 3, figsize=(15.2, 8.4))

    b = ax[0, 0]
    x = np.arange(Q.height)
    b.bar(x - 0.2, Q["pos_bonf"].to_numpy(), width=0.38, color=C_MID, lw=0,
          label="Bonferroni 陽性セル数")
    b.bar(x + 0.2, Q["pos"].to_numpy(), width=0.38, color=MUTED, lw=0,
          label="EV>0 のセル数")
    b.set_xticks(x, Q["model"].to_list(), fontsize=8.5)
    style(b, "", "セル数 (100 中)")
    legend(b, loc="lower right")
    b.set_title("(a) 共通分母をほどいた特徴量 — 4 推定量とも島は残る",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 1]
    b.axhline(0, color=BASELINE, lw=1.0)
    b.bar(x - 0.2, 1e3 * Q["ev_pos_mid"].to_numpy(), width=0.38, color=C_MID,
          lw=0, label="mid")
    b.bar(x + 0.2, 1e3 * Q["ev_pos_micro"].to_numpy(), width=0.38, color=C_MU,
          lw=0, label="microprice")
    b.set_xticks(x, Q["model"].to_list(), fontsize=8.5)
    style(b, "", "陽性セルの EV (bp/候補 ×10⁻³)")
    legend(b, loc="upper right")
    b.set_title("(b) 陽性セルの EV(手仕舞い費用なし)", color=INK,
                fontsize=9.5, loc="left")

    for k, (nm, bx) in enumerate((("OBI", ax[0, 2]), ("OFI 1s", ax[1, 0]),
                                  ("スプレッド", ax[1, 1]))):
        d = G.filter(pl.col("var") == nm).sort("q")
        q = d["q"].to_numpy()
        bx.axhline(0, color=BASELINE, lw=1.0)
        bx.plot(q, 1e3 * d["ev_mk"].to_numpy(), color=C_MID, lw=2.0, marker="o",
                ms=4, label="markout(費用なし)")
        bx.errorbar(q, 1e3 * d["ev_rt"].to_numpy(), yerr=1e3 * d["se_rt"].to_numpy(),
                    color=C_RT, lw=2.0, marker="s", ms=4, capsize=3,
                    label="往復 (Hybrid 10s)")
        bx.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)], fontsize=8.5)
        style(bx, f"{nm} の五分位(領域内)", "EV (bp/候補 ×10⁻³)")
        legend(bx, loc="center left")
        ttl = {"OBI": "(c) OBI — markout は平らだが往復は改善する",
               "OFI 1s": "(d) OFI — 同じく往復だけが動く",
               "スプレッド": "(e) ★ スプレッドは markout を 16 倍にするが往復は平ら"}[nm]
        bx.set_title(ttl, color=INK, fontsize=9.5, loc="left")

    b = ax[1, 2]
    st = ["gate なし", "+OBI", "+OFI 1s", "+スプレッド"]
    for obj, c, mk in (("markout", C_MID, "o"), ("往復", C_RT, "s")):
        d = S.filter((pl.col("set") == "評価") & (pl.col("obj") == obj))
        d = d.with_columns(o=pl.col("stage").replace_strict(
            {s: i for i, s in enumerate(st)}, default=None)).sort("o")
        b.errorbar(d["o"].to_numpy(), 1e3 * d["ev"].to_numpy(),
                   yerr=1e3 * d["se"].to_numpy(), color=c, lw=2.0, marker=mk,
                   ms=5, capsize=3, label=obj)
    b.axhline(0, color=BASELINE, lw=1.2)
    b.set_xticks(range(len(st)), st, fontsize=8)
    style(b, "門を順に足す(閾値は学習期間で選択)", "EV (bp/候補 ×10⁻³)")
    legend(b, loc="center right")
    b.set_title("(f) 評価期間 — 門で近づくが往復はゼロを超えない", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle(f"{coin} 共通分母をほどいたモデル比較と、領域内で門を順に足した効果",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_rt_gates.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    fig_micro(tag, a.coin)
    fig_gates(tag, a.coin)
    print("書き出し", CH / f"{tag}_rt_micro.png", CH / f"{tag}_rt_gates.png")
