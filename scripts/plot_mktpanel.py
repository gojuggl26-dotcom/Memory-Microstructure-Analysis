"""市場の状態と Entry EV の関係を描く。

    uv run python scripts/plot_mktpanel.py
出力: charts/mktpanel_ev.png

【読み方】
仮説は「余裕が大きく・逆行が小さく・食い切られにくく・OBI の予測力が高いほど
EV が高い」。実測では 1 つ目は向きだけ支持され(単調ではなく、効きは最上位五分位に集中)、
あとの 2 つは**符号が逆**である。
そして**どの五分位でも EV は負**なので、これは相対的な良し悪しの説明であって
黒字への道ではない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mktreg import MIN_LOT
from plot_vol import BASELINE, INK, INK2, MUTED, SURFACE, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_A, C_B, C_C, C_D = "#2a78d6", "#e34948", "#1f8a5e", "#c48a12"
NQ = 5


def quint(P, v, nq=NQ):
    """五分位ごとの EV(建玉数で加重)と、日でクラスタした標準誤差。"""
    x = P[v].to_numpy()
    y = P["ev"].to_numpy()
    w = P["n_lot"].to_numpy().astype(float)
    g = np.array([f"{a}|{b}" for a, b in zip(P["coin"], P["dt"])])
    e = np.quantile(x, np.linspace(0, 1, nq + 1)[1:-1])
    k = np.searchsorted(e, x, side="right")
    m, s = [], []
    for i in range(nq):
        sel = k == i
        m.append(float(np.average(y[sel], weights=w[sel])))
        # 日ごとの加重平均を並べ、その散らばりから誤差を出す
        dv = []
        for c in np.unique(g[sel]):
            q = sel & (g == c)
            if w[q].sum() > 0:
                dv.append(float(np.average(y[q], weights=w[q])))

        dv = np.array(dv)
        s.append(float(dv.std(ddof=1) / np.sqrt(dv.size)) if dv.size > 2 else np.nan)
    return np.array(m), np.array(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    P = pl.read_parquet(DATA / "mktpanel.parquet").sort("coin", "dt", "hour")
    P = P.with_columns(log_dot=pl.col("depth_over_trade").log(),
                       log_hs=pl.col("half_spread_bp").log(),
                       log_sig=pl.col("sigma_short_bp").log())
    P = P.filter(pl.col("n_lot") >= MIN_LOT, pl.col("ev").is_not_nan())
    R = pl.read_csv(DATA / "mktreg.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    def qplot(b, v, lab, c, title):
        m, s = quint(P, v)
        b.axhline(0, color=BASELINE, lw=1.2)
        b.errorbar(range(1, NQ + 1), m, yerr=1.96 * s, color=c, lw=2.2,
                   marker="o", ms=5, capsize=3)
        b.set_xticks(range(1, NQ + 1), [f"Q{i}" for i in range(1, NQ + 1)],
                     fontsize=8.5)
        style(b, lab, "1 組あたり往復損益 (bp)")
        b.set_title(title, color=INK, fontsize=9.5, loc="left")
        return m

    b = ax[0, 0]
    b.axhline(0, color=BASELINE, lw=1.2)
    for Q, c, nm in ((P, MUTED, "全窓(x1≤0 を含む)"),
                     (P.filter(pl.col("x1") > 0), C_A, "x1>0 の窓だけ")):
        m, sd_ = quint(Q, "x1")
        b.errorbar(range(1, NQ + 1), m, yerr=1.96 * sd_, color=c, lw=2.2,
                   marker="o", ms=5, capsize=3, label=nm)
    b.set_xticks(range(1, NQ + 1), [f"Q{i}" for i in range(1, NQ + 1)],
                 fontsize=8.5)
    style(b, "x1 = (スプレッド/2 − 1ティック) / σ_short の五分位",
          "1 組あたり往復損益 (bp)")
    legend(b, loc="lower right")
    b.set_title("(a) 向きは仮説どおりだが単調ではない — 効きは最上位五分位に集中",
                color=INK, fontsize=9.5, loc="left")
    b = ax[0, 1]
    b.axhline(0, color=BASELINE, lw=1.2)
    for v, c, nm in (("log_hs", C_C, "log(スプレッド/2)"),
                     ("log_sig", C_B, "log(σ_short)")):
        m, s = quint(P, v)
        b.errorbar(range(1, NQ + 1), m, yerr=1.96 * s, color=c, lw=2.2,
                   marker="o", ms=5, capsize=3, label=nm)
    b.set_xticks(range(1, NQ + 1), [f"Q{i}" for i in range(1, NQ + 1)],
                 fontsize=8.5)
    style(b, "五分位", "1 組あたり往復損益 (bp)")
    legend(b, loc="lower left")
    b.set_title("★(b) 比は制約が強すぎる — スプレッド単独は効かず σ だけが効く",
                color=INK, fontsize=9.5, loc="left")

    qplot(ax[0, 2], "log_dot", "log(Depth / TradeSize)", C_D,
          "★(c) 符号が逆 — 板が厚いほど EV は**低い**")

    b = ax[1, 0]
    b.axhline(0, color=BASELINE, lw=1.2)
    for v, c, nm in (("obi_ic_mid", MUTED, "mid に対する IC"),
                     ("obi_ic_micro", C_B, "microprice に対する IC")):
        m, s = quint(P, v)
        b.errorbar(range(1, NQ + 1), m, yerr=1.96 * s, color=c, lw=2.2,
                   marker="o", ms=5, capsize=3, label=nm)
    b.set_xticks(range(1, NQ + 1), [f"Q{i}" for i in range(1, NQ + 1)],
                 fontsize=8.5)
    style(b, "OBI の情報係数(IC)の五分位", "1 組あたり往復損益 (bp)")
    legend(b, loc="lower left")
    b.set_title("★(d) 符号が逆 — OBI が microprice を当てる市場ほど EV は**低い**",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    nm = {"x1": "x1", "log_dot": "log(Depth/Trade)", "obi_ic_mid": "IC(mid)",
          "obi_ic_micro": "IC(micro)"}
    S = R.filter(pl.col("spec").str.contains("のみ"))
    ks = ["同時", "1 期先"]
    vs = ["x1", "log_dot", "obi_ic_mid", "obi_ic_micro"]
    yy = np.arange(len(vs))
    b.axvline(0, color=BASELINE, lw=1.2)
    for i, k in enumerate(ks):
        d = S.filter(pl.col("kind") == k)
        bb = [float(d.filter(pl.col("var") == v)["beta"][0]) for v in vs]
        ss = [float(d.filter(pl.col("var") == v)["se"][0]) for v in vs]
        b.errorbar(bb, yy + (i - 0.5) * 0.22, xerr=[1.96 * s for s in ss],
                   fmt="o", ms=5, color=[C_A, C_D][i], capsize=3, label=k)
    b.set_yticks(yy, [nm[v] for v in vs], fontsize=8.5)
    style(b, "標準化した係数 (bp / 1 標準偏差) — 誤差は銘柄×日でクラスタ", "")
    legend(b, loc="lower right")
    b.set_title("(e) 単変量の係数 — 予測(1 期先)でも符号は変わらない",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 2]
    x = P["x1"].to_numpy()
    yv = P["obi_ic_micro"].to_numpy()
    ev = P["ev"].to_numpy()
    sc = b.scatter(np.clip(x, -1, 4), yv, c=np.clip(ev, -6, 2), s=6,
                   cmap="RdYlBu", lw=0, alpha=0.7)
    cb = fig.colorbar(sc, ax=b)
    cb.set_label("EV (bp)", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    style(b, "x1", "OBI の IC(microprice)")
    b.set_title(f"(f) 2 つは強く相関する (r={np.corrcoef(x, yv)[0,1]:+.3f}) — "
                f"別々には解けない", color=INK, fontsize=9.5, loc="left")

    fig.suptitle("xyz:MU + xyz:INTC — 市場の状態で Entry EV を説明できるか"
                 "(銘柄 2 × 日 99 × 時刻、3,845 窓)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / "mktpanel_ev.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / "mktpanel_ev.png")


if __name__ == "__main__":
    main()
