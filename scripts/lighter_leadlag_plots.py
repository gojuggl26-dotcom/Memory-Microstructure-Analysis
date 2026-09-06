r"""リードラグの図(CCF・細部・偏相関・ホライズン曲線・イベントスタディ・時計診断)。

★見出しの断定は実測値から組み立てる。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
LL = Path("E:/Memory-lighter/ll")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
PAIRS = ["MU", "DRAM", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD",
         "AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "XAG", "XAU"]
CB, CL, CP = "#dc2626", "#2563eb", "#9ca3af"


def med_ccf(C, sym, clock, dt, session="all"):
    g = C.filter((pl.col("symbol") == sym) & (pl.col("clock") == clock)
                 & (pl.col("dt") == dt) & (pl.col("session") == session))
    ks = sorted(g["k"].unique().to_list())
    med, q1, q3 = [], [], []
    for k in ks:
        v = g.filter(pl.col("k") == k)["rho"].to_numpy()
        v = v[np.isfinite(v)]
        if len(v) == 0:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        else:
            med.append(np.median(v))
            q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75))
    return np.array(ks), np.array(med), np.array(q1), np.array(q3)


def leader_stats(C, sym, clock, dt=1.0, K=10):
    ks, med, _, _ = med_ccf(C, sym, clock, dt)
    pos = (ks >= 1) & (ks <= K)
    neg = (ks <= -1) & (ks >= -K)
    sB = float(np.nansum(med[pos]))       # k>0: Binance 先行
    sL = float(np.nansum(med[neg]))       # k<0: Lighter 先行
    share = sB / (sB + sL) if (sB + sL) > 0 else np.nan
    return sB, sL, share


def fig_ccf(C):
    fig, axes = plt.subplots(3, 4, figsize=(18.4, 10.6))
    stats = []
    for ax, sym in zip(axes.ravel(), PAIRS):
        ks, med, q1, q3 = med_ccf(C, sym, "server", 1.0)
        if not len(ks):
            ax.axis("off"); continue
        ax.fill_between(ks, q1, q3, color=CB, alpha=0.15)
        ax.plot(ks, med, "-", color=CB, lw=1.8, label="実測(server)")
        kp, mp, _, _ = med_ccf(C, sym, "placebo_dayshift", 1.0)
        if len(kp):
            ax.plot(kp, mp, "x--", color=CP, lw=1.2, ms=4, label="帰無(日ずらし)")
        ax.axvline(0, color="#111827", lw=0.9)
        ax.axhline(0, color="#111827", lw=0.9)
        sB, sL, share = leader_stats(C, sym, "server")
        stats.append((sym, share))
        ax.set_title(f"{sym}\nBinance 先行側 Σρ={sB:+.3f} / Lighter 先行側 Σρ={sL:+.3f}",
                     fontsize=8.6, loc="left")
        ax.tick_params(labelsize=7)
        if sym == PAIRS[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Lighter × Binance — リターンのクロス相関 CCF(Δ=1s、±30s、"
                 "サーバ時計、日ごとに推定した中央値・帯は四分位)\n"
                 "k>0 = Binance のリターンが Lighter の k 秒後のリターンと相関"
                 "(=Binance 先行)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = CHARTS / "lighter_ll_ccf.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def fig_fine(C):
    fig, axes = plt.subplots(3, 4, figsize=(18.4, 10.6))
    for ax, sym in zip(axes.ravel(), PAIRS):
        got = False
        for clock, col, lab in (("server", CB, "server 時計"),
                                ("local", "#7c3aed", "受信時計(同一マシン)")):
            ks, med, q1, q3 = med_ccf(C, sym, clock, 0.25)
            if not len(ks):
                continue
            got = True
            ax.fill_between(ks * 0.25, q1, q3, color=col, alpha=0.12)
            ax.plot(ks * 0.25, med, "o-", color=col, lw=1.6, ms=3, label=lab)
        if not got:
            ax.axis("off"); continue
        ax.axvline(0, color="#111827", lw=0.9)
        ax.axhline(0, color="#111827", lw=0.9)
        ax.set_title(sym, fontsize=9.5, loc="left")
        ax.tick_params(labelsize=7)
        if sym == PAIRS[0]:
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("ラグ [秒]", fontsize=7.5)
    fig.suptitle("細部 CCF(Δ=0.25s、±3s)— 時計 2 通りの比較。"
                 "山の位置のずれ = 時計の差、形 = 伝播の速さ", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = CHARTS / "lighter_ll_fine.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def fig_partial_hcurve(C, P, HC):
    fig = plt.figure(figsize=(17.8, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

    # ① 偏相関(自分の過去を統制した後)
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for sym in PAIRS:
        for dr, col in (("B→L", CB), ("L→B", CL)):
            g = P.filter((pl.col("symbol") == sym) & (pl.col("clock") == "server")
                         & (pl.col("dir") == dr))
            v = g["partial"].to_numpy()
            v = v[np.isfinite(v)]
            if len(v):
                rows.append((sym, dr, np.median(v), int((v > 0).sum()), len(v)))
    ypos = np.arange(len(PAIRS))
    for i, sym in enumerate(PAIRS):
        for dr, off, col in (("B→L", -0.18, CB), ("L→B", +0.18, CL)):
            r = [x for x in rows if x[0] == sym and x[1] == dr]
            if r:
                ax.barh(i + off, r[0][2], 0.34, color=col, alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(PAIRS, fontsize=8)
    ax.axvline(0, color="#111827", lw=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("偏 Spearman(相手の直前 1s リターン、自分の直前 1s を統制)")
    ax.set_title("★① 自分の過去を差し引いても相手は情報を持つか\n"
                 "赤 = Binance→Lighter / 青 = Lighter→Binance(日ごとの中央値)",
                 fontsize=10.5, loc="left")

    # ② 偏相関の日次分布(全銘柄プールではなく銘柄別の点)
    ax = fig.add_subplot(gs[0, 1])
    for dr, col in (("B→L", CB), ("L→B", CL)):
        med = [x[2] for x in rows if x[1] == dr]
        ax.hist(med, bins=15, color=col, alpha=0.55, label=dr)
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_xlabel("偏 Spearman(銘柄ごとの中央値)")
    ax.set_ylabel("銘柄数")
    nB = sum(1 for x in rows if x[1] == "B→L" and x[2] > 0)
    nL = sum(1 for x in rows if x[1] == "L→B" and x[2] > 0)
    nn = len([x for x in rows if x[1] == "B→L"])
    ax.set_title(f"② 銘柄横断: B→L 正 {nB}/{nn}・L→B 正 {nL}/{nn}",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # ③ セッション別 CCF(米国立会 vs それ以外、代表 4 銘柄)
    ax = fig.add_subplot(gs[0, 2])
    for sym, ls in zip(("MU", "NVDA", "XAU", "SNDK"), ("-", "--", ":", "-.")):
        for sess, col in (("US", "#16a34a"), ("OFF", "#9ca3af")):
            ks, med, _, _ = med_ccf(C, sym, "server", 1.0, sess)
            if len(ks):
                ax.plot(ks, med, ls, color=col, lw=1.4,
                        label=f"{sym} {sess}" if sym in ("MU",) else None)
    ax.axvline(0, color="#111827", lw=0.9)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xlim(-15, 15)
    ax.set_xlabel("ラグ [秒]")
    ax.set_title("③ 米国立会(緑)とそれ以外(灰)\n線種 = MU/NVDA/XAU/SNDK",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # ④ ホライズン曲線
    ax = fig.add_subplot(gs[1, 0])
    for sym in PAIRS:
        g = HC.filter((pl.col("symbol") == sym) & (pl.col("clock") == "server"))
        hs = sorted(g["h"].unique().to_list())
        med = [np.nanmedian(g.filter(pl.col("h") == h)["rho"].to_numpy())
               for h in hs]
        ax.plot(range(len(hs)), med, "o-", lw=1.3, ms=3, label=sym)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"{h:g}" for h in hs], fontsize=7.5)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xlabel("Lighter 側のホライズン h [秒]")
    ax.set_ylabel("ρ(Binance 直近 1s リターン, Lighter の h 先リターン)")
    ax.set_title("④ Binance の直近 1 秒はどこまで先を当てるか", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=6.2, frameon=False, ncol=2)

    # ⑤ 時計診断
    CK = pl.read_parquet(LL / "clock.parquet")
    ax = fig.add_subplot(gs[1, 1])
    for i, sym in enumerate(PAIRS):
        g = CK.filter(pl.col("symbol") == sym)
        ax.plot([i - 0.15] * g.height, g["lighter_lag_ms"].to_numpy(), ".",
                color=CL, ms=4, alpha=0.7)
        ax.plot([i + 0.15] * g.height, g["binance_lag_ms"].to_numpy(), ".",
                color=CB, ms=4, alpha=0.7)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIRS, rotation=60, fontsize=7, ha="right")
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_ylabel("受信時刻 − サーバ時刻 [ms](日次中央値)")
    ax.set_title("⑤ 時計の診断 — 青=Lighter / 赤=Binance\n"
                 "負 = サーバの時計が受信より進んでいる", fontsize=10.5, loc="left")

    # ⑥ リーダーシェア
    ax = fig.add_subplot(gs[1, 2])
    shares, labels = [], []
    for sym in PAIRS:
        _, _, sh = leader_stats(C, sym, "server")
        if np.isfinite(sh):
            shares.append(sh); labels.append(sym)
    ax.barh(range(len(labels)), shares, 0.6, color=CB, alpha=0.85)
    ax.axvline(0.5, color="#111827", lw=1.2, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Binance 先行シェア = Σρ(k=1..10) / (両側の和)")
    nb = sum(1 for s in shares if s > 0.5)
    ax.set_title(f"⑥ リーダーシェア — 0.5 超(Binance 先行)は {nb}/{len(shares)} 銘柄",
                 fontsize=10.5, loc="left")

    fig.suptitle("Lighter × Binance — 方向の判定(偏相関・層別・ホライズン・時計)",
                 fontsize=13.5, y=0.99)
    out = CHARTS / "lighter_ll_partial.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def fig_events():
    fig, axes = plt.subplots(3, 4, figsize=(18.4, 10.8))
    t = np.arange(-20, 61) * 0.25
    for ax, sym in zip(axes.ravel(), PAIRS):
        p = LL / f"events_{sym}.npz"
        if not p.exists():
            ax.axis("off"); continue
        z = np.load(p)
        b2l, bs = z["b2l"], z["b_self"]
        l2b, ls_ = z["l2b"], z["l_self"]
        if len(b2l):
            ax.plot(t, bs.mean(axis=0), "--", color=CB, lw=1.3,
                    label=f"Binance 自身 (n={len(b2l)})")
            ax.plot(t, b2l.mean(axis=0), "-", color=CB, lw=1.9,
                    label="Lighter の応答")
        if len(l2b):
            ax.plot(t, ls_.mean(axis=0), "--", color=CL, lw=1.3,
                    label=f"Lighter 自身 (n={len(l2b)})")
            ax.plot(t, l2b.mean(axis=0), "-", color=CL, lw=1.9,
                    label="Binance の応答")
        ax.axvline(0, color="#111827", lw=0.9)
        ax.axhline(0, color="#111827", lw=0.9)
        ax.set_title(sym, fontsize=9.5, loc="left")
        ax.tick_params(labelsize=7)
        if sym == PAIRS[0]:
            ax.legend(fontsize=6.4, frameon=False)
    fig.suptitle("イベントスタディ — 片側の 1 秒ジャンプ(|r| ≥ max(3bp, p99.9)、"
                 "30 秒クールダウン)後の両取引所の平均経路[bp]\n"
                 "イベントの符号に揃えて平均。x=イベントからの秒(サーバ時計)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = CHARTS / "lighter_ll_events.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    C = pl.read_parquet(LL / "ccf.parquet")
    P = pl.read_parquet(LL / "partial.parquet")
    HC = pl.read_parquet(LL / "hcurve.parquet")
    todo = a.only.split(",") if a.only else ["ccf", "fine", "partial", "events"]
    if "ccf" in todo:
        fig_ccf(C)
    if "fine" in todo:
        fig_fine(C)
    if "partial" in todo:
        fig_partial_hcurve(C, P, HC)
    if "events" in todo:
        fig_events()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
