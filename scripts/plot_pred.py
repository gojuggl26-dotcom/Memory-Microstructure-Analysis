"""OBI / OFI / 攻撃的注文の強度の五分位 × 予測ホライズンを描く。

    uv run python scripts/plot_pred.py --coin xyz:MU
出力: charts/<coin>_pred_quintile.png  五分位ごとの前向きリターン
      charts/<coin>_pred_micro.png     それは新しい情報か・費用を引くと残るか
      data/pred_summary_<coin>.csv     数値表

【読み方】
mid のリターンだけを見てはいけない。microprice = mid + (スプレッド/2)x OBI は
T 時点で既に判っているので、「OBI が高いと mid が上がる」の大半は mid が
その既知の値へ寄るだけの機械的な動きでありうる。microprice で測った
リターンが動いて初めて「新しい情報」と言える。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_vol import (BASELINE, C_MID, C_MIC, C_NW, C_WARN, GRID, INK, INK2,  # noqa: E402
                      MUTED, SURFACE, legend, style)

ROOT = Path(__file__).resolve().parents[1]
HS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ["obi", "ofi_10s", "ai_net_10s", "ai_rate_10s"]
LAB = {"obi": "OBI  板の不均衡 (瞬間値)",
       "ofi_10s": "OFI  注文流の不均衡 (10 秒)",
       "ai_net_10s": "攻撃的注文の符号つき数量 (10 秒)",
       "ai_rate_10s": "攻撃的注文の件数 (10 秒)"}
# 五分位の色: 発散 (Q1 売り側 → Q5 買い側)。中央は無彩色
QC = ["#b5322f", "#e08b7f", "#9a9890", "#7fa8dd", "#1f5fa8"]


def agg(C: pl.DataFrame) -> pl.DataFrame:
    return (C.group_by("feat", "h", "q")
            .agg(n=pl.col("n").sum(), sr=pl.col("sum_r").sum(),
                 src=pl.col("sum_rmic").sum(), sa=pl.col("sum_abs").sum(),
                 smm=pl.col("sum_mm").sum(), ss=pl.col("sum_spr").sum(),
                 np_=pl.col("n_p").sum(), srp=pl.col("sum_r_p").sum())
            .with_columns(mid=pl.col("sr") / pl.col("n"),
                          mic=pl.col("src") / pl.col("n"),
                          mabs=pl.col("sa") / pl.col("n"),
                          mm=pl.col("smm") / pl.col("n"),
                          spr=pl.col("ss") / pl.col("n"),
                          pla=pl.col("srp") / pl.col("np_"))
            .sort("feat", "h", "q"))


def daily(C: pl.DataFrame, f: str, h: float, col: str) -> tuple[float, float]:
    """Q5-Q1 を日ごとに作り、日をまたいだ平均と標準誤差を返す。

    1 日の中の窓は重なっていないが、ボラティリティの塊があるので独立ではない。
    日単位にまとめてから誤差を取るほうが保守的で正しい。
    """
    s = C.filter((pl.col("feat") == f) & (pl.col("h") == h)
                 & pl.col("q").is_in([0, 4])).sort("dt", "q")
    d = (s.group_by("dt")
         .agg(v=(pl.col(col) / pl.col("n" if col != "sum_r_p" else "n_p")).last()
              - (pl.col(col) / pl.col("n" if col != "sum_r_p" else "n_p")).first()))
    v = d["v"].to_numpy()
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def hx(ax):
    ax.set_xscale("log")
    ax.set_xticks(HS)
    ax.set_xticklabels(["100ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"])
    ax.minorticks_off()


def fig_quintile(G, tag, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    fig.suptitle(f"{tag}  五分位ごとの前向きリターン — 97 日・1 秒格子・"
                 f"境目は前日の分布から", color=INK, fontsize=13, y=0.985)
    for ax, f in zip(axes.ravel(), FEATS):
        r = G.filter(pl.col("feat") == f)
        col = "mabs" if f == "ai_rate_10s" else "mid"
        for q in range(5):
            v = r.filter(pl.col("q") == q).sort("h")
            ax.plot(HS, v[col].to_numpy(), color=QC[q], lw=2.0, marker="o",
                    ms=4.5, label=f"Q{q+1}")
        if f != "ai_rate_10s":
            ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
        hx(ax)
        yl = ("その後 h の |リターン| の平均 (bp)" if f == "ai_rate_10s"
              else "その後 h のリターンの平均 (bp)")
        style(ax, "予測ホライズン h", yl, logy=(f == "ai_rate_10s"))
        tail = ("  ← 向きは当てない。動きの大きさを当てる"
                if f == "ai_rate_10s" else "")
        ax.set_title(LAB[f] + tail, color=INK, fontsize=10, loc="left")
        legend(ax, loc="upper left", ncol=2)
    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def fig_micro(C, G, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{tag}  その予測は新しい情報か、費用を引くと残るか",
                 color=INK, fontsize=13, y=0.985)
    sg = [f for f in FEATS if f != "ai_rate_10s"]
    ttl = {"obi": "(a) OBI — 既知の分を超えない",
           "ofi_10s": "(b) OFI — 既知の分をはるかに超える",
           "ai_net_10s": "(c) 攻撃的注文 — 機械的な向きと逆に効く"}
    for ax, f in zip(axes[0], sg):
        r = G.filter(pl.col("feat") == f)
        md, mc, sd, sc = [], [], [], []
        for h in HS:
            a, b = daily(C, f, h, "sum_r")
            c, d = daily(C, f, h, "sum_rmic")
            md.append(a); sd.append(b); mc.append(c); sc.append(d)
        md, mc = np.array(md), np.array(mc)
        sd, sc = np.array(sd), np.array(sc)
        r1 = r.filter(pl.col("h") == 1.0).sort("q")["mm"].to_numpy()
        known = float(r1[4] - r1[0])
        ax.axhline(known, color=MUTED, lw=1.4, ls=":", zorder=1)
        ax.text(HS[-1], known, f"T 時点で既知 {known:+.2f} bp ", color=MUTED,
                fontsize=8, va="bottom", ha="right")
        ax.fill_between(HS, md - sd, md + sd, color=C_MID, alpha=0.15, lw=0)
        ax.plot(HS, md, color=C_MID, lw=2.2, marker="o", ms=4.5,
                label="mid で測った Q5−Q1")
        ax.fill_between(HS, mc - sc, mc + sc, color=C_MIC, alpha=0.15, lw=0)
        ax.plot(HS, mc, color=C_MIC, lw=2.2, marker="s", ms=4.5,
                label="microprice で測った Q5−Q1")
        ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
        hx(ax)
        style(ax, "予測ホライズン h", "Q5 − Q1 (bp)")
        ax.set_title(ttl[f], color=INK, fontsize=10, loc="left")
        legend(ax, loc="upper left")

    ax = axes[1, 0]
    spread = float(G.filter(pl.col("feat") == "obi")["spr"].mean())
    # 買いは ask (mid + spread/2)、売りは bid (mid - spread/2) なので、
    # 入って出るとスプレッド 1 本ぶん払う。手数料はこれとは別に掛かる。
    for f, c in zip(sg, (C_MID, C_MIC, C_NW)):
        g = [daily(C, f, h, "sum_r")[0] / 2.0 - spread for h in HS]
        ax.plot(HS, g, color=c, lw=2.0, marker="o", ms=4.5, label=LAB[f])
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    ax.text(HS[0], 0.02, " ここより上でないと取れない", color=INK, fontsize=8,
            va="bottom")
    ax.set_ylim(top=0.25)
    hx(ax)
    style(ax, "予測ホライズン h", "1 往復あたりの純益 (bp)")
    ax.set_title(f"(d) 板を叩いて取る場合 — 往復のスプレッド {spread:.2f} bp を引く"
                 f" (手数料は別)",
                 color=INK, fontsize=10, loc="left")
    legend(ax, loc="lower right")

    ax = axes[1, 1]
    w = 0.2
    for i, f in enumerate(FEATS):
        obs = np.array([daily(C, f, h, "sum_r")[0] for h in HS])
        pla = np.array([daily(C, f, h, "sum_r_p")[0] for h in HS])
        ax.plot(HS, np.abs(obs), color=QC[[0, 4, 1, 2][i]], lw=2.0,
                marker="o", ms=4, label=f"観測 {f}")
        ax.plot(HS, np.abs(pla), color=QC[[0, 4, 1, 2][i]], lw=1.2, ls="--",
                alpha=0.7)
    ax.text(0.03, 0.22, "実線 = 観測 / 破線 = 帰無対照 (説明変数を日内で "
            "300 秒ずらす)", color=INK2, fontsize=8, transform=ax.transAxes,
            va="bottom")
    hx(ax)
    style(ax, "予測ホライズン h", "|Q5 − Q1| (bp)", logy=True)
    ax.set_title("(e) 帰無対照は全ホライズンで消える", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="lower right", ncol=2)

    ax = axes[1, 2]
    yy = np.arange(len(FEATS))
    for j, (h, mk) in enumerate(((1.0, "o"), (10.0, "s"), (60.0, "^"))):
        tv = []
        for f in FEATS:
            m, se = daily(C, f, h, "sum_rmic")
            tv.append(m / se if se > 0 else np.nan)
        ax.scatter(tv, yy + (j - 1) * 0.22, s=48, marker=mk,
                   color=[C_MID, C_MIC, C_NW][j], label=f"h = {h:g} 秒")
    ax.axvline(0, color=INK, lw=1.2, zorder=3)
    for v in (-3.29, 3.29):
        ax.axvline(v, color=C_WARN, lw=1.0, ls="--", zorder=1)
    ax.text(3.4, len(FEATS) - 0.55, "Bonferroni (64 検定)", color=C_WARN,
            fontsize=8, va="top")
    ax.set_yticks(yy, [LAB[f] for f in FEATS], fontsize=8.5)
    style(ax, "microprice で測った Q5−Q1 の t 値 (符号つき・日次 97 日)", "")
    ax.set_title("(f) 符号まで見ると OBI だけ向きが逆", color=INK, fontsize=10,
                 loc="left")
    legend(ax, loc="lower right")

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    C = pl.read_parquet(ROOT / "data" / f"pred_cells_{tag}.parquet")
    G = agg(C)
    print(f"{a.coin}: {C['dt'].n_unique()} 日 / 総観測 {int(C['n'].sum()):,}")
    out = []
    for f in FEATS:
        for h in HS:
            m1, s1 = daily(C, f, h, "sum_r")
            m2, s2 = daily(C, f, h, "sum_rmic")
            m3, s3 = daily(C, f, h, "sum_r_p")
            r = G.filter((pl.col("feat") == f) & (pl.col("h") == h)).sort("q")
            out.append({"feat": f, "h": h, "n": int(r["n"].sum()),
                        "mid_q5_q1": m1, "mid_se": s1, "mid_t": m1 / s1,
                        "mic_q5_q1": m2, "mic_se": s2, "mic_t": m2 / s2,
                        "placebo_q5_q1": m3, "placebo_se": s3,
                        "known_micro_minus_mid": float(r["mm"][4] - r["mm"][0]),
                        "abs_q1": float(r["mabs"][0]),
                        "abs_q5": float(r["mabs"][4]),
                        "spread_bp": float(r["spr"].mean())})
    pl.DataFrame(out).write_csv(ROOT / "data" / f"pred_summary_{tag}.csv")
    ch = ROOT / "charts"
    fig_quintile(G, a.coin, ch / f"{tag}_pred_quintile.png")
    fig_micro(C, G, a.coin, ch / f"{tag}_pred_micro.png")


if __name__ == "__main__":
    main()
