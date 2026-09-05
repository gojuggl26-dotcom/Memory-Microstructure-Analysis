"""在庫を持つメイカーの図。

    uv run python scripts/plot_inv.py --coin xyz:MU
出力: charts/<coin>_inv.png       相殺の生存曲線・組の損益・timeout 全探索・設定比較
      charts/<coin>_inv_gate.png  在庫方向に揃えた OBI/OFI・入口の門・手仕舞い価格

【読み方】
単価(1 組あたり)と総額(1 日あたり)を必ず並べてある。timeout を短くすると
「相殺できた組」だけの単価は上がるが、強制決済が 88% になり総額は最悪になる。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fillpnl import draw  # noqa: E402
from plot_heat import DIV, SEQ  # noqa: E402
from plot_vol import BASELINE, INK, INK2, MUTED, SURFACE, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_A, C_B, C_C, C_D = "#2a78d6", "#e34948", "#1f8a5e", "#c48a12"


def fig_main(tag: str, coin: str) -> None:
    S = pl.read_csv(DATA / f"inv_summary_{tag}.csv")
    files = sorted(Path("E:/Memory-quotes").joinpath(tag).glob("dt=*.parquet"))
    te = [f.stem.split("=")[1] for f in files[59:]]     # 評価期間だけで描く
    L = pl.read_parquet(DATA / f"inv_lots_{tag}_q1.parquet").filter(
        pl.col("dt").is_in(te))
    G = pl.read_parquet(DATA / f"inv_lots_{tag}_q1_gated.parquet").filter(
        pl.col("dt").is_in(te))
    fig, ax = plt.subplots(2, 3, figsize=(15.2, 8.6))

    b = ax[0, 0]
    for D, c, nm in ((L, C_A, "門なし"), (G, C_C, "門あり")):
        v = np.sort(D["t_off"].to_numpy())
        b.plot(v, 1.0 - np.arange(v.size) / v.size, color=c, lw=1.8, label=nm)
    for h, lab in ((1.0, "1s"), (10.0, "10s"), (60.0, "60s")):
        b.axvline(h, color=BASELINE, lw=0.9, ls="--")
    style(b, "相殺までの秒数", "まだ相殺できていない割合", logx=True)
    legend(b, loc="lower left")
    b.set_title("(a) メイカー同士の相殺は速い(評価期間 39 日)",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 1]
    bins = np.linspace(-15, 15, 121)
    for D, c, nm in ((L, C_A, "門なし"), (G, C_C, "門あり")):
        v = D["pnl"].to_numpy()
        b.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=c,
               label=f"{nm}  平均 {v.mean():+.2f} / 中央 {np.median(v):+.2f}")
    b.axvline(0, color=BASELINE, lw=1.0)
    style(b, "1 組の往復損益 (bp)", "密度")
    legend(b, loc="upper left")
    b.set_title("(b) 分布は左に裾を引く — 中央値は正でも平均は負",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 2]
    T = S.filter(pl.col("cfg").str.contains("tmax")).with_columns(
        tm=pl.col("cfg").map_elements(
            lambda s: float(re.search(r"tmax([\d.]+)", s).group(1)),
            return_dtype=pl.Float64)).sort("tm")
    x = T["tm"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.0)
    b.plot(x, T["ev_pair"].to_numpy(), color=MUTED, lw=1.8, marker="o", ms=4,
           ls="--", label="相殺できた組だけの単価(誤り)")
    b.plot(x, T["ev_lot"].to_numpy(), color=C_B, lw=2.2, marker="s", ms=4,
           label="建玉 1 本あたり(正しい)")
    base = float(S.filter(pl.col("cfg") == "q1")["ev_lot"][0])
    b.axhline(base, color=C_A, lw=1.4, ls=":", label=f"時間切れなし {base:+.2f}")
    style(b, "強制決済までの上限 T_max (秒)", "1 単位あたり損益 (bp)", logx=True)
    legend(b, loc="lower right")
    b.set_title("★(c) 短い timeout は「組の単価」だけ上げる — 総額は最悪",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 0]
    b.plot(x, 100 * T["timeout_rate"].to_numpy(), color=C_D, lw=2.0, marker="o",
           ms=4, label="強制決済の割合")
    style(b, "強制決済までの上限 T_max (秒)", "強制決済された約定の割合 (%)",
          logx=True)
    b2 = b.twinx()
    b2.plot(x, T["timeout_cost"].to_numpy(), color=C_B, lw=2.0, marker="s",
            ms=4, label="1 回あたりの費用")
    b2.set_ylabel("強制決済 1 回の費用 (bp)", color=INK2, fontsize=9)
    b2.tick_params(colors=INK2, labelsize=8.5)
    for sp in ("top",):
        b2.spines[sp].set_visible(False)
    legend(b, loc="center right")
    b.set_title("(d) 強制決済は 1 回 1.40〜1.47 bp — 発生率がそのまま損になる",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    for f, c, nm in ((f"inv_days_{tag}_q1.csv", C_A, "門なし"),
                     (f"inv_days_{tag}_q1_gated.csv", C_C, "門あり")):
        D = pl.read_csv(DATA / f).sort("dt")[59:]
        v = D["total_bp"].to_numpy() / np.maximum(
            D["n_pair"].to_numpy(), 1)
        b.plot(np.arange(v.size), v, color=c, lw=1.5, label=nm)
    b.axhline(0, color=BASELINE, lw=1.2)
    style(b, "評価期間の日 (2026-07-02 から)", "その日の 1 組あたり損益 (bp)")
    legend(b, loc="lower left")
    b.set_title("(e) 日次 — 門ありはゼロ近傍を上下する", color=INK,
                fontsize=9.5, loc="left")

    b = ax[1, 2]
    K = S.filter(~pl.col("cfg").str.contains("tmax|_te")).sort("ev_pair")
    y = np.arange(K.height)
    b.axvline(0, color=BASELINE, lw=1.2)
    b.barh(y, K["ev_pair"].to_numpy(), xerr=K["se"].to_numpy(),
           color=[C_C if v > 0 else C_B for v in K["ev_pair"]], height=0.62,
           lw=0, error_kw={"lw": 1.0, "ecolor": INK2})
    b.set_yticks(y, [c.replace("q1", "在庫上限1").replace("q3", "在庫上限3")
                     .replace("_lat130", "+遅延130ms")
                     .replace("_gated", "+入口の門").replace("_fix10", "+価格固定10s")
                     for c in K["cfg"]], fontsize=8)
    style(b, "1 組あたり往復損益 (bp) — 誤差は日次 Newey-West", "")
    b.set_title("(f) 標本外 39 日 — 門を入れたときだけゼロを跨ぐ", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle(f"{coin} 在庫を持つメイカー — 相殺・時間切れ・入口の門"
                 f"(数値はすべて標本外 39 日)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_inv.png", dpi=170)
    plt.close(fig)


def fig_gate(tag: str, coin: str) -> None:
    C = pl.read_csv(DATA / f"unwind_cells_{tag}_q1.csv").sort("obi_d", "ofi_d")
    U = pl.read_parquet(DATA / f"unwind_{tag}_q1.parquet")
    fig, ax = plt.subplots(2, 3, figsize=(15.2, 8.6))

    def gg(c):
        return C[c].to_numpy().astype(float).reshape(10, 10)

    draw(ax[0, 0], 100 * gg("p_exit_1s"), "(a) 1 秒以内に相殺できた割合 (%)",
         SEQ, False, "{:.1f}")
    draw(ax[0, 1], gg("c_unwind_1s"), "(b) 1 秒で平らにする費用 (bp)",
         SEQ, False, "{:.2f}")
    draw(ax[0, 2], gg("pnl"), "(c) 組の往復損益 (bp) — 全セル負", DIV, True,
         "{:+.2f}")
    for b in (ax[0, 0], ax[0, 1], ax[0, 2]):
        b.set_xlabel("OFI^inv 十分位", color=INK2, fontsize=8.5)
        b.set_ylabel("OBI^inv 十分位", color=INK2, fontsize=8.5)

    b = ax[1, 0]
    o = U["obi_inv"].to_numpy()
    e = np.quantile(o, np.linspace(0, 1, 6)[1:-1])
    k = np.searchsorted(e, o, side="right")
    y1 = [100 * U["y_exit_1s"].to_numpy()[k == i].mean() for i in range(5)]
    y2 = [U["pnl"].to_numpy()[k == i].mean() for i in range(5)]
    b.plot(range(1, 6), y1, color=C_A, lw=2.2, marker="o", ms=5,
           label="1 秒以内に相殺できた割合 (%)")
    b.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)], fontsize=8.5)
    style(b, "OBI^inv 五分位(在庫の向きに揃えた板の偏り)", "1 秒相殺率 (%)")
    b2 = b.twinx()
    b2.axhline(0, color=BASELINE, lw=1.0)
    b2.plot(range(1, 6), y2, color=C_B, lw=2.2, marker="s", ms=5,
            label="組の往復損益 (bp)")
    b2.set_ylabel("組の往復損益 (bp)", color=INK2, fontsize=9)
    b2.tick_params(colors=INK2, labelsize=8.5)
    b2.spines["top"].set_visible(False)
    legend(b, loc="upper left")
    legend(b2, loc="lower right")
    b.set_title("★(d) OBI^inv は将来の値動きではなく「在庫の消しやすさ」を当てる",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    kf = DATA / f"inv_summary_{tag}.csv"
    K = pl.read_csv(kf).filter(pl.col("cfg").str.contains("_k"))
    if K.height:
        K = K.with_columns(k=pl.col("cfg").map_elements(
            lambda s: int(re.search(r"_k(\d+)", s).group(1)),
            return_dtype=pl.Int64)).sort("k")
        b.axhline(0, color=BASELINE, lw=1.0)
        b.errorbar(K["k"].to_numpy(), K["ev_pair"].to_numpy(),
                   yerr=K["se"].to_numpy(), color=C_B, lw=2.0, marker="s",
                   ms=5, capsize=3, label="1 組あたり損益")
        b2 = b.twinx()
        b2.plot(K["k"].to_numpy(), K["pairs_day"].to_numpy(), color=C_A,
                lw=2.0, marker="o", ms=5)
        b2.set_ylabel("1 日の組の数", color=INK2, fontsize=9)
        b2.tick_params(colors=INK2, labelsize=8.5)
        b2.spines["top"].set_visible(False)
        style(b, "手仕舞いを最良から何ティック外へ置くか", "1 組あたり損益 (bp)")
        legend(b, loc="lower left")
    else:
        b.text(0.5, 0.5, "手仕舞い価格の階段は計算中", ha="center",
               transform=b.transAxes, color=INK2)
        style(b, "", "")
    b.set_title("(e) 深く置くほど数が減り、単価の増分では埋まらない",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 2]
    P = pl.read_parquet(DATA / f"inv_posts_{tag}_q1.parquet")
    GA = pl.read_parquet(DATA / f"entrygate_{tag}_q1.parquet")
    files = sorted(Path("E:/Memory-quotes").joinpath(tag).glob("dt=*.parquet"))
    te = [f.stem.split("=")[1] for f in files[59:]]
    J = GA.join(P.select("dt", "t", "side", "filled", "rt_pnl"),
                on=["dt", "t", "side"]).filter(pl.col("dt").is_in(te))
    ev = J["ev"].to_numpy()
    pn = J["rt_pnl"].to_numpy()
    q = np.quantile(ev, np.linspace(0, 1, 11)[1:-1])
    d = np.searchsorted(q, ev, side="right")
    v = [np.nanmean(np.where(np.isfinite(pn[d == i]), pn[d == i], np.nan))
         for i in range(10)]
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(range(1, 11), v, color=[C_C if x > 0 else C_B for x in v], width=0.66,
          lw=0)
    b.set_xticks(range(1, 11), [f"D{i}" for i in range(1, 11)], fontsize=8)
    style(b, "予測 EV(往復損益で学習)の十分位", "実現した往復損益 (bp/建玉)")
    b.set_title("(f) 標本外 — 予測 EV の順に往復損益が並ぶ", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle(f"{coin} 在庫の向きに揃えた OBI/OFI と、往復損益で学習した入口の門",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(CH / f"{tag}_inv_gate.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    fig_main(tag, a.coin)
    fig_gate(tag, a.coin)
    print("書き出し", CH / f"{tag}_inv.png", CH / f"{tag}_inv_gate.png")
