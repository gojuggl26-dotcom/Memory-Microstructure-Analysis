"""価格改善(1 ティック内側)と、その効果の分解を描く。

    uv run python scripts/plot_improve.py --coin xyz:MU
出力: charts/<coin>_improve.png

【読み方】
価格改善は「値段を 1 ティック譲る」ことと「行列の先頭に立つ」ことを同時に行う。
分解すると**利得は行列の側からしか出ていない**。値段だけ譲るのは基準より悪い。
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
from plot_vol import BASELINE, INK, INK2, MUTED, SURFACE, legend, style

ROOT = Path(__file__).resolve().parents[1]
DATA, CH = ROOT / "data", ROOT / "charts"
C_A, C_B, C_C, C_D = "#2a78d6", "#e34948", "#1f8a5e", "#c48a12"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    S = pl.read_csv(DATA / f"inv_summary_{tag}.csv")

    def g(c, col):
        d = S.filter(pl.col("cfg") == c)
        return float(d[col][0]) if d.height else np.nan

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    b = ax[0, 0]
    dec = [("基準\n最良気配・実際の行列", "q1"),
           ("値段だけ 1tick 改善\n行列は元のまま", "q1_imp1_price"),
           ("最良気配のまま\n行列の先頭", "q1_queue"),
           ("両方\n(= 価格改善)", "q1_imp1")]
    v = [g(c, "ev_pair") for _, c in dec]
    e = [g(c, "se") for _, c in dec]
    b.axhline(0, color=BASELINE, lw=1.2)
    b.axhline(v[0], color=MUTED, lw=1.0, ls=":")
    b.bar(range(4), v, yerr=[1.96 * x for x in e], color=[MUTED, C_B, C_C, C_A],
          width=0.62, lw=0, error_kw={"lw": 1.0, "ecolor": INK2})
    for i, x in enumerate(v):
        b.text(i, x - 0.16, f"{x:+.3f}", ha="center", fontsize=8.5, color=INK)
    b.set_xticks(range(4), [d[0] for d in dec], fontsize=7.5)
    style(b, "", "1 組あたり往復損益 (bp)")
    b.set_title("★(a) 分解 — 利得は行列の側からしか出ていない",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 1]
    for pre, c, nm in (("q1", C_A, "門のみ"), ("q1_imp1", C_C, "価格改善 + 門")):
        rows = []
        for cfg in S["cfg"]:
            if not cfg.endswith("_gated"):
                continue
            m = re.match(rf"^q1(_lat(\d+))?{'' if pre == 'q1' else '_imp1'}_gated$",
                         cfg)
            if pre == "q1" and "imp" in cfg:
                continue
            if pre == "q1_imp1" and "imp1" not in cfg:
                continue
            L = int(re.search(r"lat(\d+)", cfg).group(1)) if "lat" in cfg else 0
            rows.append((L, g(cfg, "ev_pair"), g(cfg, "se")))
        rows.sort()
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        s = [r[2] for r in rows]
        b.errorbar(x, y, yerr=[1.96 * z for z in s], color=c, lw=2.2, marker="o",
                   ms=5, capsize=3, label=nm)
    b.axhline(0, color=BASELINE, lw=1.2)
    b.axvline(67.3, color=MUTED, lw=1.0, ls=":")
    b.text(69, -0.45, "1 ブロック", color=INK2, fontsize=8)
    style(b, "発注から板に載るまでの遅延 (ms)", "1 組あたり往復損益 (bp)")
    legend(b, loc="lower left")
    b.set_title("(b) 遅延フロンティア — 改善しても分岐は 50 ms 台のまま",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 2]
    files = sorted(Path("E:/Memory-quotes").joinpath(tag).glob("dt=*.parquet"))
    te = [f.stem.split("=")[1] for f in files[59:]]
    bins = np.linspace(-12, 12, 121)
    for f, c, nm in ((f"inv_lots_{tag}_q1_gated.parquet", C_A, "門のみ"),
                     (f"inv_lots_{tag}_q1_imp1_gated.parquet", C_C, "価格改善 + 門")):
        L = pl.read_parquet(DATA / f).filter(pl.col("dt").is_in(te))
        v2 = L["pnl"].to_numpy()
        b.hist(v2, bins=bins, density=True, histtype="step", lw=1.8, color=c,
               label=f"{nm}  n={L.height:,}  中央 {np.median(v2):+.2f}")
    b.axvline(0, color=BASELINE, lw=1.0)
    style(b, "1 組の往復損益 (bp)", "密度")
    legend(b, loc="upper left")
    b.set_title("(c) 分布 — 改善版は中央値が +0.80 bp、正の割合 62.3%",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 0]
    cfgs = [("門のみ", "q1_gated"), ("価格改善+門", "q1_imp1_gated")]
    x = np.arange(2)
    b.bar(x, [g(c, "lots_day") for _, c in cfgs], color=[C_A, C_C], width=0.5,
          lw=0)
    for i, (_, c) in enumerate(cfgs):
        b.text(i, g(c, "lots_day") * 1.02, f"{g(c, 'lots_day'):,.0f}",
               ha="center", fontsize=9, color=INK)
    b.set_xticks(x, [c[0] for c in cfgs], fontsize=9)
    style(b, "", "1 日の往復数")
    b.set_title("(d) 同じ単価で回数が 8.2 倍になる", color=INK, fontsize=9.5,
                loc="left")

    b = ax[1, 1]
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(x, [g(c, "per_day_bp") for _, c in cfgs], color=[C_A, C_C], width=0.5,
          lw=0)
    for i, (_, c) in enumerate(cfgs):
        b.text(i, g(c, "per_day_bp") * 1.03, f"{g(c, 'per_day_bp'):+,.0f}",
               ha="center", fontsize=9, color=INK)
    b.set_xticks(x, [c[0] for c in cfgs], fontsize=9)
    style(b, "", "1 日あたり合計 (bp)")
    b.set_title("(e) 総額では 13 倍(+12 → +158 bp/日)", color=INK,
                fontsize=9.5, loc="left")

    b = ax[1, 2]
    ks = [("0(最良気配)", "q1"), ("1 ティック内側", "q1_imp1"),
          ("2 ティック内側", "q1_imp2")]
    v3 = [g(c, "ev_pair") for _, c in ks]
    e3 = [g(c, "se") for _, c in ks]
    b.axhline(0, color=BASELINE, lw=1.2)
    b.errorbar(range(3), v3, yerr=[1.96 * z for z in e3], color=C_D, lw=2.2,
               marker="o", ms=6, capsize=4)
    b.set_xticks(range(3), [k[0] for k in ks], fontsize=8.5)
    style(b, "改善の深さ", "1 組あたり往復損益 (bp)")
    b.set_title("(f) 1 ティックが最良 — 2 ティックは払いすぎ", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 1 ティック内側への価格改善(標本外 39 日)"
                 f" — 値段を譲る効果と行列の先頭に立つ効果を分ける",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / f"{tag}_improve.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_improve.png")


if __name__ == "__main__":
    main()
