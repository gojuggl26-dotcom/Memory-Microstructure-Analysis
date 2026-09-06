"""日次損益による判定と、候補 2 の信号の劣化を描く。

    uv run python scripts/plot_dailypnl.py --coin xyz:MU
出力: charts/<coin>_daily_pnl.png

【読み方】
判定は単価ではなく**日次総額**で行う。両者は重みが違うので、
単価が非有意でも総額が有意なことがある(実際そうだった)。
候補 2 の信号は約定条件ではなく**古さ**で失われている。
"""
from __future__ import annotations

import argparse
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
    D = pl.read_csv(DATA / f"dailypnl_{tag}.csv")
    PR = pl.read_csv(DATA / f"dailypnl_pairs_{tag}.csv")
    SZ = pl.read_csv(DATA / f"dailypnl_size_{tag}.csv")
    DC = pl.read_csv(DATA / f"cand2_decay_{tag}.csv")
    ST = pl.read_csv(DATA / f"cand2_stages_{tag}.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.6))

    # (a) 日次総額(門を通す設定だけ。基準は桁が違うので外す)
    b = ax[0, 0]
    K = D.filter(~pl.col("cfg").is_in(["q1", "q1_imp1"]))
    y = np.arange(K.height)
    v = K["daily"].to_numpy()
    lo = K["lo"].to_numpy()
    hi = K["hi"].to_numpy()
    b.axvline(0, color=BASELINE, lw=1.4)
    b.barh(y, v, xerr=[v - lo, hi - v], height=0.6, lw=0,
           color=[C_C if x > 0 else C_B for x in v],
           error_kw={"lw": 1.0, "ecolor": INK2})
    for i, x in enumerate(v):
        b.text(x + (6 if x > 0 else -6), i, f"{x:+.0f}", va="center",
               fontsize=8, ha="left" if x > 0 else "right", color=INK)
    b.set_yticks(y, K["label"].to_list(), fontsize=8)
    b.invert_yaxis()
    style(b, "日次総額 (bp) — 誤差はブロック bootstrap 95%", "")
    b.set_title("★(a) 判定は日次総額で — 候補 1 は t=+3.62 でゼロを跨がない",
                color=INK, fontsize=9.5, loc="left")

    # (b) 2×2
    b = ax[0, 1]
    cells = {("BBO", "G0"): "q1_gated", ("BBO", "G1"): "q1_gatedby_q1_imp1",
             ("改善", "G0"): "q1_imp1_gatedby_q1",
             ("改善", "G1"): "q1_imp1_gatedby_q1_imp1"}
    xs, vs, es, lb = [], [], [], []
    for i, (k, cfg) in enumerate(cells.items()):
        r = D.filter(pl.col("cfg") == cfg)
        if not r.height:
            continue
        xs.append(i)
        vs.append(float(r["daily"][0]))
        es.append(float(r["se"][0]))
        lb.append(f"{k[0]}\n{k[1]}")
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(xs, vs, yerr=[1.96 * e for e in es], width=0.6, lw=0,
          color=[C_C if x > 0 else C_B for x in vs],
          error_kw={"lw": 1.0, "ecolor": INK2})
    for x, val in zip(xs, vs):
        b.text(x, val + (8 if val > 0 else -18), f"{val:+.0f}", ha="center",
               fontsize=8.5, color=INK)
    b.set_xticks(xs, lb, fontsize=8.5)
    style(b, "置き方 × 門", "日次総額 (bp)")
    b.set_title("★(b) 門は置き方に合わせないと害になる(BBO×G1 は有意に負)",
                color=INK, fontsize=9.5, loc="left")

    # (c) 対にした差
    b = ax[0, 2]
    y2 = np.arange(PR.height)
    v2 = PR["diff"].to_numpy()
    b.axvline(0, color=BASELINE, lw=1.4)
    b.barh(y2, v2, xerr=[v2 - PR["lo"].to_numpy(), PR["hi"].to_numpy() - v2],
           height=0.6, lw=0, color=[C_C if x > 0 else C_B for x in v2],
           error_kw={"lw": 1.0, "ecolor": INK2})
    b.set_yticks(y2, PR["label"].to_list(), fontsize=7.5)
    b.invert_yaxis()
    style(b, "同じ日で対にした差 (bp/日)", "")
    b.set_title("(c) 対にした差 — すべて 95% 区間がゼロを跨がない",
                color=INK, fontsize=9.5, loc="left")

    # (d) ドル換算
    b = ax[1, 0]
    s = SZ["size"].to_numpy()
    u = SZ["usd"].to_numpy()
    sh = 100 * SZ["share_of_touch"].to_numpy()
    b.plot(sh, u, color=C_A, lw=2.2, marker="o", ms=5)
    for x, yv, ss in zip(sh, u, s):
        b.annotate(f"{ss:.3g} 枚", (x, yv), fontsize=7.5, color=INK2,
                   xytext=(4, -8), textcoords="offset points")
    b.axhline(10, color=MUTED, lw=1.0, ls="--")
    b.axvline(100, color=C_B, lw=1.2, ls="--")
    b.text(102, 1.2, "最良気配の全量", color=C_B, fontsize=8, rotation=90)
    style(b, "1 建玉の数量 / 最良気配の中央数量 (%)", "日次ドル損益", logx=True,
          logy=True)
    b.set_title("★(d) 規模が出ない — $10/日 で気配の 12%、$100/日 で 122%",
                color=INK, fontsize=9.5, loc="left")

    # (e) 信号の劣化
    b = ax[1, 1]
    x3 = DC["delay_s"].to_numpy()
    c3 = np.abs(DC["corr"].to_numpy())
    b.plot(x3, c3, color=C_B, lw=2.2, marker="o", ms=6)
    b.axhline(c3[0] / 2, color=MUTED, lw=1.0, ls="--")
    b.text(2.0, c3[0] / 2 + 0.003, "半分", color=INK2, fontsize=8)
    b.axvline(2.49, color=C_D, lw=1.4, ls=":")
    b.text(2.55, c3[0] * 0.6, "発注時点の\n古さ 中央 2.49s", color=C_D, fontsize=8)
    style(b, "信号を寝かせた秒数 δ", "|corr(A, 5 秒先リターン)|")
    b.set_title("★(e) 候補 2 の信号の半減期は約 250 ms",
                color=INK, fontsize=9.5, loc="left")

    # (f) 段階分解
    b = ax[1, 2]
    qs = [f"q{i}" for i in range(1, 6)]
    for nm, c in (("1 全候補 E[U(5s)]", C_A),
                  ("2 約定した候補 E[U(5s)|Fill]", C_B),
                  ("5 往復損益 E[Π|Fill]", C_D)):
        r = ST.filter(pl.col("stage") == nm)
        if not r.height:
            continue
        b.plot(range(1, 6), [float(r[q][0]) for q in qs], lw=2.0, marker="o",
               ms=5, color=c, label=nm)
    b.axhline(0, color=BASELINE, lw=1.0)
    b.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)], fontsize=8.5)
    style(b, "自分の側に揃えた A の五分位", "bp")
    legend(b, loc="center left")
    b.set_title("(f) 段階分解 — 段階 1 の時点で既に勾配が無い",
                color=INK, fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 判定を日次損益に置き換える / 候補 2 は信号の古さで失われる"
                 f"(標本外 39 日)", color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / f"{tag}_daily_pnl.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_daily_pnl.png")


if __name__ == "__main__":
    main()
