"""条件つきの出し直し(候補 1 手順②)の結果を描く。

    uv run python scripts/plot_requote.py --coin xyz:MU
出力: charts/<coin>_requote.png

【読み方】
注文を維持すると**約定率は上がるのに損益は下がる**。増える約定は
逆選択された約定だけである。価値があるのは「最良に居ること」であって
「板に居ること」ではない。
"""
from __future__ import annotations

import argparse
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
BONF = 2.891


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = pl.read_csv(DATA / f"dailypnl_rq_{tag}.csv")
    P = pl.read_csv(DATA / f"dailypnl_pairs_rq_{tag}.csv")
    M = pl.read_csv(DATA / f"requote_mech_{tag}.csv")

    fig, ax = plt.subplots(2, 2, figsize=(13.6, 8.8))

    # (a) 日次総額。keep は桁が違うので別扱い
    b = ax[0, 0]
    K = D.filter(~pl.col("label").str.starts_with("keep"))
    y = np.arange(K.height)
    v = K["daily"].to_numpy()
    b.axvline(0, color=BASELINE, lw=1.4)
    b.axvline(v[0], color=MUTED, lw=1.0, ls=":")
    b.barh(y, v, xerr=[v - K["lo"].to_numpy(), K["hi"].to_numpy() - v],
           height=0.62, lw=0,
           color=[C_A if i == 0 else (C_C if x > 0 else C_B)
                  for i, x in enumerate(v)],
           error_kw={"lw": 1.0, "ecolor": INK2})
    b.set_yticks(y, K["label"].to_list(), fontsize=7.5)
    b.invert_yaxis()
    style(b, "日次総額 (bp) — 誤差はブロック bootstrap 95%", "")
    b.set_title("(a) 出し直しを止めるほど悪くなる(点線 = 従来 always)",
                color=INK, fontsize=9.5, loc="left")

    # (b) 対にした差
    b = ax[0, 1]
    Q = P.filter(~pl.col("label").str.contains("keep"))
    t = Q["t"].to_numpy()
    y2 = np.arange(Q.height)
    b.axvline(0, color=BASELINE, lw=1.4)
    b.axvline(BONF, color=C_D, lw=1.2, ls="--")
    b.text(BONF + 0.12, -0.75, f"Bonferroni {BONF:.2f}", color=C_D,
           fontsize=8)
    b.barh(y2, t, height=0.62, lw=0,
           color=[C_C if x > BONF else (MUTED if x > 0 else C_B) for x in t])
    b.set_yticks(y2, [s.replace("always − (", "").replace(")", "")
                      for s in Q["label"].to_list()], fontsize=7.5)
    b.invert_yaxis()
    style(b, "always − 各規則 の t 値(正 = always が優る)", "")
    nsig = int((t > BONF).sum())
    b.set_title(f"★(b) {Q.height} 通り中 {nsig} 通りで always が Bonferroni 水準で優る",
                color=INK, fontsize=9.5, loc="left")

    # (c) 離れても維持するほど悪くなる(kd 別・km 別)
    b = ax[1, 0]
    alw = float(M.filter(pl.col("km") == 0)["daily"][0])
    b.axhline(0, color=BASELINE, lw=1.0)
    b.axhline(alw, color=C_A, lw=1.6, ls=":")
    b.text(0.05, alw + 8, "always(毎回出し直す)", color=C_A, fontsize=8.5)
    xs = [0, 1, 2, 3]
    for km, c in ((1, C_C), (5, C_D), (60, C_B)):
        G = M.filter(pl.col("km") == km).sort("kd")
        b.plot(xs[:G.height], G["daily"].to_numpy(), color=c, lw=2.0,
               marker="o", ms=5, label=f"維持の上限 {km} 秒")
    b.set_xticks(xs, ["1", "2", "5", "∞"], fontsize=9)
    style(b, "最良から何ティック離れても維持するか", "日次総額 (bp)")
    legend(b, loc="lower left")
    b.set_title("(c) 離れるほど悪くなる(kd=∞ は発注が 1/5 に減り別物)",
                color=INK,
                fontsize=9.5, loc="left")

    # (d) 約定率と損益が逆を向く
    b = ax[1, 1]
    fp = 100 * M["fill_per_post"].to_numpy()
    dy = M["daily"].to_numpy()
    kr = 100 * M["keep_rate"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.0)
    sc = b.scatter(fp, dy, s=58, c=kr, cmap="YlOrBr", zorder=3,
                   edgecolors=INK2, linewidths=0.5)
    b.scatter(fp[0], dy[0], s=110, facecolors="none", edgecolors=C_A,
              linewidths=2.0, zorder=4)
    b.annotate("always", (fp[0], dy[0]), fontsize=9, color=C_A,
               xytext=(7, -12), textcoords="offset points")
    r = float(np.corrcoef(fp, dy)[0, 1])
    cb = fig.colorbar(sc, ax=b, pad=0.02)
    cb.set_label("維持率 (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    style(b, "約定 / 発注 (%)", "日次総額 (bp)")
    b.set_title(f"★(d) 約定は増えるのに損益は下がる(相関 {r:+.2f})",
                color=INK, fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 条件つきの出し直しは効かない — 価値は「板に居ること」"
                 f"ではなく「最良に居ること」(標本外 39 日)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(CH / f"{tag}_requote.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_requote.png")


if __name__ == "__main__":
    main()
