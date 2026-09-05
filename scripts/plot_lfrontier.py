"""遅延フロンティアとシグナル減衰の図。

    uv run python scripts/plot_lfrontier.py --coin xyz:MU
出力: charts/<coin>_lfrontier.png

【読み方】
横軸は「発注を決めてから注文が板で約定可能になるまで」の遅延。
50 ms を超えると戦略の EV が負に落ちる。50 ms より内側では板が動いていないので
シグナルも EV もほとんど変わらない。境目はブロック周期(66.9 ms)にある。
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
BE = 50.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    F = pl.read_csv(DATA / f"lfrontier_{tag}.csv").sort("lag_ms")
    D = pl.read_csv(DATA / f"decay_{tag}.csv").sort("lag_ms")
    B = pl.read_csv(DATA / f"lfrontier_block_{tag}.csv").sort("net_delay_ms")
    G = pl.read_parquet(DATA / f"lfrontier_gaps_{tag}.parquet")["gap_ms"].to_numpy()

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.4))

    b = ax[0, 0]
    x, y, s = (F["lag_ms"].to_numpy().astype(float), F["ev_pair"].to_numpy(),
               F["se"].to_numpy())
    b.axhline(0, color=BASELINE, lw=1.2)
    b.axvspan(BE, x.max() * 1.05, color=C_B, alpha=0.08, lw=0)
    b.fill_between(x, y - 1.96 * s, y + 1.96 * s, color=C_A, alpha=0.16, lw=0)
    b.plot(x, y, color=C_A, lw=2.2, marker="o", ms=5)
    b.axvline(BE, color=C_B, lw=1.4, ls="--")
    b.text(BE + 3, y.min() * 0.85, f"損益分岐 ≈ {BE:.0f} ms", color=C_B,
           fontsize=9)
    b.set_xlim(-8, x.max() * 1.05)
    style(b, "発注を決めてから板で約定可能になるまで (ms)",
          "1 組あたり往復損益 (bp)")
    b.set_title("(A) 遅延フロンティア — 標本外 39 日・帯は 95% 区間",
                color=INK, fontsize=9.5, loc="left")

    b = ax[0, 1]
    xd = D["lag_ms"].to_numpy().astype(float)
    b.plot(xd, 100 * D["surv"].to_numpy(), color=C_C, lw=2.2, marker="o", ms=5,
           label="EV>0 の条件が残る割合")
    b.axvline(66.9, color=MUTED, lw=1.2, ls=":")
    b.text(68, 40, "1 ブロック 66.9 ms", color=INK2, fontsize=8.5)
    b.axvline(BE, color=C_B, lw=1.4, ls="--")
    style(b, "決めてから板に載るまでの遅延 (ms)", "条件の生存率 (%)", logx=True)
    b2 = b.twinx()
    b2.axhline(0, color=BASELINE, lw=1.0)
    b2.plot(xd, D["ev1"].to_numpy(), color=C_A, lw=2.0, marker="s", ms=4,
            label="そのときの予測 EV")
    b2.set_ylabel("載った時点の予測 EV (bp)", color=INK2, fontsize=9)
    b2.tick_params(colors=INK2, labelsize=8.5)
    b2.spines["top"].set_visible(False)
    legend(b, loc="lower left")
    legend(b2, loc="upper right")
    b.set_title("(B) シグナルの減衰 — 50 ms までは何も起きず、"
                "1 ブロックで崩れる", color=INK, fontsize=9.5, loc="left")

    b = ax[1, 0]
    b.axhline(0, color=BASELINE, lw=1.0)
    b.plot(xd, D["dmid"].to_numpy(), color=C_D, lw=2.2, marker="o", ms=5,
           label="自分が取ろうとした側への mid の動き (bp)")
    b.axvline(66.9, color=MUTED, lw=1.2, ls=":")
    b.axvline(BE, color=C_B, lw=1.4, ls="--")
    style(b, "決めてから板に載るまでの遅延 (ms)", "Δmid (bp)", logx=True)
    b2 = b.twinx()
    b2.plot(xd, D["dobi"].to_numpy(), color=C_B, lw=2.0, marker="s", ms=4)
    b2.set_ylabel("ΔOBI", color=INK2, fontsize=9)
    b2.tick_params(colors=INK2, labelsize=8.5)
    b2.spines["top"].set_visible(False)
    legend(b, loc="lower right")
    b.set_title("(C) 何が壊れているか — 板の偏りが先に消え、価格が追いかける",
                color=INK, fontsize=9.5, loc="left")

    # ★ 横軸の意味が違うものを重ねない。板の間隔と、ネットワーク遅延は別の図に分ける
    b = ax[0, 2]
    h, e = np.histogram(G[G < 340], bins=np.arange(0, 340, 2))
    b.bar(e[:-1], h / h.sum(), width=2, color=MUTED, lw=0, align="edge")
    for k in (1, 2, 3, 4):
        b.axvline(66.86 * k, color=C_A, lw=1.2, ls=":")
        b.text(66.86 * k + 3, 0.09, f"{k}", color=C_A, fontsize=8)
    style(b, "板更新の間隔 (ms)", "割合")
    b.set_title("(D) ブロックは 66.9 ms 周期(実測・板更新 192 万件)",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 1]
    xn = B["net_delay_ms"].to_numpy()
    yn = 100 * B["p_under_be_1block"].to_numpy()
    b.fill_between(xn, 0, yn, color=C_B, alpha=0.14, lw=0)
    b.plot(xn, yn, color=C_B, lw=2.2, marker="o", ms=5,
           label="次のブロックで入る場合")
    b.plot(xn, np.zeros_like(xn), color=MUTED, lw=2.0, ls="--",
           label="2 ブロック以上かかる場合(常に 0)")
    style(b, "ネットワーク遅延 (ms)", f"P(実効遅延 < {BE:.0f} ms) (%)")
    legend(b, loc="upper right")
    b.set_title("(E) 成立に必要な条件 — 送信が 30 ms を超えると 3 割を切る",
                color=INK, fontsize=9.5, loc="left")

    b = ax[1, 2]
    b.plot(x, F["pairs_day"].to_numpy(), color=C_C, lw=2.2, marker="o", ms=5)
    b.axvline(BE, color=C_B, lw=1.4, ls="--")
    style(b, "遅延 (ms)", "1 日の往復数")
    b.set_title("(F) 遅延は回数も削る — 130 組/日 → 65 組/日", color=INK,
                fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 新規発注の遅延はどこまで許されるか"
                 f"(実効遅延そのものは probe を出さないと測れない)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / f"{tag}_lfrontier.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_lfrontier.png")


if __name__ == "__main__":
    main()
