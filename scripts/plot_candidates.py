"""三候補のまとめ図。

    uv run python scripts/plot_candidates.py --coin xyz:MU
出力: charts/<coin>_candidates.png

【読み方】
三候補は測っている単位が違う。往復損益(1 組あたり bp)へ寄せられるのは
候補 1 だけで、候補 2 は「価格の予測」から「往復損益」へ翻訳すると消える。
候補 3 は独立した戦略というより、候補 1 の中心的な主張を実参加者で照合する材料。
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
    S = pl.read_csv(DATA / "candidates_summary.csv")
    C2 = pl.read_csv(DATA / "cand2_signal.csv")
    C2q = pl.read_csv(DATA / "cand2_quintile.csv")
    C3 = pl.read_csv(DATA / "cand3_queue.csv")
    IS = pl.read_csv(DATA / f"inv_summary_{tag}.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.6))

    # (a) 共通の物差し
    b = ax[0, 0]
    L = S.filter(pl.col("group") == "ladder")
    y = np.arange(L.height)
    v = L["ev"].to_numpy()
    e = L["se"].to_numpy()
    b.axvline(0, color=BASELINE, lw=1.4)
    b.barh(y, v, xerr=1.96 * e, height=0.6, lw=0,
           color=[C_B if x < 0 else C_C for x in v],
           error_kw={"lw": 1.0, "ecolor": INK2})
    for i, x in enumerate(v):
        b.text(x + (0.06 if x > 0 else -0.06), i,
               f"{x:+.3f}", va="center", fontsize=8,
               ha="left" if x > 0 else "right", color=INK)
    b.set_yticks(y, L["label"].to_list(), fontsize=8)
    b.invert_yaxis()
    style(b, "1 組あたり往復損益 (bp) — 誤差は日次 Newey-West の 95%", "")
    b.set_title("(a) 唯一の共通の物差し — 標本外 39 日の往復損益",
                color=INK, fontsize=9.5, loc="left")

    # (b) 候補 1 の分解
    b = ax[0, 1]
    D = S.filter(pl.col("group") == "decomp")
    v2 = D["ev"].to_numpy()
    e2 = D["se"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.2)
    b.axhline(v2[0], color=MUTED, lw=1.0, ls=":")
    b.bar(range(D.height), v2, yerr=1.96 * e2,
          color=[MUTED, C_B, C_C, C_A], width=0.62, lw=0,
          error_kw={"lw": 1.0, "ecolor": INK2})
    for i, x in enumerate(v2):
        b.text(i, x - 0.17, f"{x:+.3f}", ha="center", fontsize=8.5, color=INK)
    b.set_xticks(range(D.height), D["label"].to_list(), fontsize=8)
    style(b, "", "1 組あたり往復損益 (bp)")
    b.set_title("★(b) 候補 1 の中身 — 価値があるのは順番で、値段ではない",
                color=INK, fontsize=9.5, loc="left")

    # (c) 候補 2: 価格は当てるが往復では消える
    b = ax[0, 2]
    cp = float(C2.filter(pl.col("metric").str.contains("リターン"))["corr"][0])
    cq = float(C2.filter(pl.col("metric").str.contains("往復"))["corr"][0])
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar([0, 1], [abs(cp), abs(cq)], color=[C_A, C_B], width=0.5, lw=0)
    b.text(0, abs(cp) + 0.004, f"{cp:+.4f}", ha="center", fontsize=9, color=INK)
    b.text(1, abs(cq) + 0.004, f"{cq:+.4f}", ha="center", fontsize=9, color=INK)
    b.set_xticks([0, 1], ["5 秒先の価格\n(既報の物差し)",
                          "自分の往復損益\n(戦略の物差し)"], fontsize=8.5)
    style(b, "", "相関の絶対値")
    b.set_title("★(c) 候補 2 の信号は約定条件を通ると 20 分の 1 になる",
                color=INK, fontsize=9.5, loc="left")

    # (d) 候補 2 の五分位
    b = ax[1, 0]
    q = C2q["ev"].to_numpy()
    b.axhline(0, color=BASELINE, lw=1.0)
    b.plot(range(1, 6), q, color=C_B, lw=2.2, marker="o", ms=6)
    b.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)], fontsize=8.5)
    b.set_ylim(min(q) - 0.35, max(q) + 0.35)
    style(b, "自分の側に揃えた A の五分位", "1 組あたり往復損益 (bp)")
    b.set_title("(d) 五分位でも平ら — 往復損益に勾配が無い", color=INK,
                fontsize=9.5, loc="left")

    # (e) 候補 3: 実参加者による照合
    b = ax[1, 1]
    x3 = np.arange(C3.height)
    b.bar(x3, C3["adv10"].to_numpy(), color=C_D, width=0.62, lw=0)
    for i, r in enumerate(C3.iter_rows(named=True)):
        b.text(i, r["adv10"] + 0.03, f"{100*r['share']:.0f}%", ha="center",
               fontsize=7.5, color=INK2)
    b.set_xticks(x3, C3["bin"].to_list(), fontsize=7.5, rotation=20, ha="right")
    style(b, "発注した時点で自分の前にあった数量", "約定 10 秒後の逆選択 (bp)")
    b.set_title("★(e) 候補 3 — 実参加者でも先頭は逆選択が 37% 少ない",
                color=INK, fontsize=9.5, loc="left")

    # (f) 遅延に対する候補 1
    b = ax[1, 2]
    rows = []
    for cfg in IS["cfg"]:
        if not cfg.startswith("q1") or "imp1_gated" not in cfg or "impact" in cfg:
            continue
        L_ = int(re.search(r"lat(\d+)", cfg).group(1)) if "lat" in cfg else 0
        d = IS.filter(pl.col("cfg") == cfg)
        rows.append((L_, float(d["ev_pair"][0]), float(d["se"][0])))
    rows.sort()
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    ss = [r[2] for r in rows]
    b.axhline(0, color=BASELINE, lw=1.4)
    b.fill_between(xs, [y - 1.96 * s for y, s in zip(ys, ss)],
                   [y + 1.96 * s for y, s in zip(ys, ss)], color=C_A,
                   alpha=0.16, lw=0)
    b.plot(xs, ys, color=C_A, lw=2.2, marker="o", ms=5)
    b.axvline(67.3, color=MUTED, lw=1.0, ls=":")
    b.text(69, min(ys) * 0.8, "1 ブロック", color=INK2, fontsize=8)
    style(b, "発注から板に載るまでの遅延 (ms)", "1 組あたり往復損益 (bp)")
    b.set_title("(f) 候補 1 — 区間の下限は遅延 0 でもゼロに届かない",
                color=INK, fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 三候補のまとめ — 往復損益に翻訳できたのは候補 1 だけ"
                 f"(数値はすべて標本外 39 日)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / f"{tag}_candidates.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_candidates.png")


if __name__ == "__main__":
    main()
