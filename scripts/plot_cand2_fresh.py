"""候補 2 を新鮮な信号(100 ms 格子)で検定し直した結果を描く。

    uv run python scripts/plot_cand2_fresh.py --coin xyz:MU
出力: charts/<coin>_cand2_fresh.png

【読み方】
前報の「約定条件を通ると情報が残らない」は誤りで、原因は**信号の古さ**だった。
5 秒格子では発注時点の古さが中央 2.49 秒(半減期 250ms に対し強度 3%)。
100 ms 格子にすると 0.050 秒になり、往復損益まで情報が残る。
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
AGE = {"5 秒格子(前報)": (2.49, 4.50), "100 ms 格子": (0.050, 0.090)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    S5 = pl.read_csv(DATA / f"cand2_stages_{tag}.csv")
    S1 = pl.read_csv(DATA / f"cand2_stages_{tag}_100.csv")
    D = pl.read_csv(DATA / f"dailypnl_{tag}.csv")
    P = pl.read_csv(DATA / f"dailypnl_pairs_{tag}.csv")
    DC = pl.read_csv(DATA / f"cand2_decay_{tag}.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 8.6))

    # (a) 信号の劣化曲線と、2 つの格子での古さ
    b = ax[0, 0]
    x = DC["delay_s"].to_numpy()
    c = np.abs(DC["corr"].to_numpy())
    b.plot(x, c, color=C_B, lw=2.2, marker="o", ms=5)
    b.axhline(c[0] / 2, color=MUTED, lw=1.0, ls="--")
    b.text(0.012, c[0] / 2 + 0.004, "半分(半減期 ≒ 250 ms)", color=INK2,
           fontsize=8)
    for nm, (md, _), col in (("5 秒格子(前報)", AGE["5 秒格子(前報)"], C_B),
                             ("100 ms 格子", AGE["100 ms 格子"], C_C)):
        b.axvline(md, color=col, lw=1.5, ls=":")
        b.text(md * 1.06, c[0] * (0.75 if col == C_C else 0.45),
               f"{nm}\n古さ 中央 {md:g}s", color=col, fontsize=8)
    b.set_xscale("log")
    # ★ このリポジトリは text.parse_math=False なので、対数軸の既定の目盛
    #   ラベル($\mathdefault{...}$)がそのまま文字として出る。明示する。
    tk = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.49, 5]
    b.set_xticks(tk, ["0.01", "0.05", "0.1", "0.25", "0.5", "1", "2.49", "5"],
                 fontsize=8)
    b.set_xticks([], minor=True)
    style(b, "信号を寝かせた秒数 δ(秒・対数目盛)", "|corr(A, 5 秒先リターン)|")
    b.set_title("★(a) 5 秒格子では強度の 3% しか残らない", color=INK,
                fontsize=9.5, loc="left")

    # (b) 段階分解の Q5−Q1
    b = ax[0, 1]
    lab = [s[2:] if s[1] == " " else s for s in S1["stage"].to_list()]
    y = np.arange(len(lab))
    w = 0.38
    b.axvline(0, color=BASELINE, lw=1.2)
    b.barh(y - w / 2, S5["spread"].to_numpy(), height=w, color=C_B, lw=0,
           label="5 秒格子(前報)")
    b.barh(y + w / 2, S1["spread"].to_numpy(), height=w, color=C_C, lw=0,
           label="100 ms 格子")
    b.set_yticks(y, lab, fontsize=7.5)
    b.invert_yaxis()
    style(b, "信号の五分位の Q5 − Q1 (bp)", "")
    legend(b, loc="center right")
    b.set_title("★(b) 新鮮なら段階 1 から勾配が立ち、往復まで残る",
                color=INK, fontsize=9.5, loc="left")

    # (c) 五分位ごとの往復損益
    b = ax[0, 2]
    qs = [f"q{i}" for i in range(1, 6)]
    for D_, nm, col in ((S5, "5 秒格子(前報)", C_B), (S1, "100 ms 格子", C_C)):
        r = D_.filter(pl.col("stage").str.contains("往復"))
        b.plot(range(1, 6), [float(r[q][0]) for q in qs], lw=2.2, marker="o",
               ms=6, color=col, label=nm)
    b.axhline(0, color=BASELINE, lw=1.0)
    b.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)], fontsize=9)
    style(b, "自分の側に揃えた A の五分位", "1 組あたり往復損益 (bp)")
    legend(b, loc="upper left")
    b.set_title("(c) 往復損益の勾配 — 全部負なのは変わらない", color=INK,
                fontsize=9.5, loc="left")

    # (d) 日次総額
    b = ax[1, 0]
    want = [("候補 1", "q1_imp1_gatedby_q1_imp1", C_A),
            ("+ 古い A(5 秒)", "q1_imp1_gated_impact", C_B),
            ("+ 新鮮な A(100 ms)", "q1_imp1_gatedby_q1_imp1_impact100", C_C)]
    xs, vs, lo, hi, cs, lb = [], [], [], [], [], []
    for i, (nm, cfg, col) in enumerate(want):
        r = D.filter(pl.col("cfg") == cfg)
        if not r.height:
            continue
        xs.append(i); vs.append(float(r["daily"][0]))
        lo.append(float(r["lo"][0])); hi.append(float(r["hi"][0]))
        cs.append(col); lb.append(nm)
    v = np.array(vs)
    b.axhline(0, color=BASELINE, lw=1.2)
    b.bar(xs, v, yerr=[v - np.array(lo), np.array(hi) - v], width=0.6, lw=0,
          color=cs, error_kw={"lw": 1.0, "ecolor": INK2})
    for i, val in zip(xs, v):
        b.text(i, val + 12, f"{val:+.0f}", ha="center", fontsize=9, color=INK)
    b.set_xticks(xs, lb, fontsize=8.5)
    style(b, "", "日次総額 (bp)")
    b.set_title("★(d) 古い A は害、新鮮な A は益(標本外 39 日)",
                color=INK, fontsize=9.5, loc="left")

    # (e) 「A を足したときの効果」に符号を揃える
    b = ax[1, 1]
    # ★ 対の差の向きが混在すると読み違える。すべて「足した効果」に揃える。
    #   古い A は P の "候補 1 − (候補 1+候補 2)" なので符号を反転する。
    src = [("古い A(5 秒格子)を足す", "候補 1 − (候補 1+候補 2)", -1.0),
           ("新鮮な A(100 ms)を足す", "★ 新鮮な A − 候補 1", +1.0),
           ("新鮮な A を足す(遅延 65 ms)",
            "★ 新鮮な A(遅延 65ms)− 候補 1(遅延 65ms)", +1.0)]
    lb2, v2, e2, t2 = [], [], [], []
    for nm, key, sg in src:
        r = P.filter(pl.col("label") == key)
        if not r.height:
            continue
        lb2.append(nm)
        v2.append(sg * float(r["diff"][0]))
        e2.append((sg * float(r["lo"][0]), sg * float(r["hi"][0])))
        t2.append(sg * float(r["t"][0]))
    v2 = np.array(v2)
    lo2 = np.array([min(x) for x in e2])
    hi2 = np.array([max(x) for x in e2])
    y2 = np.arange(v2.size)
    b.axvline(0, color=BASELINE, lw=1.4)
    b.barh(y2, v2, xerr=[v2 - lo2, hi2 - v2], height=0.55, lw=0,
           color=[C_C if x > 0 else C_B for x in v2],
           error_kw={"lw": 1.0, "ecolor": INK2})
    for i, (x, t_) in enumerate(zip(v2, t2)):
        b.text(x + (3 if x > 0 else -3), i, f"{x:+.1f} (t={t_:+.2f})",
               va="center", ha="left" if x > 0 else "right", fontsize=8,
               color=INK)
    b.set_yticks(y2, lb2, fontsize=8)
    b.invert_yaxis()
    b.set_xlim(min(lo2.min(), 0) - 25, max(hi2.max(), 0) + 30)
    style(b, "A を足したときの日次総額の変化 (bp/日)", "")
    b.set_title("★(e) 古い A は −28(害)、新鮮な A は +41(益)",
                color=INK, fontsize=9.5, loc="left")

    # (f) 信号の古さに対する応答(横軸は水準なので等間隔に置く)
    b = ax[1, 2]
    base = float(D.filter(pl.col("cfg") == "q1_imp1_gatedby_q1_imp1")["daily"][0])
    pts = [("100 ms 格子\n(A 1 変数)\n古さ 0.05s",
            "q1_imp1_gatedby_q1_imp1_impact100"),
           ("同 +100 ms 寝かせ\n(先読み検査)\n古さ 0.15s",
            "q1_imp1_gatedby_q1_imp1_impact100lag100"),
           ("5 秒格子\n(8 変数)\n古さ 2.49s", "q1_imp1_gated_impact")]
    xs, vs, lb = [], [], []
    for k, (nm, cfg) in enumerate(pts):
        r = D.filter(pl.col("cfg") == cfg)
        if not r.height:
            continue
        xs.append(len(xs)); vs.append(float(r["daily"][0])); lb.append(nm)
    b.axhline(base, color=C_A, lw=1.6, ls=":")
    b.text(len(xs) - 1.45, base - 13, "候補 1(A を足さない)", color=C_A,
           fontsize=8.5)
    b.plot(xs, vs, color=MUTED, lw=1.4, zorder=2)
    b.scatter(xs, vs, s=80, zorder=3, c=[C_C if v > base else C_B for v in vs])
    for x, v in zip(xs, vs):
        b.annotate(f"{v:+.0f}", (x, v), fontsize=9.5, color=INK, ha="center",
                   xytext=(0, 12), textcoords="offset points")
    b.set_xticks(xs, lb, fontsize=8)
    b.set_xlim(-0.45, len(xs) - 0.55)
    b.set_ylim(min(vs + [base]) - 30, max(vs + [base]) + 40)
    style(b, "", "日次総額 (bp)")
    b.set_title("(f) 1 セル寝かせても 85% 残る(先読みではない)",
                color=INK, fontsize=9.5, loc="left")

    fig.suptitle(f"{a.coin} 候補 2 は「約定条件で消える」のではなく"
                 f"「信号が古かった」(標本外 39 日)",
                 color=INK, fontsize=11.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CH / f"{tag}_cand2_fresh.png", dpi=170)
    plt.close(fig)
    print("書き出し", CH / f"{tag}_cand2_fresh.png")


if __name__ == "__main__":
    main()
