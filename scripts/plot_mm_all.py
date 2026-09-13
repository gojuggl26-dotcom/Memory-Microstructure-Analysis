"""所有 7 銘柄のメイカーの経済的有意性(図)。

    uv run python scripts/plot_mm_all.py

出力: charts/allcoins_mm.png

配色: 7 銘柄を 7 色に塗り分けない(CVD 分離が取れない)。銘柄の識別は
軸のラベルで行い、色は「門なし / 無作為の門 / 門あり」と「正 / 負」の
対比にだけ使う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, INK, D, plt, save  # noqa: E402

COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]


def main() -> None:
    DC = pl.read_csv(D / "mm_all_decomp.csv")
    G = pl.read_csv(D / "mmgate_all.csv")
    SZ = pl.read_csv(D / "mm_size_curve.csv")
    SM = pl.read_csv(D / "mm_all_summary.csv")
    y = np.arange(len(COINS))

    fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.2))

    # A. 1 組あたり損益の分解 --------------------------------------------
    b = ax[0, 0]
    q = {c: DC.filter(pl.col("coin") == f"xyz:{c}").to_dicts()[0] for c in COINS
         if DC.filter(pl.col("coin") == f"xyz:{c}").height}
    quoted = np.array([q[c]["quoted_half_bp"] * 2 for c in COINS])
    decay = np.array([q[c]["predecay_bp"] * 2 for c in COINS])
    drift = np.array([-q[c]["drift_bp"] for c in COINS])
    fee = np.array([q[c]["fee_bp"] for c in COINS])
    ev = np.array([q[c]["ev_bp"] for c in COINS])
    b.barh(y, quoted, height=0.6, color=C3, label="出したときの半スプレッド ×2")
    b.barh(y, -decay, height=0.6, color=C1, label="約定までの mid の動き")
    b.barh(y, -drift, left=-decay, height=0.6, color=C2, label="保有中の drift")
    b.barh(y, -fee, left=-decay - drift, height=0.6, color=CM, label="手数料 ×2")
    b.plot(ev, y, "o", color=INK, ms=6, label="実現(= 左の和)")
    b.axvline(0, color=INK, lw=1.0)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_xlabel("bp / 1 組")
    b.set_title("A. 1 組あたり損益の分解(恒等式・門なし・全期間)\n"
                "     半スプレッドは約定するまでにほぼ消える", loc="left")
    b.legend(fontsize=6.6, frameon=False, loc="lower left")

    # B. 評価期間の 1 組あたり損益(3 設定)-------------------------------
    b = ax[0, 1]
    w = 0.26
    for k, (cfg, col) in enumerate((("門なし", C1), ("無作為の門", C3),
                                    ("門あり", C2))):
        v = []
        for c in COINS:
            r = G.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("cfg") == cfg))
            v.append(float(r["ev_pair_bp"][0]) if r.height else np.nan)
        b.barh(y + (k - 1) * w, v, height=w, color=col, label=cfg)
    b.axvline(0, color=INK, lw=1.1)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_xlabel("1 組あたり損益(bp)　0 = クオートを出さない")
    b.set_title("B. 入口の門の効果(評価期間だけ)\n"
                "     ★対照は「門なし」ではなく「無作為の門」", loc="left")
    b.legend(fontsize=7, frameon=False, loc="lower right")

    # C. 数量 × 1 組あたり bp(門あり)------------------------------------
    b = ax[0, 2]
    for c in COINS:
        z = (SZ.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("cfg") == "門あり")
                       & (pl.col("size_usd") > 0)).sort("size_usd"))
        if z.height < 2:
            continue
        col = C2 if c == "KIOXIA" else C1
        b.plot(z["size_usd"], z["pair_bp"], "-o", color=col, ms=3.4,
               lw=2.0 if c == "KIOXIA" else 1.0,
               alpha=1.0 if c == "KIOXIA" else 0.55)
        b.annotate(c, (float(z["size_usd"][-1]), float(z["pair_bp"][-1])),
                   fontsize=6.6, xytext=(4, 0), textcoords="offset points",
                   color=col)
    b.axhline(0, color=INK, lw=1.1)
    b.set_xscale("log")
    b.set_xlabel("自分のクオート数量(ドル)")
    b.set_ylabel("1 組あたり損益(bp)")
    b.set_title("C. 数量を入れると 1 組あたりの取り分は落ちる\n"
                "     太線 = xyz:KIOXIA(唯一 0 より上から始まる)", loc="left")

    # D. 数量 × 年間ドル(部分約定・門あり)--------------------------------
    b = ax[1, 0]
    for c in COINS:
        z = (SZ.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("cfg") == "門あり")
                       & (pl.col("size_usd") > 0)).sort("size_usd"))
        if z.height < 2:
            continue
        col = C2 if c == "KIOXIA" else C1
        b.plot(z["size_usd"], z["partial_usd_year"], "-o", color=col, ms=3.4,
               lw=2.0 if c == "KIOXIA" else 1.0,
               alpha=1.0 if c == "KIOXIA" else 0.55)
        b.annotate(c, (float(z["size_usd"][-1]),
                       float(z["partial_usd_year"][-1])),
                   fontsize=6.6, xytext=(4, 0), textcoords="offset points",
                   color=col)
    b.axhline(0, color=INK, lw=1.1)
    b.set_xscale("log")
    b.set_yscale("symlog", linthresh=1e3)
    b.set_ylim(-3e5, 1e5)
    b.set_xlabel("自分のクオート数量(ドル)")
    b.set_ylabel("1 年あたりの取り分(ドル、部分約定の見積り)")
    b.set_title("D. ★結論 — 金額にすると山は低く、すぐ負に折れる\n"
                "     0 = クオートを出さない。設備費は引いていない", loc="left")

    # E. xyz:KIOXIA の数量曲線(設定ごと)---------------------------------
    b = ax[1, 1]
    for cfg, col, ls in (("門なし", C1, "-"), ("門あり", C2, "-"),
                         ("門あり+130ms", C2, "--")):
        z = (SZ.filter((pl.col("coin") == "xyz:KIOXIA") & (pl.col("cfg") == cfg)
                       & (pl.col("size_usd") > 0)).sort("size_usd"))
        if z.height < 2:
            continue
        b.plot(z["size_usd"], z["partial_usd_year"], ls, marker="o", color=col,
               ms=4, lw=1.8, label=cfg)
    b.axhline(0, color=INK, lw=1.1)
    b.set_xscale("log")
    b.set_yscale("symlog", linthresh=1e3)
    b.set_ylim(-3e6, 1e5)
    b.set_xlabel("自分のクオート数量(ドル)")
    b.set_ylabel("1 年あたりの取り分(ドル)")
    b.set_title("E. xyz:KIOXIA — 唯一 0 を超える銘柄\n"
                "     山は 1,000 ドルで 2.2 万ドル。評価はわずか 19 日",
                loc="left")
    b.legend(fontsize=7.5, frameon=False)

    # F. 遅延 -------------------------------------------------------------
    b = ax[1, 2]
    for k, (L, col, m) in enumerate(((0, C1, "o"), (65, C3, "s"), (130, C2, "^"))):
        v = []
        for c in COINS:
            r = SM.filter((pl.col("coin") == f"xyz:{c}")
                          & (pl.col("size_unit") == 0) & (pl.col("lat_ms") == L))
            v.append(float(r["ev_pair_bp"][0]) if r.height else np.nan)
        b.plot(v, y, m, color=col, ms=6, label=f"{L} ms")
        if k == 0:
            for i in range(len(y)):
                b.plot([0, v[i]], [i, i], color=CM, lw=0.7, zorder=0)
    b.axvline(0, color=INK, lw=1.1)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_xlabel("1 組あたり損益(bp、幽霊注文・門なし・全期間)")
    b.set_title("F. 発注遅延の影響 — 130 ms で 0.18〜0.73bp 悪化", loc="left")
    b.legend(fontsize=7.5, frameon=False, title="実効遅延", title_fontsize=7)

    save(fig, "allcoins_mm.png",
         "所有 7 銘柄 — 在庫を持つメイカーの経済的有意性")


if __name__ == "__main__":
    main()
