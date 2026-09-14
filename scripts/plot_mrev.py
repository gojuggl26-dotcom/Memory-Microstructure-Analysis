"""1 分足の平均回帰(図)。

    uv run python scripts/plot_mrev.py

出力: charts/allcoins_mrev.png

配色: 8 銘柄を 8 色に塗り分けない(CVD 分離が取れない)。細線の束を C1、
**平均回帰が有意だった 2 銘柄(xyz:SKHX / xyz:SMSN)だけ C2 の太線**にして
区別する。ランダムウォーク対照は CM の点線。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, C4, CM, INK, D, plt, save  # noqa: E402

COINS = ["MU", "SNDK", "INTC", "AMD", "SMSN", "DRAM", "KIOXIA", "SKHX"]
HI = {"SKHX", "SMSN"}                      # 平均回帰が有意だった銘柄
FAIRS = [("micro", "microprice"), ("ewma", "EWMA(半減期 7 分)"),
         ("sma20", "SMA20(主基準)"), ("model", "モデル fair")]


def sty(c):
    return (C2, 2.0, 1.0) if c in HI else (C1, 1.0, 0.55)


def main() -> None:
    A = pl.read_csv(D / "mrev_acf.csv")
    P = pl.read_csv(D / "mrev_speed.csv")
    V = pl.read_csv(D / "mrev_vr.csv")
    S = pl.read_csv(D / "mrev_summary.csv")
    y = np.arange(len(COINS))

    fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.0))

    # A. ρ(k) — mid と約定値
    b = ax[0, 0]
    for c in COINS:
        col, lw, al = sty(c)
        q = A.filter((pl.col("coin") == f"xyz:{c}")
                     & (pl.col("px") == "mid")).sort("lag")
        if q.height:
            b.plot(q["lag"], q["rho"], color=col, lw=lw, alpha=al)
        q = A.filter((pl.col("coin") == f"xyz:{c}")
                     & (pl.col("px") == "trade")).sort("lag")
        if q.height:
            b.plot(q["lag"], q["rho"], color=C3, lw=1.0, alpha=0.55, ls="--")
    b.axhline(0, color=INK, lw=1.0)
    b.plot([], [], color=C1, lw=1.4, label="mid(細線)")
    b.plot([], [], color=C2, lw=2.0, label="mid(xyz:SKHX / xyz:SMSN)")
    b.plot([], [], color=C3, lw=1.2, ls="--", label="約定値")
    b.set_xlabel("ラグ k(分)")
    b.set_ylabel("1 分リターンの自己相関 ρ(k)")
    b.set_title("A. 単独のラグではほぼ何も見えない\n"
                "     約定値だけ k=1 が負に振れる(bid–ask bounce)", loc="left")
    b.legend(fontsize=7, frameon=False)

    # B. 分散比
    b = ax[0, 1]
    for c in COINS:
        col, lw, al = sty(c)
        q = V.filter(pl.col("coin") == f"xyz:{c}").sort("k")
        if not q.height:
            continue
        b.plot(q["k"], q["vr"], "-o", color=col, lw=lw, ms=3.6, alpha=al)
        b.plot(q["k"], q["vr_rw"], ":", color=CM, lw=0.9)
        if c in HI:
            b.annotate(c, (float(q["k"][-1]), float(q["vr"][-1])), fontsize=7,
                       xytext=(4, 0), textcoords="offset points", color=C2)
    b.axhline(1.0, color=INK, lw=1.1)
    b.set_xscale("log")
    b.set_xticks([2, 5, 10, 20, 30, 60])
    b.set_xticklabels(["2", "5", "10", "20", "30", "60"])
    b.xaxis.set_minor_locator(plt.NullLocator())
    b.xaxis.set_minor_formatter(plt.NullFormatter())
    b.set_xlabel("ホライズン k(分)")
    b.set_ylabel("分散比 VR(k)")
    b.set_title("B. 分散比だと見える — 5〜30 分で反転\n"
                "     点線 = 同じ欠測・ボラ・ティックのランダムウォーク", loc="left")

    # C. φ(SMA20) は実測とランダムウォークでほぼ同じ
    b = ax[0, 2]
    q = P.filter(pl.col("fair") == "sma20")
    xs = [float(q.filter(pl.col("coin") == f"xyz:{c}")["phi_rw"][0]) for c in COINS]
    ys = [float(q.filter(pl.col("coin") == f"xyz:{c}")["phi"][0]) for c in COINS]
    zs = [float(q.filter(pl.col("coin") == f"xyz:{c}")["z_vs_rw"][0]) for c in COINS]
    lo = min(min(xs), min(ys)) - 0.004
    hi = max(max(xs), max(ys)) + 0.004
    b.plot([lo, hi], [lo, hi], color=CM, lw=1.2, ls=":",
           label="実測 = ランダムウォーク")
    off = {"KIOXIA": (7, 6), "AMD": (7, -3), "MU": (-7, 8), "INTC": (7, -12),
           "DRAM": (-7, 7), "SNDK": (-7, -9), "SKHX": (9, 2), "SMSN": (9, -8)}
    for c, x, v, z in zip(COINS, xs, ys, zs):
        col = C2 if c in HI else C1
        b.scatter(x, v, s=52, color=col, zorder=3)
        dx, dy = off.get(c, (6, -3))
        b.annotate(f"{c} (z={z:+.1f})", (x, v), fontsize=6.6,
                   xytext=(dx, dy), ha="right" if dx < 0 else "left",
                   textcoords="offset points", color=col)
    b.set_xlabel("ランダムウォークの φ")
    b.set_ylabel("実測の φ")
    b.set_title("C. ★ φ=0.92・半減期 8 分は「平均回帰」ではない\n"
                "     移動平均の残差は RW でも同じ値になる", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    # D. 基準価格ごとの半減期
    b = ax[1, 0]
    w = 0.2
    for k, (nm, lab) in enumerate(FAIRS):
        v = []
        for c in COINS:
            r = P.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("fair") == nm))
            v.append(float(r["half_life_min"][0]) if r.height else np.nan)
        b.barh(y + (k - 1.5) * w, v, height=w,
               color=[C1, C3, C2, C4][k], label=lab)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_ylim(-0.6, len(COINS) - 1 + 1.6)
    b.set_xscale("log")
    b.set_xlabel("D_t = mid − F_t の半減期(分、対数)")
    b.set_title("D. 「何への回帰か」で半減期は 30 倍違う\n"
                "     microprice / モデルは 1 分未満 = 1 分足では測れない",
                loc="left")
    b.legend(fontsize=6.8, frameon=False, loc="upper right")

    # E. 立会と時間外
    b = ax[1, 1]
    for i, c in enumerate(COINS):
        r = S.filter(pl.col("coin") == f"xyz:{c}").to_dicts()[0]
        col = C2 if c in HI else C1
        b.plot([r["vr20_oth"], r["vr20_rth"]], [i, i], color=CM, lw=1.0, zorder=0)
        b.plot(r["vr20_oth"], i, "o", color=C1, ms=6,
               label="時間外" if i == 0 else None)
        b.plot(r["vr20_rth"], i, "D", color=col, ms=6,
               label="米国立会 13:30–20:00 UTC" if i == 0 else None)
        b.text(1.19, i, f"z={r['vr20_diff_z']:+.1f}", fontsize=6.6,
               ha="right", va="center", color=INK)
    b.axvline(1.0, color=INK, lw=1.1)
    b.set_xlim(0.69, 1.21)
    b.set_ylim(-0.7, len(COINS) - 1 + 1.3)
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_xlabel("VR(20)")
    b.set_title("E. 平均回帰は米国立会時間のほうが強い\n"
                "     8 銘柄中 7 銘柄で立会のほうが低い", loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="lower right")

    # F. 乖離の大きさ
    b = ax[1, 2]
    for k, (nm, lab) in enumerate(FAIRS):
        v = []
        for c in COINS:
            r = P.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("fair") == nm))
            v.append(float(r["sd_bp"][0]) if r.height else np.nan)
        b.barh(y + (k - 1.5) * w, v, height=w,
               color=[C1, C3, C2, C4][k], label=lab)
    sp = [float(S.filter(pl.col("coin") == f"xyz:{c}")["spread_bp"][0])
          for c in COINS]
    b.plot(sp, y, "o", color=INK, ms=5, label="中央スプレッド")
    b.set_yticks(y)
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_ylim(-0.6, len(COINS) - 1 + 1.6)
    b.set_xscale("log")
    b.set_xlabel("D_t の標準偏差(bp、対数)")
    b.set_title("F. 乖離の大きさ — SMA20 基準なら 25〜47bp\n"
                "     microprice 基準は 0.6〜7bp でスプレッド並み", loc="left")
    b.legend(fontsize=6.8, frameon=False, loc="upper left")

    save(fig, "allcoins_mrev.png",
         "1 分足の平均回帰 — 自己相関・回帰速度・分散比(8 銘柄)")


if __name__ == "__main__":
    main()
