"""ボリンジャーバンドの三本線と OBI / OFI — の図。

    uv run python scripts/plot_bb.py --coin xyz:MU

出力: charts/<tag>_bb.png

配色は 2 色(#3b6fd4 / #c2410c)+ 参照線の灰。
`palette_check.validate(['#3b6fd4','#c2410c'])` で ok=True。

x が確定する時刻 / y の期間: 図 E のみ前向き(説明変数は突破した足の終値まで、
目的変数はその後 5 分)。ほかは記述。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

BAND, MIDL = C2, C1


def merge(t: pl.DataFrame, evs, col: str) -> np.ndarray:
    """上下を件数で重み付けて 1 本にまとめる(符号は既に向きへそろえてある)。"""
    s = t.filter(pl.col("event").is_in(evs)).sort("rel_min")
    w = s.group_by("rel_min").agg(
        ((pl.col(col) * pl.col("n")).sum() / pl.col("n").sum()).alias("v")).sort("rel_min")
    return w["v"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--day", default="2026-06-09")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    pr = pl.read_csv(D / f"bb_profile_{tag}.csv")
    to = pl.read_csv(D / f"bb_touch_{tag}.csv")
    cr = pl.read_csv(D / f"bb_cross_{tag}.csv")
    bb = pl.read_parquet(D / f"bb_{tag}.parquet")

    fig, ax = plt.subplots(2, 3, figsize=(15.6, 8.0))

    b = ax[0, 0]
    b.plot(pr["z_mid"], pr["ofi_z"], color=BAND, lw=1.6, marker="o", ms=3.6,
           label="OFI(そのまま)")
    b.plot(pr["z_mid"], pr["ofi_z_resid"], color=MIDL, lw=1.6, marker="o", ms=3.6,
           label="直前 20 分のリターンを抜いた残差")
    b.axhline(0, color=CM, lw=0.8)
    for v in (-2, 0, 2):
        b.axvline(v, color=CM, lw=0.9, ls="--")
    b.set_xlabel("帯の位置 z =(終値 − MA)/ σ")
    b.set_ylabel("OFI(後ろ向き 60 分で標準化)")
    b.set_title("A. 帯の位置と OFI の向き — 端ほど強い", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax[0, 1]
    b.plot(pr["z_mid"], pr["abs_ofi_z"], color=BAND, lw=1.7, marker="o", ms=3.8)
    b.axhline(float(pr.filter(pl.col("z_mid").abs() < 0.3)["abs_ofi_z"].mean()),
              color=CM, lw=1.0, ls=":")
    for v in (-2, 0, 2):
        b.axvline(v, color=CM, lw=0.9, ls="--")
    b.text(0.02, float(pr["abs_ofi_z"].min()) * 1.02, "中間線あたりの水準",
           fontsize=7.5, color=CM)
    b.set_xlabel("帯の位置 z")
    b.set_ylabel("|OFI|(標準化)")
    b.set_title("B. 大きさで見ると U 字 — 両端は中間線の 2.2 倍", loc="left")

    b = ax[0, 2]
    x = to["rel_min"].unique().sort().to_numpy()
    for nm, evs, col in (("両バンドへの到達", ["上バンド到達", "下バンド到達"], BAND),
                         ("中間線の通過", ["中間線を上抜け", "中間線を下抜け"], MIDL)):
        b.plot(x, merge(to, evs, "ofi_mean"), color=col, lw=1.7, marker="o", ms=3.6,
               label=nm)
    b.axhline(0, color=CM, lw=0.8)
    b.axvline(0, color=CM, lw=1.1)
    b.set_xlabel("線に到達した分からの経過(分)")
    b.set_ylabel("OFI(到達の向きに符号をそろえた平均)")
    b.set_title("C. 到達した分に立ち上がり、次の 1 分で消える", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax[1, 0]
    for nm, evs, col in (("両バンドへの到達", ["上バンド到達", "下バンド到達"], BAND),
                         ("中間線の通過", ["中間線を上抜け", "中間線を下抜け"], MIDL)):
        b.plot(x, merge(to, evs, "ofi_excess"), color=col, lw=1.7, marker="o", ms=3.6,
               label=nm)
    b.axhline(0, color=CM, lw=0.8)
    b.axvline(0, color=CM, lw=1.1)
    b.set_xlabel("線に到達した分からの経過(分)")
    b.set_ylabel("超過 OFI(同じ 1 分リターンの層との差)")
    b.set_title("D. ★1 分の値動きを揃えても、到達の手前で OFI は高い", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax[1, 1]
    for nm, col in (("上バンド突破", BAND), ("下バンド突破", MIDL)):
        s = cr.filter(pl.col("event") == nm).sort("quintile")
        b.plot(s["quintile"], s["fwd5_mean"], color=col, lw=1.7, marker="o", ms=4.4,
               label=f"{nm}(n={int(s['n_all'][0]):,})")
    b.axhline(0, color=CM, lw=0.8)
    b.axhline(2.83, color=CM, lw=0.9, ls=":")
    b.text(1.05, 2.95, "往復の費用 2.83bp", fontsize=7.5, color=CM)
    b.set_xticks([1, 2, 3, 4, 5])
    b.set_xlabel("突破した分の OFI の五分位(右ほど強い)")
    b.set_ylabel("5 分後のリターン(bp・突破の向き)")
    b.set_title("E. 突破時の OFI が大きくても、その後は分けられない", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax[1, 2]
    d0 = int(np.datetime64(a.day + "T13:30:00", "s").astype("int64"))
    w = bb.filter((pl.col("ts") >= d0) & (pl.col("ts") < d0 + 3 * 3600)
                  & pl.col("usable")).sort("ts")
    xm = (w["ts"].to_numpy() - d0) / 60
    b.plot(xm, w["close"], color=CM, lw=1.1, label="終値(1 分)")
    b.plot(xm, w["ma"], color=MIDL, lw=1.2, label="MA(20)")
    b.plot(xm, w["upper"], color=BAND, lw=1.0)
    b.plot(xm, w["lower"], color=BAND, lw=1.0, label="±2σ")
    b.fill_between(xm, w["lower"], w["upper"], color=BAND, alpha=0.07, linewidth=0)
    b.set_xlabel(f"{a.day} の寄り付き(13:30 UTC)からの分")
    b.set_ylabel("価格(USD)")
    b.set_title(f"F. 実例 — {a.day} 13:30〜16:30 UTC", loc="left")
    b.legend(fontsize=7.5, frameon=False)
    save(fig, f"{tag}_bb.png",
         f"{a.coin}: ボリンジャーバンド(20, 2)の三本線と板の特徴量"
         f"(1 分足 140,499 本 / 98 日 / 2026-05-04 〜 08-09)")


if __name__ == "__main__":
    main()
