"""価格発見の検証の図。

    uv run python scripts/plot_pricedisc.py --coin xyz:MU

出力: charts/<tag>_pricedisc.png     寄り付きの跳ね・予測誤差・増分 R²・gap の追随
      charts/<tag>_pricedisc_rv.png  時刻別ボラ・夜間リターンの一致・代表日・先読み検査

配色は 2 色まで(#3b6fd4 = 現物/プレマーケット、#c2410c = perp)。
`palette_check.validate` で ok=True を確認済み。

x が確定する時刻 / y の期間: 図 1D と図 2D のみ時間の向きを持つ。
1D は説明変数が 13:30:00 までに確定し目的変数は 13:30 以降、
2D は説明変数が 13:30:00 まで、目的変数は 13:30〜14:30。先読みは無い。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

CASH, PERP = C1, C2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    p = pl.read_csv(D / f"pricedisc_panel_{tag}.csv")
    op = pl.read_csv(D / f"pricedisc_open_{tag}.csv")
    er = pl.read_csv(D / f"pricedisc_err_{tag}.csv")
    rv = pl.read_csv(D / f"pricedisc_rv_{tag}.csv")

    # ================= 図 1 =================
    fig, ax = plt.subplots(2, 2, figsize=(14.4, 7.8))

    b = ax[0, 0]
    s = op["sec"].to_numpy()
    b.plot(s, op["abs_med_trading"], color=PERP, lw=1.5, label="立会日(n=67)")
    b.plot(s, op["abs_med_holiday"], color=CM, lw=1.3, ls="--", label="非立会日(n=31)")
    b.axvline(0, color=CASH, lw=1.2)
    b.set_yscale("log")
    b.set_xlim(-900, 900)
    b.set_xlabel("米国市場の寄り付き(13:30:00 UTC)からの秒数")
    b.set_ylabel("|mid の変化|(bp・中央値)")
    b.set_title("A. 寄り付きの瞬間に perp は跳ねる — 直後 60 秒は直前 60 秒の 4.6 倍、"
                "1 秒で 13.8bp(非立会日の 271 倍)", loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    b.annotate("寄り付き", (0, op["abs_med_trading"].max()), fontsize=8, color=CASH,
               xytext=(6, -2), textcoords="offset points")

    b = ax[0, 1]
    e = er.sort("abs_med_bp", descending=True)
    y = np.arange(e.height)
    col = [PERP if "perp" in s_ else (CM if ("先物" in s_ or "何もしない" in s_) else CASH)
           for s_ in e["source"]]
    b.barh(y, e["abs_med_bp"], color=col, height=0.62)
    for i, (v, r) in enumerate(zip(e["abs_med_bp"], e["rmse_bp"])):
        b.text(v * 1.06, i, f"{v:.1f}(RMSE {r:.0f})", fontsize=7, va="center")
    b.set_yticks(y)
    b.set_yticklabels(e["source"], fontsize=8)
    b.invert_yaxis()
    b.set_xscale("log")
    b.set_xlim(3, 1400)
    b.set_xlabel("寄り値の予測誤差 |bp|(中央値・対数)")
    b.set_title(f"B. 13:29:59 の値段は寄り値をどれだけ当てるか(共通 {e['n'][0]} 日)",
                loc="left")

    b = ax[1, 0]
    lab = ["先物のみ → +プレ\n(13:00 で測る)", "先物+プレ → +perp\n(13:00 で測る)",
           "先物+プレ → +perp\n(13:29:59 で測る)", "先物+perp → +プレ\n(13:29:59 で測る)",
           "現物+先物 → +夜の perp\n(プレ抜き)", "現物+先物+プレ → +夜の perp"]
    val = [0.2817, 0.0316, 0.0002, 0.0018, 0.0174, 0.0000]
    pvv = ["p=1e-32", "p=3e-13", "p=0.094", "p=2e-05", "p=0.014", "p=0.78"]
    colb = [CASH, PERP, PERP, CASH, PERP, PERP]
    yy = np.arange(len(val))
    b.barh(yy, val, color=colb, height=0.62)
    for i, (v, q) in enumerate(zip(val, pvv)):
        b.text(max(v, 0.0002) * 1.15, i, q, fontsize=7.5, va="center")
    b.set_yticks(yy)
    b.set_yticklabels(lab, fontsize=7.5)
    b.invert_yaxis()
    b.set_xscale("log")
    b.set_xlim(1e-4, 1.0)
    b.set_xlabel("寄り値の説明力の増分 ΔR²(対数)")
    b.set_title("C. 同じ時刻でそろえると、perp の上乗せは消える", loc="left")

    b = ax[1, 1]
    g = p["gap"].to_numpy()
    j = p["R_perp_open1m"].to_numpy()
    m = np.isfinite(g) & np.isfinite(j)
    b.scatter(g[m], j[m], s=26, color=PERP, alpha=0.8, linewidths=0)
    xs = np.linspace(np.nanmin(g[m]), np.nanmax(g[m]), 10)
    A = np.c_[np.ones(m.sum()), g[m]]
    bb = np.linalg.lstsq(A, j[m], rcond=None)[0]
    b.plot(xs, bb[0] + bb[1] * xs, color=CM, lw=1.4)
    b.plot(xs, xs, color=CASH, lw=1.0, ls="--", label="差を 100% 埋める線")
    b.axhline(0, color=CM, lw=0.6)
    b.axvline(0, color=CM, lw=0.6)
    b.set_xlabel("gap = 現物の夜間リターン − perp の夜間リターン(bp)")
    b.set_ylabel("perp の 13:30:00→13:31:00 リターン(bp)")
    b.set_title(f"D. perp は寄り値を追いかける(傾き {bb[1]:.2f}・t=6.2・n={int(m.sum())})",
                loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    save(fig, f"{tag}_pricedisc.png",
         f"{a.coin}: 米国市場の寄り付きに対して perp は先行するか"
         f"(2026-05-04 〜 09-10 の 90 営業日 / 1 秒 mid は 67 日)")

    # ================= 図 2 =================
    fig2, ax2 = plt.subplots(2, 2, figsize=(14.4, 7.8))

    b = ax2[0, 0]
    pv = rv.pivot(values="rv_bp", index="hour", on="kind").sort("hour")
    h = pv["hour"].to_numpy()
    b.plot(h + 0.5, pv["立会日"], color=PERP, lw=1.6, marker="o", ms=3.4,
           label="米国市場が開く日")
    b.plot(h + 0.5, pv["非立会日"], color=CM, lw=1.4, ls="--", marker="o", ms=3,
           label="週末・休場日")
    b.axvspan(13.5, 20, color=CASH, alpha=0.10)
    b.axvspan(8, 13.5, color=CASH, alpha=0.045)
    b.text(16.7, b.get_ylim()[1] * 0.93, "レギュラー", fontsize=7.5, color=CASH, ha="center")
    b.text(10.7, b.get_ylim()[1] * 0.93, "プレ", fontsize=7.5, color=CASH, ha="center")
    b.text(4, b.get_ylim()[1] * 0.93, "現物はどこも動かない", fontsize=7.5, color=CM,
           ha="center")
    b.set_xlim(0, 24)
    b.set_xticks(range(0, 25, 3))
    b.set_xlabel("UTC の時刻")
    b.set_ylabel("1 時間あたり実現ボラティリティ(bp)")
    b.set_title("A. 現物がどこでも動かない 8 時間で、1 日の分散の 25% が作られる", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax2[0, 1]
    x = p["R_perp_on_s"].to_numpy()
    yv = p["R_cash_on"].to_numpy()
    m = np.isfinite(x) & np.isfinite(yv)
    b.scatter(x[m], yv[m], s=26, color=PERP, alpha=0.8, linewidths=0)
    lim = np.nanmax(np.abs(np.r_[x[m], yv[m]])) * 1.08
    b.plot([-lim, lim], [-lim, lim], color=CM, lw=1.0, ls="--")
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    r = np.corrcoef(x[m], yv[m])[0, 1]
    b.set_xlabel("perp の夜間リターン(20:00 UTC → 13:29:59・bp)")
    b.set_ylabel("現物の夜間リターン(前引け → 寄り値・bp)")
    b.set_title(f"B. 夜の動きはほぼ一致する(相関 {r:.4f}・n={int(m.sum())})", loc="left")

    b = ax2[1, 0]
    k = int(np.nanargmax(np.abs(np.nan_to_num(p["gap"].to_numpy()))))
    day = p["day"][k]
    z = np.load(D / f"_liqmid_{tag}.npz")
    mid, t0 = z["mid"], int(z["t0"])
    ds = int(np.datetime64(day + "T00:00:00", "s").astype("int64"))
    lo, hi = ds + 12 * 3600, ds + 15 * 3600
    xs = np.arange(lo, hi) - ds
    seg = mid[lo - t0:hi - t0]
    b.plot(xs / 3600, seg, color=PERP, lw=1.0, label="perp の mid(1 秒)")
    for nm, v, st in (("前日の引け", p["cash_prev_close"][k], ":"),
                      ("寄り値", p["cash_open"][k], "-"),
                      ("1 時間後", p["cash_h1"][k], "--")):
        b.axhline(v, color=CASH, lw=1.1, ls=st)
        b.text(15.02, v, nm, fontsize=7.5, color=CASH, va="center")
    b.axvline(13.5, color=CM, lw=1.2)
    b.set_xlim(12, 15.35)
    b.set_xticks([12, 12.5, 13, 13.5, 14, 14.5, 15])
    b.set_xticklabels(["12:00", "12:30", "13:00", "13:30\n寄り", "14:00", "14:30", "15:00"],
                      fontsize=7.5)
    b.set_xlabel("UTC")
    b.set_ylabel("価格(USD)")
    b.set_title(f"C. gap が最大だった日({day}・gap {p['gap'][k]:.0f}bp)", loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="lower left")

    b = ax2[1, 1]
    pp = p
    # 寄り 5 分前までの perp の動き(1 秒 mid から作り直す)
    x5 = np.array([np.log(mid[int(np.datetime64(d + "T00:00:00", "s").astype("int64"))
                              + 13 * 3600 + 1800 - t0]
                          / mid[int(np.datetime64(d + "T00:00:00", "s").astype("int64"))
                                + 13 * 3600 + 1500 - t0]) * 1e4
                   if (int(np.datetime64(d + "T00:00:00", "s").astype("int64"))
                       + 13 * 3600 + 1800 - t0) < mid.size else np.nan
                   for d in pp["day"]])
    yv = pp["R_h1"].to_numpy()
    m = np.isfinite(x5) & np.isfinite(yv)
    b.scatter(x5[m], yv[m], s=26, color=PERP, alpha=0.8, linewidths=0)
    A = np.c_[np.ones(m.sum()), x5[m]]
    bb = np.linalg.lstsq(A, yv[m], rcond=None)[0]
    xs = np.linspace(np.nanmin(x5[m]), np.nanmax(x5[m]), 10)
    b.plot(xs, bb[0] + bb[1] * xs, color=CM, lw=1.4)
    b.axhline(0, color=CM, lw=0.6)
    b.axvline(0, color=CM, lw=0.6)
    rr = np.corrcoef(x5[m], yv[m])[0, 1]
    b.set_xlabel("perp の 13:25→13:30 リターン(寄り前の 5 分・bp)")
    b.set_ylabel("現物の 寄り→1 時間後 リターン(bp)")
    b.set_title(f"D. 寄り前の perp は、寄り後 1 時間を当てない"
                f"(相関 {rr:.2f}・t=1.13・n={int(m.sum())})", loc="left")
    save(fig2, f"{tag}_pricedisc_rv.png",
         f"{a.coin}: 現物が閉まっている時間の動き / 夜間の一致 / 寄り付きの当日例")


if __name__ == "__main__":
    main()
