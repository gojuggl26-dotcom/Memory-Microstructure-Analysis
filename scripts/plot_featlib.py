"""analyze_featlib.py の出力を図にする。

    uv run python scripts/plot_featlib.py --coin xyz:INTC [--ref xyz:MU]

出力: charts/<tag>_featlib.png(6 枚組)と charts/<tag>_featlib_corr.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, D, plt, save  # noqa: E402

TGT = "fwd_micro_10s"
BWD = "bwd_mid_10s"
PLB = "pl_micro_10s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    pr = pl.read_csv(D / f"featlib_pred_{tag}.csv")
    st = pl.read_csv(D / f"featlib_stats_{tag}.csv")
    pr = pr.filter(pl.col(TGT).is_finite())

    fig, ax = plt.subplots(2, 3, figsize=(15.4, 7.6))

    # --- A. 分類ごとの本数と最大の予測相関 --------------------------------
    g = (pr.group_by("family").agg(pl.len().alias("n"),
                                   pl.col(TGT).abs().max().alias("mx"),
                                   pl.col(BWD).abs().max().alias("mb"))
         .sort("mx", descending=True))
    y = np.arange(g.height)
    a0 = ax[0, 0]
    a0.barh(y, g["mx"], color=C1, height=0.42, label="前向き |r| の最大")
    a0.barh(y + 0.44, g["mb"], color=CM, height=0.42, label="後ろ向き |r| の最大")
    a0.set_yticks(y + 0.22)
    a0.set_yticklabels([f"{s} ({n})" for s, n in zip(g["family"], g["n"])], fontsize=7)
    a0.invert_yaxis()
    a0.set_xlabel("|相関|")
    a0.set_title("A. 分類ごとの到達点(括弧内は特徴量の本数)")
    a0.legend(fontsize=7, frameon=False)

    # --- B. 前向き vs 後ろ向き --------------------------------------------
    # ★配色の規則(MAINTENANCE.txt §4)により、系列を色で 12 本に分けない。
    #   単色にして、目立つ点だけ名前を書く。
    a1 = ax[0, 1]
    a1.scatter(pr[BWD].abs(), pr[TGT].abs(), s=13, alpha=0.7, color=C1,
               linewidths=0)
    lim = max(float(pr[TGT].abs().max()), float(pr[BWD].abs().max())) * 1.08
    a1.plot([0, lim], [0, lim], color=CM, lw=0.9, ls="--")
    a1.set_xlim(0, lim)
    a1.set_ylim(0, lim)
    thr = float(pr[TGT].abs().quantile(0.97))
    for r in pr.filter((pl.col(TGT).abs() >= thr)
                       | (pl.col(BWD).abs() >= lim * 0.5)).iter_rows(named=True):
        a1.annotate(r["feature"], (abs(r[BWD]), abs(r[TGT])), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", color=C2)
    a1.set_xlabel("後ろ向き |r| (同時点・T−10s → T)")
    a1.set_ylabel("前向き |r| (予測・T → T+10s)")
    a1.set_title("B. 対角線より下なら「予測より同時性」")

    # --- C. 上位 15 本 -----------------------------------------------------
    top = pr.with_columns(ab=pl.col(TGT).abs()).sort("ab", descending=True).head(15)
    a2 = ax[0, 2]
    y = np.arange(top.height)
    a2.barh(y, top[TGT].abs(), color=C1, height=0.28, label="前向き (T→T+10s)")
    a2.barh(y + 0.3, top[BWD].abs(), color=CM, height=0.28, label="後ろ向き")
    a2.barh(y + 0.6, top[PLB].abs(), color=C2, height=0.28, label="帰無対照(日の入替)")
    a2.set_yticks(y + 0.3)
    a2.set_yticklabels(top["feature"], fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("|相関|")
    a2.set_title("C. 前向き相関の上位 15 本")
    a2.legend(fontsize=7, frameon=False)

    # --- D. 標本内 vs 標本外 (期間で前半・後半に割る) ---------------------
    a3 = ax[1, 0]
    s = pr.filter(pl.col("is_1st").is_finite() & pl.col("oos_2nd").is_finite())
    a3.scatter(s["is_1st"], s["oos_2nd"], s=11, alpha=0.7, color=C3, linewidths=0)
    lim = max(float(s["is_1st"].abs().max()), float(s["oos_2nd"].abs().max())) * 1.08
    a3.plot([-lim, lim], [-lim, lim], color=CM, lw=0.8, ls="--")
    a3.axhline(0, color=CM, lw=0.6)
    a3.axvline(0, color=CM, lw=0.6)
    rho = float(np.corrcoef(s["is_1st"], s["oos_2nd"])[0, 1])
    a3.set_xlabel("前半の日で測った r")
    a3.set_ylabel("後半の日で測った r")
    a3.set_title(f"D. 期間で割った標本外 (相関 {rho:.3f})")

    # --- E. ホライズンごとの符号 ------------------------------------------
    a4 = ax[1, 1]
    hs = [c for c in ("fwd_micro_1s", "fwd_micro_10s", "fwd_micro_60s") if c in pr.columns]
    # ★系列は 3 本まで。palette_check.validate に C1+C4 を通すと
    #   「#3b6fd4/#7c3aed: CVD ΔE=4.4 < 下限 6.0」で落ちるので C4 は使わない
    sel = []
    seen = set()
    for f in top["feature"].to_list():                 # 同一の量は 1 本だけ
        r = pr.filter(pl.col("feature") == f)
        k = round(float(r[TGT][0]), 3)      # ほぼ同一の量は 1 本にまとめる
        if k in seen:
            continue
        seen.add(k)
        sel.append(f)
        if len(sel) == 3:
            break
    for i, f in enumerate(sel):
        r = pr.filter(pl.col("feature") == f)
        a4.plot(range(len(hs)), [float(r[h][0]) for h in hs], marker="o", ms=3.8,
                lw=1.1, label=f, color=[C1, C2, C3][i])
    a4.axhline(0, color=CM, lw=0.8)
    a4.set_xticks(range(len(hs)))
    a4.set_xticklabels([h.replace("fwd_micro_", "") for h in hs])
    a4.set_xlabel("予測ホライズン")
    a4.set_ylabel("相関 (符号つき)")
    a4.set_title("E. ホライズンを伸ばすと符号は残るか")
    a4.legend(fontsize=7, frameon=False)

    # --- F. 有限率 ---------------------------------------------------------
    a5 = ax[1, 2]
    fr = st.sort("finite_rate")
    a5.plot(np.arange(fr.height), fr["finite_rate"] * 100, color=C1, lw=1.2)
    a5.set_xlabel("特徴量 (有限率の小さい順)")
    a5.set_ylabel("値が有限だった割合 (%)")
    a5.set_ylim(0, 103)
    a5.set_title("F. 欠損の分布")
    low = fr.head(6)
    a5.text(0.98, 0.06, "\n".join(f"{r['feature']} {r['finite_rate']*100:.0f}%"
                                  for r in low.iter_rows(named=True)),
            transform=a5.transAxes, ha="right", va="bottom", fontsize=6, color=C2)
    save(fig, f"{tag}_featlib.png",
         f"{a.coin}: 特徴量ライブラリ {pr.height} 本の棚卸しと予測力"
         f"(目的変数 = microprice の 10 秒先 log リターン)")

    # --- 相関行列 ---------------------------------------------------------
    R = np.load(D / f"featlib_corr_{tag}.npy")
    cols = (D / f"featlib_corr_{tag}.cols").read_text(encoding="utf-8").splitlines()
    fmap = dict(zip(st["feature"], st["family"]))
    order = sorted(range(len(cols)), key=lambda i: (fmap.get(cols[i], "z"), cols[i]))
    Ro = R[np.ix_(order, order)]
    fig2, ax2 = plt.subplots(figsize=(8.6, 7.4))
    im = ax2.imshow(Ro, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    bnd, lab = [], []
    prevf = None
    for k, i in enumerate(order):
        f = fmap.get(cols[i], "z")
        if f != prevf:
            bnd.append(k)
            lab.append(f)
            prevf = f
    for b in bnd[1:]:
        ax2.axhline(b - 0.5, color="k", lw=0.5)
        ax2.axvline(b - 0.5, color="k", lw=0.5)
    ax2.set_xticks(bnd)
    ax2.set_xticklabels(lab, rotation=90, fontsize=6)
    ax2.set_yticks(bnd)
    ax2.set_yticklabels(lab, fontsize=6)
    ax2.grid(False)
    fig2.colorbar(im, ax=ax2, fraction=0.035, label="相関")
    save(fig2, f"{tag}_featlib_corr.png",
         f"{a.coin}: 特徴量どうしの相関({len(cols)} 本・分類順)")


if __name__ == "__main__":
    main()
