"""十分位分析の図(2 銘柄 × 229 特徴量 × 9 ホライズン)。

    uv run python scripts/plot_decile.py

出力: charts/xyz_MU_INTC_decile.png

配色は 2 色(#3b6fd4 = xyz:MU / mid、#c2410c = xyz:INTC / micro)。
系列を分けるのは色ではなく線種にする(palette_check の規則)。

x が確定する時刻 / y の期間: 説明変数は格子点 T まで、目的変数は (T, T+h]。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, D, plt, save  # noqa: E402

HZ = ["100ms", "300ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"]
HS = [0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 60]
COST = 2.83
FIN = pl.col("t").is_finite() & pl.col("spread_bp").is_finite()
KEY = "delta1_bp"          # 代表として見る特徴量(= (スプレッド/2) × OBI1)


def main() -> None:
    R = pl.read_csv(D / "decile_report.csv")
    X = pl.read_csv(D / "decile_xcoin.csv")
    SM = pl.read_csv(D / "decile_sum_xyz_MU.csv").filter(FIN)
    SI = pl.read_csv(D / "decile_sum_xyz_INTC.csv").filter(FIN)
    NM = pl.read_csv(D / "decile_null_xyz_MU.csv").filter(FIN)
    DM = pl.read_parquet(D / "decile_xyz_MU.parquet")
    xs = np.log10(HS)

    fig, ax = plt.subplots(2, 3, figsize=(15.6, 8.0))

    b = ax[0, 0]
    for coin, col in (("xyz:MU", C1), ("xyz:INTC", C2)):
        for px, ls in (("mid", "-"), ("micro", "--")):
            q = R.filter((pl.col("coin") == coin) & (pl.col("px") == px))
            q = q.join(pl.DataFrame({"horizon": HZ, "i": range(9)}), on="horizon").sort("i")
            b.plot(xs, q["n_bonf"] / q["n_feat"] * 100, color=col, ls=ls, lw=1.6,
                   marker="o", ms=3.6, label=f"{coin} {px}")
    b.axhline(0, color=CM, lw=0.9)
    b.set_xticks(xs)
    b.set_xticklabels(HZ, fontsize=7, rotation=40)
    b.set_xlabel("予測ホライズン")
    b.set_ylabel("Bonferroni を通った特徴量の割合(%)")
    b.set_title("A. 300ms〜5 秒が台地、10 秒以降で落ちる。\n     プラセボはどのホライズンでも 0〜0.1%", loc="left")
    b.legend(fontsize=7, frameon=False)

    b = ax[0, 1]
    for nm, T, col, ls in (("実測(xyz:MU)", SM, C1, "-"),
                           ("プラセボ(1 営業日ずらし)", NM, C1, "--"),
                           ("実測(xyz:INTC)", SI, C2, "-")):
        v = np.sort(np.abs(T["spread_bp"].to_numpy()))
        b.plot(v, 1 - np.arange(v.size) / v.size, color=col, ls=ls, lw=1.5, label=nm)
    b.axvline(COST, color=CM, lw=1.2, ls=":")
    b.text(COST * 1.15, 2e-3, f"往復の費用\n{COST}bp", fontsize=7.5, color=CM)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("|D10 − D1|(bp)")
    b.set_ylabel("これ以上である割合")
    b.set_title("B. 大きさの分布 — 費用の線を越える有意なセルは 0", loc="left")
    b.legend(fontsize=7, frameon=False)

    b = ax[1, 0]
    for px, col, ls in (("mid", C1, "-"), ("micro", C2, "-")):
        q = (DM.filter((pl.col("feature") == KEY) & (pl.col("target") == f"fwd_{px}_10s"))
             .sort("decile"))
        b.plot(q["decile"], q["mean_bp"], color=col, ls=ls, lw=1.7, marker="o", ms=4.5,
               label=f"{px} 建て")
    b.axhline(0, color=CM, lw=0.9)
    b.set_xticks(range(1, 11))
    b.set_xlabel(f"{KEY} の十分位(前日の分位で切る)")
    b.set_ylabel("10 秒先のリターンの平均(bp)")
    b.set_title(f"C. {KEY} の十分位カーブ(xyz:MU・10 秒先)— "
                f"mid と micro で符号が逆", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    b = ax[1, 1]
    for coin, S_, col in (("xyz:MU", SM, C1), ("xyz:INTC", SI, C2)):
        for px, ls in (("mid", "-"), ("micro", "--")):
            v = [float(S_.filter((pl.col("feature") == KEY)
                                 & (pl.col("target") == f"fwd_{px}_{h}"))["spread_bp"][0])
                 for h in HZ]
            b.plot(xs, v, color=col, ls=ls, lw=1.6, marker="o", ms=3.6,
                   label=f"{coin} {px}")
    b.axhline(0, color=CM, lw=0.9)
    b.set_xticks(xs)
    b.set_xticklabels(HZ, fontsize=7, rotation=40)
    b.set_xlabel("予測ホライズン")
    b.set_ylabel("D10 − D1(bp)")
    b.set_title(f"D. {KEY} のホライズン曲線 — mid は正、micro は負", loc="left")
    b.legend(fontsize=7, frameon=False)

    b = ax[0, 2]
    j = SM.join(SI, on=["feature", "target"], how="inner", suffix="_i")
    q = j.filter(pl.col("target") == "fwd_mid_10s")
    b.scatter(q["spread_bp"], q["spread_bp_i"], s=20, color=C1, alpha=0.75, linewidths=0,
              label="mid 10 秒")
    q2 = j.filter(pl.col("target") == "fwd_micro_10s")
    b.scatter(q2["spread_bp"], q2["spread_bp_i"], s=20, color=C2, alpha=0.75,
              linewidths=0, label="micro 10 秒")
    lim = float(np.nanmax(np.abs(np.r_[q["spread_bp"].to_numpy(),
                                       q2["spread_bp"].to_numpy(),
                                       q["spread_bp_i"].to_numpy(),
                                       q2["spread_bp_i"].to_numpy()]))) * 1.1
    b.plot([-lim, lim], [-lim, lim], color=CM, lw=0.9, ls="--")
    b.axhline(0, color=CM, lw=0.6)
    b.axvline(0, color=CM, lw=0.6)
    b.set_xlabel("xyz:MU の D10 − D1(bp)")
    b.set_ylabel("xyz:INTC の D10 − D1(bp)")
    xm = X.filter((pl.col("horizon") == "10s") & (pl.col("px") == "mid"))
    b.set_title(f"E. 2 銘柄で同じ向きか(10 秒・順位相関 "
                f"{float(xm['rank_corr'][0]):.2f}・符号一致 "
                f"{float(xm['same_sign'][0])*100:.0f}%)", loc="left")
    b.legend(fontsize=7, frameon=False)

    b = ax[1, 2]
    TOP = pl.read_csv(D / "decile_top_xyz_MU.csv")   # 既に Bonferroni 通過のみ・family つき
    g = (TOP.group_by("family").agg(pl.col("spread_bp").abs().max().alias("mx"),
                                    pl.len().alias("n")).sort("mx", descending=True))
    y = np.arange(g.height)
    b.barh(y, g["mx"], color=C1, height=0.62)
    b.axvline(COST, color=CM, lw=1.2, ls=":")
    b.text(COST * 1.02, g.height - 0.4, f"費用 {COST}bp", fontsize=7.5, color=CM)
    b.set_yticks(y)
    b.set_yticklabels([f"{a}({n})" for a, n in zip(g["family"], g["n"])], fontsize=7.5)
    b.invert_yaxis()
    b.set_xlabel("Bonferroni を通ったセルの |D10 − D1| の最大(bp)")
    b.set_title("F. 分類ごとの到達点(xyz:MU・括弧は通過セル数)", loc="left")
    save(fig, "xyz_MU_INTC_decile.png",
         "xyz:MU と xyz:INTC: 特徴量 229 本 × 9 ホライズン × mid/micro の十分位分析"
         "(十分位の境目は前日の分布・標準誤差は日でクラスタ)")


if __name__ == "__main__":
    main()
