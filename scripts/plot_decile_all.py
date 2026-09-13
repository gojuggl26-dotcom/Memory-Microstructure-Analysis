"""所有 7 銘柄の十分位分析の図。

    uv run python scripts/plot_decile_all.py

出力: charts/allcoins_decile.png

配色: 7 銘柄を 7 色で塗り分けると CVD 分離が確保できない(palette_check の
規則)。**銘柄の識別は色ではなく、行列(ヒートマップ)と細線の束**で行い、
色は「実測 / プラセボ」「正 / 負」の対比にだけ使う。

★費用は銘柄ごとに違う。中央スプレッドは xyz:MU 1.09bp に対し
xyz:KIOXIA 13.9bp なので、全銘柄に 2.83bp を当てるのは誤り。
往復費用 = 2 × (中央スプレッド / 2 + テイカー手数料 0.79bp) を銘柄ごとに使う。

★判定は「片側」で行う。|D10 − D1| は 2 建玉ぶんの差なので、1 往復の費用と
比べるのは甘い。**max(|D1|, |D10|) > 往復費用** が正しい比較。

x が確定する時刻 / y の期間: 説明変数は格子点 T まで、目的変数は (T, T+h]。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, CM, INK, D, plt, save  # noqa: E402

COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
HZ = ["100ms", "300ms", "500ms", "1s", "3s", "5s", "10s", "30s", "60s"]
FIN = pl.col("t").is_finite() & pl.col("spread_bp").is_finite()
KEY_F, KEY_T = "ofi_ewma", "fwd_mid_60s"   # mid 建てで最も大きかった規則


def grid(ax, M, rows, cols, fmt, title, cmap="Blues", vmin=None, vmax=None):
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    lo = np.nanmin(M) if vmin is None else vmin
    hi = np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            c = "white" if (M[i, j] - lo) / max(hi - lo, 1e-9) > 0.6 else INK
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=6.4, color=c)
    ax.set_title(title, loc="left")
    ax.grid(False)


def main() -> None:
    R = pl.read_csv(D / "decile_all_report.csv")
    X = pl.read_csv(D / "decile_all_xcoin.csv")
    CO = pl.read_csv(D / "decile_all_cost.csv")
    cost = {r["coin"].split(":")[1]: r["cost_bp"] for r in CO.iter_rows(named=True)}
    S = {c: pl.read_csv(D / f"decile_sum_xyz_{c}_bbo.csv").filter(FIN) for c in COINS}
    N = {c: pl.read_csv(D / f"decile_null_xyz_{c}_bbo.csv").filter(FIN) for c in COINS}
    thr = {c: float(R.filter(pl.col("coin") == f"xyz:{c}")["thr"][0]) for c in COINS}

    fig, ax = plt.subplots(2, 3, figsize=(16.4, 8.8))

    # A. 銘柄 × ホライズン の通過率
    M = np.array([[float(R.filter((pl.col("coin") == f"xyz:{c}")
                                  & (pl.col("px") == "mid")
                                  & (pl.col("horizon") == h))
                         .select(pl.col("n_bonf") / pl.col("n_feat") * 100).item())
                   for h in HZ] for c in COINS])
    grid(ax[0, 0], M, [f"xyz:{c}" for c in COINS], HZ, "{:.0f}",
         "A. Bonferroni を通った割合(%、mid 建て)\n"
         "     厚い銘柄は 100ms から効き、薄い銘柄は 3〜30 秒に寄る")

    # B. 有意なセルの「片側の大きさ」と、その銘柄の往復費用
    b = ax[1, 2]
    for i, c in enumerate(COINS):
        one = (S[c].filter(pl.col("t").abs() > thr[c])
               .with_columns(o=pl.max_horizontal(pl.col("d1").abs(),
                                                 pl.col("d10").abs()))["o"].to_numpy())
        q = np.nanpercentile(one, [50, 90, 100])
        b.plot([q[0], q[2]], [i, i], color=C1, lw=2.2, solid_capstyle="butt",
               label="有意なセルの片側の大きさ(中央値〜最大)" if i == 0 else None)
        b.plot(q[2], i, "o", color=C1, ms=5)
        b.plot(cost[c], i, "D", color=C2, ms=6.5,
               label="その銘柄の往復費用" if i == 0 else None)
        b.text(cost[c] * 1.06, i + 0.22, f"{cost[c]:.1f}", fontsize=6.6, color=C2)
    b.set_yticks(range(len(COINS)))
    b.set_yticklabels([f"xyz:{c}" for c in COINS], fontsize=7.5)
    b.set_xscale("log")
    b.set_xlabel("bp(対数)")
    b.set_ylim(-0.7, len(COINS) - 1 + 1.5)
    b.set_title("F. 片側で取れる大きさは、どの銘柄でも費用に届かない\n"
                "     ◆ が費用。棒の右端(最大)がその左にある", loc="left")
    b.legend(fontsize=7, frameon=False, loc="upper left")

    # C. 銘柄ペアの順位相関
    P = np.full((len(COINS), len(COINS)), np.nan)
    for r in X.filter((pl.col("px") == "mid")
                      & (pl.col("horizon") == "ALL")).iter_rows(named=True):
        i, j = COINS.index(r["a"]), COINS.index(r["b"])
        P[i, j] = P[j, i] = r["rank_corr"]
    np.fill_diagonal(P, 1.0)
    grid(ax[0, 2], P, COINS, COINS, "{:.2f}",
         "C. 効果の並び方の銘柄間一致(順位相関)\n"
         "     0.78〜0.98 — 同じ特徴量が同じ向きに効く", vmin=0.7, vmax=1.0)

    # B(左上の右). |D10 − D1| の分布 実測 vs プラセボ
    b = ax[0, 1]
    for k, c in enumerate(COINS):
        for T, col, lw in ((S[c], C1, 1.1), (N[c], C2, 0.9)):
            v = np.sort(np.abs(T["spread_bp"].to_numpy()))
            b.plot(v, 1 - np.arange(v.size) / v.size, color=col, lw=lw, alpha=0.75,
                   label=("実測(7 銘柄)" if col == C1 else "プラセボ(7 銘柄)")
                   if k == 0 else None)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("|D10 − D1|(bp)")
    b.set_ylabel("これ以上である割合")
    b.set_title("B. 効果の大きさ — 実測はプラセボの 10〜30 倍。\n"
                "     配管が偽物を作っているのではない", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    # D. micro 建ての delta1 は機械的な平均回帰であること
    b = ax[1, 0]
    xs, ys, lb = [], [], []
    for c in COINS:
        q = S[c].filter(pl.col("feature") == "delta1_bp")
        m = float(q.filter(pl.col("target") == "fwd_mid_60s")["spread_bp"][0])
        mi = float(q.filter(pl.col("target") == "fwd_micro_60s")["spread_bp"][0])
        f = (pl.scan_parquet(str(D / f"featbbo_xyz_{c}" / "dt=*.parquet"))
             .select("delta1_bp").collect()["delta1_bp"].to_numpy())
        f = f[np.isfinite(f)]
        xs.append(float(np.nanquantile(f, 0.95) - np.nanquantile(f, 0.05)))
        ys.append(m - mi)
        lb.append(c)
    b.plot([0, max(xs) * 1.05], [0, max(xs) * 1.05], color=CM, lw=1.2, ls=":",
           label="y = x(完全に機械的)")
    b.scatter(xs, ys, s=44, color=C1, zorder=3)
    for x_, y_, l_ in zip(xs, ys, lb):
        b.annotate(l_, (x_, y_), fontsize=7,
                   xytext=(6, 2 if l_ in ("SNDK", "AMD") else -11),
                   textcoords="offset points", color=INK)
    b.set_xlabel("delta1_bp 自身の広がり(5〜95% 点、bp)")
    b.set_ylabel("mid 建ての効果 − micro 建ての効果(bp)")
    b.set_title("D. micro 建ての「強い予測」は microprice の定義から出る\n"
                "     両者がほぼ一致 = 機械的な平均回帰。執行はできない", loc="left")
    b.legend(fontsize=7.5, frameon=False)

    # E. mid 建てで最大だった規則の十分位曲線
    b = ax[1, 1]
    cur = []
    for c in COINS:
        q = (pl.read_parquet(D / f"decile_xyz_{c}_bbo.parquet")
             .filter((pl.col("feature") == KEY_F) & (pl.col("target") == KEY_T))
             .sort("decile"))
        if q.height == 10:
            v = q["mean_bp"].to_numpy()
            cur.append(v)
            b.plot(range(1, 11), v, color=C1, lw=1.0, alpha=0.6)
    if cur:
        b.plot(range(1, 11), np.mean(cur, axis=0), color=C2, lw=2.4,
               marker="o", ms=4, label="7 銘柄の平均")
    b.axhline(0, color=CM, lw=0.9)
    b.axhspan(-min(cost.values()), min(cost.values()), color=CM, alpha=0.14,
              label=f"いちばん安い銘柄でも越えられない帯(±{min(cost.values()):.1f}bp)")
    b.set_xticks(range(1, 11))
    b.set_xlabel("前日の分位で切った十分位")
    b.set_ylabel("60 秒先の mid リターンの平均(bp)")
    b.set_title(f"E. {KEY_F} → 60 秒先の mid(mid 建てで最大の規則)\n"
                "     単調は 7 銘柄で揃う。平均は帯の中。はみ出す線は\n"
                "     xyz:KIOXIA で、この銘柄の費用は 15.5bp と最も高い", loc="left")
    b.legend(fontsize=7, frameon=False)

    save(fig, "allcoins_decile.png",
         "所有 7 銘柄 × 84 特徴量 × 9 ホライズン × mid/micro の十分位分析")


if __name__ == "__main__":
    main()
