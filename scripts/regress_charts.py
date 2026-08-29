"""regression.json から回帰レポート用のチャートを描く。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("C:/Users/ii562/Downloads/Memory")
CH = OUT / "charts"
C_PRIMARY = "#3b6fd4"
C_SECOND = "#c2410c"
C_MUTED = "#9aa3b2"
INK = "#1f2733"
GRID = "#e3e7ee"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})
HS = list(range(1, 21))


def main() -> None:
    r = json.loads((OUT / "data" / "regression.json").read_text(encoding="utf-8"))
    bak_p = OUT / "data" / "regression.json.bak"
    bak = json.loads(bak_p.read_text(encoding="utf-8")) if bak_p.exists() else None

    # ---- 12) 項構造: 日次 β の中央値と四分位、プール OLS ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind, xlab in ((axes[0], "event", "地平(イベント数)"),
                           (axes[1], "clock", "地平(秒)")):
        med = [r["fama_macbeth"][f"{kind}_{h}"]["beta_median"] for h in HS]
        q25 = [r["fama_macbeth"][f"{kind}_{h}"]["beta_q25"] for h in HS]
        q75 = [r["fama_macbeth"][f"{kind}_{h}"]["beta_q75"] for h in HS]
        mean = [r["fama_macbeth"][f"{kind}_{h}"]["beta_mean"] for h in HS]
        pooled = [r["specs"][f"{kind}_{h}"]["beta"] for h in HS]
        ax.fill_between(HS, q25, q75, color=C_PRIMARY, alpha=0.18, linewidth=0,
                        label="日次 β の四分位範囲")
        ax.plot(HS, med, color=C_PRIMARY, linewidth=2, marker="o", markersize=4,
                label="日次 β の中央値")
        ax.plot(HS, mean, color=C_SECOND, linewidth=2, label="Fama-MacBeth 平均")
        ax.plot(HS, pooled, color=C_MUTED, linewidth=2, linestyle="--", label="プール OLS")
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel(xlab)
        ax.set_ylabel("β")
        ax.set_title(f"{'イベント時間' if kind == 'event' else '実時間'}の項構造")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("β < 0 = 中値はマイクロプライスの方向へ動く(m = mid − micro)", y=1.02)
    fig.savefig(CH / "xyz_DRAM_12_beta_term_structure.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 13) レバレッジ集中: クロス行を除く前と後(日次 Σ(x−x̄)² を実データから算出) ----
    import polars as pl
    src_dir = OUT / "data" / "microprice"
    lev_in, lev_ex = [], []
    for p in sorted(src_dir.iterdir()):
        df = pl.read_parquet(p / "part-000.parquet", columns=["micro_dev_bp", "is_crossed"])
        for lev, d in ((lev_in, df), (lev_ex, df.filter(~pl.col("is_crossed")))):
            x = d["micro_dev_bp"].to_numpy()
            lev.append(float(((x - x.mean()) ** 2).sum()) if x.size else 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for lev, lab, col in ((lev_in, "クロス状態を含む(汚染)", C_SECOND),
                          (lev_ex, "クロス状態を除く", C_PRIMARY)):
        s = np.sort(np.array(lev))[::-1]
        s = np.cumsum(s) / s.sum()
        ax.plot(range(1, len(s) + 1), s * 100, linewidth=2, color=col, marker="o",
                markersize=3, label=f"{lab}(上位5日で {s[4] * 100:.0f}%)")
    ax.axhline(80, color=C_MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("寄与の大きい日から数えた日数")
    ax.set_ylabel("Σ(x−x̄)² の累積シェア(%)")
    ax.set_title("プール OLS のレバレッジがどれだけ少数日に集中しているか")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(CH / "xyz_DRAM_13_leverage_concentration.png")
    plt.close(fig)

    # ---- 14) 安定性: 日次 β の推移 ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax, spec, ttl in ((axes[0], "clock_1", "地平 1 秒"),
                          (axes[1], "clock_10", "地平 10 秒")):
        d = r["daily_beta"][spec]
        x = np.arange(len(d))
        b = np.array([q["beta"] for q in d], dtype=float)
        roll = np.array([np.median(b[max(0, i - 19):i + 1]) for i in range(len(b))])
        ax.plot(x, b, color=C_MUTED, linewidth=1, label="日次 β")
        ax.plot(x, roll, color=C_PRIMARY, linewidth=2, label="20 日移動中央値")
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_ylabel("β")
        ax.set_title(f"{ttl} — 板が成熟するほど β は 0 から離れる")
        ax.legend(frameon=False, fontsize=8)
    ticks = [i for i in range(0, len(r["daily_beta"]["clock_1"]), 14)]
    axes[1].set_xticks(ticks, [r["daily_beta"]["clock_1"][i]["dt"][5:] for i in ticks])
    fig.savefig(CH / "xyz_DRAM_14_beta_stability.png")
    plt.close(fig)

    # ---- 15) 経済的有意性 ----
    ec = r["economics"]
    keys = [k for k in ["event_1", "event_5", "event_20", "clock_1", "clock_5", "clock_20"]
            if k in ec]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(range(len(keys)), [ec[k]["mean_pred_move_bp"] for k in keys],
           color=C_PRIMARY, width=0.6, label="予測幅 |β·m| の平均")
    ax.axhline(ec[keys[0]]["mean_half_spread_bp"], color=C_SECOND, linewidth=2,
               linestyle="--", label="ハーフスプレッド(執行コスト)")
    ax.set_xticks(range(len(keys)), keys, rotation=20)
    ax.set_ylabel("bp")
    ax.set_title("シグナルの大きさは執行コストの 2〜5% にしかならない")
    ax.legend(frameon=False)
    fig.savefig(CH / "xyz_DRAM_15_economic_significance.png")
    plt.close(fig)
    print("charts done")


if __name__ == "__main__":
    main()
