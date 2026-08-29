"""build_microprice_dram.py の出力から追加診断とチャートを作る。

追加診断:
  - 滞留時間(BBO が動かない時間)の分布 = 「最高解像度」の実像
  - クロス/ロック状態の件数と滞留時間(板再構成の健全性)
  - マイクロプライス乖離のヒストグラム(イベント加重 / 時間加重)
  - 鮮度(stale)別の予測力 β … イベント時間と実時間で結論が割れる理由の説明
  - UTC 時刻別のスプレッドと更新頻度
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

OUT = Path("C:/Users/ii562/Downloads/Memory")
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)
FULL = OUT / "data" / "microprice" / "**" / "*.parquet"

# dataviz: カテゴリ色は固定順で使い、順位で塗り替えない。1 系列に凡例は付けない
C_PRIMARY = "#3b6fd4"   # マイクロプライス
C_SECOND = "#c2410c"    # 中値・比較対象
C_MUTED = "#9aa3b2"     # 参照線・帯
INK = "#1f2733"
GRID = "#e3e7ee"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    # 日本語ラベルが豆腐にならないよう Windows 同梱フォントを優先する
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})


def fit_stats(df: pl.DataFrame, xcol: str, ycol: str) -> dict:
    s = df.select(
        n=pl.len(), sx=pl.col(xcol).sum(), sy=pl.col(ycol).sum(),
        sxx=(pl.col(xcol) ** 2).sum(), syy=(pl.col(ycol) ** 2).sum(),
        sxy=(pl.col(xcol) * pl.col(ycol)).sum(),
    ).to_dicts()[0]
    n = s["n"]
    if n < 100:
        return {"n": n}
    vx = s["sxx"] / n - (s["sx"] / n) ** 2
    vy = s["syy"] / n - (s["sy"] / n) ** 2
    cxy = s["sxy"] / n - (s["sx"] / n) * (s["sy"] / n)
    return {"n": n, "beta": cxy / vx if vx > 0 else None,
            "corr": cxy / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None}


def main() -> None:
    res = json.loads((OUT / "data" / "analysis.json").read_text(encoding="utf-8"))
    daily = pl.read_csv(OUT / "data" / "daily_summary.csv").with_columns(
        pl.col("dt").str.to_date()
    )
    # ★クロス状態を除く(全体の 0.045% だが |MP−mid| が最大 7,349bp に飛ぶ)
    lf = pl.scan_parquet(str(FULL)).filter(pl.col("best_ask") > pl.col("best_bid"))

    # ---- 滞留時間・クロス/ロック・時刻別プロファイル ----
    q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    dwell = lf.select(
        [pl.col("dwell_ns").quantile(p).alias(f"p{int(p*100)}") for p in q]
        + [pl.col("dwell_ns").mean().alias("mean")]
    ).collect().to_dicts()[0]

    xstate = lf.select(
        crossed_n=(pl.col("best_ask") < pl.col("best_bid")).sum(),
        crossed_ns=pl.when(pl.col("best_ask") < pl.col("best_bid"))
        .then(pl.col("dwell_ns")).otherwise(0).sum(),
        locked_n=(pl.col("best_ask") == pl.col("best_bid")).sum(),
        locked_ns=pl.when(pl.col("best_ask") == pl.col("best_bid"))
        .then(pl.col("dwell_ns")).otherwise(0).sum(),
        total_ns=pl.col("dwell_ns").sum(),
    ).collect().to_dicts()[0]

    hourly = (
        lf.with_columns(((pl.col("ts") // 3_600_000_000_000) % 24).alias("hour"))
        .group_by("hour")
        .agg(
            n=pl.len(),
            spread_bp=(pl.col("spread_bp") * pl.col("dwell_ns")).sum() / pl.col("dwell_ns").sum(),
            dev_bp=(pl.col("micro_dev_bp").abs() * pl.col("dwell_ns")).sum()
            / pl.col("dwell_ns").sum(),
        )
        .sort("hour").collect()
    )

    # ---- 乖離のヒストグラム(0.25bp 刻み、±10bp にクリップ) ----
    hist = (
        lf.select(
            pl.col("micro_dev_bp").clip(-10, 10).mul(4).round(0).truediv(4).alias("bin"),
            pl.col("dwell_ns").cast(pl.Float64),
        )
        .group_by("bin").agg(n=pl.len(), t=pl.col("dwell_ns").sum())
        .sort("bin").collect()
    )

    # ---- 鮮度別の予測力(1 秒グリッド) ----
    g = pl.read_parquet(OUT / "data" / "microprice_1s.parquet").sort(["dt", "ts"])
    g = g.with_columns(
        ((pl.col("mid").shift(-1) - pl.col("mid")) / pl.col("mid") * 1e4).alias("y1"),
        ((pl.col("microprice") - pl.col("mid")) / pl.col("mid") * 1e4).alias("x"),
        (pl.col("ts").shift(-1) - pl.col("ts")).alias("gap"),
    ).filter((pl.col("gap") == 1_000_000_000) & pl.col("y1").is_not_null())
    stale_bands = [(0, 0.1), (0.1, 1), (1, 10), (10, 60), (60, 1e9)]
    stale_fit = []
    for lo, hi in stale_bands:
        sub = g.filter(
            (pl.col("stale_ns") >= lo * 1e9) & (pl.col("stale_ns") < hi * 1e9)
        )
        f = fit_stats(sub, "x", "y1")
        f["band"] = f"{lo}-{hi:g}s" if hi < 1e8 else f"{lo}s+"
        stale_fit.append(f)

    # ---- スプレッド体制別の予測力(全期間プールの回帰は体制混合で壊れる) ----
    spread_bands = [(0, 1), (1, 3), (3, 10), (10, 30), (30, 1e9)]
    spread_fit = []
    for lo, hi in spread_bands:
        sub = g.filter(
            (pl.col("spread_bp") >= lo) & (pl.col("spread_bp") < hi)
            & (pl.col("stale_ns") < 60_000_000_000)
        )
        f = fit_stats(sub, "x", "y1")
        f["band"] = f"{lo}-{hi:g}bp" if hi < 1e8 else f"{lo}bp+"
        f["mean_abs_dev_bp"] = sub["x"].abs().mean() if sub.height else None
        f["mean_abs_move_bp"] = sub["y1"].abs().mean() if sub.height else None
        spread_fit.append(f)

    extra = {"dwell_ns": dwell, "book_state": xstate, "hourly": hourly.to_dicts(),
             "stale_beta_1s": stale_fit, "spread_beta_1s": spread_fit}
    (OUT / "data" / "diagnostics.json").write_text(
        json.dumps(extra, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # ================= チャート =================
    m1 = pl.read_csv(OUT / "data" / "microprice_1m.csv").with_columns(
        pl.from_epoch(pl.col("ts"), time_unit="ns").alias("t")
    )

    # 1) 全期間のマイクロプライス(1 分・最良気配の帯つき)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(m1["t"], m1["best_bid"], m1["best_ask"], color=C_MUTED, alpha=0.45,
                    linewidth=0, label="最良買気配〜最良売気配")
    ax.plot(m1["t"], m1["microprice"], color=C_PRIMARY, linewidth=1.0, label="マイクロプライス")
    ax.set_title("xyz:DRAM マイクロプライス 全期間(1 分グリッド表示・原系列は 23,470,241 点)")
    ax.set_ylabel("USDC")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(CH / "xyz_DRAM_01_microprice_full_period.png")
    plt.close(fig)

    # 2) 上場初日の高解像度(イベント単位)
    d0 = pl.read_parquet(OUT / "data" / "microprice" / "dt=2026-05-04" / "part-000.parquet")
    d0 = d0.with_columns(pl.from_epoch(pl.col("ts"), time_unit="ns").alias("t"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(d0["t"], d0["best_bid"], d0["best_ask"], color=C_MUTED, alpha=0.45,
                    linewidth=0, step="post", label="最良買気配〜最良売気配")
    ax.step(d0["t"], d0["microprice"], where="post", color=C_PRIMARY, linewidth=0.9,
            label="マイクロプライス")
    ax.set_title("上場初日 2026-05-04(イベント単位・4,930 点)— 板が立ち上がる過程")
    ax.set_ylabel("USDC")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(CH / "xyz_DRAM_02_launch_day_event_resolution.png")
    plt.close(fig)

    # 3) 日次スプレッドと更新頻度(2 軸禁止 → 縦積みの小倍数)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(daily["dt"], daily["spread_bp_tw"], color=C_PRIMARY, linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("bp(対数)")
    axes[0].set_title("日次スプレッド(時間加重)— 上場直後 142bp から 0.5bp へ")
    axes[1].plot(daily["dt"], daily["n_ticks"], color=C_SECOND, linewidth=1.6)
    axes[1].set_ylabel("件/日")
    axes[1].set_title("日次 BBO 変化イベント数 = マイクロプライスの更新回数")
    fig.savefig(CH / "xyz_DRAM_03_daily_spread_and_ticks.png")
    plt.close(fig)

    # 4) 乖離の分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, ttl in ((axes[0], "n", "イベント加重"), (axes[1], "t", "時間加重")):
        w = hist[col].to_numpy() / hist[col].sum()
        ax.bar(hist["bin"].to_numpy(), w, width=0.22, color=C_PRIMARY, linewidth=0)
        ax.axvline(0, color=C_MUTED, linewidth=1)
        ax.set_title(f"マイクロプライス − 中値({ttl})")
        ax.set_xlabel("bp")
        ax.set_ylabel("構成比")
    fig.savefig(CH / "xyz_DRAM_04_micro_deviation_distribution.png")
    plt.close(fig)

    # 5) インバランス → 1 秒先の中値変化
    ic = pl.DataFrame(res["imb_curve"])
    x = ((ic["imb_lo"] + ic["imb_hi"]) / 2).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.axhline(0, color=C_MUTED, linewidth=1)
    ax.plot(x, ic["mean_micro_dev_bp"], color=C_MUTED, linewidth=2, marker="o", markersize=4,
            label="マイクロプライスの示す幅")
    ax.plot(x, ic["mean_future_bp"], color=C_PRIMARY, linewidth=2, marker="o", markersize=5,
            label="実際に起きた 1 秒後の中値変化")
    ax.set_xlabel("買い側インバランス I = bid_sz/(bid_sz+ask_sz)")
    ax.set_ylabel("bp")
    ax.set_title("板の偏りは次の値動きを当てる(全 23.5M イベント・単調)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(CH / "xyz_DRAM_05_imbalance_vs_future_move.png")
    plt.close(fig)

    # 6) β の層別(全期間プールの 1 本の回帰では意味が壊れることを示す)
    sf = [f for f in spread_fit if f.get("beta") is not None]
    tf = [f for f in stale_fit if f.get("beta") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar([f["band"] for f in sf], [f["beta"] for f in sf], color=C_PRIMARY, width=0.6)
    axes[0].axhline(1.0, color=C_SECOND, linewidth=1.2, linestyle="--")
    axes[0].text(0.45, 0.90, "β=1 なら較正済み", transform=axes[0].transAxes, color=C_SECOND)
    axes[0].set_title("スプレッド体制別 β(1 秒先・実時間)")
    axes[0].set_ylabel("β")
    axes[1].bar([f["band"] for f in tf], [f["beta"] for f in tf], color=C_MUTED, width=0.6)
    axes[1].set_title("直近更新からの経過時間別 β")
    axes[1].set_ylabel("β")
    fig.savefig(CH / "xyz_DRAM_06_beta_stratified.png")
    plt.close(fig)

    # 7) UTC 時刻別プロファイル
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(hourly["hour"], hourly["n"], color=C_PRIMARY, width=0.7)
    axes[0].set_title("時刻別 BBO 更新数(UTC)")
    axes[0].set_xlabel("hour (UTC)")
    axes[1].bar(hourly["hour"], hourly["spread_bp"], color=C_SECOND, width=0.7)
    axes[1].set_title("時刻別スプレッド(時間加重, bp)")
    axes[1].set_xlabel("hour (UTC)")
    fig.savefig(CH / "xyz_DRAM_07_intraday_profile.png")
    plt.close(fig)

    print(json.dumps(extra, indent=2, ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main()
