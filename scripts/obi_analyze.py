"""OBI(L=1..10) の記述統計・レベル間相関・予測力を集計し、チャートを作る。

問いは 1 つ: 「板を何段まで見ると次の値動きが読めるのか」。
各 L について、1 秒先の中値変化に対する回帰 β と相関を、
イベント時間・スプレッド体制別に出す(体制混合の罠は microprice_report.md §4 と同じ)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path("C:/Users/ii562/Downloads/Memory")
CH = OUT / "charts"
OBI = OUT / "data" / "obi"
LS = list(range(1, 11))
DEC = 10  # OBI を 10 分位に切って将来リターンを見る

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

SPREAD_BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9)]


def band_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}bp" if hi < 1e8 else f"{lo:g}bp+"


def acc(d: dict, key, vals: dict) -> None:
    t = d.setdefault(key, {})
    for k, v in vals.items():
        t[k] = t.get(k, 0.0) + v


def fit(s: dict) -> dict:
    n = s["n"]
    if n < 1000:
        return {"n": n}
    vx = s["sxx"] / n - (s["sx"] / n) ** 2
    vy = s["syy"] / n - (s["sy"] / n) ** 2
    cxy = s["sxy"] / n - (s["sx"] / n) * (s["sy"] / n)
    return {
        "n": int(n),
        "beta_bp_per_unit": cxy / vx if vx > 0 else None,   # OBI 1 単位あたり bp
        "corr": cxy / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None,
        "mean": s["sx"] / n, "std": vx ** 0.5,
    }


def charts(res: dict, daily: pl.DataFrame) -> None:
    corr_mat = np.array(res["corr_matrix"])
    daily_rows = daily.to_dicts()
    # ---- 1) L 別の予測力(スプレッド体制ごとに最適な段数が違う) ----
    bands = ["0-1bp", "1-3bp", "3-10bp", "10bp+"]
    cols = [C_PRIMARY, C_SECOND, "#0f766e", C_MUTED]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for band, col in zip(bands, cols):
        c = [res["predictive_daily"].get(f"L{L}|{band}", {}).get("median") for L in LS]
        b = [res["predictive"].get(f"L{L}|{band}", {}).get("beta_bp_per_unit") for L in LS]
        axes[0].plot(LS, c, marker="o", markersize=4, linewidth=2, color=col, label=band)
        axes[1].plot(LS, b, marker="o", markersize=4, linewidth=2, color=col, label=band)
        best = int(np.nanargmax([x if x is not None else np.nan for x in c])) + 1
        axes[0].annotate(f"L={best}", (best, c[best - 1]), textcoords="offset points",
                         xytext=(0, 6), ha="center", color=col)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xticks(LS)
    axes[0].set_xlabel("L(合計する板の段数)")
    axes[0].set_ylabel("相関")
    axes[0].set_title("1 秒先の中値変化との相関(日次中央値)— スプレッド体制別")
    axes[0].legend(frameon=False, title="スプレッド")
    axes[1].set_xticks(LS)
    axes[1].set_xlabel("L")
    axes[1].set_ylabel("β(OBI 1 単位あたり bp)")
    axes[1].set_title("回帰係数 β")
    axes[1].legend(frameon=False, title="スプレッド")
    fig.savefig(CH / "xyz_DRAM_08_obi_predictive_by_level.png")
    plt.close(fig)

    # ---- 2) 十分位ごとの将来リターン(L=1,3,10) ----
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for L, col in ((1, C_PRIMARY), (3, C_SECOND), (10, C_MUTED)):
        cur = res["decile_curve"][f"L{L}"]
        x = [(r["obi_lo"] + r["obi_hi"]) / 2 for r in cur]
        ax.plot(x, [r["mean_future_bp"] for r in cur], marker="o", markersize=5,
                linewidth=2, color=col, label=f"OBI(L={L})")
    ax.axhline(0, color=C_MUTED, linewidth=1)
    ax.set_xlabel("OBI(L)")
    ax.set_ylabel("1 秒後の中値変化(平均, bp)")
    ax.set_title("OBI の水準と次の値動き(全期間・毎ティック)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(CH / "xyz_DRAM_09_obi_decile_curve.png")
    plt.close(fig)

    # ---- 3) レベル間相関(逐次ヒートマップ) ----
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr_mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), LS)
    ax.set_yticks(range(10), LS)
    ax.set_xlabel("L")
    ax.set_ylabel("L")
    ax.grid(False)
    for a in range(10):
        for b2 in range(10):
            ax.text(b2, a, f"{corr_mat[a, b2]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if corr_mat[a, b2] > 0.6 else INK)
    ax.set_title("OBI(L) 同士の相関(毎ティック)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(CH / "xyz_DRAM_10_obi_level_correlation.png")
    plt.close(fig)

    # ---- 4) 分布と時間推移 ----
    dd = pl.DataFrame(daily_rows).with_columns(pl.col("dt").str.to_date())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(dd["dt"], dd["obi1_tw"], color=C_PRIMARY, linewidth=1.4, label="L=1")
    axes[0].plot(dd["dt"], dd["obi10_tw"], color=C_SECOND, linewidth=1.4, label="L=10")
    axes[0].axhline(0, color=C_MUTED, linewidth=1)
    axes[0].set_title("日次 OBI(時間加重平均)")
    axes[0].legend(frameon=False)
    axes[1].plot(LS, [res["levels"][str(L)]["std_ev"] for L in LS], marker="o",
                 color=C_PRIMARY, linewidth=2)
    axes[1].set_xticks(LS)
    axes[1].set_xlabel("L")
    axes[1].set_ylabel("標準偏差")
    axes[1].set_title("段数を増やすと OBI は落ち着く")
    fig.savefig(CH / "xyz_DRAM_11_obi_daily_and_dispersion.png")
    plt.close(fig)


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in OBI.iterdir() if p.is_dir())
    stats: dict = {}        # (L, band) -> 十分統計量
    tw: dict = {}           # L -> 時間加重の平均・二乗和
    corr_mat = np.zeros((10, 10))
    corr_n = 0.0
    dec_curve: dict = {}    # (L, decile) -> n, sy
    daily_corr: dict = {}   # (L, band) -> 日次相関のリスト(監査の作法: プールせず日次で読む)
    daily_rows = []

    for i, dt in enumerate(days, 1):
        df = pl.read_parquet(OBI / f"dt={dt}" / "part-000.parquet")
        # ★クロス状態は mid が意味を失う(全体の 0.045%)。統計・回帰からは除く
        df = df.drop_nulls(["mid"]).filter(pl.col("best_ask") > pl.col("best_bid"))
        if df.height < 10:
            continue
        # 1 秒先の中値(その時刻以前の直近値)
        fut = df.select(pl.col("ts").alias("ts_f"), pl.col("mid").alias("mid_f")).sort("ts_f")
        j = (
            df.with_columns((pl.col("ts") + 1_000_000_000).alias("ts_h"))
            .sort("ts_h")
            .join_asof(fut, left_on="ts_h", right_on="ts_f", strategy="backward")
            .drop_nulls(["mid_f"])
            .with_columns(
                ((pl.col("mid_f") - pl.col("mid")) / pl.col("mid") * 1e4).alias("y"),
                ((pl.col("best_ask") - pl.col("best_bid")) / pl.col("mid") * 1e4).alias("spread_bp"),
            )
        )
        w = pl.col("dwell_ns").cast(pl.Float64)
        tot_w = df.select(w.sum()).item()
        for L in LS:
            c = f"obi_{L}"
            acc(tw, L, df.select(
                w_sum=w.sum(),
                wx=(pl.col(c) * w).sum(),
                wxx=(pl.col(c) ** 2 * w).sum(),
                wabs=(pl.col(c).abs() * w).sum(),
                n=pl.len(),
                sx=pl.col(c).sum(),
                sxx=(pl.col(c) ** 2).sum(),
            ).to_dicts()[0])
            for lo, hi in SPREAD_BANDS:
                sub = j.filter((pl.col("spread_bp") >= lo) & (pl.col("spread_bp") < hi))
                if sub.height < 10:
                    continue
                if sub.height >= 500:
                    xv, yv = sub[c].to_numpy(), sub["y"].to_numpy()
                    if xv.std() > 0 and yv.std() > 0:
                        daily_corr.setdefault((L, band_name(lo, hi)), []).append(
                            float(np.corrcoef(xv, yv)[0, 1]))
                acc(stats, (L, band_name(lo, hi)), sub.select(
                    n=pl.len(), sx=pl.col(c).sum(), sy=pl.col("y").sum(),
                    sxx=(pl.col(c) ** 2).sum(), syy=(pl.col("y") ** 2).sum(),
                    sxy=(pl.col(c) * pl.col("y")).sum(),
                ).to_dicts()[0])
            if j.height >= 500:
                xv, yv = j[c].to_numpy(), j["y"].to_numpy()
                if xv.std() > 0 and yv.std() > 0:
                    daily_corr.setdefault((L, "all"), []).append(
                        float(np.corrcoef(xv, yv)[0, 1]))
            acc(stats, (L, "all"), j.select(
                n=pl.len(), sx=pl.col(c).sum(), sy=pl.col("y").sum(),
                sxx=(pl.col(c) ** 2).sum(), syy=(pl.col("y") ** 2).sum(),
                sxy=(pl.col(c) * pl.col("y")).sum(),
            ).to_dicts()[0])
            b = (
                j.with_columns(
                    ((pl.col(c) + 1) / 2 * DEC).floor().clip(0, DEC - 1).alias("d")
                ).group_by("d").agg(n=pl.len(), sy=pl.col("y").sum()).sort("d")
            )
            for r in b.to_dicts():
                acc(dec_curve, (L, int(r["d"])), {"n": r["n"], "sy": r["sy"]})

        m = df.select([f"obi_{L}" for L in LS]).to_numpy()
        corr_mat += np.corrcoef(m, rowvar=False) * df.height
        corr_n += df.height
        daily_rows.append({
            "dt": dt, "n_ticks": df.height,
            "obi1_tw": df.select((pl.col("obi_1") * w).sum() / tot_w).item(),
            "obi10_tw": df.select((pl.col("obi_10") * w).sum() / tot_w).item(),
            "lv_short_bid": df.select((pl.col("n_lv_bid") < 10).mean()).item(),
            "lv_short_ask": df.select((pl.col("n_lv_ask") < 10).mean()).item(),
        })
        print(f"[{i:3d}/{len(days)}] {dt} {df.height:>9,}", flush=True)

    corr_mat /= corr_n
    res = {
        "days": len(days),
        "n_ticks": int(sum(r["n_ticks"] for r in daily_rows)),
        "levels": {
            str(L): {
                "mean_tw": tw[L]["wx"] / tw[L]["w_sum"],
                "std_tw": (tw[L]["wxx"] / tw[L]["w_sum"] - (tw[L]["wx"] / tw[L]["w_sum"]) ** 2) ** 0.5,
                "mean_abs_tw": tw[L]["wabs"] / tw[L]["w_sum"],
                "mean_ev": tw[L]["sx"] / tw[L]["n"],
                "std_ev": (tw[L]["sxx"] / tw[L]["n"] - (tw[L]["sx"] / tw[L]["n"]) ** 2) ** 0.5,
            } for L in LS
        },
        "predictive": {
            f"L{L}|{band}": fit(s) for (L, band), s in sorted(stats.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        },
        "predictive_daily": {
            f"L{L}|{band}": {
                "days": len(v), "median": float(np.median(v)),
                "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
                "n_days_negative": int(sum(1 for x in v if x < 0)),
            } for (L, band), v in sorted(daily_corr.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        },
        "corr_matrix": corr_mat.round(4).tolist(),
        "decile_curve": {
            f"L{L}": [
                {"decile": d, "obi_lo": -1 + 2 * d / DEC, "obi_hi": -1 + 2 * (d + 1) / DEC,
                 "n": int(dec_curve[(L, d)]["n"]),
                 "mean_future_bp": dec_curve[(L, d)]["sy"] / dec_curve[(L, d)]["n"]}
                for d in range(DEC) if (L, d) in dec_curve
            ] for L in LS
        },
    }
    (OUT / "data" / "obi_analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pl.DataFrame(daily_rows).write_csv(OUT / "data" / "obi_daily.csv")

    charts(res, pl.DataFrame(daily_rows))
    print("done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "charts":   # 集計をやり直さず図だけ描く
        charts(json.loads((OUT / "data" / "obi_analysis.json").read_text(encoding="utf-8")),
               pl.read_csv(OUT / "data" / "obi_daily.csv"))
    else:
        main()
