"""book_slope を核にした予測モデル — 段階 1〜6(データ構築)。

  L4 → book_slope 算出 → データ洗浄 → 特徴量設計 → 未来リターン → 学習/検証/OOS 分割

【★時間契約(CLAUDE.md 厳禁事項)】
  すべての説明変数は時刻 T までに確定した情報のみで作る。
    ・窓集計は [T−Δ, T) の**後ろ向き**のみ。rolling は shift(1) を必ず挟む
    ・asof 結合は backward のみ
    ・目的変数は y = (log mid(T+h) − log mid(T)) × 10⁴ … T **以降**の区間
    ・標準化・分位・winsorize の統計量は**学習期間だけ**から推定する
  したがって「T の特徴量」と「T→T+h のリターン」は時間的に重ならない。

【book_slope の定義】
  S^side = Σ_i q_i·|p_i − best| / Σ_i q_i     (最良気配から 25bp 以内、i は注文)
  = 「板の重心が最良気配からどれだけ離れているか」。大きいほど手前が薄い。
  深さ帯 25bp は book_slope_grain_report.md の粒度スイープで最良だった帯。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
SL25 = D / "slope25"
PANEL = D / "panel_v2"
OUT = D / "slope_model"
H_SEC = 1                      # 予測地平 1 秒(粒度検証で最適)
BURN_IN_DAYS = 7               # 立ち上げ期は関係が 5〜7 倍動くため除外
TRAIN_FRAC, VAL_FRAC = 0.50, 0.20

# 統制変数(book_slope 以外。既に効果が確認済みのもの)
CTRL = ["m_bp", "spread_bp", "obi_5", "log_depth", "r_prev", "voi_sum", "ofi_sum",
        "oi", "n_trades", "cancel_diff_vol", "new_intensity_diff", "hhi_diff",
        "top1_diff", "stale_ns"]


def build_day(dt: str) -> pl.DataFrame | None:
    """1 日分: slope25 と panel_v2 を 1 秒グリッドで突き合わせ、slope 系特徴量を作る。"""
    f1, f2 = SL25 / f"{dt}.parquet", PANEL / f"dt={dt}"
    if not f1.exists() or not f2.exists():
        return None
    s = pl.read_parquet(f1).sort("ts")
    p = pl.read_parquet(f2 / "part-000.parquet").sort("ts")

    # --- 段階 2: book_slope 本体 ---------------------------------------
    s = s.with_columns(
        (pl.col("s_bid") - pl.col("s_ask")).alias("slope_diff"),
        (pl.col("s_bid") + pl.col("s_ask")).alias("slope_sum"),
        (pl.col("q_bid") + pl.col("q_ask")).alias("q_tot"),
    )
    # --- 段階 4: 特徴量設計(すべて後ろ向き) --------------------------
    # shift(1) を挟むのは「T の行に T の値を含む窓」を作らないため。
    # rolling_mean(k) は [T−k+1, T] を見るので、shift(1) で [T−k, T−1] にする。
    s = s.with_columns(
        # 水準の非対称
        (pl.col("slope_diff") / pl.col("slope_sum")).alias("slope_ratio"),
        # 変化(直前 1 秒との差)= 板の重心がどちらへ動いたか
        pl.col("slope_diff").diff().alias("d_slope_diff"),
        pl.col("s_bid").diff().alias("d_s_bid"),
        pl.col("s_ask").diff().alias("d_s_ask"),
        # 後ろ向き移動平均(平滑化した水準)
        pl.col("slope_diff").shift(1).rolling_mean(5).alias("slope_diff_ma5"),
        pl.col("slope_diff").shift(1).rolling_mean(30).alias("slope_diff_ma30"),
        # 後ろ向きの散らばり(体制の指標)
        pl.col("slope_diff").shift(1).rolling_std(30).alias("slope_diff_sd30"),
        # 数量で重み付けした非対称(薄い板での slope は信用しない)
        (pl.col("q_bid") - pl.col("q_ask")).alias("q_diff"),
    ).with_columns(
        # 平均からの乖離(体制を除いた「今の異常さ」)。分母は後ろ向き標準偏差
        ((pl.col("slope_diff") - pl.col("slope_diff_ma30"))
         / (pl.col("slope_diff_sd30") + 1e-9)).alias("slope_diff_dev"),
        (pl.col("slope_diff") * pl.col("q_tot").log1p()).alias("slope_x_qty"),
    )
    j = p.join(s, on="ts", how="inner")
    if j.height == 0:
        return None
    j = j.with_columns(
        (pl.col("slope_diff") * pl.col("spread_bp")).alias("slope_x_spread"),
        (pl.col("slope_diff") * pl.col("log_depth")).alias("slope_x_depth"),
        pl.lit(dt).alias("dt"),
    )
    return j


SLOPE_FEATS = ["slope_diff", "s_bid", "s_ask", "slope_sum", "slope_ratio",
               "d_slope_diff", "d_s_bid", "d_s_ask", "slope_diff_ma5",
               "slope_diff_ma30", "slope_diff_sd30", "slope_diff_dev",
               "q_diff", "slope_x_qty", "slope_x_spread", "slope_x_depth"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(p.stem for p in SL25.glob("*.parquet"))
    log = {"stage": [], "days_total": len(days)}
    frames = []
    for dt in days:
        d = build_day(dt)
        if d is not None:
            frames.append(d.select(["dt", "ts", "y", *SLOPE_FEATS, *CTRL]))
    df = pl.concat(frames)
    del frames
    n0 = df.height
    log["stage"].append({"名称": "結合直後(クロス・resync 除去済みの panel_v2 基準)",
                         "行数": n0, "日数": df["dt"].n_unique()})

    # --- 段階 3: データ洗浄 --------------------------------------------
    days_kept = sorted(df["dt"].unique().to_list())
    burn = days_kept[:BURN_IN_DAYS]
    df = df.filter(~pl.col("dt").is_in(burn))
    log["stage"].append({"名称": f"立ち上げ {BURN_IN_DAYS} 日を除外", "行数": df.height,
                         "日数": df["dt"].n_unique(), "除外日": burn})

    df = df.filter(pl.all_horizontal([pl.col(c).is_finite()
                                      for c in ["y", *SLOPE_FEATS, *CTRL]]))
    log["stage"].append({"名称": "非有限値(NaN/Inf)を除去", "行数": df.height})

    # 板が 5 秒以上更新されていない行 = 実質的に情報が無い
    df = df.filter(pl.col("stale_ns") < 5e9)
    log["stage"].append({"名称": "板が 5 秒以上停止した行を除去", "行数": df.height})

    # --- 段階 5〜6: y は構築済み(T→T+1s)。日で時系列分割 -------------
    dl = sorted(df["dt"].unique().to_list())
    n = len(dl)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    split = {d: ("train" if k < i1 else "val" if k < i2 else "oos") for k, d in enumerate(dl)}
    df = df.with_columns(pl.col("dt").replace_strict(split).alias("split"))

    # winsorize は**学習期間の分位だけ**から作る(段階 3 の続き・順序が重要)
    tr = df.filter(pl.col("split") == "train")
    ql, qh = tr["y"].quantile(0.0005), tr["y"].quantile(0.9995)
    df = df.with_columns(pl.col("y").clip(ql, qh).alias("y"))
    log["winsor"] = {"lower_bp": float(ql), "upper_bp": float(qh),
                     "source": "学習期間のみの 0.05% / 99.95% 分位"}

    # 因果的な日次ボラ(**前日**の実現ボラ)。MSE を分散の大きい日に支配させないため
    dv = (df.group_by("dt").agg(pl.col("y").std().alias("vol")).sort("dt")
            .with_columns(pl.col("vol").shift(1).alias("vol_prev")))
    dv = dv.with_columns(pl.col("vol_prev").fill_null(pl.col("vol_prev").drop_nulls().first()))
    df = df.join(dv.select(["dt", "vol_prev"]), on="dt", how="left")
    log["stage"].append({"名称": "前日ボラで正規化する列 vol_prev を付与(因果)",
                         "行数": df.height})

    df.write_parquet(OUT / "dataset.parquet", compression="zstd")
    cnt = df.group_by("split").agg(n=pl.len(), days=pl.col("dt").n_unique(),
                                   d0=pl.col("dt").min(), d1=pl.col("dt").max())
    log["splits"] = cnt.sort("d0").to_dicts()
    log["features"] = {"slope": SLOPE_FEATS, "control": CTRL}
    log["horizon_sec"] = H_SEC
    (D / "slope_model_data.json").write_text(json.dumps(log, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print("=== 段階 3: 洗浄の記録 ===")
    prev = n0
    for s in log["stage"]:
        print(f"  {s['名称']:<44} {s['行数']:>10,}  ({s['行数']-prev:+,})")
        prev = s["行数"]
    print(f"\nwinsorize 閾値(学習期間のみ): [{ql:.3f}, {qh:.3f}] bp")
    print("\n=== 段階 6: 時系列分割 ===")
    for r in log["splits"]:
        print(f"  {r['split']:>5}  {r['d0']} 〜 {r['d1']}  {r['days']:>2} 日  {r['n']:>9,} 行")
    print(f"\nslope 系特徴量 {len(SLOPE_FEATS)} 個 / 統制 {len(CTRL)} 個")


if __name__ == "__main__":
    main()
