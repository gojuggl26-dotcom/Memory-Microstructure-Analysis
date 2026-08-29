"""xyz:DRAM の全期間マイクロプライスを最高解像度(BBO 変化イベント単位)で算出する。

入力: L2 bbo テーブル(ts, best_bid, best_ask, bid_sz, ask_sz)= 板の最良気配が
      変化した瞬間だけを ns 精度で記録したもの。これがこのデータで到達できる
      最高解像度であり、これ以上細かい板状態は原理的に観測できない。

出力: OUT_DIR/
  data/microprice/dt=YYYY-MM-DD/part-000.parquet   全解像度シリーズ
  data/microprice_1s.parquet                        1 秒グリッド(直近値・stale 情報つき)
  data/microprice_1m.csv                            1 分グリッド(表計算で開ける版)
  data/daily_summary.csv                            日次サマリ
  data/analysis.json                                レポート生成用の集計結果

マイクロプライス(重み付き中値)の定義:
    I  = bid_sz / (bid_sz + ask_sz)                 … 買い側インバランス
    MP = I * best_ask + (1 - I) * best_bid
       = (best_bid * ask_sz + best_ask * bid_sz) / (bid_sz + ask_sz)
厚みの薄い側へ寄る。板が偏っているとき次の約定が起きやすい側の価格を強く見る。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

SRC = Path("data/l2_v99/bbo")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/ii562/Downloads/Memory")
NS_DAY = 86_400_000_000_000

# 予測力の検証に使う先読み地平(ns)
HORIZONS = {"100ms": 100_000_000, "1s": 1_000_000_000, "10s": 10_000_000_000}
GRID_STEPS = {"1s": 1, "10s": 10, "60s": 60}  # 1 秒グリッド上での先読み(秒数)
N_IMB_BINS = 20


def fit(s: dict) -> dict:
    """十分統計量から回帰係数と予測誤差を出す。"""
    n, sx, sy, sxx, syy, sxy = s["n"], s["sx"], s["sy"], s["sxx"], s["syy"], s["sxy"]
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    cxy = sxy / n - (sx / n) * (sy / n)
    beta = cxy / vx if vx > 0 else float("nan")
    return {
        "n": n,
        "beta": beta,
        "corr": cxy / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan"),
        "r2": (cxy ** 2) / (vx * vy) if vx > 0 and vy > 0 else float("nan"),
        "std_x_bp": vx ** 0.5,
        "std_y_bp": vy ** 0.5,
        "rmse_mid_bp": (s["sse_mid"] / n) ** 0.5,
        "rmse_micro_bp": (s["sse_mp"] / n) ** 0.5,
        "rmse_micro_calibrated_bp": ((syy - 2 * beta * sxy + beta * beta * sxx) / n) ** 0.5,
    }


def day_bounds(dt: str) -> tuple[int, int]:
    d = datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    t0 = int(d.timestamp()) * 1_000_000_000
    return t0, t0 + NS_DAY


def build_day(path: Path, dt: str) -> tuple[pl.DataFrame, dict]:
    """1 日分の BBO を読み、マイクロプライスと滞留時間を付けて返す。"""
    t0, t1 = day_bounds(dt)
    df = (
        pl.read_parquet(path)
        .sort("ts")
        .unique(subset=["ts"], keep="last", maintain_order=True)  # 同 ns は最終状態のみ
        .with_columns(
            (pl.col("bid_sz") + pl.col("ask_sz")).alias("tot_sz"),
        )
        .with_columns(
            (pl.col("bid_sz") / pl.col("tot_sz")).alias("imbalance"),
        )
        .with_columns(
            (
                (pl.col("best_bid") * pl.col("ask_sz") + pl.col("best_ask") * pl.col("bid_sz"))
                / pl.col("tot_sz")
            ).alias("microprice"),
        )
        .with_columns(
            ((pl.col("microprice") - pl.col("mid")) / pl.col("mid") * 1e4).alias("micro_dev_bp"),
            # 次の BBO 変化までの滞留時間。時間加重統計と 1s グリッドの stale 判定に使う
            (pl.col("ts").shift(-1).fill_null(t1) - pl.col("ts")).alias("dwell_ns"),
            # ★クロス状態。件数は全体の 0.045% だが mid が意味を失い |MP−mid| が最大
            #   7,349bp に飛ぶため、統計・グリッド・回帰からは必ず除く
            (pl.col("best_ask") <= pl.col("best_bid")).alias("is_crossed"),
        )
        .drop("tot_sz")
    )
    ok = df.filter(~pl.col("is_crossed"))

    # --- 日次統計(時間加重・クロス除外) ---
    w = pl.col("dwell_ns").cast(pl.Float64)
    stat = ok.select(
        n_ticks=pl.len(),
        ts_min=pl.col("ts").min(),
        ts_max=pl.col("ts").max(),
        covered_ns=pl.col("dwell_ns").sum(),
        px_first=pl.col("microprice").first(),
        px_last=pl.col("microprice").last(),
        px_min=pl.col("microprice").min(),
        px_max=pl.col("microprice").max(),
        mp_tw=(pl.col("microprice") * w).sum() / w.sum(),
        mid_tw=(pl.col("mid") * w).sum() / w.sum(),
        spread_bp_tw=(pl.col("spread_bp") * w).sum() / w.sum(),
        spread_bp_med=pl.col("spread_bp").median(),
        dev_bp_tw_abs=(pl.col("micro_dev_bp").abs() * w).sum() / w.sum(),
        dev_bp_ev_abs=pl.col("micro_dev_bp").abs().mean(),
        dev_bp_p99=pl.col("micro_dev_bp").abs().quantile(0.99),
        imb_tw=(pl.col("imbalance") * w).sum() / w.sum(),
        bid_sz_tw=(pl.col("bid_sz") * w).sum() / w.sum(),
        ask_sz_tw=(pl.col("ask_sz") * w).sum() / w.sum(),
        dwell_med=pl.col("dwell_ns").median(),
        dwell_p99=pl.col("dwell_ns").quantile(0.99),
        gap_ns=pl.when(pl.col("dwell_ns") > 60_000_000_000)
        .then(pl.col("dwell_ns"))
        .otherwise(0)
        .sum(),
    ).to_dicts()[0]
    stat.update(df.select(
        n_all=pl.len(),
        n_crossed=(pl.col("best_ask") < pl.col("best_bid")).sum(),
        n_locked=(pl.col("best_ask") == pl.col("best_bid")).sum(),
        crossed_ns=pl.when(pl.col("best_ask") < pl.col("best_bid"))
        .then(pl.col("dwell_ns")).otherwise(0).sum(),
    ).to_dicts()[0])
    stat["dt"] = dt
    # 板が片側だけ/空で BBO 行が出ない時間(= 観測不能時間)
    stat["uncovered_ns"] = (t1 - t0) - int(stat["covered_ns"] or 0) - max(0, int(stat["ts_min"]) - t0)
    return df, stat


def predictive(df: pl.DataFrame) -> dict:
    """先の中値をマイクロプライスがどれだけ当てるかの十分統計量を貯める。

    x  = (MP_t - mid_t)/mid_t*1e4      … マイクロプライスの示す方向(bp)
    y  = (mid_{t+h} - mid_t)/mid_t*1e4 … 実際に起きた中値変化(bp)
    回帰 y = a + b x の b が 1 に近ければマイクロプライスは較正済み。
    """
    out: dict = {}
    base = df.filter(~pl.col("is_crossed")).select("ts", "mid", "micro_dev_bp", "imbalance")
    for name, h in HORIZONS.items():
        fut = (
            base.select(pl.col("ts").alias("ts_f"), pl.col("mid").alias("mid_f"))
            .sort("ts_f")
        )
        j = (
            base.with_columns((pl.col("ts") + h).alias("ts_h"))
            .sort("ts_h")
            .join_asof(fut, left_on="ts_h", right_on="ts_f", strategy="backward")
            .drop_nulls(["mid_f"])
            .with_columns(((pl.col("mid_f") - pl.col("mid")) / pl.col("mid") * 1e4).alias("y"))
            .rename({"micro_dev_bp": "x"})
        )
        if j.height == 0:
            continue
        s = j.select(
            n=pl.len(),
            sx=pl.col("x").sum(),
            sy=pl.col("y").sum(),
            sxx=(pl.col("x") ** 2).sum(),
            syy=(pl.col("y") ** 2).sum(),
            sxy=(pl.col("x") * pl.col("y")).sum(),
            # 予測誤差: 中値をそのまま使う場合 vs マイクロプライスを使う場合
            sse_mid=(pl.col("y") ** 2).sum(),
            sse_mp=((pl.col("y") - pl.col("x")) ** 2).sum(),
        ).to_dicts()[0]
        out[name] = s
        if name == "1s":
            b = (
                j.with_columns(
                    pl.col("imbalance").mul(N_IMB_BINS).floor().clip(0, N_IMB_BINS - 1).alias("ib")
                )
                .group_by("ib")
                .agg(n=pl.len(), sy=pl.col("y").sum(), sx=pl.col("x").sum())
                .sort("ib")
            )
            out["imb_curve"] = b.to_dicts()
    return out


def merge_pred(acc: dict, cur: dict) -> None:
    for k, v in cur.items():
        if k == "imb_curve":
            tgt = acc.setdefault("imb_curve", {})
            for r in v:
                t = tgt.setdefault(int(r["ib"]), {"n": 0, "sy": 0.0, "sx": 0.0})
                t["n"] += r["n"]
                t["sy"] += r["sy"]
                t["sx"] += r["sx"]
        else:
            tgt = acc.setdefault(k, {})
            for kk, vv in v.items():
                tgt[kk] = tgt.get(kk, 0) + vv


def predictive_grid(g: pl.DataFrame, steps: dict[str, int]) -> dict:
    """1 秒グリッド上でも同じ回帰を行う(イベント時間の偏りに対する頑健性チェック)。

    イベント単位の集計は活発な時間帯を過大に重み付けする。等間隔の実時間で見ても
    同じ結論になるかを確かめる。stale(板が止まっている)行は除く。
    """
    out: dict = {}
    base = g.filter(pl.col("stale_ns") < 60_000_000_000).sort("ts")
    for name, k in steps.items():
        j = base.with_columns(
            ((pl.col("mid").shift(-k) - pl.col("mid")) / pl.col("mid") * 1e4).alias("y"),
            ((pl.col("microprice") - pl.col("mid")) / pl.col("mid") * 1e4).alias("x"),
            (pl.col("ts").shift(-k) - pl.col("ts")).alias("gap"),
        ).filter(pl.col("gap") == k * 1_000_000_000).drop_nulls(["y"])
        if j.height == 0:
            continue
        out[name] = j.select(
            n=pl.len(), sx=pl.col("x").sum(), sy=pl.col("y").sum(),
            sxx=(pl.col("x") ** 2).sum(), syy=(pl.col("y") ** 2).sum(),
            sxy=(pl.col("x") * pl.col("y")).sum(),
            sse_mid=(pl.col("y") ** 2).sum(),
            sse_mp=((pl.col("y") - pl.col("x")) ** 2).sum(),
        ).to_dicts()[0]
    return out


def grid(df: pl.DataFrame, dt: str, step_ns: int) -> pl.DataFrame:
    """秒/分グリッドに直近の板状態を貼り付ける(前方補完)。stale_ns で鮮度が分かる。"""
    t0, t1 = day_bounds(dt)
    g = pl.DataFrame({"ts": list(range(t0, t1, step_ns))}, schema={"ts": pl.Int64})
    j = g.join_asof(
        df.filter(~pl.col("is_crossed")).select("ts", "best_bid", "best_ask", "bid_sz", "ask_sz", "mid", "microprice",
                  "imbalance", "spread_bp").rename({"ts": "ts_src"}),
        left_on="ts", right_on="ts_src", strategy="backward",
    )
    return j.with_columns(
        (pl.col("ts") - pl.col("ts_src")).alias("stale_ns"),
        pl.lit(dt).alias("dt"),
    ).drop("ts_src").drop_nulls(["microprice"])


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in SRC.iterdir() if p.is_dir())
    out_full = OUT_DIR / "data" / "microprice"
    out_full.mkdir(parents=True, exist_ok=True)

    stats: list[dict] = []
    pred: dict = {}
    gpred: dict = {}
    g1s: list[pl.DataFrame] = []
    g1m: list[pl.DataFrame] = []

    for i, dt in enumerate(days, 1):
        src = SRC / f"dt={dt}" / "part-000.parquet"
        df, st = build_day(src, dt)
        d = out_full / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / "part-000.parquet", compression="zstd", statistics=True)
        stats.append(st)
        merge_pred(pred, predictive(df))
        gs = grid(df, dt, 1_000_000_000)
        merge_pred(gpred, predictive_grid(gs, GRID_STEPS))
        g1s.append(gs)
        g1m.append(grid(df, dt, 60_000_000_000))
        print(f"[{i:3d}/{len(days)}] {dt} ticks={st['n_ticks']:>8,} "
              f"mp_tw={st['mp_tw']:.6f} spread={st['spread_bp_tw']:.1f}bp", flush=True)

    sdf = pl.DataFrame(stats)
    sdf.write_csv(OUT_DIR / "data" / "daily_summary.csv")

    s1s = pl.concat(g1s)
    s1s.write_parquet(OUT_DIR / "data" / "microprice_1s.parquet", compression="zstd")
    pl.concat(g1m).write_csv(OUT_DIR / "data" / "microprice_1m.csv")

    # --- 全期間の集計 ---
    res: dict = {"days": len(days), "dt_min": days[0], "dt_max": days[-1]}
    res["n_ticks"] = int(sdf["n_ticks"].sum())
    res["event_time"] = {name: fit(pred[name]) for name in HORIZONS if pred.get(name)}
    res["clock_time_1s_grid"] = {name: fit(gpred[name]) for name in GRID_STEPS if gpred.get(name)}
    res["imb_curve"] = [
        {"bin": k, "imb_lo": k / N_IMB_BINS, "imb_hi": (k + 1) / N_IMB_BINS,
         "n": v["n"], "mean_future_bp": v["sy"] / v["n"], "mean_micro_dev_bp": v["sx"] / v["n"]}
        for k, v in sorted(pred.get("imb_curve", {}).items())
    ]
    w = pl.col("covered_ns").cast(pl.Float64)
    agg = sdf.select(
        spread_bp_tw=(pl.col("spread_bp_tw") * w).sum() / w.sum(),
        dev_bp_tw_abs=(pl.col("dev_bp_tw_abs") * w).sum() / w.sum(),
        dev_bp_ev_abs=(pl.col("dev_bp_ev_abs") * pl.col("n_ticks")).sum() / pl.col("n_ticks").sum(),
        imb_tw=(pl.col("imb_tw") * w).sum() / w.sum(),
        covered_ns=pl.col("covered_ns").sum(),
        uncovered_ns=pl.col("uncovered_ns").sum(),
        gap_ns=pl.col("gap_ns").sum(),
        n_crossed=pl.col("n_crossed").sum(),
        px_min=pl.col("px_min").min(),
        px_max=pl.col("px_max").max(),
    ).to_dicts()[0]
    res["overall"] = agg
    (OUT_DIR / "data" / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("done:", OUT_DIR)


if __name__ == "__main__":
    main()
