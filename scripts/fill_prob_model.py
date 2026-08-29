"""項目 2: 約定確率モデル  P(fill | キュー位置, 距離, スプレッド, シグナル, 数量)

【目的】
  「どこに、どんな状況で出せば、どれだけ約定するか」を推定する。
  これが無いと選別(不利なときに引く)の**機会費用**が計算できない。

【標本】
  成熟期 10 日。**発注時点で最良気配から 5bp 以内**に置かれた指値のみを対象にする
  (マーケットメイクが実際に置く範囲。深い注文を混ぜると母集団が別物になる)。

【説明変数】すべて**発注時点(ts_open)までに確定**した情報
  log1p(q_ahead)   自分の前にあった数量(= キュー位置)
  n_ahead          前にいた注文の本数
  dist_bp          最良気配からの距離(0 = 最良気配ちょうど)
  spread_bp        そのときのスプレッド
  align            方向 s × (−book_slope_diff)  … シグナルが自分に有利か
  log1p(size)      自分の注文数量
  is_bid           側

【目的変数】 is_filled(その注文が約定で終わったか)

【統計的正当性】
  - 学習は前半 6 日、検証は後半 3 日(時間ブロック分割・シャッフルなし)
  - 標準化の統計量は学習期間だけから推定
  - 評価は AUC と較正(予測 10 分位ごとの実約定率)。**日ごとに計算して中央値と
    符号一貫性で報告**(プールした指標は少数日に支配されるため)
  - プラセボ: align を 1 時間ずらした系列で置き換え、AUC が落ちることを確認
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
SL = D / "book_slope_grain"
OUT = D / "fill_prob"
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
MAX_DIST_BP = 5.0
FEATS = ["log_q_ahead", "n_ahead", "dist_bp", "spread_bp", "align", "log_size", "is_bid"]
TRAIN_DAYS = 6
PLACEBO_SHIFT = 3_600_000_000_000


def build_day(dt: str) -> pl.DataFrame | None:
    f = SL / f"{dt}.parquet"
    if not f.exists():
        return None
    life = (pl.read_parquet(LIFE / f"dt={dt}" / "part-000.parquet",
                            columns=["oid", "side", "px", "orig_sz", "ts_open",
                                     "terminal_status", "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null())
            .sort("ts_open"))
    q = pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                        columns=["oid", "is_filled", "q_ahead_open", "n_ahead_open",
                                 "open_unknown"])
    df = life.join(q, on="oid", how="inner").filter(~pl.col("open_unknown"))
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "best_bid", "best_ask", "mid", "spread_bp",
                                   "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    df = df.join_asof(mp, left_on="ts_open", right_on="ts", strategy="backward").drop_nulls(["mid"])
    s = pl.read_parquet(f)
    sig = pl.DataFrame({"ts": s["ts"], "slope": s["bid_25bp"] - s["ask_25bp"]}).sort("ts")
    sigp = sig.with_columns((pl.col("ts") + PLACEBO_SHIFT).alias("ts")).sort("ts")
    df = (df.join_asof(sig, left_on="ts_open", right_on="ts", strategy="backward")
            .join_asof(sigp.rename({"slope": "slope_pl", "ts": "ts_pl"}),
                       left_on="ts_open", right_on="ts_pl", strategy="backward"))
    s_dir = pl.when(pl.col("side") == "B").then(1.0).otherwise(-1.0)
    df = df.with_columns(
        # 自分の側の最良気配からの距離(内側=負、外側=正)
        (pl.when(pl.col("side") == "B")
         .then((pl.col("best_bid") - pl.col("px")) / pl.col("mid") * 1e4)
         .otherwise((pl.col("px") - pl.col("best_ask")) / pl.col("mid") * 1e4)).alias("dist_bp"),
        (s_dir * (-pl.col("slope"))).alias("align"),
        (s_dir * (-pl.col("slope_pl"))).alias("align_placebo"),
        pl.col("q_ahead_open").log1p().alias("log_q_ahead"),
        pl.col("orig_sz").log1p().alias("log_size"),
        pl.col("n_ahead_open").cast(pl.Float64).alias("n_ahead"),
        (pl.col("side") == "B").cast(pl.Float64).alias("is_bid"),
        pl.col("is_filled").cast(pl.Int8).alias("y"),
    ).filter(pl.col("dist_bp").abs() <= MAX_DIST_BP)
    return df.select(["ts_open", "y", "align_placebo", *FEATS]).drop_nulls()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(p.stem for p in SL.glob("*.parquet"))
    frames = []
    for k, dt in enumerate(days):
        d = build_day(dt)
        if d is None or d.height < 1000:
            continue
        d.write_parquet(OUT / f"{dt}.parquet", compression="zstd")
        frames.append(d.with_columns(pl.lit(k).alias("day")))
        print(f"{dt}: {d.height:,} 件 / 約定率 {d['y'].mean():.4f}", flush=True)
    df = pl.concat(frames)
    day = df["day"].to_numpy()
    y = df["y"].to_numpy()
    X = df.select(FEATS).to_numpy()
    Xp = df.select([c if c != "align" else "align_placebo" for c in FEATS]).to_numpy()
    tr = day < TRAIN_DAYS
    te = ~tr
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Zp = (Xp - mu) / sd

    mdl = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])
    mdl_p = LogisticRegression(max_iter=2000, C=1.0).fit(Zp[tr], y[tr])

    def auc(sc, yy):
        o = np.argsort(sc)
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(1, len(sc) + 1)
        n1 = yy.sum(); n0 = len(yy) - n1
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan

    res: dict = {"n": int(len(y)), "days": days, "fill_rate": float(y.mean()),
                 "train_days": TRAIN_DAYS, "features": FEATS,
                 "coef": {f: float(c) for f, c in zip(FEATS, mdl.coef_[0])}}
    p_te = mdl.predict_proba(Z)[:, 1]
    pp_te = mdl_p.predict_proba(Zp)[:, 1]
    daily, daily_p = [], []
    for k in np.unique(day[te]):
        m = day == k
        if m.sum() < 500 or y[m].sum() == 0:
            continue
        daily.append(auc(p_te[m], y[m]))
        daily_p.append(auc(pp_te[m], y[m]))
    res["auc_oos_daily"] = {"median": float(np.median(daily)), "days": len(daily),
                            "min": float(np.min(daily)), "values": daily}
    res["auc_placebo_daily"] = {"median": float(np.median(daily_p)), "days": len(daily_p)}
    # 較正(検証期間・予測 10 分位)
    q = np.quantile(p_te[te], np.linspace(0, 1, 11))
    cal = []
    for i in range(10):
        sel = te & (p_te >= q[i]) & ((p_te <= q[i + 1]) if i == 9 else (p_te < q[i + 1]))
        if sel.sum() < 100:
            continue
        cal.append({"decile": i + 1, "n": int(sel.sum()),
                    "pred": float(p_te[sel].mean()), "actual": float(y[sel].mean())})
    res["calibration_oos"] = cal
    # 実務用の表: キュー位置 × 距離 別の実測約定率(検証期間)
    tbl = []
    qa = df["log_q_ahead"].to_numpy()
    dbp = df["dist_bp"].to_numpy()
    for lo, hi, lab in ((-0.1, 0.1, "0(先頭)"), (0.1, 3.0, "小"), (3.0, 6.0, "中"),
                        (6.0, 99.0, "大")):
        for d0, d1, dl in ((-5.1, -0.01, "内側"), (-0.01, 0.01, "最良気配"),
                           (0.01, 1.0, "1bp 外"), (1.0, 5.1, "1-5bp 外")):
            sel = te & (qa >= lo) & (qa < hi) & (dbp >= d0) & (dbp < d1)
            if sel.sum() < 200:
                continue
            tbl.append({"q_ahead": lab, "dist": dl, "n": int(sel.sum()),
                        "fill_rate": float(y[sel].mean())})
    res["empirical_table"] = tbl
    (D / "fill_prob_model.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    print(f"\n件数 {res['n']:,} / 全体の約定率 {res['fill_rate']:.4f}")
    print(f"標本外 AUC(日次中央値) = {res['auc_oos_daily']['median']:.4f} "
          f"(最小 {res['auc_oos_daily']['min']:.4f}, {res['auc_oos_daily']['days']} 日)")
    print(f"プラセボ AUC        = {res['auc_placebo_daily']['median']:.4f}")
    print("\n係数(標準化):")
    for f, c in sorted(res["coef"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:>14} {c:+.4f}")
    print("\n較正(標本外・予測 10 分位):")
    for c in res["calibration_oos"]:
        print(f"  {c['decile']:>2} 予測 {c['pred']:.4f} / 実測 {c['actual']:.4f} (n={c['n']:,})")
    print("\n実測の約定率(キュー位置 × 距離):")
    for t in tbl:
        print(f"  前の数量={t['q_ahead']:>8} 位置={t['dist']:>10} "
              f"約定率 {t['fill_rate']:.4f} (n={t['n']:,})")


if __name__ == "__main__":
    main()
