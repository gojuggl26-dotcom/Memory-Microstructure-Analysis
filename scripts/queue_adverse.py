"""キュー順位ごとの (1) 約定確率 (2) 約定後の平均価格変動 (3) その分布。

【問い】
  キューの先頭に居ることは、逆選択から守ってくれるのか。

  行列の最後尾に付くと、自分に約定が届くのは**行列が一掃されるとき** —
  つまり大口が板を薙ぎ払う瞬間である。だとすれば後ろほど不利なはずである。
  一方で先頭は「約定しやすい」ので、平凡な流れでも約定してしまう。
  どちらが効くかは測らないと判らない。

【損益の分解】(メイカー 1 約定あたり、bp)
  粗利     gross_bp = s × (mid(t) − px_fill) / mid(t) × 10⁴
             … 約定した瞬間に中値から見て取れている分(= 実質の半スプレッド)
  逆選択   adv_bp   = s × (mid(t+Δ) − mid(t)) / mid(t) × 10⁴
             … 約定後に中値が自分に対してどう動いたか(負なら不利)
  正味     net_bp   = gross_bp + adv_bp
  s = +1(メイカーが買った) / −1(メイカーが売った)

【時間契約】
  キュー順位は**発注時点**(q_ahead_open)の値。約定より前に確定している。
  価格変動は約定時刻から**前向き**に測る(これは結果であって説明変数ではない)。
  中値の参照は backward asof のみ(t+Δ 以前の最後の中値)。

【推論】
  日ごとに集計し、28 日の中央値・符号一貫性・ブロックブートストラップで報告する。
  プールした平均は少数日に支配されるため主指標にしない。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
OUT = D / "queue_adverse2"
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
HORIZONS = {"100ms": 100_000_000, "1s": 1_000_000_000,
            "10s": 10_000_000_000, "60s": 60_000_000_000}
# 自分の前にある数量の区分(p50=19.6, p75=221.6, p90=604, p95=1000)
QB = [(-0.5, 0.5, "0 (先頭)"), (0.5, 20, "0<q<=20"), (20, 100, "20<q<=100"),
      (100, 400, "100<q<=400"), (400, 1000, "400<q<=1000"), (1000, np.inf, "q>1000")]
# 発注時の最良気配からの距離(内側は負)
DB = [(-99, -1e-9, "内側"), (-1e-9, 1e-9, "最良気配"), (1e-9, 1.0, "0-1bp"),
      (1.0, 5.0, "1-5bp")]


def bucket(v: np.ndarray, spec) -> np.ndarray:
    out = np.full(v.shape, -1, dtype=np.int8)
    for i, (lo, hi, _) in enumerate(spec):
        out[(v > lo) & (v <= hi)] = i
    return out


def run_day(dt: str) -> dict | None:
    f = OUT / f"{dt}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    lf = LIFE / f"dt={dt}" / "part-000.parquet"
    if not lf.exists():
        return None
    life = (pl.read_parquet(lf, columns=["oid", "side", "px", "orig_sz", "ts_open",
                                         "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null()))
    q = pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                        columns=["oid", "is_filled", "q_ahead_open", "n_ahead_open",
                                 "open_unknown"]).filter(~pl.col("open_unknown"))
    # データ窓の終端では結果が判らない注文が残る(右打ち切り)。
    # 全期間で 2026-08-10 の 0.14% のみ。件数を記録して除外する。
    n_cens = int(q["is_filled"].null_count())
    q = q.filter(pl.col("is_filled").is_not_null())
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "mid", "best_bid", "best_ask", "spread_bp",
                                   "bid_sz", "ask_sz", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    mts = mp["ts"].to_numpy(); mid = mp["mid"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    spr = mp["spread_bp"].to_numpy()
    dep = (mp["bid_sz"].to_numpy() + mp["ask_sz"].to_numpy())

    # ---- (1) 約定確率: 発注時点の情報だけで層別する ----------------------
    df = life.join(q, on="oid", how="inner").sort("ts_open")
    j = np.searchsorted(mts, df["ts_open"].to_numpy(), side="right") - 1
    ok = j >= 0
    df = df.filter(ok); j = j[ok]
    s_dir = np.where(df["side"].to_numpy() == "B", 1.0, -1.0)
    px = df["px"].to_numpy()
    # 自分の側の最良気配からの距離(内側 = 負、外側 = 正)
    ref = np.where(s_dir > 0, bb[j], ba[j])
    dist = s_dir * (ref - px) / mid[j] * 1e4
    o_spr, o_dep = spr[j], dep[j]          # 発注時点のスプレッドと 1 段目の厚み
    qb = bucket(df["q_ahead_open"].to_numpy().astype(float), QB)
    db = bucket(dist, DB)
    filled = df["is_filled"].to_numpy().astype(np.int8)
    prob = []
    for bi, (_, _, bl) in enumerate(QB):
        for di, (_, _, dl) in enumerate(DB):
            m = (qb == bi) & (db == di)
            if m.sum() < 200:
                continue
            prob.append({"q": bl, "dist": dl, "n": int(m.sum()),
                         "fill_rate": float(filled[m].mean())})
    # 距離を最良気配に固定した「純粋なキュー効果」
    prob_best = []
    for bi, (_, _, bl) in enumerate(QB):
        m = (qb == bi) & (db == 1)
        if m.sum() < 200:
            continue
        prob_best.append({"q": bl, "n": int(m.sum()), "fill_rate": float(filled[m].mean())})

    # ---- (2)(3) 約定後の価格変動 ----------------------------------------
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fl = pl.concat([pl.read_parquet(x, columns=["oid", "ts", "px", "sz", "crossed", "tid"])
                    for x in fs], how="diagonal_relaxed")
    fl = (fl.filter(~pl.col("crossed"))          # メイカー側の約定だけ
            .with_columns(pl.col("ts").cast(pl.Int64).alias("t")))
    key = (df.select(["oid", "side", "q_ahead_open", "n_ahead_open"])
             .with_columns(pl.Series("dist_bp", dist), pl.Series("qb", qb),
                           pl.Series("db", db), pl.Series("o_spr", o_spr),
                           pl.Series("o_dep", o_dep),
                           pl.Series("t_open", df["ts_open"].to_numpy())))
    fj = fl.join(key, on="oid", how="inner")
    if fj.height < 500:
        return None
    t = fj["t"].to_numpy()
    fpx = fj["px"].to_numpy()
    sd = np.where(fj["side"].to_numpy() == "B", 1.0, -1.0)
    j0 = np.searchsorted(mts, t, side="right") - 1
    good = j0 >= 0
    t, fpx, sd, j0 = t[good], fpx[good], sd[good], j0[good]
    fq = fj["qb"].to_numpy()[good]; fd = fj["db"].to_numpy()[good]
    f_spr = fj["o_spr"].to_numpy()[good]; f_dep = fj["o_dep"].to_numpy()[good]
    delay = (t - fj["t_open"].to_numpy()[good]) / 1e9
    # 発注時のスプレッド・厚みの三分位(その日の分布から)。交絡の切り分け用
    st_ = np.digitize(f_spr, np.quantile(f_spr, [1/3, 2/3]))
    dt_ = np.digitize(f_dep, np.quantile(f_dep, [1/3, 2/3]))
    m0 = mid[j0]
    gross = sd * (m0 - fpx) / m0 * 1e4
    adv = {}
    for hn, dt_ns in HORIZONS.items():
        j1 = np.searchsorted(mts, t + dt_ns, side="right") - 1
        v = np.where(j1 >= 0, sd * (mid[np.clip(j1, 0, len(mid) - 1)] - m0) / m0 * 1e4, np.nan)
        v[t + dt_ns > mts[-1]] = np.nan
        adv[hn] = v

    rows = []
    for bi, (_, _, bl) in enumerate(QB):
        for scope, msk in (("全体", fq == bi), ("最良気配のみ", (fq == bi) & (fd == 1))):
            if msk.sum() < 100:
                continue
            r = {"q": bl, "scope": scope, "n": int(msk.sum()),
                 "gross_bp": float(np.mean(gross[msk])),
                 "spread_at_open": float(np.mean(f_spr[msk])),
                 "depth_at_open": float(np.mean(f_dep[msk])),
                 "delay_sec_median": float(np.median(delay[msk])),
                 "delay_sec_mean": float(np.mean(delay[msk]))}
            for hn in HORIZONS:
                v = adv[hn][msk]
                v = v[np.isfinite(v)]
                if v.size < 50:
                    continue
                r[f"adv_{hn}"] = float(np.mean(v))
                r[f"net_{hn}"] = float(np.mean(gross[msk][np.isfinite(adv[hn][msk])]) + np.mean(v))
                r[f"neg_{hn}"] = float(np.mean(v < 0))
                if hn == "1s":
                    r["q_adv"] = {str(p): float(np.percentile(v, p))
                                  for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
                    r["sd_1s"] = float(v.std())
            rows.append(r)

    strat = []
    for bi, (_, _, bl) in enumerate(QB):
        for nm, arr in (("spread", st_), ("depth", dt_)):
            for k in (0, 1, 2):
                m = (fq == bi) & (fd == 1) & (arr == k) & np.isfinite(adv["1s"])
                if m.sum() < 100:
                    continue
                strat.append({"q": bl, "by": nm, "tercile": k, "n": int(m.sum()),
                              "adv_1s": float(np.mean(adv["1s"][m])),
                              "gross_bp": float(np.mean(gross[m]))})
    out = {"dt": dt, "n_orders": int(df.height), "n_maker_fills": int(len(t)),
           "strat": strat,
           "n_censored_dropped": n_cens,
           "fill_prob": prob, "fill_prob_best": prob_best, "post_fill": rows}
    # 分布図用に 1 秒の逆選択を層別で間引き保存
    samp = {}
    for bi, (_, _, bl) in enumerate(QB):
        m = (fq == bi) & (fd == 1) & np.isfinite(adv["1s"])
        v = adv["1s"][m]
        if v.size > 20000:
            v = v[np.random.default_rng(0).choice(v.size, 20000, replace=False)]
        samp[bl] = v.astype(np.float32).tolist()
    out["sample_1s_best"] = samp
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    days = sys.argv[1:]
    for dt in days:
        r = run_day(dt)
        print(f"{dt}: 注文 {r['n_orders']:,} / メイカー約定 {r['n_maker_fills']:,}"
              if r else f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
