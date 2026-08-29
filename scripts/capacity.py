"""規模(容量)の測定 — 「前に 400 単位超」の局面で実際にどれだけ捌けるか。

【なぜ必要か】
  pull_rule_99_report.md は「引く価値 +25.5 bp/日」を出したが、
  **これは 298 件の約定すべてを自分が取れた場合の値**である。
  実際には他のメイカーと競合するので、取れるのは一部にすぎない。
  「2.9%」という割合を規模の根拠に使ったのは誤用だった(§10-2)。

【測る 3 つ】
  1. 発注機会の総量 … 「最良気配に 400 単位超」の状態が 1 日何秒続くか
                      (時間加重。イベント間の継続時間で重み付ける)
  2. 1 件あたりの規模 … その局面での板の厚みの分布と、
                      他者が実際に置いている注文数量の分布
  3. 競合 …           その 298 件/日を何人のメイカーが分け合っているか。
                      上位者のシェア(HHI・上位1者シェア)

【この分析が答えないこと】
  自分が入ることで他者の行動が変わる効果は測れない(観察データの限界)。
  したがってここで出る数字は**上限**である。
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
OUT = D / "capacity"
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]
Q_THRESH = 400.0


def run_day(dt: str) -> dict | None:
    f = OUT / f"{dt}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    lf = LIFE / f"dt={dt}" / "part-000.parquet"
    if not lf.exists():
        return None
    mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                          columns=["ts", "best_bid", "best_ask", "bid_sz", "ask_sz",
                                   "mid", "is_crossed"])
          .filter(~pl.col("is_crossed")).sort("ts"))
    if mp.height < 5000:
        return None
    ts = mp["ts"].to_numpy()
    dur = np.diff(ts, append=ts[-1])                       # 各状態の継続時間 ns
    tot = float(dur.sum())
    bsz, asz = mp["bid_sz"].to_numpy(), mp["ask_sz"].to_numpy()
    mid = mp["mid"].to_numpy()

    # ---- 1. 発注機会の総量(時間加重)--------------------------------
    occ = {}
    for nm, sz in (("bid", bsz), ("ask", asz)):
        m = sz > Q_THRESH
        occ[nm] = {"sec": float(dur[m].sum() / 1e9), "share": float(dur[m].sum() / tot)}
    m_any = (bsz > Q_THRESH) | (asz > Q_THRESH)
    occ["どちらか"] = {"sec": float(dur[m_any].sum() / 1e9),
                       "share": float(dur[m_any].sum() / tot)}
    # その局面での板の厚みの分布
    thick = {}
    for nm, sz in (("bid", bsz), ("ask", asz)):
        v = sz[sz > Q_THRESH]
        if v.size:
            thick[nm] = {str(p): float(np.percentile(v, p)) for p in (25, 50, 75, 90, 99)}

    # ---- 2/3. その局面で起きたメイカー約定 ----------------------------
    life = (pl.read_parquet(lf, columns=["oid", "side", "px", "orig_sz", "ts_open",
                                         "tif", "is_rejected", "is_trigger"])
            .filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                    & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                    & pl.col("ts_open").is_not_null()))
    qi = (pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                          columns=["oid", "q_ahead_open", "open_unknown"])
          .filter((~pl.col("open_unknown")) & pl.col("q_ahead_open").is_not_null()))
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    fl = (pl.concat([pl.read_parquet(x, columns=["oid", "user", "sz", "px", "crossed"])
                     for x in fs], how="diagonal_relaxed")
          .filter(~pl.col("crossed")))                     # メイカー側のみ
    df = life.join(qi, on="oid", how="inner").join(
        fl.group_by("oid").agg(pl.col("sz").sum().alias("fsz"),
                               pl.col("user").first().alias("user")),
        on="oid", how="inner")
    if df.height < 100:
        return None
    # 発注時に最良気配だったか
    j = np.clip(np.searchsorted(ts, df["ts_open"].to_numpy(), side="right") - 1, 0, len(ts) - 1)
    sd = np.where(df["side"].to_numpy() == "B", 1.0, -1.0)
    ref = np.where(sd > 0, mp["best_bid"].to_numpy()[j], mp["best_ask"].to_numpy()[j])
    at_best = np.abs(sd * (ref - df["px"].to_numpy()) / mid[j] * 1e4) < 1e-9
    sel = at_best & (df["q_ahead_open"].to_numpy() > Q_THRESH)
    if sel.sum() < 20:
        return None
    sub = df.filter(sel)
    users = sub["user"].to_numpy()
    u, c = np.unique(users, return_counts=True)
    share = c / c.sum()
    vol = float(sub["fsz"].sum())
    px_med = float(np.median(mid[j][sel]))

    out = {"dt": dt, "occupancy": occ, "thickness": thick,
           "n_fills": int(sel.sum()), "volume_units": vol,
           "notional_usd": vol * px_med,
           "order_size": {str(p): float(np.percentile(sub["orig_sz"].to_numpy(), p))
                          for p in (25, 50, 75, 90, 99)},
           "n_makers": int(len(u)), "top1_share": float(share.max()),
           "top3_share": float(np.sort(share)[::-1][:3].sum()),
           "hhi": float((share ** 2).sum()),
           "fills_per_maker_median": float(np.median(c))}
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    for dt in sys.argv[1:]:
        r = run_day(dt)
        print(f"{dt}: 占有 {r['occupancy']['どちらか']['share']*100:5.1f}% / "
              f"約定 {r['n_fills']:>4} 件 {r['volume_units']:>8,.0f} 単位 / "
              f"メイカー {r['n_makers']:>3} 者 / 上位1者 {r['top1_share']*100:4.1f}%"
              if r else f"{dt}: スキップ", flush=True)


if __name__ == "__main__":
    main()
