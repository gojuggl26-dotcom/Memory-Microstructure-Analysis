"""イベント到着の「かたまり具合」(burstiness)を 7 つの系列で測る。

【指標】
到着間隔 τ の平均 μ と標準偏差 σ から Goh–Barabási の

```
B = (σ − μ) / (σ + μ)
```

を出す。B ∈ [−1, +1] で、**−1 = 完全に規則的(等間隔)、0 = ポアソン(無記憶)、
+1 = 極端にかたまる**。

【★ B は標本の大きさに強く依存する — 補正版を必ず併記する】
B はイベント数 n が小さいと 0 へ引き寄せられる。ポアソン過程から n 個取って
B を計算しても 0 にはならず、n によって系統的にずれる。
Kim & Jo (2016) の有限標本補正

```
A_n = ( √(n+1)·r − √(n−1) ) / ( (√(n+1) − 2)·r + √(n−1) ),   r = σ/μ
```

を併せて出す。**イベント数が桁違いに違う系列どうし(例: ウォレット別)を
生の B で比べてはいけない。**

【7 つの系列】
    NEW burstiness        open イベントの到着間隔
    REMOVE burstiness     終端イベント(約定 + 取消)の到着間隔
    UPDATE burstiness     板を触った全イベントの到着間隔
    bid-event burstiness  買い側のイベントだけ
    ask-event burstiness  売り側のイベントだけ
    wallet-event burstiness  ウォレットごとのイベント列(分布として報告)
    BBO-change burstiness    最良気配が動いた瞬間の到着間隔(l2/bbo)

★この列構成には「注文の訂正(amend)」という状態が無い。取引所側で訂正が
取消 + 新規に分解されているため、**UPDATE は「板を触った全イベント」**と定義した。
訂正だけを取り出すことはできない。

【★★同時刻のイベントが 7〜8 割を占める】
同じナノ秒に入るイベントの割合(τ = 0)は実測で **71〜83%** ある。取引所が
ブロック単位で同じ時刻を打っているためで、**生の B は「取引所のバッチ処理の
かたまり具合」を測っていて、参加者の到着過程ではない。**

そこで 2 通りを併記する。

    B        全イベントの到着間隔(τ = 0 を含む)。バッチの中身まで含む
    B_uniq   **同時刻を 1 つに畳んだ**到着間隔。バッチそのものの到着過程

参加者の行動を論じるなら `B_uniq` を見ること。`zero_share` も必ず併記する。

【帰無対照】
同じ本数・同じ平均率のポアソン過程を 200 回発生させ、B の分布を出す。
**B の絶対値ではなく、ポアソンからのずれで読むこと。**

【x が確定する時刻】
1 日の中で閉じた記述統計。将来の値は使っていない。

    uv run python scripts/build_burstiness.py --coin xyz:MU
出力: data/burst_daily_<coin>.csv   … 日 × 系列 の B / A_n / 帰無分布
      data/burst_wallet_<coin>.csv  … ウォレット × B / A_n
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
L1COLS = ["ts", "oid", "side", "status", "tif", "is_trigger"]
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}
N_NULL = 200
MIN_EV = 50
SEED = 20260901
SERIES = ["NEW", "REMOVE", "UPDATE", "bid", "ask", "BBO"]


def burst(tau: np.ndarray) -> tuple[float, float, float, int]:
    """B と有限標本補正 A_n を返す。tau は到着間隔(秒)。"""
    n = len(tau)
    if n < MIN_EV:
        return np.nan, np.nan, np.nan, n
    mu = float(tau.mean())
    sd = float(tau.std(ddof=1))
    if mu <= 0:
        return np.nan, np.nan, np.nan, n
    B = (sd - mu) / (sd + mu)
    r = sd / mu
    s1, s2 = np.sqrt(n + 1), np.sqrt(n - 1)
    den = (s1 - 2) * r + s2
    A = (s1 * r - s2) / den if den != 0 else np.nan
    return B, float(A), mu, n


def null_B(n: int, rng) -> tuple[float, float]:
    """同じ本数のポアソン過程から得られる B の分布(平均と 97.5% 点)。"""
    if n < MIN_EV:
        return np.nan, np.nan
    m = min(n, 200_000)                 # 巨大な系列は打ち切って計算量を抑える
    v = np.empty(N_NULL)
    for i in range(N_NULL):
        t = rng.exponential(1.0, m)
        v[i] = (t.std(ddof=1) - t.mean()) / (t.std(ddof=1) + t.mean())
    return float(v.mean()), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(SEED)
    l1dir, l1udir = ROOT / "data" / f"l1_{tag}", ROOT / "data" / f"l1u_{tag}"
    days = sorted(p.stem.split("=")[1] for p in l1udir.glob("dt=*.parquet"))
    if a.days:
        days = days[: a.days]
    bbo = pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
    print(f"[日] {len(days)}", file=sys.stderr)

    rows, wrows = [], []
    nullcache: dict[int, tuple] = {}
    for n_, day in enumerate(days, 1):
        try:
            d = pl.read_parquet(l1dir / f"dt={day}.parquet", columns=L1COLS)
            u = pl.read_parquet(l1udir / f"dt={day}.parquet")
        except Exception as e:
            print(f"  {day} 読めない: {type(e).__name__}", file=sys.stderr)
            continue
        if u.height != d.height:
            sys.exit(f"{day}: user の行数が合わない")
        d = d.with_columns(user=u["user"], ts=pl.col("ts").cast(pl.Int64)).sort("ts")
        st = d["status"].to_numpy()
        rest = (~d["is_trigger"].to_numpy()) & np.isin(d["tif"].to_numpy(), list(RESTING))
        ts = d["ts"].to_numpy()
        isb = d["side"].to_numpy() == "B"
        is_open = (st == "open") & rest
        is_term = np.isin(st, ["filled"] + list(CANCEL))
        touched = is_open | is_term

        sel = {"NEW": is_open, "REMOVE": is_term, "UPDATE": touched,
               "bid": touched & isb, "ask": touched & ~isb}
        B = (bbo.filter(pl.col("dt") == day).select("ts").collect()
                .with_columns(ts=pl.col("ts").cast(pl.Int64)).sort("ts"))
        for name, m in list(sel.items()) + [("BBO", None)]:
            t = (B["ts"].to_numpy() if name == "BBO" else ts[m])
            if len(t) < MIN_EV + 1:
                continue
            tau = np.diff(t) / 1e9
            b, an, mu, nn = burst(tau)
            tu = np.unique(t)                     # 同時刻を 1 つに畳む
            bu, anu, muu, nnu = burst(np.diff(tu) / 1e9)
            if nn not in nullcache:
                nullcache[nn] = null_B(nn, rng)
            nb, nhi = nullcache[nn]
            rows.append({"dt": day, "series": name, "n": nn + 1,
                         "rate_per_s": 1.0 / mu if mu and mu > 0 else np.nan,
                         "B": b, "A_n": an, "zero_share": float((tau == 0).mean()),
                         "B_uniq": bu, "A_n_uniq": anu, "n_uniq": nnu + 1,
                         "rate_uniq_per_s": 1.0 / muu if muu and muu > 0 else np.nan,
                         "B_null": nb, "B_null_hi": nhi})

        # ---- ウォレットごと -------------------------------------------------
        mw = touched
        uu = d["user"].to_numpy()[mw]
        tw = ts[mw]
        # ★同時刻を畳んでからウォレット別の間隔を取る(バッチの影響を外す)
        o = np.lexsort((tw, uu))
        us, tsort = uu[o], tw[o]
        du = np.r_[True, (tsort[1:] != tsort[:-1]) | (us[1:] != us[:-1])]
        us, tsort = us[du], tsort[du]
        newu = np.r_[True, us[1:] != us[:-1]]
        tau = np.diff(tsort) / 1e9
        keep = ~newu[1:]
        uid_all, uidx = np.unique(us, return_inverse=True)
        gid = uidx[1:][keep]
        tv = tau[keep]
        cnt = np.bincount(gid, minlength=len(uid_all))
        s1 = np.bincount(gid, weights=tv, minlength=len(uid_all))
        s2 = np.bincount(gid, weights=tv * tv, minlength=len(uid_all))
        ok = cnt >= MIN_EV
        mu_ = np.where(ok, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(ok, (s2 - cnt * mu_ ** 2) / np.maximum(cnt - 1, 1), np.nan)
        sd_ = np.sqrt(np.maximum(var, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            Bw = (sd_ - mu_) / (sd_ + mu_)
            r = sd_ / mu_
            s1n, s2n = np.sqrt(cnt + 1), np.sqrt(cnt - 1)
            Aw = (s1n * r - s2n) / ((s1n - 2) * r + s2n)
        for i in np.flatnonzero(ok):
            wrows.append({"dt": day, "user": uid_all[i], "n": int(cnt[i]) + 1,
                          "B": float(Bw[i]), "A_n": float(Aw[i]),
                          "rate_per_s": float(1 / mu_[i]) if mu_[i] > 0 else np.nan})
        if n_ % 10 == 0 or n_ == len(days):
            print(f"  {day}  ({n_}/{len(days)} 日)", file=sys.stderr)

    D = pl.DataFrame(rows)
    D.write_csv(ROOT / "data" / f"burst_daily_{tag}.csv")
    W = pl.DataFrame(wrows)
    Wg = (W.group_by("user").agg(pl.col("n").sum(), pl.col("B").median(),
                                 pl.col("A_n").median(), pl.col("rate_per_s").median(),
                                 pl.len().alias("n_days"))
           .filter(pl.col("n_days") >= 10).sort("n", descending=True))
    Wg.write_csv(ROOT / "data" / f"burst_wallet_{tag}.csv")
    print(f"\n[集計] 日 × 系列 {D.height:,} 行 / ウォレット {Wg.height:,} 者",
          file=sys.stderr)
    print(D.group_by("series").agg(pl.col("B").median().round(3),
                                   pl.col("B_uniq").median().round(3),
                                   pl.col("A_n_uniq").median().round(3),
                                   pl.col("B_null").median().round(3),
                                   pl.col("zero_share").median().round(3),
                                   pl.col("rate_per_s").median().round(1),
                                   pl.col("rate_uniq_per_s").median().round(1))
          .sort("series").to_pandas().to_string(), file=sys.stderr)
    print(f"\n-> data/burst_daily_{tag}.csv / burst_wallet_{tag}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
