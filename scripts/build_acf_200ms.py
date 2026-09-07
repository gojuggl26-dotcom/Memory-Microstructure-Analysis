"""OBI と OFI を 200ms 格子に載せ、それぞれの自己相関を算出して特徴量に保存する。

【なぜ格子に載せ直すか】
OBI と OFI はイベント時刻(最良気配が動いた瞬間)に定義されている。イベントの
間隔は一定でないので、そのまま並べた自己相関は「200ms 後との相関」ではなく
「次のイベントとの相関」になる。時計で 200ms ごとに測り直す。

【載せ方が 2 つの量で違う】
    OBI … **状態**(今どちら側が厚いか)。格子点 T の値は T 以前の最後の
           イベントの OBI。イベントが無い格子点は直前の値を持ち越す
    OFI … **流量**(この間にどれだけ足し引きされたか)。格子点 T の値は
           区間 (T−200ms, T] に入るイベントの OFI の合計。イベントが
           無ければ 0(何も起きなかった = 流量 0 は正しい)

どちらも T 時点の情報だけで決まる。

【★持ち越しは自己相関を機械的に押し上げる】
イベントが無い格子点で OBI は前の値のままなので、空の格子点が多いほど
OBI の自己相関は 1 に近づく。これは板の粘りではなく標本化の副作用である。
そのため空の格子点の割合を必ず併記する(`empty_share`)。

【自己相関の推定】
日ごとに、平均を日内平均としてラグ k の標本自己相関を出す。

    rho_k = Σ_t (x_t − x̄)(x_{t+k} − x̄) / Σ_t (x_t − x̄)^2

日をまたぐ集計は日を単位にしたブートストラップで区間を出す(格子点どうしは
強く相関しているので、点を単位にした区間は狭すぎる)。立会日と閉場日は分ける。

【帰無対照は「並べ替え」であって「巡回シフト」ではない】
このリポジトリの他のレポートでは日内の巡回シフトを帰無対照に使っているが、
それは x と y の対応だけを壊す道具である。**自己相関に対しては巡回シフトは
無効**で、系列を回しても自分自身との相関はそのまま残る。ここでは日内で
**無作為に並べ替える**。時間の順序が消えるので自己相関は 0 に潰れるはず。

【x が確定する時刻 / y の期間】
保存する特徴量はすべて、その格子点 T までの情報だけで計算している。

    obi / ofi          … 区間の終端 T で確定
    *_rho1_60s / _5m   … T で終わる後ろ向き窓の中だけで計算した 1 ラグ自己相関

`shift(-k)` は使っていない。将来の値は一切入らない。
なお自己相関表(acf_200ms_*)のほうは**日内平均で中心化した記述統計**であり、
標本外評価に使うなら中心化も窓の中だけでやり直すこと。

    uv run python scripts/build_acf_200ms.py --coin xyz:MU
出力: data/acf_200ms_<coin>.parquet     … 変数 × 日区分 × ラグ の自己相関と区間
      data/acf_200ms_daily_<coin>.parquet … 変数 × 日 × ラグ の素の自己相関
      data/feat200_<coin>/dt=*.parquet  … 200ms 格子の特徴量(日ごと)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STEP_NS = 200_000_000                 # 200ms
PER_DAY = 24 * 60 * 60 * 5            # 432,000 格子点
LAGS = list(range(1, 151)) + [200, 300, 450, 600, 900, 1500]   # 200ms 〜 5 分
W60, W5M = 300, 1500                  # 後ろ向き窓(格子点数)= 60 秒 / 5 分
CHUNK = 10                            # 一度に読む日数
N_BOOT = 400
SEED = 20260828


def load_days(coin: str) -> list[str]:
    p = ROOT / "data" / f"bbo_{coin.replace(':', '_')}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い。先に fetch_bbo.py を実行すること")
    return sorted(pl.scan_parquet(p).select("dt").unique().collect()["dt"].to_list())


def read_chunk(coin: str, days: list[str]) -> tuple[pl.DataFrame, int]:
    """その日群の BBO を読み、クロス行を落として OBI / OFI を付ける。"""
    p = ROOT / "data" / f"bbo_{coin.replace(':', '_')}.parquet"
    d = (pl.scan_parquet(p).filter(pl.col("dt").is_in(days)).collect().sort("ts"))
    n0 = d.height
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    pb, pa = pl.col("best_bid"), pl.col("best_ask")
    qb, qa = pl.col("bid_sz"), pl.col("ask_sz")
    d = d.with_columns(
        obi=(qb - qa) / (qb + qa),
        ofi=(pl.when(pb >= pb.shift(1)).then(qb).otherwise(0.0)
             - pl.when(pb <= pb.shift(1)).then(qb.shift(1)).otherwise(0.0)
             - pl.when(pa <= pa.shift(1)).then(qa).otherwise(0.0)
             + pl.when(pa >= pa.shift(1)).then(qa.shift(1)).otherwise(0.0)),
    ).with_columns(
        # 日の最初の行は前の行が前日なので OFI を無効(0 ではなく欠測)にする
        ofi=pl.when(pl.col("dt") == pl.col("dt").shift(1)).then(pl.col("ofi")).otherwise(None)
    )
    return d.select("ts", "dt", "obi", "ofi"), n0 - d.height


def to_grid(day: str, e: pl.DataFrame) -> pl.DataFrame:
    """その日のイベントを 200ms 格子に載せる。"""
    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns")
             .cast(pl.Int64)[0])
    b = ((e["ts"].to_numpy() - t0) // STEP_NS).astype(np.int64)
    ok = (b >= 0) & (b < PER_DAY)
    b, obi, ofi = b[ok], e["obi"].to_numpy()[ok], e["ofi"].to_numpy()[ok]

    n_ev = np.bincount(b, minlength=PER_DAY).astype(np.int32)
    # OFI は区間内の合計。欠測(日の初回)は 0 として足さない
    fo = np.nan_to_num(ofi, nan=0.0)
    ofi_g = np.bincount(b, weights=fo, minlength=PER_DAY)
    # OBI は区間内の最後の値。b は ts 昇順なので後ろから書けば最後が残る
    obi_g = np.full(PER_DAY, np.nan)
    obi_g[b] = obi                      # 同じ b が複数あれば最後の代入が残る
    # 空の格子点は直前の値を持ち越す(先頭の空白は nan のまま)
    idx = np.where(np.isfinite(obi_g), np.arange(PER_DAY), 0)
    np.maximum.accumulate(idx, out=idx)
    obi_g = np.where(np.isfinite(obi_g[idx]), obi_g[idx], np.nan)

    ts = t0 + (np.arange(PER_DAY, dtype=np.int64) + 1) * STEP_NS   # 区間の終端
    return pl.DataFrame({"ts": ts, "dt": [day] * PER_DAY,
                         "n_ev": n_ev, "obi": obi_g, "ofi": ofi_g})


def acf(x: np.ndarray, lags: list[int]) -> np.ndarray:
    """標本自己相関。欠測を含む先頭は呼ぶ側で落としておくこと。"""
    z = x - x.mean()
    d0 = float(z @ z)
    if d0 <= 0:
        return np.full(len(lags), np.nan)
    return np.array([float(z[:-k] @ z[k:]) / d0 for k in lags])


def roll_rho1(x: pl.Expr, w: int) -> pl.Expr:
    """T で終わる後ろ向き窓 w の中だけで測った 1 ラグ自己相関。"""
    y = x.shift(1)
    sxy, sx, sy = (x * y).rolling_sum(w), x.rolling_sum(w), y.rolling_sum(w)
    sxx, syy = (x * x).rolling_sum(w), (y * y).rolling_sum(w)
    num = w * sxy - sx * sy
    den = ((w * sxx - sx * sx) * (w * syy - sy * sy)).sqrt()
    return pl.when(den > 0).then(num / den).otherwise(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0, help="先頭 N 日だけ(0 = 全日)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(SEED)

    days = load_days(a.coin)
    if a.days:
        days = days[: a.days]
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days}
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    outdir = ROOT / "data" / f"feat200_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows, empties, n_drop = [], [], 0

    for c0 in range(0, len(days), CHUNK):
        part = days[c0:c0 + CHUNK]
        E, nd = read_chunk(a.coin, part)
        n_drop += nd
        for day in part:
            g = to_grid(day, E.filter(pl.col("dt") == day))
            first = int(np.argmax(np.isfinite(g["obi"].to_numpy())))
            emp = float((g["n_ev"] == 0).mean())
            empties.append({"dt": day, "empty_share": emp,
                            "n_ev_mean": float(g["n_ev"].mean())})

            # ---- 自己相関(その日の格子全体) --------------------------------
            ob = g["obi"].to_numpy()[first:]
            of = g["ofi"].to_numpy()[first:]
            for nm, x in (("OBI", ob), ("OFI", of)):
                r = acf(x, LAGS)
                sh = acf(rng.permutation(x), LAGS)      # 帰無対照(日内で並べ替え)
                for k, rk, sk in zip(LAGS, r, sh):
                    rows.append({"dt": day, "var": nm, "lag": k,
                                 "lag_ms": k * 200, "rho": rk, "rho_null": sk,
                                 "n": len(x)})

            # ---- 特徴量(後ろ向き窓の自己相関)--------------------------------
            g = g.with_columns(
                obi_rho1_60s=roll_rho1(pl.col("obi"), W60),
                obi_rho1_5m=roll_rho1(pl.col("obi"), W5M),
                ofi_rho1_60s=roll_rho1(pl.col("ofi"), W60),
                ofi_rho1_5m=roll_rho1(pl.col("ofi"), W5M),
            ).with_columns(
                pl.col("obi", "ofi", "obi_rho1_60s", "obi_rho1_5m",
                       "ofi_rho1_60s", "ofi_rho1_5m").cast(pl.Float32)
            )
            g.write_parquet(outdir / f"dt={day}.parquet", compression="zstd")
        print(f"  {part[-1]}  ({c0 + len(part)}/{len(days)} 日)", file=sys.stderr)

    D = pl.DataFrame(rows)
    D.write_parquet(ROOT / "data" / f"acf_200ms_daily_{tag}.parquet")
    EM = pl.DataFrame(empties)
    # ★plot_acf_200ms が読む日次 lag1 の表。以前は別セッションのワンオフ生成で、
    #   他銘柄に回したときここが無くて図が落ちた。builder が責任を持って書く。
    L1 = (D.filter(pl.col("lag") == 1)
          .pivot(on="var", index="dt", values="rho")
          .join(EM.select("dt", empty=pl.col("empty_share")), on="dt")
          .sort("dt"))
    L1.write_csv(ROOT / "data" / f"acf_200ms_daily_lag1_{tag}.csv")
    print(f"[格子] 空の格子点の割合 中央値 {EM['empty_share'].median():.1%} / "
          f"最小 {EM['empty_share'].min():.1%} / 最大 {EM['empty_share'].max():.1%}",
          file=sys.stderr)
    print(f"[除去] クロス・数量 0 の行 {n_drop:,}", file=sys.stderr)

    # ---- 日をまたぐ集計(日単位のブートストラップ)----------------------------
    D = D.with_columns(day_type=pl.col("dt").replace_strict(dtype))
    out = []
    for (nm, dty), sub in D.group_by("var", "day_type"):
        P = sub.pivot(on="dt", index="lag", values="rho").sort("lag")
        M = P.drop("lag").to_numpy()                      # ラグ × 日
        Pn = sub.pivot(on="dt", index="lag", values="rho_null").sort("lag")
        Mn = Pn.drop("lag").to_numpy()
        nd = M.shape[1]
        bs = np.empty((N_BOOT, M.shape[0]))
        for b in range(N_BOOT):
            bs[b] = np.nanmean(M[:, rng.integers(0, nd, nd)], axis=1)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        for i, k in enumerate(P["lag"].to_list()):
            out.append({"var": nm, "day_type": dty, "lag": k, "lag_ms": k * 200,
                        "n_days": nd, "rho": float(np.nanmean(M[i])),
                        "lo": float(lo[i]), "hi": float(hi[i]),
                        "rho_null": float(np.nanmean(Mn[i]))})
    A = pl.DataFrame(out).sort("var", "day_type", "lag")
    A.write_parquet(ROOT / "data" / f"acf_200ms_{tag}.parquet")
    A.write_csv(ROOT / "data" / f"acf_200ms_{tag}.csv")

    print("\n=== 200ms 格子の自己相関(日をまたいだ平均)===", file=sys.stderr)
    show = [1, 5, 25, 50, 150, 300, 1500]
    for nm in ("OBI", "OFI"):
        for dty in ("立会日", "閉場日"):
            s = A.filter((pl.col("var") == nm) & (pl.col("day_type") == dty))
            if s.height == 0:
                continue
            txt = "  ".join(
                f"{k*200/1000:g}s {s.filter(pl.col('lag') == k)['rho'][0]:+.3f}"
                for k in show if s.filter(pl.col("lag") == k).height)
            nl = s.filter(pl.col("lag") == 1)["rho_null"][0]
            print(f"  {nm} {dty}({s['n_days'][0]} 日)  {txt}   帰無 {nl:+.4f}",
                  file=sys.stderr)
    print(f"\n-> data/acf_200ms_{tag}.parquet / .csv", file=sys.stderr)
    print(f"-> data/acf_200ms_daily_{tag}.parquet", file=sys.stderr)
    print(f"-> {outdir}/dt=*.parquet ({len(days)} 日 × {PER_DAY:,} 格子点)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
