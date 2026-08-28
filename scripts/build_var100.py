"""10ms 格子の板の量を 100ms 窓で分散にし、将来 log リターンへ OLS で当てる。

【何を作るか】
10ms の時計に載せた 4 つの量(買い・売りを分ける)

    bid_depth … 最良買い気配の数量(状態。空区間は持ち越し)
    ask_depth … 最良売り気配の数量(同上)
    obi       … (bid_sz − ask_sz) / (bid_sz + ask_sz)(状態)
    ofi       … Cont–Kukanov–Stoikov の板流量(流量。空区間は 0)

を 10 点ずつ(= 100ms)まとめ、その **標本分散**(ddof=1)を特徴量にする。

10 レベルの指数減衰合計 depth は板の全価格帯が要る。l2/book_px は
DEEP_ARCHIVE で読めないうえ 1 秒スナップショットなので 10ms には使えない。
l1 から板を組み直す別のスクリプトで作る(build_var100_l1.py)。

【x が確定する時刻 / y の期間】
窓 j は 10ms 格子の点 10j..10j+9 を含み、最後の点の時刻を T と書く。

    x = その 10 点の分散      … T で確定する(未来の点は入らない)
    y = log mid_{T+h} − log mid_T   … 期間は (T, T+h]

先読みは無い。`shift(-k)` は y にしか使っていない。

【★分散は符号を持たない — 符号つきリターンとの関係は構造的にほぼ 0 になる】
分散が大きいことは「どちらへ動くか」を含まない。したがって符号つきの
log リターンに当てた OLS は、当てはまるほうがおかしい。指示どおり符号つきも
出すが、**意味があるのは絶対値と二乗のほう**なので 3 つとも出して並べる。

    y_signed = log mid_{T+h} − log mid_T
    y_abs    = |y_signed|
    y_sq     = y_signed^2

【★重なる窓は t 値を水増しする】
ホライズン h の窓は隣どうしが重なるので、古典的な t 値は使えない
(book_slope のレポートで最大 6.3 倍の水増しを実測した)。
**日の中で h ごとに 1 つだけ取る重ならない部分標本**を併記し、
判定はそちらで行う。

【★分散は右に大きく歪む】
少数の異常窓が係数を支配しうる(CLAUDE.md の既知の事故型)。
生の分散に加えて log(1 + x) 版も同じ手順で出し、結論が変わらないか見る。

【★空の窓】
10ms 格子の大半は最良気配が動かない。動かない窓では分散が厳密に 0 になり、
これは板の静けさであって欠測ではないが、標本の大半が同じ値 0 だと
回帰の当てはまりは「動いたか否か」の指標に近づく。窓ごとのイベント数
`n_ev` を保存し、集計は **全窓** と **n_ev ≥ 1 の窓** の両方で出す
(結果を見てからの選別ではなく、既知の副作用に対して先に決めた層別)。

    uv run python scripts/build_var100.py --coin xyz:MU
出力: data/var100_<coin>/dt=*.parquet … 100ms 窓の特徴量
      data/var100_ols_<coin>.parquet / .csv … OLS の結果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STEP_NS = 10_000_000                       # 10ms
PER_DAY = 24 * 60 * 60 * 100               # 8,640,000 格子点
K = 10                                     # 100ms = 10ms × 10
NW = PER_DAY // K                          # 864,000 窓
# 予測ホライズン(10ms 格子の歩数)
HOR = {"10ms": 1, "20ms": 2, "50ms": 5, "100ms": 10, "500ms": 50, "1s": 100,
       "3s": 300, "5s": 500, "10s": 1000, "30s": 3000, "50s": 5000, "100s": 10000}
FEATS = ["var_bid_depth", "var_ask_depth", "var_obi", "var_ofi"]
CHUNK = 7


def read_chunk(coin: str, days: list[str]):
    p = ROOT / "data" / f"bbo_{coin.replace(':', '_')}.parquet"
    d = pl.scan_parquet(p).filter(pl.col("dt").is_in(days)).collect().sort("ts")
    n0 = d.height
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    pb, pa = pl.col("best_bid"), pl.col("best_ask")
    qb, qa = pl.col("bid_sz"), pl.col("ask_sz")
    d = d.with_columns(
        mid=(pb + pa) / 2,
        obi=(qb - qa) / (qb + qa),
        ofi=(pl.when(pb >= pb.shift(1)).then(qb).otherwise(0.0)
             - pl.when(pb <= pb.shift(1)).then(qb.shift(1)).otherwise(0.0)
             - pl.when(pa <= pa.shift(1)).then(qa).otherwise(0.0)
             + pl.when(pa >= pa.shift(1)).then(qa.shift(1)).otherwise(0.0)),
    ).with_columns(
        ofi=pl.when(pl.col("dt") == pl.col("dt").shift(1)).then(pl.col("ofi")).otherwise(None)
    )
    return d.select("ts", "dt", "bid_sz", "ask_sz", "mid", "obi", "ofi"), n0 - d.height


def ffill_last(b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """区間の最後の値を置き、空区間は直前を持ち越す(先頭の空白は nan)。"""
    g = np.full(PER_DAY, np.nan)
    g[b] = v                                   # ts 昇順なので最後の代入が残る
    idx = np.where(np.isfinite(g), np.arange(PER_DAY), 0)
    np.maximum.accumulate(idx, out=idx)
    return g[idx]


def to_grid(day: str, e: pl.DataFrame) -> dict[str, np.ndarray]:
    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
    b = ((e["ts"].to_numpy() - t0) // STEP_NS).astype(np.int64)
    ok = (b >= 0) & (b < PER_DAY)
    b = b[ok]
    out = {"n_ev": np.bincount(b, minlength=PER_DAY).astype(np.int32)}
    for c in ("bid_sz", "ask_sz", "mid", "obi"):
        out[c] = ffill_last(b, e[c].to_numpy()[ok])
    out["ofi"] = np.bincount(b, weights=np.nan_to_num(e["ofi"].to_numpy()[ok]),
                             minlength=PER_DAY)
    out["t0"] = t0
    return out


def windows(g: dict[str, np.ndarray]) -> pl.DataFrame:
    """10ms 格子を 100ms 窓へ。分散は ddof=1。"""
    def var(c):
        x = g[c].reshape(NW, K)
        return np.nanvar(x, axis=1, ddof=1)

    def mean(c):
        return np.nanmean(g[c].reshape(NW, K), axis=1)

    ts = g["t0"] + (np.arange(NW, dtype=np.int64) + 1) * (STEP_NS * K)
    return pl.DataFrame({
        "ts": ts,
        "n_ev": g["n_ev"].reshape(NW, K).sum(axis=1).astype(np.int32),
        "var_bid_depth": var("bid_sz"), "var_ask_depth": var("ask_sz"),
        "var_obi": var("obi"), "var_ofi": var("ofi"),
        "mean_bid_depth": mean("bid_sz"), "mean_ask_depth": mean("ask_sz"),
        "mean_obi": mean("obi"),
        "mid_T": g["mid"][K - 1::K],           # 窓の最後の点の mid
    })


class Acc:
    """OLS の十分統計量。全標本と重ならない部分標本を同時に貯める。"""

    def __init__(self):
        self.s = {}

    def add(self, key, x, y, sub):
        for tag, m in (("full", slice(None)), ("sub", sub)):
            xa, ya = x[m], y[m]
            k = (*key, tag)
            a = self.s.get(k)
            v = np.array([len(xa), xa.sum(), ya.sum(), (xa * ya).sum(),
                          (xa * xa).sum(), (ya * ya).sum()], dtype=np.float64)
            self.s[k] = v if a is None else a + v

    def raw(self) -> pl.DataFrame:
        """十分統計量のまま返す。日ごとに保存して後から足せるようにする。"""
        KEY = ("feat", "x_form", "y_kind", "day_type", "hor", "sample")
        SUM = ("n", "sx", "sy", "sxy", "sxx", "syy")
        return pl.DataFrame([dict(zip(KEY, k)) | dict(zip(SUM, v.tolist()))
                             for k, v in self.s.items()])


KEY = ["feat", "x_form", "y_kind", "day_type", "hor", "sample"]


def finalize(S: pl.DataFrame) -> pl.DataFrame:
    """日ごとの十分統計量を足して OLS の推定値にする。"""
    G = S.group_by(KEY).agg(pl.col("n", "sx", "sy", "sxy", "sxx", "syy").sum())
    n = G["n"].to_numpy()
    sx, sy = G["sx"].to_numpy(), G["sy"].to_numpy()
    sxy, sxx, syy = G["sxy"].to_numpy(), G["sxx"].to_numpy(), G["syy"].to_numpy()
    den = n * sxx - sx * sx
    dy = n * syy - sy * sy
    with np.errstate(invalid="ignore", divide="ignore"):
        b1 = (n * sxy - sx * sy) / den
        b0 = (sy - b1 * sx) / n
        r = (n * sxy - sx * sy) / np.sqrt(den * dy)
        sse = np.maximum(syy - b0 * sy - b1 * sxy, 0.0)
        se = np.sqrt(sse / (n - 2) * n / den)
        t = b1 / se
    ok = (n >= 30) & (den > 0) & (dy > 0)
    return (G.with_columns(beta=pl.Series(b1), r=pl.Series(r),
                           r2=pl.Series(r * r), t=pl.Series(t))
             .filter(pl.Series(ok))
             .select(*KEY, "n", "beta", "r", "r2", "t"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い")
    days = sorted(pl.scan_parquet(p).select("dt").unique().collect()["dt"].to_list())
    if a.days:
        days = days[: a.days]
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days}
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    outdir = ROOT / "data" / f"var100_{tag}"
    accdir = ROOT / "data" / f"var100_acc_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    accdir.mkdir(parents=True, exist_ok=True)
    n_drop, zero_share = 0, []
    # ★再開できるようにする。日ごとに十分統計量を書き、済んだ日は飛ばす
    todo = [d for d in days if not (accdir / f"dt={d}.parquet").exists()]
    print(f"[再開] 未処理 {len(todo)} 日 / 済み {len(days) - len(todo)} 日", file=sys.stderr)

    for c0 in range(0, len(todo), CHUNK):
        part = todo[c0:c0 + CHUNK]
        E, nd = read_chunk(a.coin, part)
        n_drop += nd
        for day in part:
            acc = Acc()
            g = to_grid(day, E.filter(pl.col("dt") == day))
            W = windows(g)
            logmid = np.log(np.where(g["mid"] > 0, g["mid"], np.nan))
            base = logmid[K - 1::K]                     # 窓の終端 T の log mid
            act = (W["n_ev"].to_numpy() >= 1)
            zero_share.append({"dt": day, "active": float(act.mean())})

            for hname, h in HOR.items():
                fut = np.full(NW, np.nan)
                avail = logmid[K - 1 + h::K]             # T+h の log mid
                nf = min(NW, len(avail))                 # 未来が取れる窓の数
                fut[:nf] = avail[:nf]
                y = fut - base
                stride = max(1, int(np.ceil(h / K)))     # 重ならない部分標本の間隔
                for feat in FEATS:
                    xv = W[feat].to_numpy()
                    for lay, mask in (("全窓", np.ones(NW, bool)), ("動いた窓", act)):
                        m = mask & np.isfinite(xv) & np.isfinite(y)
                        if m.sum() < 30:
                            continue
                        sub = np.zeros(NW, bool)
                        sub[::stride] = True
                        sub = sub[m]
                        x0 = xv[m]
                        y0 = y[m]
                        for xf, xx in (("生", x0), ("log1p", np.log1p(np.maximum(x0, 0)))):
                            for yk, yy in (("符号つき", y0), ("絶対値", np.abs(y0)),
                                           ("二乗", y0 * y0)):
                                acc.add((f"{feat}|{lay}", xf, yk, dtype[day], hname),
                                        xx, yy, sub)
            W.with_columns(pl.col(pl.Float64).exclude("mid_T").cast(pl.Float32)) \
             .with_columns(dt=pl.lit(day)) \
             .write_parquet(outdir / f"dt={day}.parquet", compression="zstd")
            acc.raw().write_parquet(accdir / f"dt={day}.parquet")
        print(f"  {part[-1]}  ({c0 + len(part)}/{len(todo)} 日)", file=sys.stderr)

    S = pl.read_parquet(accdir / "dt=*.parquet")
    R = finalize(S).with_columns(
        lay=pl.col("feat").str.split("|").list.get(1),
        feat=pl.col("feat").str.split("|").list.get(0),
        hor_ms=pl.col("hor").replace_strict({k: v * 10 for k, v in HOR.items()},
                                            return_dtype=pl.Int64),
    ).sort("feat", "lay", "x_form", "y_kind", "day_type", "hor_ms", "sample")
    R.write_parquet(ROOT / "data" / f"var100_ols_{tag}.parquet")
    R.write_csv(ROOT / "data" / f"var100_ols_{tag}.csv")
    Z = pl.DataFrame(zero_share)
    print(f"[窓] イベントのある窓の割合 中央 {Z['active'].median():.1%} / "
          f"最小 {Z['active'].min():.1%} / 最大 {Z['active'].max():.1%}", file=sys.stderr)
    print(f"[除去] クロス・数量 0 の行 {n_drop:,}", file=sys.stderr)
    print(f"[OLS] {R.height:,} 行", file=sys.stderr)
    print(f"\n-> data/var100_ols_{tag}.parquet / .csv\n-> {outdir}/dt=*.parquet",
          file=sys.stderr)


if __name__ == "__main__":
    main()
