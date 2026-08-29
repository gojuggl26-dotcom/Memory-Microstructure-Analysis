"""板の弾力性(elasticity)— mid から離れるにつれて厚みがどう増えるかを測る。

【定義】
mid から距離 δ(bp)以内に積まれている累積数量を Q(δ) とする。

    Q(δ) ∝ δ^ε      ⟺      ε = d ln Q / d ln δ

この ε を **liquidity elasticity** と呼ぶ。ε = 1 なら板の密度が距離に対して一様、
ε > 1 なら遠いほど厚く(外側に積まれる)、ε < 1 なら最良気配の近くに集中している。
推定は ln δ に対する ln Q の OLS の傾き。

    liquidity elasticity   買い + 売りの合計に対する ε(1〜100 bp)
    bid elasticity         買い側だけの ε
    ask elasticity         売り側だけの ε
    elasticity asymmetry   (ε_bid − ε_ask) / (ε_bid + ε_ask)
    local elasticity       近い側だけで測った ε(1〜10 bp)
    deep-book elasticity   遠い側だけで測った ε(25〜100 bp)

**local と deep を分けるのは、板が単一のべき乗則に従う保証が無いため。** 1 本の
直線で全体を要約すると、内側と外側で傾きが違うときにどちらでもない値になる。

【なぜ l1 から板を組み直すか】
距離帯ごとの厚みは `l2/book_bp` にそのまま入っているが、バケット上で
**DEEP_ARCHIVE に移行済みで読めない**(復元に 12〜48 時間)。l1 は GLACIER_IR で
取得済みなので、open と終端イベントから板を組み直す。
価格は 0.01 刻みなので**ティック番号の密な配列**で持つ。100 bp は mid の 1% で、
$200 なら 200 ティック分しかないので、断面ごとの累積和は窓を切れば安い。

【★板に入れてはいけない注文(build_fill_rate.py と同じ)】
トリガー注文は発火するまで板に載らない。テイカー(Ioc / FrontendMarket /
LiquidationMarket)は板に留まらない。どちらも除く。

【★既知の精度の限界】
この列構成には部分約定の途中経過が無いので、注文の生存中は発注数量(orig_sz)の
まま板に置いている。部分約定した注文の分だけ厚みをわずかに過大に見る。

【x が確定する時刻】
断面は 1 秒ごと。各断面の ε はその時刻までの板だけで決まる。将来の値は入らない。
リターンとの関係を見るのは別のスクリプト(analyze_elasticity.py)。

    uv run python scripts/build_elasticity.py --coin xyz:MU
出力: data/elast_<coin>/dt=*.parquet … 1 秒ごとの断面と 6 つの弾力性
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.01
SNAP_NS = 1_000_000_000                  # 1 秒ごとの断面
PER_DAY = 24 * 60 * 60
BPS = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])   # mid からの距離
LOCAL = BPS <= 10.0                      # local elasticity に使う範囲
DEEP = BPS >= 25.0                       # deep-book elasticity に使う範囲
EPS_Q = 1e-9    # ★数量の残差。0 と見なす閾値(下のコメント参照)
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}


def slope(lx: np.ndarray, ly: np.ndarray, use: np.ndarray) -> np.ndarray:
    """ln δ に対する ln Q の OLS の傾き。ly は (断面, δ) の行列。

    Q = 0 の点は対数が取れないので落とす。使える点が 2 未満なら nan。
    """
    m = np.isfinite(ly) & use[None, :]
    n = m.sum(axis=1)
    x = np.where(m, lx[None, :], 0.0)
    y = np.where(m, ly, 0.0)
    sx, sy = x.sum(1), y.sum(1)
    sxx, sxy = (x * x).sum(1), (x * y).sum(1)
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (n * sxy - sx * sy) / den
    return np.where((n >= 2) & (den > 0), b, np.nan)


def day_snapshots(fp: Path, day: str) -> pl.DataFrame:
    d = pl.read_parquet(fp).sort("ts")
    ts = d["ts"].to_numpy()
    isbid = (d["side"].to_numpy() == "B")
    px = np.rint(d["px"].to_numpy() / TICK).astype(np.int64)
    st = d["status"].to_numpy()
    osz = d["orig_sz"].to_numpy()
    trg = d["is_trigger"].to_numpy()
    tif = d["tif"].to_numpy()
    oid = d["oid"].to_numpy()

    lo_t, hi_t = int(px.min()) - 2, int(px.max()) + 2
    n_t = hi_t - lo_t + 1
    if n_t > 5_000_000:
        sys.exit(f"{day}: 価格の幅が広すぎる({n_t:,} ティック)")
    depb = np.zeros(n_t)
    depa = np.zeros(n_t)
    live: dict[int, tuple] = {}
    hb: list[int] = []          # bid ティックの最大ヒープ(負で持つ)
    ha: list[int] = []
    inb: set[int] = set()
    ina: set[int] = set()

    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
    snap_t = t0 + (np.arange(PER_DAY, dtype=np.int64) + 1) * SNAP_NS
    # 断面の直前までイベントを流すための境界
    cut = np.searchsorted(ts, snap_t, side="right")

    nb = len(BPS)
    QB = np.full((PER_DAY, nb), np.nan)
    QA = np.full((PER_DAY, nb), np.nan)
    MID = np.full(PER_DAY, np.nan)
    SPR = np.full(PER_DAY, np.nan)

    # ★浮動小数の残差を 0 と見なさないと板が壊れる。
    #   足し引きを繰り返すと数量に 1e-15 程度の残りが出る。これを「生きている
    #   水準」と判定すると、消えたはずの価格が板に残り続け、再構成した板が
    #   ほぼ常時クロスする(実測で 86,400 断面中 86,286 がクロスした)。
    def best_bid():
        while hb:
            t = -hb[0]
            if depb[t - lo_t] > EPS_Q:
                return t
            heapq.heappop(hb); inb.discard(t)
        return None

    def best_ask():
        while ha:
            t = ha[0]
            if depa[t - lo_t] > EPS_Q:
                return t
            heapq.heappop(ha); ina.discard(t)
        return None

    diag = [0, 0, 0, 0]      # 片側なし / クロス / 最大クロス幅 / 正常
    i = 0
    for s_i in range(PER_DAY):
        end = cut[s_i]
        while i < end:
            s = st[i]
            if s == "open":
                if trg[i] or tif[i] not in RESTING:
                    i += 1
                    continue
                b, t = bool(isbid[i]), int(px[i])
                dep = depb if b else depa
                if dep[t - lo_t] <= EPS_Q:
                    if b:
                        if t not in inb:
                            heapq.heappush(hb, -t); inb.add(t)
                    elif t not in ina:
                        heapq.heappush(ha, t); ina.add(t)
                dep[t - lo_t] += float(osz[i])
                live[int(oid[i])] = (b, t, float(osz[i]))
            elif s == "filled" or s in CANCEL:
                o = live.pop(int(oid[i]), None)
                if o is not None:
                    b, t, q = o
                    dep = depb if b else depa
                    v = dep[t - lo_t] - q
                    dep[t - lo_t] = v if v > EPS_Q else 0.0
            i += 1

        bb, ba = best_bid(), best_ask()
        if bb is None or ba is None:
            diag[0] += 1
            continue
        if ba <= bb:
            diag[1] += 1
            diag[2] = max(diag[2], bb - ba)
            continue
        diag[3] += 1
        mid = (bb + ba) / 2.0
        MID[s_i] = mid * TICK
        SPR[s_i] = (ba - bb) / mid * 1e4
        # 100 bp = mid の 1%。窓を切って累積和を取る
        w = int(np.ceil(mid * BPS[-1] / 1e4)) + 1
        b_lo = max(bb - w - lo_t, 0)
        cb = np.cumsum(depb[b_lo:bb - lo_t + 1][::-1])          # bb から下へ累積
        a_hi = min(ba + w - lo_t, n_t - 1)
        ca = np.cumsum(depa[ba - lo_t:a_hi + 1])                 # ba から上へ累積
        # δ bp 以内に入る「最良気配から何ティック分か」。
        # mid から δ の距離は mid ± mid·δ/1e4 なので、最良気配からは半スプレッド分だけ近い
        half = (ba - bb) / 2.0
        kk = np.floor(mid * BPS / 1e4 - half).astype(np.int64)
        mb = np.minimum(kk, len(cb) - 1)
        ma = np.minimum(kk, len(ca) - 1)
        QB[s_i] = np.where(mb >= 0, cb[np.maximum(mb, 0)], 0.0)
        QA[s_i] = np.where(ma >= 0, ca[np.maximum(ma, 0)], 0.0)

    lx = np.log(BPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        lb = np.where(QB > 0, np.log(QB), np.nan)
        la = np.where(QA > 0, np.log(QA), np.nan)
        lt = np.where((QB + QA) > 0, np.log(QB + QA), np.nan)
    all_ = np.ones(len(BPS), bool)
    e_bid = slope(lx, lb, all_)
    e_ask = slope(lx, la, all_)
    out = {
        "ts": snap_t, "dt": [day] * PER_DAY, "mid": MID, "spread_bp": SPR,
        "eps_liq": slope(lx, lt, all_),
        "eps_bid": e_bid, "eps_ask": e_ask,
        "eps_asym": (e_bid - e_ask) / (e_bid + e_ask),
        "eps_local": slope(lx, lt, LOCAL),
        "eps_deep": slope(lx, lt, DEEP),
    }
    for j, b in enumerate(BPS):
        out[f"qb_{b:g}"] = QB[:, j]
        out[f"qa_{b:g}"] = QA[:, j]
    print(f"    [検算] 片側なし {diag[0]:,} / クロス {diag[1]:,}"
          f"(最大 {diag[2]} ティック) / 正常 {diag[3]:,}", file=sys.stderr)
    return pl.DataFrame(out).filter(pl.col("mid").is_not_nan())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    l1dir = ROOT / "data" / f"l1_{tag}"
    days = sorted(p.stem.split("=")[1] for p in l1dir.glob("dt=*.parquet"))
    if a.days:
        days = days[: a.days]
    outdir = ROOT / "data" / f"elast_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (outdir / f"dt={d}.parquet").exists()]
    print(f"[日] 全 {len(days)} / 未処理 {len(todo)}", file=sys.stderr)

    for n, day in enumerate(todo, 1):
        S = day_snapshots(l1dir / f"dt={day}.parquet", day)
        S.with_columns(pl.col(pl.Float64).exclude("mid").cast(pl.Float32)) \
         .write_parquet(outdir / f"dt={day}.parquet", compression="zstd")
        e = S.select(pl.col("eps_liq").median(), pl.col("eps_local").median(),
                     pl.col("eps_deep").median()).row(0)
        print(f"  {day}  断面 {S.height:,}  ε={e[0]:.3f} local={e[1]:.3f} deep={e[2]:.3f}"
              f"  ({n}/{len(todo)})", file=sys.stderr)
    print(f"-> {outdir}/dt=*.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
