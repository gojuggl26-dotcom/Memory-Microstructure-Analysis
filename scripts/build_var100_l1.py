"""l1 から板を組み直し、上位 10 レベルの指数減衰合計 depth を 10ms 格子で作る。

【なぜ l1 から組むか】
上位 10 レベルの depth には板の全価格帯が要る。バケットの `l2/book_px` は
**DEEP_ARCHIVE で読めない**うえ、読めたとしても **1 秒スナップショット**なので
10ms の格子には使えない。l1(注文イベント)は GLACIER_IR で取得済みなので、
open と終端イベントから板を組み直す。

【指数減衰の重み】
上から i 番目(i = 1..10)の価格水準の数量 q_i に

    w_i = exp(−λ (i − 1)),   λ = ln 2 / 3   → 3 レベルごとに半分、10 番目は 0.125

を掛けて足す。λ は分布を見る前に機械的に決めた(結果を見てから選ばない)。
比較のため重みなしの 10 レベル合計も同時に出す。

【★板に入れてはいけない注文(build_fill_rate.py と同じ)】
トリガー注文は発火するまで板に載らない。テイカー(Ioc / FrontendMarket /
LiquidationMarket)は板に留まらない。どちらも除く。

【★既知の精度の限界】
この列構成には部分約定の途中経過が無いので、注文の生存中は発注数量
(orig_sz)のまま板に置いている。部分約定した注文の分だけ depth を
わずかに過大に見る。約定する注文は全体の 2.1% なので影響は小さいが、
ゼロではない。

【計算量の工夫】
上位 10 レベルは、触られた価格が現在の 10 番目より内側のときだけ計算し直す。
外側の価格が動いても上位 10 レベルは変わらない(10 番目自身が消える場合は
その価格が「10 番目以内」なので拾える)。実測では注文の 79% が最良気配から
50 ティック以上離れた場所に置かれるので、大半は再計算を飛ばせる。

【x が確定する時刻 / y の期間】
build_var100.py と同じ。窓 j の最後の 10ms 点の時刻を T として

    x = その 100ms 窓の 10 点の標本分散 … T で確定
    y = log mid_{T+h} − log mid_T      … 期間は (T, T+h]

mid は l2/bbo から取る(l1 の板の最良気配ではなく、他の分析と同じ定義に揃える)。

    uv run python scripts/build_var100_l1.py --coin xyz:MU
出力: data/var100l1_<coin>/dt=*.parquet … 100ms 窓の特徴量
      data/var100_l1_ols_<coin>.parquet / .csv … OLS の結果
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_var100 import (HOR, K, NW, PER_DAY, STEP_NS, Acc, finalize,  # noqa: E402
                          read_chunk)

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.01
NLV = 10
LAM = np.log(2.0) / 3.0
W = np.exp(-LAM * np.arange(NLV))
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}
FEATS = ["var_bid_exp10", "var_ask_exp10", "var_bid_sum10", "var_ask_sum10"]


def top10(heap, in_h, dep, sign):
    """有効な上位 10 水準の (価格, 数量) を返す。遅延削除つき。"""
    got, popped = [], []
    while heap and len(got) < NLV:
        p = heapq.heappop(heap)
        pr = -p if sign < 0 else p
        q = dep.get(pr, 0.0)
        if q > 0:
            got.append((pr, q))
            popped.append(p)
        else:
            in_h.discard(pr)
    for p in popped:
        heapq.heappush(heap, p)
    return got


def day_series(fp: Path):
    """その日の l1 を舐めて、上位 10 レベルの指標が変わった時刻だけ返す。"""
    d = pl.read_parquet(fp).sort("ts")
    ts = d["ts"].to_numpy()
    isbid = (d["side"].to_numpy() == "B")
    px = np.rint(d["px"].to_numpy() / TICK).astype(np.int64)
    st = d["status"].to_numpy()
    osz = d["orig_sz"].to_numpy()
    trg = d["is_trigger"].to_numpy()
    tif = d["tif"].to_numpy()

    live: dict[int, tuple] = {}
    depb: dict[int, float] = {}
    depa: dict[int, float] = {}
    hb: list[int] = []      # bid 価格の最大ヒープ(負で持つ)
    ha: list[int] = []
    inb: set[int] = set()
    ina: set[int] = set()
    oid = d["oid"].to_numpy()

    out_t, out_v = [], []
    cur = np.zeros(4)                 # bid_exp, ask_exp, bid_sum, ask_sum
    tenth_b, tenth_a = None, None     # 10 番目の価格(内側判定に使う)

    for i in range(len(st)):
        s = st[i]
        b = bool(isbid[i])
        p = int(px[i])
        if s == "open":
            if trg[i] or tif[i] not in RESTING:
                continue
            dep = depb if b else depa
            if dep.get(p, 0.0) <= 0:
                if b:
                    if p not in inb:
                        heapq.heappush(hb, -p); inb.add(p)
                elif p not in ina:
                    heapq.heappush(ha, p); ina.add(p)
            dep[p] = dep.get(p, 0.0) + float(osz[i])
            live[int(oid[i])] = (b, p, float(osz[i]))
        elif s == "filled" or s in CANCEL:
            o = live.pop(int(oid[i]), None)
            if o is None:
                continue
            b, p, q = o
            dep = depb if b else depa
            dep[p] = dep.get(p, 0.0) - q
            if dep[p] <= 1e-12:
                dep.pop(p, None)
        else:
            continue

        # 上位 10 レベルの外側なら再計算しない
        if b:
            if tenth_b is not None and p < tenth_b:
                continue
        else:
            if tenth_a is not None and p > tenth_a:
                continue

        gb = top10(hb, inb, depb, -1)
        ga = top10(ha, ina, depa, +1)
        tenth_b = gb[-1][0] if len(gb) == NLV else None
        tenth_a = ga[-1][0] if len(ga) == NLV else None
        qb = np.array([q for _, q in gb])
        qa = np.array([q for _, q in ga])
        v = np.array([float(W[:len(qb)] @ qb), float(W[:len(qa)] @ qa),
                      float(qb.sum()), float(qa.sum())])
        if not np.array_equal(v, cur):
            out_t.append(ts[i]); out_v.append(v); cur = v
    return np.asarray(out_t, dtype=np.int64), np.asarray(out_v, dtype=np.float64)


def to_grid(day: str, t: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """変化点を 10ms 格子へ。状態量なので最後の値を置いて持ち越す。"""
    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
    b = ((t - t0) // STEP_NS).astype(np.int64)
    ok = (b >= 0) & (b < PER_DAY)
    b, v = b[ok], v[ok]
    g = {"t0": t0, "n_ev": np.bincount(b, minlength=PER_DAY).astype(np.int32)}
    for j, nm in enumerate(("bid_exp10", "ask_exp10", "bid_sum10", "ask_sum10")):
        a = np.full(PER_DAY, np.nan)
        a[b] = v[:, j]
        idx = np.where(np.isfinite(a), np.arange(PER_DAY), 0)
        np.maximum.accumulate(idx, out=idx)
        g[nm] = a[idx]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    l1dir = ROOT / "data" / f"l1_{tag}"
    days = sorted(p.stem.split("=")[1] for p in l1dir.glob("dt=*.parquet"))
    bbo_days = set(pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
                   .select("dt").unique().collect()["dt"].to_list())
    days = [d for d in days if d in bbo_days]
    if a.days:
        days = days[: a.days]
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days}
    print(f"[日] {len(days)} 日(l1 と bbo が両方ある日)", file=sys.stderr)

    outdir = ROOT / "data" / f"var100l1_{tag}"
    accdir = ROOT / "data" / f"var100l1_acc_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    accdir.mkdir(parents=True, exist_ok=True)
    # ★1 日 60 秒かかるので再開できるようにする。済んだ日は飛ばす
    todo = [d for d in days if not (accdir / f"dt={d}.parquet").exists()]
    print(f"[再開] 未処理 {len(todo)} 日 / 済み {len(days) - len(todo)} 日", file=sys.stderr)

    for n, day in enumerate(todo, 1):
        acc = Acc()
        t, v = day_series(l1dir / f"dt={day}.parquet")
        if len(t) == 0:
            print(f"  {day} 変化点なし — 飛ばす", file=sys.stderr)
            continue
        g = to_grid(day, t, v)
        E, _ = read_chunk(a.coin, [day])
        mg = np.full(PER_DAY, np.nan)
        bb = ((E["ts"].to_numpy() - g["t0"]) // STEP_NS).astype(np.int64)
        okb = (bb >= 0) & (bb < PER_DAY)
        mg[bb[okb]] = E["mid"].to_numpy()[okb]
        idx = np.where(np.isfinite(mg), np.arange(PER_DAY), 0)
        np.maximum.accumulate(idx, out=idx)
        mg = mg[idx]

        cols = {"ts": g["t0"] + (np.arange(NW, dtype=np.int64) + 1) * (STEP_NS * K),
                "n_ev": g["n_ev"].reshape(NW, K).sum(axis=1).astype(np.int32)}
        for src, nm in (("bid_exp10", "var_bid_exp10"), ("ask_exp10", "var_ask_exp10"),
                        ("bid_sum10", "var_bid_sum10"), ("ask_sum10", "var_ask_sum10")):
            cols[nm] = np.nanvar(g[src].reshape(NW, K), axis=1, ddof=1)
            cols["mean_" + src] = np.nanmean(g[src].reshape(NW, K), axis=1)
        Wd = pl.DataFrame(cols)

        logmid = np.log(np.where(mg > 0, mg, np.nan))
        base = logmid[K - 1::K]
        act = Wd["n_ev"].to_numpy() >= 1
        for hname, h in HOR.items():
            fut = np.full(NW, np.nan)
            avail = logmid[K - 1 + h::K]
            nf = min(NW, len(avail))
            fut[:nf] = avail[:nf]
            y = fut - base
            stride = max(1, int(np.ceil(h / K)))
            for feat in FEATS:
                xv = Wd[feat].to_numpy()
                for lay, mask in (("全窓", np.ones(NW, bool)), ("動いた窓", act)):
                    m = mask & np.isfinite(xv) & np.isfinite(y)
                    if m.sum() < 30:
                        continue
                    sub = np.zeros(NW, bool)
                    sub[::stride] = True
                    sub = sub[m]
                    x0, y0 = xv[m], y[m]
                    for xf, xx in (("生", x0), ("log1p", np.log1p(np.maximum(x0, 0)))):
                        for yk, yy in (("符号つき", y0), ("絶対値", np.abs(y0)),
                                       ("二乗", y0 * y0)):
                            acc.add((f"{feat}|{lay}", xf, yk, dtype[day], hname),
                                    xx, yy, sub)
        Wd.with_columns(pl.col(pl.Float64).cast(pl.Float32), dt=pl.lit(day)) \
          .write_parquet(outdir / f"dt={day}.parquet", compression="zstd")
        print(f"  {day}  変化点 {len(t):,}  ({n}/{len(days)} 日)", file=sys.stderr)

    R = acc.rows().with_columns(
        lay=pl.col("feat").str.split("|").list.get(1),
        feat=pl.col("feat").str.split("|").list.get(0),
        hor_ms=pl.col("hor").replace_strict({k: v * 10 for k, v in HOR.items()},
                                            return_dtype=pl.Int64),
    ).sort("feat", "lay", "x_form", "y_kind", "day_type", "hor_ms", "sample")
    R.write_parquet(ROOT / "data" / f"var100_l1_ols_{tag}.parquet")
    R.write_csv(ROOT / "data" / f"var100_l1_ols_{tag}.csv")
    print(f"\n[OLS] {R.height:,} 行\n-> data/var100_l1_ols_{tag}.parquet / .csv\n"
          f"-> {outdir}/dt=*.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
