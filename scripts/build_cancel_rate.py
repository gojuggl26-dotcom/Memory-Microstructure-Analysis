"""キャンセル率 CR と キャンセル不均衡 CI を 100ms 刻みで作り、その傾きで将来の
log リターンを説明できるかを検証する。

【定義】

```
CR_ask(t) = [t-Δ, t] のキャンセルされた ask 数量 / 同じ窓の平均 ask 深さ
CR_bid(t) = [t-Δ, t] のキャンセルされた bid 数量 / 同じ窓の平均 bid 深さ
CI(t)     = (CR_ask - CR_bid) / (CR_ask + CR_bid + ε)
```

【★キャンセル数量をどう得たか(重要な制約)】
本来は l2/lifecycle(注文 1 本 1 行、終端が cancelled のもの)を使うべきだが、
**このバケットの lifecycle / book_bp / book_px は DEEP_ARCHIVE に移行済みで
直接読めない**(復元に 12〜48 時間と追加費用)。そこで手元にある

    l2/bbo  … 最良気配の価格と数量(イベントごと)
    fills   … 約定(テイカー側の行で向きが決まる)

の 2 つから、**最良気配で起きたキャンセルだけ**を復元した。連続する 2 つの
bbo 行の間で、

    価格が同じ         … 数量の減少分のうち、約定で説明できない分をキャンセルとする
    気配が不利に動いた … その水準が丸ごと消えたので、約定分を引いた残りをキャンセル
    気配が有利に動いた … 内側に新しい注文が入っただけ。キャンセルは 0 とする

したがって本スクリプトの CR は **「最良気配のキャンセル率」** であって、
板の奥のキャンセルは含まない。式の一般形とは範囲が違う点を明記しておく。
分子(キャンセル)も分母(深さ)もどちらも最良気配なので、比としては整合している。

さらに次の 2 つを取り逃す:
- 同じ価格での「取消して出し直し」が 1 区間内で相殺されると見えない(過小評価)
- 気配が有利に動いた区間では、消えた古い水準の内訳が分からない

【x が確定する時刻 / y の期間】
    x = CI の直近 m 点(既定 10 点 = 1 秒)に当てた OLS の傾き
        … CI(t) は [t-Δ, t] の情報だけで決まり、傾きは CI(t-(m-1)δ) … CI(t) から
          決まる。すべて時刻 t までに確定する
    y = log(mid(t+h) / mid(t))   … 期間は (t, t+h]
先読みは無い。

【★判定は 0 ではなく「何もしない場合」と比べる】
標本期間中 mid は上昇しているので、無条件の平均 log リターンは正である。
「傾きが正のとき y > 0」は、それだけでは何もしなくても成り立ってしまう。
よって
    (a) 何もしない場合(無条件)の平均
    (b) 傾きが正のときの平均
    (c) 傾きが負のときの平均
を並べ、**(b) − (a)** と **(b) − (c)** で判定する。

【不確かさ】窓が重なるので日単位のブロックブートストラップ(400 反復)。
【帰無対照】傾きを日内で巡回シフトして同じ集計をする。

    uv run python scripts/build_cancel_rate.py --coin xyz:MU
出力: data/cancel_rate_daily_<coin>.parquet  … 日 × 傾きの符号 × ホライズン
      data/cancel_rate_bins_<coin>.parquet   … 傾きの十分位 × ホライズン
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
GRID_NS = 100_000_000                      # 100ms
HORIZONS_MS = [100, 300, 500, 1_000, 5_000, 10_000, 30_000, 60_000]
EPS = 1e-9
N_BOOT = 400
SEED = 20260828
NDEC = 10                                  # 傾きの十分位


def cancels_for_day(bb: pl.DataFrame, fl: pl.DataFrame) -> dict[str, np.ndarray]:
    """bbo と fills から、区間ごとの最良気配キャンセル数量を復元する。"""
    ts = bb["ts"].to_numpy()
    pa, pb = bb["best_ask"].to_numpy(), bb["best_bid"].to_numpy()
    qa, qb = bb["ask_sz"].to_numpy(), bb["bid_sz"].to_numpy()

    # 約定をテイカーの向きで分け、bbo の区間 (ts[i], ts[i+1]] に割り当てる。
    # fills はミリ秒精度で板イベントより最大 ~1ms 早いことがあるが、
    # 100ms 格子に対しては十分小さい(CLAUDE.md の既知の注意点)。
    ft = fl["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    fsz, fbuy = fl["sz"].to_numpy(), (fl["side"].to_numpy() == "B")
    # searchsorted(side="left") - 1 で「その約定を含む区間の左端」を得る
    j = np.searchsorted(ts, ft, side="left") - 1
    ok = (j >= 0) & (j < len(ts) - 1)
    n = len(ts)
    fill_a = np.zeros(n); fill_b = np.zeros(n)
    np.add.at(fill_a, j[ok & fbuy], fsz[ok & fbuy])    # テイカー買い → ask を消費
    np.add.at(fill_b, j[ok & ~fbuy], fsz[ok & ~fbuy])  # テイカー売り → bid を消費

    # --- ask 側 ---
    same = pa[1:] == pa[:-1]
    worse = pa[1:] > pa[:-1]                # 気配が上へ = 不利に動いた
    drop = np.where(same, qa[:-1] - qa[1:], np.where(worse, qa[:-1], 0.0))
    can_a = np.maximum(drop - fill_a[:-1], 0.0)
    # --- bid 側 ---
    same = pb[1:] == pb[:-1]
    worse = pb[1:] < pb[:-1]                # 気配が下へ = 不利に動いた
    drop = np.where(same, qb[:-1] - qb[1:], np.where(worse, qb[:-1], 0.0))
    can_b = np.maximum(drop - fill_b[:-1], 0.0)
    # ★can_a[i] は区間 i→i+1 のキャンセルで、qa[i+1] を使って初めて計算できる。
    #   つまり **時刻 ts[i+1] にならないと観測できない**。ts[i] のバケットに
    #   入れると未来の情報が説明変数へ混入する。観測できる側の時刻を返す。
    return {"ts": ts, "can_ts": ts[1:], "can_a": can_a, "can_b": can_b,
            "qa": qa, "qb": qb, "mid": (pa + pb) / 2}


def grid_day(c: dict, delta_ms: int, m: int) -> dict[str, np.ndarray] | None:
    """100ms 格子に載せ、CR / CI / 傾き / 将来 log リターンを返す。"""
    ts = c["ts"]
    t0 = (ts[0] // GRID_NS) * GRID_NS
    t1 = (ts[-1] // GRID_NS) * GRID_NS
    ng = int((t1 - t0) // GRID_NS) + 1
    if ng < 10_000:
        return None
    gi = ((ts - t0) // GRID_NS).astype(np.int64)
    gi = np.clip(gi, 0, ng - 1)

    # キャンセル数量は「観測できた時刻」のバケットに足し込む(先読み防止)
    gc = np.clip(((c["can_ts"] - t0) // GRID_NS).astype(np.int64), 0, ng - 1)
    ca = np.zeros(ng); cb = np.zeros(ng)
    np.add.at(ca, gc, c["can_a"]); np.add.at(cb, gc, c["can_b"])

    # 深さと mid は「その格子時刻で最後に観測された値」= 後ろ向き asof
    last = np.full(ng, -1, dtype=np.int64)
    last[gi] = np.arange(len(ts))
    np.maximum.accumulate(last, out=last)
    valid = last >= 0
    if valid.sum() < 10_000:
        return None
    idx = np.where(valid, last, 0)
    qa, qb, mid = c["qa"][idx], c["qb"][idx], c["mid"][idx]

    # [t-Δ, t] の移動和 / 移動平均(W バケット)
    W = max(int(delta_ms // 100), 1)
    def roll_sum(v):
        cs = np.concatenate([[0.0], np.cumsum(v)])
        out = np.full(len(v), np.nan)
        out[W - 1:] = cs[W:] - cs[:-W]
        return out
    sa, sb = roll_sum(ca), roll_sum(cb)
    da, db = roll_sum(qa) / W, roll_sum(qb) / W

    with np.errstate(divide="ignore", invalid="ignore"):
        cra = np.where(da > 0, sa / da, np.nan)
        crb = np.where(db > 0, sb / db, np.nan)
    ci = (cra - crb) / (cra + crb + EPS)

    # 直近 m 点に当てた OLS の傾き = 線形カーネルとの畳み込み
    w = np.arange(m) - (m - 1) / 2.0
    w = w / (w ** 2).sum()
    slope = np.full(ng, np.nan)
    cif = np.where(np.isfinite(ci), ci, 0.0)
    good = np.convolve(np.isfinite(ci).astype(float), np.ones(m), mode="valid")
    conv = np.convolve(cif, w[::-1], mode="valid")
    slope[m - 1:] = np.where(good == m, conv, np.nan)

    lm = np.where(mid > 0, np.log(mid), np.nan)
    rets = {}
    for h in HORIZONS_MS:
        s = h // 100
        r = np.full(ng, np.nan)
        r[:-s] = lm[s:] - lm[:-s]
        rets[h] = r
    return {"slope": slope, "ci": ci, "cra": cra, "crb": crb,
            "valid": valid, "rets": rets}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--delta-ms", type=int, default=1000, help="キャンセル率の平均窓 Δ")
    ap.add_argument("--m", type=int, default=10, help="傾きを当てる点数(100ms 刻み)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bbo = pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet").sort("ts")
    bbo = bbo.filter((pl.col("best_ask") > pl.col("best_bid"))
                     & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0))
    fills = pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                            columns=["ts", "sz", "side", "crossed", "dt"]
                            ).filter(pl.col("crossed")).sort("ts")
    days = sorted(bbo["dt"].unique().to_list())
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    rng = np.random.default_rng(SEED)

    rows, brows = [], []
    for day in days:
        bb = bbo.filter(pl.col("dt") == day)
        fl = fills.filter(pl.col("dt") == day)
        if bb.height < 5_000:
            continue
        g = grid_day(cancels_for_day(bb, fl), a.delta_ms, a.m)
        if g is None:
            continue
        dtl = "立会日" if day in sess else "閉場日"
        sl = g["slope"]
        # 帰無対照: 傾きだけを日内で巡回シフト(y との対応を壊す)
        fin = np.isfinite(sl)
        if fin.sum() < 1000:
            continue
        sh = int(rng.integers(len(sl) // 5, 4 * len(sl) // 5))
        sl_pl = np.roll(sl, sh)
        # 十分位の境目はその日の中で決める(全標本から作らない)
        q = np.quantile(sl[fin], np.linspace(0, 1, NDEC + 1)[1:-1])
        for h in HORIZONS_MS:
            y = g["rets"][h]
            m0 = fin & np.isfinite(y)
            if m0.sum() < 1000:
                continue
            ys = y[m0]; ss = sl[m0]; sp = sl_pl[m0]
            rows.append({
                "dt": day, "day_type": dtl, "h_ms": h, "n": int(m0.sum()),
                "sum_all": float(ys.sum()),
                "n_pos": int((ss > 0).sum()), "sum_pos": float(ys[ss > 0].sum()),
                "up_pos": int((ys[ss > 0] > 0).sum()),
                "n_neg": int((ss < 0).sum()), "sum_neg": float(ys[ss < 0].sum()),
                "up_neg": int((ys[ss < 0] > 0).sum()),
                "n_up_all": int((ys > 0).sum()),
                "n_pos_pl": int((sp > 0).sum()), "sum_pos_pl": float(ys[sp > 0].sum()),
            })
            b = np.digitize(ss, q)
            for i in range(NDEC):
                s2 = b == i
                if s2.sum() == 0:
                    continue
                brows.append({"dt": day, "day_type": dtl, "h_ms": h, "dec": i,
                              "n": int(s2.sum()), "sum": float(ys[s2].sum()),
                              "n_up": int((ys[s2] > 0).sum())})
        print(f"  {day} {dtl} 格子 {int(fin.sum()):,}", file=sys.stderr)

    D = pl.DataFrame(rows); B = pl.DataFrame(brows)
    D.write_parquet(ROOT / "data" / f"cancel_rate_daily_{tag}.parquet")
    B.write_parquet(ROOT / "data" / f"cancel_rate_bins_{tag}.parquet")

    # ---- 集計とブートストラップ ---------------------------------------------
    print(f"\nΔ={a.delta_ms}ms / 傾きは直近 {a.m} 点(={a.m*100}ms)の OLS", file=sys.stderr)
    out = []
    for dtl in ["立会日", "閉場日"]:
        s = D.filter(pl.col("day_type") == dtl)
        dl = sorted(s["dt"].unique().to_list())
        ix = {v: i for i, v in enumerate(dl)}
        print(f"\n=== {dtl}({len(dl)} 日) 平均 log リターン[bp] ===", file=sys.stderr)
        print(f"{'h':>8}{'何もしない':>12}{'傾き>0':>12}{'傾き<0':>12}"
              f"{'(b)-(a)':>11}{'95%区間':>18}{'帰無':>10}", file=sys.stderr)
        for h in HORIZONS_MS:
            t = s.filter(pl.col("h_ms") == h)
            if t.height == 0:
                continue
            A = np.zeros((len(dl), 6))
            for r in t.iter_rows(named=True):
                i = ix[r["dt"]]
                A[i] += [r["n"], r["sum_all"], r["n_pos"], r["sum_pos"],
                         r["n_pos_pl"], r["sum_pos_pl"]]
            neg_n = t["n_neg"].sum(); neg_s = t["sum_neg"].sum()
            base = A[:, 1].sum() / A[:, 0].sum() * 1e4
            pos = A[:, 3].sum() / A[:, 2].sum() * 1e4
            neg = neg_s / neg_n * 1e4 if neg_n else np.nan
            pl_ = A[:, 5].sum() / A[:, 4].sum() * 1e4
            pick = rng.integers(0, len(dl), size=(N_BOOT, len(dl)))
            bb_ = A[:, 3][pick].sum(1) / A[:, 2][pick].sum(1) * 1e4
            ba_ = A[:, 1][pick].sum(1) / A[:, 0][pick].sum(1) * 1e4
            lo, hi = np.percentile(bb_ - ba_, [2.5, 97.5])
            star = "*" if lo * hi > 0 else " "
            print(f"{h:>7}ms{base:>12.3f}{pos:>12.3f}{neg:>12.3f}"
                  f"{pos-base:>10.3f}{star}[{lo:>+7.3f},{hi:>+7.3f}]{pl_-base:>10.3f}",
                  file=sys.stderr)
            out.append({"day_type": dtl, "h_ms": h, "base_bp": base, "pos_bp": pos,
                        "neg_bp": neg, "diff_bp": pos - base, "ci_lo": lo, "ci_hi": hi,
                        "placebo_diff_bp": pl_ - base,
                        "n_pos": int(A[:, 2].sum()), "n": int(A[:, 0].sum())})
    pl.DataFrame(out).write_parquet(ROOT / "data" / f"cancel_rate_cells_{tag}.parquet")
    print(f"\n-> data/cancel_rate_cells_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
