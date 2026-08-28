"""板の入れ替わり(churn)を 7 通りで測り、将来 log リターンへ OLS で当てる。

【churn とは何か】
板に注文が**入った量**と**出ていった量**の合計。板がどれだけ書き換わったかで、
板の厚み(残高)でも、約定した量でもない。100ms の窓ごとに合計する。

    churn_total  買い + 売り の (入った量 + 出ていった量)
    churn_bid    買い側だけ
    churn_ask    売り側だけ
    churn_top    最良気配「以上」の価格で起きた分(最良気配そのもの + 内側)
    churn_deep   それより外側で起きた分
    churn_imb    (churn_bid − churn_ask) / (churn_bid + churn_ask)  ← 符号を持つ
    churn_cnt    量ではなく**本数**(入った本数 + 出ていった本数)

churn_total = churn_bid + churn_ask = churn_top + churn_deep(2 通りの分け方)。

【板に載る注文だけを数える】
build_var100_l1.py / build_fill_rate.py と同じ規則。

    入り  status == "open"、is_trigger でない、tif ∈ {Alo, Gtc}
    出   status ∈ {canceled, reduceOnlyCanceled, selfTradeCanceled,
                   siblingFilledCanceled, marginCanceled, scheduledCancel, filled}
         かつ同じ tif / is_trigger の条件

Ioc / FrontendMarket / LiquidationMarket は板に留まらないので入れない。
`...Rejected` は板に載っていないので出でもない。

★終端イベントの行は、その注文を出したときの px / side / orig_sz をそのまま
持っている(2026-06-15 で実測: px 一致 100%、side 一致 100%、sz 一致
99.99993%)。したがって注文の一生を追わずに、行の値だけで量と価格が決まる。
日をまたいで生きていた注文の取消も正しく数えられる(同日 open との突合に
頼らないため。実測でその取消は終端の 0.06%)。

★部分約定の途中経過はこの列構成に無いので、注文は発注数量のまま板にいて、
終端で発注数量が出ていく、として数える。約定する注文は全体の 2.4% なので
影響は小さいが 0 ではない(build_var100_l1.py と同じ既知の限界)。

【top / deep の分け方】
その事象より**厳密に前**の BBO を asof(backward, allow_exact_matches=False)で
引き当て、

    買い: px >= best_bid なら top      売り: px <= best_ask なら top

とする。同時刻の BBO はその事象自身を映している可能性があるので使わない。
ティック幅を仮定しないので、刻みが 0.01 固定でなくても壊れない。

【x が確定する時刻 / y の期間】★先読みをしていないことの説明
窓 j は時刻 [T_j − 100ms, T_j) の事象を集める。

    x = その窓の churn      … T_j で確定する(未来の事象は入らない)
    y = log mid_{T_j+h} − log mid_{T_j}   … 期間は (T_j, T_j+h]

`shift(-k)` は y にしか使っていない。mid は l2/bbo から取り、クロスした断面
(best_ask <= best_bid)は除く(build_var100.read_chunk と同じ規則)。

【★帰無対照 — 過去向きの同じ回帰】
churn と**過去**のリターンの関係も同じ手順で出す。

    y_past = log mid_{T_j} − log mid_{T_j−h}   … 期間は (T_j−h, T_j]

これは予測ではなく同時点・過去の関係である。板が書き換わったのは値段が
動いたからで、順序は逆でありうる。前向きと後ろ向きを並べて初めて
「予測できている」と言えるかが判る。

【★量は符号を持たない】
churn_imb 以外の 6 つは大きさであって向きを含まない。符号つきの log リターンに
当てた OLS は、当てはまるほうがおかしい。指示どおり符号つきも出すが、
大きさの特徴量で意味があるのは |y| のほうなので両方出して並べる。

【★標準化しない】
churn の水準は日によって大きく変わるが、日ごとに標準化すると
「その日の全標本から作ったパラメータ」で説明変数を作ることになり、
CLAUDE.md の禁止事項に触れる。生のまま当て、裾の重さには log1p 版を併記する。

【★重なる窓は t 値を水増しする】
ホライズン h の窓は隣どうしが重なる。日の中で h 窓ごとに 1 つだけ取る
**重ならない部分標本**を併記し、判定はそちらで行う(sample = sub)。

    uv run python scripts/build_churn.py --coin xyz:MU
出力: data/churn_<coin>/dt=*.parquet      … 100ms 窓の 7 特徴量
      data/churn_acc_<coin>/dt=*.parquet  … 日ごとの OLS 十分統計量
      data/churn_ols_<coin>.parquet / .csv … OLS の結果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_var100 import KEY, finalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

WIN_NS = 100_000_000                     # 100ms
NW = 24 * 60 * 60 * 10                   # 864,000 窓
RESTING = {"Alo", "Gtc"}
CLOSE = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
         "siblingFilledCanceled", "marginCanceled", "scheduledCancel", "filled"}

# 予測ホライズン(窓の歩数)。窓が 100ms なのでこれが下限
HOR = {"100ms": 1, "200ms": 2, "500ms": 5, "1s": 10, "2s": 20, "5s": 50,
       "10s": 100, "30s": 300, "1m": 600, "2m": 1200, "5m": 3000}

RAW = ["churn_total", "churn_bid", "churn_ask", "churn_top", "churn_deep",
       "churn_imb", "churn_cnt"]
LOGGABLE = [f for f in RAW if f != "churn_imb"]        # 符号つきは log1p にしない
COLS = [(f, "生") for f in RAW] + [(f, "log1p") for f in LOGGABLE]
NCOL = len(COLS)


def bbo_day(tag: str, day: str) -> pl.DataFrame:
    """その日の BBO。クロスした断面は除く(build_var100.read_chunk と同じ規則)。"""
    return (
        pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
        .filter(pl.col("dt") == day)
        .filter((pl.col("best_ask") > pl.col("best_bid"))
                & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
                & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite())
        .select("ts", "best_bid", "best_ask",
                mid=(pl.col("best_bid") + pl.col("best_ask")) / 2)
        .collect()
        .sort("ts")
    )


def day_windows(tag: str, day: str) -> pl.DataFrame | None:
    """1 日分の 100ms 窓の特徴量。板の事象が 1 つも無ければ None。"""
    fp = ROOT / "data" / f"l1_{tag}" / f"dt={day}.parquet"
    if not fp.exists():
        return None
    B = bbo_day(tag, day)
    if B.height == 0:
        return None

    L = (
        pl.scan_parquet(fp)
        .select("ts", "side", "px", "status", "orig_sz", "tif", "is_trigger")
        .filter(pl.col("tif").is_in(RESTING) & (~pl.col("is_trigger"))
                & ((pl.col("status") == "open") | pl.col("status").is_in(CLOSE)))
        # bbo の ts は生のナノ秒(Int64)、l1 は Datetime(ns)。結合のため揃える
        .with_columns(ts=pl.col("ts").cast(pl.Int64))
        .collect()
        .sort("ts")
    )
    if L.height == 0:
        return None

    # ★その事象より厳密に前の BBO。同時刻はその事象自身を映しうる
    L = (L.join_asof(B.select("ts", "best_bid", "best_ask"), on="ts",
                     strategy="backward", allow_exact_matches=False)
          .drop_nulls("best_bid"))
    if L.height == 0:
        return None

    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns")
             .cast(pl.Int64)[0])
    w = ((L["ts"].to_numpy() - t0) // WIN_NS).astype(np.int64)
    ok = (w >= 0) & (w < NW)
    w = w[ok]
    isbid = (L["side"].to_numpy() == "B")[ok]
    q = L["orig_sz"].to_numpy()[ok]
    px = L["px"].to_numpy()[ok]
    top = np.where(isbid, px >= L["best_bid"].to_numpy()[ok],
                   px <= L["best_ask"].to_numpy()[ok])

    bc = lambda m, v=None: np.bincount(w[m], weights=None if v is None else v[m],
                                       minlength=NW).astype(np.float64)
    all_ = np.ones(len(w), bool)
    c_bid, c_ask = bc(isbid, q), bc(~isbid, q)
    c_top, c_deep = bc(top, q), bc(~top, q)
    c_tot = c_bid + c_ask
    n_ev = bc(all_)
    # 板が動かなかった窓は「偏りが無い」とみなして 0(先に決めた規則)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_imb = np.where(c_tot > 0, (c_bid - c_ask) / np.where(c_tot > 0, c_tot, 1.0), 0.0)

    # 窓の終わりの mid(空区間は直前を持ち越す)
    mg = np.full(NW, np.nan)
    wb = ((B["ts"].to_numpy() - t0) // WIN_NS).astype(np.int64)
    okb = (wb >= 0) & (wb < NW)
    mg[wb[okb]] = B["mid"].to_numpy()[okb]        # ts 昇順なので最後の代入が残る
    idx = np.where(np.isfinite(mg), np.arange(NW), 0)
    np.maximum.accumulate(idx, out=idx)
    mg = mg[idx]

    return pl.DataFrame({
        "churn_total": c_tot, "churn_bid": c_bid, "churn_ask": c_ask,
        "churn_top": c_top, "churn_deep": c_deep, "churn_imb": c_imb,
        "churn_cnt": n_ev, "n_ev": n_ev.astype(np.int32),
        "logmid": np.log(np.where(mg > 0, mg, np.nan)),
    })


def stats(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """列ごとの十分統計量 (n, sx, sy, sxy, sxx, syy) を一度に。"""
    n = float(len(y))
    return np.column_stack([
        np.full(X.shape[1], n), X.sum(0), np.full(X.shape[1], y.sum()),
        X.T @ y, (X * X).sum(0), np.full(X.shape[1], float(y @ y)),
    ])


def day_acc(W: pl.DataFrame, dty: str) -> pl.DataFrame:
    """1 日分の窓から OLS の十分統計量を作る。"""
    X = np.empty((NW, NCOL))
    for j, (f, form) in enumerate(COLS):
        v = W[f].to_numpy().astype(np.float64)
        X[:, j] = v if form == "生" else np.log1p(np.maximum(v, 0.0))
    lm = W["logmid"].to_numpy()
    act = W["n_ev"].to_numpy() >= 1

    rows = []
    for hname, h in HOR.items():
        fwd = np.full(NW, np.nan); fwd[:NW - h] = lm[h:] - lm[:NW - h]
        pst = np.full(NW, np.nan); pst[h:] = lm[h:] - lm[:NW - h]
        for direc, y in (("前向き", fwd), ("後ろ向き(帰無対照)", pst)):
            base = np.isfinite(y)
            for lay, lmask in (("全窓", base), ("動いた窓", base & act)):
                idx = np.flatnonzero(lmask)
                if len(idx) < 30:
                    continue
                Xm, ym = X[idx], y[idx]
                sub = idx[(idx % h) == 0] if h > 1 else idx     # 重ならない部分標本
                Xs, ys = X[sub], y[sub]
                for ykind, yy, yys in (("符号つき", ym, ys),
                                       ("絶対値", np.abs(ym), np.abs(ys))):
                    for smp, A, b in (("full", Xm, yy), ("sub", Xs, yys)):
                        if len(b) < 30:
                            continue
                        S = stats(A, b)
                        for j, (f, form) in enumerate(COLS):
                            rows.append((f"{f}|{lay}", form, ykind, dty,
                                         f"{direc}|{hname}", smp, *S[j]))
    return pl.DataFrame(rows, schema=[*KEY, "n", "sx", "sy", "sxy", "sxx", "syy"],
                        orient="row")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    days = sorted(p.name.split("=")[1].removesuffix(".parquet")
                  for p in (ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    have = set(pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
               .select("dt").unique().collect()["dt"].to_list())
    skip = [d for d in days if d not in have]
    days = [d for d in days if d in have]
    if skip:
        print(f"[skip] BBO が無い日: {skip}", file=sys.stderr)
    if a.days:
        days = days[: a.days]

    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dtype = {d: ("立会日" if d in sess else "閉場日") for d in days}
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    outdir = ROOT / "data" / f"churn_{tag}"
    accdir = ROOT / "data" / f"churn_acc_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    accdir.mkdir(parents=True, exist_ok=True)

    for i, day in enumerate(days, 1):
        ap_ = accdir / f"dt={day}.parquet"
        if ap_.exists():
            continue
        wp = outdir / f"dt={day}.parquet"
        W = pl.read_parquet(wp) if wp.exists() else day_windows(tag, day)
        if W is None:
            print(f"  [skip] {day} 事象なし", file=sys.stderr)
            continue
        if not wp.exists():
            W.write_parquet(wp)
        day_acc(W, dtype[day]).write_parquet(ap_)
        del W
        print(f"  {i}/{len(days)} {day}", file=sys.stderr, flush=True)

    S = pl.read_parquet(accdir / "dt=*.parquet")
    # 立会日 / 閉場日 に加えて、日区分で条件づけない「全日」も出す
    S = pl.concat([S, S.with_columns(day_type=pl.lit("全日"))])
    R = finalize(S)
    R.write_parquet(ROOT / "data" / f"churn_ols_{tag}.parquet")
    R.write_csv(ROOT / "data" / f"churn_ols_{tag}.csv")
    print(f"\n[OLS] {R.height:,} 行 -> data/churn_ols_{tag}.parquet / .csv",
          file=sys.stderr)


if __name__ == "__main__":
    main()
