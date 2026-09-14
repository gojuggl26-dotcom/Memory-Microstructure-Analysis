r"""発注候補 1 件ごとに会場間乖離 d を付け、往復損益との関係を直接見る。

    uv run python scripts/xv_postsig.py --coin xyz:MU --delay 130
    uv run python scripts/xv_postsig.py --coin xyz:MU --delay 0 --sfx _lat0

出力:
    data/xv_sig_<coin>_d<delay>.parquet … dt / t / side / d_bp / age_ms / sess
    data/xv_dec_<coin>_d<delay>.csv     … side·d の十分位ごとの約定率と往復損益

=============================================================================
★時間契約(これを外すと全部が無意味になる)
=============================================================================
発注時刻 t(HL のイベント時刻 ns)で使える Binance の情報は
**`transact_time ≤ t − Δ`** のものだけである。Δ は

    Δ = 会場間の配送遅延 + 自分の計算時間

で、観測できない。よって **Δ を振って感度を見る**。既定は 130ms
(在庫シミュレータで検証済みの遅延水準に合わせた保守側)。

- Binance 側は **debounce 済みの mid 推定**を使う(生の約定値には
  板幅ぶんの跳ね返りが乗っており、取れない見かけの乖離になる)
- 基差は **過去 30 分の移動中央値**で抜く。全標本の中央値は使わない
- HL 側の mid は**その発注イベント時点の最良気配**から作る
  (シミュレータが見ているものと同じ)

★`rt_pnl` は**未来を見る量**であって目的変数専用である。
  d は説明変数側で、確定時刻は t−Δ ≤ t なので契約を満たす。

=============================================================================
★この十分位表が「本命の検定」である理由
=============================================================================
MU では markout だけを学習して何度も偽の edge が出ている。価格を当てる
情報が**約定という条件を通ると残らない**ことが候補 2 で実証された
(5 秒先との相関 −0.0888 に対し、往復損益との相関は +0.0045 で 20 分の 1)。

だから最初に見るのは「d が将来価格を当てるか」ではなく、
**「d が高いときに発注して約定した組の往復損益が高いか」**である。
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BNC = Path("E:/Binance-perp-data/data/history/parquet/kind=aggTrades")
BP = 1e4
NS = 1_000_000_000
SESS = [("00-08 夜", 0, 8), ("08-13.5 プレ", 8, 13.5),
        ("13.5-20 現物", 13.5, 20), ("20-24 アフター", 20, 24)]


def debounce(px, sgn):
    """Δp = h·Δs + ε の切片なし OLS で実効半スプレッドを抜く。"""
    dp, ds = np.diff(px), np.diff(sgn)
    den = float((ds * ds).sum())
    h = max(float((dp * ds).sum() / den) if den > 0 else 0.0, 0.0)
    return px - sgn * h, h


def locf(t_src, v_src, q):
    i = np.searchsorted(t_src, q, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(t_src) - 1)
    return np.where(ok, v_src[i], np.nan), np.where(ok, q - t_src[i], np.inf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sym", default="")
    ap.add_argument("--delay", type=float, default=130.0, help="Δ (ms)")
    ap.add_argument("--posts", default="", help="既定 inv_posts_<tag>_q1")
    ap.add_argument("--maxage", type=float, default=5.0,
                    help="Binance の鮮度の上限 (秒)。超えたら d を欠測にする")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="プラセボ: d を何秒前のものにずらすか(0=しない)")
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    sym = a.sym or (a.coin.split(":")[-1] + "USDT")
    dly = int(a.delay * 1_000_000)
    P = pl.read_parquet(a.posts or DATA / f"inv_posts_{tag}_q1.parquet")
    pday = {k[0] if isinstance(k, tuple) else k: v
            for k, v in P.partition_by("dt", as_dict=True).items()}
    bbo = {os.path.basename(f).split("=")[1].split(".")[0]: f
           for f in glob.glob(str(DATA / f"bbo_l1_{tag}" / "*.parquet"))}

    print(f"{a.coin} × {sym} / Δ={a.delay:g}ms / 候補 {P.height:,} 件 / "
          f"{len(pday)} 日", flush=True)
    out = []
    for day in sorted(pday):
        Q = pday[day].sort("t")
        f = bbo.get(day)
        fb = BNC / f"symbol={sym}" / f"{sym}-aggTrades-{day}.parquet"
        if f is None or not fb.exists():
            continue
        B = pl.read_parquet(f)
        bt = B["ts"].to_numpy()
        bb, ba = B["best_bid"].to_numpy(), B["best_ask"].to_numpy()
        okb = np.isfinite(bb) & np.isfinite(ba) & (ba > bb) & (bb > 0)
        bt, hm = bt[okb], 0.5 * (bb[okb] + ba[okb])

        T = pl.read_parquet(fb, columns=["transact_time", "price",
                                         "quantity", "is_buyer_maker"])
        tt = T["transact_time"].to_numpy().astype(np.int64) * 1_000_000
        o = np.argsort(tt, kind="stable")
        tt = tt[o]
        px = T["price"].to_numpy().astype(np.float64)[o]
        sg = np.where(T["is_buyer_maker"].to_numpy()[o], -1.0, 1.0)
        px, _ = debounce(px, sg)

        # ---- 基差: 1 秒格子の過去 30 分の移動中央値 ----
        d0 = (bt[0] // (86400 * NS)) * 86400 * NS
        g = np.arange(d0, d0 + 86400 * NS, NS, dtype=np.int64)
        gh, _ = locf(bt, hm, g)
        gb, gage = locf(tt, px, g)
        gm = np.isfinite(gh) & np.isfinite(gb) & (gage <= a.maxage * NS)
        graw = np.where(gm, (np.log(gb) - np.log(gh)) * BP, np.nan)
        gbas = (pl.Series(graw).rolling_median(1800, min_samples=300)
                .to_numpy())

        # ---- 発注時刻へ ----
        qt = Q["t"].to_numpy()
        mh, _ = locf(bt, hm, qt)                    # HL の mid(同時刻)
        # ★プラセボ: d だけを大きく過去へずらす。配管が正しければ勾配は消える
        shift = int(a.shift * NS)
        mb, age = locf(tt, px, qt - dly - shift)    # Binance(Δ 前まで)
        if shift:
            # HL 側も同じだけずらして「同時点の乖離」の形を保つ
            mh, _ = locf(bt, hm, qt - shift)
        bas, _ = locf(g, gbas, qt)                  # 基差(過去のみ)
        d = (np.log(mb) - np.log(mh)) * BP - bas
        bad = ~np.isfinite(d) | (age > a.maxage * NS)
        d = np.where(bad, np.nan, d)
        hour = (qt - d0) / (3600 * NS)
        sid = np.full(qt.size, -1, np.int8)
        for i, (_, x, y) in enumerate(SESS):
            sid[(hour >= x) & (hour < y)] = i
        out.append(pl.DataFrame({
            "dt": Q["dt"], "t": qt, "side": Q["side"],
            "filled": Q["filled"], "rt_pnl": Q["rt_pnl"],
            "d_bp": d, "age_ms": age / 1e6, "sess": sid}))
        if len(out) % 20 == 0:
            print(f"  {len(out)}/{len(pday)} {day}", flush=True)

    S = pl.concat(out)
    # ★出力名に**発注候補の出どころ**を必ず入れる。入れなかったせいで
    #   imp1 の実行が q1 の結果を黙って上書きした(同じ事故を 3 度目)。
    src = ""
    if a.posts:
        st = pathlib.Path(a.posts).stem
        src = "_" + st.split(f"inv_posts_{tag}", 1)[-1].lstrip("_")
    sfx = (f"{src}_d{int(a.delay)}"
           + (f"_shift{int(a.shift)}s" if a.shift else ""))
    S.write_parquet(DATA / f"xv_sig_{tag}{sfx}.parquet")
    n_ok = int(S["d_bp"].is_not_nan().sum())
    print(f"\nd が付いた候補 {n_ok:,}/{S.height:,} "
          f"({100*n_ok/S.height:.1f}%)")

    # ---- side·d の十分位 × 約定率 × 往復損益 ----
    # ★十分位の境目は**その日より前**の分布で切る(全標本から作らない)
    S = S.with_columns((pl.col("side") * pl.col("d_bp")).alias("sd"))
    rows = []
    days = sorted(S["dt"].unique().to_list())
    prev = None
    for day in days:
        cur = S.filter(pl.col("dt") == day)
        if prev is not None and prev.height > 5000:
            q = np.nanquantile(prev["sd"].to_numpy(), np.arange(1, 10) / 10)
            b = np.digitize(cur["sd"].to_numpy(), q)
            cur = cur.with_columns(pl.Series("dec", b))
            rows.append(cur)
        prev = cur if prev is None else pl.concat(
            [prev, cur.select(S.columns)]).tail(400000)
    if rows:
        R = pl.concat([r.select(*S.columns, "dec") for r in rows])
        G = (R.filter(pl.col("d_bp").is_not_nan())
             .group_by("dec").agg([
                 pl.len().alias("n"),
                 pl.col("sd").median().alias("sd_med"),
                 (100 * pl.col("filled").mean()).alias("fill_pct"),
                 pl.col("rt_pnl").filter(
                     pl.col("rt_pnl").is_not_nan()).mean().alias("rt_mean"),
                 pl.col("rt_pnl").filter(
                     pl.col("rt_pnl").is_not_nan()).len().alias("n_pair"),
             ]).sort("dec"))
        G.write_csv(DATA / f"xv_dec_{tag}{sfx}.csv")
        print(f"\n{'十分位':>6s}{'side·d 中央':>12s}{'候補':>11s}"
              f"{'約定率%':>9s}{'組':>10s}{'往復損益 bp':>13s}")
        for r in G.iter_rows(named=True):
            print(f"{r['dec']+1:>6d}{r['sd_med']:>12.2f}{r['n']:>11,}"
                  f"{r['fill_pct']:>9.2f}{r['n_pair']:>10,}"
                  f"{r['rt_mean']:>13.3f}")
        lo = G.filter(pl.col("dec") == 0)
        hi = G.filter(pl.col("dec") == 9)
        if lo.height and hi.height:
            print(f"\nD10 − D1 = "
                  f"{float(hi['rt_mean'][0]) - float(lo['rt_mean'][0]):+.3f} bp"
                  f"  (約定率 {float(hi['fill_pct'][0]):.2f}% 対 "
                  f"{float(lo['fill_pct'][0]):.2f}%)")
    print(f"\n書き出し {DATA}/xv_{{sig,dec}}_{tag}{sfx}")


if __name__ == "__main__":
    main()
