"""実際に清算された建玉の一覧と、ヒートマップ用の格子を作る。

    uv run python scripts/build_liq.py --coin xyz:MU

入力: data/liq_<coin>.parquet(fetch_liq.py)/ data/fills_<coin>.parquet /
      data/oi_series_<coin>.parquet / data/bbo_<coin>.parquet
出力: data/liqev_<coin>.parquet   1 行 = 1 清算イベント
      data/liqheat_<coin>.parquet 時刻 × 価格 の格子(side 別)
      data/liqpx_<coin>.parquet   1 時間ごとの価格と出来高(重ね描き用)
      data/liq_summary_<coin>.csv 要約

★数え方(二重計上をしないための定義)
------------------------------------
1 つの清算約定は必ず 2 行になる。清算された口座の行と、その相手方の行である。
両方に `liquidation = true` が立つので、**真偽だけで数えると 2 倍になる**。
清算された建玉は

    user == liquidatedUser の行の sz

で一意に決まる。実測でこの行は 100% が `crossed = true`(テイカー)で、
`dir` は `Close Long`(ロングの清算=強制売り)か `Close Short`(ショートの清算=
強制買い)のどちらかしか出ない。

★OI への効き方は相手方で決まる
------------------------------
清算は建玉を消すとは限らない。相手方が新規に建てれば建玉は**移るだけ**である。

    相手方が Open Long / Open Short   → OI は変わらない(建玉の移転)
    相手方が Close Short / Close Long → OI が sz だけ減る
    相手方が Long > Short など        → 途中まで減る(ここでは減少と数える)

x が確定する時刻 / y の期間: 該当なし(記述統計)。OI と mid は
**清算時刻以前の最後の値**を backward で引く(先読みなし)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
KEY = ["ts", "user", "px", "sz", "side", "dir", "tid", "oid",
       "liquidatedUser", "startPosition", "closedPnl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--px-bin", type=float, default=2.5, help="価格格子の幅(USD)")
    ap.add_argument("--t-bin", type=int, default=3600, help="時刻格子の幅(秒)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    q = pl.read_parquet(D / f"liq_{tag}.parquet")
    n0 = q.height
    q = q.unique(subset=KEY, keep="first").sort("ts")
    print(f"[liq] 取得 {n0:,} 行 -> 重複除去後 {q.height:,} 行 "
          f"(完全重複 {n0 - q.height} 行)", file=sys.stderr)

    g = q.group_by("tid").agg(pl.len().alias("n"))
    assert int((g["n"] != 2).sum()) == 0, "1 約定 2 行でない tid がある"
    assert q.select((pl.col("liquidationMethod") == "market").all()).item(), "market 以外あり"

    liq = q.filter(pl.col("user") == pl.col("liquidatedUser"))
    cp = q.filter(pl.col("user") != pl.col("liquidatedUser")).select(
        "tid", cp_user="user", cp_dir="dir")
    assert liq.height == cp.height == g.height

    ev = (liq.join(cp, on="tid", how="inner")
          .with_columns(
              lside=pl.when(pl.col("dir") == "Close Long").then(pl.lit("long"))
                     .otherwise(pl.lit("short")),
              notional=pl.col("px") * pl.col("sz"),
              notional_mark=pl.col("liquidationMarkPx") * pl.col("sz"),
              is_full=(pl.col("sz") - pl.col("startPosition").abs()).abs() < 1e-9,
              oi_cut=pl.when(pl.col("cp_dir").str.starts_with("Open"))
                      .then(pl.lit(0.0)).otherwise(pl.col("sz")),
          )
          .with_columns(
              # スリッページ: 清算された側にとって不利な向きを正にする
              slip_bp=pl.when(pl.col("lside") == "long")
                       .then((pl.col("liquidationMarkPx") / pl.col("px") - 1) * 1e4)
                       .otherwise((pl.col("px") / pl.col("liquidationMarkPx") - 1) * 1e4))
          .rename({"liquidationMarkPx": "mark_px"}))

    # --- 口座に通し番号を振る(住所は出さない) -----------------------------
    rank = (ev.group_by("user").agg(pl.col("notional").sum().alias("v"))
            .sort("v", descending=True).with_row_index("i")
            .with_columns(luid=pl.format("L{}", pl.col("i") + 1)).select("user", "luid"))
    crank = (ev.group_by("cp_user").agg(pl.col("notional").sum().alias("v"))
             .sort("v", descending=True).with_row_index("i")
             .with_columns(cuid=pl.format("C{}", pl.col("i") + 1)).select("cp_user", "cuid"))
    ev = ev.join(rank, on="user").join(crank, on="cp_user")

    # --- 清算時点の OI と mid を backward で引く ---------------------------
    oi = pl.read_parquet(D / f"oi_series_{tag}.parquet").sort("ts")
    ev = ev.sort("ts").join_asof(oi.select("ts", "oi"), on="ts", strategy="backward")

    bb = (pl.scan_parquet(D / f"bbo_{tag}.parquet")
          .select(ts=pl.col("ts").cast(pl.Datetime("ns")),
                  mid=(pl.col("best_bid") + pl.col("best_ask")) / 2,
                  crossed_book=pl.col("best_ask") < pl.col("best_bid"))
          .sort("ts").collect())
    ev = ev.join_asof(bb, on="ts", strategy="backward", tolerance="60s")

    ev = ev.select("ts", "dt", "luid", "cuid", "lside", "sz", "px", "mark_px",
                   "notional", "notional_mark", "closedPnl", "fee", "startPosition",
                   "is_full", "cp_dir", "oi_cut", "slip_bp", "oi", "mid",
                   "crossed_book", "tid")
    ev.write_parquet(D / f"liqev_{tag}.parquet", compression="zstd")

    # --- 1 時間ごとの価格と出来高 ------------------------------------------
    f = (pl.scan_parquet(D / f"fills_{tag}.parquet").filter(pl.col("crossed"))
         .with_columns(tb=(pl.col("ts").dt.epoch("s") // a.t_bin * a.t_bin))
         .group_by("tb").agg(
             pl.col("px").first().alias("o"), pl.col("px").max().alias("hi"),
             pl.col("px").min().alias("lo"), pl.col("px").last().alias("c"),
             (pl.col("px") * pl.col("sz")).sum().alias("vol_usd"),
             pl.col("sz").sum().alias("vol_sz"), pl.len().alias("n_tr"))
         .sort("tb").collect())
    f.write_parquet(D / f"liqpx_{tag}.parquet", compression="zstd")

    # --- ヒートマップ格子 ---------------------------------------------------
    b = a.px_bin
    heat = (ev.with_columns(
                tb=(pl.col("ts").dt.epoch("s") // a.t_bin * a.t_bin),
                pb=(pl.col("mark_px") / b).floor() * b)
            .group_by("tb", "pb", "lside").agg(
                pl.col("notional").sum().alias("usd"),
                pl.col("sz").sum().alias("sz"),
                pl.len().alias("n"),
                pl.col("luid").n_unique().alias("n_acct"),
                pl.col("oi_cut").sum().alias("oi_cut"))
            .sort("tb", "pb"))
    heat.write_parquet(D / f"liqheat_{tag}.parquet", compression="zstd")

    # --- 要約 ---------------------------------------------------------------
    vol = float(f["vol_usd"].sum())
    tot = float(ev["notional"].sum())
    rows = []
    for s, sub in [("all", ev), ("long", ev.filter(pl.col("lside") == "long")),
                   ("short", ev.filter(pl.col("lside") == "short"))]:
        rows.append({
            "side": s, "n_events": sub.height,
            "notional_usd": float(sub["notional"].sum()),
            "sz": float(sub["sz"].sum()),
            "n_accounts": sub["luid"].n_unique(),
            "n_counterparties": sub["cuid"].n_unique(),
            "median_usd": float(sub["notional"].median()),
            "p99_usd": float(sub["notional"].quantile(0.99)),
            "max_usd": float(sub["notional"].max()),
            "full_close_pct": float(sub["is_full"].mean() * 100),
            "oi_cut_sz": float(sub["oi_cut"].sum()),
            "oi_cut_pct": float(sub["oi_cut"].sum() / sub["sz"].sum() * 100),
            "closed_pnl_usd": float(sub["closedPnl"].sum()),
            "fee_usd": float(sub["fee"].sum()),
            "slip_bp_median": float(sub["slip_bp"].median()),
            "slip_bp_mean": float(sub["slip_bp"].mean()),
            "pct_of_volume": float(sub["notional"].sum() / vol * 100),
        })
    sm = pl.DataFrame(rows)
    sm.write_csv(D / f"liq_summary_{tag}.csv")
    with pl.Config(tbl_width_chars=220, tbl_cols=25):
        print(sm)
    print(f"\n[liq] 清算 {ev.height:,} 件 / ${tot/1e6:.2f}M / "
          f"出来高 ${vol/1e9:.2f}B の {tot/vol*100:.3f}%", file=sys.stderr)
    print(f"[liq] 清算された口座 {ev['luid'].n_unique():,} / "
          f"相手方 {ev['cuid'].n_unique():,}", file=sys.stderr)
    print(f"[liq] mid が引けなかった清算 {int(ev['mid'].is_null().sum())} 件 / "
          f"板がクロスしていた {int(ev['crossed_book'].fill_null(False).sum())} 件",
          file=sys.stderr)
    print(f"[liq] 格子 {heat.height:,} セル", file=sys.stderr)


if __name__ == "__main__":
    main()
