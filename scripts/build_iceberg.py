"""隠れ数量(iceberg)と注文分割の代理指標 8 つ。

【問い】
取引所が iceberg を明示しなくても、「見えている以上の数量が裏にある」なら
行動に痕跡が残る。同じ口座が同じ値段へ、削られるたびに同じ大きさで出し直す。
その痕跡を測る。

【★ 区別の鍵 — 何の**後**に出し直したか】
この市場では出し直しは全員の既定行動である([見せかけの板を疑う 18 指標](
../reports/hyperliquid/MU/mu_manipulation_report.md): 取消率 98.9%、
[生存率](../reports/hyperliquid/MU/mu_hazard_report.md): 寿命の中央値 565ms)。
したがって「同じ値段に出し直す」だけでは何の証拠にもならない。分けるべきは:

    約定の後の補充  … 削られたから足す = 裏に数量がある証拠になりうる
    取消の後の補充  … 値段の付け直し   = 隠れ数量とは無関係

本スクリプトは全指標をこの 2 つに割り、**取消後を対照**として並べる。

【8 指標】
  1 repeated same-wallet same-price replenishment
        スロット(口座 × 側 × 価格)ごとの「一続きの補充」= episode の子注文数
  2 identical-size repetition
        補充のとき前と**同じ数量**か / その口座の最頻数量が占める割合
  3 child-order frequency
        episode の中の子注文の到着間隔(秒)と、毎秒の子注文数
  4 rapid refill after depletion
        約定で終わった注文のうち、同じ値段へ 1 秒以内に出し直した割合
        (対照 = 取消で終わった注文の同じ割合)
  5 persistent hidden-size proxy
        episode 中に約定した数量 ÷ その episode で見せた最大の数量
        「見せた最大の何倍を約定させたか」
  6 replenishment-to-visible-depth ratio
        episode 中に出した数量の合計 ÷ 見せた最大の数量
  7 same-price cumulative turnover
        距離帯ごとに Σ(出した数量) ÷ Σ(数量 × 表示時間)  …… 毎秒
        = 数量加重の平均表示時間の逆数。「その水準は毎秒何回入れ替わるか」
  8 same-wallet turnover
        口座ごとに同じ計算。「その口座の板は毎秒何回入れ替わるか」

【x が確定する時刻 / y の期間】
記述の量であって予測の説明変数ではない。4・5 は注文が終わった**後**の
出し直しを見るので、設計上ルックアヘッドを含む(監視・実態把握のための量)。

【落とし穴】
- 部分約定して生き残る注文は実測 0.09%(2026-06-24 で 99,294 件中 91 件)。
  注文は open → 終端の 2 イベントで閉じるとみなしてよい。
- 数量は 0.001 の整数ロット、価格は有効数字 5 桁でティックが変わる。
  合計は必ず Int64 へ上げてから(Int32 のままだと日次合計で桁あふれする)。

    uv run python scripts/build_iceberg.py --coin xyz:MU --days 3
    uv run python scripts/build_iceberg.py --coin xyz:MU
出力: data/ice_wallet_<coin>.parquet  口座 × 日 の集計
      data/ice_daily_<coin>.csv       市場全体の日次
      data/ice_band_<coin>.csv        距離帯 × 日 の回転率
      data/ice_hist_<coin>.csv        分布(子注文数・間隔・隠れ倍率)
      data/ice_meta_<coin>.csv        日ごとの検算
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, PX_UNIT, RESTING,  # noqa: E402
                              SZ_LOT, TERMINAL, clean_bbo)

ROOT = Path(__file__).resolve().parents[1]

FAST_NS = 1_000_000_000        # 「すぐ」出し直した(1 秒)
GAP_NS = 60_000_000_000        # これ以上空いたら別の episode(60 秒)
PT_SPAN = 200_000              # スロット id を作るときの価格の幅
D_EDGES = [0, 1, 3, 6, 11, 26, 51, 101]
D_LAB = ["改善 <0", "0 最良", "1-2", "3-5", "6-10", "11-25", "26-50", "51-100",
         "101+"]
# 分布のビン
NCH_EDGES = [1, 2, 3, 4, 5, 6, 11, 21, 51, 101]
GAP_EDGES = 10.0 ** np.arange(0, 6.51, 0.5)      # ミリ秒 1 〜 3.16e6(≈1 時間)
HID_EDGES = 10.0 ** np.arange(-2, 3.01, 0.25)    # 隠れ倍率 0.01 〜 1000


def bin_dist(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), -1, np.int8)
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    out[ok] = np.where(x < 0, 0, np.searchsorted(D_EDGES, x, side="right"))[ok]
    return out


def day_orders(fp: Path, bts, bbid, bask, um: pl.DataFrame):
    """その日に出て消えた指値。スロット id と距離帯まで付ける。"""
    d = pl.read_parquet(fp)
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & (~pl.col("reduce_only"))
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pt=(pl.col("px") / PX_UNIT).round().cast(pl.Int64),
                  sz=(pl.col("orig_sz") / SZ_LOT).round().cast(pl.Int64),
                  rs=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  op=(pl.col("status") == "open"),
                  canc=pl.col("status").is_in(CANCELS)))
    op = ev.filter(pl.col("op")).select("oid", "is_bid", "pt", "sz",
                                        t0=pl.col("ts"))
    tm = (ev.filter(~pl.col("op"))
          .select("oid", t1=pl.col("ts"), canc="canc", rs="rs")
          .unique(subset="oid", keep="first"))
    n_open = op.height
    o = op.join(tm, on="oid", how="inner").join(um, on="oid", how="inner")
    del d, ev, op, tm
    t0 = o["t0"].to_numpy()
    j = np.searchsorted(bts, t0, side="left") - 1
    ok = j >= 0
    jj = np.clip(j, 0, len(bts) - 1)
    isb = o["is_bid"].to_numpy()
    pt = o["pt"].to_numpy()
    ref = np.where(isb, bbid[jj], bask[jj]).astype(np.float64)
    # ティック幅は価格で変わる(1000 未満 0.01 / 以上 0.1)
    tick = np.where(ref >= 100_000, 10.0, 1.0)
    dd = np.where(ok, np.where(isb, ref - pt, pt - ref) / tick, np.nan)
    o = o.with_columns(
        band=pl.Series(bin_dist(dd)),
        fsz=(pl.col("sz") - pl.col("rs")).clip(0),
        life=(pl.col("t1") - pl.col("t0")).clip(0),
        slot=((pl.col("wid").cast(pl.Int64) * 2
               + pl.col("is_bid").cast(pl.Int64)) * PT_SPAN + pl.col("pt")),
    ).drop("oid", "rs", "pt")
    return o, n_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    bb_all, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    bb_all = bb_all.sort("ts")
    have = set(bb_all["dt"].unique().to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]
    bidx = {d: (int(x), int(y)) for d, x, y in
            bb_all.with_row_index("r").group_by("dt")
            .agg(lo=pl.col("r").min(), hi=pl.col("r").max()).iter_rows()}
    BTS = bb_all["ts"].cast(pl.Int64).to_numpy()
    BBID = np.rint(bb_all["best_bid"].to_numpy() / PX_UNIT).astype(np.int64)
    BASK = np.rint(bb_all["best_ask"].to_numpy() / PX_UNIT).astype(np.int64)
    del bb_all
    gc.collect()
    print(f"[load] {len(files)} 日 / bbo 除外 {n_drop:,}", file=sys.stderr)

    wparts, dparts, bparts, metas = [], [], [], []
    H = {"nchild": np.zeros(len(NCH_EDGES), np.int64),
         "gap_fill": np.zeros(len(GAP_EDGES) + 1, np.int64),
         "gap_canc": np.zeros(len(GAP_EDGES) + 1, np.int64),
         "hid_max": np.zeros(len(HID_EDGES) + 1, np.int64),
         "hid_avg": np.zeros(len(HID_EDGES) + 1, np.int64),
         "replen": np.zeros(len(HID_EDGES) + 1, np.int64),
         # 1 本で終わった episode は定義上 hid<=1 になるので、子が 3 本以上の
         # ものだけを別に貯める(ここが iceberg らしさの本丸)
         "hid_max_multi": np.zeros(len(HID_EDGES) + 1, np.int64)}

    for fp in files:
        dt = fp.stem.split("=")[1]
        up = ROOT / "data" / f"l1user_{tag}" / f"dt={dt}.parquet"
        if not up.exists() or dt not in bidx:
            continue
        lo_, hi_ = bidx[dt]
        o, n_open = day_orders(fp, BTS[lo_:hi_ + 1], BBID[lo_:hi_ + 1],
                               BASK[lo_:hi_ + 1], pl.read_parquet(up))
        if not o.height:
            continue
        # ---- スロット内の並び ------------------------------------------
        o = o.sort("slot", "t0")
        o = o.with_columns(
            same=pl.col("slot") == pl.col("slot").shift(1),
            prev_end=pl.col("t1").cum_max().shift(1).over("slot"),
            nxt_t0=pl.col("t0").shift(-1).over("slot"),
            nxt_sz=pl.col("sz").shift(-1).over("slot"))
        o = o.with_columns(
            new_ep=(~pl.col("same"))
            | ((pl.col("t0") - pl.col("prev_end")) > GAP_NS))
        o = o.with_columns(ep=pl.col("new_ep").cum_sum())
        o = o.with_columns(
            gap=pl.col("nxt_t0") - pl.col("t1"),
            same_sz=pl.col("nxt_sz") == pl.col("sz"))
        o = o.with_columns(
            refill=(pl.col("gap").is_not_null() & (pl.col("gap") >= 0)
                    & (pl.col("gap") <= FAST_NS)),
            byfill=~pl.col("canc"))

        # ---- 分布 ------------------------------------------------------
        for nm, msk in (("gap_fill", ~o["canc"].to_numpy()),
                        ("gap_canc", o["canc"].to_numpy())):
            g = o["gap"].to_numpy().astype(np.float64)[msk]
            g = g[np.isfinite(g) & (g >= 0)] / 1e6            # ミリ秒
            H[nm] += np.bincount(np.searchsorted(GAP_EDGES, g, side="right"),
                                 minlength=len(GAP_EDGES) + 1)

        # ---- episode 集計 ----------------------------------------------
        E = o.group_by("ep").agg(
            wid=pl.col("wid").first(), band=pl.col("band").first(),
            n_child=pl.len(),
            dur=(pl.col("t1").max() - pl.col("t0").min()),
            sz_pl=pl.col("sz").sum(), sz_fl=pl.col("fsz").sum(),
            mx=pl.col("sz").max(),
            szt=(pl.col("sz").cast(pl.Float64)
                 * pl.col("life").cast(pl.Float64)).sum())
        E = E.with_columns(
            avg_disp=pl.when(pl.col("dur") > 0)
            .then(pl.col("szt") / pl.col("dur")).otherwise(None))
        nc = E["n_child"].to_numpy()
        H["nchild"] += np.bincount(np.searchsorted(NCH_EDGES, nc, side="right") - 1,
                                   minlength=len(NCH_EDGES))[: len(NCH_EDGES)]
        mx = E["mx"].to_numpy().astype(np.float64)
        multi = E["n_child"].to_numpy() >= 3
        for nm, v in (("hid_max", E["sz_fl"].to_numpy() / np.maximum(mx, 1)),
                      ("hid_max_multi",
                       (E["sz_fl"].to_numpy() / np.maximum(mx, 1))[multi]),
                      ("hid_avg", E["sz_fl"].to_numpy()
                       / np.maximum(E["avg_disp"].to_numpy().astype(np.float64), 1)),
                      ("replen", E["sz_pl"].to_numpy() / np.maximum(mx, 1))):
            v = v[np.isfinite(v) & (v > 0)]
            H[nm] += np.bincount(np.searchsorted(HID_EDGES, v, side="right"),
                                 minlength=len(HID_EDGES) + 1)

        # ---- 口座 × 日 ---------------------------------------------------
        mode = (o.group_by("wid", "sz").len()
                .group_by("wid").agg(n_modal=pl.col("len").max(),
                                     n_sz=pl.len()))
        w = (o.group_by("wid").agg(
            n_ord=pl.len(), n_slot=pl.col("slot").n_unique(),
            n_fill=(~pl.col("canc")).sum(), n_canc=pl.col("canc").sum(),
            n_rf_fill=((~pl.col("canc")) & pl.col("refill")).sum(),
            n_rf_canc=(pl.col("canc") & pl.col("refill")).sum(),
            n_rf_same=(pl.col("refill") & pl.col("same_sz")).sum(),
            n_rf_fill_same=((~pl.col("canc")) & pl.col("refill")
                            & pl.col("same_sz")).sum(),
            sz_pl=pl.col("sz").sum(), sz_fl=pl.col("fsz").sum(),
            szt=(pl.col("sz").cast(pl.Float64)
                 * pl.col("life").cast(pl.Float64)).sum())
            .join(mode, on="wid", how="left")
            .join(E.group_by("wid").agg(n_ep=pl.len(),
                                        ep_child=pl.col("n_child").mean(),
                                        ep_child_max=pl.col("n_child").max()),
                  on="wid", how="left")
            .with_columns(pl.lit(dt).alias("dt")))
        wparts.append(w)

        # ---- 距離帯 × 日 ---------------------------------------------------
        b = (o.group_by("band").agg(
            n=pl.len(), sz_pl=pl.col("sz").sum(), sz_fl=pl.col("fsz").sum(),
            szt=(pl.col("sz").cast(pl.Float64)
                 * pl.col("life").cast(pl.Float64)).sum(),
            n_fill=(~pl.col("canc")).sum(),
            n_rf_fill=((~pl.col("canc")) & pl.col("refill")).sum(),
            n_canc=pl.col("canc").sum(),
            n_rf_canc=(pl.col("canc") & pl.col("refill")).sum())
            .with_columns(pl.lit(dt).alias("dt")))
        bparts.append(b)

        nf = int((~o["canc"]).sum())
        ncn = int(o["canc"].sum())
        dparts.append({
            "dt": dt, "n_ord": o.height, "n_slot": int(o["slot"].n_unique()),
            "n_ep": E.height, "ep_child_mean": float(E["n_child"].mean()),
            "n_fill": nf, "n_canc": ncn,
            "rf_after_fill": float(((~o["canc"]) & o["refill"]).sum() / max(nf, 1)),
            "rf_after_canc": float((o["canc"] & o["refill"]).sum() / max(ncn, 1)),
            "rf_fill_same_sz": float(
                ((~o["canc"]) & o["refill"] & o["same_sz"]).sum()
                / max(int(((~o["canc"]) & o["refill"]).sum()), 1)),
            "rf_canc_same_sz": float(
                (o["canc"] & o["refill"] & o["same_sz"]).sum()
                / max(int((o["canc"] & o["refill"]).sum()), 1)),
            "turnover_s": float(o["sz"].cast(pl.Int64).sum()
                                / max(float((o["sz"].cast(pl.Float64)
                                             * o["life"].cast(pl.Float64)).sum())
                                      / 1e9, 1)),
            # 約定のあった episode だけ(無い episode は定義上 0 になる)
            "hid_max_med": float(np.nanmedian(
                (E["sz_fl"].to_numpy() / np.maximum(mx, 1))[E["sz_fl"].to_numpy() > 0]))
            if int((E["sz_fl"] > 0).sum()) else float("nan"),
            "hid_max_med_multi": float(np.nanmedian(
                (E["sz_fl"].to_numpy() / np.maximum(mx, 1))[multi & (E["sz_fl"].to_numpy() > 0)]))
            if int((multi & (E["sz_fl"].to_numpy() > 0)).sum()) else float("nan"),
            "n_ep_fill": int((E["sz_fl"] > 0).sum()),
        })
        metas.append({"dt": dt, "n_open": n_open, "n_kept": o.height,
                      "n_no_band": int((o["band"] < 0).sum()),
                      "n_slot": int(o["slot"].n_unique()), "n_ep": E.height})
        print(f"  {dt} 注文 {o.height:>9,} スロット {int(o['slot'].n_unique()):>8,} "
              f"episode {E.height:>8,} 子/ep {float(E['n_child'].mean()):.2f} "
              f"約定後補充 {dparts[-1]['rf_after_fill']:.1%} "
              f"取消後 {dparts[-1]['rf_after_canc']:.1%}", file=sys.stderr)
        del o, E, w, b, mode
        gc.collect()

    pl.concat(wparts, how="diagonal_relaxed").write_parquet(
        ROOT / "data" / f"ice_wallet_{tag}.parquet", compression="zstd")
    pl.DataFrame(dparts).write_csv(ROOT / "data" / f"ice_daily_{tag}.csv")
    pl.concat(bparts, how="diagonal_relaxed").write_csv(
        ROOT / "data" / f"ice_band_{tag}.csv")
    pl.DataFrame(metas).write_csv(ROOT / "data" / f"ice_meta_{tag}.csv")
    rows = []
    for k, v in H.items():
        for i, c in enumerate(v):
            rows.append({"hist": k, "bin": i, "count": int(c)})
    pl.DataFrame(rows).write_csv(ROOT / "data" / f"ice_hist_{tag}.csv")
    print(f"\n[out] -> data/ice_wallet_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
