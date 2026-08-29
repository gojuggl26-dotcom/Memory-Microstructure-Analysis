"""特定のウォレットがマーケットメイカーらしいかを 13 の基準で判定する。

【対象の選び方】
`wallet_*` の **depth contribution(mid ±10bp の板厚に占める割合)が大きい順**に
上位 N 者を取る。結果を見てから選ぶのではなく、**板をどれだけ出しているか**という
事前に決めた基準で選ぶ(生存バイアスを避けるため)。

【13 の基準】
    two-sided quoting ratio     買いと売りを同時に出していた時間の割合
    continuous quoting ratio    注文を 1 本でも出していた時間の割合
    BBO participation           最良気配に居た時間の割合
    symmetric placement         1 − |bid距離 − ask距離| / (bid距離 + ask距離)
    high update frequency       1 時間あたりの発注数
    high cancellation rate      取消数 / 発注数
    low median lifetime         注文の生存時間の中央値
    replenishment behavior      自分の注文が消えてから同じ側へ出し直すまでの時間と、
                                1 秒以内に出し直した割合
    inventory-sensitive skew    自分の建玉と気配の傾きの相関(建玉は fills の
                                startPosition から復元。proxy ではなく実測値)
    depth contribution          mid ±10bp の板厚に占める割合
    market coverage             mid ±10bp に注文を置いていた時間の割合
    quote width stability       自分の建てるスプレッドの変動係数(1 分ごと)
    quote-size stability        自分の出す数量の変動係数(1 分ごと)

【計算の経路】
板の全体像は要らない。**対象ウォレットの l1 の行だけ**を取り出し、
最良気配は取得済みの `l2/bbo` を asof で貼る。板の再構成(1 日 1 分以上)を
回避できるので、対象を絞れば全期間を短時間で処理できる。

【x が確定する時刻】
すべてその日の中で閉じた記述統計。将来の値は使っていない。

【★注意】
`bbo_presence` は「自分の指値の価格が最良気配と一致していた時間」であって、
キューの何番目かは見ていない。前に大量に並んでいれば約定はしない
([約定率のレポート](../reports/MU/mu_fill_rate_report.md)参照)。

    uv run python scripts/build_wallet_mm.py --coin xyz:MU --top 6
出力: data/wallet_mm_<coin>.csv       … ウォレット × 13 基準
      data/wallet_mm_daily_<coin>.csv … ウォレット × 日 の内訳
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.01
NEAR_BP = 10.0
MIN_S = 60                       # 安定性を測る刻み(秒)
REPLENISH_MS = 1_000             # 出し直しとみなす上限
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}


def day_rows(l1: Path, l1u: Path, targets: set[str]) -> pl.DataFrame:
    d = pl.read_parquet(l1)
    u = pl.read_parquet(l1u)
    if u.height != d.height:
        sys.exit(f"{l1.name}: user の行数が合わない")
    return (d.with_columns(user=u["user"])
             .filter(pl.col("user").is_in(list(targets)))
             .with_columns(pl.col("ts").cast(pl.Int64)).sort("ts"))


def lifecycles(d: pl.DataFrame, day_end: int) -> pl.DataFrame:
    """対象ウォレットの注文を 1 本 1 行にする。"""
    op = d.filter((pl.col("status") == "open") & ~pl.col("is_trigger")
                  & pl.col("tif").is_in(list(RESTING)))
    tm = (d.filter(pl.col("status").is_in(["filled"] + list(CANCEL)))
           .select("oid", t_close="ts",
                   is_cancel=pl.col("status").is_in(list(CANCEL)))
           .group_by("oid").first())
    return (op.select("oid", "user", "side", "px", "orig_sz", t_open="ts")
              .join(tm, on="oid", how="left")
              .with_columns(censored=pl.col("t_close").is_null(),
                            t_close=pl.col("t_close").fill_null(day_end))
              .with_columns(dur_s=(pl.col("t_close") - pl.col("t_open")) / 1e9,
                            bid=pl.col("side") == "B"))


def presence(t_open, t_close, bid, t0, day_ns):
    """区間の和集合から在席時間を出す。買い・売り・両建てを別々に。"""
    def union(a, b):
        if len(a) == 0:
            return 0
        o = np.argsort(a)
        a, b = a[o], b[o]
        tot, cur_s, cur_e = 0, a[0], b[0]
        for s_, e_ in zip(a[1:], b[1:]):
            if s_ > cur_e:
                tot += cur_e - cur_s
                cur_s, cur_e = s_, e_
            else:
                cur_e = max(cur_e, e_)
        return tot + cur_e - cur_s

    def overlap(a1, b1, a2, b2):
        """2 つの和集合の重なり。掃引で求める。"""
        ev = np.concatenate([np.stack([a1, np.ones(len(a1))]),
                             np.stack([b1, -np.ones(len(b1))]),
                             np.stack([a2, np.full(len(a2), 2.0)]),
                             np.stack([b2, np.full(len(b2), -2.0)])], axis=1)
        o = np.argsort(ev[0])
        t, d = ev[0][o], ev[1][o]
        c1 = c2 = 0
        tot, prev = 0, t[0] if len(t) else 0
        for ti, di in zip(t, d):
            if c1 > 0 and c2 > 0:
                tot += ti - prev
            if abs(di) == 1:
                c1 += 1 if di > 0 else -1
            else:
                c2 += 1 if di > 0 else -1
            prev = ti
        return tot

    ob, cb = t_open[bid], t_close[bid]
    oa, ca = t_open[~bid], t_close[~bid]
    any_t = union(t_open, t_close)
    two_t = overlap(ob, cb, oa, ca) if len(ob) and len(oa) else 0
    return any_t / day_ns, two_t / day_ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    W = pl.concat([pl.read_parquet(f)
                   for f in sorted(glob.glob(str(ROOT / "data" / f"wallet_{tag}" / "dt=*.parquet")))])
    rank = (W.group_by("user").agg(pl.col("liq_contribution").mean().alias("liq"),
                                   pl.col("displayed_notional").mean().alias("ntl"))
             .sort("liq", descending=True).head(a.top))
    targets = set(rank["user"].to_list())
    print(f"[対象] depth contribution 上位 {len(targets)} 者", file=sys.stderr)

    l1dir, l1udir = ROOT / "data" / f"l1_{tag}", ROOT / "data" / f"l1u_{tag}"
    days = sorted(p.stem.split("=")[1] for p in l1udir.glob("dt=*.parquet"))
    bbo = pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
    fills = (pl.scan_parquet(ROOT / "data" / f"fills_{tag}.parquet")
               .filter(pl.col("user").is_in(list(targets)))
               .select("ts", "user", "startPosition", "dt", "sz", "side").collect())

    # depth contribution の分母。elast_* の 1 秒断面から mid ±10bp の平均厚み
    ELAST = {}
    for f in sorted(glob.glob(str(ROOT / "data" / f"elast_{tag}" / "dt=*.parquet"))):
        try:
            e = pl.read_parquet(f, columns=["qb_10", "qa_10"])
        except Exception:
            continue
        ELAST[Path(f).stem.split("=")[1]] = float((e["qb_10"] + e["qa_10"]).mean())
    print(f"[分母] elast のある日 {len(ELAST)}", file=sys.stderr)

    rows, mins = [], []
    for n, day in enumerate(days, 1):
        t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
        day_end = t0 + 86_400_000_000_000
        try:
            d = day_rows(l1dir / f"dt={day}.parquet", l1udir / f"dt={day}.parquet", targets)
        except Exception as e:      # 取得中の書きかけファイルを掴むことがある
            print(f"  {day} 読めないので飛ばす: {type(e).__name__}", file=sys.stderr)
            continue
        if d.height == 0:
            continue
        L = lifecycles(d, day_end)
        B = (bbo.filter(pl.col("dt") == day)
                .select("ts", "best_bid", "best_ask", "bid_sz", "ask_sz").collect()
                .filter(pl.col("best_ask") > pl.col("best_bid")).sort("ts")
                .with_columns(ts=pl.col("ts").cast(pl.Int64),
                              mid=(pl.col("best_bid") + pl.col("best_ask")) / 2))
        if B.height < 10:
            continue
        bt = B["ts"].to_numpy()
        bmid, bb, ba = B["mid"].to_numpy(), B["best_bid"].to_numpy(), B["best_ask"].to_numpy()

        # 注文が生きていた間の最良気配(開始時点で貼る。距離の基準)
        j0 = np.clip(np.searchsorted(bt, L["t_open"].to_numpy(), side="right") - 1, 0, len(bt) - 1)
        mid0 = bmid[j0]
        px = L["px"].to_numpy()
        dist = np.abs(px - mid0) / mid0 * 1e4
        at_touch = np.where(L["bid"].to_numpy(), px == bb[j0], px == ba[j0])
        L = L.with_columns(dist_bp=dist, mid0=mid0, at_touch=at_touch,
                           wgt=pl.col("orig_sz") * pl.col("dur_s"))

        # ---- 出し直し(replenishment): 終了から同じ側の次の発注まで --------
        for u in targets:
            s = L.filter(pl.col("user") == u)
            if s.height < 20:
                continue
            rep = []
            for side in (True, False):
                q = s.filter(pl.col("bid") == side).sort("t_open")
                if q.height < 5:
                    continue
                opens = q["t_open"].to_numpy()
                closes = np.sort(q["t_close"].to_numpy())
                k = np.searchsorted(opens, closes, side="left")
                ok = k < len(opens)
                rep.append((opens[np.clip(k, 0, len(opens) - 1)][ok] - closes[ok]) / 1e6)
            gap = np.concatenate(rep) if rep else np.array([])
            gap = gap[np.isfinite(gap) & (gap >= 0)]

            # ---- 1 分刻みの安定性と建玉感応 -------------------------------
            mb = np.clip((s["t_open"].to_numpy() - t0) // (MIN_S * 10 ** 9), 0, 1439)
            bidm = s["bid"].to_numpy()
            dm, szm = s["dist_bp"].to_numpy(), s["orig_sz"].to_numpy()
            wid, siz, skw, idx = [], [], [], []
            for m in np.unique(mb):
                sel = mb == m
                db = dm[sel & bidm]
                da = dm[sel & ~bidm]
                if len(db) >= 3 and len(da) >= 3:
                    wid.append(np.median(db) + np.median(da))
                    skw.append(np.median(db) - np.median(da))
                    idx.append(m)
                if sel.sum() >= 3:
                    siz.append(np.median(szm[sel]))
            f = fills.filter((pl.col("user") == u) & (pl.col("dt").cast(pl.String) == day))
            inv_r = np.nan
            if f.height > 20 and len(idx) > 10:
                ft = f["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
                fp = f["startPosition"].to_numpy().astype(float)
                o = np.argsort(ft)
                fm = np.clip((ft[o] - t0) // (MIN_S * 10 ** 9), 0, 1439)
                pos = np.full(1440, np.nan)
                pos[fm] = fp[o]
                good = np.isfinite(pos[np.array(idx)]) & np.isfinite(np.array(skw))
                if good.sum() > 10:
                    inv_r = float(np.corrcoef(pos[np.array(idx)][good],
                                              np.array(skw)[good])[0, 1])
            # ★在席は wallet_* に頼らずここで区間の和集合から出す
            cont, two = presence(s["t_open"].to_numpy(), s["t_close"].to_numpy(),
                                 s["bid"].to_numpy(), t0, 86_400_000_000_000)
            wb = s.filter(pl.col("bid"))
            wa = s.filter(~pl.col("bid"))
            db_ = (float((wb["dist_bp"] * wb["wgt"]).sum() / wb["wgt"].sum())
                   if wb.height and wb["wgt"].sum() > 0 else np.nan)
            da_ = (float((wa["dist_bp"] * wa["wgt"]).sum() / wa["wgt"].sum())
                   if wa.height and wa["wgt"].sum() > 0 else np.nan)
            sym = (1 - abs(db_ - da_) / (db_ + da_)) if np.isfinite(db_ + da_) else np.nan
            dep = np.nan
            if day in ELAST:
                tot10 = ELAST[day]
                own10 = float(s.filter(pl.col("dist_bp") <= NEAR_BP)["wgt"].sum()) / 86_400.0
                dep = own10 / tot10 if tot10 > 0 else np.nan
            near = s.filter(pl.col("dist_bp") <= NEAR_BP)
            rows.append({
                "user": u, "dt": day,
                "two_sided_ratio": two,
                "continuous_ratio": cont,
                "bbo_participation": float(s["at_touch"].mean()),
                "symmetric_placement": sym,
                "bid_dist_bp": db_, "ask_dist_bp": da_,
                "update_freq_per_h": s.height / 24.0,
                "cancel_rate": float(s["is_cancel"].fill_null(False).mean()),
                "median_lifetime_s": float(s["dur_s"].median()),
                "replenish_med_ms": float(np.median(gap)) if len(gap) else np.nan,
                "replenish_1s_share": float((gap <= REPLENISH_MS).mean()) if len(gap) else np.nan,
                "inv_skew_corr": inv_r,
                "depth_contribution": dep,
                "market_coverage": float(near["wgt"].sum() / s["wgt"].sum()) if s["wgt"].sum() else np.nan,
                "quote_width_cv": float(np.std(wid) / np.mean(wid)) if len(wid) > 5 else np.nan,
                "quote_size_cv": float(np.std(siz) / np.mean(siz)) if len(siz) > 5 else np.nan,
                "touch_share": float(s["at_touch"].mean()),
                "n_orders": s.height,
            })
        if n % 10 == 0 or n == len(days):
            print(f"  {day}  ({n}/{len(days)} 日)", file=sys.stderr)

    D = pl.DataFrame(rows)
    D.write_csv(ROOT / "data" / f"wallet_mm_daily_{tag}.csv")
    num = [c for c in D.columns if c not in ("user", "dt")]
    S = (D.group_by("user").agg([pl.col(c).median().alias(c) for c in num]
                                + [pl.len().alias("n_days")])
          .join(rank, on="user", how="left").sort("liq", descending=True))
    S.write_csv(ROOT / "data" / f"wallet_mm_{tag}.csv")
    print(f"\n[集計] {S.height} 者 × {D['dt'].n_unique()} 日", file=sys.stderr)
    print(S.to_pandas().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
