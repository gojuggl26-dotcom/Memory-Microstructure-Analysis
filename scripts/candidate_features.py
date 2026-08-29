"""候補特徴量を算出し、既存 25 変数への線形従属性を測る。

「新しい情報か」は、候補 z を既存 25 変数 X に回帰した R² で判定する。
    R²(z | X) が低い = 既存では説明できない = 線形従属性が低い = 新規性がある
    固有分散 = 1 − R²

候補は feature_inventory.md の 8 ファミリー。重いので代表 8 日(成熟期中心)で測る。
出力は 1 秒グリッド。時間契約は既存パネルと同じ(窓 [T−Δ, T) / 板は T 時点)。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
LIFE = Path("data/l2_v99/lifecycle")
BOOKPX = Path("data/l2_v99/book_px")
OUT = D / "candidate_features"
DAYS = ["2026-06-15", "2026-06-30", "2026-07-05", "2026-07-15",
        "2026-07-22", "2026-07-29", "2026-08-03", "2026-08-06"]
STEP = 1_000_000_000
NS_DAY = 86_400_000_000_000
NON_RESTING = ["Ioc", "FrontendMarket", "LiquidationMarket"]


def grid_of(dt: str) -> np.ndarray:
    t0 = int(datetime.strptime(dt, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    return np.arange(t0, t0 + NS_DAY, STEP, dtype=np.int64)


def bsum(ts: np.ndarray, val: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, ts, side="right")
    o = np.zeros(grid.size + 1)
    np.add.at(o, np.clip(idx, 0, grid.size), val)
    return o[: grid.size]


def lifecycle_feats(dt: str, grid: np.ndarray, mid_at: np.ndarray) -> dict:
    df = pl.read_parquet(
        LIFE / f"dt={dt}" / "part-000.parquet",
        columns=["oid", "user", "side", "px", "orig_sz", "filled_sz", "ts_open",
                 "ts_close", "lifetime_ns", "terminal_status", "tif", "order_type",
                 "is_trigger", "trigger_px", "is_rejected", "reduce_only"])
    n = grid.size
    out: dict[str, np.ndarray] = {}

    # ---- 板に載った注文 ----
    rest = df.filter((~pl.col("is_rejected")) & (~pl.col("is_trigger"))
                     & (~pl.col("tif").is_in(NON_RESTING).fill_null(False))
                     & pl.col("ts_open").is_not_null())
    o_ts = rest["ts_open"].to_numpy()
    o_sz = rest["orig_sz"].to_numpy()
    is_b = (rest["side"].to_numpy() == "B")
    win = np.clip(np.searchsorted(grid, o_ts, side="right"), 0, n)

    # A. 参加者 / ウォレット
    u = rest["user"].to_numpy()
    tmp = pl.DataFrame({"w": win, "u": u, "sz": o_sz, "b": is_b}).filter(pl.col("w") < n)
    g = tmp.group_by("w").agg(nu=pl.col("u").n_unique())
    v = np.zeros(n); v[g["w"].to_numpy()] = g["nu"].to_numpy(); out["n_users_new"] = v
    for side, tag in ((True, "bid"), (False, "ask")):
        s = tmp.filter(pl.col("b") == side)
        gg = (s.group_by(["w", "u"]).agg(sz=pl.col("sz").sum())
                .group_by("w").agg(tot=pl.col("sz").sum(),
                                   hhi=(pl.col("sz") ** 2).sum(),
                                   top=pl.col("sz").max()))
        w_ = gg["w"].to_numpy()
        tot = gg["tot"].to_numpy(); hhi = gg["hhi"].to_numpy(); top = gg["top"].to_numpy()
        a = np.zeros(n); a[w_] = np.where(tot > 0, hhi / np.maximum(tot ** 2, 1e-12), 0)
        out[f"hhi_new_{tag}"] = a
        a = np.zeros(n); a[w_] = np.where(tot > 0, top / np.maximum(tot, 1e-12), 0)
        out[f"top1_share_new_{tag}"] = a
    # 参加者の入替率(前窓との Jaccard 距離)
    su = tmp.group_by("w").agg(us=pl.col("u").unique())
    churn = np.zeros(n)
    prev: set = set()
    wmap = {int(w): set(x) for w, x in zip(su["w"], su["us"])}
    for i in range(n):
        cur = wmap.get(i, set())
        if cur or prev:
            inter = len(cur & prev); uni = len(cur | prev)
            churn[i] = 1.0 - (inter / uni if uni else 0.0)
        prev = cur
    out["user_churn"] = churn

    # C. 寿命・キャンセル挙動(その窓に消滅した注文)
    c_ts = rest["ts_close"].to_numpy()
    lif = rest["lifetime_ns"].to_numpy().astype(float)
    cw = np.clip(np.searchsorted(grid, c_ts, side="right"), 0, n)
    cnt = bsum(c_ts, np.ones(c_ts.size), grid)
    out["fleeting_share_100ms"] = np.where(
        cnt > 0, bsum(c_ts, (lif < 1e8).astype(float), grid) / np.maximum(cnt, 1), 0.0)
    out["mean_lifetime_log"] = np.where(
        cnt > 0, bsum(c_ts, np.log1p(lif / 1e6), grid) / np.maximum(cnt, 1), 0.0)
    filled = (rest["terminal_status"].to_numpy() == "filled").astype(float)
    nf = bsum(c_ts, filled, grid)
    out["fill_rate_realized"] = np.where(cnt > 0, nf / np.maximum(cnt, 1), 0.0)

    # D. 種別構成(棄却も母数に入れる)
    allo = df.filter(pl.col("ts_open").is_not_null())
    a_ts = allo["ts_open"].to_numpy()
    n_all = bsum(a_ts, np.ones(a_ts.size), grid)
    is_alo = (allo["tif"].to_numpy() == "Alo").astype(float)
    out["alo_share"] = np.where(n_all > 0, bsum(a_ts, is_alo, grid) / np.maximum(n_all, 1), 0.0)
    rej = allo["is_rejected"].to_numpy().astype(float)
    out["reject_rate"] = np.where(n_all > 0, bsum(a_ts, rej, grid) / np.maximum(n_all, 1), 0.0)
    out["reduce_only_share"] = np.where(
        n_all > 0, bsum(a_ts, allo["reduce_only"].to_numpy().astype(float), grid)
        / np.maximum(n_all, 1), 0.0)
    out["market_order_n"] = bsum(
        a_ts, (allo["order_type"].to_numpy() == "Market").astype(float), grid)

    # E. 発注価格の攻撃性(mid からの距離)
    m = mid_at[np.clip(win - 1, 0, n - 1)]
    good = (win < n) & (m > 0)
    dist = np.where(good, np.abs(o_sz * 0 + rest["px"].to_numpy() - m) / np.maximum(m, 1e-9) * 1e4, 0)
    sw = bsum(o_ts, np.where(good, 1.0, 0.0), grid)
    out["dist_bp_new_mean"] = np.where(sw > 0, bsum(o_ts, np.where(good, dist, 0.0), grid)
                                       / np.maximum(sw, 1), 0.0)
    out["at_touch_share"] = np.where(
        sw > 0, bsum(o_ts, np.where(good & (dist < 0.5), 1.0, 0.0), grid) / np.maximum(sw, 1), 0.0)

    # B. トリガー注文
    tg = df.filter(pl.col("is_trigger") & pl.col("trigger_px").is_not_null()
                   & (pl.col("trigger_px") > 0))
    if tg.height:
        t_o = tg["ts_open"].fill_null(grid[0]).to_numpy()
        t_c = tg["ts_close"].to_numpy()
        t_px = tg["trigger_px"].to_numpy()
        t_sz = tg["orig_sz"].to_numpy()
        out["n_trigger_new"] = bsum(t_o, np.ones(t_o.size), grid)
        out["trigger_fired_n"] = bsum(
            t_c, (tg["terminal_status"].to_numpy() == "filled").astype(float), grid)
        # 生存中トリガーの数量(mid より上 / 下)
        above = np.zeros(n); below = np.zeros(n)
        io = np.clip(np.searchsorted(grid, t_o, side="right"), 0, n)
        ic = np.clip(np.searchsorted(grid, t_c, side="right"), 0, n)
        for a, b, px, sz in zip(io, ic, t_px, t_sz):
            if b <= a:
                continue
            seg = slice(a, min(b, n))
            up = t_px[0] * 0 + px > mid_at[seg]
            above[seg] += np.where(up, sz, 0.0)
            below[seg] += np.where(~up, sz, 0.0)
        out["trigger_stock_above"] = above
        out["trigger_stock_below"] = below
    else:
        for k in ("n_trigger_new", "trigger_fired_n",
                  "trigger_stock_above", "trigger_stock_below"):
            out[k] = np.zeros(n)

    # G. キュー(queue_qi)
    q = pl.read_parquet(D / f"queue_qi/dt={dt}/part-000.parquet",
                        columns=["oid", "q_ahead_open", "open_unknown", "qi_close"])
    j = rest.select("oid", "ts_open", "ts_close").join(q, on="oid", how="inner")
    if j.height:
        jo = j["ts_open"].to_numpy()
        qa = np.nan_to_num(j["q_ahead_open"].to_numpy().astype(float))
        nq = bsum(jo, np.ones(jo.size), grid)
        out["front_arrival_share"] = np.where(
            nq > 0, bsum(jo, (qa == 0).astype(float), grid) / np.maximum(nq, 1), 0.0)
        jc = j["ts_close"].to_numpy()
        qi = np.nan_to_num(j["qi_close"].to_numpy().astype(float))
        nc = bsum(jc, np.ones(jc.size), grid)
        out["qi_close_mean"] = np.where(nc > 0, bsum(jc, qi, grid) / np.maximum(nc, 1), 0.0)
    return out


def book_feats(dt: str, grid: np.ndarray) -> dict:
    """book_px を再生して板形状(F)を 1 秒グリッドで出す。"""
    from bisect import bisect_left
    df = (pl.read_parquet(BOOKPX / f"dt={dt}" / "part-000.parquet",
                          columns=["ts", "side", "px", "total_sz", "is_snapshot"])
          .sort(["ts", "is_snapshot"], descending=[False, True]))
    ts = df["ts"].to_list(); sd = df["side"].to_list()
    px = df["px"].to_list(); sz = df["total_sz"].to_list()
    n = grid.size
    keys = {"B": [], "A": []}
    vals = {"B": [], "A": []}
    res = {k: np.zeros(n) for k in
           ("n_levels_bid", "n_levels_ask", "depth_total_bid", "depth_total_ask",
            "depth_25bp_bid", "depth_25bp_ask", "book_slope_bid", "book_slope_ask")}
    gi = 0
    i = 0
    N = len(ts)
    while gi < n:
        while i < N and ts[i] <= grid[gi]:
            s = sd[i]
            k = -px[i] if s == "B" else px[i]
            arr, va = keys[s], vals[s]
            j = bisect_left(arr, k)
            hit = j < len(arr) and arr[j] == k
            if sz[i] > 0:
                if hit:
                    va[j] = sz[i]
                else:
                    arr.insert(j, k); va.insert(j, sz[i])
            elif hit:
                del arr[j]; del va[j]
            i += 1
        for s, tag in (("B", "bid"), ("A", "ask")):
            arr, va = keys[s], vals[s]
            if not arr:
                continue
            best = -arr[0] if s == "B" else arr[0]
            res[f"n_levels_{tag}"][gi] = len(arr)
            res[f"depth_total_{tag}"][gi] = sum(va)
            lim = best * (1 - 0.0025) if s == "B" else best * (1 + 0.0025)
            d25 = 0.0; num = 0.0; den = 0.0
            for kk, vv in zip(arr[:60], va[:60]):
                p = -kk if s == "B" else kk
                if (s == "B" and p >= lim) or (s == "A" and p <= lim):
                    d25 += vv
                dd = abs(p - best) / best * 1e4
                num += dd * vv; den += vv
            res[f"depth_25bp_{tag}"][gi] = d25
            res[f"book_slope_{tag}"][gi] = num / den if den > 0 else 0.0
        gi += 1
        if i >= N and gi < n:
            for s, tag in (("B", "bid"), ("A", "ask")):
                for key in ("n_levels", "depth_total", "depth_25bp", "book_slope"):
                    res[f"{key}_{tag}"][gi:] = res[f"{key}_{tag}"][gi - 1]
            break
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for dt in DAYS:
        grid = grid_of(dt)
        mp = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                              columns=["ts", "mid", "is_crossed"])
              .filter(~pl.col("is_crossed")).sort("ts"))
        mts = mp["ts"].to_numpy(); mv = mp["mid"].to_numpy()
        jj = np.clip(np.searchsorted(mts, grid, side="right") - 1, 0, mts.size - 1)
        mid_at = mv[jj]
        f = lifecycle_feats(dt, grid, mid_at)
        f.update(book_feats(dt, grid))
        pl.DataFrame({"ts": grid, **{k: v.astype(np.float64) for k, v in f.items()}}
                     ).write_parquet(OUT / f"{dt}.parquet", compression="zstd")
        print(f"{dt}: {len(f)} 個", flush=True)
    print("done")


if __name__ == "__main__":
    main()
