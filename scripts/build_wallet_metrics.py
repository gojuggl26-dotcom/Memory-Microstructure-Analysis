"""ウォレット(user)ごとに 24 の板行動指標を算出する。

【入力】
l1 の注文イベント(取得済み)+ `user` 列(fetch_l1_user.py で追加取得)。
板の再構成は build_elasticity.py と同じ方式(ティック番号の密な配列 + 遅延削除の
ヒープ)。**数量の残差 1e-9 を 0 と見なさないと板が壊れる**(密な配列で足し引きを
繰り返すと 1e-15 が残り、消えた価格が生き続けて板が常時クロスする。実測済み)。

【24 の指標】
時間加重(その値で過ごした時間の重み)で出すものと、注文の母集団で出すものがある。

  板の状態(時間加重の平均。1 日 86,400 秒で割る)
    number of active orders     生きている注文の本数
    total bid size / ask size   買い・売りの合計数量
    net displayed inventory     total bid − total ask(板に見えている偏り)
    displayed notional          Σ 数量 × 価格 [USD]
    number of price levels      注文を置いている異なる価格の数
  気配の位置(数量 × 時間で重みづけ)
    mean / bid / ask quote distance   mid からの距離 [bp]
    quoted spread               自分の最良買いと最良売りの差 [bp]
    quote symmetry              1 − |bid距離 − ask距離| / (bid距離 + ask距離)
  注文の性質(注文を単位に)
    order size mean / distribution    平均と分位(p10/50/90)と変動係数
    order lifetime              生存時間の中央値 [s]
    update frequency            1 時間あたりの発注数
    cancel/removal frequency    1 時間あたりの取消数
    quote persistence           1 秒以上生きた注文の割合
    liquidity contribution      mid ±10bp の板厚に占める自分の割合
  在席(1 秒刻みの断面で数える)
    BBO presence frequency      最良買いか最良売りに居た時間の割合
    bid-touch / ask-touch presence
    two-sided / one-sided presence
    quote uptime                注文を 1 本でも置いていた時間の割合

【x が確定する時刻】
すべてその日の中で閉じた記述統計であり、将来の値は使っていない。
リターンとの関係を見る場合は、指標が確定する時刻(日の終わり)より後の
リターンに限ること。

【★右打ち切り】
日の終わりまでに終端イベントが来ない注文は、その日の終わりで打ち切って扱う
(`n_censored` に件数を残す)。生存時間の中央値はこの打ち切りの影響を受ける。

    uv run python scripts/build_wallet_metrics.py --coin xyz:MU
出力: data/wallet_<coin>/dt=*.parquet … 日 × ウォレット × 24 指標
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
SNAP_NS = 1_000_000_000
PER_DAY = 24 * 60 * 60
DAY_NS = PER_DAY * SNAP_NS
EPS_Q = 1e-9
NEAR_BP = 10.0                 # liquidity contribution を測る帯
PERSIST_S = 1.0                # quote persistence の閾値
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}


def load_day(l1: Path, l1u: Path) -> pl.DataFrame:
    d = pl.read_parquet(l1)
    u = pl.read_parquet(l1u)
    if u.height != d.height:
        sys.exit(f"{l1.name}: user の行数が合わない")
    # ★l1 の ts は datetime[ns]。整数で差分を取るので明示的に落とす
    return (d.with_columns(user=u["user"])
             .with_columns(pl.col("ts").cast(pl.Int64)).sort("ts"))


def run_day(day: str, d: pl.DataFrame) -> pl.DataFrame:
    ts = d["ts"].to_numpy()
    isbid = (d["side"].to_numpy() == "B")
    px = np.rint(d["px"].to_numpy() / TICK).astype(np.int64)
    st = d["status"].to_numpy()
    osz = d["orig_sz"].to_numpy()
    trg = d["is_trigger"].to_numpy()
    tif = d["tif"].to_numpy()
    oid = d["oid"].to_numpy()
    usr = d["user"].to_numpy()

    lo_t = int(px.min()) - 2
    n_t = int(px.max()) + 3 - lo_t
    depb, depa = np.zeros(n_t), np.zeros(n_t)
    ub: dict[int, dict] = {}          # bid ティック -> {user: 本数}
    ua: dict[int, dict] = {}
    hb: list[int] = []
    ha: list[int] = []
    inb: set[int] = set()
    ina: set[int] = set()
    live: dict[int, tuple] = {}

    t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
    snap_t = t0 + (np.arange(PER_DAY, dtype=np.int64) + 1) * SNAP_NS
    cut = np.searchsorted(ts, snap_t, side="right")

    # ウォレットごとの状態と時間加重の積算
    W: dict[str, dict] = {}
    T_LAST: dict[str, int] = {}

    def wal(u):
        w = W.get(u)
        if w is None:
            w = W[u] = {"nb": 0, "na": 0, "sb": 0.0, "sa": 0.0, "no": 0.0,
                        "lv": {}, "i_n": 0.0, "i_sb": 0.0, "i_sa": 0.0,
                        "i_no": 0.0, "i_lv": 0.0, "t_any": 0, "t_two": 0,
                        "t_one": 0, "t_bid": 0, "t_ask": 0,
                        "s_bidtouch": 0, "s_asktouch": 0, "s_bbo": 0}
            T_LAST[u] = t0
        return w

    def flush(u, t):
        """状態が変わる直前までの時間を積算する。"""
        w = W[u]
        dt = t - T_LAST[u]
        if dt <= 0:
            T_LAST[u] = t
            return
        n = w["nb"] + w["na"]
        w["i_n"] += n * dt
        w["i_sb"] += w["sb"] * dt
        w["i_sa"] += w["sa"] * dt
        w["i_no"] += w["no"] * dt
        w["i_lv"] += len(w["lv"]) * dt
        if n > 0:
            w["t_any"] += dt
            if w["nb"] > 0 and w["na"] > 0:
                w["t_two"] += dt
            else:
                w["t_one"] += dt
            if w["nb"] > 0:
                w["t_bid"] += dt
            if w["na"] > 0:
                w["t_ask"] += dt
        T_LAST[u] = t

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

    mid_s = np.full(PER_DAY, np.nan)
    i = 0
    for s_i in range(PER_DAY):
        end = cut[s_i]
        while i < end:
            s = st[i]
            if s == "open":
                if trg[i] or tif[i] not in RESTING:
                    i += 1
                    continue
                b, t, q, u = bool(isbid[i]), int(px[i]), float(osz[i]), usr[i]
                dep, umap = (depb, ub) if b else (depa, ua)
                if dep[t - lo_t] <= EPS_Q:
                    if b:
                        if t not in inb:
                            heapq.heappush(hb, -t); inb.add(t)
                    elif t not in ina:
                        heapq.heappush(ha, t); ina.add(t)
                dep[t - lo_t] += q
                m = umap.setdefault(t, {})
                m[u] = m.get(u, 0) + 1
                w = wal(u)
                flush(u, ts[i])
                if b:
                    w["nb"] += 1; w["sb"] += q
                else:
                    w["na"] += 1; w["sa"] += q
                w["no"] += q * t * TICK
                w["lv"][(b, t)] = w["lv"].get((b, t), 0) + 1
                live[int(oid[i])] = (b, t, q, u, ts[i])
            elif s == "filled" or s in CANCEL:
                o = live.pop(int(oid[i]), None)
                if o is not None:
                    b, t, q, u, _ = o
                    dep, umap = (depb, ub) if b else (depa, ua)
                    v = dep[t - lo_t] - q
                    dep[t - lo_t] = v if v > EPS_Q else 0.0
                    m = umap.get(t)
                    if m is not None:
                        c = m.get(u, 0) - 1
                        if c <= 0:
                            m.pop(u, None)
                            if not m:
                                umap.pop(t, None)
                        else:
                            m[u] = c
                    w = W[u]
                    flush(u, ts[i])
                    if b:
                        w["nb"] -= 1; w["sb"] -= q
                    else:
                        w["na"] -= 1; w["sa"] -= q
                    w["no"] -= q * t * TICK
                    c = w["lv"].get((b, t), 0) - 1
                    if c <= 0:
                        w["lv"].pop((b, t), None)
                    else:
                        w["lv"][(b, t)] = c
            i += 1

        bb, ba = best_bid(), best_ask()
        if bb is None or ba is None or ba <= bb:
            continue
        mid_s[s_i] = (bb + ba) / 2.0 * TICK
        at_b = set(ub.get(bb, {}))
        at_a = set(ua.get(ba, {}))
        for u in at_b:
            wal(u)["s_bidtouch"] += 1
        for u in at_a:
            wal(u)["s_asktouch"] += 1
        for u in at_b | at_a:
            wal(u)["s_bbo"] += 1

    for u in list(W):
        flush(u, t0 + DAY_NS)

    # ---- 注文を単位にした指標(打ち切りは日の終わりで揃える)------------------
    rest = (st == "open") & (~trg) & np.isin(tif, list(RESTING))
    op = pl.DataFrame({"oid": oid[rest], "user": usr[rest], "bid": isbid[rest],
                       "px": px[rest] * TICK, "sz": osz[rest], "t_open": ts[rest]})
    term = (st == "filled") | np.isin(st, list(CANCEL))
    tm = (pl.DataFrame({"oid": oid[term], "t_close": ts[term],
                        "is_cancel": np.isin(st[term], list(CANCEL))})
          .group_by("oid").first())
    L = (op.join(tm, on="oid", how="left")
           .with_columns(censored=pl.col("t_close").is_null(),
                         t_close=pl.col("t_close").fill_null(t0 + DAY_NS))
           .with_columns(dur_s=(pl.col("t_close") - pl.col("t_open")) / 1e9))
    # mid の時間加重平均(1 秒刻みの mid を階段関数として積分)
    mfill = pl.Series(mid_s).fill_nan(None).fill_null(strategy="forward") \
              .fill_null(strategy="backward").to_numpy()
    C = np.concatenate([[0.0], np.cumsum(mfill)]) * SNAP_NS
    a = np.clip((L["t_open"].to_numpy() - t0) // SNAP_NS, 0, PER_DAY)
    b = np.clip((L["t_close"].to_numpy() - t0) // SNAP_NS, 0, PER_DAY)
    # ★1 秒未満で消える注文は a == b になる。差分を取ると 0 になってしまうので、
    #   その場合はその秒の mid をそのまま使う(実測で注文の過半がここに落ちる)
    inst = mfill[np.clip(a, 0, PER_DAY - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        mid_bar = np.where(b > a, (C[b] - C[a]) / (np.maximum(b - a, 1) * SNAP_NS), inst)
    mid_bar = np.where(np.isfinite(mid_bar) & (mid_bar > 0), mid_bar, np.nan)
    L = L.with_columns(mid_bar=mid_bar,
                       dist_bp=np.abs(L["px"].to_numpy() - mid_bar) / mid_bar * 1e4)
    L = L.with_columns(wgt=pl.col("sz") * pl.col("dur_s"))

    tot_near = float(L.filter(pl.col("dist_bp") <= NEAR_BP)["wgt"].sum())
    G = L.group_by("user").agg(
        n_orders_placed=pl.len(),
        n_censored=pl.col("censored").sum(),
        order_size_mean=pl.col("sz").mean(),
        order_size_p10=pl.col("sz").quantile(0.1),
        order_size_p50=pl.col("sz").quantile(0.5),
        order_size_p90=pl.col("sz").quantile(0.9),
        order_size_cv=pl.col("sz").std() / pl.col("sz").mean(),
        order_lifetime_med=pl.col("dur_s").median(),
        order_lifetime_mean=pl.col("dur_s").mean(),
        quote_persistence=(pl.col("dur_s") >= PERSIST_S).mean(),
        n_cancel=pl.col("is_cancel").sum(),
        mean_quote_distance=((pl.col("dist_bp") * pl.col("wgt")).filter(pl.col("dist_bp").is_not_nan()).sum()
                             / pl.col("wgt").filter(pl.col("dist_bp").is_not_nan()).sum()),
        bid_quote_distance=((pl.col("dist_bp") * pl.col("wgt"))
                            .filter(pl.col("bid") & pl.col("dist_bp").is_not_nan()).sum()
                            / pl.col("wgt").filter(pl.col("bid") & pl.col("dist_bp").is_not_nan()).sum()),
        ask_quote_distance=((pl.col("dist_bp") * pl.col("wgt"))
                            .filter(~pl.col("bid") & pl.col("dist_bp").is_not_nan()).sum()
                            / pl.col("wgt").filter(~pl.col("bid") & pl.col("dist_bp").is_not_nan()).sum()),
        liq_contribution=(pl.col("wgt").filter(pl.col("dist_bp") <= NEAR_BP).sum()
                          / (tot_near if tot_near > 0 else float("nan"))),
    )

    rows = []
    for u, w in W.items():
        up = w["t_any"] / DAY_NS
        rows.append({
            "dt": day, "user": u,
            "n_active_orders": w["i_n"] / DAY_NS,
            "total_bid_size": w["i_sb"] / DAY_NS,
            "total_ask_size": w["i_sa"] / DAY_NS,
            "net_displayed_inventory": (w["i_sb"] - w["i_sa"]) / DAY_NS,
            "displayed_notional": w["i_no"] / DAY_NS,
            "n_price_levels": w["i_lv"] / DAY_NS,
            "quote_uptime": up,
            "two_sided_presence": w["t_two"] / DAY_NS,
            "one_sided_presence": w["t_one"] / DAY_NS,
            "bid_presence": w["t_bid"] / DAY_NS,
            "ask_presence": w["t_ask"] / DAY_NS,
            "bid_touch_presence": w["s_bidtouch"] / PER_DAY,
            "ask_touch_presence": w["s_asktouch"] / PER_DAY,
            "bbo_presence": w["s_bbo"] / PER_DAY,
        })
    B = pl.DataFrame(rows)
    out = B.join(G, on="user", how="left").with_columns(
        quoted_spread=pl.col("bid_quote_distance") + pl.col("ask_quote_distance"),
        quote_symmetry=1 - (pl.col("bid_quote_distance") - pl.col("ask_quote_distance")).abs()
        / (pl.col("bid_quote_distance") + pl.col("ask_quote_distance")),
        update_freq_per_h=pl.col("n_orders_placed") / 24.0,
        cancel_freq_per_h=pl.col("n_cancel") / 24.0,
        cancel_ratio=pl.col("n_cancel") / pl.col("n_orders_placed"),
    )
    return out.sort("displayed_notional", descending=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    l1dir = ROOT / "data" / f"l1_{tag}"
    l1udir = ROOT / "data" / f"l1u_{tag}"
    outdir = ROOT / "data" / f"wallet_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    days = sorted(p.stem.split("=")[1] for p in l1udir.glob("dt=*.parquet"))
    if a.days:
        days = days[: a.days]
    todo = [d for d in days if not (outdir / f"dt={d}.parquet").exists()]
    print(f"[日] user のある日 {len(days)} / 未処理 {len(todo)}", file=sys.stderr)

    for n, day in enumerate(todo, 1):
        d = load_day(l1dir / f"dt={day}.parquet", l1udir / f"dt={day}.parquet")
        R = run_day(day, d)
        R.write_parquet(outdir / f"dt={day}.parquet", compression="zstd")
        print(f"  {day}  ウォレット {R.height:,}  "
              f"上位の板金額 ${R['displayed_notional'][0]:,.0f}  ({n}/{len(todo)})",
              file=sys.stderr)
    print(f"-> {outdir}/dt=*.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
