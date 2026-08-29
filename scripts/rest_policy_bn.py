"""報告 44 の Binance リードラグ信号を、退出規則として損益で検証する。

【規則(各時点で判定できる形 — CLAUDE.md §D-16)】
  時刻 t に自分が知っている Binance の情報は **t − lat_read までのもの**。
  その直近 Δ の対数リターン sig を見て
      sig < −thr  → **買い気配を引く**(価格が下がる = 自分の買いが食われる側)
      sig > +thr  → **売り気配を引く**
  引いた側は t + hold_ns まで再発注しない。取消自体も lat_cancel だけ遅れて効く
  (stale を食われる経路を残す)。

【時間契約】
  sig は [t − lat_read − Δ, t − lat_read) の窓で、**t の時点で確定している**。
  未来の Binance 価格は一切使わない。

【判定】
  対照は「同じ方策で信号を使わない場合」(thr = ∞)。ゼロとの比較ではない
  (CLAUDE.md §B-5: 主判定がベースライン依存で自動成立しないようにする)。

【検算】
  thr = ∞ のとき rest_policy.simulate_rest と**完全一致**することを確認してから使う。

【出力】 data/rest_policy_bn.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from capacity_v3 import TICK, FEE_BP, load          # noqa: E402
from backtester_v3 import load_model                # noqa: E402

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
MS = 1_000_000
INF64 = np.int64(np.iinfo(np.int64).max)


def binance_signal(dt: str, win_ns: int, lat_read_ns: int):
    """(信号が届く時刻, その時点の直近 win の対数リターン[bp]) を返す。"""
    f = D / f"binance/dt={dt}.parquet"
    if not f.exists():
        return None
    b = pl.read_parquet(f).sort("ts_ms")
    if len(b) < 2000:
        return None
    ts = b["ts_ms"].to_numpy().astype(np.int64) * MS
    lp = np.log(b["mid_proxy"].to_numpy())
    j = np.searchsorted(ts, ts - win_ns, side="right") - 1
    ok = j >= 0
    sig = np.zeros(len(ts))
    sig[ok] = (lp[ok] - lp[j[ok]]) * 1e4
    return ts + lat_read_ns, sig


def simulate(d, qty, inv_limit, k_ticks, lat_open, lat_cancel,
             sig_t=None, sig_v=None, thr=np.inf, hold_ns=0):
    """rest_policy.simulate_rest に「Binance 信号で片側を引く」を足したもの。"""
    mp = d["mp"]
    mts = mp["ts"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    mid = mp["mid"].to_numpy()
    fl = d["fill"]
    f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy()
    f_sz = fl["sz"].to_numpy(); f_side = fl["side"].to_numpy()

    ev_t = []; ev_k = []; ev_i = []
    for kind, ser in ((0, bb), (1, ba)):
        chg = np.flatnonzero(np.r_[True, ser[1:] != ser[:-1]])
        ev_t.append(mts[chg]); ev_k.append(np.full(chg.size, kind, np.int8))
        ev_i.append(chg)
    ev_t.append(f_t); ev_k.append(np.full(f_t.size, 2, np.int8))
    ev_i.append(np.arange(f_t.size))
    use_sig = sig_t is not None and np.isfinite(thr)
    if use_sig:                                   # 種別 3 = Binance 信号の到着
        m = np.abs(sig_v) > thr                   # 閾値未満は何もしないので event にしない
        ev_t.append(sig_t[m]); ev_k.append(np.full(int(m.sum()), 3, np.int8))
        ev_i.append(np.flatnonzero(m))
    ev_t = np.concatenate(ev_t); ev_k = np.concatenate(ev_k); ev_i = np.concatenate(ev_i)
    o = np.argsort(ev_t, kind="stable")
    ev_t = ev_t[o]; ev_k = ev_k[o]; ev_i = ev_i[o]

    lvl = d["lvl"]
    always = k_ticks < 0
    tol = max(k_ticks, 0) * TICK + 1e-9
    st = {0: [], 1: []}
    blocked = {0: np.int64(-1), 1: np.int64(-1)}   # その時刻まで再発注しない
    cash = 0.0; pos = 0.0
    n_place = 0; n_fill = 0; vol = 0.0; n_pull = 0
    gross_num = 0.0
    eq_min = 0.0

    def place(side_k, t_now, i0):
        nonlocal n_place
        if t_now < blocked[side_k]:
            return None
        s_dir = 1.0 if side_k == 0 else -1.0
        if inv_limit > 0 and s_dir * pos >= inv_limit:
            return None
        p = bb[i0] if side_k == 0 else ba[i0]
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:
            px_ = p + TICK if side_k == 0 else p - TICK
        n_place += 1
        t_eff = t_now + lat_open
        t_front = t_eff
        if px_ == p:
            lv = lvl.get(int(round(px_ * 1e6)) * 2 + (side_k == 0))
            if lv is not None:
                jj = int(np.searchsorted(lv[0], t_now, side="right")) - 1
                if jj >= 0 and lv[1][jj] > t_eff:
                    t_front = int(lv[1][jj])
        return [px_, t_eff, t_front, qty, INF64]

    for n in range(ev_t.size):
        t = ev_t[n]; kind = ev_k[n]; idx = ev_i[n]
        if kind == 3:
            # --- Binance 信号: 不利な側を引く ---
            s = sig_v[idx]
            side_k = 0 if s < 0 else 1             # 下落 → 買いを引く / 上昇 → 売りを引く
            blocked[side_k] = max(blocked[side_k], t + hold_ns)
            lst = [o_ for o_ in st[side_k] if t < o_[4]]
            for o_ in lst:
                if o_[4] == INF64:
                    o_[4] = int(t) + lat_cancel    # ★取消も遅れて効く
                    n_pull += 1
            st[side_k] = lst
            continue
        if kind == 2:
            tp = f_px[idx]; tz = f_sz[idx]
            side_k = 0 if f_side[idx] == "A" else 1
            lst = st[side_k]
            if not lst:
                continue
            for o_ in lst:
                if t >= o_[4] or abs(tp - o_[0]) > 1e-9:
                    continue
                if t < o_[1] or t < o_[2]:
                    continue
                take = min(o_[3], float(tz))
                if take <= 0:
                    continue
                s_dir = 1.0 if side_k == 0 else -1.0
                px_ = o_[0]
                cash += -s_dir * px_ * take - FEE_BP / 1e4 * px_ * take
                pos += s_dir * take
                vol += take; n_fill += 1; o_[3] -= take
                j0 = int(np.searchsorted(mts, t, side="right")) - 1
                m0 = mid[max(j0, 0)]
                gross_num += s_dir * (m0 - px_) / m0 * 1e4 * take
                eq_min = min(eq_min, cash + pos * m0)
                break
            st[side_k] = [o_ for o_ in lst if o_[3] > 1e-9 and t < o_[4]]
        else:
            side_k = kind
            best = bb[idx] if side_k == 0 else ba[idx]
            lst = [o_ for o_ in st[side_k] if t < o_[4]]
            live = [o_ for o_ in lst if o_[4] == INF64]
            if live:
                o_ = live[0]
                if always or abs(best - o_[0]) > tol:
                    o_[4] = int(t) + lat_cancel
                    live = []
            if not live:
                new = place(side_k, int(t), int(idx))
                if new is not None:
                    lst.append(new)
            st[side_k] = lst
        if (n & 8191) == 0:
            j0 = int(np.searchsorted(mts, t, side="right")) - 1
            eq_min = min(eq_min, cash + pos * mid[max(j0, 0)])

    if vol <= 0:
        return None
    return {"pnl_usd": float(cash + pos * mid[-1]), "n_place": n_place,
            "n_fill": n_fill, "n_pull": n_pull, "volume": float(vol),
            "gross_bp": float(gross_num / vol),
            "ops_per_s": float(n_place * 2 / 86400.0),
            "eq_min": float(eq_min), "end_inv": float(pos)}


WIN_NS = 250 * MS            # Binance 側の窓 250ms(報告 44 の最良付近)
LAT_READ_NS = 100 * MS       # Binance を見るまでの遅延(報告 45 の実測 86〜108ms)
# (閾値 bp, 引いておく時間 ms)。thr=inf が対照(信号なし)
CFG = [(np.inf, 0), (1.0, 500), (2.0, 500), (4.0, 500), (2.0, 250), (2.0, 1000)]
LATS = [0, 100 * MS]
K = 0                        # 報告 37 の最良方策


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    q = float(pl.read_csv(D / "capacity_100usd.csv")
              .filter(pl.col("lev") == 1)["qty"][0])
    out = D / "rest_policy_bn.csv"
    rows = []
    if out.exists():
        old = pl.read_csv(out)
        done = {dt for dt, c in old.group_by("dt").len().iter_rows()
                if c == len(CFG) * len(LATS)}
        rows = old.filter(pl.col("dt").is_in(list(done))).to_dicts()
        days = [d_ for d_ in days if d_ not in done]
        print(f"再開: 完了 {len(done)} 日 / 残り {len(days)} 日", flush=True)
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        sg = binance_signal(dt, WIN_NS, LAT_READ_NS)
        if sg is None:
            print(f"  {dt}: Binance データなし — 飛ばす", flush=True)
            continue
        for lat in LATS:
            for thr, hold in CFG:
                r = simulate(d, q, 5.0 * q, K, lat, lat,
                             sg[0], sg[1], thr, hold * MS)
                if r:
                    rows.append({"dt": dt, "lat_ms": lat // MS,
                                 "thr_bp": float(thr), "hold_ms": hold, **r})
        print(f"{i+1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(out)
    pl.DataFrame(rows).write_csv(out)
    print("完了")


if __name__ == "__main__":
    main()
