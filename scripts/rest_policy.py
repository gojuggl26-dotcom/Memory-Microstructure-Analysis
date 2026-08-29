"""トレード頻度を落とすべきか — 「置いたまま」方策の測定。

【問い】
  現行は最良気配が動くたびに取消・再発注する(= 追いかける)。これは
    ・発注レートが上限の 116〜122%(報告 33 §4)→ 実行不能
    ・遅延 100ms で δ が 0.7096 → 0.1826 bp に崩れる(報告 36 §10)
  頻度を落とせばどちらも緩和されるはずだが、代償があるかを測る。

【方策】
  自分の指値から最良気配が **k ティック以上離れたときだけ**建て直す。
    k=0  … 最良が動くたびに建て直す(現行)
    k 大 … 置きっぱなしに近づく

【何が起きるか(事前の見立て)】
  (−) 置きっぱなしの指値は「相場が自分のところへ降りてきたとき」に約定する。
      これは逆選択そのもので、δ が落ちるはず
  (+) 追いかけるたびに行列の最後尾に付き直している。置けば先頭に到達できる
      (t_front の機構。報告 19 の気配改善が効いたのと同じ理由)
  (+) 発注レートが下がる
  どちらが勝つかは事前には決まらない。だから測る。

【★決定的な検証】
  頻度を落とす主張の核心は「レースに参加しなければ遅延は効かない」である。
  したがって **遅延 100ms のもとで k>0 が k=0 に勝つか**が判定である。

【時間契約】
  建て直しの判定はその時刻の板のみ。注文は t+lat_open に有効化、
  取消は判定時刻 +lat_cancel に効く。未来の価格は使わない。

【実装】
  価格変化イベントと約定を時刻順にマージした 1 本のループで処理する。
  在庫は両側で共有されるので、約定は発生時点で在庫に反映させる
  (取消時にまとめて精算すると在庫の時系列が崩れる)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from capacity_v3 import load, TICK, FEE_BP          # noqa: E402
from backtester_v3 import load_model                # noqa: E402

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
MS = 1_000_000


def simulate_rest(d, qty, inv_limit, k_ticks, lat_open=0, lat_cancel=0):
    mp = d["mp"]
    mts = mp["ts"].to_numpy()
    bb = mp["best_bid"].to_numpy(); ba = mp["best_ask"].to_numpy()
    mid = mp["mid"].to_numpy()
    fl = d["fill"]
    f_t = fl["t"].to_numpy(); f_px = fl["px"].to_numpy()
    f_sz = fl["sz"].to_numpy(); f_side = fl["side"].to_numpy()

    # --- イベント列を作る -------------------------------------------------
    # 種別 0 = 気配変化(買い側), 1 = 気配変化(売り側), 2 = 約定
    ev_t = []; ev_k = []; ev_i = []
    for kind, ser in ((0, bb), (1, ba)):
        chg = np.flatnonzero(np.r_[True, ser[1:] != ser[:-1]])
        ev_t.append(mts[chg]); ev_k.append(np.full(chg.size, kind, np.int8))
        ev_i.append(chg)
    ev_t.append(f_t); ev_k.append(np.full(f_t.size, 2, np.int8))
    ev_i.append(np.arange(f_t.size))
    ev_t = np.concatenate(ev_t); ev_k = np.concatenate(ev_k); ev_i = np.concatenate(ev_i)
    o = np.argsort(ev_t, kind="stable")
    ev_t = ev_t[o]; ev_k = ev_k[o]; ev_i = ev_i[o]

    lvl = d["lvl"]
    # k_ticks = -1 は「気配変化のたび無条件に建て直す」= capacity_v3 と同じ方策。
    # 対照として用意する(本エンジンの検算に使う)。
    always = k_ticks < 0
    tol = max(k_ticks, 0) * TICK + 1e-9
    # 各側の生存注文リスト。要素 = [価格, 有効時刻, 先頭到達時刻, 残数量, 失効時刻]
    # ★取消にも遅延がある。取消を決めた時刻 t の注文は t+lat_cancel まで約定しうる
    #   (stale な指値を食われる経路。報告 35 §5-3 の (ii))。
    #   したがって取消直後は「消えつつある古い注文」と「これから効く新しい注文」が
    #   一時的に併存する。lat_open = lat_cancel なら実質どの瞬間も 1 本だけ生きる。
    INF = np.int64(np.iinfo(np.int64).max)
    st = {0: [], 1: []}              # 0 = 買い, 1 = 売り
    cash = 0.0; pos = 0.0
    n_place = 0; n_fill = 0; vol = 0.0
    gross_num = 0.0                  # 数量加重の粗利 [bp·単位]
    eq_min = 0.0

    def place(side_k, t_now, i0):
        """side_k 側に発注する。side_k: 0=買い, 1=売り"""
        nonlocal n_place
        s_dir = 1.0 if side_k == 0 else -1.0
        if inv_limit > 0 and s_dir * pos >= inv_limit:
            return None
        p = bb[i0] if side_k == 0 else ba[i0]
        px_ = p
        if (ba[i0] - bb[i0]) > 2 * TICK * 1.5:            # A 気配改善
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
        return [px_, t_eff, t_front, qty, INF]

    for n in range(ev_t.size):
        t = ev_t[n]; kind = ev_k[n]; idx = ev_i[n]
        if kind == 2:
            # --- 約定イベント: 自分の指値に当たるか ---
            tp = f_px[idx]; tz = f_sz[idx]
            # テイカー売り → 買い板に当たる。capacity_v3 と同じ対応
            side_k = 0 if f_side[idx] == "A" else 1
            lst = st[side_k]
            if not lst:
                continue
            for o_ in lst:
                if t >= o_[4]:                             # 取消が効いた注文
                    continue
                if abs(tp - o_[0]) > 1e-9:
                    continue
                if t < o_[1] or t < o_[2]:                 # 未到達 / 行列の後ろ
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
                break                                      # 1 約定は 1 注文にしか当たらない
            st[side_k] = [o for o in lst if o[3] > 1e-9 and t < o[4]]
        else:
            side_k = kind
            best = bb[idx] if side_k == 0 else ba[idx]
            lst = [o for o in st[side_k] if t < o[4]]       # 取消済みを掃除
            live = [o for o in lst if o[4] == INF]          # 取消を決めていない注文
            if live:
                o_ = live[0]
                # 建て直すか: 自分の指値から最良が k ティック以上離れたら
                if always or abs(best - o_[0]) > tol:
                    o_[4] = int(t) + lat_cancel             # ★取消は遅れて効く
                    live = []
            if not live:
                new = place(side_k, int(t), int(idx))
                if new is not None:
                    lst.append(new)
            st[side_k] = lst
            # 在庫が上限側なら place が None を返すのでそのまま
        if (n & 8191) == 0:
            j0 = int(np.searchsorted(mts, t, side="right")) - 1
            eq_min = min(eq_min, cash + pos * mid[max(j0, 0)])

    if vol <= 0:
        return None
    pnl = cash + pos * mid[-1]
    ops_per_s = n_place * 2 / 86400.0
    return {"k_ticks": k_ticks, "lat_ms": lat_open // MS,
            "pnl_usd": float(pnl), "n_place": n_place, "n_fill": n_fill,
            "volume": float(vol), "gross_bp": float(gross_num / vol),
            "ops_per_s": float(ops_per_s), "eq_min": float(eq_min),
            "end_inv": float(pos)}


def main() -> None:
    model = load_model()
    days = sorted(pl.read_csv(D / "capacity_100usd.csv")["dt"].unique().to_list())
    if len(sys.argv) > 1:
        days = days[::int(sys.argv[1])]
    q = float(pl.read_csv(D / "capacity_100usd.csv")
              .filter(pl.col("lev") == 1)["qty"][0])
    KS = [-1, 0, 1, 2, 3, 5, 10, 20, 50]
    CFG = [(k, lat) for lat in (0, 100 * MS) for k in KS]
    # 途中で止まっても続きから走れるようにする(既存 CSV の完了日は飛ばす)
    out = D / "rest_policy.csv"
    rows = []
    if out.exists():
        old = pl.read_csv(out)
        done = {dt for dt, c in old.group_by("dt").len().iter_rows()
                if c == len(CFG)}
        rows = old.filter(pl.col("dt").is_in(list(done))).to_dicts()
        days = [d_ for d_ in days if d_ not in done]
        print(f"再開: 完了 {len(done)} 日 / 残り {len(days)} 日", flush=True)
    for i, dt in enumerate(days):
        d = load(dt, model)
        if d is None:
            continue
        for k, lat in CFG:
            r = simulate_rest(d, q, 5.0 * q, k, lat, lat)
            if r:
                rows.append({"dt": dt, **r})
        print(f"{i+1}/{len(days)} {dt}", flush=True)
        pl.DataFrame(rows).write_csv(D / "rest_policy.csv")
    pl.DataFrame(rows).write_csv(D / "rest_policy.csv")
    print("完了")


if __name__ == "__main__":
    main()
