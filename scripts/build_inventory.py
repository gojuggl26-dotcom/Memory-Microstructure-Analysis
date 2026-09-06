"""両側にメイカーを出し続け、在庫を FIFO で相殺する事象駆動シミュレータ。

    uv run python scripts/build_inventory.py --coin xyz:MU [--qmax 1] [--lat 0.065]

これまでは 1 約定ごとに即座に手仕舞っていた。そこでは出口の脚に必ず
半スプレッドかテイカー手数料がかかるので、負けるのが当たり前だった。
ここでは**在庫 q_t を持つ**。買いで約定したら q=+1、その後の売り約定は
「新規ショート」ではなく**まず既存ロングの相殺**に使う(FIFO)。

    PnL_pair = (p_sell − p_buy)/mid_open × 1e4 − 2·Fee_maker

在庫上限 qmax(既定 1)
----------------------
上限を入れないと在庫は乱歩して |q| が数十まで伸び、FIFO の「相殺までの時間」は
市場ではなく**自分の在庫の行列長**を測ってしまう(実測で中央 6,800 秒)。
|q| が上限にあるとき、それを増やす側は出さない。つまり qmax=1 なら

    q=0  → 両側を出す
    q=+1 → 売りだけ(在庫を減らす注文)
    q=−1 → 買いだけ

となり、指示された「entry 注文と inventory-reducing 注文を分ける」状態機械の
最小形になっている。

発注の規則
----------
- `--mode follow`(既定): 自分側の最良気配が動いた瞬間に出し直す
  (待ち行列の最後尾に付き直す)。動かない間は並び続ける
- `--mode fixed`: 一度出したら値段を動かさず、約定するか T_hold 秒経つまで置く

★ 約定と気配変化が**同じブロック時刻**になることが多い(板を消し切る約定が
気配を動かすため)。有効期限の判定は `tau <= exp` にする。`<` にすると
1 日 3,742 件の約定を取りこぼし、約定率が 6.61% → 1.75% に落ちる。

条件つきの出し直し(--requote)
------------------------------
`follow` は自分側の気配が動くたびに無条件で取り消して並び直す。これは
**待ち行列の順番を毎回捨てている**。順番に +1.15 bp の価値があると分かった
以上、捨てるかどうかは条件で決めるべきである。

    always : 既定。毎回取り消して最良気配に出し直す(従来の挙動)
    cond   : 自分側の最良が **mid 側へ動いた**ときだけ維持する。
             このとき自分の指値は最良より不利な位置に隠れ、最良は他人が作る。
             幽霊注文(自分の注文が板に影響しない)の仮定が壊れない場面だけを選ぶ
    keep   : 向きを問わず維持する。mid から遠ざかる向きでも維持するので、
             **自分が唯一の最良気配になってしまう**。幽霊注文の仮定が効くため
             上界としてのみ読むこと

★ 維持する場合、**約定しない注文も板を占有させる**。占有させないと
「その注文が約定するかどうか」を見てから持ち続けるか捨てるかを決めることに
なり、各時点で判定できない規則になる。実際、占有させずに試した最初の実装は
+11.2 bp/組(半スプレッドの 27 倍)という明らかに過大な値を出した。

`--keepmax` で維持の上限秒数を入れる(既定 60 秒)。維持中に在庫が動いて
上限を破る側になった注文は取り消す(always では従来どおり判定しない。
公表済みの結果では |q| の最大が 1 で、上限を破る事象は起きていない)。

在庫の評価
----------
相殺できなかった在庫を捨てないため、

    Equity_t = RealizedPnL_t + q_t·(Mid_t − 平均建値)

とし、**日の終わりに残った在庫はテイカーで強制決済**して費用も計上する。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (GRID_NS, PX_UNIT, SZ_LOT,  # noqa: E402
                              clean_bbo, day_features as ladder_day)
from build_quotes import DAY_NS, NG, fill_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
MAKER_FEE = 0.088
TAKER_FEE = 0.846
HS = [0.25, 1.0, 5.0, 10.0, 30.0, 60.0]
QCAP = 60


def posts(ts, px):
    """自分側の気配が動いた点。次に動くまでがその注文の有効期間。"""
    chg = np.empty(px.size, bool)
    chg[0] = True
    chg[1:] = px[1:] != px[:-1]
    i = np.flatnonzero(chg)
    exp = np.empty(i.size, np.int64)
    exp[:-1] = ts[i[1:]]
    exp[-1] = ts[-1] + 1
    return i, exp


def day_arrays(dt, bpath, fpath):
    d = clean_bbo(pl.scan_parquet(bpath).filter(pl.col("dt") == dt)
                  .collect())[0].sort("ts")
    ts = d["ts"].cast(pl.Int64).to_numpy()
    pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
    qb, qa = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
    nb = ts.size
    d0 = int(ts[0]) // DAY_NS * DAY_NS
    tg = d0 + np.arange(NG, dtype=np.int64) * 10 ** 9
    gj = np.clip(np.searchsorted(ts, tg, side="right") - 1, 0, nb - 1)
    F = pl.scan_parquet(fpath).filter(
        pl.col("crossed") & (pl.col("dt") == dt)).select(
        "ts", "px", "sz", "side").collect().sort("ts")
    return (ts, pb, pa, qb, qa, 0.5 * (pb + pa), d0,
            np.round(pb[gj] / PX_UNIT), np.round(pa[gj] / PX_UNIT),
            F["ts"].cast(pl.Int64).to_numpy(),
            np.round(F["px"].to_numpy() / PX_UNIT),
            F["sz"].to_numpy(), F["side"].to_numpy() == "B")


def simulate(dt, bpath, fpath, lat_ns, qmax, hold_ns, tmax_ns=0,
             gate=None, exitk=0, lad=None, improve=0, front=False,
             requote="always", keepmax_ns=0, keepdt=0):
    (ts, pb, pa, qb, qa, mid, d0, btg, atg,
     ft, fpx, fsz, fbuy) = day_arrays(dt, bpath, fpath)
    nb = ts.size

    P = {}
    for sgn, px, qsz, grid, msk in ((1, pb, qb, btg, ~fbuy),
                                    (-1, pa, qa, atg, fbuy)):
        i, exp = posts(ts, px)
        t0 = ts[i] + lat_ns
        if hold_ns:                      # 値段を固定して置く上限
            exp = np.minimum(exp, t0 + hold_ns)
        P[sgn] = {"i": i, "t": ts[i], "exp": exp}
        # k=0 は最良気配(新規)、k=exitk は在庫を減らす注文の置き場所。
        # 在庫を減らす側は mid から**遠ざける**ので、売りなら高く、買いなら安く。
        ks = [0]
        if exitk:
            ks.append(exitk)
        if improve:
            ks.append(-improve)                # 負の k = スプレッドの内側へ
        for k in ks:
            tick = np.where(0.5 * (pb[i] + pa[i]) >= 1000.0, 0.1, 0.01)
            pk = px[i] - sgn * k * tick        # 買いは下へ、売りは上へ
            # k<0 は価格改善(内側)。スプレッドが 2|k| ティック以上あるときだけ。
            # 反対気配を跨がない post-only を保つ条件でもある。
            if k < 0:
                spt = (pa[i] - pb[i]) / tick
                okk = spt >= 2 * (-k)
                pk = np.where(okk, pk, px[i])  # 足りなければ最良気配のまま
            if k == 0:
                q0 = np.full(i.size, SZ_LOT) if front == "queue" else qsz[i]
            elif k < 0:
                # 内側の値段には誰も並んでいない(スプレッドの中は空)。
                # ただし待ち行列 0 だと fill_times が値段を無視して埋まるので
                # 1 ロットを下限にする。改善できなかった行は元の行列を使う。
                q0 = np.where(okk, SZ_LOT, qsz[i])
                if front == "price":
                    # 値段だけ改善し、行列は元のまま(先頭に立つ利得を消す)
                    q0 = qsz[i]
            else:
                # その値段の待ち行列は再構成した板の第 k 階層の数量
                g = np.clip((ts[i] - d0) // GRID_NS, 0, lad["gi"][-1])
                lg = np.clip(np.searchsorted(lad["gi"], g, side="right") - 1,
                             0, lad["gi"].size - 1)
                Q = lad["qb"] if sgn > 0 else lad["qa"]
                q0 = np.nan_to_num(Q[lg, min(k, Q.shape[1] - 1)]) * SZ_LOT
                # ★ 待ち行列 0 だと fill_times が「値段を問わず最初の約定」を
                #   返してしまう(need=0 が常に満たされるため)。深い階層では
                #   16.7% が 0 になり、そこだけ約定率が 35.6% と跳ねた。
                #   自分が先頭でも「自分の値段で 1 ロット約定する」ことは要る。
                q0 = np.maximum(q0, SZ_LOT)
            tau = fill_times(t0, np.round(pk / PX_UNIT), q0, sgn, grid,
                             ft, fpx, fsz, msk, d0)
            good = (tau >= 0) & (tau <= exp)
            P[sgn][f"tau{k}"] = np.where(good, tau, np.iinfo(np.int64).max)
            # 維持する場合は「次に気配が動くまで」で打ち切らない。値段は
            # 動かないので fill_times の物理はそのまま使える(自分の値段以下の
            # 約定だけが待ち行列を進める)。
            P[sgn][f"raw{k}"] = np.where(tau >= 0, tau, np.iinfo(np.int64).max)
            P[sgn][f"px{k}"] = pk
    for sgn in (1, -1):
        P[sgn]["tau"] = P[sgn]["tau0"]
        P[sgn]["px"] = P[sgn]["px0"]

    # 発注機会を時刻順に並べる
    et = np.concatenate([P[1]["t"], P[-1]["t"]])
    es = np.concatenate([np.ones(P[1]["t"].size, np.int8),
                         -np.ones(P[-1]["t"].size, np.int8)])
    ek = np.concatenate([np.arange(P[1]["t"].size), np.arange(P[-1]["t"].size)])
    o = np.argsort(et, kind="stable")
    et, es, ek = et[o], es[o], ek[o]

    INF = np.iinfo(np.int64).max
    live = {1: None, -1: None}           # (tau, price)
    q = 0
    lots_t, lots_p, lots_m = [], [], []
    realized = 0.0
    pair_dt, pair_pnl = [], []
    # 建玉 1 本ずつの記録(項目 3 の特徴量突き合わせ用)
    lot_rec = {"t_in": [], "side": [], "p_in": [], "m_in": [],
               "t_off": [], "pnl": [], "forced": []}
    # 発注 1 件ごとの記録(項目 5 の入口ゲート学習用)。
    # 約定したか、その約定が建てた玉の最終的な往復損益はいくらか。
    po_t, po_s, po_f, po_p = [], [], [], []
    lot_row = []                      # 建玉ごとに、それを建てた発注の行番号
    n_fill = {1: 0, -1: 0}
    n_post = {1: 0, -1: 0}
    n_keep = {1: 0, -1: 0}       # 気配が動いたが取り消さなかった回数
    qs = []

    def do_fill(s, tau, price, row=-1):
        nonlocal q, realized
        j = np.clip(np.searchsorted(ts, tau, side="right") - 1, 0, nb - 1)
        m = float(mid[j])
        n_fill[s] += 1
        if q != 0 and (1 if q > 0 else -1) != s:
            t0 = lots_t.pop(0); p0 = lots_p.pop(0); m0 = lots_m.pop(0)
            r0 = lot_row.pop(0)
            # 売り約定ならロング(建値 p0)を price で閉じる。買い約定はその逆
            gross = (price - p0) if s < 0 else (p0 - price)
            pnl = gross / m0 * 1e4 - 2 * MAKER_FEE
            realized += pnl
            if r0 >= 0:
                po_p[r0] = pnl
            pair_dt.append((tau - t0) / 1e9)
            pair_pnl.append(pnl)
            lot_rec["t_in"].append(t0); lot_rec["side"].append(-s)
            lot_rec["p_in"].append(p0); lot_rec["m_in"].append(m0)
            lot_rec["t_off"].append((tau - t0) / 1e9)
            lot_rec["pnl"].append(pnl); lot_rec["forced"].append(0)
            q -= 1 if q > 0 else -1
        else:
            lots_t.append(tau); lots_p.append(price); lots_m.append(m)
            lot_row.append(row)
            q += s
        qs.append(q)

    n_to = 0
    to_pnl = to_cost = 0.0

    def timeout_flat(T):
        """保有が上限を超えた建玉をテイカーで強制的に閉じる。"""
        nonlocal q, n_to, to_pnl, to_cost
        while lots_t and (T - lots_t[0]) > tmax_ns:
            t0 = lots_t.pop(0); p0 = lots_p.pop(0); m0 = lots_m.pop(0)
            r0 = lot_row.pop(0)
            j = np.clip(np.searchsorted(ts, T, side="right") - 1, 0, nb - 1)
            sgn = 1 if q > 0 else -1
            xp = pb[j] if sgn > 0 else pa[j]
            g = (xp - p0) * sgn / m0 * 1e4 - MAKER_FEE - TAKER_FEE
            to_pnl += g
            if r0 >= 0:
                po_p[r0] = g
            to_cost += 0.5 * (pa[j] - pb[j]) / m0 * 1e4 + TAKER_FEE
            lot_rec["t_in"].append(t0); lot_rec["side"].append(sgn)
            lot_rec["p_in"].append(p0); lot_rec["m_in"].append(m0)
            lot_rec["t_off"].append((T - t0) / 1e9)
            lot_rec["pnl"].append(g); lot_rec["forced"].append(2)
            q -= sgn
            n_to += 1
            qs.append(q)

    for k in range(et.size):
        T = int(et[k])
        if tmax_ns:
            timeout_flat(T)
        # この時点より前に起きる約定を先に処理する
        while True:
            cand = [(live[s][0], s) for s in (1, -1)
                    if live[s] is not None and live[s][0] <= T]
            if not cand:
                break
            tau, s = min(cand)
            po_f[live[s][2]] = 1
            do_fill(s, tau, live[s][1], live[s][2])
            live[s] = None
            if requote != "always":
                # 在庫が動いた。上限を破る側の注文は取り消す。
                for s2 in (1, -1):
                    if live[s2] is not None and abs(q + s2) > qmax:
                        live[s2] = None
        s = int(es[k])
        idx = int(ek[k])
        cur = live[s]
        if cur is not None and requote != "always":
            # 自分側の最良が mid 側へ動いたか(= 自分の指値が最良より不利に
            # なったか)。買いなら最良買気配が上がった場合。
            gap = (float(P[s]["px0"][idx]) - cur[1]) * s
            tick = 0.1 if cur[1] >= 1000.0 else 0.01
            toward = gap > 1e-12
            # 最良から離れすぎた注文は維持しない(η の代わりの 1 つの閾値)
            near = (not keepdt) or (gap <= keepdt * tick + 1e-12)
            fresh = ((not keepmax_ns) or (T - cur[3] <= keepmax_ns)) and near
            if fresh and (requote == "keep" or toward):
                n_keep[s] += 1
                continue                  # 維持する(取り消さない)
        live[s] = None                    # 気配が動いたので出し直し(取り消し)
        allow = abs(q + s) <= qmax        # 在庫が増える側は上限で止める
        if allow and gate is not None and abs(q + s) > abs(q):
            g = gate[s]               # 在庫を増やす発注だけ門を通す
            u = np.searchsorted(g["t"], et[k])
            allow = bool(g["ok"][u]) if (u < g["t"].size
                                         and g["t"][u] == et[k]) else False
        if allow:
            n_post[s] += 1
            row = len(po_t)
            po_t.append(int(et[k])); po_s.append(s); po_f.append(0)
            po_p.append(np.nan)
            # 在庫を減らす発注なら exitk ティック外へ、
            # 在庫を増やす発注は improve ティック内へ(価格改善)
            if exitk and abs(q + s) < abs(q):
                kk = exitk
            elif improve and abs(q + s) > abs(q):
                kk = -improve
            else:
                kk = 0
            if requote == "always":
                tau = int(P[s][f"tau{kk}"][idx])
                live[s] = ((tau, float(P[s][f"px{kk}"][idx]), row)
                           if tau != INF else None)
            else:
                # ★ 約定しない注文も live に載せる。載せないと「60 秒以内に
                #   約定する注文だけを持ち続ける」ことになり、**将来を知って
                #   いないと判定できない規則**になる(自己精査 D16)。
                #   置いた注文はその側を占有し、次の判定まで動かない。
                tau = int(P[s][f"raw{kk}"][idx])
                if keepmax_ns and tau != INF and tau - int(et[k]) > keepmax_ns:
                    tau = INF
                live[s] = (tau, float(P[s][f"px{kk}"][idx]), row, int(et[k]))
    for s in (1, -1):                     # 日の終わりまでに残った約定
        if live[s] is not None and live[s][0] < INF:  # INF = 約定しない注文
            po_f[live[s][2]] = 1
            do_fill(s, live[s][0], live[s][1], live[s][2])

    forced_n = len(lots_t)
    forced_pnl = forced_cost = 0.0
    if forced_n:
        sgn = 1 if q > 0 else -1
        xp = pb[-1] if sgn > 0 else pa[-1]
        for t0, p0, m0, r0 in zip(lots_t, lots_p, lots_m, lot_row):
            g = (xp - p0) * sgn / m0 * 1e4 - MAKER_FEE - TAKER_FEE
            forced_pnl += g
            if r0 >= 0:
                po_p[r0] = g
            forced_cost += 0.5 * (pa[-1] - pb[-1]) / m0 * 1e4 + TAKER_FEE
            lot_rec["t_in"].append(t0); lot_rec["side"].append(sgn)
            lot_rec["p_in"].append(p0); lot_rec["m_in"].append(m0)
            lot_rec["t_off"].append((ts[-1] - t0) / 1e9)
            lot_rec["pnl"].append(g); lot_rec["forced"].append(1)
    pd_ = np.array(pair_dt)
    pp_ = np.array(pair_pnl)
    qa_ = np.array(qs) if qs else np.zeros(1)
    out = {"dt": dt, "n_keep": n_keep[1] + n_keep[-1],
           "n_post_bid": n_post[1], "n_post_ask": n_post[-1],
           "n_fill_bid": n_fill[1], "n_fill_ask": n_fill[-1],
           "n_fill": n_fill[1] + n_fill[-1], "n_pair": int(pd_.size),
           "realized_bp": realized, "forced_n": forced_n,
           "forced_pnl_bp": forced_pnl, "forced_cost_bp": forced_cost,
           "n_timeout": n_to, "timeout_pnl_bp": to_pnl,
           "timeout_cost_bp": to_cost,
           "total_bp": realized + forced_pnl + to_pnl,
           "q_abs_mean": float(np.abs(qa_).mean()),
           "q_abs_max": int(np.abs(qa_).max()),
           "p_absq_ge2": float((np.abs(qa_) >= 2).mean()),
           "t_off_med": float(np.median(pd_)) if pd_.size else np.nan,
           "pair_pnl_mean": float(pp_.mean()) if pp_.size else np.nan}
    for h in HS:
        out[f"p_off_{h:g}s"] = float((pd_ <= h).mean()) if pd_.size else np.nan
    posts_rec = {"t": np.array(po_t, np.int64),
                 "side": np.array(po_s, np.int8),
                 "filled": np.array(po_f, np.int8),
                 "rt_pnl": np.array(po_p, np.float32)}
    return out, pd_, pp_, qa_, lot_rec, posts_rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--qmax", type=int, default=1, help="在庫の上限 |q|")
    ap.add_argument("--lat", type=float, default=0.0, help="発注遅延(秒)")
    ap.add_argument("--mode", choices=["follow", "fixed"], default="follow")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="mode=fixed のときの値段固定の上限(秒)")
    ap.add_argument("--tmax", type=float, default=0.0,
                    help="保有の上限(秒)。超えたらテイカーで強制決済。0=無制限")
    ap.add_argument("--front", choices=["", "price", "queue"], default="",
                    help="価格改善の効果の分解。price=値段だけ改善(行列は元のまま)、"
                         "queue=最良気配のまま行列の先頭に立つ")
    ap.add_argument("--improve", type=int, default=0,
                    help="在庫を増やす注文を最良より何ティック内側に置くか"
                         "(スプレッドが 2 倍以上あるときだけ改善する)")
    ap.add_argument("--exitk", type=int, default=0,
                    help="在庫を減らす注文を最良から何ティック外へ置くか")
    ap.add_argument("--requote", choices=["always", "cond", "keep"],
                    default="always",
                    help="自分側の気配が動いたときの扱い。always=毎回出し直す"
                         "(従来)、cond=mid 側へ動いたときだけ維持、"
                         "keep=向きを問わず維持(幽霊注文の仮定が効くので上界)")
    ap.add_argument("--keepmax", type=float, default=60.0,
                    help="維持の上限(秒)。0 で無制限")
    ap.add_argument("--keepdt", type=int, default=0,
                    help="最良から何ティックまで離れても維持するか。0 で無制限")
    ap.add_argument("--posts", action="store_true",
                    help="発注 1 件ごとの記録を書く(項目 5 の学習用)")
    ap.add_argument("--gate", default="",
                    help="在庫を増やす発注に掛ける門(data/... の parquet)")
    ap.add_argument("--skip", type=int, default=0,
                    help="先頭 N 日を飛ばす(評価期間だけ回すとき 59)")
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    sfx = f"_q{a.qmax}"
    if a.lat:
        sfx += f"_lat{int(round(a.lat*1000))}"
    if a.mode == "fixed":
        sfx += f"_fix{a.hold:g}"
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    if not files:
        # 候補テーブルが無い銘柄は bbo から日付を取る(このシミュレータは
        # bbo と fills だけで動くので、54 列の表は本来要らない)
        days = (pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                .select(pl.col("dt").unique()).collect()["dt"].sort().to_list())
        files = [Path(f"dt={d}.parquet") for d in days]
        print(f"候補テーブルが無いので bbo から {len(files)} 日を取得", flush=True)
    if a.skip:
        files = files[a.skip:]
        sfx += "_te"
    if a.days:
        files = files[:a.days]
        sfx += f"_d{a.days}"      # 試験実行が全期間の出力を上書きしないように
    bpath, fpath = DATA / f"bbo_{tag}.parquet", DATA / f"fills_{tag}.parquet"
    lat_ns = int(round(a.lat * 1e9))
    hold_ns = int(round(a.hold * 1e9)) if a.mode == "fixed" else 0
    tmax_ns = int(round(a.tmax * 1e9))
    keepmax_ns = int(round(a.keepmax * 1e9))
    if a.requote != "always":
        sfx += f"_rq{a.requote}"
        if a.keepmax:
            sfx += f"km{a.keepmax:g}"
        if a.keepdt:
            sfx += f"kd{a.keepdt}"
    if a.tmax:
        sfx += f"_tmax{a.tmax:g}"
    print(f"{len(files)} 日 / qmax {a.qmax} / 遅延 {1000*a.lat:.0f} ms / "
          f"{a.mode}" + (f" (固定 {a.hold:g}s)" if hold_ns else ""), flush=True)

    if a.exitk:
        sfx += f"_k{a.exitk}"
    if a.improve:
        sfx += f"_imp{a.improve}"
    if a.front:
        sfx += f"_{a.front}"
    if a.gate:
        GA = pl.read_parquet(a.gate)
        gday = {k[0] if isinstance(k, tuple) else k: v
                for k, v in GA.partition_by("dt", as_dict=True).items()}
        # ★ 門ごとに出力名を分ける。以前は常に "_gated" だったので、
        #   別の門で回すと前の結果を黙って上書きしていた。
        stem = pathlib.Path(a.gate).stem
        tail = stem.split(f"entrygate_{tag}", 1)[-1].lstrip("_")
        # ★ 門の識別子は**必ず全部**付ける。重複するトークンを省く実装にしたら、
        #   別の門で回した結果が同じ名前になり、2 度も静かに上書きした。
        #   冗長でも一意であることを優先する。
        sfx += "_gatedby_" + tail
    rows, alld, allp, lots, posts = [], [], [], [], []
    carry = pl.DataFrame()
    hq = np.zeros(2 * QCAP + 1, np.int64)
    for k, f in enumerate(files):
        dt = f.stem.split("=")[1]
        gate = None
        if a.gate:
            G = gday.get(dt)
            gate = {}
            for sg in (1, -1):
                g = G.filter(pl.col("side") == sg).sort("t")
                gate[sg] = {"t": g["t"].to_numpy(),
                            "ok": g["pass"].to_numpy().astype(bool)}
        lad = None
        if a.exitk:
            keep = {"step": 1, "gi": [], "qb": [], "qa": [], "ok": []}
            bb = clean_bbo(pl.scan_parquet(bpath)
                           .filter(pl.col("dt") == dt).collect())[0].sort("ts")
            # 板の再構成は L1 のイベント列から作る(候補テーブルではない)
            _, _, _, _, _, carry, _ = ladder_day(
                DATA / f"l1_{tag}" / f"dt={dt}.parquet", bb, carry, keep=keep)
            gi = np.concatenate(keep["gi"])
            o2 = np.argsort(gi)
            lad = {"gi": gi[o2],
                   "qb": np.concatenate(keep["qb"])[o2],
                   "qa": np.concatenate(keep["qa"])[o2]}
            del keep, bb
        r, pd_, pp_, qa_, lr, pr = simulate(dt, bpath, fpath, lat_ns, a.qmax,
                                            hold_ns, tmax_ns, gate, a.exitk,
                                            lad, a.improve, a.front,
                                            a.requote, keepmax_ns, a.keepdt)
        if a.posts:
            posts.append(pl.DataFrame({"dt": [dt] * pr["t"].size, **pr}))
        rows.append(r)
        lots.append(pl.DataFrame({
            "dt": [dt] * len(lr["t_in"]),
            "t_in": np.array(lr["t_in"], np.int64),
            "side": np.array(lr["side"], np.int8),
            "p_in": np.array(lr["p_in"], np.float64),
            "m_in": np.array(lr["m_in"], np.float64),
            "t_off": np.array(lr["t_off"], np.float32),
            "pnl": np.array(lr["pnl"], np.float32),
            "forced": np.array(lr["forced"], np.int8)}))
        alld.append(pd_.astype(np.float32))
        allp.append(pp_.astype(np.float32))
        hq += np.bincount(np.clip(qa_.astype(int) + QCAP, 0, 2 * QCAP),
                          minlength=2 * QCAP + 1)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(files)}] {dt} 約定 {r['n_fill']:,} "
                  f"組 {r['n_pair']:,} 60s 相殺 {100*r['p_off_60s']:.1f}% "
                  f"残 {r['forced_n']}", flush=True)
    D = pl.DataFrame(rows)
    D.write_csv(DATA / f"inv_days_{tag}{sfx}.csv")
    td = np.concatenate(alld)
    tp = np.concatenate(allp)
    pl.concat(lots).write_parquet(DATA / f"inv_lots_{tag}{sfx}.parquet")
    if a.posts:
        pl.concat(posts).write_parquet(DATA / f"inv_posts_{tag}{sfx}.parquet")
    pl.DataFrame({"q": np.arange(-QCAP, QCAP + 1), "n": hq}).write_csv(
        DATA / f"inv_qhist_{tag}{sfx}.csv")

    tot_f = int(D["n_fill"].sum())
    tot_p = int(D["n_pair"].sum())
    print(f"\n発注 {int(D['n_post_bid'].sum()+D['n_post_ask'].sum()):,} / "
          f"約定 {tot_f:,} (買 {int(D['n_fill_bid'].sum()):,} / "
          f"売 {int(D['n_fill_ask'].sum()):,})")
    print(f"メイカー同士で相殺できた組 {tot_p:,} "
          f"= 約定の {100*2*tot_p/max(tot_f,1):.1f}%")
    print("相殺までの時間(完成した組の中で)")
    for h in HS:
        print(f"  {h:>5g}s 以内 {100*float((td <= h).mean()):5.1f}%")
    if td.size:
        print(f"  中央 {np.median(td):.3f}s / 平均 {td.mean():.3f}s")
    print(f"\n組の損益 平均 {tp.mean():+.4f} bp / 中央 {np.median(tp):+.4f} bp "
          f"/ 正 {100*float((tp > 0).mean()):.1f}% / 合計 {tp.sum():+,.0f} bp")
    nt = int(D["n_timeout"].sum())
    if nt:
        print(f"時間切れの強制決済 {nt:,} 件 = 約定の "
              f"{100*nt/max(tot_f,1):.2f}%  損益 "
              f"{float(D['timeout_pnl_bp'].sum()):+,.0f} bp "
              f"(1 件 {float(D['timeout_pnl_bp'].sum())/nt:+.3f} bp / "
              f"費用のみ {float(D['timeout_cost_bp'].sum())/nt:.3f} bp)")
    fn = int(D["forced_n"].sum())
    print(f"強制決済 {fn:,} 件 = 約定の {100*fn/max(tot_f,1):.2f}%  "
          f"損益 {float(D['forced_pnl_bp'].sum()):+,.0f} bp "
          f"(1 件 {float(D['forced_pnl_bp'].sum())/max(fn,1):+.3f} bp / "
          f"費用のみ {float(D['forced_cost_bp'].sum())/max(fn,1):.3f} bp)")
    tb = float(D["total_bp"].sum())
    print(f"★ 合計 {tb:+,.0f} bp / 1 約定 {tb/max(tot_f,1):+.4f} bp / "
          f"1 日 {tb/len(files):+,.1f} bp")
    print(f"在庫 |q| 平均 {float(D['q_abs_mean'].mean()):.3f} "
          f"最大 {int(D['q_abs_max'].max())} "
          f"P(|q|>=2) {100*float(D['p_absq_ge2'].mean()):.2f}%")
    print(f"\n書き出し {DATA}/inv_days_{tag}{sfx}.csv ほか")


if __name__ == "__main__":
    main()
