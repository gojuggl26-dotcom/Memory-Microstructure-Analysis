"""反実仮想: 特徴量が Q1/Q5 に入った瞬間に BBO へメイカー注文を置いたら約定するか。

    uv run python scripts/build_maker.py --coin xyz:MU

置き方
------
時刻 T で特徴量が
  Q5 (最も買い側) → **最良買い気配に買い指値**を置く
  Q1 (最も売り側) → **最良売り気配に売り指値**を置く
つまり信号の向きに受動的に構える。数量は無限小 (自分の存在が板を変えない)。

約定の判定 (保守的な待ち行列モデル)
-----------------------------------
買い指値を値段 p_b(T) に置いたとき、自分の前にいる数量は **Q_ahead = bid_sz(T)**
である (T の時点で最後尾に並ぶ)。以後、値段 p_b(T) **以下**で約定したテイカー
売りの数量を積み上げ、それが Q_ahead を超えた時点で約定とみなす。

  - 最初の 1 秒は約定 1 件ずつ実価格で判定する (h=1s はこれだけで決まる)
  - 2 秒目以降は「その秒の最良買い気配が p_b(T) 以下か」で秒単位に判定する。
    テイカー売りは必ずその時の最良買い気配以下で約定するので、
    気配が p_b(T) 以下なら約定値段も p_b(T) 以下である

**この規則は約定率を過小に見積もる側に倒れている。** 理由は 3 つ:
  1. 自分より前の注文が**取り消される**ことを数えていない
     (実際には取り消しで順番が繰り上がる)
  2. 最良気配が自分より内側にある間に、大きなテイカーが板を食い下がって
     自分の値段に届く場合を 2 秒目以降は数えていない。実測ではテイカー売りの
     87.7% はその時の最良買い気配ちょうどで約定し 96.5% は 5 ティック以内
     なので、この取りこぼしは小さい
  3. 板が薄くなって自分が最良になっても、待ち行列は減らさない

そこで**上限側の規則**も同時に出す。「自分の値段に 1 枚でもテイカーが来たら
約定」= 待ち行列の先頭にいた場合である。真の約定率はこの 2 つの間にある。

対照
----
信号のない **Q3(中央の五分位)** でも同じ発注をして比べる。
「信号を見て置いたら損」なのか「そもそも受動的に置いたら損」なのかは
別の話であり、対照が無いと区別できない (CLAUDE.md checklist B5/B6)。
さらに **買い指値と売り指値の両方**を全五分位で出す。

時間契約 (CLAUDE.md の厳禁事項)
------------------------------
- 発注の判断に使う特徴量は build_pred.day_features のもので、すべて T で確定する
- 約定判定に使うテイカー約定は **(T + 5ms, T+h]** のものだけ。5ms の余裕は
  node_fills の時刻がミリ秒精度で板イベントより最大 ~1ms 早いことへの備え
- 五分位も十分位も境目は**前日の分布**から作る。当日の実現値を知らないと
  決まらない境目は各時点で計算できず、実装不能な規則になる
- 損益は「約定値段」と「T+h の mid」だけで決まり、未来の情報は入らない

費用
----
メイカー手数料 **0.088 bp/約定** を控除した値も出す (この案件で先に確定した値)。
mid で評価した損益は建てた側の評価益であって、そこから降りる費用は別に掛かる。
反対売買を板を叩いて行う場合 (片道スプレッドの半分) の値も出せるよう、
T 時点のスプレッドをセルごとに記録する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import PX_UNIT, clean_bbo  # noqa: E402
from build_pred import (FILL_GUARD_NS, GRID_N, day_features, quintile,  # noqa: E402
                        usable)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HORIZONS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ("obi", "ofi_10s", "ai_net_10s")
QS = (0, 2, 4)          # Q1 / Q3(対照) / Q5
SIDES = (1, -1)         # +1 = 最良買い気配に買い指値 / -1 = 最良売り気配に売り指値
RATE_W = 10                 # 待ち行列がはける速さを測る窓 (秒)
MAKER_FEE_BP = 0.088        # メイカー手数料 (bp/約定)
NDEC = 10


def pair_volume(ft, fp, fs, isb, tg, thr_b, thr_a, t_lo_ns, t_hi_ns):
    """各格子点について (T+t_lo, T+t_hi] の約定を 1 件ずつ見て数量を積む。

    買い指値側は「約定値段 <= その格子点の最良買い気配」、
    売り指値側は「約定値段 >= その格子点の最良売り気配」だけを数える。
    窓が短いので (T, fill) の組を実体化しても数万件で済む。
    """
    n = tg.size
    lo = np.searchsorted(ft, tg + t_lo_ns, side="right")
    hi = np.searchsorted(ft, tg + t_hi_ns, side="right")
    cnt = np.maximum(hi - lo, 0)
    tot = int(cnt.sum())
    vb = np.zeros(n)
    va = np.zeros(n)
    if tot == 0:
        return vb, va
    g = np.repeat(np.arange(n), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    fi = np.repeat(lo, cnt) + off
    mb = (~isb[fi]) & (fp[fi] <= thr_b[g])
    ma = isb[fi] & (fp[fi] >= thr_a[g])
    np.add.at(vb, g[mb], fs[fi][mb])
    np.add.at(va, g[ma], fs[fi][ma])
    return vb, va


def day_fills(D, fd):
    """1 日ぶんの「約定したか」の判定材料と、待ち行列がはける推定時間を返す。

    build_maker.py と build_heat.py で判定がずれないよう、ここ 1 か所に置く。
    戻り値 FILL[h] = (買い指値を食った数量, 売り指値を食った数量)、
    TAU[side] = 待ち行列がはけるまでの推定秒数。
    """
    tg = D["tg"]
    bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
    bt = np.round(bid / PX_UNIT)
    at = np.round(ask / PX_UNIT)
    d0 = int(tg[0])
    idx = np.arange(GRID_N + 1)

    ft = fd["ts"].cast(pl.Int64).to_numpy()
    fp = np.round(fd["px"].to_numpy() / PX_UNIT)
    fs = fd["sz"].to_numpy()
    isb = fd["side"].to_numpy() == "B"        # テイカー買い = 売り板を食う
    sec = np.clip((ft - d0) // 10 ** 9, 0, GRID_N).astype(np.int64)
    vsell = np.zeros(GRID_N + 1)
    vbuy = np.zeros(GRID_N + 1)
    np.add.at(vsell, sec[~isb], fs[~isb])
    np.add.at(vbuy, sec[isb], fs[isb])
    csell = np.concatenate([[0.0], np.cumsum(vsell)])
    cbuy = np.concatenate([[0.0], np.cumsum(vbuy)])
    lo10 = np.maximum(idx - RATE_W, 0)
    rate_s = (csell[idx] - csell[lo10]) / RATE_W
    rate_b = (cbuy[idx] - cbuy[lo10]) / RATE_W

    FILL = {}
    for h in HORIZONS:
        if h < 1.0:
            FILL[h] = pair_volume(ft, fp, fs, isb, tg, bt, at,
                                  FILL_GUARD_NS, int(h * 1e9))
    Vb, Va = pair_volume(ft, fp, fs, isb, tg, bt, at, FILL_GUARD_NS, 10 ** 9)
    FILL[1.0] = (Vb.copy(), Va.copy())
    done = 1
    for h in [x for x in HORIZONS if x > 1.0]:
        for d in range(done, int(h)):
            j = np.minimum(idx + d, GRID_N)
            Vb += np.where(bt[j] <= bt, vsell[j], 0.0)
            Va += np.where(at[j] >= at, vbuy[j], 0.0)
        done = int(h)
        FILL[h] = (Vb.copy(), Va.copy())

    TAU = {1: np.where(rate_s > 0, qb / np.maximum(rate_s, 1e-12), np.inf),
           -1: np.where(rate_b > 0, qa / np.maximum(rate_b, 1e-12), np.inf)}
    return FILL, TAU


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 行を除外)", flush=True)
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "px", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    days = sorted(bb["dt"].unique().to_list())

    rows, edrows = [], []
    prev_q: dict[str, np.ndarray] = {}
    prev_d: dict[int, np.ndarray] = {}
    for k, dt in enumerate(days):
        D = day_features(bb.filter(pl.col("dt") == dt).sort("ts"), FID.get(dt))
        if D is None:
            continue
        F, R, RC, tg = D["F"], D["R"], D["RC"], D["tg"]
        mmg = D["mmg"]      # T 時点の microprice - mid (bp)
        bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
        mid = 0.5 * (bid + ask)
        spr = (ask - bid) / mid * 1e4
        bt = np.round(bid / PX_UNIT)
        at = np.round(ask / PX_UNIT)
        d0 = int(tg[0])
        idx = np.arange(GRID_N + 1)

        FILL, TAU = day_fills(D, FID[dt])
        SZ = {1: qb, -1: qa}
        PX = {1: bid, -1: ask}

        pe_ok = {f: usable(prev_q.get(f)) for f in FEATS}
        de_ok = {sd: (prev_d.get(sd) is not None
                      and bool(np.all(np.diff(prev_d[sd]) > 0))) for sd in SIDES}
        for f in FEATS:
            if not pe_ok[f]:
                continue
            qq = quintile(F[f], prev_q[f])
            fin = np.isfinite(F[f])
            for sd in SIDES:
                if not de_ok[sd]:
                    continue
                dec = np.searchsorted(prev_d[sd], TAU[sd], side="right")
                for q in QS:
                    base = fin & (qq == q) & np.isfinite(mid) & (SZ[sd] > 0)
                    for h in HORIZONS:
                        vb, va = FILL[h]
                        v = vb if sd == 1 else va
                        got = v >= SZ[sd]          # 待ち行列を抜いた
                        touch = v > 0              # 自分の値段に来た (上限側)
                        m_end = mid * np.exp(R[h] / 1e4)
                        sel = base & np.isfinite(m_end)
                        pnl = (m_end - PX[sd]) * sd / mid * 1e4
                        ret = R[h] * sd
                        # microprice で測ったリターン。十分位の点数には板の厚み
                        # (= OBI の情報) が入っているので、mid だけ見ると
                        # 「mid が既知の microprice に寄る」分を信号と誤認しうる
                        retc = RC[h] * sd
                        # T+h 時点の microprice - mid。mid 建ての損益にこれを
                        # 足すと microprice 建てになる。板が傾いた状態で
                        # mid 評価すると損益が偏るのでの確認用
                        mm_end = mmg + RC[h] - R[h]
                        for dd in range(NDEC):
                            s_ = sel & (dec == dd)
                            if not s_.any():
                                continue
                            g = s_ & got
                            tc = s_ & touch
                            rows.append({
                                "dt": dt, "feat": f, "q": q, "side": sd,
                                "h": h, "dec": dd,
                                "n": int(s_.sum()), "n_fill": int(g.sum()),
                                "n_touch": int(tc.sum()),
                                "sum_pnl": float(pnl[g].sum()),
                                "sum_pnl_touch": float(pnl[tc].sum()),
                                "sum_ret": float(ret[s_].sum()),
                                "sum_ret_fill": float(ret[g].sum()),
                                "sum_retc": float(np.nansum(retc[s_])),
                                "sum_retc_fill": float(np.nansum(retc[g])),
                                "sum_mmend_fill": float(np.nansum(mm_end[g] * sd)),
                                "sum_spr": float(np.nansum(spr[s_])),
                                "sum_tau": float(np.minimum(TAU[sd][s_], 1e6).sum()),
                                "sum_q": float(SZ[sd][s_].sum())})
        prev_q = {f: np.nanquantile(F[f][np.isfinite(F[f])], [0.2, 0.4, 0.6, 0.8])
                  for f in FEATS if np.isfinite(F[f]).any()}
        prev_d = {}
        for sd in SIDES:
            v = TAU[sd][np.isfinite(TAU[sd])]
            if v.size:
                e = np.quantile(v, np.arange(1, NDEC) / NDEC)
                prev_d[sd] = e
                edrows.append({"dt": dt, "side": sd,
                               **{f"d{i+1}": float(x) for i, x in enumerate(e)}})
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True)

    pl.DataFrame(rows).write_parquet(DATA / f"maker_cells_{tag}.parquet")
    pl.DataFrame(edrows).write_csv(DATA / f"maker_edges_{tag}.csv")
    print(f"書き出し maker_cells({len(rows):,} セル) / maker_edges({len(edrows)})")


if __name__ == "__main__":
    main()


def day_fill_times(D, fd, max_s: int = 60):
    """各発注について **実際の約定時刻 tau** を返す (ns、約定しなければ -1)。

    判定規則は day_fills と同じ。違いは「h までに約定したか」ではなく
    「いつ約定したか」を出すこと。逆選択を測るには発注時刻 T ではなく
    約定時刻 tau からのリターンが要る (T から tau までの動きは、
    まだ何も持っていない区間なので損益に入らない)。

    手順は 2 段。まず秒単位で「待ち行列を抜けた秒」を探し、
    次にその秒の中の約定を 1 件ずつたどって正確な時刻を取る。
    """
    tg = D["tg"]
    bid, ask, qb, qa = D["bid"], D["ask"], D["bsz"], D["asz"]
    bt = np.round(bid / PX_UNIT)
    at = np.round(ask / PX_UNIT)
    d0 = int(tg[0])
    n = tg.size
    idx = np.arange(n)

    ft = fd["ts"].cast(pl.Int64).to_numpy()
    fp = np.round(fd["px"].to_numpy() / PX_UNIT)
    fs = fd["sz"].to_numpy()
    isb = fd["side"].to_numpy() == "B"
    sec = np.clip((ft - d0) // 10 ** 9, 0, GRID_N).astype(np.int64)
    vsell = np.zeros(n)
    vbuy = np.zeros(n)
    np.add.at(vsell, sec[~isb], fs[~isb])
    np.add.at(vbuy, sec[isb], fs[isb])

    e_b, e_a = pair_volume(ft, fp, fs, isb, tg, bt, at, FILL_GUARD_NS, 10 ** 9)
    out = {}
    for sd, Q, e0, vv, thr, cmp_ok, msk in (
            (1, qb, e_b, vsell, bt, np.less_equal, ~isb),
            (-1, qa, e_a, vbuy, at, np.greater_equal, isb)):
        good = np.isfinite(Q) & (Q > 0)
        C = np.where(good, e0, 0.0)
        dstar = np.full(n, -1, dtype=np.int64)
        cprev = np.zeros(n)
        # ★既知の小さな取りこぼし: ここ (A 段) には許容誤差を入れていないので、
        # 約定数量が Q とちょうど一致する場合を桁落ちで落とすことがある。
        # 実測の影響は約定率で +0.02pp、tau の 3.2% (2026-06-24 の買い側)。
        # 既に公表した数値の再現性を保つため、この関数は据え置く。
        # 許容誤差を入れた正しい実装は build_quotes.fill_times にある。
        hit = good & (C >= Q)
        dstar[hit] = 0
        for d in range(1, max_s):
            j = np.minimum(idx + d, GRID_N)
            add = np.where(cmp_ok(thr[j], thr), vv[j], 0.0)
            newly = good & (dstar < 0) & ((C + add) >= Q)
            cprev[newly] = C[newly]
            dstar[newly] = d
            C = C + add
        tau = np.full(n, -1, dtype=np.int64)
        k = np.flatnonzero(dstar >= 0)
        if k.size:
            st = np.where(dstar[k] == 0, tg[k] + FILL_GUARD_NS,
                          d0 + (idx[k] + dstar[k]) * 10 ** 9)
            en = np.where(dstar[k] == 0, tg[k] + 10 ** 9,
                          d0 + (idx[k] + dstar[k] + 1) * 10 ** 9)
            lo = np.searchsorted(ft, st, side="right")
            hi = np.searchsorted(ft, en, side="right")
            cnt = np.maximum(hi - lo, 0)
            tot = int(cnt.sum())
            if tot:
                g = np.repeat(np.arange(k.size), cnt)
                off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                fi = np.repeat(lo, cnt) + off
                # dstar==0 の秒だけは値段も見る (exact0 と同じ規則)
                px_ok = np.where(np.repeat(dstar[k] == 0, cnt),
                                 cmp_ok(fp[fi], thr[k][g]), True)
                w = np.where(msk[fi] & px_ok, fs[fi], 0.0)
                cs = np.cumsum(w)
                cs -= np.repeat(cs[np.cumsum(cnt) - cnt] - w[np.cumsum(cnt) - cnt],
                                cnt)
                need = np.repeat(Q[k] - cprev[k], cnt)
                # 区分累積和は「全体の cumsum から群の直前を引く」形なので
                # 桁落ちが出る。1.193 が 1.1929999... になって比較に落ちると、
                # **待ち行列が小さい = 最も速く約定する発注**だけが取りこぼされ、
                # 逆選択を過大に見積もる方向へ偏る。数量の最小単位は 0.001 なので
                # 1e-9 の許容は安全側。
                okc = cs >= need - 1e-9
                first = np.full(k.size, -1, dtype=np.int64)
                ordi = np.flatnonzero(okc)
                if ordi.size:
                    gg = g[ordi]
                    # 各発注について最初に閾値を超えた約定
                    sel = np.ones(ordi.size, dtype=bool)
                    sel[1:] = gg[1:] != gg[:-1]
                    first[gg[sel]] = fi[ordi[sel]]
                got = first >= 0
                tau[k[got]] = ft[first[got]]
                # 秒の中で越えが見つからない場合はその秒の終わりに丸める
                tau[k[~got]] = en[~got]
        out[sd] = tau
    return out
