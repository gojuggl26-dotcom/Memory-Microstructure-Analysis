"""仮想発注候補テーブル — 1 行 = 1 つの発注候補。

    uv run python scripts/build_quotes.py --coin xyz:MU [--days N]

設計
----
候補時点は **BBO が更新された瞬間**(= event-driven)。同じ状態を時計時間で
大量に重複させないためで、指示どおり最終バックテストがそのまま event-driven に
なる。1 つの時点につき **買い候補と売り候補の 2 行**を作る。

    1 row = 1 potential quote  (timestamp, side)

★最重要規則
-----------
    X_t は時刻 t までに観測可能な値だけ

- 板・OBI・depth は t 以前の最後の状態(後ろ向き asof)
- OFI・攻撃的流量・ボラは t で閉じた後ろ向き窓
- 待ち行列は t 時点の自分側の最良気配の数量
- **label_ で始まる列だけが未来を使う**。説明変数に混ぜないこと

多水準の板
----------
build_obi_levels.day_features を keep つきで呼び、100 ms 格子の水準別数量
Qb/Qa (最良から 10 ティック) を取り出して候補時点へ後ろ向きに貼る。
100 ms 以内の古さが乗るが、ブロックが約 65 ms 周期なのでほぼ最新である。

Hazard (fill probability) について
----------------------------------
「予測約定確率」はモデルの出力なので、この表そのものから当てはめる必要がある
(この表が無いと学習できない)。ここでは**その入力**である

    tau_hat = 自分側の待ち行列 ÷ 直近 10 秒の同方向テイカー流量

と、実現した約定時刻 label_fill_lat_s を持たせる。確率の当てはめは Step 2。

保存先
------
本体は E:(容量のため)、リポジトリ内には間引き版を置く。
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, GRID_NS, NLV, PX_UNIT,  # noqa: E402
                              RESTING, SZ_LOT, TERMINAL, clean_bbo,
                              day_features as ladder_day)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
DAY_NS = 86_400_000_000_000
NG = 86_400
GUARD_NS = 5_000_000            # fills の時刻ずれへの安全余裕
MAXW = 60                       # 約定を探す最大秒数
PRE_NS = 2_000_000              # 約定直前の板を見る位置
FEE_BP = 0.088
OFI_WIN = [0.1, 0.25, 0.5, 1.0, 10.0]
AGG_WIN = [0.1, 1.0, 10.0]
MK_H = [0.1, 1.0, 10.0]         # markout のホライズン
SUB = 20                        # リポジトリ内の間引き版の間隔


def rsum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size) + 1
    return c[i] - c[np.maximum(i - w, 0)]


def fill_times(tc, thr, Q0, sgn, bt_grid, ft, fp, fs, msk, d0):
    """候補時点ごとの約定時刻 tau (ns)。届かなければ -1。

    判定は既報と同じ「自分の値段以下(売りなら以上)で約定したテイカーの数量が
    自分の前の待ち行列を超えたら約定」。3 段で解く。

      0 段: t_c + 5ms 〜 その秒の終わり  … 約定 1 件ずつ実価格で
      A 段: 次の秒から 59 秒先まで        … 秒ごとに気配で門を開閉
      B 段: 超えた秒の中を 1 件ずつ       … 正確な時刻を取る
    """
    n = tc.size
    cmp_ = np.less_equal if sgn > 0 else np.greater_equal
    sec_c = np.clip((tc - d0) // 10 ** 9, 0, NG - 1)
    eos = d0 + (sec_c + 1) * 10 ** 9

    def pair_vol(lo_t, hi_t, thr_v, want_time=False, need=None, base=None):
        lo = np.searchsorted(ft, lo_t, side="right")
        hi = np.searchsorted(ft, hi_t, side="right")
        cnt = np.maximum(hi - lo, 0)
        tot = int(cnt.sum())
        v = np.zeros(lo.size)
        tt = np.full(lo.size, -1, dtype=np.int64)
        if tot == 0:
            return (v, tt) if want_time else v
        g = np.repeat(np.arange(lo.size), cnt)
        off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        fi = np.repeat(lo, cnt) + off
        ok = msk[fi] & cmp_(fp[fi], thr_v[g])
        w = np.where(ok, fs[fi], 0.0)
        np.add.at(v, g, w)
        if not want_time:
            return v
        # cnt=0 の群では開始添字が末尾を越えるので丸める
        # (repeat が 0 回なのでその値は使われない)
        st = np.minimum(np.cumsum(cnt) - cnt, tot - 1)
        cs = np.cumsum(w)
        cs -= np.repeat(cs[st] - w[st], cnt)
        hit = cs >= (need[g] - base[g] - 1e-9)
        oi = np.flatnonzero(hit)
        if oi.size:
            gg = g[oi]
            sel = np.ones(oi.size, dtype=bool)
            sel[1:] = gg[1:] != gg[:-1]
            tt[gg[sel]] = ft[fi[oi[sel]]]
        return v, tt

    # 0 段
    zero = np.zeros(n)
    v0, t0 = pair_vol(tc + GUARD_NS, eos, thr, True, Q0, zero)
    tau = np.where(v0 >= Q0 - 1e-9, t0, -1).astype(np.int64)
    C = v0.copy()
    dstar = np.where(tau >= 0, 0, -1).astype(np.int64)
    cprev = np.zeros(n)

    # A 段: 秒ごとの数量 (自分の側)
    vsec = np.zeros(NG + MAXW + 2)
    ssec = np.clip((ft - d0) // 10 ** 9, 0, NG - 1).astype(np.int64)
    np.add.at(vsec, ssec[msk], fs[msk])
    gate_thr = np.concatenate([bt_grid, np.full(MAXW + 2, bt_grid[-1])])
    for d in range(1, MAXW):
        j = sec_c + d
        add = np.where(cmp_(gate_thr[j], thr), vsec[j], 0.0)
        newly = (dstar < 0) & ((C + add) >= Q0 - 1e-9)
        cprev[newly] = C[newly]
        dstar[newly] = d
        C = C + add

    # B 段
    k = np.flatnonzero((dstar > 0) & (tau < 0))
    if k.size:
        st = d0 + (sec_c[k] + dstar[k]) * 10 ** 9
        en = st + 10 ** 9
        _, tt = pair_vol(st, en, thr[k], True, Q0[k], cprev[k])
        got = tt >= 0
        tau[k[got]] = tt[got]
        tau[k[~got]] = en[~got]
    return tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--out", default=str(BULK))
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    outd = Path(a.out) / tag
    outd.mkdir(parents=True, exist_ok=True)
    subd = DATA / f"quotes_sub_{tag}"
    subd.mkdir(parents=True, exist_ok=True)

    files = sorted((DATA / f"l1_{tag}").glob("dt=*.parquet"))
    # ★常駐させない。bbo 3,517 万行と fills を丸ごと持つと、板の再構成が
    #   使う数百 MB と競合してメモリ不足で落ちる。日ごとに読み直す。
    bpath = DATA / f"bbo_{tag}.parquet"
    fpath = DATA / f"fills_{tag}.parquet"
    bdays = set(pl.scan_parquet(bpath).select(pl.col("dt").unique())
                .collect()["dt"].to_list())

    def bbo_of(dt):
        return clean_bbo(pl.scan_parquet(bpath).filter(pl.col("dt") == dt)
                         .collect())[0].sort("ts")

    def fills_of(dt):
        return (pl.scan_parquet(fpath)
                .filter(pl.col("crossed") & (pl.col("dt") == dt))
                .select("ts", "px", "sz", "side").collect().sort("ts"))

    files = [f for f in files if f.stem.split("=")[1] in bdays]
    if a.days:
        files = files[: a.days]
    print(f"対象 {len(files)} 日 -> {outd}", flush=True)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64})
    cdir = outd / "_carry"
    cdir.mkdir(exist_ok=True)
    for kf, fp_ in enumerate(files):
        dt = fp_.stem.split("=")[1]
        # 中断・再開できるように: 済んだ日は飛ばし、繰越状態はファイルから戻す
        if (outd / f"dt={dt}.parquet").exists() and (cdir / f"dt={dt}.parquet").exists():
            carry = pl.read_parquet(cdir / f"dt={dt}.parquet")
            continue
        if kf > 0 and carry.height == 0:
            prev = cdir / f"dt={files[kf-1].stem.split('=')[1]}.parquet"
            if prev.exists():
                carry = pl.read_parquet(prev)
        d = bbo_of(dt)
        keep = {"step": 1, "gi": [], "qb": [], "qa": [], "ok": []}
        _, _, _, _, _, nxt, _ = ladder_day(fp_, d, carry, keep=keep)
        carry = nxt
        gi = np.concatenate(keep["gi"])
        Qb = np.concatenate(keep["qb"])
        Qa = np.concatenate(keep["qa"])
        okl = np.concatenate(keep["ok"])
        del keep
        ordr = np.argsort(gi)
        gi, Qb, Qa, okl = gi[ordr], Qb[ordr], Qa[ordr], okl[ordr]
        Qb = np.where(okl[:, None], Qb, np.nan)
        Qa = np.where(okl[:, None], Qa, np.nan)

        ts = d["ts"].cast(pl.Int64).to_numpy()
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        pb = d["best_bid"].to_numpy()
        pa = d["best_ask"].to_numpy()
        qb1 = d["bid_sz"].to_numpy()
        qa1 = d["ask_sz"].to_numpy()
        mid = 0.5 * (pb + pa)
        n = ts.size
        # 板 (100ms 格子) を候補時点へ後ろ向きに貼る
        lg = np.clip(np.searchsorted(gi, (ts - d0) // GRID_NS, side="right") - 1,
                     0, gi.size - 1)
        QB, QA = Qb[lg], Qa[lg]
        cb = np.nancumsum(QB, axis=1)
        ca = np.nancumsum(QA, axis=1)

        ft_ = fills_of(dt)
        ft = ft_["ts"].cast(pl.Int64).to_numpy()
        fpx = ft_["px"].to_numpy()
        fsz = ft_["sz"].to_numpy()
        fbuy = ft_["side"].to_numpy() == "B"
        cnet = np.concatenate([[0.0], np.cumsum(fsz * np.where(fbuy, 1.0, -1.0))])
        csell = np.concatenate([[0.0], np.cumsum(np.where(~fbuy, fsz, 0.0))])
        cbuy = np.concatenate([[0.0], np.cumsum(np.where(fbuy, fsz, 0.0))])

        e = np.concatenate([[0.0], (
            (pb[1:] >= pb[:-1]) * qb1[1:] - (pb[1:] <= pb[:-1]) * qb1[:-1]
            - ((pa[1:] <= pa[:-1]) * qa1[1:] - (pa[1:] >= pa[:-1]) * qa1[:-1]))])
        cofi = np.cumsum(e)

        # L1 の板イベント (流動性の補充・消滅)
        ev = pl.read_parquet(fp_, columns=["ts", "side", "px", "status",
                                           "orig_sz", "remaining_sz", "tif",
                                           "is_trigger"]).filter(
            (~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
            & pl.col("status").is_in(["open"] + TERMINAL))
        et = ev["ts"].cast(pl.Int64).to_numpy()
        eb = ev["side"].to_numpy() == "B"
        est = ev["status"].to_numpy()
        evol = np.where(est == "open", ev["orig_sz"].to_numpy(),
                        ev["remaining_sz"].to_numpy())
        is_add = est == "open"
        is_can = np.isin(est, CANCELS)
        cadd = np.concatenate([[0.0], np.cumsum(np.where(is_add, evol, 0.0))])
        ccan = np.concatenate([[0.0], np.cumsum(np.where(is_can, evol, 0.0))])
        cadd_b = np.concatenate([[0.0], np.cumsum(np.where(is_add & eb, evol, 0.0))])
        ccan_b = np.concatenate([[0.0], np.cumsum(np.where(is_can & eb, evol, 0.0))])
        cadd_a = np.concatenate([[0.0], np.cumsum(np.where(is_add & ~eb, evol, 0.0))])
        ccan_a = np.concatenate([[0.0], np.cumsum(np.where(is_can & ~eb, evol, 0.0))])

        # 1 秒格子の mid (ボラと、約定判定の門に使う)
        tg = d0 + np.arange(NG, dtype=np.int64) * 10 ** 9
        gj = np.clip(np.searchsorted(ts, tg, side="right") - 1, 0, n - 1)
        midg = mid[gj]
        r1 = np.diff(np.log(midg), prepend=0.0) * 1e4
        rv60 = np.sqrt(rsum(np.nan_to_num(r1 ** 2), 60))
        ofg = cofi[gj]
        ofd = np.diff(ofg, prepend=0.0)
        ofivar = rsum(np.nan_to_num(ofd ** 2), 60) / 60.0
        btg = np.round(pb[gj] / PX_UNIT)
        atg = np.round(pa[gj] / PX_UNIT)
        sec_c = np.clip((ts - d0) // 10 ** 9, 0, NG - 1)

        def back(cum, arr, w_ns):
            j = np.searchsorted(arr, ts - w_ns, side="right")
            return cum[np.searchsorted(arr, ts, side="right")] - cum[j]

        base = {}
        base["ts"] = ts
        base["sec"] = sec_c.astype(np.int32)
        base["bid"] = pb
        base["ask"] = pa
        base["mid"] = mid
        base["spread_abs"] = pa - pb
        tick = np.where(mid >= 1000.0, 0.1, 0.01)
        base["tick"] = tick
        base["spread_bp"] = (pa - pb) / mid * 1e4
        base["spread_tick"] = (pa - pb) / tick
        chg = np.concatenate([[True], (pb[1:] != pb[:-1]) | (pa[1:] != pa[:-1])])
        idx = np.where(chg, np.arange(n), -1)
        base["bbo_age_s"] = (ts - ts[np.maximum.accumulate(idx)]) / 1e9
        j5 = np.searchsorted(ts, ts - 5 * 10 ** 9, side="left")
        base["bbo_turnover"] = (np.arange(n) - j5) / 5.0
        for w in OFI_WIN:
            j = np.clip(np.searchsorted(ts, ts - int(w * 1e9), side="right") - 1,
                        0, n - 1)
            base[f"ofi_{w:g}s"] = cofi - cofi[j]
        for w in AGG_WIN:
            base[f"aggr_{w:g}s"] = back(cnet, ft, int(w * 1e9))
        base["rv_60s_bp"] = rv60[sec_c]
        base["ofi_var_60s"] = ofivar[sec_c]
        base["cancel_10s"] = back(ccan, et, 10 ** 10)
        base["add_10s"] = back(cadd, et, 10 ** 10)
        hh = (sec_c // 3600).astype(np.int32)
        base["hour_utc"] = hh
        base["is_us_session"] = ((sec_c >= 13 * 3600 + 1800)
                                 & (sec_c < 20 * 3600)).astype(np.int8)
        import datetime as _dt
        wd = _dt.date.fromisoformat(dt).weekday()
        base["is_weekend"] = np.full(n, 1 if wd >= 5 else 0, np.int8)

        rows = []
        for sd in (1, -1):
            F = dict(base)
            F["side"] = np.full(n, sd, np.int8)
            F["quote_px"] = pb if sd == 1 else pa
            for K in (1, 2, 4, 10):
                F[f"obi{K}"] = np.where(cb[:, K - 1] + ca[:, K - 1] > 0,
                                        sd * (cb[:, K - 1] - ca[:, K - 1])
                                        / np.maximum(cb[:, K - 1] + ca[:, K - 1], 1e-9),
                                        np.nan)
            F["obi_l2"] = np.where(QB[:, 1] + QA[:, 1] > 0,
                                   sd * (QB[:, 1] - QA[:, 1])
                                   / np.maximum(QB[:, 1] + QA[:, 1], 1e-9), np.nan)
            F["depth_own"] = (QB[:, 0] if sd == 1 else QA[:, 0]) * SZ_LOT
            F["depth_opp"] = (QA[:, 0] if sd == 1 else QB[:, 0]) * SZ_LOT
            F["depth_cum10"] = (cb[:, 9] + ca[:, 9]) * SZ_LOT
            F["queue_ahead_qty"] = qb1 if sd == 1 else qa1
            madd = back(cadd_b if sd == 1 else cadd_a, et, 10 ** 10)
            F["queue_ahead_est_cnt"] = np.where(
                madd > 0, F["queue_ahead_qty"] / np.maximum(madd / 100.0, 1e-9),
                np.nan)
            rate = back(csell if sd == 1 else cbuy, ft, 10 ** 10) / 10.0
            F["taker_rate_10s"] = rate
            F["tau_hat_s"] = np.where(rate > 0, F["queue_ahead_qty"]
                                      / np.maximum(rate, 1e-12), np.inf)
            # 分子は自分側の**全水準**の取消/追加なので、分母も累積の板厚に揃える
            # (最良気配だけで割ると桁が 2 つずれる)
            own_cum = (cb[:, 9] if sd == 1 else ca[:, 9]) * SZ_LOT
            F["depth_own_cum10"] = own_cum
            F["fragility"] = (back(ccan_b if sd == 1 else ccan_a, et, 10 ** 10)
                              / np.maximum(own_cum * 10.0, 1e-9))
            F["resilience"] = (back(cadd_b if sd == 1 else cadd_a, et, 10 ** 10)
                               / np.maximum(own_cum * 10.0, 1e-9))
            # 最良気配だけの取消圧 (待ち行列がどれだけ速く溶けるか)
            F["touch_cancel_rate"] = (
                back(ccan_b if sd == 1 else ccan_a, et, 10 ** 10)
                / np.maximum(F["queue_ahead_qty"] * 10.0, 1e-9))
            for w in OFI_WIN:
                F[f"ofi_{w:g}s"] = sd * base[f"ofi_{w:g}s"]
            for w in AGG_WIN:
                F[f"aggr_{w:g}s"] = sd * base[f"aggr_{w:g}s"]
            thr = np.round((pb if sd == 1 else pa) / PX_UNIT)
            tau = fill_times(ts, thr, F["queue_ahead_qty"], sd,
                             btg if sd == 1 else atg, ft, np.round(fpx / PX_UNIT),
                             fsz, (~fbuy) if sd == 1 else fbuy, d0)
            has = tau >= 0
            F["label_filled_60s"] = has.astype(np.int8)
            F["label_fill_lat_s"] = np.where(has, (tau - ts) / 1e9, np.nan)
            ip = np.clip(np.searchsorted(ts, np.where(has, tau - PRE_NS, ts[0]),
                                         side="right") - 1, 0, n - 1)
            mpre = np.where(has, mid[ip], np.nan)
            F["label_edge_bp"] = (mpre - F["quote_px"]) * sd / mpre * 1e4
            for h in MK_H:
                te = tau + int(h * 1e9)
                ih = np.clip(np.searchsorted(ts, np.where(has, te, ts[0]),
                                             side="right") - 1, 0, n - 1)
                mk = np.where(has & (te <= d0 + DAY_NS),
                              sd * np.log(mid[ih] / mpre) * 1e4, np.nan)
                F[f"label_markout_{h:g}s_bp"] = mk
                F[f"label_pnl_{h:g}s_bp"] = F["label_edge_bp"] - FEE_BP + mk
            rows.append(F)

        cols = {}
        for k_ in rows[0]:
            v = np.concatenate([rows[0][k_], rows[1][k_]])
            cols[k_] = v.astype(np.float32) if v.dtype.kind == "f" else v
        T = pl.DataFrame({"dt": [dt] * (2 * n), **cols})
        T.write_parquet(outd / f"dt={dt}.parquet", compression="zstd")
        carry.write_parquet(cdir / f"dt={dt}.parquet")
        T[::SUB].write_parquet(subd / f"dt={dt}.parquet", compression="zstd")
        nrow, ncol = 2 * n, T.width
        frate = 100 * np.mean(np.concatenate(
            [rows[0]['label_filled_60s'], rows[1]['label_filled_60s']]))
        del d, T, rows, QB, QA, cb, ca
        gc.collect()
        if (kf + 1) % 5 == 0 or kf == 0:
            print(f"  [{kf+1}/{len(files)}] {dt}  候補 {nrow:,} 行 × {ncol} 列 "
                  f"約定率 {frate:.1f}%",
                  flush=True)
    print(f"書き出し {outd} / 間引き版 {subd}")


if __name__ == "__main__":
    main()
