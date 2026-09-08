r"""Lighter maker 戦略の約定再生シミュレータ(第 2 段)。

    uv run python scripts/lighter_maker_sim.py --symbols MU
    uv run python scripts/lighter_maker_sim.py --arm placebo_flip --symbols MU

主構成は config/lighter_maker_frozen_v1.json(結果を見る前に凍結済み)。
本ファイルは**その構成をそのまま実行する**。探索的な感応度は --arm / --grid で
別枠として回し、主判定には混ぜない。

=============================================================================
★時間契約(凍結 JSON の time_contract と同じもの。実装上の要点)
=============================================================================
すべての時刻は **recv_ns(このマシンの受信時計)** で組む。取引所の
last_updated_at は「その秒が終わったか」の判定と監査にしか使わない。

  t_decide   秒 s の特徴量が使えると**観測できた**時刻
             = ts_us > (s+1)*1e6 を持つ最初の ticker 行の recv_ns
  t_live     = t_decide + eff_maker
  t_cxreq    = t_live + lifetime
  t_cxeff    = t_cxreq + eff_cancel
  t_fill     行列を約定が食い切った時刻(t_live < t_fill <= t_cxeff)
  t_xreq     = t_fill + exit_s
  t_xexec    = t_xreq + eff_taker

  eff_X = compute + net_rtt + institutional_X

★eff に往復(片道でなく)を入れる理由: 受信時計の上では、受信 r の約定は
取引所時刻 r − oneway に起きている。自分の注文が取引所時刻
t + oneway + inst に載るなら、それに当たり得る約定の受信時刻は
r >= t + 2*oneway + inst = t + rtt + inst である。片道で済ませると
自分の注文が実際より早く板に載ることになる。

=============================================================================
★待ち行列モデル(L2 では真の順番が観測できないので 3 通り出す)
=============================================================================
  Q1 strict-trade      前方数量は約定だけが削る(主モデル・保守的)
  Q2 proportional      説明できない減少を 前方/表示 の比だけ前方から削る
  Q3 front-cancel      説明できない減少を可能な限り前方から削る(楽観・上界)

板の監査(lighter_maker_parse.py --audit)で、最良気配の数量減少のうち
**約定で説明できるのは 0.3〜10% しかない**ことが判っている。つまり行列は
ほぼ取消で消える。Q1 と Q3 の差はこの市場では極端に大きくなる。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lighter_features import build  # noqa: E402

WF = Path("E:/Memory-lighter/wf")
MK = Path("E:/Memory-lighter/mk")
CFG = Path(__file__).resolve().parents[1] / "config/lighter_maker_frozen_v1.json"
BP = 1e4
HORS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0)
NS = 1_000_000_000
INF = np.iinfo(np.int64).max

ACC = {
    "Standard": dict(mk_fee=0.0, tk_fee=0.0, mk_lat=200.0, cx_lat=300.0, tk_lat=300.0),
    "Standard_mk0": dict(mk_fee=0.0, tk_fee=0.0, mk_lat=0.0, cx_lat=300.0, tk_lat=300.0),
    "Premium": dict(mk_fee=0.40, tk_fee=2.80, mk_lat=0.0, cx_lat=0.0, tk_lat=140.0),
    "Premium_stake500k": dict(mk_fee=0.28, tk_fee=1.96, mk_lat=0.0, cx_lat=0.0, tk_lat=140.0),
    "Plus": dict(mk_fee=0.5, tk_fee=0.5, mk_lat=200.0, cx_lat=200.0, tk_lat=300.0),
}


def load_cfg() -> dict:
    return json.loads(CFG.read_text(encoding="utf-8"))


def load_sym(sym: str):
    """grid → 信号、bbo → 気配列、trades → 約定列。"""
    g = pl.read_parquet(WF / f"grid_{sym}.parquet")
    F = build(g)
    ts_s = F["ts_s"].to_numpy().astype(np.int64)
    bad = F["_bad"].to_numpy()
    sig = np.where(bad, np.nan, F["gap_12_asym"].to_numpy().astype(np.float64))
    mid = np.where(bad, np.nan, F["_mid"].to_numpy().astype(np.float64))
    spr = np.where(bad, np.nan, F["spr_bp"].to_numpy().astype(np.float64))
    W = {w: (g[f"W{w}_b"].to_numpy().astype(np.float64),
             g[f"W{w}_a"].to_numpy().astype(np.float64)) for w in (5, 10, 25, 50, 100)}
    del F, g
    B = pl.read_parquet(WF / f"bbo_{sym}.parquet").sort("recv_ns")
    bo = dict(recv=B["recv_ns"].to_numpy().astype(np.int64),
              ts=B["ts_us"].to_numpy().astype(np.int64),
              bid=B["bid"].to_numpy().astype(np.float64),
              ask=B["ask"].to_numpy().astype(np.float64),
              bsz=B["bid_sz"].to_numpy().astype(np.float64),
              asz=B["ask_sz"].to_numpy().astype(np.float64))
    del B
    T = pl.read_parquet(MK / f"trades_{sym}.parquet").sort("recv_ns")
    tr = dict(recv=T["recv_ns"].to_numpy().astype(np.int64),
              px=T["px"].to_numpy().astype(np.float64),
              sz=T["sz"].to_numpy().astype(np.float64),
              mka=T["is_maker_ask"].to_numpy())
    del T
    return ts_s, sig, mid, spr, bad, W, bo, tr


def decide_recv(ts_s: np.ndarray, bo: dict) -> np.ndarray:
    """秒 s が完全に終わったと**観測できた**受信時刻。

    (s+1) 秒以降の取引所刻印を持つ最初の ticker 行の受信時刻。これより前に
    秒 s の特徴量を確定させることはできない。
    """
    j = np.searchsorted(bo["ts"], (ts_s + 1) * 1_000_000, side="left")
    out = np.full(ts_s.size, INF, np.int64)
    ok = j < bo["recv"].size
    out[ok] = bo["recv"][j[ok]]
    # ticker の取引所刻印は単調でない場合があるので、受信時刻で累積最大を取る
    return np.maximum.accumulate(out)


def calib(day: np.ndarray, sig: np.ndarray, ret60: np.ndarray, q: float, nwin: int):
    """テスト日ごとに、直前 nwin 完全日だけで 閾値と符号 を引く。"""
    days = np.unique(day)
    thr = {}
    sgn = {}
    for d in days:
        m = (day >= d - nwin) & (day < d) & np.isfinite(sig)
        if m.sum() < 1000:
            continue
        a = np.abs(sig[m])
        thr[int(d)] = float(np.quantile(a, q))
        m2 = m & np.isfinite(ret60)
        if m2.sum() < 1000:
            continue
        x, y = sig[m2], ret60[m2]
        c = float(np.corrcoef(x, y)[0, 1])
        sgn[int(d)] = 1.0 if c >= 0 else -1.0
    return thr, sgn


def bbo_at(bo: dict, t: int):
    """受信時刻 t 時点で観測済みの最新 BBO(後ろ向き)。"""
    i = np.searchsorted(bo["recv"], t, side="right") - 1
    if i < 0:
        return None
    return i


def vwap(side: int, i: int, bo: dict, W: dict, gi: int, q: float):
    """テイカーで数量 q を食べたときの平均約定価格。

    side=+1 は買い(ask を食べる)、-1 は売り(bid を食べる)。
    最良の表示数量で足りればその値段。足りなければ mid からの bp 帯の
    累積深さ(W5/W10/W25/W50/W100)を線形補間して残りの平均価格を作る。
    """
    px = bo["ask"][i] if side > 0 else bo["bid"][i]
    sz = bo["asz"][i] if side > 0 else bo["bsz"][i]
    if q <= sz or gi < 0:
        return px, 0.0
    m = 0.5 * (bo["bid"][i] + bo["ask"][i])
    ws = [5, 10, 25, 50, 100]
    cum = [W[w][0 if side < 0 else 1][gi] for w in ws]
    prev_w, prev_c = 0.0, sz
    bpx = 100.0
    for w, c in zip(ws, cum):
        if not np.isfinite(c):
            continue
        if c >= q:
            bpx = prev_w + (w - prev_w) * (q - prev_c) / max(c - prev_c, 1e-12)
            break
        prev_w, prev_c = float(w), float(c)
    # 平均約定価格 ≒ 最良と最深の中点(帯内は一様と仮定)
    edge = m * (1 + side * bpx / BP)
    return 0.5 * (px + edge), bpx


def fill_scan(side: int, p: float, q: float, t0: int, t1: int, Q0: float,
              bo: dict, tr: dict, eps: float):
    """t0 から t1 までの受信イベントを走らせ、Q1/Q2/Q3 の約定を同時に出す。

    戻り: dict(モデル名 -> (約定数量, 約定受信時刻 or INF))
    """
    lo = np.searchsorted(tr["recv"], t0, side="right")
    hi = np.searchsorted(tr["recv"], t1, side="right")
    # 自分に当たり得る攻撃的約定だけを残す
    if side > 0:                       # 買い: 攻撃的売り(is_maker_ask=False)
        m = (~tr["mka"][lo:hi]) & (tr["px"][lo:hi] <= p + eps)
    else:                              # 売り: 攻撃的買い(is_maker_ask=True)
        m = (tr["mka"][lo:hi]) & (tr["px"][lo:hi] >= p - eps)
    trec = tr["recv"][lo:hi][m]
    tsz = tr["sz"][lo:hi][m]
    # 同価格の表示数量(自分の指値が最良である間だけ観測できる)
    blo = np.searchsorted(bo["recv"], t0, side="right")
    bhi = np.searchsorted(bo["recv"], t1, side="right")
    brec = bo["recv"][blo:bhi]
    bpx = bo["bid"][blo:bhi] if side > 0 else bo["ask"][blo:bhi]
    bsz = bo["bsz"][blo:bhi] if side > 0 else bo["asz"][blo:bhi]
    # ★自分の指値が最良である行だけ残す。残さないと、指値が最良でない構成
    #   (内側/外側 1 ティック)で「全部 NaN の行」を Python ループが舐めて
    #   3 倍遅くなる(実測 16s -> 44s)。
    okv = np.abs(bpx - p) < eps
    brec = brec[okv]
    vis = bsz[okv]

    out = {}
    for mdl in ("Q1", "Q2", "Q3"):
        Q = Q0
        D = Q0                          # 直前に観測した表示数量
        got = 0.0
        tf = INF
        ib = 0
        for k in range(trec.size + 1):
            tnow = trec[k] if k < trec.size else t1
            # このイベントまでに観測した表示数量の減少を処理する
            if mdl != "Q1":
                while ib < brec.size and brec[ib] <= tnow:
                    v = vis[ib]
                    ib += 1
                    if not np.isfinite(v):
                        continue
                    dec = D - v
                    if dec > 0:
                        # ★同一 50ms バッチでは約定を先に前方へ割り当てる。
                        #   ここでは約定処理の前に来る減少だけを取消として扱う。
                        # ★比率は**減少前**の表示数量で取る。減少後で割ると
                        #   r が過大になり Q2 が Q3 に潰れる(実際に潰れた)。
                        if mdl == "Q3":
                            Q = max(0.0, Q - dec)
                        else:
                            r = min(Q / D, 1.0) if D > 0 else 1.0
                            Q = max(0.0, Q - dec * r)
                    D = v               # 追加は自分の後ろ(前方は変わらない)
            if k == trec.size:
                break
            s = tsz[k]
            if Q > 0:
                use = min(Q, s)
                Q -= use
                s -= use
                if mdl != "Q1":
                    D = max(0.0, D - use)
            if s > 0 and got < q:
                add = min(s, q - got)
                if got == 0.0:
                    tf = int(trec[k])
                got += add
        out[mdl] = (got, tf)
    return out


def blank(sym, day, t0, side, p, q, stale=0, reject=0):
    """提出できなかった quote も同じ列を持たせる(後段の schema を揃える)。"""
    r = dict(sym=sym, day=day, t0=t0, tl=0, tcx=0, teff=0, side=side, px=p, q=q,
             Q0=float("nan"), stale=stale, reject=reject, alone=0, behind=0)
    for m in ("Q1", "Q2", "Q3"):
        r[f"fq_{m}"] = 0.0
        r[f"tf_{m}"] = INF
        r[f"cxpend_{m}"] = 0
    return r


def run_symbol(sym: str, cfg: dict, acc: str, arm: str, size_usd: float,
               lifetime: float, exit_s: float, qmult: float, dpos: int,
               rtt_ms: float, mk_lat_over=None, pos: int = 0,
               sizepct: float = 0.0, noe2: bool = False) -> pl.DataFrame:
    ts_s, sig, mid, spr, bad, W, bo, tr = load_sym(sym)
    meta = json.loads((MK / "market_meta.json").read_text(encoding="utf-8"))
    mm = meta["markets"][sym]
    tick = 10.0 ** (-int(mm["price_decimals"]))
    sdec = int(mm["size_decimals"])
    minq = float(mm["min_base_amount"])
    eps = tick * 0.4

    a = ACC[acc]
    comp = float(cfg["clock"]["compute_ms"])
    mk_lat = a["mk_lat"] if mk_lat_over is None else float(mk_lat_over)
    eff_mk = int((comp + rtt_ms + mk_lat) * 1e6)
    eff_cx = int((comp + rtt_ms + a["cx_lat"]) * 1e6)
    eff_tk = int((comp + rtt_ms + a["tk_lat"]) * 1e6)

    day = (ts_s // 86400).astype(np.int64)
    j60 = np.searchsorted(ts_s, ts_s + 60, side="right") - 1
    ok60 = (ts_s[j60] >= ts_s + 58) & np.isfinite(mid[j60]) & np.isfinite(mid)
    ret60 = np.where(ok60, (mid[j60] - mid) / mid * BP, np.nan)
    thr, sgn = calib(day, sig, ret60, cfg["signal"].get("q", 0.99),
                     int(cfg["signal"]["calibration_window_days"]))
    tdec = decide_recv(ts_s, bo)

    # 発火判定(すべて当日より前の情報だけ)
    fire = np.zeros(ts_s.size, bool)
    sd = np.zeros(ts_s.size, np.int8)
    for d in np.unique(day):
        di = int(d)
        if di not in thr or di not in sgn:
            continue
        m = (day == di) & np.isfinite(sig) & (~bad)
        v = sgn[di] * sig
        hit = m & (np.abs(sig) > thr[di])
        fire |= hit
        sd = np.where(hit, np.where(v > 0, 1, -1).astype(np.int8), sd)
    if arm == "placebo_flip":
        sd = (-sd).astype(np.int8)
    if arm == "placebo_shift":
        # 銘柄×日の内側で発火時刻を循環シフト(帰無)
        rng = np.random.default_rng(20260908)
        for d in np.unique(day):
            m = day == d
            k = int(rng.integers(1000, 80000))
            fire[m] = np.roll(fire[m], k)
            sd[m] = np.roll(sd[m], k)
    if arm in ("cadence", "puremm_bid", "puremm_ask"):
        fire = (ts_s % dpos == 0) & (~bad) & np.isfinite(mid)
        if arm == "cadence":
            rng = np.random.default_rng(7)
            sd = np.where(rng.random(ts_s.size) < 0.5, 1, -1).astype(np.int8)
        else:
            # ★E3 Pure MM の片脚。両側を同時に出す構成は、買い脚と売り脚を
            #   別々に回して足し合わせる(在庫の相殺は別途上界として評価する)
            sd = np.full(ts_s.size, 1 if arm == "puremm_bid" else -1, np.int8)

    idx = np.flatnonzero(fire & np.isfinite(mid) & (tdec < INF))
    rows = []
    for i in idx:
        t0 = int(tdec[i])
        side = int(sd[i])
        if side == 0:
            continue
        i0 = bbo_at(bo, t0)
        if i0 is None:
            continue
        p = bo["bid"][i0] if side > 0 else bo["ask"][i0]
        if not np.isfinite(p) or p <= 0:
            continue
        # 指値の置き場所: 0=BBO / -1=スプレッドの内側 1 ティック / +1=外側 1 ティック
        if pos:
            # 買いの内側は高い買い気配、売りの内側は安い売り気配
            p = round((p + side * (-pos) * tick) / tick) * tick
        if sizepct > 0:
            # 到着時 BBO 表示数量の一定割合(容量の感応度)
            base = bo["bsz"][i0] if side > 0 else bo["asz"][i0]
            q = max(minq, round(sizepct * float(base), sdec))
        else:
            q = max(minq, round(size_usd / p, sdec))
        tl = t0 + eff_mk
        il = bbo_at(bo, tl)
        if il is None:
            continue
        # データ切断中(直近更新が 60 秒より古い)なら出さない
        if tl - bo["recv"][il] > 60 * NS:
            rows.append(blank(sym, int(day[i]), t0, side, p, q, stale=1))
            continue
        bidl, askl = bo["bid"][il], bo["ask"][il]
        # post-only reject: 到着時に marketable なら約定扱いにしない
        if (side > 0 and p >= askl - eps) or (side < 0 and p <= bidl + eps):
            rows.append(blank(sym, int(day[i]), t0, side, p, q, reject=1))
            continue
        touch = bidl if side > 0 else askl
        alone = int((side > 0 and p > touch + eps) or (side < 0 and p < touch - eps))
        behind = int((side > 0 and p < touch - eps) or (side < 0 and p > touch + eps))
        if alone:
            Q0 = 0.0
        elif behind:
            Q0 = (bo["bsz"][i0] if side > 0 else bo["asz"][i0])
        else:
            Q0 = (bo["bsz"][il] if side > 0 else bo["asz"][il])
        Q0 *= qmult
        tcx = tl + int(lifetime * NS)
        teff = tcx + eff_cx
        f = fill_scan(side, p, q, tl, teff, Q0, bo, tr, eps)
        r = dict(sym=sym, day=int(day[i]), t0=t0, tl=tl, tcx=tcx, teff=teff,
                 side=side, px=p, q=q, Q0=Q0, stale=0, reject=0,
                 alone=alone, behind=behind)
        for mdl, (got, tf) in f.items():
            r[f"fq_{mdl}"] = got
            r[f"tf_{mdl}"] = tf
            r[f"cxpend_{mdl}"] = int(tf != INF and tf > tcx)
        rows.append(r)
    if not rows:
        return pl.DataFrame()
    L = pl.DataFrame(rows)
    return finish(L, sym, bo, tr, W, ts_s, a, eff_mk, eff_cx, eff_tk,
                  exit_s, eps, tmo=() if noe2 else (1.0, 5.0, 30.0, 300.0))


def finish(L, sym, bo, tr, W, ts_s, a, eff_mk, eff_cx, eff_tk, exit_s, eps,
           tmo=(1.0, 5.0, 30.0, 300.0)):
    # ★探索格子では E2 を回さない(tmo=())。1 約定あたり 12 回の約定再走査が
    #   要り、格子 21 構成では 5 時間かかる。格子の主指標は E1 の純 EV である。
    """約定した quote に markout と往復損益(E1 / E2)を付ける。

    ★1 銘柄 1 ポジションの制約は **待ち行列モデルごとに** 事後で適用する。
      Q1 の約定履歴で全モデルを塞ぐと、約定しやすい Q2/Q3 が実際より多く
      建てられることになり、PnL/日 を過大評価する。
    """
    n = L.height
    out = {}
    side = L["side"].to_numpy()
    px = L["px"].to_numpy()
    t0a = L["t0"].to_numpy()
    okq = (L["reject"].to_numpy() == 0) & (L["stale"].to_numpy() == 0)
    for mdl in ("Q1", "Q2", "Q3"):
        fq0 = L[f"fq_{mdl}"].to_numpy().copy()
        tf0 = L[f"tf_{mdl}"].to_numpy().copy()
        # --- 1 ポジション制約(モデルごと)---
        sub = np.zeros(n, bool)
        free = -1
        for r in range(n):
            if not okq[r] or t0a[r] <= free:
                continue
            sub[r] = True
            if fq0[r] > 0 and tf0[r] < INF:
                free = tf0[r] + int(exit_s * NS) + eff_tk
        fq = np.where(sub, fq0, 0.0)
        tf = np.where(sub, tf0, INF)
        fil = (fq > 0) & (tf < INF)
        out[f"sub_{mdl}"] = sub
        out[f"fil_{mdl}"] = fil
        ii = np.clip(np.searchsorted(bo["recv"], np.where(fil, tf, 0), "right") - 1,
                     0, bo["recv"].size - 1)
        mf = np.where(fil, 0.5 * (bo["bid"][ii] + bo["ask"][ii]), np.nan)
        out[f"mid_f_{mdl}"] = mf
        # 入口の価格改善 = 約定時点の mid から見た自分の値段の有利さ
        out[f"imp_{mdl}"] = np.where(fil, BP * side * (mf - px) / px, np.nan)
        for h in HORS:
            th = np.where(fil, tf + int(h * NS), 0)
            jj = np.clip(np.searchsorted(bo["recv"], th, "right") - 1,
                         0, bo["recv"].size - 1)
            mh = 0.5 * (bo["bid"][jj] + bo["ask"][jj])
            okh = fil & (th <= bo["recv"][-1])
            # M_h = 入口の価格改善 + 約定後の価格変化(分解して両方持つ)
            out[f"mk{h:g}_{mdl}"] = np.where(okh, BP * side * (mh - px) / px, np.nan)
            out[f"dr{h:g}_{mdl}"] = np.where(okh, BP * side * (mh - mf) / px, np.nan)
        # ---- E1: maker entry -> exit_s 秒後に taker ----
        txe = np.where(fil, tf + int(exit_s * NS), 0) + eff_tk
        kk = np.clip(np.searchsorted(bo["recv"], txe, "right") - 1,
                     0, bo["recv"].size - 1)
        gi = np.clip(np.searchsorted(ts_s, txe // NS, "right") - 1, 0, ts_s.size - 1)
        pxx = np.full(n, np.nan)
        slip = np.full(n, np.nan)
        okx = fil & (txe <= bo["recv"][-1])
        for r in np.flatnonzero(okx):
            v, s_ = vwap(-int(side[r]), int(kk[r]), bo, W, int(gi[r]), float(fq[r]))
            pxx[r] = v
            slip[r] = s_
        notion = np.where(fil, px * fq, np.nan)
        fee1 = (a["mk_fee"] / BP) * notion + (a["tk_fee"] / BP) * np.abs(pxx * fq)
        out[f"px_x_{mdl}"] = pxx
        out[f"slip_x_{mdl}"] = slip
        out[f"notional_{mdl}"] = notion
        out[f"fee_{mdl}"] = fee1
        out[f"pnl_{mdl}"] = np.where(okx, side * (pxx - px) * fq - fee1, np.nan)
        out[f"held_{mdl}"] = np.where(okx, (txe - tf) / NS, np.nan)
        out[f"xhour_{mdl}"] = np.where(
            fil, (txe // (3600 * NS)) != (tf // (3600 * NS)), False)
        # ---- E2: maker entry -> 反対側 BBO へ maker exit -> timeout で taker ----
        for T in (tmo or ()):
            pe = np.full(n, np.nan)
            hp = np.zeros(n, np.int8)      # 1 = maker で出られた
            pn = np.full(n, np.nan)
            for r in np.flatnonzero(fil):
                tfi = int(tf[r])
                i1 = bbo_at(bo, tfi)
                if i1 is None:
                    continue
                s2 = -int(side[r])
                pe_px = bo["ask"][i1] if s2 < 0 else bo["bid"][i1]
                if not np.isfinite(pe_px) or pe_px <= 0:
                    continue
                lv = tfi + eff_mk
                iv = bbo_at(bo, lv)
                if iv is None:
                    continue
                bl, al = bo["bid"][iv], bo["ask"][iv]
                mk_ok = not ((s2 > 0 and pe_px >= al - eps)
                             or (s2 < 0 and pe_px <= bl + eps))
                got, tfe = 0.0, INF
                if mk_ok:
                    tch = bl if s2 > 0 else al
                    if abs(pe_px - tch) < eps:
                        Q0e = bo["bsz"][iv] if s2 > 0 else bo["asz"][iv]
                    elif (s2 > 0 and pe_px > tch) or (s2 < 0 and pe_px < tch):
                        Q0e = 0.0
                    else:
                        Q0e = bo["bsz"][i1] if s2 > 0 else bo["asz"][i1]
                    fe = fill_scan(s2, pe_px, float(fq[r]), lv,
                                   lv + int(T * NS) + eff_cx, Q0e, bo, tr, eps)
                    got, tfe = fe[mdl]
                if got >= fq[r] - 1e-12 and tfe < INF:
                    pe[r] = pe_px
                    hp[r] = 1
                    fee2 = (a["mk_fee"] / BP) * (px[r] + pe_px) * fq[r]
                    pn[r] = side[r] * (pe_px - px[r]) * fq[r] - fee2
                else:
                    tq = lv + int(T * NS) + eff_tk
                    if tq > bo["recv"][-1]:
                        continue
                    iq = bbo_at(bo, tq)
                    gq = int(np.clip(np.searchsorted(ts_s, tq // NS, "right") - 1,
                                     0, ts_s.size - 1))
                    rem = float(fq[r] - got)
                    v, _ = vwap(s2, int(iq), bo, W, gq, max(rem, 1e-9))
                    # 部分的に maker で出られた分と、残りを taker で清算した分
                    avg = (got * pe_px + rem * v) / fq[r] if fq[r] > 0 else v
                    pe[r] = avg
                    fee2 = ((a["mk_fee"] / BP) * (px[r] * fq[r] + pe_px * got)
                            + (a["tk_fee"] / BP) * abs(v * rem))
                    pn[r] = side[r] * (avg - px[r]) * fq[r] - fee2
            out[f"e2px{T:g}_{mdl}"] = pe
            out[f"e2mk{T:g}_{mdl}"] = hp
            out[f"e2pnl{T:g}_{mdl}"] = pn
    return L.with_columns([pl.Series(k, v) for k, v in out.items()])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--acc", default="Standard", choices=list(ACC))
    ap.add_argument("--arm", default="main",
                    choices=["main", "placebo_flip", "placebo_shift", "cadence",
                             "puremm_bid", "puremm_ask"])
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--life", type=float, default=1.0)
    ap.add_argument("--exit", type=float, default=300.0)
    ap.add_argument("--qmult", type=float, default=1.0)
    ap.add_argument("--cadence", type=int, default=60)
    ap.add_argument("--rtt", type=float, default=-1.0)
    ap.add_argument("--mklat", type=float, default=-1.0)
    ap.add_argument("--noe2", action="store_true",
                    help="E2 Hybrid を計算しない(探索格子の高速化)")
    ap.add_argument("--sizepct", type=float, default=0.0,
                    help=">0 なら 到着時 BBO 表示数量のこの割合を数量にする")
    ap.add_argument("--pos", type=int, default=0,
                    help="0=BBO / -1=内側1ティック / +1=外側1ティック")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_cfg()
    syms = a.symbols.split(",") if a.symbols else cfg["symbols"]
    rtt = cfg["clock"]["net_rtt_ms"] if a.rtt < 0 else a.rtt
    mkl = None if a.mklat < 0 else a.mklat
    MK.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.arm}_{a.acc}"
    for sym in syms:
        t0 = time.time()
        L = run_symbol(sym, cfg, a.acc, a.arm, a.size, a.life, a.exit,
                       a.qmult, a.cadence, rtt, mkl, a.pos, a.sizepct, a.noe2)
        if not L.height:
            print(f"{sym}: quote なし", flush=True)
            continue
        L.write_parquet(MK / f"mk_{tag}_{sym}.parquet")
        sb = L["sub_Q1"].to_numpy()
        fl = L["fil_Q1"].to_numpy()
        pn = L["pnl_Q1"].to_numpy()
        no = L["notional_Q1"].to_numpy()
        net = float(np.nansum(pn))
        nn = float(np.nansum(no))
        nb = BP * net / nn if nn > 0 else float("nan")
        print(f"{sym}: 発火 {L.height:,} / 提出 {int(sb.sum()):,} / "
              f"reject {int(L['reject'].sum()):,} / Q1 約定 {int(fl.sum()):,} "
              f"({100*fl.sum()/max(sb.sum(),1):.2f}%) / "
              f"純 ${net:+.4f} = {nb:+.3f}bp ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
