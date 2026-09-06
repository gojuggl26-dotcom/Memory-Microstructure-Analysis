r"""Lighter 1 秒グリッド → 派生特徴量(第2段)。lighter_analyze.py から import する。

=============================================================================
★時間契約(CLAUDE.md の厳禁事項)
=============================================================================
- すべての特徴量は時刻 t(その秒の終わり)までに判る値だけで作る
- ローリング窓は (t−W, t] の閉じた過去。roll_sum は NaN を 0 と数える
  (cumsum の NaN 伝播事故の再発防止 — Boros 報告17 と同じガード)
- 拡大窓 z / 分位は t より前の標本だけで標準化する
- shift(-k) は目的変数専用。ここでは一切使わない

★グリッドに穴がある(60 秒超の沈黙は行ごと無い)。窓は ts で引くので
  穴をまたいでも「実時間 (t−W, t]」の意味は壊れない。

接頭辞 = 系統: px/ret/rv/spr(価格・BBO) obi(不均衡) dep(深さ) mic(microprice)
shp(形状) gap(隙間) flw(指値フロー) cxl(取消) ofi trd(約定) act(活動・品質)
x(横断の掛け合わせ)
"""
from __future__ import annotations

import numpy as np
import polars as pl

BP = 1e4
EPS = 1e-12
WINS = (5, 30, 60, 300)          # 秒
BURN = 600                        # 拡大窓の助走(秒)


def roll_sum(ts: np.ndarray, v: np.ndarray, w: float) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    c = np.concatenate(([0.0], np.cumsum(v)))
    lo = np.searchsorted(ts, ts - w, side="right")
    return c[np.arange(1, len(ts) + 1)] - c[lo]


def roll_n(ts: np.ndarray, w: float) -> np.ndarray:
    lo = np.searchsorted(ts, ts - w, side="right")
    return (np.arange(len(ts)) + 1 - lo).astype(np.float64)


def roll_mean(ts, v, w):
    ok = np.isfinite(v).astype(np.float64)
    return roll_sum(ts, v, w) / np.maximum(roll_sum(ts, ok, w), 1.0)


def ewma(v: np.ndarray, hl: float) -> np.ndarray:
    """イベント(=秒)単位の指数移動平均。NaN は直前値を保つ。"""
    al = 1.0 - 0.5 ** (1.0 / hl)
    x = np.nan_to_num(v, nan=0.0)
    out = np.empty(len(v))
    acc = 0.0
    for i in range(len(v)):
        acc = al * x[i] + (1 - al) * acc
        out[i] = acc
    return out


def expand_z(v: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(v, nan=0.0)
    ok = np.isfinite(v).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cq = np.concatenate(([0.0], np.cumsum(x * x)))
    cn = np.concatenate(([0.0], np.cumsum(ok)))
    m = cn[:-1]
    mu = np.where(m > 0, cs[:-1] / np.maximum(m, 1), np.nan)
    var = np.where(m > 1, cq[:-1] / np.maximum(m, 1) - mu ** 2, np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    z = (v - mu) / np.where(sd > EPS, sd, np.nan)
    z[:BURN] = np.nan
    return z


def streak(sign: np.ndarray) -> np.ndarray:
    out = np.zeros(len(sign))
    run = 0.0
    prev = 0.0
    for i, s in enumerate(sign):
        if s != 0 and s == prev:
            run += s
        else:
            run = s
        prev = s if s != 0 else prev
        out[i] = run
    return out


def dif(v: np.ndarray) -> np.ndarray:
    return np.r_[np.nan, np.diff(v)]


def imb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, EPS)


def build(d: pl.DataFrame) -> pl.DataFrame:
    """grid_{SYM}.parquet(1 秒グリッド)→ 特徴量 DataFrame。"""
    g = {c: d[c].to_numpy().astype(np.float64) for c in d.columns}
    ts = g["ts_s"]
    n = len(ts)
    F: dict[str, np.ndarray] = {}

    def put(name, v):
        F[name] = np.asarray(v, dtype=np.float64)

    bb, ba = g["bb"], g["ba"]
    mid = (bb + ba) / 2
    bad = ~np.isfinite(mid) | (g["crossed"] > 0) | (g["suspect"] > 0)
    mid_ok = np.where(bad, np.nan, mid)

    # ---------------- px / ret / rv / spr ----------------
    put("px_mid", mid_ok)
    spr_bp = (ba - bb) / mid * BP
    put("spr_bp", np.where(bad, np.nan, spr_bp))
    put("spr_mean_bp", g["spr_mean"] / mid * BP)
    put("spr_z", expand_z(F["spr_bp"]))
    for w in (10, 60, 300):
        put(f"spr_ma_{w}s", roll_mean(ts, F["spr_bp"], w))
    # 後ろ向きリターン(bp)
    lm = np.log(np.where(mid_ok > 0, mid_ok, np.nan))
    for w in (1, 5, 10, 30, 60, 300):
        j = np.searchsorted(ts, ts - w, side="right") - 1
        ok = (j >= 0) & (ts[np.maximum(j, 0)] >= ts - w - 2)
        r = np.where(ok, (lm - lm[np.maximum(j, 0)]) * BP, np.nan)
        put(f"ret_{w}s", r)
        put(f"ret_abs_{w}s", np.abs(r))
    d1 = dif(mid_ok) / mid * BP
    put("ret_1s_simple", d1)
    for w in (10, 60, 300):
        put(f"rv_{w}s", np.sqrt(roll_sum(ts, d1 ** 2, w)))
        put(f"path_{w}s", roll_sum(ts, g["path"], w) / mid * BP)
    hi60 = roll_sum(ts, np.nan_to_num(g["mid_hi"] - g["mid_lo"]), 60)
    put("range_sum_60s_bp", hi60 / mid * BP)
    put("ret_sign_streak", streak(np.sign(np.nan_to_num(d1))))
    put("px_tick_regime", np.floor(np.log10(np.maximum(mid, EPS))))

    # ---------------- obi ----------------
    b1, a1 = g["bb_sz"], g["ba_sz"]
    put("obi_1", imb(b1, a1))
    for k in (2, 3, 5, 10, 20):
        put(f"obi_D{k}", imb(g[f"D{k}_b"], g[f"D{k}_a"]))
    for w in (5, 10, 25, 50, 100):
        put(f"obi_W{w}", imb(g[f"W{w}_b"], g[f"W{w}_a"]))
    put("obi_tot", imb(g["tot_b"], g["tot_a"]))
    put("obi_near_deep", F["obi_W10"] - F["obi_W100"])
    put("obi_d1", dif(F["obi_1"]))
    for hl in (5, 30, 120):
        put(f"obi_ew{hl}", ewma(F["obi_1"], hl))
    put("obi_abs", np.abs(F["obi_1"]))
    put("obi_sign_streak", streak(np.sign(np.nan_to_num(F["obi_1"]))))
    put("obi_z", expand_z(F["obi_1"]))

    # ---------------- dep ----------------
    for t_ in ("b", "a"):
        for k in (1, 5, 20):
            put(f"dep_D{k}_{t_}", np.log1p(g[f"D{k}_{t_}"] if k > 1 else
                                           (b1 if t_ == "b" else a1)))
        for w in (10, 100):
            put(f"dep_W{w}_{t_}", np.log1p(g[f"W{w}_{t_}"]))
        put(f"dep_tot_{t_}", np.log1p(g[f"tot_{t_}"]))
        put(f"dep_nlv100_{t_}", g[f"nlv100_{t_}"])
        put(f"dep_nd_ratio_{t_}",
            g[f"W10_{t_}"] / np.maximum(g[f"W100_{t_}"], EPS))
        put(f"dep_d_W10_{t_}", dif(g[f"W10_{t_}"]))
    put("dep_sum_W10", np.log1p(g["W10_b"] + g["W10_a"]))
    put("dep_z", expand_z(F["dep_sum_W10"]))
    for w in (60, 300):
        put(f"dep_vol_{w}s", np.sqrt(roll_mean(
            ts, np.nan_to_num(dif(g["W10_b"] + g["W10_a"])) ** 2, w)))

    # ---------------- mic ----------------
    tw = b1 + a1
    micro1 = np.where(tw > 0, (bb * a1 + ba * b1) / np.maximum(tw, EPS), np.nan)
    put("mic_delta1_bp", np.where(bad, np.nan, (micro1 - mid) / mid * BP))
    for w in (10, 100):
        twd = g[f"W{w}_b"] + g[f"W{w}_a"]
        mw = np.where(twd > 0,
                      (bb * g[f"W{w}_a"] + ba * g[f"W{w}_b"]) / np.maximum(twd, EPS),
                      np.nan)
        put(f"mic_deltaW{w}_bp", np.where(bad, np.nan, (mw - mid) / mid * BP))
    put("mic_slope_1_W10", F["mic_delta1_bp"] - F["mic_deltaW10_bp"])
    put("mic_slope_W10_W100", F["mic_deltaW10_bp"] - F["mic_deltaW100_bp"])
    put("mic_d1", dif(F["mic_delta1_bp"]))
    for hl in (5, 30):
        put(f"mic_ew{hl}", ewma(F["mic_delta1_bp"], hl))

    # ---------------- shp / gap ----------------
    for base, pref in (("wavg_bp", "shp_wavg"), ("hhi10", "shp_hhi"),
                       ("wallsh10", "shp_wallsh"), ("px_range20_bp", "shp_rng"),
                       ("walldist10_bp", "shp_walld"),
                       ("gap12_bp", "gap_12"), ("maxgap10_bp", "gap_max")):
        vb, va = g[f"{base}_b"], g[f"{base}_a"]
        put(f"{pref}_b", vb)
        put(f"{pref}_a", va)
        put(f"{pref}_asym", vb - va)
    put("shp_elast_b", np.log(np.maximum(g["D20_b"], EPS))
        - np.log(np.maximum(b1, EPS)))
    put("shp_elast_a", np.log(np.maximum(g["D20_a"], EPS))
        - np.log(np.maximum(a1, EPS)))
    put("shp_elast_asym", F["shp_elast_b"] - F["shp_elast_a"])

    # ---------------- flw / cxl(指値フロー)----------------
    inc_b, inc_a = g["inc_b"], g["inc_a"]
    dec_b, dec_a = g["dec_b"], g["dec_a"]
    tb, tsv = g["tb_vol"], g["ts_vol"]
    # 取消の近似: 減少 − その側で食われた約定(テイカー買いは ask を食う)
    cxl_a1 = np.maximum(dec_a - tb, 0.0)
    cxl_b1 = np.maximum(dec_b - tsv, 0.0)
    for w in WINS:
        ib, ia = roll_sum(ts, inc_b, w), roll_sum(ts, inc_a, w)
        db_, da = roll_sum(ts, dec_b, w), roll_sum(ts, dec_a, w)
        put(f"flw_add_imb_{w}s", imb(ib, ia))
        put(f"flw_dec_imb_{w}s", imb(db_, da))
        put(f"flw_net_b_{w}s", ib - db_)
        put(f"flw_net_a_{w}s", ia - da)
        put(f"flw_net_imb_{w}s",
            ((ib - db_) - (ia - da)) / np.maximum(ib + ia + db_ + da, EPS))
        put(f"flw_churn_{w}s", (ib + ia + db_ + da)
            / np.maximum(g["W100_b"] + g["W100_a"], EPS))
        cb_ = roll_sum(ts, cxl_b1, w)
        ca_ = roll_sum(ts, cxl_a1, w)
        put(f"cxl_imb_{w}s", imb(cb_, ca_))
        put(f"cxl_rate_{w}s", (cb_ + ca_) / np.maximum(ib + ia, EPS))
        put(f"cxl_intensity_{w}s", (cb_ + ca_) / w)
        inb = roll_sum(ts, g["incn_b"], w)
        ina = roll_sum(ts, g["incn_a"], w)
        dnb = roll_sum(ts, g["decn_b"], w)
        dna = roll_sum(ts, g["decn_a"], w)
        put(f"flw_addn_imb_{w}s", imb(inb, ina))
        put(f"flw_decn_imb_{w}s", imb(dnb, dna))
    put("flw_add_burst", expand_z(inc_b + inc_a))
    put("cxl_burst", expand_z(cxl_b1 + cxl_a1))

    # ---------------- ofi ----------------
    ofi1 = g["ofi"]
    put("ofi_1s", ofi1)
    for w in WINS:
        r = roll_sum(ts, ofi1, w)
        put(f"ofi_{w}s", r)
        put(f"ofi_norm_{w}s", r / np.maximum(g["W10_b"] + g["W10_a"], EPS))
    for hl in (20, 100):
        put(f"ofi_ew{hl}", ewma(np.nan_to_num(ofi1), hl))
    put("ofi_z", expand_z(ofi1))

    # ---------------- trd ----------------
    sv = tb - tsv
    put("trd_sv_1s", sv)
    for w in WINS:
        rb = roll_sum(ts, tb, w)
        rs = roll_sum(ts, tsv, w)
        cb2 = roll_sum(ts, g["tb_cnt"], w)
        cs2 = roll_sum(ts, g["ts_cnt"], w)
        put(f"trd_sv_{w}s", rb - rs)
        put(f"trd_imb_{w}s", imb(rb, rs))
        put(f"trd_cnt_imb_{w}s", imb(cb2, cs2))
        put(f"trd_intensity_{w}s", (rb + rs) / w)
        put(f"trd_usd_{w}s", roll_sum(ts, g["usd_vol"], w))
        put(f"trd_abssv_ratio_{w}s", np.abs(rb - rs) / np.maximum(rb + rs, EPS))
    for hl in (20, 100):
        put(f"trd_sv_ew{hl}", ewma(sv, hl))
    put("trd_max_z", expand_z(g["max_trd"]))
    put("trd_z", expand_z(sv))
    put("trd_to_depth", sv / np.maximum(g["W10_b"] + g["W10_a"], EPS))
    put("trd_liq_60s", roll_sum(ts, g["n_liq"], 60))
    # 最後の約定からの経過秒(各時点で計算可能)
    had = (g["tb_cnt"] + g["ts_cnt"]) > 0
    last = np.where(had, ts, np.nan)
    np.maximum.accumulate(np.nan_to_num(last, nan=-1e18), out=last)
    put("trd_age_s", np.where(last > 0, ts - last, np.nan))

    # ---------------- act(活動・品質)----------------
    put("act_n_frames", g["n_frames"])
    put("act_n_bbo", g["n_bbo"])
    put("act_n_lvl", g["n_lvl_ch"])
    for w in (10, 60, 300):
        put(f"act_bbo_{w}s", roll_sum(ts, g["n_bbo"], w))
        put(f"act_lvl_{w}s", roll_sum(ts, g["n_lvl_ch"], w))
    put("act_q2t_60s", F["act_lvl_60s"]
        / np.maximum(roll_sum(ts, g["tb_cnt"] + g["ts_cnt"], 60), 1.0))
    put("act_bbo_z", expand_z(g["n_bbo"]))
    put("act_tk_lag_ms", g["tk_lag_ms"])
    hadq = g["n_bbo"] > 0
    lastq = np.where(hadq, ts, np.nan)
    np.maximum.accumulate(np.nan_to_num(lastq, nan=-1e18), out=lastq)
    put("act_bbo_age_s", np.where(lastq > 0, ts - lastq, np.nan))

    for w in (5, 30, 60, 300):
        put(f"obi_ma_{w}s", roll_mean(ts, F["obi_1"], w))
    put("ret_z", expand_z(F["ret_60s"]))
    put("rv_ratio_10_300", F["rv_10s"] / np.maximum(F["rv_300s"], EPS))
    put("spr_rel_ma300", F["spr_bp"] / np.maximum(F["spr_ma_300s"], EPS))
    put("trd_usd_z", expand_z(F["trd_usd_60s"]))
    put("flw_net_imb_ma300", roll_mean(ts, F["flw_net_imb_5s"], 300))
    put("cxl_imb_ma300", roll_mean(ts, F["cxl_imb_5s"], 300))

    # ---------------- x(掛け合わせ)----------------
    put("x_obi_spr", F["obi_1"] * F["spr_bp"])
    put("x_obi_rv", F["obi_1"] * F["rv_60s"])
    put("x_sv_over_depth_60s", F["trd_sv_60s"]
        / np.maximum(g["W10_b"] + g["W10_a"], EPS))
    put("x_impact_60s", F["ret_abs_60s"] / np.maximum(F["trd_usd_60s"], 1.0))
    put("x_ofi_over_rv", F["ofi_60s"] / np.maximum(F["rv_60s"], EPS))
    put("x_delta_minus_halfspread_obi",
        F["mic_delta1_bp"] - F["spr_bp"] / 2 * F["obi_1"])

    out = pl.DataFrame({"ts_s": ts.astype(np.int64)})
    for k, v in F.items():
        out = out.with_columns(pl.Series(k, v.astype(np.float32)))
    out = out.with_columns([
        pl.Series("_bad", bad),
        pl.Series("_mid", mid),
    ])
    return out
