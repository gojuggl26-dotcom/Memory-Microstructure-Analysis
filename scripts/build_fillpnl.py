"""250ms の約定ハザード + 条件つき損益モデル(二本立て)。

    uv run python scripts/build_fillpnl.py --coin xyz:MU
    uv run python scripts/build_fillpnl.py --coin xyz:MU --lag 0.065
    uv run python scripts/build_fillpnl.py --coin xyz:MU --simple
    uv run python scripts/build_fillpnl.py --coin xyz:MU --target mu

狙い
----
「60 秒 Hazard 1 本」ではなく、

    (1) P(Fill within 250ms | X_t)          ロジスティック
    (2) E[PnL_1s | Fill within 250ms, X_t]  最小二乗

を別々に当てはめ、**予測値の十分位 10×10** で
P(Fill) / PnL per fill / EV per candidate / N を出す。
EV > 0 の島があるかを見るのが目的で、無ければ Rare Maker 仮説を見直す。

**同じセルについて mid 基準と microprice 基準の EV を並べる**(h = 100ms/1s/10s)。
mid が BBO の不均衡を遅れて反映しているだけなら、mid 基準の島は microprice で消える。

時間契約
--------
- 説明変数 X_t は候補時点 t までに観測可能な値のみ(`build_quotes.py` の 45 列)
- 目的変数は t 以降: 250ms 以内に約定したか / 約定後の net PnL
- **学習は前 59 日、評価は後 39 日**(時間ブロック分割。行のシャッフルはしない)
- 標準化の平均・分散、十分位の境界、係数は**すべて学習期間だけ**から作る

費用
----
`label_pnl*` は既にメイカー手数料 0.088 bp を引いてある
(= 実現エッジ − 手数料 + markout)。**手仕舞いの費用は入っていない**
(往復は `build_roundtrip.py`)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_latency import fit_logit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")

H = 0.25              # 約定の判定窓 (秒)
LAG = 0.0             # 発注が板に載るまでの遅延 (秒)。--lag で与える。
#                       t に決めて t+LAG に載り、そこから H 秒待つ。
TRAIN_FRAC = 0.60     # 前半を学習に使う (既報と同じ)
SUB_FIT = 25          # 学習用に間引く間隔 (行が多すぎるため)
NQ = 10               # 十分位
MIN_N_DAY = 50        # 1 日 1 セルの最小候補数。これ未満の日はそのセルで使わない
NW_LAGS = 14          # 日次の Newey-West (週次の山がラグ 7 に出るため 2 周期)
SEED = 20260905
SIMPLE_MODE = False
TARGET = "mid"        # 損益モデルが当てはめる先 (mid / mu)

POS = ["spread_bp", "spread_tick", "bbo_age_s", "bbo_turnover",
       "queue_ahead_qty", "queue_ahead_est_cnt", "depth_own", "depth_opp",
       "depth_own_cum10", "depth_cum10", "taker_rate_10s", "rv_60s_bp",
       "ofi_var_60s", "cancel_10s", "add_10s", "fragility", "resilience",
       "touch_cancel_rate"]
SGN = ["ofi_0.1s", "ofi_0.25s", "ofi_0.5s", "ofi_1s", "ofi_10s",
       "aggr_0.1s", "aggr_1s", "aggr_10s"]
RAW = ["obi1", "obi2", "obi4", "obi10", "is_us_session", "is_weekend", "side"]
MKH = ["0.1", "1", "10"]
LABS = ([f"label_pnl_{h}s_bp" for h in MKH]
        + [f"label_pnlmu_{h}s_bp" for h in MKH])
NEED = (POS + SGN + RAW + ["obi_l2", "tau_hat_s", "hour_utc",
                           "label_fill_lat_s"] + [f"label_pnl_{h}s_bp" for h in MKH])
NEED_MU = [f"label_pnlmu_{h}s_bp" for h in MKH]
# 往復(手仕舞い込み)の損益。`build_roundtrip.py` が別ファイルに書いている。
# passive は T_max までに閉じないことがあるので、約定した件数を別に数える。
RTL = ["rt_taker_1s_bp", "rt_hyb_10s_bp", "rt_hyb_60s_bp"]
RTP = ["rt_pass_10s_bp", "rt_pass_60s_bp"]
NEED_RT = RTL + RTP
NAMES = ([f"log_{c}" for c in POS] + [f"slog_{c}" for c in SGN] + RAW
         + ["obi_l2", "obi_l2_na", "inv_tau", "hour_sin", "hour_cos"])
# ★ fragility = 取消/累積板厚、resilience = 追加/累積板厚 は分母が同じで
#   ほぼ同値。差 log(取消/追加) だけが効くため、全変数版では係数が ±8.8 まで
#   暴れる(多重共線性)。解釈可能な最小形として 3 変数版も回して比べる。
SIMPLE = ["log_ratio_cancel_add", "log_queue_ahead_qty", "log_spread_bp"]


def load(f: Path) -> pl.DataFrame:
    """本体と microprice ラベルを横に連結する(行順は同一)。"""
    T = pl.read_parquet(f, columns=NEED)
    dt = f.stem.split("=")[1]
    mu = f.parent.parent / f"{f.parent.name}_mu" / f"dt={dt}.parquet"
    M = pl.read_parquet(mu, columns=NEED_MU)
    assert M.height == T.height, (M.height, T.height)
    rt = f.parent.parent / f"{f.parent.name}_rt" / f"dt={dt}.parquet"
    if rt.exists():
        R = pl.read_parquet(rt, columns=NEED_RT)
        assert R.height == T.height, (R.height, T.height)
        M = M.hstack(R)
    else:
        M = M.hstack(pl.DataFrame({c: np.full(T.height, np.nan, np.float32)
                                   for c in NEED_RT}))
    return T.hstack(M)


def design(T: pl.DataFrame) -> np.ndarray:
    """54 列の生の表から説明変数行列を作る。t 以降の情報は一切使わない。"""
    cols = []
    for c in POS:
        v = T[c].to_numpy().astype(np.float64)
        cols.append(np.log1p(np.maximum(np.nan_to_num(v, nan=0.0), 0.0)))
    for c in SGN:
        v = np.nan_to_num(T[c].to_numpy().astype(np.float64), nan=0.0)
        cols.append(np.sign(v) * np.log1p(np.abs(v)))
    for c in RAW:
        cols.append(np.nan_to_num(T[c].to_numpy().astype(np.float64), nan=0.0))
    l2 = T["obi_l2"].to_numpy().astype(np.float64)
    cols.append(np.nan_to_num(l2, nan=0.0))
    cols.append(np.isnan(l2).astype(np.float64))          # 欠損そのものを説明変数に
    tau = T["tau_hat_s"].to_numpy().astype(np.float64)
    cols.append(1.0 / (1.0 + np.where(np.isfinite(tau), np.maximum(tau, 0.0),
                                      np.inf)))            # inf は 0 に落ちる
    hh = T["hour_utc"].to_numpy().astype(np.float64)
    cols.append(np.sin(2 * np.pi * hh / 24.0))
    cols.append(np.cos(2 * np.pi * hh / 24.0))
    X = np.column_stack(cols)
    assert X.shape[1] == len(NAMES), (X.shape, len(NAMES))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not SIMPLE_MODE:
        return X
    i = {c: k for k, c in enumerate(NAMES)}
    return np.column_stack([X[:, i["log_cancel_10s"]] - X[:, i["log_add_10s"]],
                            X[:, i["log_queue_ahead_qty"]],
                            X[:, i["log_spread_bp"]]])


def labels(T: pl.DataFrame):
    """約定したか (250ms 窓) と、6 通りの損益ラベル。"""
    lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
    P = {c: T[c].to_numpy().astype(np.float64) for c in LABS + RTL + RTP}
    tgt = "label_pnl_1s_bp" if TARGET == "mid" else "label_pnlmu_1s_bp"
    y = (np.isfinite(lat) & (lat > LAG) & (lat <= LAG + H)
         & np.isfinite(P[tgt]))                            # 損益が取れる約定のみ
    return y.astype(np.float64), P, tgt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--simple", action="store_true",
                    help="解釈可能な 3 変数版で回す")
    ap.add_argument("--target", choices=["mid", "mu"], default="mid",
                    help="損益モデルの当てはめ先。mid が既定 (セルの定義を揃える)")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="発注が板に載るまでの秒数 (例 0.065 = 1 ブロック)")
    a = ap.parse_args()
    global LAG, SIMPLE_MODE, TARGET
    LAG, SIMPLE_MODE, TARGET = a.lag, a.simple, a.target
    sfx = (("" if LAG == 0 else f"_lag{int(round(LAG*1000))}")
           + ("_simple" if SIMPLE_MODE else "")
           + ("" if TARGET == "mid" else "_mu"))
    tag = a.coin.replace(":", "_")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    tr, te = files[:ntr], files[ntr:]
    print(f"遅延 {1000*LAG:.0f} ms / 判定窓 {1000*H:.0f} ms / 当てはめ先 {TARGET}")
    print(f"学習 {len(tr)} 日 ({tr[0].stem[3:]}〜{tr[-1].stem[3:]}) / "
          f"評価 {len(te)} 日 ({te[0].stem[3:]}〜{te[-1].stem[3:]})", flush=True)

    # ---- 学習標本を集める(間引き。約定した行は全部残す) ----
    Xs, ys, ps, ms = [], [], [], []
    for k, f in enumerate(tr):
        T = load(f)
        X = design(T)
        y, P, tgt = labels(T)
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32))
        ys.append(y[sl])
        keep = y > 0                                       # 損益モデル用は全件
        Xs.append(X[keep].astype(np.float32))
        ys.append(np.full(int(keep.sum()), np.nan))        # 印。ロジットには使わない
        ps.append(P[tgt][keep])
        ms.append(np.zeros(X[sl].shape[0], bool))
        ms.append(np.ones(int(keep.sum()), bool))
        del T, X
        if (k + 1) % 20 == 0:
            print(f"  学習読み込み {k+1}/{len(tr)}", flush=True)
    XA = np.concatenate(Xs).astype(np.float64)
    yA = np.concatenate(ys)
    isf = np.concatenate(ms)                               # True = 約定行(全件側)
    PN = np.concatenate(ps)
    del Xs, ys, ps, ms

    # 標準化は**学習期間の間引き標本**から。評価期間は一切見ない
    base = XA[~isf]
    mu = base.mean(axis=0)
    sd = base.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (XA - mu) / sd

    print(f"ロジット学習 {int((~isf).sum()):,} 行 "
          f"(うち {1000*H:.0f}ms 約定 {int(yA[~isf].sum()):,} = "
          f"{100*float(yA[~isf].mean()):.2f}%)", flush=True)
    w_fill = fit_logit(Z[~isf], yA[~isf])
    print(f"損益モデル学習 {int(isf.sum()):,} 行 "
          f"(平均 {PN.mean():+.4f} bp)", flush=True)
    Zf = np.column_stack([np.ones(int(isf.sum())), Z[isf]])
    w_pnl, *_ = np.linalg.lstsq(Zf, PN, rcond=None)

    # ---- 十分位の境界も学習期間だけから ----
    def phat(X):
        z = np.column_stack([np.ones(X.shape[0]), (X - mu) / sd]) @ w_fill
        return 1.0 / (1.0 + np.exp(-z))

    def qhat(X):
        return np.column_stack([np.ones(X.shape[0]), (X - mu) / sd]) @ w_pnl

    pf_tr = phat(XA[~isf])
    qh_tr = qhat(XA[~isf])
    eh = np.linspace(0, 1, NQ + 1)[1:-1]
    e_p = np.quantile(pf_tr, eh)
    e_q = np.quantile(qh_tr, eh)
    assert np.all(np.diff(e_p) > 0) and np.all(np.diff(e_q) > 0), "境界が退化"
    del XA, Z, yA, PN, isf, base

    # ---- 評価期間を 1 日ずつ流して、セル × 日 で貯める ----
    rng = np.random.default_rng(SEED)
    rows = []
    for k, f in enumerate(te):
        T = load(f)
        X = design(T)
        y, P, _ = labels(T)
        spr = np.nan_to_num(T["spread_bp"].to_numpy().astype(np.float64))
        ob1 = np.nan_to_num(T["obi1"].to_numpy().astype(np.float64))
        of1 = np.nan_to_num(T["ofi_1s"].to_numpy().astype(np.float64))
        sd_ = T["side"].to_numpy()
        pf, qh = phat(X), qhat(X)
        ip = np.searchsorted(e_p, pf, side="right")
        iq = np.searchsorted(e_q, qh, side="right")
        cell = ip * NQ + iq
        rc = rng.integers(0, NQ * NQ, size=cell.size)      # 帰無対照(無作為割り当て)
        W = {c: np.where(y > 0, np.nan_to_num(P[c]), 0.0) for c in LABS + RTL}
        # passive は閉じなかった行を 0 と数えてはいけないので、件数を別に持つ
        for c in RTP:
            fin = (y > 0) & np.isfinite(P[c])
            W[c] = np.where(fin, np.nan_to_num(P[c]), 0.0)
            W["n_" + c] = fin.astype(np.float64)
        dt = f.stem.split("=")[1]
        L_ = NQ * NQ + 1
        # 買い候補と売り候補で島が同じか。片側だけなら結論が変わる。
        # 対象外の行は添字 NQ*NQ の「捨て箱」へ送り、集計の最後に落とす
        for nm, cc in (("model", cell), ("random", rc),
                       ("buy", np.where(sd_ > 0, cell, NQ * NQ)),
                       ("sell", np.where(sd_ < 0, cell, NQ * NQ))):
            n = np.bincount(cc, minlength=L_)[:NQ * NQ]
            nf = np.bincount(cc, weights=y, minlength=L_)[:NQ * NQ]
            Dd = {"dt": [dt] * (NQ * NQ), "kind": [nm] * (NQ * NQ),
                  "hd": np.repeat(np.arange(NQ), NQ).astype(np.int8),
                  "qd": np.tile(np.arange(NQ), NQ).astype(np.int8),
                  "n": n.astype(np.int64), "n_fill": nf}
            for c in W:
                Dd[f"s_{c}"] = np.bincount(cc, weights=W[c],
                                           minlength=L_)[:NQ * NQ]
            for nm2, v in (("mean_p", pf), ("mean_q", qh), ("mean_spread", spr),
                           ("mean_obi", ob1), ("mean_ofi", of1)):
                Dd[nm2] = (np.bincount(cc, weights=v, minlength=L_)[:NQ * NQ]
                           / np.maximum(n, 1))
            rows.append(pl.DataFrame(Dd))
        del T, X
        if (k + 1) % 10 == 0:
            print(f"  評価 {k+1}/{len(te)}", flush=True)
    D = pl.concat(rows)
    D.write_parquet(DATA / f"fillpnl_days_{tag}{sfx}.parquet")

    # ---- セルごとに集計 ----
    def nw(v):
        e = v - v.mean()
        gg = float((e * e).sum() / v.size)
        for lg in range(1, NW_LAGS + 1):
            gg += 2.0 * (1.0 - lg / (NW_LAGS + 1.0)) * float(
                (e[lg:] * e[:-lg]).sum() / v.size)
        return float(v.mean()), float(np.sqrt(max(gg, 0.0) / v.size))

    def agg(kind: str, Dx: pl.DataFrame) -> pl.DataFrame:
        P = (Dx.filter(pl.col("kind") == kind).group_by("hd", "qd").agg(
            n=pl.col("n").sum(), n_fill=pl.col("n_fill").sum(),
            **{f"s_{c}": pl.col(f"s_{c}").sum()
               for c in LABS + RTL + RTP + ["n_" + c for c in RTP]},
            mean_p=(pl.col("mean_p") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_q=(pl.col("mean_q") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_spread=(pl.col("mean_spread") * pl.col("n")).sum()
            / pl.col("n").sum(),
            mean_obi=(pl.col("mean_obi") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_ofi=(pl.col("mean_ofi") * pl.col("n")).sum() / pl.col("n").sum())
            .sort("hd", "qd"))
        ALL = LABS + RTL
        out = {c: [] for c in ALL}
        ses = {c: [] for c in ALL}
        nd = []
        S = Dx.filter(pl.col("kind") == kind).sort("dt")
        for hd, qd in zip(P["hd"], P["qd"]):
            s = S.filter((pl.col("hd") == hd) & (pl.col("qd") == qd))
            m = s["n"].to_numpy() >= MIN_N_DAY
            nn = np.maximum(s["n"].to_numpy(), 1)
            nd.append(int(m.sum()))
            for c in ALL:
                v = (s[f"s_{c}"].to_numpy() / nn)[m]
                if v.size < 5:
                    out[c].append(np.nan)
                    ses[c].append(np.nan)
                    continue
                a_, b_ = nw(v)
                out[c].append(a_)
                ses[c].append(b_)
        E = {}
        for c in ALL:
            k2 = (c.replace("label_pnl", "ev").replace("rt_", "evrt_")
                  .replace("_bp", ""))
            E[k2 + "_pooled"] = pl.col(f"s_{c}") / pl.col("n")
            E[k2] = pl.Series(out[c])
            E["se_" + k2] = pl.Series(ses[c])
        return P.with_columns(
            p_fill=pl.col("n_fill") / pl.col("n"),
            pnl_per_fill=pl.col("s_label_pnl_1s_bp") / pl.col("n_fill"),
            pnlmu_per_fill=pl.col("s_label_pnlmu_1s_bp") / pl.col("n_fill"),
            ev_pooled=pl.col("s_label_pnl_1s_bp") / pl.col("n"),
            ev_daily=pl.Series(out["label_pnl_1s_bp"]),
            se_daily=pl.Series(ses["label_pnl_1s_bp"]),
            n_days=pl.Series(nd), **E)

    M = agg("model", D)
    R = agg("random", D)
    agg("buy", D).write_parquet(DATA / f"fillpnl_buy_{tag}{sfx}.parquet")
    agg("sell", D).write_parquet(DATA / f"fillpnl_sell_{tag}{sfx}.parquet")
    days = sorted(D["dt"].unique().to_list())
    h1, h2 = days[:len(days) // 2], days[len(days) // 2:]
    H1 = agg("model", D.filter(pl.col("dt").is_in(h1)))
    H2 = agg("model", D.filter(pl.col("dt").is_in(h2)))
    M = M.with_columns(ev_h1=H1["ev_pooled"], ev_h2=H2["ev_pooled"])
    M.write_parquet(DATA / f"fillpnl_cells_{tag}{sfx}.parquet")
    R.write_parquet(DATA / f"fillpnl_random_{tag}{sfx}.parquet")

    # ---- 報告 ----
    NT = int(M["n"].sum())
    print(f"\n評価期間 {NT:,} 候補 / {1000*H:.0f}ms 約定 {int(M['n_fill'].sum()):,} "
          f"({100*float(M['n_fill'].sum())/NT:.2f}%)")
    ev = M["ev_daily"].to_numpy()
    se = M["se_daily"].to_numpy()
    ok = np.isfinite(ev) & np.isfinite(se)
    t = np.where(ok, ev / se, 0.0)
    zb = 3.481                                            # 両側 5% / 100 セル
    print(f"無条件の EV/候補 = {float(M['s_label_pnl_1s_bp'].sum())/NT:+.4f} bp (mid) "
          f"/ {float(M['s_label_pnlmu_1s_bp'].sum())/NT:+.4f} bp (microprice)")
    print(f"EV>0 のセル {int(np.sum(ok & (ev > 0)))}/100  "
          f"うち Bonferroni {int(np.sum(ok & (ev > 0) & (t > zb)))}")
    for h in MKH:
        for lab, nm in ((f"label_pnl_{h}s_bp", "mid       "),
                        (f"label_pnlmu_{h}s_bp", "microprice")):
            k2 = lab.replace("label_pnl", "ev").replace("_bp", "")
            e2 = M[k2].to_numpy()
            s2 = M["se_" + k2].to_numpy()
            o2 = np.isfinite(e2) & np.isfinite(s2)
            t2 = np.where(o2, e2 / np.where(s2 == 0, np.nan, s2), 0.0)
            print(f"  h={h:>4}s {nm}: EV>0 {int(np.sum(o2 & (e2 > 0)))}/100  "
                  f"Bonferroni {int(np.sum(o2 & (e2 > 0) & (t2 > zb)))}  "
                  f"全体 {float(M['s_' + lab].sum())/NT:+.4f} bp")
    win = ok & (ev > 0) & (t > zb)
    if win.any():
        nn = M["n"].to_numpy().astype(float)
        print(f"  ★ mid 基準の陽性 {int(win.sum())} セル(候補 {nn[win].sum():,.0f})の中で:")
        for h in MKH:
            a1 = float(M[f"s_label_pnl_{h}s_bp"].to_numpy()[win].sum() / nn[win].sum())
            a2 = float(M[f"s_label_pnlmu_{h}s_bp"].to_numpy()[win].sum()
                       / nn[win].sum())
            print(f"     h={h:>4}s  EV_mid {a1:+.4f}  /  EV_micro {a2:+.4f} bp")
    rv = R["ev_daily"].to_numpy()
    j = int(np.nanargmax(np.where(np.isfinite(rv), rv, -np.inf)))
    print(f"無作為対照の最良セル EV {float(R['ev_daily'][j]):+.4f} bp")

    json.dump({"train_days": len(tr), "test_days": len(te), "h": H, "lag": LAG,
               "target": TARGET, "simple": SIMPLE_MODE, "n_test": NT,
               "coef_fill": dict(zip(["const"] + (SIMPLE if SIMPLE_MODE else NAMES),
                                     map(float, w_fill))),
               "coef_pnl": dict(zip(["const"] + (SIMPLE if SIMPLE_MODE else NAMES),
                                    map(float, w_pnl)))},
              open(DATA / f"fillpnl_meta_{tag}{sfx}.json", "w"), indent=1)
    print(f"\n書き出し {DATA}/fillpnl_cells_{tag}{sfx}.parquet ほか")


if __name__ == "__main__":
    main()
