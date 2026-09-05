"""250ms の約定ハザード + 条件つき損益モデル(二本立て)。

    uv run python scripts/build_fillpnl.py --coin xyz:MU

狙い
----
「60 秒 Hazard 1 本」ではなく、

    (1) P(Fill within 250ms | X_t)          ロジスティック
    (2) E[PnL_1s | Fill within 250ms, X_t]  最小二乗

を別々に当てはめ、**予測値の十分位 10×10** で
P(Fill) / PnL per fill / EV per candidate / N を出す。
EV > 0 の島があるかを見るのが目的で、無ければ Rare Maker 仮説を見直す。

時間契約
--------
- 説明変数 X_t は候補時点 t までに観測可能な値のみ(`build_quotes.py` の 45 列)
- 目的変数は t 以降: 250ms 以内に約定したか / 約定後 1 秒の net PnL
- **学習は前 59 日、評価は後 39 日**(時間ブロック分割。行のシャッフルはしない)
- 標準化の平均・分散、十分位の境界、係数は**すべて学習期間だけ**から作る

費用
----
`label_pnl_1s_bp` は既にメイカー手数料 0.088 bp を引いてある
(= 実現エッジ − 手数料 + 1 秒 markout)。
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
#                       t+LAG より前の約定は「まだ板に居ない」ので約定しない。
TRAIN_FRAC = 0.60     # 前半を学習に使う (既報と同じ)
SUB_FIT = 25          # 学習用に間引く間隔 (行が多すぎるため)
NQ = 10               # 十分位
MIN_N_DAY = 50        # 1 日 1 セルの最小候補数。これ未満の日はそのセルで使わない
NW_LAGS = 14          # 日次の Newey-West (週次の山がラグ 7 に出るため 2 周期)
SEED = 20260905
SIMPLE_MODE = False

POS = ["spread_bp", "spread_tick", "bbo_age_s", "bbo_turnover",
       "queue_ahead_qty", "queue_ahead_est_cnt", "depth_own", "depth_opp",
       "depth_own_cum10", "depth_cum10", "taker_rate_10s", "rv_60s_bp",
       "ofi_var_60s", "cancel_10s", "add_10s", "fragility", "resilience",
       "touch_cancel_rate"]
SGN = ["ofi_0.1s", "ofi_0.25s", "ofi_0.5s", "ofi_1s", "ofi_10s",
       "aggr_0.1s", "aggr_1s", "aggr_10s"]
RAW = ["obi1", "obi2", "obi4", "obi10", "is_us_session", "is_weekend", "side"]
NEED = (POS + SGN + RAW + ["obi_l2", "tau_hat_s", "hour_utc",
                           "label_fill_lat_s", "label_pnl_1s_bp"])
# ★ fragility = 取消/累積板厚、resilience = 追加/累積板厚 は分母が同じで
#   ほぼ同値。差 log(取消/追加) だけが効くため、全変数版では係数が ±8.8 まで
#   暴れる(多重共線性)。解釈可能な最小形として 3 変数版も回して比べる。
SIMPLE = ["log_ratio_cancel_add", "log_queue_ahead_qty", "log_spread_bp"]
NAMES = ([f"log_{c}" for c in POS] + [f"slog_{c}" for c in SGN] + RAW
         + ["obi_l2", "obi_l2_na", "inv_tau", "hour_sin", "hour_cos"])


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
    lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
    pnl = T["label_pnl_1s_bp"].to_numpy().astype(np.float64)
    y = (np.isfinite(lat) & (lat > LAG) & (lat <= LAG + H)
         & np.isfinite(pnl))                               # 損益が取れる約定のみ
    return y.astype(np.float64), pnl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--simple", action="store_true",
                    help="解釈可能な 3 変数版で回す")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="発注が板に載るまでの秒数 (例 0.065 = 1 ブロック)")
    a = ap.parse_args()
    global LAG, SIMPLE_MODE
    LAG = a.lag
    SIMPLE_MODE = a.simple
    sfx = ("" if LAG == 0 else f"_lag{int(round(LAG*1000))}") +           ("_simple" if SIMPLE_MODE else "")
    tag = a.coin.replace(":", "_")
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    ntr = int(round(len(files) * TRAIN_FRAC))
    tr, te = files[:ntr], files[ntr:]
    print(f"遅延 {1000*LAG:.0f} ms / 判定窓 {1000*H:.0f} ms")
    print(f"学習 {len(tr)} 日 ({tr[0].stem[3:]}〜{tr[-1].stem[3:]}) / "
          f"評価 {len(te)} 日 ({te[0].stem[3:]}〜{te[-1].stem[3:]})", flush=True)

    # ---- 学習標本を集める(間引き。約定した行は全部残す) ----
    Xs, ys, ps, ms = [], [], [], []
    for k, f in enumerate(tr):
        T = pl.read_parquet(f, columns=NEED)
        X = design(T)
        y, pnl = labels(T)
        sl = slice(None, None, SUB_FIT)
        Xs.append(X[sl].astype(np.float32))
        ys.append(y[sl])
        keep = y > 0                                       # 損益モデル用は全件
        Xs.append(X[keep].astype(np.float32))
        ys.append(np.full(int(keep.sum()), np.nan))        # 印。ロジットには使わない
        ps.append(pnl[keep])
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
          f"(うち 250ms 約定 {int(yA[~isf].sum()):,} = "
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
        T = pl.read_parquet(f, columns=NEED)
        X = design(T)
        y, pnl = labels(T)
        spr = np.nan_to_num(T["spread_bp"].to_numpy().astype(np.float64))
        ob1 = np.nan_to_num(T["obi1"].to_numpy().astype(np.float64))
        of1 = np.nan_to_num(T["ofi_1s"].to_numpy().astype(np.float64))
        pf, qh = phat(X), qhat(X)
        ip = np.searchsorted(e_p, pf, side="right")
        iq = np.searchsorted(e_q, qh, side="right")
        cell = ip * NQ + iq
        rc = rng.integers(0, NQ * NQ, size=cell.size)      # 帰無対照(無作為割り当て)
        w = np.where(y > 0, np.nan_to_num(pnl), 0.0)
        dt = f.stem.split("=")[1]
        sd_ = T["side"].to_numpy()
        # 買い候補と売り候補で島が同じか。片側だけなら結論が変わる。
        # 対象外の行は添字 NQ*NQ の「捨て箱」へ送り、集計の最後に落とす
        for nm, cc in (("model", cell), ("random", rc),
                       ("buy", np.where(sd_ > 0, cell, NQ * NQ)),
                       ("sell", np.where(sd_ < 0, cell, NQ * NQ))):
            L_ = NQ * NQ + 1
            n = np.bincount(cc, minlength=L_)[:NQ * NQ]
            nf = np.bincount(cc, weights=y, minlength=L_)[:NQ * NQ]
            sp = np.bincount(cc, weights=w, minlength=L_)[:NQ * NQ]
            rows.append(pl.DataFrame({
                "dt": [dt] * (NQ * NQ), "kind": [nm] * (NQ * NQ),
                "hd": np.repeat(np.arange(NQ), NQ).astype(np.int8),
                "qd": np.tile(np.arange(NQ), NQ).astype(np.int8),
                "n": n.astype(np.int64), "n_fill": nf,
                "sum_pnl": sp,
                "mean_p": np.bincount(cc, weights=pf, minlength=L_)[:NQ * NQ] /
                np.maximum(n, 1),
                "mean_q": np.bincount(cc, weights=qh, minlength=L_)[:NQ * NQ] /
                np.maximum(n, 1),
                "mean_spread": np.bincount(cc, weights=spr, minlength=L_)[:NQ * NQ] /
                np.maximum(n, 1),
                "mean_obi": np.bincount(cc, weights=ob1, minlength=L_)[:NQ * NQ] /
                np.maximum(n, 1),
                "mean_ofi": np.bincount(cc, weights=of1, minlength=L_)[:NQ * NQ] /
                np.maximum(n, 1)}))
        del T, X
        if (k + 1) % 10 == 0:
            print(f"  評価 {k+1}/{len(te)}", flush=True)
    D = pl.concat(rows)
    D.write_parquet(DATA / f"fillpnl_days_{tag}{sfx}.parquet")

    # ---- セルごとに集計 ----
    def agg(kind: str) -> pl.DataFrame:
        S = D.filter(pl.col("kind") == kind)
        P = (S.group_by("hd", "qd").agg(
            n=pl.col("n").sum(), n_fill=pl.col("n_fill").sum(),
            sum_pnl=pl.col("sum_pnl").sum(),
            mean_p=(pl.col("mean_p") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_q=(pl.col("mean_q") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_spread=(pl.col("mean_spread") * pl.col("n")).sum()
            / pl.col("n").sum(),
            mean_obi=(pl.col("mean_obi") * pl.col("n")).sum() / pl.col("n").sum(),
            mean_ofi=(pl.col("mean_ofi") * pl.col("n")).sum() / pl.col("n").sum())
            .sort("hd", "qd"))
        ev, se, nd = [], [], []
        for hd, qd in zip(P["hd"], P["qd"]):
            s = S.filter((pl.col("hd") == hd) & (pl.col("qd") == qd)).sort("dt")
            m = s["n"].to_numpy() >= MIN_N_DAY
            v = (s["sum_pnl"].to_numpy() / np.maximum(s["n"].to_numpy(), 1))[m]
            nd.append(int(v.size))
            if v.size < 5:
                ev.append(np.nan); se.append(np.nan); continue
            e = v - v.mean()
            g = float((e * e).sum() / v.size)
            for lg in range(1, NW_LAGS + 1):
                g += 2.0 * (1.0 - lg / (NW_LAGS + 1.0)) * float(
                    (e[lg:] * e[:-lg]).sum() / v.size)
            ev.append(float(v.mean()))
            se.append(float(np.sqrt(max(g, 0.0) / v.size)))
        return P.with_columns(
            p_fill=pl.col("n_fill") / pl.col("n"),
            pnl_per_fill=pl.col("sum_pnl") / pl.col("n_fill"),
            ev_pooled=pl.col("sum_pnl") / pl.col("n"),
            ev_daily=pl.Series(ev), se_daily=pl.Series(se),
            n_days=pl.Series(nd))

    M = agg("model")
    R = agg("random")
    agg("buy").write_parquet(DATA / f"fillpnl_buy_{tag}{sfx}.parquet")
    agg("sell").write_parquet(DATA / f"fillpnl_sell_{tag}{sfx}.parquet")
    days = sorted(D["dt"].unique().to_list())
    h1, h2 = set(days[:len(days) // 2]), set(days[len(days) // 2:])
    Dall = D

    def half(keep):
        nonlocal D
        D = Dall.filter(pl.col("dt").is_in(list(keep)))
        out = agg("model")
        D = Dall
        return out

    H1, H2 = half(h1), half(h2)
    M = M.with_columns(ev_h1=H1["ev_pooled"], ev_h2=H2["ev_pooled"])
    M.write_parquet(DATA / f"fillpnl_cells_{tag}{sfx}.parquet")
    R.write_parquet(DATA / f"fillpnl_random_{tag}{sfx}.parquet")

    # ---- 報告 ----
    NT = int(M["n"].sum())
    print(f"\n評価期間 {NT:,} 候補 / 250ms 約定 {int(M['n_fill'].sum()):,} "
          f"({100*float(M['n_fill'].sum())/NT:.2f}%)")
    print(f"無条件の EV/候補 = {float(M['sum_pnl'].sum())/NT:+.4f} bp")
    t = M["ev_daily"].to_numpy() / M["se_daily"].to_numpy()
    pos = (M["ev_daily"].to_numpy() > 0)
    zb = 3.481                                            # 両側 5% / 100 セル
    print(f"EV>0 のセル {int(np.nansum(pos))}/100  "
          f"うち Bonferroni (|t|>{zb}) を超えるもの "
          f"{int(np.nansum(pos & (t > zb)))}")
    print(f"EV<0 で Bonferroni を超えるもの "
          f"{int(np.nansum((~pos) & (t < -zb)))}")
    i = int(np.nanargmax(M["ev_daily"].to_numpy()))
    print(f"最良セル H=D{int(M['hd'][i])+1} Q=D{int(M['qd'][i])+1}  "
          f"EV {float(M['ev_daily'][i]):+.4f} ± {float(M['se_daily'][i]):.4f} bp  "
          f"(t={t[i]:+.2f})  N={int(M['n'][i]):,}  "
          f"P(Fill) {100*float(M['p_fill'][i]):.2f}%  "
          f"PnL/fill {float(M['pnl_per_fill'][i]):+.3f}")
    e1 = M["ev_h1"].to_numpy(); e2 = M["ev_h2"].to_numpy()
    both = np.isfinite(e1) & np.isfinite(e2)
    print(f"評価期間の前半・後半とも正のセル "
          f"{int(((e1 > 0) & (e2 > 0))[both].sum())}/{int(both.sum())}  "
          f"前後半の相関 {np.corrcoef(e1[both], e2[both])[0,1]:+.3f}")
    j = int(np.nanargmax(R["ev_daily"].to_numpy()))
    print(f"無作為対照の最良セル EV {float(R['ev_daily'][j]):+.4f} bp "
          f"(100 セルの最大値。模様が無くてもこの程度は出る)")

    json.dump({"train_days": len(tr), "test_days": len(te), "h": H,
               "n_test": NT, "coef_fill": dict(zip(["const"] + NAMES,
                                                   map(float, w_fill))),
               "coef_pnl": dict(zip(["const"] + (SIMPLE if SIMPLE_MODE else NAMES), map(float, w_pnl)))},
              open(DATA / f"fillpnl_meta_{tag}{sfx}.json", "w"), indent=1)
    print(f"\n書き出し {DATA}/fillpnl_cells_{tag}{sfx}.parquet ほか")


if __name__ == "__main__":
    main()
