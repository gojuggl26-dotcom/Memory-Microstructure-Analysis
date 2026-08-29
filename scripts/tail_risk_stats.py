"""E[PnL] の分散推定とテールリスク + 既発表の検定の頑健性再計算。

【入力】data/capacity_100usd.csv(30 日 × レバ 1/3/5/10)
       data/capacity_v3.csv(Q=1 系列)/ data/liq_100usd.csv / data/backtest_v3.csv
【出力】data/tail_risk_stats.json + 標準出力

【x/y の時間関係】予測モデルではない。既に確定した日次損益系列の記述統計。

内容:
  A. 平均の分散 — iid SE と Newey-West HAC SE(自己相関補正)、有効標本数
  B. 分布診断 — 歪度・尖度・Jarque-Bera・Shapiro、ACF・Ljung-Box
  C. 頑健な検定 — Wilcoxon / ブロック符号反転 / ブロックブートストラップ t
  D. 集中度 — 上位日を除いた平均と p(結論が少数日に依存していないか)
  E. 安定性 — 前半/後半、時間トレンド、出来高・レンジとの相関
  F. 日次テール — VaR/ES とそのブートストラップ区間
  G. 年次テール — 二重ブロックブートストラップ + 経路の最大ドローダウン・破産確率
     ブロック長感度 b∈{1,2,3,5}、Student-t 当てはめによる感度
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
rng = np.random.default_rng(20260817)
NB = 20_000
YR = 365
CAP = 100.0

c = pl.read_csv(D / "capacity_100usd.csv")
ds = pl.read_csv(D / "daily_summary.csv").with_columns(
    ((pl.col("px_max") - pl.col("px_min")) / pl.col("mid_tw") * 100).alias("range_pct"))
out = {}


def acf(u, kmax):
    ub = u - u.mean()
    g0 = float(np.mean(ub * ub))
    return [float(np.mean(ub[k:] * ub[:-k]) / g0) for k in range(1, kmax + 1)]


def nw_se(u, L):
    n = len(u); ub = u - u.mean()
    v = np.mean(ub * ub)
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * np.mean(ub[k:] * ub[:-k])
    return float(np.sqrt(max(v, 1e-12) / n))


def block_idx(n, n_out, size, b):
    nb = int(np.ceil(n_out / b))
    st = rng.integers(0, n, size=(size, nb))
    ix = (st[:, :, None] + np.arange(b)[None, None, :]) % n
    return ix.reshape(size, -1)[:, :n_out]


def annual_double(u, B, b):
    """二重ブロックブートストラップ。経路 (B, 365) を返す。"""
    n = len(u)
    pop = u[block_idx(n, n, B, b)]                       # ① 母集団の再抽出
    ix = block_idx(n, YR, B, b)                          # ② そこから 1 年
    return np.take_along_axis(pop, ix, axis=1)


for L in sorted(c["lev"].unique().to_list()):
    sub = c.filter(pl.col("lev") == L).sort("dt")
    u = sub["pnl_usd"].to_numpy()
    n = len(u)
    r = {"n": n, "mean": float(u.mean()), "median": float(np.median(u)),
         "sd": float(u.std(ddof=1)), "se_iid": float(u.std(ddof=1) / np.sqrt(n))}

    # A. 自己相関と HAC
    r["acf"] = acf(u, 5)
    lb = n * (n + 2) * sum(rho**2 / (n - k - 1) for k, rho in enumerate(r["acf"]))
    r["ljung_box_p"] = float(stats.chi2.sf(lb, 5))
    Lnw = 3
    r["se_hac"] = nw_se(u, Lnw)
    r["t_hac"] = r["mean"] / r["se_hac"]
    r["p_hac"] = float(2 * stats.t.sf(abs(r["t_hac"]), n - 1))
    r["n_eff"] = float(n / (1 + 2 * sum(max(x, 0.0) for x in r["acf"][:3])))

    # B. 分布診断
    r["skew"] = float(stats.skew(u)); r["exkurt"] = float(stats.kurtosis(u))
    r["jb_p"] = float(stats.jarque_bera(u).pvalue)
    r["shapiro_p"] = float(stats.shapiro(u).pvalue)

    # C. 検定いろいろ
    r["p_t_iid"] = float(stats.ttest_1samp(u, 0).pvalue)
    r["p_wilcoxon"] = float(stats.wilcoxon(u).pvalue)
    npos = int((u > 0).sum())
    r["n_pos"] = npos
    r["p_sign"] = float(stats.binomtest(npos, n, 0.5, alternative="greater").pvalue)
    # ブロック符号反転(H0: 0 のまわりで対称、2 日ブロックで依存を保つ)
    b = 2; nb2 = int(np.ceil(n / b))
    ub2 = np.resize(u, nb2 * b).reshape(nb2, b); ub2[nb2 - 1, n - (nb2 - 1) * b:] = 0
    sgn = rng.choice([-1.0, 1.0], size=(NB, nb2))
    flips = (ub2[None, :, :] * sgn[:, :, None]).reshape(NB, -1)[:, :]
    m_null = flips.sum(axis=1) / n
    r["p_signflip_mean"] = float((m_null >= u.mean()).mean())
    pos_null = ((ub2[None, :, :] * sgn[:, :, None]) > 0).reshape(NB, -1).sum(axis=1)
    r["p_signflip_npos"] = float((pos_null >= npos).mean())
    # ブロックブートストラップ t(中心化)
    uc = u - u.mean()
    bs = uc[block_idx(n, n, NB, 2)]
    tstar = bs.mean(axis=1) / (bs.std(axis=1, ddof=1) / np.sqrt(n))
    t_obs = u.mean() / (u.std(ddof=1) / np.sqrt(n))
    r["p_boot_t"] = float((np.abs(tstar) >= abs(t_obs)).mean())

    # D. 集中度(上位日を除く)
    conc = {}
    order = np.argsort(u)[::-1]
    for k in (1, 2, 3):
        uu = np.delete(u, order[:k])
        tt_ = stats.ttest_1samp(uu, 0)
        conc[f"drop_top{k}"] = {"mean": float(uu.mean()), "p": float(tt_.pvalue)}
    conc["top3_share_of_sum"] = float(u[order[:3]].sum() / u.sum())
    r["concentration"] = conc

    # E. 安定性
    h = n // 2
    r["half1_mean"] = float(u[:h].mean()); r["half2_mean"] = float(u[h:].mean())
    r["p_halves"] = float(stats.ttest_ind(u[:h], u[h:], equal_var=False).pvalue)
    sp = stats.spearmanr(np.arange(n), u)
    r["trend_rho"] = float(sp.statistic); r["trend_p"] = float(sp.pvalue)
    vol_own = sub["volume"].to_numpy()
    r["corr_own_volume"] = float(np.corrcoef(u, vol_own)[0, 1])
    dsj = ds.filter(pl.col("dt").is_in(sub["dt"].to_list())).sort("dt")
    rp = dsj["range_pct"].to_numpy()
    r["corr_range"] = float(np.corrcoef(u, rp)[0, 1])

    # F. 日次テール(実測 + ブートストラップ区間)
    q05, q10 = np.percentile(u, [5, 10])
    r["var95_daily"] = float(q05); r["var90_daily"] = float(q10)
    r["es95_daily"] = float(u[u <= q05].mean()); r["worst_daily"] = float(u.min())
    bs2 = u[block_idx(n, n, NB, 2)]
    v95 = np.percentile(bs2, 5, axis=1)
    es95 = np.array([row[row <= q].mean() for row, q in zip(bs2, v95)])
    r["var95_ci80"] = [float(np.percentile(v95, 10)), float(np.percentile(v95, 90))]
    r["es95_ci80"] = [float(np.percentile(es95, 10)), float(np.percentile(es95, 90))]

    # G. 年次(二重・b=2)+ 経路リスク
    paths = annual_double(u, NB, 2)
    ann = paths.sum(axis=1)
    cum = np.cumsum(paths, axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0), axis=1)
    mdd = (peak - cum).max(axis=1)
    cmin = cum.min(axis=1)
    r["annual"] = {
        "mean": float(ann.mean()), "sd": float(ann.std(ddof=1)),
        "n_neg": int((ann < 0).sum()), "prob_neg": float((ann < 0).mean()),
        "p1": float(np.percentile(ann, 1)), "var5": float(np.percentile(ann, 5)),
        "es5": float(ann[ann <= np.percentile(ann, 5)].mean()),
        "mdd_med": float(np.median(mdd)), "mdd_p90": float(np.percentile(mdd, 90)),
        "mdd_p99": float(np.percentile(mdd, 99)),
        "prob_cum_below_m50": float((cmin < -50).mean()),
        "prob_cum_below_m100": float((cmin < -100).mean())}
    del paths, cum, peak
    out[f"lev{L}"] = r

# --- 1× のブロック長感度と Student-t 感度 ---
u1 = c.filter(pl.col("lev") == 1).sort("dt")["pnl_usd"].to_numpy()
sens = {}
for b in (1, 2, 3, 5):
    ann = annual_double(u1, NB, b).sum(axis=1)
    sens[f"b{b}"] = {"sd": float(ann.std(ddof=1)),
                     "prob_neg": float((ann < 0).mean()),
                     "p1": float(np.percentile(ann, 1)),
                     "var5": float(np.percentile(ann, 5))}
out["block_sensitivity_1x"] = sens

df_, loc_, sc_ = stats.t.fit(u1)
tsim = []
n1 = len(u1)
for _ in range(500):
    ub = u1[block_idx(n1, n1, 1, 2)[0]]
    d2, l2, s2 = stats.t.fit(ub)
    d2 = float(np.clip(d2, 2.05, 200))
    tsim.append(stats.t.rvs(d2, l2, s2, size=(40, YR), random_state=rng).sum(axis=1))
tann = np.concatenate(tsim)
out["t_fit_1x"] = {"df": float(df_), "loc": float(loc_), "scale": float(sc_),
                   "prob_neg": float((tann < 0).mean()),
                   "p1": float(np.percentile(tann, 1)),
                   "es5": float(tann[tann <= np.percentile(tann, 5)].mean())}

# --- capacity_v3 Q=1 と backtest_v3 A+B の再検定 ---
cv = pl.read_csv(D / "capacity_v3.csv").filter(pl.col("qty") == 1.0).sort("dt")
uq = cv["pnl_usd"].to_numpy(); nq = len(uq)
out["capacity_q1"] = {
    "n": nq, "median": float(np.median(uq)), "mean": float(uq.mean()),
    "n_pos": int((uq > 0).sum()),
    "p_sign": float(stats.binomtest(int((uq > 0).sum()), nq, 0.5, "greater").pvalue),
    "p_t_iid": float(stats.ttest_1samp(uq, 0).pvalue),
    "p_hac": float(2 * stats.t.sf(abs(uq.mean() / nw_se(uq, 3)), nq - 1)),
    "acf1": acf(uq, 1)[0]}
v3 = pl.read_csv(D / "backtest_v3.csv").filter(pl.col("policy") == "A+B 基準").sort("dt")
ub_ = v3["pnl_bp"].to_numpy()
out["v3_ab"] = {"n": len(ub_), "dt_min": str(v3["dt"].min()), "dt_max": str(v3["dt"].max()),
                "n_pos": int((ub_ > 0).sum()),
                "p_sign": float(stats.binomtest(int((ub_ > 0).sum()), len(ub_), 0.5,
                                                "greater").pvalue)}

# --- 清算検査との突合(日中最小 equity) ---
lq = pl.read_csv(D / "liq_100usd.csv")
liq = {}
for L in (1, 3, 5, 10):
    e = lq.filter(pl.col("lev") == L)["eq_min"].to_numpy()
    liq[f"lev{L}"] = {"worst": float(e.min()), "p10": float(np.percentile(e, 10))}
out["intraday_eq_min"] = liq

(D / "tail_risk_stats.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

# --- 表示 ---
for L in (1, 3, 5, 10):
    r = out[f"lev{L}"]
    print(f"\n===== レバ {L}× =====")
    print(f"日次: mean {r['mean']:.3f}  med {r['median']:.3f}  sd {r['sd']:.3f}"
          f"  skew {r['skew']:.2f}  exkurt {r['exkurt']:.2f}"
          f"  JB p {r['jb_p']:.3f}  Shapiro p {r['shapiro_p']:.3f}")
    print(f"ACF: {['%.2f' % x for x in r['acf']]}  LB p {r['ljung_box_p']:.3f}"
          f"  n_eff {r['n_eff']:.1f}")
    print(f"SE: iid {r['se_iid']:.3f}  HAC {r['se_hac']:.3f}"
          f"  | p: t_iid {r['p_t_iid']:.4f}  HAC {r['p_hac']:.4f}"
          f"  boot_t {r['p_boot_t']:.4f}  wilcoxon {r['p_wilcoxon']:.4f}"
          f"  sign {r['p_sign']:.4f}  signflip(mean) {r['p_signflip_mean']:.4f}"
          f"  signflip(npos) {r['p_signflip_npos']:.4f}")
    cc = r["concentration"]
    print(f"集中: top3/sum {cc['top3_share_of_sum']*100:.1f}%  "
          + "  ".join(f"-top{k}: mean {cc[f'drop_top{k}']['mean']:.2f}"
                      f" p {cc[f'drop_top{k}']['p']:.4f}" for k in (1, 2, 3)))
    print(f"安定: 前半 {r['half1_mean']:.2f} 後半 {r['half2_mean']:.2f}"
          f" p {r['p_halves']:.3f}  trend rho {r['trend_rho']:.2f} p {r['trend_p']:.3f}"
          f"  corr(出来高) {r['corr_own_volume']:.2f}  corr(レンジ) {r['corr_range']:.2f}")
    print(f"日次テール: VaR95 {r['var95_daily']:.2f} CI80 {r['var95_ci80']}"
          f"  ES95 {r['es95_daily']:.2f} CI80 {r['es95_ci80']}  最悪 {r['worst_daily']:.2f}")
    a = r["annual"]
    print(f"年次: mean {a['mean']:.0f}  sd {a['sd']:.0f}  P(<0) {a['n_neg']}/{NB}"
          f"  p1 {a['p1']:.0f}  VaR5 {a['var5']:.0f}  ES5 {a['es5']:.0f}")
    print(f"経路: MDD med {a['mdd_med']:.0f} p90 {a['mdd_p90']:.0f} p99 {a['mdd_p99']:.0f}"
          f"  P(累積<-50) {a['prob_cum_below_m50']*100:.2f}%"
          f"  P(累積<-100) {a['prob_cum_below_m100']*100:.2f}%")

print("\n===== 1× ブロック長感度 =====")
for k, v in out["block_sensitivity_1x"].items():
    print(f"  {k}: 年sd {v['sd']:.0f}  P(<0) {v['prob_neg']*100:.3f}%"
          f"  p1 {v['p1']:.0f}  VaR5 {v['var5']:.0f}")
tf = out["t_fit_1x"]
print(f"\n===== 1× Student-t 感度 =====  df {tf['df']:.1f}  "
      f"P(<0) {tf['prob_neg']*100:.3f}%  p1 {tf['p1']:.0f}  ES5 {tf['es5']:.0f}")
q = out["capacity_q1"]
print(f"\ncapacity Q=1: med {q['median']:.2f}  p_sign {q['p_sign']:.4f}"
      f"  p_t {q['p_t_iid']:.4f}  p_HAC {q['p_hac']:.4f}  acf1 {q['acf1']:.2f}")
v = out["v3_ab"]
print(f"v3 A+B: {v['n']} 日 {v['dt_min']}..{v['dt_max']}  黒字 {v['n_pos']}/{v['n']}"
      f"  p_sign {v['p_sign']:.4f}")
