"""確率微分方程式によるマーケットメイク損益のシミュレーション(ブラウン運動)。

【模型 — Avellaneda-Stoikov 型】
  幾何ブラウン運動の中値に、強度過程で到来する約定を重ねる。

      dS_t = σ S_t dW_t                          (中値。ドリフトは 0 に置く)
      dN^b_t, dN^a_t                             (買い/売りの約定。強度 λ^b, λ^a)
      dq_t = v (dN^b_t − dN^a_t)                 (在庫。|q| ≤ L で片側停止)
      dX_t = v[(S+δ)dN^a − (S−δ)dN^b] − 手数料    (現金)
      PnL_T = X_T + q_T S_T

  したがって損益は **スプレッド獲得 λvδ dt** と **在庫項 ∫ q dS** の和になる。

【★ドリフトを 0 に置く理由】
  中値に正のドリフトを入れると、在庫の偏りがそのまま利益になる。それは
  マーケットメイクではなく方向性の賭けの模擬である。μ=0(マルチンゲール)が
  中立な仮定であり、これを置くと **∫q dS は q が可予測なら期待値 0** になる。

【★逆選択 — これを入れないと模型は必ず儲かる】
  上の中立模型では E[PnL] = λvδT > 0 が自動的に成立してしまう(帰無対照 §5)。
  現実には**価格が動く方向の側で約定させられる**。これを強度に入れる:

      λ^b ∝ 1 − β Z_t,   λ^a ∝ 1 + β Z_t     (Z_t はその刻みの標準化ブラウン増分)

  買い注文は価格が下がる刻みで約定しやすくなり、E[∫q dS] < 0 になる。
  β は**実測した在庫コスト(−3.751 USD/日)に合わせて 1 個だけ較正**する。

【時間契約】
  予測模型ではない。約定は「その刻みの価格増分」と同時に決まり、
  執行価格は**刻みの開始時点の S**(注文はその前に置かれている)。
  未来の価格を見てから執行価格を決めることはしていない。

【較正値の出所】すべて scripts/sde_calibrate.py と sde_realized_vol.py の実測
  λ 5,694 約定/日 / 約定量 0.3624 単位 / δ 0.7096 bp(数量加重)
  σ 5.156 %/日(30 秒標本化 RV、評価窓中央値)/ 在庫上限 L=1.862, Q=0.3724
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")

N_STEPS = 86_400                 # Δt = 1 秒
CAP = 100.0
QTY = 0.37243947858473           # 資金 100 USD / レバ 1× のときの発注数量
L_INV = 5.0 * QTY                # 在庫上限 1.862 単位
V_FILL = 0.3624                  # 1 約定あたり数量(実測平均)
DELTA_BP = 0.7096                # スプレッド獲得(数量加重平均)
FEE_BP = 0.088
S0 = 53.7

INV_COST_TARGET = -3.751         # 実測の在庫コスト [USD/日](β の較正先)


def run(n_path, sigma_day, lam_day, beta, rng, delta_bp=DELTA_BP, fee_bp=FEE_BP,
        qty=QTY, keep_path=0, n_steps=N_STEPS, s0=S0, v_fill=V_FILL):
    """SDE を Euler-Maruyama で刻む。n_path 本を同時に進める。

    戻り値: pnl, eq_min, n_fill, inv_cost, gross, q_end,(必要なら)equity 経路
    """
    L = 5.0 * qty
    sig_s = sigma_day / np.sqrt(n_steps)
    p0 = lam_day / (2.0 * n_steps)
    d_rel = delta_bp / 1e4
    f_rel = fee_bp / 1e4

    S = np.full(n_path, s0)
    q = np.zeros(n_path)
    cash = np.zeros(n_path)
    eq_min = np.zeros(n_path)
    n_fill = np.zeros(n_path, dtype=np.int32)
    gross = np.zeros(n_path)                 # スプレッド獲得の累計 [USD]
    keep = np.empty((keep_path, n_steps // 60 + 1)) if keep_path else None
    if keep_path:
        keep[:, 0] = 0.0

    for t in range(n_steps):
        Z = rng.standard_normal(n_path)
        pb = p0 * (1.0 - beta * Z)
        pa = p0 * (1.0 + beta * Z)
        np.clip(pb, 0.0, 1.0, out=pb)
        np.clip(pa, 0.0, 1.0, out=pa)
        pb *= (q < L)                         # 在庫上限に達した側は出さない
        pa *= (q > -L)
        hb = rng.random(n_path) < pb
        ha = rng.random(n_path) < pa
        # 執行は刻み開始時点の価格。買いは S(1−δ)、売りは S(1+δ)
        pxb = S * (1.0 - d_rel)
        pxa = S * (1.0 + d_rel)
        vb = v_fill * hb
        va = v_fill * ha
        cash += -pxb * vb + pxa * va - f_rel * (pxb * vb + pxa * va)
        gross += d_rel * S * (vb + va)
        q += vb - va
        n_fill += hb + ha
        # 価格が動く(在庫はこの動きを丸ごと被る = 逆選択の経路)
        S *= np.exp(-0.5 * sig_s * sig_s + sig_s * Z)
        eq = cash + q * S
        np.minimum(eq_min, eq, out=eq_min)
        if keep_path and (t + 1) % 60 == 0:
            keep[:, (t + 1) // 60] = eq[:keep_path]

    pnl = cash + q * S
    inv_cost = pnl - gross + f_rel * 0.0      # 下で厳密に計算し直す
    # 厳密な分解: PnL = 粗利 − 手数料 + 在庫コスト
    fee_tot = f_rel * s0 * v_fill * n_fill    # 近似(価格変動ぶんは 1% 未満)
    inv_cost = pnl - gross + fee_tot
    out = {"pnl": pnl, "eq_min": eq_min, "n_fill": n_fill,
           "gross": gross, "fee": fee_tot, "inv_cost": inv_cost, "q_end": q}
    if keep_path:
        out["path"] = keep
    return out


def calib_lambda(sigma_day, lam_target, beta, rng, n=1500, iters=3):
    """在庫上限で片側が止まるぶん実現約定数は入力 λ より減る。

    実測の λ は**制約下で実現した約定数**なので、模型の入力側を逆算して合わせる。
    """
    lam_in = lam_target
    for _ in range(iters):
        r = run(n, sigma_day, lam_in, beta, np.random.default_rng(rng.integers(1 << 30)))
        got = float(r["n_fill"].mean())
        if got <= 0:
            break
        lam_in *= lam_target / got
    return float(lam_in)


def _ens_inv(beta, lam_arr, sig_arr, scale, rng, n_per_day):
    """30 日ぶんの母数で回したときの在庫コストの平均。"""
    tot = []
    for lm, sg in zip(lam_arr, sig_arr):
        r = run(n_per_day, float(sg), float(lm) * scale, beta,
                np.random.default_rng(rng.integers(1 << 30)))
        tot.append(r["inv_cost"])
    return float(np.concatenate(tot).mean())


def calib_beta(lam_arr, sig_arr, scale, rng, target=INV_COST_TARGET,
               n_per_day=400, iters=4):
    """在庫コストが実測値に一致する β を求める(1 母数 1 モーメント)。

    ★解析的には E[∫q dS] = −v β S σ_step λ で β に比例するが、
      在庫上限があると β が大きいほど在庫が偏りやすくなり **超線形**になる。
      初版は β≤0.05 の 2 点から線形外挿したため目標を 23% 外した。
      割線法で反復する。較正は**実測 30 日の母数集合の上で**行う
      (最終的な年次シミュレーションがその集合を使うため)。
    """
    hist = []
    b0 = 0.0
    c0 = _ens_inv(b0, lam_arr, sig_arr, scale, rng, n_per_day)
    b1 = 0.20
    c1 = _ens_inv(b1, lam_arr, sig_arr, scale, rng, n_per_day)
    hist += [(b0, c0), (b1, c1)]
    for _ in range(iters):
        if abs(c1 - c0) < 1e-9:
            break
        b2 = b1 + (target - c1) * (b1 - b0) / (c1 - c0)
        b2 = float(np.clip(b2, 0.0, 2.0))
        c2 = _ens_inv(b2, lam_arr, sig_arr, scale, rng, n_per_day)
        hist.append((b2, c2))
        b0, c0, b1, c1 = b1, c1, b2, c2
        if abs(c1 - target) < 0.02:
            break
    return float(b1), hist


def main() -> None:
    rng = np.random.default_rng(20260817)
    calib = pl.read_csv(D / "sde_calib_daily.csv")
    rv = pl.read_csv(D / "sde_realized_vol.csv")
    j = calib.join(rv.select(["dt", "rv30", "rv1", "rv300"]), on="dt")
    lam_d = j["n_fill"].to_numpy().astype(float)
    sig_d = j["rv30"].to_numpy()
    pnl_meas = j["pnl_usd"].to_numpy()
    SIG = float(np.median(sig_d)); LAM = float(np.median(lam_d))
    out = {"calib": {"sigma_day_med": SIG, "lam_day_med": LAM,
                     "sigma_mean": float(sig_d.mean()), "lam_mean": float(lam_d.mean()),
                     "v_fill": V_FILL, "delta_bp": DELTA_BP, "fee_bp": FEE_BP,
                     "qty": QTY, "inv_limit": L_INV, "S0": S0,
                     "corr_lam_sigma": float(np.corrcoef(lam_d, sig_d)[0, 1])}}
    print("=== 較正値 ===")
    print(f"  σ  中央値 {SIG*100:.3f} %/日   平均 {sig_d.mean()*100:.3f}%")
    print(f"  λ  中央値 {LAM:,.0f} 約定/日  平均 {lam_d.mean():,.0f}")
    print(f"  δ {DELTA_BP} bp / 手数料 {FEE_BP} bp / 約定量 {V_FILL} / 在庫上限 {L_INV:.3f}")
    print(f"  corr(λ, σ) = {out['calib']['corr_lam_sigma']:.3f}")

    # --- λ の較正(在庫上限で止まるぶんを逆算)---
    LAM_IN = calib_lambda(SIG, LAM, 0.2, rng)
    SCALE = LAM_IN / LAM
    out["calib"]["lam_input"] = LAM_IN
    out["calib"]["lam_scale"] = SCALE
    print(f"  λ 入力(在庫制約の逆算) {LAM_IN:,.0f} → 実現 {LAM:,.0f} 約定/日"
          f"  (倍率 {SCALE:.4f})", flush=True)

    # --- β の較正(在庫コストを 1 点だけ合わせる。実測 30 日の母数集合の上で)---
    beta, hist = calib_beta(lam_d, sig_d, SCALE, rng)
    out["calib"]["beta"] = beta
    out["calib"]["beta_iters"] = [[float(a), float(b)] for a, b in hist]
    sig1s = SIG / np.sqrt(86400)
    beta_1s = 0.1253 / (sig1s * 1e4)
    out["calib"]["beta_from_1s_adv"] = float(beta_1s)
    out["calib"]["horizon_equiv_s"] = float((beta / beta_1s) ** 2)
    print(f"\n=== β の較正(割線法・30 日母数集合)===")
    for b_, c_ in hist:
        print(f"  β={b_:.4f} → 在庫コスト {c_:+.3f} USD/日")
    print(f"  → 実測 {INV_COST_TARGET:+.3f} に合う β = {beta:.4f}")
    print(f"  報告32 の 1 秒逆選択から出る β = {beta_1s:.4f}"
          f"  → 比 {beta/beta_1s:.2f} 倍 = 逆選択の等価地平 "
          f"{out['calib']['horizon_equiv_s']:.1f} 秒", flush=True)

    # --- ★本体: 1,000 本のシミュレーション(中央値パラメータ)---
    N = 1000
    base = run(N, SIG, LAM_IN, beta, rng, keep_path=N)
    out["base"] = {k: {"mean": float(base[k].mean()), "sd": float(base[k].std(ddof=1)),
                       "med": float(np.median(base[k])),
                       "p": {f"p{p}": float(np.percentile(base[k], p))
                             for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}}
                   for k in ("pnl", "eq_min", "inv_cost", "gross", "n_fill", "q_end")}
    np.savez_compressed(D / "sde_paths.npz", path=base["path"],
                        pnl=base["pnl"], eq_min=base["eq_min"],
                        inv_cost=base["inv_cost"], gross=base["gross"])
    p = base["pnl"]
    print(f"\n=== ★1,000 本 / 中央値パラメータ / 1 日 ===")
    print(f"  E[PnL] {p.mean():+.3f} USD/日  sd {p.std(ddof=1):.3f}  中央値 {np.median(p):+.3f}")
    print(f"  分位 p1 {np.percentile(p,1):+.2f}  p5 {np.percentile(p,5):+.2f}"
          f"  p25 {np.percentile(p,25):+.2f}  p75 {np.percentile(p,75):+.2f}"
          f"  p95 {np.percentile(p,95):+.2f}  p99 {np.percentile(p,99):+.2f}")
    print(f"  黒字割合 {(p>0).mean()*100:.1f}%   日中最小 equity: 中央値"
          f" {np.median(base['eq_min']):.2f}  最悪 {base['eq_min'].min():.2f}")
    print(f"  内訳: 粗利 {base['gross'].mean():+.3f}  手数料 {-base['fee'].mean():+.3f}"
          f"  在庫 {base['inv_cost'].mean():+.3f}")
    print(f"  実測(30 日): E[PnL] {pnl_meas.mean():+.3f}  sd {pnl_meas.std(ddof=1):.3f}", flush=True)

    # --- 帰無対照 ---
    print(f"\n=== 帰無対照(模型が自動的に儲けを作っていないかの検査)===")
    ctrl = {}
    for nm, bt, dl in [("β=0 逆選択なし", 0.0, DELTA_BP),
                       ("δ=0 獲得なし", beta, 0.0),
                       ("両方 0", 0.0, 0.0)]:
        r = run(N, SIG, LAM_IN, bt, rng, delta_bp=dl)
        ctrl[nm] = {"pnl_mean": float(r["pnl"].mean()), "pnl_sd": float(r["pnl"].std(ddof=1)),
                    "inv": float(r["inv_cost"].mean())}
        print(f"  {nm:<16} E[PnL] {r['pnl'].mean():+8.3f}  sd {r['pnl'].std(ddof=1):6.3f}"
              f"  在庫コスト {r['inv_cost'].mean():+7.3f}")
    out["null_controls"] = ctrl

    # --- σ 掃引(ボラティリティ応答曲線)---
    print(f"\n=== σ 掃引(λ は中央値に固定)===")
    print(f"{'σ/日':>8}{'年率':>8}{'E[PnL]':>10}{'sd':>8}{'粗利':>9}{'在庫':>9}{'黒字%':>8}{'最悪eq':>9}")
    sweep = []
    for s in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.30]:
        r = run(N, s, LAM_IN, beta, rng)
        sweep.append({"sigma": s, "pnl_mean": float(r["pnl"].mean()),
                      "pnl_sd": float(r["pnl"].std(ddof=1)),
                      "gross": float(r["gross"].mean()), "inv": float(r["inv_cost"].mean()),
                      "pos_frac": float((r["pnl"] > 0).mean()),
                      "eq_min_worst": float(r["eq_min"].min()),
                      "eq_min_p1": float(np.percentile(r["eq_min"], 1))})
        w = sweep[-1]
        print(f"{s*100:>7.1f}%{s*np.sqrt(365)*100:>7.0f}%{w['pnl_mean']:>10.3f}"
              f"{w['pnl_sd']:>8.3f}{w['gross']:>9.3f}{w['inv']:>9.3f}"
              f"{w['pos_frac']*100:>7.1f}%{w['eq_min_worst']:>9.2f}", flush=True)
    out["sigma_sweep"] = sweep

    # --- 実測 30 日のパラメータで回す(分散の再現性検査)---
    print(f"\n=== 実測 30 日のパラメータで各 2,000 本 ===")
    day_pnl = np.empty((len(lam_d), 2000))
    day_eqmin = np.empty((len(lam_d), 2000))
    for i, (lm, sg) in enumerate(zip(lam_d, sig_d)):
        r = run(2000, float(sg), float(lm) * SCALE, beta, rng)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/30 日", flush=True)
        day_pnl[i] = r["pnl"]; day_eqmin[i] = r["eq_min"]
    mix = day_pnl.ravel()
    print(f"  SDE(母数変動込み): E {mix.mean():+.3f}  sd {mix.std(ddof=1):.3f}"
          f"  p5 {np.percentile(mix,5):+.2f}  p95 {np.percentile(mix,95):+.2f}")
    print(f"  実測 30 日        : E {pnl_meas.mean():+.3f}  sd {pnl_meas.std(ddof=1):.3f}"
          f"  p5 {np.percentile(pnl_meas,5):+.2f}  p95 {np.percentile(pnl_meas,95):+.2f}")
    print(f"  日ごとの平均の相関(SDE vs 実測): "
          f"{np.corrcoef(day_pnl.mean(axis=1), pnl_meas)[0,1]:.3f}")
    out["day_level"] = {"sde_mean": float(mix.mean()), "sde_sd": float(mix.std(ddof=1)),
                        "meas_mean": float(pnl_meas.mean()),
                        "meas_sd": float(pnl_meas.std(ddof=1)),
                        "corr_daily": float(np.corrcoef(day_pnl.mean(axis=1), pnl_meas)[0, 1]),
                        "sd_fixed_param": float(base["pnl"].std(ddof=1))}
    np.savez_compressed(D / "sde_day_ensemble.npz", pnl=day_pnl, eq_min=day_eqmin,
                        lam=lam_d, sigma=sig_d, pnl_meas=pnl_meas)

    # --- ★年次 1,000 本(日次型をブロック復元抽出 → SDE の条件付き分布から抽出)---
    print(f"\n=== ★年次 1,000 本(365 日、2 日ブロック)===")
    NB = 1000; YR = 365; BLK = 2
    nd = len(lam_d)
    nblk = int(np.ceil(YR / BLK))
    st = rng.integers(0, nd, size=(NB, nblk))
    ix = (st[:, :, None] + np.arange(BLK)[None, None, :]) % nd
    ix = ix.reshape(NB, -1)[:, :YR]
    pick = rng.integers(0, 2000, size=(NB, YR))
    ann_daily = day_pnl[ix, pick]                     # (1000, 365)
    ann = ann_daily.sum(axis=1)
    cum = np.cumsum(ann_daily, axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0), axis=1)
    mdd = (peak - cum).max(axis=1)
    eq_daily = day_eqmin[ix, pick]                    # その日の日中最小 equity
    ruin = ((cum - ann_daily + eq_daily) <= -CAP).any(axis=1)   # 日中に資金消尽
    qs = [1, 5, 10, 25, 50, 70, 75, 80, 90, 95, 99]
    out["annual"] = {"mean": float(ann.mean()), "sd": float(ann.std(ddof=1)),
                     "prob_neg": float((ann < 0).mean()),
                     "prob_ruin": float(ruin.mean()),
                     "mdd_med": float(np.median(mdd)), "mdd_p90": float(np.percentile(mdd, 90)),
                     "mdd_p99": float(np.percentile(mdd, 99)),
                     "pct": {f"p{q}": float(np.percentile(ann, q)) for q in qs}}
    print(f"  E[年次] {ann.mean():+,.0f} USD  sd {ann.std(ddof=1):,.0f}"
          f"  → APY {ann.mean()/CAP*100:,.0f}%")
    print("  分位: " + "  ".join(f"p{q} {np.percentile(ann,q)/CAP*100:,.0f}%" for q in qs))
    print(f"  P(年次<0) {(ann<0).mean()*100:.2f}%   P(日中に資金消尽) {ruin.mean()*100:.2f}%")
    print(f"  最大ドローダウン 中央値 {np.median(mdd):.0f}  p90 {np.percentile(mdd,90):.0f}"
          f"  p99 {np.percentile(mdd,99):.0f}")
    np.savez_compressed(D / "sde_annual.npz", ann=ann, ann_daily=ann_daily, mdd=mdd)

    (D / "sde_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print("\n保存: sde_results.json / sde_paths.npz / sde_day_ensemble.npz / sde_annual.npz")


if __name__ == "__main__":
    main()
