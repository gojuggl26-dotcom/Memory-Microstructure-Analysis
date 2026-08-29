"""★結合掃引 — λ と σ は独立に動かせない。

【なぜ必要か】
  sde_simulate.py の σ 掃引は λ を固定した「他の条件を同じにした」比較である。
  しかし実測では **corr(λ, σ) = 0.909**、当てはめると

      λ = 174 + 112,311·σ        (σ は小数。切片は λ の 3% 未満なのでほぼ比例)

  つまり**ボラティリティが高い日は約定も多い**。分離できない。
  λ ∝ σ を代入すると解析解は σ の二次関数になる:

      E[PnL] = k·v·S·[ σ(δ−f) − β σ²/√N ]

  → **上に凸**。σ→0 では約定が来ず 0、σ が大きいと逆選択に食われて 0 を割る。
      最適 σ = (δ−f)√N / (2β)、損益 0 になる σ = その 2 倍。

  「ボラが高いほうがマーケットメイクは儲かる」でも「低いほうが安全」でもなく、
  **内点に最適がある**というのがこの模型の予測である。実測がその内点の
  どちら側にいるかで、取るべき行動が変わる。

【時間契約】sde_simulate.py と同一。予測模型ではない。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sde_simulate import run, D, CAP                     # noqa: E402

res = json.loads((D / "sde_results.json").read_text(encoding="utf-8"))
c = res["calib"]
beta = c["beta"]
SCALE = c["lam_scale"]
fit = np.load(D / "sde_lam_sigma_fit.npz")
A, B = float(fit["a"]), float(fit["b"])
rng = np.random.default_rng(20260818)
N = 1000

print(f"β = {beta:.4f} / λ = {A:,.0f} + {B:,.0f}·σ / λ 入力倍率 {SCALE:.4f}")
print("\n=== 結合掃引(λ が σ に追従)===")
print(f"{'σ/日':>8}{'年率':>7}{'λ/日':>9}{'E[PnL]':>10}{'sd':>8}{'粗利':>9}"
      f"{'在庫':>9}{'黒字%':>8}{'eq_min p1':>11}")
rows = []
for s in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05156, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]:
    lam = max(A + B * s, 50.0)
    r = run(N, s, lam * SCALE, beta, rng)
    row = {"sigma": s, "lam": lam, "pnl_mean": float(r["pnl"].mean()),
           "pnl_sd": float(r["pnl"].std(ddof=1)), "gross": float(r["gross"].mean()),
           "inv": float(r["inv_cost"].mean()), "pos_frac": float((r["pnl"] > 0).mean()),
           "eq_min_p1": float(np.percentile(r["eq_min"], 1)),
           "eq_min_worst": float(r["eq_min"].min()),
           "n_fill": float(r["n_fill"].mean())}
    rows.append(row)
    print(f"{s*100:>7.2f}%{s*np.sqrt(365)*100:>6.0f}%{lam:>9,.0f}{row['pnl_mean']:>10.3f}"
          f"{row['pnl_sd']:>8.3f}{row['gross']:>9.3f}{row['inv']:>9.3f}"
          f"{row['pos_frac']*100:>7.1f}%{row['eq_min_p1']:>11.2f}", flush=True)

# 解析解による最適点と分岐点
d_bp, f_bp = c["delta_bp"], c["fee_bp"]
sig_star = (d_bp - f_bp) / 1e4 * np.sqrt(86400) / beta
print(f"\n=== 解析解 ===")
print(f"  最適 σ   = (δ−f)√N / (2β) = {sig_star/2*100:.3f} %/日"
      f"  (年率 {sig_star/2*np.sqrt(365)*100:.0f}%)")
print(f"  分岐点 σ = (δ−f)√N / β    = {sig_star*100:.3f} %/日"
      f"  (年率 {sig_star*np.sqrt(365)*100:.0f}%)")
print(f"  実測 σ   = {c['sigma_day_med']*100:.3f} %/日"
      f"  → 最適の {c['sigma_day_med']/(sig_star/2):.2f} 倍、分岐点の"
      f" {c['sigma_day_med']/sig_star*100:.0f}%")
pv = np.array([r["pnl_mean"] for r in rows])
sv = np.array([r["sigma"] for r in rows])
i = int(np.argmax(pv))
print(f"  シミュレーションの最大は σ={sv[i]*100:.2f}% で {pv[i]:+.3f} USD/日")
k = np.where(np.diff(np.sign(pv)))[0]
if k.size:
    jj = k[-1]
    sb = sv[jj] + (sv[jj + 1] - sv[jj]) * (0 - pv[jj]) / (pv[jj + 1] - pv[jj])
    print(f"  シミュレーションの損益 0 は σ={sb*100:.2f}%")

out = {"rows": rows, "sigma_opt_analytic": float(sig_star / 2),
       "sigma_breakeven_analytic": float(sig_star), "beta": beta,
       "lam_fit_a": A, "lam_fit_b": B}
(D / "sde_joint_sweep.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
print("\n保存: data/sde_joint_sweep.json")
