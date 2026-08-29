"""SDE の解析解と損益分岐点 — シミュレーションの独立検算も兼ねる。

【解析解】
  模型の 1 日あたり期待損益は、在庫上限に当たらない極限で閉じた形になる。

      E[PnL] = λ v S (δ − f)  +  E[∫ q dS]
      E[∫ q dS] = −λ v S β σ_Δ            (σ_Δ = σ_day / √n_steps)

  導出: 刻み t で E[dq_t | Z_t] = v·p0·(1−βZ) − v·p0·(1+βZ) = −2 v p0 β Z_t。
  在庫は刻み内の価格変化 ΔS = S σ_Δ Z_t を丸ごと被るので
      E[q_t ΔS_t] = E[dq_t ΔS_t] = −2 v p0 β S σ_Δ E[Z²] = −2 v p0 β S σ_Δ。
  1 日で n_steps 刻み、λ = 2 p0 n_steps だから合計は −λ v S β σ_Δ。
  (q_{t−1} は Z_t と独立なので E[q_{t−1} ΔS_t] = 0 — これが「ドリフト 0 なら
   在庫項はマルチンゲール」ということで、逆選択 β だけが期待値を生む。)

  したがって **1 約定あたり**では

      E[PnL]/約定 = v S (δ − f − β σ_Δ)          [USD]
      損益分岐:      δ* = f + β σ_Δ               [bp]

  δ は手数料と**逆選択コスト β σ_Δ** を両方賄えなければならない。
  σ_Δ が σ_day/√n に比例するので、**逆選択コストは √σ ではなく σ に比例**する。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
res = json.loads((D / "sde_results.json").read_text(encoding="utf-8"))
c = res["calib"]

N = 86400
v, S, f_bp = c["v_fill"], c["S0"], c["fee_bp"]
d_bp, beta = c["delta_bp"], c["beta"]
sig = c["sigma_day_med"]
lam = c["lam_day_med"]
sig_d_bp = sig / np.sqrt(N) * 1e4                    # 1 刻みの σ [bp]

adv_bp = beta * sig_d_bp                             # 逆選択コスト [bp/約定]
net_bp = d_bp - f_bp - adv_bp
be_bp = f_bp + adv_bp

print("=== 解析解(在庫上限を無視した極限)===")
print(f"  1 刻み(1 秒)の σ                    {sig_d_bp:.4f} bp")
print(f"  スプレッド獲得        δ             +{d_bp:.4f} bp")
print(f"  手数料                f             −{f_bp:.4f} bp")
print(f"  逆選択コスト          β σ_Δ         −{adv_bp:.4f} bp")
print(f"  ------------------------------------------------")
print(f"  1 約定あたり純益                    {net_bp:+.4f} bp"
      f"  = {net_bp/1e4*v*S*1e3:+.4f} mUSD")
print(f"  → 1 日({lam:,.0f} 約定)             {net_bp/1e4*v*S*lam:+.3f} USD/日")
print()
print(f"  ★損益分岐スプレッド δ* = f + βσ_Δ = {be_bp:.4f} bp")
print(f"    実際の獲得 δ = {d_bp:.4f} bp → 余裕 {d_bp-be_bp:+.4f} bp "
      f"({(d_bp-be_bp)/d_bp*100:.1f}% ぶん)")
print(f"    δ が {(1-be_bp/d_bp)*100:.1f}% 減るか、σ が {d_bp/be_bp if be_bp else 0:.2f}… "
      f"厳密には σ が {(d_bp-f_bp)/adv_bp:.2f} 倍になると損益 0")

sim = res["base"]
print("\n=== シミュレーションとの突き合わせ ===")
print(f"{'':22}{'解析解':>12}{'SDE 1,000 本':>16}{'実測 30 日':>14}")
g_an = d_bp / 1e4 * v * S * lam
i_an = -beta * sig_d_bp / 1e4 * v * S * lam
p_an = net_bp / 1e4 * v * S * lam
print(f"  {'スプレッド獲得':<18}{g_an:>12.3f}{sim['gross']['mean']:>16.3f}{7.933:>14.3f}")
print(f"  {'在庫コスト':<20}{i_an:>12.3f}{sim['inv_cost']['mean']:>16.3f}{-3.751:>14.3f}")
print(f"  {'E[PnL]':<22}{p_an:>12.3f}{sim['pnl']['mean']:>16.3f}{3.204:>14.3f}")
print(f"\n  解析解と SDE の差は在庫上限(|q| ≤ {c['inv_limit']:.3f})の効果。")
print(f"  在庫が上限に貼り付くと約定が片側で止まるので、両方とも縮む。")

print("\n=== 感度: どれだけ悪化すると損益 0 になるか ===")
for nm, cur, be in [("スプレッド獲得 δ [bp]", d_bp, be_bp),
                    ("ボラティリティ σ [%/日]", sig * 100,
                     sig * 100 * (d_bp - f_bp) / adv_bp),
                    ("手数料 f [bp]", f_bp, d_bp - adv_bp)]:
    print(f"  {nm:<26} 現在 {cur:>8.4f} → 分岐点 {be:>8.4f}"
          f"  ({(be/cur-1)*100:+.1f}%)")
