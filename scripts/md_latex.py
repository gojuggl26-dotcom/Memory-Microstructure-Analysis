"""インラインコードの数式を LaTeX ($...$) へ変換する。

方針: 「視覚理解が困難にならない範囲で」。
  変換する  … 純粋に数学的で、LaTeX にすると読みやすくなるもの
  変換しない … 式に日本語が混ざるもの(`E[損益 | 出す] × P(fill)` など)、
               散文の掛け算(`最良気配 × 前に 100〜400 単位` など)、
               列名の羅列(`Σ(vol_buy + vol_sell)` など。\\mathrm 漬けになって
               かえって読みにくい)

対応は下の MAP に**完全一致**で持つ。正規表現で推測すると
パス表記や散文まで巻き込むため、意図した式だけを列挙している。

使い方:
    uv run python scripts/md_latex.py --check
    uv run python scripts/md_latex.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("C:/Users/ii562/Downloads/Memory")

# `元の表記` -> $LaTeX$
MAP: dict[str, str] = {
    # --- 定義式 ---
    "r_{t+1} = log Price_{t+1} − log Price_t":
        r"r_{t+1} = \log P_{t+1} - \log P_{t}",
    "j_i = ceil((τ_i − t0)/Δ)":
        r"j_i = \lceil (\tau_i - t_0)/\Delta \rceil",
    "Σ_j r_j r_{j−k}":
        r"\sum_j r_j\, r_{j-k}",
    "E[PnL] = E[Revenue] − E[Cost]":
        r"E[\mathrm{PnL}] = E[\mathrm{Revenue}] - E[\mathrm{Cost}]",
    "adv = s×(mid(t+Δ) − mid(t))":
        r"\mathrm{adv} = s\,(m_{t+\Delta} - m_t)",
    "β_d = corr_d × σ_y,d / σ_x,d":
        r"\beta_d = \rho_d\,\sigma_{y,d}/\sigma_{x,d}",
    "m_bp = −OBI(1)/2 × spread":
        r"m_{\mathrm{bp}} = -\tfrac{1}{2}\,\mathrm{OBI}(1)\cdot s",
    "ρ(k) = corr( r_HL(t), r_BN(t + kΔ) )":
        r"\rho(k) = \mathrm{corr}\!\left(r^{\mathrm{HL}}_t,\; r^{\mathrm{BN}}_{t+k\Delta}\right)",
    "y = (log mid(T+1s) − log mid(T)) × 10⁴":
        r"y = \left(\log m_{T+1s} - \log m_T\right)\times 10^{4}",
    "y = α + Σ_{L=1..10} f_L(I_L) + ε":
        r"y = \alpha + \sum_{L=1}^{10} f_L(I_L) + \varepsilon",
    "y = α + β·x + ε":
        r"y = \alpha + \beta x + \varepsilon",
    "MP = mid + OBI(1)/2 · spread":
        r"\mathrm{MP} = m + \tfrac{1}{2}\,\mathrm{OBI}(1)\cdot s",
    "MP − mid = OBI(1)/2 · spread":
        r"\mathrm{MP} - m = \tfrac{1}{2}\,\mathrm{OBI}(1)\cdot s",
    "2·bid_sz/(bid_sz+ask_sz) − 1":
        r"\frac{2\,q_b}{q_b+q_a} - 1",
    "(Σ_{i≤L} q_b − Σ_{i≤L} q_a)/(Σ_{i≤L} q_b + Σ_{i≤L} q_a)":
        r"\frac{\sum_{i\le L} q_b - \sum_{i\le L} q_a}{\sum_{i\le L} q_b + \sum_{i\le L} q_a}",
    "Δmid_{t+h}/mid × 10⁴ = a + β · micro_dev_bp":
        r"\frac{\Delta m_{t+h}}{m}\times 10^{4} = a + \beta\,d_{\mathrm{bp}}",
    "mid + β̂ ·(MP − mid)":
        r"m + \hat{\beta}\,(\mathrm{MP} - m)",
    "mid + β̂·(MP − mid)":
        r"m + \hat{\beta}\,(\mathrm{MP} - m)",
    "mid + (I − 1/2)·spread":
        r"m + (I - \tfrac{1}{2})\,s",
    "(I − 1/2) × spread":
        r"(I - \tfrac{1}{2})\,s",
    "(I−1/2)×spread":
        r"(I - \tfrac{1}{2})\,s",
    "log mid(T+2Δ) − log mid(T+Δ)":
        r"\log m_{T+2\Delta} - \log m_{T+\Delta}",
    "log mid(T+2·step) − log mid(T+step)":
        r"\log m_{T+2\delta} - \log m_{T+\delta}",
    "ΔQuantity_bid_t − ΔQuantity_ask_t":
        r"\Delta Q^{b}_{t} - \Delta Q^{a}_{t}",
    "λ_cancel^bid / (λ_cancel^bid + λ_cancel^ask)":
        r"\frac{\lambda^{b}_{\mathrm{cxl}}}{\lambda^{b}_{\mathrm{cxl}}+\lambda^{a}_{\mathrm{cxl}}}",
    "λ_cancel^bid/(λ_cancel^bid+λ_cancel^ask)":
        r"\frac{\lambda^{b}_{\mathrm{cxl}}}{\lambda^{b}_{\mathrm{cxl}}+\lambda^{a}_{\mathrm{cxl}}}",
    "λ_new^buy − λ_new^sell":
        r"\lambda^{\mathrm{buy}}_{\mathrm{new}} - \lambda^{\mathrm{sell}}_{\mathrm{new}}",
    "λ_cancel^bid": r"\lambda^{b}_{\mathrm{cxl}}",
    "λ_cancel^ask": r"\lambda^{a}_{\mathrm{cxl}}",
    "E[mid_{t+∞}]": r"E[m_{t+\infty}]",
    "exp(−λ(L−1))": r"e^{-\lambda(L-1)}",
    "exp(2×0.2/τ)": r"e^{2\times 0.2/\tau}",
    "Σ|w| = 1": r"\textstyle\sum |w| = 1",
    "w ≥ 0": r"w \ge 0",
    "y ≠ 0": r"y \ne 0",
    "log|β_d|": r"\log|\beta_d|",
    "a_δ > 0": r"a_\delta > 0",
    "a_δ": r"a_\delta",
    "t − δ": r"t-\delta",
    "I_1 = ±0.9": r"I_1 = \pm 0.9",
    "I_1 ≈ 0": r"I_1 \approx 0",
    "I_1 × I_2": r"I_1 \times I_2",
    "n > (2/0.16)² ≈ 156": r"n > (2/0.16)^2 \approx 156",
    "(k − corr)² = 1.08": r"(k-\rho)^2 = 1.08",
    "ŷ → λ ŷ": r"\hat{y} \to \lambda \hat{y}",
    "ŷ → (σ_{m−1}/σ̄) ŷ": r"\hat{y} \to (\sigma_{m-1}/\bar{\sigma})\,\hat{y}",
    "mid ≈ 約定価格 ± ティック/2": None,          # 日本語混在 — 変換しない
    "[T, T+Δ)": r"[T,\; T+\Delta)",
    "[T−Δ, T)": r"[T-\Delta,\; T)",
    "(t0, t_fill−δ]": r"(t_0,\; t_{\mathrm{fill}}-\delta]",
}

SKIP_NOTE = """変換しなかったもの(意図的):
  日本語が式に混ざる  E[損益 | 出す] × P(fill) / Σ(n_bid + n_ask) = その日のキャンセル数
                      p(上乗せ ≤ 0) / mid ≈ 約定価格 ± ティック/2
  散文の掛け算        最良気配 × 前に 100〜400 単位 × 両側 など
  列名の羅列          Σ(vol_buy + vol_sell) / ΔQuantity など(\\mathrm 漬けで読みにくくなる)
  複数行の定義ブロック 式と日本語注記が横並びで、$$ に入れると注記まで数式になる"""

FENCE = re.compile(r"```.*?```", re.S)


def convert(text: str) -> tuple[str, int]:
    """フェンス外のインラインコードだけを見て、MAP に完全一致するものを $...$ にする。"""
    spans = [m.span() for m in FENCE.finditer(text)]

    def in_fence(i: int) -> bool:
        return any(a <= i < b for a, b in spans)

    n = 0
    out, last = [], 0
    for m in re.finditer(r"`([^`\n]+)`", text):
        if in_fence(m.start()):
            continue
        tex = MAP.get(m.group(1).strip())
        if not tex:
            continue
        out.append(text[last:m.start()])
        out.append(f"${tex}$")
        last = m.end()
        n += 1
    out.append(text[last:])
    return "".join(out), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    total = 0
    for f in sorted(ROOT.glob("*.md")):
        src = f.read_text(encoding="utf-8")
        dst, n = convert(src)
        if n:
            total += n
            print(f"  {f.name:<34} {n:>3} 箇所")
            if not a.check:
                f.write_text(dst, encoding="utf-8")
    print(f"\n{'検査' if a.check else '変換'}: 合計 {total} 箇所")
    print(SKIP_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
