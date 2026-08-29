"""MLOBI の重み(w2/w3/w4)にロバスト最適化が必要かを診断し、必要なら実行する。

【ロバスト最適化とは】
  推定した重みが「たまたまその標本で最適だっただけ」になるのを防ぐ手当て。
  縮小(Ridge)・ブートストラップ平均(bagging)・最悪期最適化(minimax)などがある。
  **必要でないのに入れると、無駄に偏りを持ち込むだけ**なので、要否を先に判定する。

【★判定基準(結果を見る前に固定する)】
  次の 4 つのうち **1 つでも該当すれば「必要」** と判定する。

    (a) 多重共線性  … 相関行列の条件数 κ > 30
                      (κ は固有値の最大/最小の比。30 超は中程度以上の共線性の目安)
    (b) 推定不安定  … 分割標本(奇数日 / 偶数日)で推定した重みベクトルの相関 < 0.8
    (c) 最適点が尖る … 重みを 20% 摂動したとき標本外 R² が 10% 以上劣化
    (d) 体制依存    … 立ち上げ期と成熟期の重みの余弦類似度 < 0.8

  (a)(b) は推定の安定性、(c) は目的関数の形状、(d) は転移可能性を見る。
  どれも該当しなければ「不要」と結論し、**やらないことを根拠つきで報告する**。

【必要と判定された場合に実行する手当て】
  R1 bagging      … ブートストラップ標本で重みを推定して平均
  R2 minimax      … 分割した部分期間の最悪 R² を最大化する縮小係数を選ぶ
  R3 事前形状縮小 … 指数減衰の形へ縮める(w5 の形を事前分布とみなす)
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "obi_levels"
NL = 10
BURN = 7
MIN_TRAIN = 20
RIDGE = 1.0
RNG = np.random.default_rng(20260816)

# ---- 判定基準(事前に固定)----
KAPPA_MAX = 30.0        # (a) 条件数
SPLIT_CORR_MIN = 0.80   # (b) 分割標本間の重み相関
PERTURB_FRAC = 0.20     # (c) 摂動の大きさ
DEGRADE_MAX = 0.10      # (c) 許容する劣化率
COS_MIN = 0.80          # (d) 体制間の余弦類似度


def load():
    days = []
    for f in sorted(glob.glob(str(OUT / "*.npz")))[BURN:]:
        z = np.load(f)
        days.append({"dt": Path(f).stem, "I": z["I"].astype(np.float64),
                     "y": z["y"].astype(np.float64)})
    return days


def suff(days):
    """中心化した十分統計量。"""
    XtX = np.zeros((NL, NL)); Xty = np.zeros(NL)
    un = np.zeros(NL); ud = np.zeros(NL)
    for d in days:
        Ic = d["I"] - d["I"].mean(0)
        yc = d["y"] - d["y"].mean()
        XtX += Ic.T @ Ic; Xty += Ic.T @ yc
        un += Ic.T @ yc; ud += (Ic * Ic).sum(0)
    return XtX, Xty, un, ud


def weights(XtX, Xty, un, ud):
    w = {}
    w["w2"] = np.where(ud > 0, un / np.maximum(ud, 1e-12), 0.0)
    try:
        w["w3"] = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        w["w3"] = w["w2"].copy()
    sc = np.trace(XtX) / NL
    w["w4"] = np.linalg.solve(XtX + RIDGE * sc * np.eye(NL), Xty)
    return w


def norm(w):
    s = np.abs(w).sum()
    return w / s if s > 0 else w


def fit_ab(days, w):
    sx = sy = sxx = sxy = 0.0; N = 0
    for d in days:
        x = d["I"] @ w; y = d["y"]
        sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); sxy += (x * y).sum(); N += len(y)
    den = N * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0, 0.0
    b = (N * sxy - sx * sy) / den
    return (sy - b * sx) / N, b


def oos(day, w, ab, ybar):
    a, b = ab
    p = a + b * (day["I"] @ w)
    y = day["y"]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2))


def main() -> None:
    days = load()
    n = len(days)
    print(f"=== {n} 日 / 判定基準は事前に固定(κ>30 / 相関<0.8 / 劣化>10% / 余弦<0.8)===\n")
    verdict = {}

    # ---- (a) 多重共線性 -------------------------------------------------
    XtX, Xty, un, ud = suff(days)
    d_ = np.sqrt(np.diag(XtX))
    R = XtX / np.outer(d_, d_)
    ev = np.linalg.eigvalsh(R)
    kappa = float(ev.max() / max(ev.min(), 1e-12))
    verdict["a_kappa"] = {"value": kappa, "threshold": KAPPA_MAX, "need": bool(kappa > KAPPA_MAX)}
    print(f"(a) 条件数 κ = {kappa:.2f}  (閾値 {KAPPA_MAX})  "
          f"→ {'★必要' if kappa > KAPPA_MAX else '不要'}")
    print(f"    固有値 最大 {ev.max():.3f} / 最小 {ev.min():.3f} / "
          f"レベル間の最大相関 {np.abs(R - np.eye(NL)).max():.3f}")

    # ---- (b) 分割標本での推定安定性 -------------------------------------
    A, Bd = days[0::2], days[1::2]
    wA = weights(*suff(A)); wB = weights(*suff(Bd))
    print(f"\n(b) 分割標本(奇数日 / 偶数日)で推定した重みの相関  (閾値 {SPLIT_CORR_MIN})")
    verdict["b_split"] = {}
    for k in ("w2", "w3", "w4"):
        c = float(np.corrcoef(norm(wA[k]), norm(wB[k]))[0, 1])
        verdict["b_split"][k] = {"corr": c, "need": bool(c < SPLIT_CORR_MIN)}
        print(f"    {k}: 相関 {c:.4f}  → {'★必要' if c < SPLIT_CORR_MIN else '不要'}")

    # ---- (c) 最適点の尖り具合(摂動感度)--------------------------------
    print(f"\n(c) 重みを ±{PERTURB_FRAC:.0%} 摂動したときの標本外 R² の劣化  "
          f"(閾値 {DEGRADE_MAX:.0%})")
    verdict["c_perturb"] = {}
    for k in ("w2", "w3", "w4"):
        base, pert = [], []
        for i in range(MIN_TRAIN, n):
            tr, ev_d = days[:i], days[i]
            w = weights(*suff(tr))[k]
            ybar = float(np.mean([t["y"].mean() for t in tr]))
            ab = fit_ab(tr, w)
            base.append(oos(ev_d, w, ab, ybar))
            acc = []
            for _ in range(20):
                wp = w * (1 + PERTURB_FRAC * RNG.standard_normal(NL))
                acc.append(oos(ev_d, wp, fit_ab(tr, wp), ybar))
            pert.append(float(np.mean(acc)))
        b_, p_ = np.median(base), np.median(pert)
        deg = (b_ - p_) / abs(b_) if abs(b_) > 1e-12 else 0.0
        verdict["c_perturb"][k] = {"base": b_, "perturbed": p_, "degrade": float(deg),
                                   "need": bool(deg > DEGRADE_MAX)}
        print(f"    {k}: {b_:.5f} → {p_:.5f}  劣化 {deg:+.1%}  "
              f"→ {'★必要' if deg > DEGRADE_MAX else '不要'}")

    # ---- (d) 体制間の転移 -----------------------------------------------
    half = n // 2
    wL = weights(*suff(days[:half])); wM = weights(*suff(days[half:]))
    print(f"\n(d) 立ち上げ期 vs 成熟期の重みの余弦類似度  (閾値 {COS_MIN})")
    verdict["d_regime"] = {}
    for k in ("w2", "w3", "w4"):
        u, v = norm(wL[k]), norm(wM[k])
        cs = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        verdict["d_regime"][k] = {"cos": cs, "need": bool(cs < COS_MIN)}
        print(f"    {k}: 余弦 {cs:.4f}  → {'★必要' if cs < COS_MIN else '不要'}")

    need = (verdict["a_kappa"]["need"]
            or any(v["need"] for v in verdict["b_split"].values())
            or any(v["need"] for v in verdict["c_perturb"].values())
            or any(v["need"] for v in verdict["d_regime"].values()))
    verdict["need_robust"] = bool(need)
    print("\n" + "=" * 60)
    print(f"判定: ロバスト最適化は **{'必要' if need else '不要'}**")
    print("=" * 60)
    if not need:
        print("  4 基準すべてが閾値内。重みは安定しており、目的関数も平坦である。")
        print("  ロバスト化は偏りを持ち込むだけで利得が無いと判断する。")
        print("  ※ w4 Ridge 自体が既に縮小を含んでおり、それで十分ということでもある。")
    (D / "mlobi_robust.json").write_text(json.dumps(verdict, indent=1, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
