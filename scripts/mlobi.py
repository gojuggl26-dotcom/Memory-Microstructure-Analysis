"""MLOBI — レベル別 I_L を重み付けして統合した多段不均衡指標。

【定義】2 通りを作る
  A. 不均衡加重  MLOBI_A = Σ_L w_L · I_L / Σ_L |w_L|
        レベル別の不均衡そのものを重み付ける。[-1, +1] に収まる
  B. 数量加重    MLOBI_B = Σ_L w_L(q_b^L − q_a^L) / Σ_L w_L(q_b^L + q_a^L)
        文献で標準的な形。数量の重みなので w_L ≥ 0 を課す

【重みの決め方】5 通りを比較する
  w1 均等              w_L = 1
  w2 単変量 β          w_L = β_L(各段を単独で回帰したときの係数)
  w3 重回帰 OLS        w = argmin ‖y − I·w‖²(相関を考慮した最適線形結合)
  w4 Ridge             w3 に縮小を入れる。10 段は互いに相関するため
  w5 指数減衰          w_L = exp(−λ(L−1))、λ は学習期間で当てはめる

【★ルックアヘッドの排除】
  重みは**評価日より前のデータだけ**で推定する(拡張窓のウォークフォワード)。
  全期間で当てはめた重みで全期間を評価すると、必ず良く見える。
  CLAUDE.md 厳禁事項 4「標準化・分位のパラメータを全標本から作らない」と同じ理屈。

【比較対象】
  OBI(1) 単独 / 累積 OBI(5) / 累積 OBI(10) / MLOBI 各種

【★評価指標 — 2 つを厳密に区別する(2026-08-16 訂正)】
  当初 R² を「x と y の当日の回帰」で計算していた。重みは標本外でも、
  **最後の傾き b と切片 a を評価日の y で当てはめていた**ため、
  これは標本外 R² ではなく「標本外の重みを使った当日の相関の 2 乗」だった。

  今版は 2 つを併記する。
    (i) 当日相関²  … b, a も当日で当てはめる。**情報量の指標**。常に ≥ 0
    (ii) 真の標本外 R² … w も b も a も**学習期間だけ**で決め、評価日に適用する。
                        1 − Σ(y−ŷ)²/Σ(y−ȳ_train)²。符号や尺度が転移しなければ**負になる**

  (i) は「その日どれだけ相関したか」、(ii) は「事前に決めた式がその日通用したか」。
  実務で使えるのは (ii) である。
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
MIN_TRAIN = 20          # 重み推定に最低これだけの日数を使う
RIDGE = 1.0
RNG = np.random.default_rng(20260816)
B = 5000


def boot(x, blk=2):
    n = len(x); nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def sgn(a):
    p = int((np.asarray(a) > 0).sum())
    return p, float(stats.binomtest(p, len(a), .5, alternative="greater").pvalue)


def load():
    days = []
    for f in sorted(glob.glob(str(OUT / "*.npz")))[BURN:]:
        z = np.load(f)
        days.append({"dt": Path(f).stem, "I": z["I"].astype(np.float64),
                     "C": z["C"].astype(np.float64),
                     "qb": z["qb"].astype(np.float64), "qa": z["qa"].astype(np.float64),
                     "y": z["y"].astype(np.float64)})
    return days


def fit_weights(train):
    """学習期間の日から 5 通りの重みを作る。"""
    XtX = np.zeros((NL, NL)); Xty = np.zeros(NL)
    uni_num = np.zeros(NL); uni_den = np.zeros(NL)
    for d in train:
        I, y = d["I"], d["y"] - d["y"].mean()
        Ic = I - I.mean(0)
        XtX += Ic.T @ Ic
        Xty += Ic.T @ y
        uni_num += Ic.T @ y
        uni_den += (Ic * Ic).sum(0)
    w = {}
    w["w1 均等"] = np.ones(NL)
    w["w2 単変量β"] = np.where(uni_den > 0, uni_num / np.maximum(uni_den, 1e-12), 0.0)
    try:
        w["w3 重回帰OLS"] = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        w["w3 重回帰OLS"] = w["w2 単変量β"].copy()
    sc = np.trace(XtX) / NL
    w["w4 Ridge"] = np.linalg.solve(XtX + RIDGE * sc * np.eye(NL), Xty)
    # 指数減衰: 単変量βの絶対値に log-linear を当てはめる
    b = np.abs(w["w2 単変量β"])
    ok = b > 0
    if ok.sum() >= 3:
        lam = -np.polyfit(np.arange(NL)[ok], np.log(b[ok]), 1)[0]
        lam = float(np.clip(lam, 0.0, 5.0))
    else:
        lam = 1.0
    s1 = np.sign(w["w2 単変量β"][0]) or 1.0
    w["w5 指数減衰"] = s1 * np.exp(-lam * np.arange(NL))
    return w, lam


def corr2_of(x, y, ybar):
    """(i) 当日相関² — b, a も当日で当てはめる。情報量の指標。"""
    v = x.var()
    if v <= 1e-18:
        return 0.0
    b = np.cov(x, y, bias=True)[0, 1] / v
    a = y.mean() - b * x.mean()
    p = a + b * x
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2))


def oos_r2(x, y, ab, ybar):
    """(ii) 真の標本外 R² — 傾き・切片は学習期間で決めたものを使う。"""
    a, b = ab
    p = a + b * x
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2))


def fit_ab(train, xf):
    """学習期間で傾きと切片を当てはめる。xf は日 → 指標系列 を返す関数。"""
    sx = sy = sxx = sxy = 0.0; N = 0
    for d in train:
        x, y = xf(d), d["y"]
        sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); sxy += (x * y).sum()
        N += len(y)
    den = N * sxx - sx * sx
    if abs(den) < 1e-12:
        return (0.0, 0.0)
    b = (N * sxy - sx * sy) / den
    return ((sy - b * sx) / N, b)


def main() -> None:
    days = load()
    n = len(days)
    print(f"=== {n} 日 / 1 秒グリッド / 重みは評価日より前のデータだけで推定 ===\n")

    cand = ["OBI(1) 単独", "累積 OBI(5)", "累積 OBI(10)",
            "w1 均等", "w2 単変量β", "w3 重回帰OLS", "w4 Ridge", "w5 指数減衰"]
    r2A = {c: [] for c in cand}
    oosA = {c: [] for c in cand}
    r2B = {c: [] for c in cand if c.startswith("w")}
    lams = []
    for k in range(MIN_TRAIN, n):
        train, d = days[:k], days[k]
        w, lam = fit_weights(train)
        lams.append(lam)
        ybar = float(np.mean([t["y"].mean() for t in train]))
        y = d["y"]
        for nm, col in (("OBI(1) 単独", 0), ("累積 OBI(5)", 4), ("累積 OBI(10)", 9)):
            xf = (lambda dd, c=col: dd["C"][:, c])
            r2A[nm].append(corr2_of(xf(d), y, ybar))
            oosA[nm].append(oos_r2(xf(d), y, fit_ab(train, xf), ybar))
        for nm, ww in w.items():
            sw = np.abs(ww).sum() or 1.0
            xf = (lambda dd, v=ww, s_=sw: dd["I"] @ v / s_)    # A: 不均衡加重
            r2A[nm].append(corr2_of(xf(d), y, ybar))
            oosA[nm].append(oos_r2(xf(d), y, fit_ab(train, xf), ybar))
            wp = np.abs(ww)                                    # B: 数量加重(w ≥ 0)
            num = d["qb"] @ wp - d["qa"] @ wp
            den = d["qb"] @ wp + d["qa"] @ wp
            xb = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
            r2B[nm].append(corr2_of(xb, y, ybar))

    base = np.array(r2A["OBI(1) 単独"])
    print(f"評価日数 {len(base)} 日(最初の {MIN_TRAIN} 日は重み推定に使用)")
    print(f"指数減衰の λ 中央値 = {np.median(lams):.3f}\n")
    print("【A】不均衡加重  MLOBI_A = Σ w_L·I_L / Σ|w_L|")
    print("   (i) 当日相関² = 情報量 / (ii) ★真の標本外 R² = 事前に決めた式が通用したか")
    print(f"{'指標':<16}{'(i)相関²':>11}{'比':>7}{'★(ii)OOS R²':>13}{'比':>8}"
          f"{'正の日':>9}{'上回った日':>12}{'p':>9}")
    res = {}
    obase = np.array(oosA["OBI(1) 単独"])
    for c in cand:
        a = np.array(r2A[c]); o = np.array(oosA[c])
        rel = np.median(np.where(np.abs(base) > 1e-12, a / base, np.nan))
        orel = np.median(np.where(np.abs(obase) > 1e-12, o / obase, np.nan))
        opos, _ = sgn(o)
        pos, p = sgn(o - obase)
        mk = " ★" if p < 0.05 else ""
        print(f"{c:<16}{np.median(a):>11.5f}{rel:>7.3f}{np.median(o):>13.5f}{orel:>8.3f}"
              f"{opos:>5}/{len(o):<3}{pos:>7}/{len(o):<4}{p:>9.4f}{mk}")
        res[f"A|{c}"] = {"corr2": float(np.median(a)), "corr2_rel": float(rel),
                         "oos_r2": float(np.median(o)), "oos_rel": float(orel),
                         "oos_pos": opos, "pos": pos, "n": len(o), "p": p}
    print("\n【B】数量加重  MLOBI_B = Σ w_L(q_b−q_a) / Σ w_L(q_b+q_a)   (w ≥ 0)")
    print(f"{'指標':<16}{'OOS R²':>11}{'★OBI(1)=1':>12}{'上回った日':>12}{'p':>9}")
    for c in r2B:
        a = np.array(r2B[c])
        rel = np.median(np.where(np.abs(base) > 1e-12, a / base, np.nan))
        pos, p = sgn(a - base)
        mk = " ★" if p < 0.05 else ""
        print(f"{c:<16}{np.median(a):>11.5f}{rel:>12.3f}{pos:>7}/{len(a):<4}{p:>9.4f}{mk}")
        res[f"B|{c}"] = {"r2": float(np.median(a)), "rel": float(rel),
                         "pos": pos, "n": len(a), "p": p}

    # 最終日までの全データで推定した重み(実装用に提示)
    w, lam = fit_weights(days)
    print("\n【参考】全期間で推定した重み(実装に使うならこれ。上の評価には未使用)")
    for nm in ("w2 単変量β", "w3 重回帰OLS", "w4 Ridge"):
        v = w[nm] / np.abs(w[nm]).sum()
        print(f"  {nm:<12} " + " ".join(f"{x:+.3f}" for x in v))
    print(f"  w5 指数減衰   λ = {lam:.3f}  → " +
          " ".join(f"{x:+.3f}" for x in (w['w5 指数減衰'] / np.abs(w['w5 指数減衰']).sum())))
    res["weights_full_sample"] = {k: v.tolist() for k, v in w.items()}
    res["lambda"] = lam
    (D / "mlobi.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    print("\n★ = OBI(1) 単独を有意に上回る")


if __name__ == "__main__":
    main()
