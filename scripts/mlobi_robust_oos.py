"""ロバスト最適化を実際に走らせ、標本外で w2/w3/w4 と比較する。

【なぜ必要か】
  mlobi_robust.py は「ロバスト化が必要になる条件(共線性・不安定・尖り・体制依存)が
  揃っていない」ことを示した。しかし **ロバスト手法を実際に走らせて負けることは
  示していない**。診断だけで「不要」と結論するのは、対立仮説を実行せずに棄却するのと同じ。

【比較する手法】すべて拡張窓のウォークフォワード、重み w も傾き a,b も学習期間だけで決める
  基準  w2 単変量β / w3 重回帰OLS / w4 Ridge(λ=1.0)
  R1    bagging      … 学習日をブロック復元抽出して w3 を推定、200 回の平均
  R2    minimax λ    … 学習期間を 4 分割し、最悪部分期間の R² を最大化する λ を選ぶ
  R3    形状縮小     … w3 を指数減衰の形へ縮める  w = (1−γ)·w3 + γ·w5、γ は学習期間で選択

【判定】
  真の標本外 R²(1 − Σ(y−ŷ)²/Σ(y−ȳ_train)²)の日次中央値と、
  w4 を上回った日数の符号検定。**w4 を有意に上回る手法があれば「必要」**と結論する。
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); OUT = D / "obi_levels"
NL, BURN, MIN_TRAIN = 10, 7, 20
RNG = np.random.default_rng(20260816)
NBAG = 200


def load():
    days = []
    for f in sorted(glob.glob(str(OUT / "*.npz")))[BURN:]:
        z = np.load(f)
        Ic = z["I"].astype(np.float64)
        y = z["y"].astype(np.float64)
        Icc = Ic - Ic.mean(0)
        days.append({"dt": Path(f).stem, "I": Ic, "y": y,
                     "XtX": Icc.T @ Icc, "Xty": Icc.T @ (y - y.mean()),
                     "ud": (Icc * Icc).sum(0)})
    return days


def agg(sel):
    XtX = sum(d["XtX"] for d in sel); Xty = sum(d["Xty"] for d in sel)
    ud = sum(d["ud"] for d in sel)
    return XtX, Xty, ud


def w_ridge(XtX, Xty, lam):
    sc = np.trace(XtX) / NL
    return np.linalg.solve(XtX + lam * sc * np.eye(NL), Xty)


def fit_ab(sel, w):
    sx = sy = sxx = sxy = 0.0; N = 0
    for d in sel:
        x = d["I"] @ w; y = d["y"]
        sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); sxy += (x * y).sum(); N += len(y)
    den = N * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0, 0.0
    b = (N * sxy - sx * sy) / den
    return (sy - b * sx) / N, b


def r2(day, w, ab, ybar):
    a, b = ab
    p = a + b * (day["I"] @ w)
    y = day["y"]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2))


def main() -> None:
    days = load(); n = len(days)
    names = ["w2 単変量β", "w3 重回帰OLS", "w4 Ridge",
             "R1 bagging", "R2 minimax λ", "R3 形状縮小"]
    res = {k: [] for k in names}
    lam_sel, gam_sel = [], []
    for i in range(MIN_TRAIN, n):
        tr, ev = days[:i], days[i]
        ybar = float(np.mean([t["y"].mean() for t in tr]))
        XtX, Xty, ud = agg(tr)
        W = {}
        W["w2 単変量β"] = np.where(ud > 0, np.diag(XtX) * 0 + Xty / np.maximum(ud, 1e-12), 0.0)
        W["w3 重回帰OLS"] = np.linalg.solve(XtX, Xty)
        W["w4 Ridge"] = w_ridge(XtX, Xty, 1.0)
        # R1 bagging: 学習日をブロック(長さ2)復元抽出
        acc = np.zeros(NL)
        for _ in range(NBAG):
            st = RNG.integers(0, max(len(tr) - 1, 1), size=int(np.ceil(len(tr) / 2)))
            idx = np.clip(np.concatenate([st, st + 1]), 0, len(tr) - 1)
            A, B, _u = agg([tr[j] for j in idx])
            try:
                acc += np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                pass
        W["R1 bagging"] = acc / NBAG
        # R2 minimax λ: 学習期間を 4 分割し最悪の R² を最大化
        q = max(len(tr) // 4, 1)
        folds = [tr[k * q:(k + 1) * q] for k in range(4) if tr[k * q:(k + 1) * q]]
        best, bl = None, 1.0
        for lam in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
            worst = 1e9
            for k, fo in enumerate(folds):
                rest = [d for j, d in enumerate(tr) if not (k * q <= j < (k + 1) * q)]
                if not rest:
                    continue
                A, B, _u = agg(rest)
                ww = w_ridge(A, B, lam)
                ab = fit_ab(rest, ww)
                yb = float(np.mean([t["y"].mean() for t in rest]))
                worst = min(worst, float(np.median([r2(d, ww, ab, yb) for d in fo])))
            if best is None or worst > best:
                best, bl = worst, lam
        lam_sel.append(bl)
        W["R2 minimax λ"] = w_ridge(XtX, Xty, bl)
        # R3 形状縮小: w3 を指数減衰の形へ縮める。γ は学習期間の後半で選ぶ
        b_ = np.abs(W["w2 単変量β"]); ok = b_ > 0
        lam_e = float(np.clip(-np.polyfit(np.arange(NL)[ok], np.log(b_[ok]), 1)[0], 0, 5)) \
            if ok.sum() >= 3 else 1.0
        shape = np.sign(W["w2 単変量β"][0] or 1.0) * np.exp(-lam_e * np.arange(NL))
        shape = shape / np.abs(shape).sum() * np.abs(W["w3 重回帰OLS"]).sum()
        h = max(len(tr) // 2, 1)
        va, bg, bs = tr[h:], None, 0.0
        for g in (0.0, 0.25, 0.5, 0.75, 1.0):
            ww = (1 - g) * W["w3 重回帰OLS"] + g * shape
            ab = fit_ab(tr[:h], ww)
            yb = float(np.mean([t["y"].mean() for t in tr[:h]]))
            v = float(np.median([r2(d, ww, ab, yb) for d in va]))
            if bg is None or v > bg:
                bg, bs = v, g
        gam_sel.append(bs)
        W["R3 形状縮小"] = (1 - bs) * W["w3 重回帰OLS"] + bs * shape
        for k in names:
            res[k].append(r2(ev, W[k], fit_ab(tr, W[k]), ybar))

    print(f"=== 標本外 R²(評価 {len(res[names[0]])} 日)===")
    print(f"  minimax の λ 中央値 = {np.median(lam_sel):.2f} / "
          f"形状縮小の γ 中央値 = {np.median(gam_sel):.2f}\n")
    base = np.array(res["w4 Ridge"])
    print(f"{'手法':<16}{'OOS R²':>11}{'w4=1 の比':>11}{'w4 を上回った日':>16}{'符号検定 p':>12}")
    out = {}
    for k in names:
        a = np.array(res[k])
        rel = np.median(np.where(np.abs(base) > 1e-12, a / base, np.nan))
        d_ = a - base
        pos = int((d_ > 0).sum())
        p = float(stats.binomtest(pos, len(d_), .5, alternative="greater").pvalue)
        mk = " ★" if p < 0.05 else ""
        print(f"{k:<16}{np.median(a):>11.5f}{rel:>11.3f}{pos:>11}/{len(d_):<4}{p:>12.4f}{mk}")
        out[k] = {"oos_r2": float(np.median(a)), "rel": float(rel),
                  "pos": pos, "n": len(d_), "p": p}
    win = [k for k in names if k.startswith("R") and out[k]["p"] < 0.05]
    print("\n" + "=" * 62)
    print(f"判定: ロバスト手法が w4 Ridge を有意に上回るか → "
          f"{'★あり: ' + ', '.join(win) if win else '**なし**'}")
    print("=" * 62)
    out["lambda_median"] = float(np.median(lam_sel))
    out["gamma_median"] = float(np.median(gam_sel))
    (D / "mlobi_robust_oos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
