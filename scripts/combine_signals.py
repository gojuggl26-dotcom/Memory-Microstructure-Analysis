"""短期価格予測(book_slope)と前方約定フローを組み合わせて E[PnL] を改善する。

【分解の枠組み】
    E[PnL] = E[Revenue] − E[Cost]
           = E[粗利] − ( E[逆選択] + 手数料 + 手仕舞いコスト )

  規則は 2 経路で効く。
    ・悪い局面を避ける → 残る約定の**粗利が上がる**
    ・不利な約定を避ける → **逆選択が減る**

【組み合わせる 2 つのシグナル】
  A. 前方約定フロー(ns 精度)  exec/10ms/abs5 · δ=100ms
       「前で 5 単位が約定し始めたら引く」= 掃きの検知
  B. 短期価格予測(1 秒グリッド) align = s × (−(S^Bid − S^Ask))
       slope_regression_report.md より予測リターンは −slope_diff に比例する
       B1 参入ゲート … align_open が閾値未満なら**置かない**
       B2 退出ゲート … **保有中に一度でも**閾値を下回ったら引く(align_min で判定)

  ★当初 B2 を「約定 δ 前の値」で判定していたが、これは実装不可能な後知恵だった。
    途中で不利になっても直前に回復していれば約定を残してしまう。
    実際の規則は「下回った瞬間に引く。その後回復しても戻れない」なので、
    保有中の最小値 align_min で判定する。
    実測では align_min の中央値 −0.996 に対し align_fill は +0.081 で、
    **大半の注文は保有中に一度は不利へ転じている**。差は大きい。

【評価する方策】
    0. 何もしない(全部の約定を受ける)
    A   前方フローのみ
    B1  参入ゲートのみ
    B2  退出ゲートのみ
    A+B1 / A+B2 / A+B1+B2

【検定】
  ・推論の単位は日。中央値・符号一貫性・ブロックブートストラップ
  ・閾値は**前日の分布**から取る(当日の分布を使うと結果を見てから線を引くことになる)
  ・全方策を報告する。良いものだけ拾わない
  ・★「改善が有意」と「改善後の水準が有意に正」は別物なので両方出す
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "pull_rule"
FEE_BP = 0.088
SIG = ["cancel", "exec", "total"]
W = [10, 50, 100, 500]
THR = ["frac0.25", "frac0.5", "frac0.75", "abs5", "abs20", "abs50", "abs150", "abs400"]
CELL = (SIG.index("exec"), W.index(10), THR.index("abs5"))
DELTA_NS = 100_000_000
BURN_IN = 7
Q_MIN = 400.0
QUANTS = [0.0, 0.2, 0.4, 0.6]          # 前日 align 分布の下位を切る割合
RNG = np.random.default_rng(20260816)
B = 5000


def boot(x, blk=2):
    n = len(x); nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def sgn(x):
    a = np.asarray(x)
    pos = int((a > 0).sum())
    return pos, float(stats.binomtest(pos, len(a), 0.5, alternative="greater").pvalue)


def load():
    keep = [f for f in sorted(OUT.glob("*.npz"))
            if not json.loads((OUT / f"{f.stem}.json").read_text(encoding="utf-8")).get("suspect")]
    days = []
    for f in keep[BURN_IN:]:
        z = np.load(f)
        m = z["good"] & (z["qo"] > Q_MIN) & (np.abs(z["dist_bp"]) < 1e-9)
        if m.sum() < 100:
            continue
        mg = z["margin"][m][:, CELL[0], CELL[1], CELL[2]].astype(np.float64)
        days.append({"dt": f.stem,
                     "gross": z["gross"][m].astype(np.float64),
                     "adv": z["adv"][m].astype(np.float64),
                     "pull_A": np.isfinite(mg) & (mg > DELTA_NS),
                     "a_open": z["align_open"][m].astype(np.float64),
                     "a_fill": z["align_fill"][m].astype(np.float64),
                     "a_min": z["align_min"][m].astype(np.float64)})
    return days


def main() -> None:
    days = load()
    n = len(days)
    print(f"層: 最良気配 × 前に {Q_MIN:.0f} 単位超 / {n} 日 / 手数料 {FEE_BP} 控除済み\n")

    def evaluate(keep_fn, label):
        """keep_fn(d, thr) -> bool 配列(残す約定)。閾値は前日から取る。"""
        rows = []
        prev = None
        for d in days:
            if prev is None:
                prev = d
                continue
            k = keep_fn(d, prev)
            if k.sum() < 10:
                prev = d
                continue
            net = d["gross"] + d["adv"] - FEE_BP
            base = float(net.sum())
            rows.append({"n": len(net), "kept": int(k.sum()),
                         "base_tot": base, "tot": float(net[k].sum()),
                         "base_per": float(net.mean()), "per": float(net[k].mean()),
                         "gross": float(d["gross"][k].mean()),
                         "adv": float(d["adv"][k].mean())})
            prev = d
        if len(rows) < 8:
            return None
        tot = np.array([r["tot"] for r in rows])
        bt = np.array([r["base_tot"] for r in rows])
        per = np.array([r["per"] for r in rows])
        gain = tot - bt
        gp, gpv = sgn(gain)
        lp, lpv = sgn(per)
        lo, hi = boot(gain)
        return {"label": label, "n_days": len(rows),
                "keep_rate": float(np.median([r["kept"] / r["n"] for r in rows])),
                "gross": float(np.median([r["gross"] for r in rows])),
                "adv": float(np.median([r["adv"] for r in rows])),
                "per_fill": float(np.median(per)), "per_pos": lp, "per_p": lpv,
                "total": float(np.median(tot)),
                "gain": float(np.median(gain)), "gain_pos": gp, "gain_p": gpv,
                "gain_ci": [lo, hi]}

    res = []
    res.append(evaluate(lambda d, p: np.ones(len(d["gross"]), bool), "0  何もしない"))
    res.append(evaluate(lambda d, p: ~d["pull_A"], "A  前方約定フロー"))
    for q in QUANTS[1:]:
        t = q
        res.append(evaluate(
            lambda d, p, q=q: d["a_open"] >= np.nanquantile(p["a_open"], q),
            f"B1 参入ゲート q={q}"))
        res.append(evaluate(
            lambda d, p, q=q: d["a_min"] >= np.nanquantile(p["a_min"], q),
            f"B2 退出ゲート q={q}"))
        res.append(evaluate(
            lambda d, p, q=q: (~d["pull_A"]) & (d["a_open"] >= np.nanquantile(p["a_open"], q)),
            f"A+B1        q={q}"))
        res.append(evaluate(
            lambda d, p, q=q: (~d["pull_A"]) & (d["a_min"] >= np.nanquantile(p["a_min"], q)),
            f"A+B2        q={q}"))
        res.append(evaluate(
            lambda d, p, q=q: d["a_fill"] >= np.nanquantile(p["a_fill"], q),
            f"[参考]後知恵版 q={q}"))
    res = [r for r in res if r]

    print(f"{'方策':<20}{'残す率':>7}{'粗利':>8}{'逆選択':>8}"
          f"{'E[PnL]/約定':>12}{'正の日':>8}{'p':>8}{'合計bp/日':>10}"
          f"{'改善':>9}{'改善日':>8}{'p':>8}")
    print("-" * 118)
    for r in res:
        m1 = "★" if r["gain_p"] < 0.05 else " "
        m2 = "●" if r["per_p"] < 0.05 else " "
        print(f"{r['label']:<20}{r['keep_rate']*100:>6.0f}%{r['gross']:>8.4f}{r['adv']:>8.4f}"
              f"{r['per_fill']:>12.4f}{r['per_pos']:>4}/{r['n_days']:<3}{r['per_p']:>8.4f}"
              f"{r['total']:>10.1f}{r['gain']:>9.1f}{r['gain_pos']:>4}/{r['n_days']:<3}"
              f"{r['gain_p']:>8.4f} {m1}{m2}")
    print("\n★ = 何もしないより有意に改善 / ● = E[PnL] の水準が有意に正")
    (D / "combine_signals.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
