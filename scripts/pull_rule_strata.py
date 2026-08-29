"""段階①の層別評価 — 行列の深さ × 売買の別 × 置き場所。

【なぜ層別するか】
  前版(pull_rule_eval.py)は**最良気配・全深さ込み**で評価し、
  「規則は 1 約定あたりでは効くが、総額で『そもそも出さない』に勝てない」
  という結論だった。しかし母集団の平均が −0.206 bp と赤字だったため、
  **母集団の選び方そのものが結論を決めていた**可能性がある。

  queue_clearing_report.md §4 の 2 元表では
  **深い行列(q_ahead > 100)はどの消え方でも黒字**(+0.048 〜 +0.603 bp)だった。
  したがって深い行列を母集団にすれば「出さない」を超えられるかもしれない。

【層】
  side      … Bid / Ask(板の非対称性は既知。cancel_report.md で ask−bid の差が有意)
  q_ahead   … 行列の深さ 4 区分
  placement … 最良気配 / 内側(気配改善)/ 外側

【検定手法】pull_rule_eval.py と同一(手法 1〜9)。層ごとに同じ手順を回す。

【★主判定の訂正(2026-08-16)】
  当初「残り合計が 0(そもそも出さない)を上回るか」を主判定にしていたが、
  **ベースラインが黒字の層ではこれがほぼ自動的に成立してしまう**。
  実際 92 日に拡張したところ、最良気配|100<q<=400 は
    出し続ける +93.3 / 引く +75.6
  で「0 を上回る」は通るのに**規則は損をしている**。

  したがって比較すべき対照は 3 つあり、すべて報告する。
    (a) ベースライン vs 0    … そもそも出す価値があるか
    (b) 規則 vs 0            … 引いて出す価値があるか
    (c) ★規則 vs ベースライン … **引くこと自体に価値があるか**(これが規則の検定)
  セルの選択は (c) で行う。(b) で選ぶと「何もしない方策」が勝ってしまう。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "pull_rule"
# 破損日は日付をハードコードせず、日次 json の suspect フラグで機械的に弾く
#   (2026-08-10 は ts_close の null で build_queue_qi が壊れ q_ahead_open が 1,135 倍)
BURN_IN_DAYS = 7                    # 立ち上げ期は関係が 5〜7 倍動くため除外
SIGNALS = ["cancel", "exec", "total"]
WINDOWS_MS = [10, 50, 100, 500]
# 割合(q_ahead 比)3 点 + 絶対量 5 点。pull_rule.py の保存順と一致させる
THRESHOLDS = ["frac0.25", "frac0.5", "frac0.75", "abs5", "abs20", "abs50", "abs150", "abs400"]
N_FRAC = 3
DELTA_MS = 100                      # 実装可能な反応時間として固定(§3-3 で 200ms まで頑健)
QB = [(-0.5, 20, "q<=20"), (20, 100, "20<q<=100"),
      (100, 400, "100<q<=400"), (400, np.inf, "q>400")]
PB = [(-99, -1e-9, "内側"), (-1e-9, 1e-9, "最良気配"), (1e-9, 99, "外側")]
# ★メイカー手数料(2026-08-16 訂正)
#   当初 net = gross + adv で手数料を控除していなかった。
#   net は 1 約定あたりの bp なので、定数 0.088 を引くのは
#   日次処理をやり直すのと数学的に同一である(再処理は不要)。
#   これを入れると「出す価値」は回数の多い層ほど削られ、
#   「引く価値」は引いた分の手数料を払わずに済むぶん増える。
FEE_BP = 0.088
MIN_PULL = 0.05                     # 発火率の下限。これ未満は規則とみなさない
RNG = np.random.default_rng(20260815)
B = 5000


def block_boot(x: np.ndarray, blk: int = 2):
    n = len(x)
    nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def load(regime: str = "all"):
    """regime: all / launch(立ち上げ)/ mature(成熟期)"""
    files = sorted(OUT.glob("*.npz"))
    keep, dropped = [], []
    for f in files:
        j = OUT / f"{f.stem}.json"
        if j.exists() and json.loads(j.read_text(encoding="utf-8")).get("suspect"):
            dropped.append(f.stem)          # q_ahead_open が異常な日
            continue
        keep.append(f)
    keep = keep[BURN_IN_DAYS:]              # 立ち上げ直後を除外
    if regime == "mature":
        keep = keep[len(keep) // 2:]
    elif regime == "launch":
        keep = keep[:len(keep) // 2]
    days = []
    for f in keep:
        z = np.load(f)
        days.append({"dt": f.stem, "margin": z["margin"].astype(np.float64),
                     "net": z["net"].astype(np.float64) - FEE_BP, "qo": z["qo"].astype(np.float64),
                     "is_bid": z["is_bid"], "dist": z["dist_bp"].astype(np.float64),
                     "good": z["good"]})
    return days, dropped


def evaluate(days, mask_fn, label: str) -> dict | None:
    """1 層について 60 セル(3 シグナル × 4 窓 × 5 閾値、δ=100ms)を回す。"""
    sub = []
    for d in days:
        m = d["good"] & mask_fn(d)
        if m.sum() < 100:
            continue
        sub.append({"net": d["net"][m], "margin": d["margin"][m],
                    "var": float(d["net"][m].var(ddof=1)) if m.sum() > 1 else 0.0})
    if len(sub) < 8:
        return None
    n_fills = int(sum(len(x["net"]) for x in sub))
    base_mean = np.array([x["net"].mean() for x in sub])
    base_tot = np.array([x["net"].sum() for x in sub])
    best = None
    for si in range(len(SIGNALS)):
        for wi in range(len(WINDOWS_MS)):
            for ti in range(len(THRESHOLDS)):
                lifts, rem_tot, rates, zs, gain = [], [], [], [], []
                for x in sub:
                    mg = x["margin"][:, si, wi, ti]
                    p = np.isfinite(mg) & (mg > DELTA_MS * 1_000_000)
                    k, n = int(p.sum()), len(x["net"])
                    if k == 0 or k >= n:
                        continue
                    mu = x["net"].mean()
                    rule = x["net"][~p].mean()
                    lifts.append(rule - mu)
                    rem_tot.append(float(x["net"][~p].sum()))
                    gain.append(float(x["net"][~p].sum() - x["net"].sum()))  # (c)
                    rates.append(k / n)
                    v = k * x["var"] * (n - k) / max(n - 1, 1)
                    zs.append((rule - mu) / (np.sqrt(v) / (n - k)) if v > 0 else 0.0)
                if len(rem_tot) < 8:
                    continue
                # ★発火率がほぼ 0 のセルは「規則」ではなく、
                #   残り合計 = ベースラインそのものになる。主判定から除外する
                #   (これを入れないと「何もしない方策」が規則として通ってしまう)
                if float(np.median(rates)) < MIN_PULL:
                    continue
                rt = np.array(rem_tot)
                gn = np.array(gain)
                pos = int((rt > 0).sum())
                gpos = int((gn > 0).sum())
                # (b) 0 を上回るか
                sp = float(stats.binomtest(pos, len(rt), 0.5, alternative="greater").pvalue)
                # (c) ★主判定: ベースラインを上回るか(引くこと自体の価値)
                gp = float(stats.binomtest(gpos, len(gn), 0.5, alternative="greater").pvalue)
                cand = {"signal": SIGNALS[si], "window_ms": WINDOWS_MS[wi],
                        "threshold": THRESHOLDS[ti],
                        "is_abs": ti >= N_FRAC, "n_days": len(rt),
                        "remaining_total_median": float(np.median(rt)),
                        "pos_days_vs_zero": pos, "sign_p_vs_zero": sp,
                        "lift_median": float(np.median(lifts)),
                        "lift_pos_days": int((np.array(lifts) > 0).sum()),
                        "gain_vs_base_median": float(np.median(gn)),
                        "gain_pos_days": gpos, "gain_sign_p": gp,
                        "pull_rate": float(np.median(rates)),
                        "rand_beat_days": int(np.sum(np.array(zs) > 1.645))}
                # ★セルの選択は (c) で行う
                if best is None or cand["gain_vs_base_median"] > best["gain_vs_base_median"]:
                    best = cand
    if best is None:
        return None
    lo, hi = block_boot(base_tot)
    bp_ = int((base_tot > 0).sum())
    return {"label": label, "n_days": len(sub), "n_fills": n_fills,
            "base_mean_median": float(np.median(base_mean)),
            "base_total_median": float(np.median(base_tot)),
            "base_neg_days": int((base_tot < 0).sum()),
            "base_pos_days": bp_,
            # ★規則なしでそのまま黒字かどうか。これ自体が独立した問い
            "base_sign_p": float(stats.binomtest(bp_, len(base_tot), 0.5,
                                                 alternative="greater").pvalue),
            "base_ci": [lo, hi], "best": best}


def main() -> None:
    regime = sys.argv[1] if len(sys.argv) > 1 else "all"
    days, dropped = load(regime)
    if dropped:
        print(f"※ 品質検査で除外した日: {', '.join(dropped)}")
    res: dict = {"n_days": len(days), "regime": regime, "dropped_days": dropped,
                 "delta_ms": DELTA_MS,
                 "cells_per_stratum": len(SIGNALS) * len(WINDOWS_MS) * len(THRESHOLDS)}
    rows = []

    def add(fn, lab):
        r = evaluate(days, fn, lab)
        if r:
            rows.append(r)

    # ---- 行列の深さ × 売買の別(最良気配に固定)-------------------------
    for qlo, qhi, ql in QB:
        for sd, sl in ((None, "両側"), (True, "Bid"), (False, "Ask")):
            def fn(d, qlo=qlo, qhi=qhi, sd=sd):
                m = (d["qo"] > qlo) & (d["qo"] <= qhi) & (np.abs(d["dist"]) < 1e-9)
                return m if sd is None else (m & (d["is_bid"] == sd))
            add(fn, f"最良気配|{ql}|{sl}")
    # ---- 置き場所 × 深さ(両側)------------------------------------------
    for plo, phi, pl_ in PB:
        if pl_ == "最良気配":
            continue                      # 側別ループと重複するため
        for qlo, qhi, ql in QB:
            def fn(d, plo=plo, phi=phi, qlo=qlo, qhi=qhi):
                return ((d["dist"] > plo) & (d["dist"] <= phi)
                        & (d["qo"] > qlo) & (d["qo"] <= qhi))
            add(fn, f"{pl_}|{ql}|両側")
    res["strata"] = rows
    res["n_strata"] = len(rows)
    (D / f"pull_rule_strata_{regime}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                             encoding="utf-8")

    print(f"=== {res['n_days']} 日 / δ={DELTA_MS}ms / 層 {len(rows)} 個 × "
          f"{res['cells_per_stratum']} セル ===")
    print("主判定 = 各層の最良セルの『残り合計』が 0(出さない)を上回った日数の符号検定\n")
    hdr = (f"{'層':<24}{'約定数':>9}{'(a)出す':>9}{'黒字日':>8}{'p(a)':>8}"
           f"{'(c)引く効果':>12}{'改善日':>8}{'p(c)':>8}{'引率':>7}{'乱数超':>7}")
    print(hdr)
    print("-" * len(hdr.encode("shift_jis", "ignore")))
    for r in rows:
        b = r["best"]
        mark = " ★" if b["gain_sign_p"] < 0.05 else ""
        bm = " ●" if r["base_sign_p"] < 0.05 else ""
        print(f"{r['label']:<24}{r['n_fills']:>9,}{r['base_total_median']:>9.1f}"
              f"{r['base_pos_days']:>5}/{r['n_days']}{r['base_sign_p']:>8.4f}"
              f"{b['gain_vs_base_median']:>12.1f}"
              f"{b['gain_pos_days']:>5}/{b['n_days']}{b['gain_sign_p']:>8.4f}"
              f"{b['pull_rate']*100:>6.0f}%"
              f"{b['rand_beat_days']:>4}/{b['n_days']}{mark}{bm}")

    print("\n● = 規則なしでそのまま有意に黒字 / ★ = 引く規則が『出さない』を有意に上回る")

    # ★規則の要否より先に問うべきこと: そもそも黒字の層はあるか
    print("\n=== ●規則なしで黒字の層 ===")
    ok0 = [r for r in rows if r["base_sign_p"] < 0.05]
    if not ok0:
        print("  なし")
    for r in ok0:
        print(f"  {r['label']:<24} 出し続けるだけで {r['base_total_median']:+.1f} bp/日 "
              f"({r['base_pos_days']}/{r['n_days']} 日、p={r['base_sign_p']:.4f}、"
              f"CI[{r['base_ci'][0]:+.0f},{r['base_ci'][1]:+.0f}])")

    print("\n=== ★主判定を通った層(そもそも出さないより有意に良い)===")
    ok = [r for r in rows if r["best"]["gain_sign_p"] < 0.05]
    if not ok:
        print("  なし")
    for r in ok:
        b = r["best"]
        print(f"  {r['label']:<26} {b['signal']}/{b['window_ms']}ms/{b['threshold']} → "
              f"改善 {b['gain_vs_base_median']:+.1f} bp/日 "
              f"({b['gain_pos_days']}/{b['n_days']} 日、p={b['gain_sign_p']:.4f}、"
              f"引率 {b['pull_rate']*100:.0f}%)")
    print(f"\n  多重比較: {len(rows)} 層 × {res['cells_per_stratum']} セルを探索。"
          f"層数だけで Bonferroni なら閾値 {0.05/max(len(rows),1):.4f}")


if __name__ == "__main__":
    main()
