"""段階①の統計的評価 — 用いた手法をすべて明示する。

【評価する量】
  引かなかった場合の損益  base = mean(net) over 全約定
  引いた場合の損益        rule = mean(net) over 引かなかった約定
  改善                    lift = rule − base
  引いた割合              pull_rate

  引いた約定は「起きなかった」ことにする。母集団は実際に起きたメイカー約定なので、
  引いた分の損益がそのまま差し引かれる。**良い約定を逃す損も自動的に計上される。**

【用いた手法(すべて報告する)】
  (1) 推論の単位は「日」。プールした t 統計量は使わない。
      1 秒観測の実効標本は n/14 しかない(regression_report.md §9 T5)。
      27 日の中央値・符号一貫性を主指標にする
  (2) 符号検定(二項検定)。分布の仮定を置かない
  (3) ブロック・ブートストラップ(ブロック長 2 日、5,000 回)で 95% CI と p(lift≤0)。
      日次系列の自己相関を壊さない
  (4) ★ランダム対照(解析解): 同じ日に**同じ件数 k** を無作為に引いた場合の
      「残った約定の平均」の分布を、有限母集団からの非復元抽出の理論値で出す。
          E[S_k] = k·μ ,  Var[S_k] = k·σ²·(n−k)/(n−1)   (有限母集団修正つき)
          残りの平均 = (S − S_k)/(n−k) ,  E = μ
          z = (rule_mean − μ) / ( sqrt(Var[S_k]) / (n−k) )
      これが「選別が情報を持つか」の帰無仮説。
      単に約定数を減らせば平均が動くという可能性を排除する。
      モンテカルロではなく解析解なので抽出誤差がない
  (5) ★シグナル対照: 取消フロー / 約定フロー / 合計フロー の 3 種を同一手順で比較。
      機構が「取消は情報」なら取消だけが効くはず。
      「前で何か起きたら危ない」という一般論なら 3 種とも効くはず
  (6) 全格子を報告する(3 シグナル × 4 窓 × 5 閾値 × 3 反応時間 = 180 通り)。
      良い組合せだけを拾わない。多重比較の規模を明示する
  (7) ★標本外の方策選択: k 日目の (シグナル, 窓, 閾値) を 1..k−1 日だけで選び、
      k 日目で評価する。全期間を見てから селで選んだ値との差が選択バイアスそのもの
  (8) ★総額でも評価する。1 約定あたりの改善(lift)だけを見ると
      「引きまくって単価を上げた」方策が勝ってしまう。
      **合計損益 = Σ net(引かなかった約定)** を併記し、
      ランダム対照も総額で行う。fill_prob_backtest_report.md で
      「単価は 68% 改善するが総額は 8.5% 減る」と測ったのと同じ落とし穴を避ける
  (9) ★引いた率を揃えた比較。シグナル間で引いた率が違うと公平でない。
      各シグナルについて引いた率が近いセル同士を比べる
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "pull_rule"
EXCLUDE = {"2026-08-10"}          # queue_qi 破損日(queue_clearing_report.md §6-2)
SIGNALS = ["cancel", "exec", "total"]
WINDOWS_MS = [10, 50, 100, 500]
THRESHOLDS = [0.10, 0.25, 0.50, 0.75, 0.90]
DELTAS_MS = [0, 100, 200]
RNG = np.random.default_rng(20260815)
B = 5000
NRAND = 1000


def block_boot(x: np.ndarray, blk: int = 2):
    n = len(x)
    nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975)), float((m <= 0).mean())


def load():
    days = []
    for f in sorted(OUT.glob("*.npz")):
        if f.stem in EXCLUDE:
            continue
        z = np.load(f)
        g = z["good"] & z["at_best"]          # 最良気配に置いた注文のみ(前報告と揃える)
        if g.sum() < 200:
            continue
        net = z["net"][g].astype(np.float64)
        days.append({"dt": f.stem, "margin": z["margin"][g].astype(np.float64),
                     "net": net, "var": float(net.var(ddof=1))})
    return days


def main() -> None:
    days = load()
    nd = len(days)
    res: dict = {"n_days": nd, "days": [d["dt"] for d in days],
                 "n_fills": int(sum(len(d["net"]) for d in days)),
                 "grid_size": len(SIGNALS) * len(WINDOWS_MS) * len(THRESHOLDS) * len(DELTAS_MS)}
    base_daily = np.array([d["net"].mean() for d in days])
    res["baseline"] = {"median": float(np.median(base_daily)),
                       "mean": float(base_daily.mean()),
                       "neg_days": int((base_daily < 0).sum())}

    cells = {}
    for si, sn in enumerate(SIGNALS):
        for wi, w in enumerate(WINDOWS_MS):
            for ti, thr in enumerate(THRESHOLDS):
                for dl in DELTAS_MS:
                    lifts, rates, rnd_p = [], [], []
                    dtot, rnd_tot_p = [], []
                    for d in days:
                        mg = d["margin"][:, si, wi, ti]
                        pulled = np.isfinite(mg) & (mg > dl * 1_000_000)
                        k = int(pulled.sum())
                        n = len(d["net"])
                        if k == 0 or k >= n:
                            continue
                        base = d["net"].mean()
                        rule = d["net"][~pulled].mean()
                        lifts.append(rule - base)
                        rates.append(k / n)
                        # ---- ランダム対照(解析解・非復元抽出)----
                        tot = float(d["net"].sum())
                        var_sk = k * d["var"] * (n - k) / max(n - 1, 1)
                        sd_rem = np.sqrt(var_sk) / (n - k)
                        z = (rule - base) / sd_rem if sd_rem > 0 else 0.0
                        rnd_p.append(float(stats.norm.cdf(z)))
                        # ---- 総額(1 日の合計損益)。単価だけ見る罠を避ける ----
                        t_rule = float(d["net"][~pulled].sum())
                        dtot.append(t_rule - tot)
                        # 総額のランダム対照: 残り合計 = S − S_k、E = S(n−k)/n
                        e_rem = tot * (n - k) / n
                        zt = (t_rule - e_rem) / np.sqrt(var_sk) if var_sk > 0 else 0.0
                        rnd_tot_p.append(float(stats.norm.cdf(zt)))
                    if len(lifts) < 5:
                        continue
                    a = np.array(lifts)
                    lo, hi, ple = block_boot(a)
                    pos = int((a > 0).sum())
                    at = np.array(dtot)
                    tlo, thi, tple = block_boot(at)
                    cells[f"{sn}|{w}|{thr}|{dl}"] = {
                        "signal": sn, "window_ms": w, "threshold": thr, "delta_ms": dl,
                        "n_days": len(a), "lift_median": float(np.median(a)),
                        "lift_mean": float(a.mean()), "pos_days": pos,
                        "sign_p": float(stats.binomtest(pos, len(a), 0.5,
                                                        alternative="greater").pvalue),
                        "ci": [lo, hi], "p_le0": ple,
                        "pull_rate": float(np.median(rates)),
                        "rand_pct_median": float(np.median(rnd_p)),
                        "rand_beat_days": int(np.sum(np.array(rnd_p) > 0.95)),
                        "rand_z_median": float(stats.norm.ppf(np.clip(np.median(rnd_p), 1e-9, 1-1e-9))),
                        "total_gain_median": float(np.median(at)),
                        "total_pos_days": int((at > 0).sum()),
                        "total_ci": [tlo, thi], "total_p_le0": tple,
                        "total_sign_p": float(stats.binomtest(int((at > 0).sum()), len(at),
                                                              0.5, alternative="greater").pvalue),
                        "rand_total_beat_days": int(np.sum(np.array(rnd_tot_p) > 0.95))}
    res["cells"] = cells

    # ---- 標本外の方策選択 ------------------------------------------------
    keys = list(cells)
    per_day = {k: {} for k in keys}
    for k in keys:
        c = cells[k]
        si, wi, ti = (SIGNALS.index(c["signal"]), WINDOWS_MS.index(c["window_ms"]),
                      THRESHOLDS.index(c["threshold"]))
        for d in days:
            mg = d["margin"][:, si, wi, ti]
            p = np.isfinite(mg) & (mg > c["delta_ms"] * 1_000_000)
            if 0 < p.sum() < len(d["net"]):
                per_day[k][d["dt"]] = float(d["net"][~p].mean() - d["net"].mean())
    dts = [d["dt"] for d in days]
    oos, chosen = [], []
    for i in range(3, nd):
        past = {k: np.mean([v[t] for t in dts[:i] if t in v])
                for k, v in per_day.items() if sum(t in v for t in dts[:i]) >= 3}
        if not past:
            continue
        best = max(past, key=past.get)
        if dts[i] in per_day[best]:
            oos.append(per_day[best][dts[i]]); chosen.append(best)
    if oos:
        a = np.array(oos)
        lo, hi, ple = block_boot(a)
        ins = max(cells, key=lambda k: cells[k]["lift_median"])
        res["oos_selection"] = {
            "n_eval": len(a), "median": float(np.median(a)), "mean": float(a.mean()),
            "pos_days": int((a > 0).sum()), "ci": [lo, hi], "p_le0": ple,
            "sign_p": float(stats.binomtest(int((a > 0).sum()), len(a), 0.5,
                                            alternative="greater").pvalue),
            "in_sample_best": ins, "in_sample_best_lift": cells[ins]["lift_median"],
            "chosen": chosen}
    (D / "pull_rule_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")

    # ---- 表示 ----
    print(f"=== {nd} 日 / 最良気配の約定 {res['n_fills']:,} 件 / 格子 {res['grid_size']} 通り ===")
    b = res["baseline"]
    print(f"引かない場合の損益: 中央値 {b['median']:+.4f} bp / 平均 {b['mean']:+.4f} / "
          f"赤字の日 {b['neg_days']}/{nd}")

    print("\n【0】★総額での評価(1 日の合計損益の増分 bp)")
    print("   単価(lift)だけ見ると「引きまくって単価を上げた」方策が勝つので必ず併記する")
    print(f"{'シグナル':<8}{'窓ms':>6}{'閾値':>7}{'引いた率':>9}{'総額増':>10}"
          f"{'正の日':>8}{'符号p':>9}{'乱数超え':>9}")
    for sn in SIGNALS:
        sub = [c for c in cells.values() if c["signal"] == sn and c["delta_ms"] == 100]
        if not sub:
            continue
        c = max(sub, key=lambda x: x["total_gain_median"])
        print(f"{sn:<8}{c['window_ms']:>6}{c['threshold']:>7.2f}{c['pull_rate']*100:>8.1f}%"
              f"{c['total_gain_median']:>10.1f}{c['total_pos_days']:>4}/{c['n_days']}"
              f"{c['total_sign_p']:>9.4f}{c['rand_total_beat_days']:>5}/{c['n_days']}")

    # ★最重要の対照: そもそも出さない(総額 0)と比べる。
    #   母集団の平均が負なら「引く」より「出さない」が強い可能性がある
    print("\n【0c】★『そもそも出さない』との比較(この母集団は平均が負なので必須の対照)")
    tot_base = np.array([d["net"].sum() for d in days])
    res["baseline_total_median"] = float(np.median(tot_base))
    res["baseline_total_neg_days"] = int((tot_base < 0).sum())
    print(f"  出し続ける: 合計 {np.median(tot_base):+.1f} bp/日(中央値)、"
          f"赤字 {(tot_base < 0).sum()}/{nd} 日")
    print(f"  出さない  : 合計 0.0 bp/日(定義)")
    print(f"{'シグナル':<8}{'窓ms':>6}{'閾値':>7}{'引いた率':>9}{'残り合計':>10}"
          f"{'0 を上回る日':>13}{'符号p':>9}")
    for sn in SIGNALS:
        sub = [c for c in cells.values() if c["signal"] == sn and c["delta_ms"] == 100]
        if not sub:
            continue
        best, bv = None, -1e18
        for c in sub:
            si, wi, ti = (SIGNALS.index(c["signal"]), WINDOWS_MS.index(c["window_ms"]),
                          THRESHOLDS.index(c["threshold"]))
            rem = []
            for d in days:
                mg = d["margin"][:, si, wi, ti]
                p_ = np.isfinite(mg) & (mg > c["delta_ms"] * 1_000_000)
                rem.append(float(d["net"][~p_].sum()))
            r = np.array(rem)
            if np.median(r) > bv:
                best, bv, br = c, float(np.median(r)), r
        pos = int((br > 0).sum())
        print(f"{sn:<8}{best['window_ms']:>6}{best['threshold']:>7.2f}"
              f"{best['pull_rate']*100:>8.1f}%{bv:>10.1f}{pos:>9}/{nd}"
              f"{stats.binomtest(pos, nd, 0.5, alternative='greater').pvalue:>9.4f}")
        res.setdefault("vs_no_quote", {})[sn] = {
            "cell": f"{sn}|{best['window_ms']}|{best['threshold']}|100",
            "remaining_total_median": bv, "pos_days": pos, "n_days": nd}

    print("\n【0b】引いた率を揃えた比較(引いた率が最も近いセル同士・δ=100ms)")
    for target in (0.2, 0.5, 0.8):
        print(f"  引いた率 ≈ {target*100:.0f}%: ", end="")
        for sn in SIGNALS:
            sub = [c for c in cells.values() if c["signal"] == sn and c["delta_ms"] == 100]
            if not sub:
                continue
            c = min(sub, key=lambda x: abs(x["pull_rate"] - target))
            print(f"{sn} lift{c['lift_median']:+.4f}(実{c['pull_rate']*100:.0f}%)  ", end="")
        print()

    print("\n【1】シグナル別の最良セル(δ=100ms、lift の日次中央値で選択)")
    print(f"{'シグナル':<8}{'窓ms':>6}{'閾値':>7}{'引いた率':>9}{'lift':>9}"
          f"{'正の日':>8}{'符号p':>9}{'95%CI':>20}{'乱数超え日':>10}")
    for sn in SIGNALS:
        sub = [c for c in cells.values() if c["signal"] == sn and c["delta_ms"] == 100]
        if not sub:
            continue
        c = max(sub, key=lambda x: x["lift_median"])
        print(f"{sn:<8}{c['window_ms']:>6}{c['threshold']:>7.2f}{c['pull_rate']*100:>8.1f}%"
              f"{c['lift_median']:>9.4f}{c['pos_days']:>4}/{c['n_days']}{c['sign_p']:>9.4f}"
              f"  [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]{c['rand_beat_days']:>6}/{c['n_days']}")

    print("\n【2】反応時間 δ の影響(cancel シグナルの最良セル)")
    for dl in DELTAS_MS:
        sub = [c for c in cells.values() if c["signal"] == "cancel" and c["delta_ms"] == dl]
        if not sub:
            continue
        c = max(sub, key=lambda x: x["lift_median"])
        print(f"  δ={dl:>3}ms  窓{c['window_ms']:>4}ms 閾値{c['threshold']:.2f}  "
              f"引いた率 {c['pull_rate']*100:>5.1f}%  lift {c['lift_median']:+.4f}  "
              f"正 {c['pos_days']}/{c['n_days']}  p={c['sign_p']:.4f}")

    print("\n【3】cancel シグナルの全格子(δ=100ms)")
    print(f"{'窓ms':>6}" + "".join(f"{'閾値'+str(t):>16}" for t in THRESHOLDS))
    for w in WINDOWS_MS:
        line = f"{w:>6}"
        for t in THRESHOLDS:
            c = cells.get(f"cancel|{w}|{t}|100")
            line += (f"{c['lift_median']:>+9.4f}({c['pos_days']:>2}/{c['n_days']:<2})"
                     if c else f"{'-':>16}")
        print(line)

    if "oos_selection" in res:
        o = res["oos_selection"]
        print(f"\n【4】方策選択の標本外検証({o['n_eval']} 日)")
        print(f"  過去だけで選んだ組合せの当日 lift: 中央値 {o['median']:+.4f} / "
              f"平均 {o['mean']:+.4f} / 正 {o['pos_days']}/{o['n_eval']} / "
              f"符号検定 p={o['sign_p']:.4f}")
        print(f"  95%CI [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}]  p(≤0)={o['p_le0']:.4f}")
        print(f"  全期間を見てから選んだ最良: {o['in_sample_best']} → "
              f"{o['in_sample_best_lift']:+.4f}(選択バイアス "
              f"{o['in_sample_best_lift']-np.median(oos):+.4f})")
        u, ct = np.unique(chosen, return_counts=True)
        print("  選ばれた組合せ: " + " / ".join(f"{x}×{y}" for x, y in
                                                 sorted(zip(u, ct), key=lambda z: -z[1])[:4]))

    # 総額・対照の結果は表示の過程で計算しているので、最後にもう一度書き出す
    (D / "pull_rule_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
