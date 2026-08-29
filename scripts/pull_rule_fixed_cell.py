"""固定セルでの体制間・売買間の頑健性検定。

【なぜ必要か】
  層ごと・体制ごとに最良セルを選び直すと、選ばれるセルが体制間で違ってしまい
  「同じ規則が両体制で効くか」の検定になっていない。
  全期間で選んだ 1 つのセルを固定し、それを各体制・各サイドで評価する。

  ★セルの選択は全期間(= 評価期間を含む)で行っているので、
    これは「頑健性の確認」であって「標本外の検証」ではない。混同しないこと。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "pull_rule"
FEE_BP = 0.088
SIGNALS = ["cancel", "exec", "total"]
WINDOWS_MS = [10, 50, 100, 500]
THR = ["frac0.25", "frac0.5", "frac0.75", "abs5", "abs20", "abs50", "abs150", "abs400"]
DELTA_MS = 100
BURN_IN = 7
CELL = ("exec", 10, "abs5")        # 全期間で最良だったセル(最良気配|q>400)


def load():
    keep = [f for f in sorted(OUT.glob("*.npz"))
            if not json.loads((OUT / f"{f.stem}.json").read_text(encoding="utf-8")).get("suspect")]
    keep = keep[BURN_IN:]
    days = []
    for f in keep:
        z = np.load(f)
        days.append({"dt": f.stem, "margin": z["margin"], "net": z["net"].astype(np.float64) - FEE_BP,
                     "qo": z["qo"], "is_bid": z["is_bid"], "dist": z["dist_bp"], "good": z["good"]})
    return days


def main() -> None:
    days = load()
    half = len(days) // 2
    si, wi, ti = SIGNALS.index(CELL[0]), WINDOWS_MS.index(CELL[1]), THR.index(CELL[2])
    print(f"固定セル: {CELL[0]}/{CELL[1]}ms/{CELL[2]} · δ={DELTA_MS}ms · 手数料 {FEE_BP} 控除済み")
    print(f"層: 最良気配 × 前に 400 単位超 / 全 {len(days)} 日\n")
    print(f"{'体制':<10}{'サイド':<8}{'日数':>5}{'出す':>9}{'引く効果':>10}{'改善日':>9}"
          f"{'符号p':>9}{'引率':>7}{'乱数超':>8}")
    res = {}
    for reg, sel in (("全期間", days), ("立ち上げ", days[:half]), ("成熟", days[half:])):
        for sd, sl in ((None, "両側"), (True, "Bid"), (False, "Ask")):
            g, b, rt, zs = [], [], [], []
            for d in sel:
                m = d["good"] & (d["qo"] > 400) & (np.abs(d["dist"]) < 1e-9)
                if sd is not None:
                    m = m & (d["is_bid"] == sd)
                if m.sum() < 100:
                    continue
                net = d["net"][m]
                mg = d["margin"][m][:, si, wi, ti].astype(np.float64)
                p = np.isfinite(mg) & (mg > DELTA_MS * 1_000_000)
                k, n = int(p.sum()), len(net)
                if k == 0 or k >= n:
                    continue
                g.append(float(net[~p].sum() - net.sum()))
                b.append(float(net.sum()))
                rt.append(k / n)
                v = k * net.var(ddof=1) * (n - k) / max(n - 1, 1)
                zs.append((net[~p].mean() - net.mean()) / (np.sqrt(v) / (n - k)) if v > 0 else 0)
            if len(g) < 8:
                continue
            ga, ba = np.array(g), np.array(b)
            pos = int((ga > 0).sum())
            sp = float(stats.binomtest(pos, len(ga), 0.5, alternative="greater").pvalue)
            bp_ = int((ba > 0).sum())
            bsp = float(stats.binomtest(bp_, len(ba), 0.5, alternative="greater").pvalue)
            mark = " ★" if sp < 0.05 else ""
            print(f"{reg:<10}{sl:<8}{len(ga):>5}{np.median(ba):>9.1f}{np.median(ga):>10.1f}"
                  f"{pos:>5}/{len(ga):<3}{sp:>9.4f}{np.median(rt)*100:>6.0f}%"
                  f"{int(np.sum(np.array(zs) > 1.645)):>5}/{len(ga):<3}{mark}")
            res[f"{reg}|{sl}"] = {"n_days": len(ga), "base_median": float(np.median(ba)),
                                  "base_sign_p": bsp,
                                  "gain_median": float(np.median(ga)), "gain_pos_days": pos,
                                  "gain_sign_p": sp, "pull_rate": float(np.median(rt)),
                                  "rand_beat": int(np.sum(np.array(zs) > 1.645))}
    (D / "pull_rule_fixed_cell.json").write_text(
        json.dumps({"cell": list(CELL), "fee_bp": FEE_BP, "results": res},
                   indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
