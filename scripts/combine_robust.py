"""採用方策 A+B2 q=0.4 の頑健性 — 体制 × 売買サイド。

方策は全期間で選んでいるので、これは**頑健性の確認**であって標本外検証ではない。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); OUT = D / "pull_rule"
FEE = 0.088; Q = 0.4; QMIN = 400.0; DELTA = 100_000_000; BURN = 7
SIG = ["cancel", "exec", "total"]; W = [10, 50, 100, 500]
THR = ["frac0.25", "frac0.5", "frac0.75", "abs5", "abs20", "abs50", "abs150", "abs400"]
C = (SIG.index("exec"), W.index(10), THR.index("abs5"))


def load():
    keep = [f for f in sorted(OUT.glob("*.npz"))
            if not json.loads((OUT / f"{f.stem}.json").read_text(encoding="utf-8")).get("suspect")]
    out = []
    for f in keep[BURN:]:
        z = np.load(f)
        base = z["good"] & (z["qo"] > QMIN) & (np.abs(z["dist_bp"]) < 1e-9)
        if base.sum() < 100:
            continue
        mg = z["margin"][base][:, C[0], C[1], C[2]].astype(np.float64)
        out.append({"dt": f.stem, "gross": z["gross"][base].astype(np.float64),
                    "adv": z["adv"][base].astype(np.float64),
                    "pull": np.isfinite(mg) & (mg > DELTA),
                    "amin": z["align_min"][base].astype(np.float64),
                    "bid": z["is_bid"][base]})
    return out


def sgn(a):
    p = int((np.asarray(a) > 0).sum())
    return p, float(stats.binomtest(p, len(a), .5, alternative="greater").pvalue)


def main() -> None:
    days = load()
    half = len(days) // 2
    print(f"方策: A(exec/10ms/abs5・δ=100ms)+ B2(align_min 下位 {Q:.0%} を切る)")
    print(f"層: 最良気配 × 前に {QMIN:.0f} 単位超 / 全 {len(days)} 日 / 手数料控除済み\n")
    print(f"{'体制':<10}{'サイド':<8}{'日数':>5}{'残す率':>8}{'E[PnL]':>10}{'正の日':>9}{'p(水準)':>10}"
          f"{'改善':>9}{'改善日':>9}{'p(改善)':>10}")
    res = {}
    for reg, sel in (("全期間", days), ("立ち上げ", days[:half]), ("成熟", days[half:])):
        for sd, sl in ((None, "両側"), (True, "Bid"), (False, "Ask")):
            per, gain, kr = [], [], []
            prev = None
            for d in sel:
                if prev is None:
                    prev = d; continue
                m = np.ones(len(d["gross"]), bool) if sd is None else (d["bid"] == sd)
                pm = np.ones(len(prev["gross"]), bool) if sd is None else (prev["bid"] == sd)
                if m.sum() < 50 or pm.sum() < 50:
                    prev = d; continue
                thr = np.nanquantile(prev["amin"][pm], Q)
                k = m & (~d["pull"]) & (d["amin"] >= thr)
                if k.sum() < 10:
                    prev = d; continue
                net = d["gross"] + d["adv"] - FEE
                per.append(float(net[k].mean()))
                gain.append(float(net[k].sum() - net[m].sum()))
                kr.append(k.sum() / m.sum())
                prev = d
            if len(per) < 8:
                continue
            lp, lpv = sgn(per); gp, gpv = sgn(gain)
            mk = " ★●" if (lpv < .05 and gpv < .05) else (" ●" if lpv < .05 else
                                                          (" ★" if gpv < .05 else ""))
            print(f"{reg:<10}{sl:<8}{len(per):>5}{np.median(kr)*100:>7.0f}%"
                  f"{np.median(per):>10.4f}{lp:>5}/{len(per):<3}{lpv:>10.4f}"
                  f"{np.median(gain):>9.1f}{gp:>5}/{len(gain):<3}{gpv:>10.4f}{mk}")
            res[f"{reg}|{sl}"] = {"n_days": len(per), "keep_rate": float(np.median(kr)),
                                  "per_fill": float(np.median(per)), "per_pos": lp, "per_p": lpv,
                                  "gain": float(np.median(gain)), "gain_pos": gp, "gain_p": gpv}
    (D / "combine_robust.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print("\n★ = 改善が有意 / ● = E[PnL] の水準が有意に正")


if __name__ == "__main__":
    main()
