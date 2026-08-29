"""clear_ratio と逆選択・期待損益の関係を集計する(日次推論)。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "queue_clearing"
# ★メイカー手数料(2026-08-16 訂正)
#   当初 net = gross + adv で手数料を控除していなかった。
#   net は 1 約定あたりの bp なので、定数 0.088 を引くのは
#   日次処理をやり直すのと数学的に同一である(再処理は不要)。
#   これを入れると「出す価値」は回数の多い層ほど削られ、
#   「引く価値」は引いた分の手数料を払わずに済むぶん増える。
FEE_BP = 0.088
RNG = np.random.default_rng(20260815)
B = 5000
CB = ["-1.0〜-0.8 (ほぼ全部取消)", "-0.8〜-0.4", "-0.4〜0.0",
      "0.0〜0.4", "0.4〜0.8", "0.8〜1.0 (ほぼ全部約定)"]
SH = ["-1.0〜-0.8", "-0.8〜-0.4", "-0.4〜0.0", "0.0〜0.4", "0.4〜0.8", "0.8〜1.0"]
HZ = ["100ms", "1s", "10s", "60s"]

# ★2026-08-10 は除外する。lifecycle の ts_close に null がある唯一の日(13,458 件)で、
#   polars の to_numpy() が float64 を返すため ns タイムスタンプ(~1.79e18)が
#   2^53 の 199 倍で精度を失い、build_queue_qi の到着/消滅マージが壊れている。
#   実際 q_ahead_open の中央値が 7,947(前日は 7)と 1,135 倍に膨らんでいる。
EXCLUDE = {"2026-08-10"}



def boot(x: np.ndarray, blk: int = 2):
    n = len(x)
    nb = int(np.ceil(n / blk))
    st = RNG.integers(0, max(n - blk + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def dstat(v: list) -> dict:
    a = np.array([x for x in v if x is not None and np.isfinite(x)])
    if a.size < 3:
        return {}
    lo, hi = boot(a)
    pos = int((a > 0).sum())
    return {"median": float(np.median(a)), "mean": float(a.mean()), "n_days": int(a.size),
            "pos_days": pos, "ci": [lo, hi],
            "sign_p": float(stats.binomtest(max(pos, a.size - pos), a.size, 0.5).pvalue)}


def main() -> None:
    days = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))
            if f.stem not in EXCLUDE]
    res: dict = {"n_days": len(days), "days": [d["dt"] for d in days],
                 "n_targets": sum(d["n_targets"] for d in days),
                 "consistency": float(np.median([d["consistency_tot_over_qahead"]
                                                 for d in days])),
                 # ★中央値だけだと 1 日の破綻を隠す(実際 08-10 の 0.006 を見逃した)
                 "consistency_worst": float(max(
                     abs(d["consistency_tot_over_qahead"] - 1.0) for d in days)),
                 "exec_share_overall": float(np.median([d["exec_share_overall"]
                                                        for d in days])),
                 "share_all_cancel": float(np.median([d["share_all_cancel"] for d in days])),
                 "share_all_exec": float(np.median([d["share_all_exec"] for d in days]))}
    agg: dict = {}
    for scope in ("最良気配のみ", "全体"):
        for b in CB:
            rows = [next((r for r in d["buckets"] if r["bucket"] == b
                          and r["scope"] == scope), None) for d in days]
            rows = [r for r in rows if r]
            if len(rows) < 3:
                continue
            e = {"n": sum(r["n"] for r in rows),
                 "clear_ratio": float(np.median([r["clear_ratio_mean"] for r in rows])),
                 "q_ahead": float(np.median([r["q_ahead_median"] for r in rows])),
                 "gross": dstat([r["gross_bp"] for r in rows])}
            for h in HZ:
                e[f"adv_{h}"] = dstat([r.get(f"adv_{h}") for r in rows])
                e[f"net_{h}"] = dstat([(r[f"net_{h}"] - FEE_BP)
                                   if r.get(f"net_{h}") is not None else None
                                   for r in rows])
            e["neg_1s"] = float(np.median([r["neg_1s"] for r in rows if "neg_1s" in r]))
            e["sd_1s"] = float(np.median([r["sd_1s"] for r in rows if "sd_1s" in r]))
            e["quantiles_1s"] = {p: float(np.median([r["q_adv"][p] for r in rows
                                                     if "q_adv" in r]))
                                 for p in ("5", "25", "50", "75", "95")}
            agg[f"{scope}|{b}"] = e
    res["buckets"] = agg
    (D / "queue_clearing_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")

    print(f"=== {res['n_days']} 日 / 対象 {res['n_targets']:,} 件 ===")
    print(f"整合性(消えた総量 / q_ahead_open)中央値 = {res['consistency']:.4f} / "
          f"1.0 からの最大乖離 = {res['consistency_worst']:.4f}  ← 両方見る")
    print(f"前の行列のうち約定で消えた割合 = {res['exec_share_overall']*100:.1f}%"
          f" / 残り {100-res['exec_share_overall']*100:.1f}% は取消")
    print(f"clear_ratio < −0.8(ほぼ全部取消)の注文 = {res['share_all_cancel']*100:.1f}%"
          f" / > +0.8(ほぼ全部約定)= {res['share_all_exec']*100:.1f}%")

    for scope in ("最良気配のみ", "全体"):
        print(f"\n=== {scope} ===")
        print(f"{'clear_ratio':<13}{'件数':>10}{'前の数量':>9}{'粗利':>8}"
              + "".join(f"{'逆選択'+h:>10}" for h in HZ)
              + f"{'正味1s':>9}{'不利率':>8}{'sd':>7}{'負の日':>8}{'p':>8}")
        for b, sh in zip(CB, SH):
            k = f"{scope}|{b}"
            if k not in agg:
                continue
            e = agg[k]
            line = (f"{sh:<13}{e['n']:>10,}{e['q_ahead']:>9.0f}"
                    f"{e['gross']['median']:>8.3f}")
            for h in HZ:
                line += f"{e[f'adv_{h}'].get('median', float('nan')):>10.3f}"
            a = e["adv_1s"]
            line += (f"{e['net_1s']['median']:>9.3f}{e['neg_1s']*100:>7.1f}%"
                     f"{e['sd_1s']:>7.2f}"
                     f"{a['n_days']-a['pos_days']:>5}/{a['n_days']}{a['sign_p']:>8.4f}")
            print(line)

    # 単調性の検定: clear_ratio と逆選択の関係(バケット中央値の順位相関)
    print("\n=== clear_ratio と逆選択の関係(最良気配)===")
    xs, ys = [], []
    for b in CB:
        k = f"最良気配のみ|{b}"
        if k in agg:
            xs.append(agg[k]["clear_ratio"]); ys.append(agg[k]["adv_1s"]["median"])
    if len(xs) >= 4:
        r = stats.spearmanr(xs, ys)
        print(f"  順位相関 ρ = {r.statistic:+.4f} (p = {r.pvalue:.4f}, n = {len(xs)} バケット)")
        print(f"  端の差: 全部取消 {ys[0]:+.3f} → 全部約定 {ys[-1]:+.3f} "
              f"= {ys[-1]-ys[0]:+.3f} bp")


if __name__ == "__main__":
    main()
