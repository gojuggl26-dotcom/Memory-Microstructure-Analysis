"""キュー順位分析の集計 — 日次推論・符号一貫性・ブロックブートストラップ。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "queue_adverse2"
# ★メイカー手数料(2026-08-16 訂正)
#   当初 net = gross + adv で手数料を控除していなかった。
#   net は 1 約定あたりの bp なので、定数 0.088 を引くのは
#   日次処理をやり直すのと数学的に同一である(再処理は不要)。
#   これを入れると「出す価値」は回数の多い層ほど削られ、
#   「引く価値」は引いた分の手数料を払わずに済むぶん増える。
FEE_BP = 0.088
RNG = np.random.default_rng(20260815)
B = 5000
QORDER = ["0 (先頭)", "0<q<=20", "20<q<=100", "100<q<=400", "400<q<=1000", "q>1000"]
DORDER = ["内側", "最良気配", "0-1bp", "1-5bp"]
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


def daily(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if a.size < 3:
        return {}
    lo, hi = boot(a)
    pos = int((a > 0).sum())
    return {"median": float(np.median(a)), "mean": float(a.mean()), "n_days": int(a.size),
            "pos_days": pos, "ci": [lo, hi],
            "sign_p": float(stats.binomtest(max(pos, a.size - pos), a.size, 0.5).pvalue)}


def main() -> None:
    files = [f for f in sorted(OUT.glob("*.json")) if f.stem not in EXCLUDE]
    days = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    res: dict = {"n_days": len(days), "days": [d["dt"] for d in days],
                 "n_maker_fills": sum(d["n_maker_fills"] for d in days),
                 "n_orders": sum(d["n_orders"] for d in days)}

    # ---- (1) 約定確率 ---------------------------------------------------
    fp = {}
    for qb in QORDER:
        for db in DORDER:
            v = [next((r["fill_rate"] for r in d["fill_prob"]
                       if r["q"] == qb and r["dist"] == db), None) for d in days]
            s = daily(v)
            if s:
                nn = sum(next((r["n"] for r in d["fill_prob"]
                               if r["q"] == qb and r["dist"] == db), 0) for d in days)
                fp[f"{qb}|{db}"] = {**s, "n": nn}
    res["fill_prob"] = fp

    # ---- (2) 約定後の平均価格変動 ----------------------------------------
    pf = {}
    for scope in ("最良気配のみ", "全体"):
        for qb in QORDER:
            rows = [next((r for r in d["post_fill"]
                          if r["q"] == qb and r["scope"] == scope), None) for d in days]
            rows = [r for r in rows if r]
            if len(rows) < 3:
                continue
            e = {"n": sum(r["n"] for r in rows),
                 "gross": daily([r["gross_bp"] for r in rows])}
            for h in HZ:
                e[f"adv_{h}"] = daily([r.get(f"adv_{h}") for r in rows])
                e[f"net_{h}"] = daily([(r[f"net_{h}"] - FEE_BP)   # 手数料控除
                                       if r.get(f"net_{h}") is not None else None
                                       for r in rows])
                e[f"neg_{h}"] = daily([r.get(f"neg_{h}") for r in rows])
            # 分布(1 秒)は日次の分位の中央値をとる
            qs = {}
            for p in ("1", "5", "10", "25", "50", "75", "90", "95", "99"):
                vv = [r["q_adv"][p] for r in rows if "q_adv" in r]
                if vv:
                    qs[p] = float(np.median(vv))
            e["quantiles_1s"] = qs
            e["sd_1s"] = float(np.median([r["sd_1s"] for r in rows if "sd_1s" in r]))
            pf[f"{scope}|{qb}"] = e
    res["post_fill"] = pf
    (D / "queue_adverse_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")

    print(f"=== {res['n_days']} 日 / 注文 {res['n_orders']:,} / メイカー約定 "
          f"{res['n_maker_fills']:,} ===")
    print("\n【1】キュー順位 × 置き場所 別の約定確率(日次中央値)")
    print(f"{'自分の前の数量':<14}" + "".join(f"{d:>12}" for d in DORDER))
    for qb in QORDER:
        line = f"{qb:<14}"
        for db in DORDER:
            k = f"{qb}|{db}"
            line += f"{fp[k]['median']*100:>11.2f}%" if k in fp else f"{'-':>12}"
        print(line)

    print("\n【2】約定後の価格変動(最良気配に置いた注文のみ・日次中央値 bp)")
    print(f"{'自分の前の数量':<14}{'件数':>10}{'粗利':>8}"
          + "".join(f"{'逆選択'+h:>11}" for h in HZ) + f"{'正味1s':>9}{'不利率1s':>10}")
    for qb in QORDER:
        k = f"最良気配のみ|{qb}"
        if k not in pf:
            continue
        e = pf[k]
        line = f"{qb:<14}{e['n']:>10,}{e['gross']['median']:>8.3f}"
        for h in HZ:
            line += f"{e[f'adv_{h}']['median']:>11.3f}"
        line += f"{e['net_1s']['median']:>9.3f}{e['neg_1s']['median']*100:>9.1f}%"
        print(line)

    print("\n【3】約定 1 秒後の価格変動の分布(最良気配・日次分位の中央値 bp)")
    ps = ["1", "5", "10", "25", "50", "75", "90", "95", "99"]
    print(f"{'自分の前の数量':<14}" + "".join(f"{'p'+p:>8}" for p in ps) + f"{'標準偏差':>9}")
    for qb in QORDER:
        k = f"最良気配のみ|{qb}"
        if k not in pf:
            continue
        e = pf[k]
        print(f"{qb:<14}" + "".join(f"{e['quantiles_1s'].get(p, float('nan')):>8.2f}"
                                    for p in ps) + f"{e['sd_1s']:>9.2f}")

    print("\n【4】統計的裏づけ(逆選択 1 秒・最良気配)")
    print(f"{'自分の前の数量':<14}{'中央値':>9}{'平均':>9}{'負の日':>8}{'符号検定p':>10}"
          f"{'95%CI':>22}")
    for qb in QORDER:
        k = f"最良気配のみ|{qb}"
        if k not in pf:
            continue
        a = pf[k]["adv_1s"]
        print(f"{qb:<14}{a['median']:>9.3f}{a['mean']:>9.3f}"
              f"{a['n_days']-a['pos_days']:>5}/{a['n_days']}{a['sign_p']:>10.4f}"
              f"  [{a['ci'][0]:>+7.3f},{a['ci'][1]:>+7.3f}]")


if __name__ == "__main__":
    main()
