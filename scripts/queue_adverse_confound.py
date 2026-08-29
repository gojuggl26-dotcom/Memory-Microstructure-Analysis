"""交絡の切り分け — キュー順位の非単調性は「板の厚み/スプレッド」の構成効果か。

【検証したい命題】
  約定後の価格変動がキュー順位に対し非単調(中間層 100〜1000 が有利)だった。
  しかし q_ahead は「行列の順番」であると同時に**発注時の板の厚み**の代理でもある。
  厚い板 = 穏やかな相場 = 逆選択が少ない、という経路がありうる。

  → もし構成効果なら、**スプレッド/厚みを層内に固定すれば非単調性は消える**。
    消えなければキュー順位そのものの効果である。

  さらに補助証拠として、約定までの待ち時間と発注時スプレッドを層別に出す。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "queue_adverse2"
QORDER = ["0 (先頭)", "0<q<=20", "20<q<=100", "100<q<=400", "400<q<=1000", "q>1000"]
SHORT = {"0 (先頭)": "0", "0<q<=20": "1-20", "20<q<=100": "21-100",
         "100<q<=400": "101-400", "400<q<=1000": "401-1k", "q>1000": ">1k"}
TERC = {0: "狭い/薄い", 1: "中", 2: "広い/厚い"}

# ★2026-08-10 は除外する。lifecycle の ts_close に null がある唯一の日(13,458 件)で、
#   polars の to_numpy() が float64 を返すため ns タイムスタンプ(~1.79e18)が
#   2^53 の 199 倍で精度を失い、build_queue_qi の到着/消滅マージが壊れている。
#   実際 q_ahead_open の中央値が 7,947(前日は 7)と 1,135 倍に膨らんでいる。
EXCLUDE = {"2026-08-10"}



def dstat(v: list[float]) -> dict:
    a = np.array([x for x in v if x is not None and np.isfinite(x)])
    if a.size < 3:
        return {}
    pos = int((a > 0).sum())
    return {"median": float(np.median(a)), "n_days": int(a.size), "pos_days": pos,
            "sign_p": float(stats.binomtest(max(pos, a.size - pos), a.size, 0.5).pvalue)}


def main() -> None:
    days = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))
            if f.stem not in EXCLUDE]
    res: dict = {"n_days": len(days), "days": [d["dt"] for d in days]}

    # ---- 記述: 各バケットが「どういう局面の約定」なのか ------------------
    desc = {}
    for qb in QORDER:
        rows = [next((r for r in d["post_fill"]
                      if r["q"] == qb and r["scope"] == "最良気配のみ"), None) for d in days]
        rows = [r for r in rows if r]
        if len(rows) < 3:
            continue
        desc[qb] = {"n": sum(r["n"] for r in rows),
                    "spread_at_open": float(np.median([r["spread_at_open"] for r in rows])),
                    "depth_at_open": float(np.median([r["depth_at_open"] for r in rows])),
                    "delay_median_sec": float(np.median([r["delay_sec_median"] for r in rows])),
                    "delay_mean_sec": float(np.median([r["delay_sec_mean"] for r in rows])),
                    "gross": dstat([r["gross_bp"] for r in rows]),
                    "adv_1s": dstat([r.get("adv_1s") for r in rows])}
    res["describe"] = desc

    # ---- 層別: スプレッド三分位 / 厚み三分位を固定した中での adv_1s -------
    st: dict = {}
    for by in ("spread", "depth"):
        for qb in QORDER:
            for k in (0, 1, 2):
                v, g = [], []
                for d in days:
                    r = next((x for x in d["strat"]
                              if x["q"] == qb and x["by"] == by and x["tercile"] == k), None)
                    if r:
                        v.append(r["adv_1s"]); g.append(r["gross_bp"])
                s = dstat(v)
                if s:
                    st[f"{by}|{qb}|{k}"] = {**s, "gross": float(np.median(g))}
    res["stratified"] = st
    (D / "queue_confound.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")

    print(f"=== {res['n_days']} 日 ===")
    print("\n【A】各バケットは「どういう局面の約定」か(日次中央値)")
    print(f"{'前の数量':<10}{'件数':>10}{'発注時spread':>13}{'発注時厚み':>12}"
          f"{'約定待ち中央値':>14}{'待ち平均':>10}{'粗利':>8}{'逆選択1s':>10}")
    for qb in QORDER:
        if qb not in desc:
            continue
        e = desc[qb]
        print(f"{SHORT[qb]:<10}{e['n']:>10,}{e['spread_at_open']:>13.3f}"
              f"{e['depth_at_open']:>12.0f}{e['delay_median_sec']:>13.2f}s"
              f"{e['delay_mean_sec']:>9.1f}s{e['gross']['median']:>8.3f}"
              f"{e['adv_1s']['median']:>10.3f}")

    for by, tlab in (("spread", ("狭", "中", "広")), ("depth", ("薄", "中", "厚"))):
        nm = "発注時スプレッド" if by == "spread" else "発注時の板の厚み"
        print(f"\n【B】{nm}を層内に固定した逆選択 1s(日次中央値 bp)")
        print(f"{'前の数量':<10}" + "".join(f"{t:>14}" for t in tlab))
        for qb in QORDER:
            line = f"{SHORT[qb]:<10}"
            for k in (0, 1, 2):
                key = f"{by}|{qb}|{k}"
                if key in st:
                    e = st[key]
                    line += f"{e['median']:>9.3f}({e['pos_days']:>2}/{e['n_days']:<2})"
                else:
                    line += f"{'-':>14}"
            print(line)
        # 層内で単調性が回復するか: 各層で先頭バケットと中間層の差
        print("   層内での「中間層(101-400) − 先頭」の差:", end=" ")
        for k in (0, 1, 2):
            a = st.get(f"{by}|100<q<=400|{k}"); b = st.get(f"{by}|0 (先頭)|{k}")
            if a and b:
                print(f"{tlab[k]} {a['median']-b['median']:+.3f}", end="  ")
        print()


if __name__ == "__main__":
    main()
