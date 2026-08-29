"""交絡の切り分け — clear_ratio の効果は q_ahead(行列の深さ)の言い換えか。

【問題】
  clear_ratio 別の逆選択を測ったところ、
    全部取消(clear_ratio<−0.8) … −0.317 bp(28/28 日で負)
    全部約定(clear_ratio>+0.8) … −0.093 bp
  と、事前予測と**逆**の結果になった。
  しかし前の数量(q_ahead)の中央値が 29 / 9 / 82〜105 とバケット間で大きく違う。
  前報告(queue_position_report.md)で q_ahead 自体が逆選択と非単調な関係を持つと
  判っている以上、**clear_ratio の効果が q_ahead の言い換えである可能性**がある。

【検証】
  q_ahead を層内に固定して clear_ratio の効果が残るかを見る。逆も見る。
  残れば独立した効果、消えれば言い換え。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
OUT = D / "queue_clearing"
# 注文単位配列の行の意味(queue_clearing.py の np.save と対応)
CR, QO, EX, CN, GR, AB, A100, A1S, A10, A60, GOOD = range(11)
# ★メイカー手数料(2026-08-16 訂正)
#   当初 net = gross + adv で手数料を控除していなかった。
#   net は 1 約定あたりの bp なので、定数 0.088 を引くのは
#   日次処理をやり直すのと数学的に同一である(再処理は不要)。
#   これを入れると「出す価値」は回数の多い層ほど削られ、
#   「引く価値」は引いた分の手数料を払わずに済むぶん増える。
FEE_BP = 0.088
QB = [(-0.5, 20, "q<=20"), (20, 100, "20<q<=100"), (100, 400, "100<q<=400"),
      (400, np.inf, "q>400")]
CBK = [(-1.01, -0.8, "全部取消"), (-0.8, 0.0, "取消優勢"),
       (0.0, 0.8, "約定優勢"), (0.8, 1.01, "全部約定")]

# ★2026-08-10 は除外する。lifecycle の ts_close に null がある唯一の日(13,458 件)で、
#   polars の to_numpy() が float64 を返すため ns タイムスタンプ(~1.79e18)が
#   2^53 の 199 倍で精度を失い、build_queue_qi の到着/消滅マージが壊れている。
#   実際 q_ahead_open の中央値が 7,947(前日は 7)と 1,135 倍に膨らんでいる。
EXCLUDE = {"2026-08-10"}



def dstat(v: list) -> dict:
    a = np.array([x for x in v if x is not None and np.isfinite(x)])
    if a.size < 3:
        return {}
    pos = int((a > 0).sum())
    return {"median": float(np.median(a)), "n_days": int(a.size), "pos_days": pos,
            "sign_p": float(stats.binomtest(max(pos, a.size - pos), a.size, 0.5).pvalue)}


def main() -> None:
    files = [f for f in sorted(OUT.glob("*.npy")) if f.stem not in EXCLUDE]
    if not files:
        print("注文単位データがまだありません")
        return
    per_day = []
    for f in files:
        a = np.load(f).astype(np.float64)
        m = (a[GOOD] > 0.5) & (a[AB] > 0.5) & np.isfinite(a[A1S])   # 最良気配のみ
        per_day.append(a[:, m])
    print(f"=== {len(per_day)} 日 / 最良気配の約定 "
          f"{sum(x.shape[1] for x in per_day):,} 件 ===")

    res: dict = {"n_days": len(per_day), "cells": {}, "marginal": {}}

    def cell(qlo, qhi, clo, chi, key):
        adv, net, n = [], [], 0
        for a in per_day:
            m = ((a[QO] > qlo) & (a[QO] <= qhi)
                 & (a[CR] > clo) & (a[CR] <= chi))
            if m.sum() < 60:
                continue
            adv.append(float(a[A1S][m].mean()))
            net.append(float(a[GR][m].mean() - FEE_BP + a[A1S][m].mean()))
            n += int(m.sum())
        s = dstat(adv)
        if s:
            res["cells"][key] = {**s, "n": n, "net": dstat(net).get("median", np.nan)}
        return res["cells"].get(key)

    # ---- 2 元表 ----
    print("\n【A】逆選択 1s: q_ahead(行) × clear_ratio(列)  日次中央値 bp")
    print(f"{'前の数量':<12}" + "".join(f"{c[2]:>17}" for c in CBK))
    for qlo, qhi, ql in QB:
        line = f"{ql:<12}"
        for clo, chi, cl in CBK:
            e = cell(qlo, qhi, clo, chi, f"{ql}|{cl}")
            line += (f"{e['median']:>9.3f}({e['n_days']-e['pos_days']:>2}/{e['n_days']:<2})"
                     if e else f"{'-':>17}")
        print(line)

    # ---- 周辺: 層内で clear_ratio の効果が残るか ----
    print("\n【B】各 q_ahead 層での「全部取消 − 全部約定」の差(負なら取消側が不利)")
    for qlo, qhi, ql in QB:
        a_ = res["cells"].get(f"{ql}|全部取消"); b_ = res["cells"].get(f"{ql}|全部約定")
        if a_ and b_:
            print(f"  {ql:<12} 全部取消 {a_['median']:+.3f} − 全部約定 {b_['median']:+.3f}"
                  f" = {a_['median']-b_['median']:+.3f} bp")

    print("\n【C】各 clear_ratio 層での「浅い(q<=20) − 深い(q>400)」の差")
    for clo, chi, cl in CBK:
        a_ = res["cells"].get(f"q<=20|{cl}"); b_ = res["cells"].get(f"q>400|{cl}")
        if a_ and b_:
            print(f"  {cl:<10} 浅い {a_['median']:+.3f} − 深い {b_['median']:+.3f}"
                  f" = {a_['median']-b_['median']:+.3f} bp")

    # ---- 連続値での確認: 日ごとの偏回帰(adv ~ clear_ratio + log q_ahead)----
    co_c, co_q = [], []
    for a in per_day:
        if a.shape[1] < 500:
            continue
        X = np.column_stack([np.ones(a.shape[1]), a[CR], np.log1p(a[QO])])
        try:
            b = np.linalg.lstsq(X, a[A1S], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        co_c.append(float(b[1])); co_q.append(float(b[2]))
    sc, sq = dstat(co_c), dstat(co_q)
    res["partial_regression"] = {"clear_ratio": sc, "log_q_ahead": sq}
    print("\n【D】日ごとの偏回帰  adv_1s = α + β₁·clear_ratio + β₂·log(1+q_ahead)")
    print(f"  β₁(clear_ratio) 中央値 {sc['median']:+.4f}  "
          f"正の日 {sc['pos_days']}/{sc['n_days']}  符号検定 p={sc['sign_p']:.4f}")
    print(f"  β₂(log q_ahead) 中央値 {sq['median']:+.4f}  "
          f"正の日 {sq['pos_days']}/{sq['n_days']}  符号検定 p={sq['sign_p']:.4f}")

    # ---- 期待損益(発注 1 件あたりではなく約定 1 件あたりの正味)----
    print("\n【E】正味損益 1s(粗利 + 逆選択)  q_ahead × clear_ratio")
    print(f"{'前の数量':<12}" + "".join(f"{c[2]:>12}" for c in CBK))
    for qlo, qhi, ql in QB:
        line = f"{ql:<12}"
        for clo, chi, cl in CBK:
            e = res["cells"].get(f"{ql}|{cl}")
            line += f"{e['net']:>12.3f}" if e and np.isfinite(e["net"]) else f"{'-':>12}"
        print(line)

    (D / "queue_clearing_cross.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
