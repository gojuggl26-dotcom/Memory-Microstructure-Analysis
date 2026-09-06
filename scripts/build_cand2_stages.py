"""候補 2 の信号が、どの段階で価値を失うかを分解する。

    uv run python scripts/build_cand2_stages.py --coin xyz:MU

前回のまとめで「価格との相関 −0.0888 が往復損益で +0.0045 になったので 20 分の 1」
と書いたが、これは**厳密でない**。標本(5 秒格子の全点 対 約定した建玉)も、
目的変数(mid リターン 対 往復損益)も、ホライズンも違い、片方だけ −side を
掛けていた。ここでは**同じ発注時点 t・同じ側 s・同じホライズン h** に揃え、
段階を追って落ちる場所を特定する。

    U_i(h) = s_i · log( mid_{t_i+h} / mid_{t_i} )   … 自分の側から見た価格変化

  段階 1  全候補        E[U(h) | A]              そもそもの価格予測
  段階 2  約定した候補   E[U(h) | Fill, A]        約定による選別
  段階 3  選別の効果     SP(A) = 段階2 − 段階1     同じ価格窓で引く
  段階 4  約定後         約定時刻からの markout    入口価格の不利・約定後の逆行
  段階 5  往復          E[Π | Fill, A]           在庫解消まで含めた実損益

信号は `A = I_sell(5) − I_buy(5) = imp_b_q5 − imp_a_q5` を自分の側に揃えた
`−s·A`(買いなら A が負のときに正)。段階 1〜5 すべて**同じ向きの信号**で切る。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entrygate import (impact100_feats,  # noqa: E402
                             impact_feats)
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
H = 5.0        # 元の信号と同じ 5 秒
NQ = 5


def sig_age(tag, dt, ts, src):
    """発注時点で使う信号が、何秒前の格子点のものか。"""
    f = DATA / (f"impact_{tag}.parquet" if src == "impact"
                else f"impact100_{tag}.parquet")
    I = (pl.scan_parquet(f).filter(pl.col("dt") == dt).select("ts")
         .collect().sort("ts"))
    if not I.height:
        return np.full(ts.size, np.nan)
    it = I["ts"].cast(pl.Int64).to_numpy()
    j = np.clip(np.searchsorted(it, ts, side="right") - 1, 0, it.size - 1)
    return (ts - it[j]) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sfx", default="_q1_imp1")
    ap.add_argument("--src", choices=["impact", "impact100"], default="impact",
                    help="信号の格子。impact=5 秒(既報)、impact100=100 ms(新鮮)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    te = [f.stem.split("=")[1] for f in sorted((BULK / tag).glob("dt=*.parquet"))[59:]]
    P = pl.read_parquet(DATA / f"inv_posts_{tag}{a.sfx}.parquet")
    L = pl.read_parquet(DATA / f"inv_lots_{tag}{a.sfx}.parquet")
    rows, age = [], []
    for dt in te:
        G = P.filter(pl.col("dt") == dt)
        if not G.height:
            continue
        d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                      .filter(pl.col("dt") == dt).collect())[0].sort("ts")
        bt = d["ts"].cast(pl.Int64).to_numpy()
        mid = 0.5 * (d["best_bid"].to_numpy() + d["best_ask"].to_numpy())
        t = G["t"].to_numpy()
        s = G["side"].to_numpy().astype(np.float64)
        A = (impact_feats(tag, dt, t, s)[:, 1] if a.src == "impact"
             else impact100_feats(tag, dt, t, s)[:, 0])   # −s·A(自分の側)
        age.append(sig_age(tag, dt, t, a.src))
        j0 = np.clip(np.searchsorted(bt, t, "right") - 1, 0, bt.size - 1)
        jh = np.clip(np.searchsorted(bt, t + int(H * 1e9), "right") - 1,
                     0, bt.size - 1)
        U = s * np.log(mid[jh] / mid[j0]) * 1e4      # 発注時点からの自分側リターン
        # 約定した建玉については、約定時刻からの markout も出す
        Ld = L.filter(pl.col("dt") == dt)
        tin = Ld["t_in"].to_numpy()
        sin = Ld["side"].to_numpy().astype(np.float64)
        ji = np.clip(np.searchsorted(bt, tin, "right") - 1, 0, bt.size - 1)
        jf = np.clip(np.searchsorted(bt, tin + int(H * 1e9), "right") - 1,
                     0, bt.size - 1)
        Uf = sin * np.log(mid[jf] / mid[ji]) * 1e4
        # 建玉を発注記録へ戻す(t_in は約定時刻なので、発注時刻ではない)
        Af_ = (impact_feats(tag, dt, tin, sin)[:, 1] if a.src == "impact"
               else impact100_feats(tag, dt, tin, sin)[:, 0])
        rows.append((A, U, G["filled"].to_numpy(), G["rt_pnl"].to_numpy(),
                     Af_, Uf, Ld["pnl"].to_numpy().astype(np.float64)))
    A = np.concatenate([r[0] for r in rows])
    U = np.concatenate([r[1] for r in rows])
    fl = np.concatenate([r[2] for r in rows])
    rt = np.concatenate([r[3] for r in rows])
    Af = np.concatenate([r[4] for r in rows])
    Uf = np.concatenate([r[5] for r in rows])
    pnl = np.concatenate([r[6] for r in rows])

    e = np.quantile(A, np.linspace(0, 1, NQ + 1)[1:-1])
    k = np.searchsorted(e, A, "right")
    ef = np.quantile(Af, np.linspace(0, 1, NQ + 1)[1:-1])
    kf = np.searchsorted(ef, Af, "right")
    fin = np.isfinite(rt)
    out = []
    print(f"発注 {A.size:,} / 約定して建玉になった {int(fin.sum()):,} / "
          f"建玉表 {pnl.size:,}(標本外 39 日、h={H:g}s)")
    print(f"\n{'段階':32s} " + " ".join(f"{'Q'+str(i+1):>8s}" for i in range(NQ))
          + "   Q5−Q1")
    def line(nm, vals):
        out.append({"stage": nm, **{f"q{i+1}": vals[i] for i in range(NQ)},
                    "spread": vals[-1] - vals[0]})
        print(f"{nm:32s} " + " ".join(f"{v:>+8.4f}" for v in vals)
              + f"   {vals[-1]-vals[0]:+.4f}")
    v1 = [float(U[k == i].mean()) for i in range(NQ)]
    line("1 全候補 E[U(5s)]", v1)
    m = fl > 0
    v2 = [float(U[m & (k == i)].mean()) for i in range(NQ)]
    line("2 約定した候補 E[U(5s)|Fill]", v2)
    line("3 選別の効果 SP = 2 − 1", [v2[i] - v1[i] for i in range(NQ)])
    v4 = [float(Uf[kf == i].mean()) for i in range(NQ)]
    line("4 約定時刻からの markout(5s)", v4)
    v5 = [float(pnl[kf == i].mean()) for i in range(NQ)]
    line("5 往復損益 E[Π|Fill]", v5)
    tail = "" if a.src == "impact" else "_100"
    pl.DataFrame(out).write_csv(DATA / f"cand2_stages_{tag}{tail}.csv")
    ag = np.concatenate(age)
    ag = ag[np.isfinite(ag)]
    print()
    print(f"信号の古さ({a.src}): 中央 {np.median(ag):.3f}s / "
          f"p90 {np.quantile(ag, 0.9):.3f}s / 平均 {ag.mean():.3f}s")
    print(f"\n相関(同じ向きの信号 −s·A で):")
    print(f"  全候補 U(5s)          {np.corrcoef(A, U)[0,1]:+.4f}  n={A.size:,}")
    print(f"  約定候補 U(5s)        {np.corrcoef(A[m], U[m])[0,1]:+.4f}  n={int(m.sum()):,}")
    print(f"  約定時刻からの markout {np.corrcoef(Af, Uf)[0,1]:+.4f}  n={Af.size:,}")
    print(f"  往復損益              {np.corrcoef(Af, pnl)[0,1]:+.4f}  n={pnl.size:,}")
    print()
    print(f"書き出し {DATA}/cand2_stages_{tag}{tail}.csv")


if __name__ == "__main__":
    main()
