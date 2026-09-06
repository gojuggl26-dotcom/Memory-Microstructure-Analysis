"""三つの候補を同じ物差しに載せてまとめる。

    uv run python scripts/build_candidates.py --coin xyz:MU

三候補は**測っている単位が違う**。ここでは可能な限り
「往復損益(1 組あたり bp)」という同じ軸へ寄せ、寄せられないものは
何の単位で測られているかを明記して並べる。

  候補 1  選択的・在庫型 MM(キュー優先順位)
          → 往復損益で完結して測れる唯一の候補
  候補 2  sweep インパクトの非対称 + 観測済み補充
          → 元の証拠は「5 秒先の価格との相関」。ここで往復損益へ翻訳する
  候補 3  実参加者の約定品質
          → 元の証拠は「口座別の 10 秒逆選択」。ここでは候補 1 の中心的な
            主張(順番に価値がある)を実参加者のデータで照合するのに使う

出力: data/candidates_summary.csv / cand2_signal.csv / cand3_queue.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_entrygate import impact_feats  # noqa: E402
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
LADDER = [
    ("基準: 在庫 MM(門なし)", "q1"),
    ("+ 価格改善 1 ティック", "q1_imp1"),
    ("+ 門のみ(改善なし)", "q1_gated"),
    ("★ 候補 1 = 改善 + 門", "q1_imp1_gated"),
    ("候補 1 + 候補 2 の変数", "q1_imp1_gated_impact"),
]
DECOMP = [("基準", "q1"), ("値段だけ改善", "q1_imp1_price"),
          ("行列の先頭", "q1_queue"), ("両方", "q1_imp1")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    S = pl.read_csv(DATA / f"inv_summary_{tag}.csv")
    te = [f.stem.split("=")[1] for f in
          sorted((BULK / tag).glob("dt=*.parquet"))[59:]]

    rows = []
    for nm, c in LADDER + DECOMP:
        d = S.filter(pl.col("cfg") == c)
        if not d.height:
            continue
        rows.append({"group": "ladder" if (nm, c) in LADDER else "decomp",
                     "label": nm, "cfg": c,
                     "lots_day": float(d["lots_day"][0]),
                     "ev": float(d["ev_pair"][0]), "se": float(d["se"][0]),
                     "per_day": float(d["per_day_bp"][0])})
    pl.DataFrame(rows).write_csv(DATA / "candidates_summary.csv")
    print("=== 往復損益(1 組あたり bp・標本外 39 日)===")
    for r in rows:
        print(f"  [{r['group']:6s}] {r['label']:26s} {r['ev']:+.4f} ± {r['se']:.4f}"
              f"  建玉/日 {r['lots_day']:>8,.0f}  1 日 {r['per_day']:>+9,.0f} bp")

    # ---- 候補 2: 価格は当てるが、往復損益は当てない ----
    ca, cr = [], []
    for dt in te[::3]:
        I = (pl.scan_parquet(DATA / f"impact_{tag}.parquet")
             .filter(pl.col("dt") == dt)
             .select("ts", "imp_b_q5", "imp_a_q5").collect().sort("ts"))
        if I.height < 100:
            continue
        d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                      .filter(pl.col("dt") == dt).collect())[0].sort("ts")
        bt = d["ts"].cast(pl.Int64).to_numpy()
        mid = 0.5 * (d["best_bid"].to_numpy() + d["best_ask"].to_numpy())
        it = I["ts"].cast(pl.Int64).to_numpy()
        A = (I["imp_b_q5"].to_numpy() - I["imp_a_q5"].to_numpy()).astype(float)
        j0 = np.clip(np.searchsorted(bt, it, "right") - 1, 0, bt.size - 1)
        jh = np.clip(np.searchsorted(bt, it + 5 * 10 ** 9, "right") - 1,
                     0, bt.size - 1)
        r = np.log(mid[jh] / mid[j0]) * 1e4
        v = np.isfinite(A) & np.isfinite(r)
        ca.append(A[v])
        cr.append(r[v])
    A5 = np.concatenate(ca)
    R5 = np.concatenate(cr)
    c_price = float(np.corrcoef(A5, R5)[0, 1])

    P = pl.read_parquet(DATA / f"inv_posts_{tag}_q1_imp1.parquet").filter(
        pl.col("dt").is_in(te))
    xs, ps = [], []
    for dt in te:
        G = P.filter(pl.col("dt") == dt)
        if not G.height:
            continue
        X = impact_feats(tag, dt, G["t"].to_numpy(),
                         G["side"].to_numpy().astype(float))
        xs.append(X[:, 1])
        ps.append(G["rt_pnl"].to_numpy())
    x = np.concatenate(xs)
    pn = np.concatenate(ps)
    fin = np.isfinite(pn)
    c_pnl = float(np.corrcoef(x[fin], pn[fin])[0, 1])
    e = np.quantile(x, [.2, .4, .6, .8])
    k = np.searchsorted(e, x, "right")
    q = [float(pn[fin & (k == i)].mean()) for i in range(5)]
    pl.DataFrame({"metric": ["A と 5 秒先リターン", "A と往復損益"],
                  "corr": [c_price, c_pnl],
                  "n": [A5.size, int(fin.sum())]}).write_csv(
        DATA / "cand2_signal.csv")
    pl.DataFrame({"q": list(range(1, 6)), "ev": q}).write_csv(
        DATA / "cand2_quintile.csv")
    print("\n=== 候補 2: 同じ信号を 2 つの物差しで ===")
    print(f"  A と 5 秒先リターンの相関   {c_price:+.4f}  (n={A5.size:,}、既報 −0.0699)")
    print(f"  A と往復損益の相関          {c_pnl:+.4f}  (n={int(fin.sum()):,})")
    print("  往復損益の五分位: " + " ".join(f"{v:+.3f}" for v in q))

    # ---- 候補 3: 実参加者で「順番」を照合 ----
    F = (pl.scan_parquet(DATA / f"wfill_fills_{tag}.parquet")
         .filter(pl.col("dt").is_in(te))
         .select("qa_place", "adv1", "adv10", "is_full", "ttf_ns").collect())
    qp = F["qa_place"].to_numpy().astype(float)
    a10 = F["adv10"].to_numpy().astype(float)
    v = np.isfinite(qp) & np.isfinite(a10)
    qp, a10 = qp[v], a10[v]
    tt = F["ttf_ns"].to_numpy().astype(float)[v] / 1e9
    edges = [1e-9, 1, 5, 20, 100, 300, np.inf]
    lab = ["0(先頭)", "0〜1", "1〜5", "5〜20", "20〜100", "100〜300", "300 超"]
    kk = np.digitize(qp, edges[:-1], right=True)
    out = []
    for i in range(len(lab)):
        m = kk == i
        if m.sum() < 200:
            continue
        out.append({"bin": lab[i], "n": int(m.sum()), "share": float(m.mean()),
                    "adv10": float(a10[m].mean()),
                    "ttf_med": float(np.median(tt[m]))})
    pl.DataFrame(out).write_csv(DATA / "cand3_queue.csv")
    print("\n=== 候補 3: 実参加者の発注時の待ち行列と 10 秒逆選択 ===")
    for r in out:
        print(f"  {r['bin']:>10s} n {r['n']:>9,} ({100*r['share']:5.1f}%)  "
              f"adv10 {r['adv10']:+.3f} bp  約定まで {r['ttf_med']:.2f}s")
    f0 = qp <= 0
    print(f"  先頭 {100*f0.mean():.1f}% は {a10[f0].mean():+.3f} bp、"
          f"それ以外は {a10[~f0].mean():+.3f} bp  差 "
          f"{a10[~f0].mean() - a10[f0].mean():+.3f} bp")
    print(f"\n書き出し {DATA}/candidates_summary.csv ほか")


if __name__ == "__main__":
    main()
