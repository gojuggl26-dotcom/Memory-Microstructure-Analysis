"""判定を日次損益に置き換える(単価ではなく総額で検定する)。

    uv run python scripts/build_dailypnl.py --coin xyz:MU

単価(1 組あたり)は日次等重みの平均、総額は取引ごとの bp の合計なので、
片方の有意性から他方は導けない。ここでは**日次総額の系列そのもの**を
Newey-West と巡回移動ブロック bootstrap で検定し、同じ日で対にした差も出す。
最後にドル換算し、必要な数量が最良気配に対してどれだけかを示す。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
NW_LAGS, BL, B = 14, 7, 10000
CFG = [
    ("基準(門なし)", "q1"),
    ("+ 価格改善", "q1_imp1"),
    ("門のみ(BBO × G0)", "q1_gated"),
    ("BBO × G1", "q1_gatedby_q1_imp1"),
    ("改善 × G0", "q1_imp1_gatedby_q1"),
    ("★ 候補 1 = 改善 × G1", "q1_imp1_gatedby_q1_imp1"),
    ("候補 1 + 候補 2 の変数", "q1_imp1_gated_impact"),
    ("候補 1 + 遅延 65ms", "q1_lat65_imp1_gated"),
    ("候補 1 + 遅延 130ms", "q1_lat130_imp1_gated"),
]
PAIRS = [("置き方だけ変更(門 G0)", "q1_imp1_gatedby_q1", "q1_gated"),
         ("置き方だけ変更(門 G1)", "q1_imp1_gatedby_q1_imp1",
          "q1_gatedby_q1_imp1"),
         ("門だけ学習し直す(BBO)", "q1_gatedby_q1_imp1", "q1_gated"),
         ("両方変更(候補 1 の全効果)", "q1_imp1_gatedby_q1_imp1", "q1_gated"),
         ("候補 1 − (候補 1+候補 2)", "q1_imp1_gatedby_q1_imp1",
          "q1_imp1_gated_impact"),
         ("遅延 0 − 遅延 65ms", "q1_imp1_gatedby_q1_imp1",
          "q1_lat65_imp1_gated")]


# --- 条件つきの出し直し(候補 1 手順②)---
# 格子は 13 通り。全件を報告し、Bonferroni の閾値も併記する。
B_ALW = "q1_imp1_gatedby_q1_imp1"
RQ = ([("★ always(従来・毎回出し直す)", B_ALW)]
      + [(f"cond km={km:g}s kd={kd if kd else '∞'}",
          f"q1_te_rqcondkm{km:g}kd{kd}_imp1_gatedby_q1_imp1"
          if kd else f"q1_te_rqcondkm{km:g}_imp1_gatedby_q1_imp1")
         for km in (1, 5, 60) for kd in (1, 2, 5, 0)]
      + [("keep km=60s(上界・幽霊注文)",
          "q1_te_rqkeepkm60_imp1_gatedby_q1_imp1")])
RQ_PAIRS = [(f"always − ({lab})", B_ALW, cfg) for lab, cfg in RQ[1:]]


def nw(v):
    n = v.size
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, NW_LAGS + 1):
        g += 2 * (1 - lg / (NW_LAGS + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n))


def boot(v, seed=7):
    rng = np.random.default_rng(seed)
    n = v.size
    k = int(np.ceil(n / BL))
    o = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, k)
        idx = (s[:, None] + np.arange(BL)[None, :]).ravel() % n
        o[b] = v[idx[:n]].mean()
    return float(np.quantile(o, 0.025)), float(np.quantile(o, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--set", choices=["main", "requote"], default="main")
    a = ap.parse_args()
    cfgs, pairs, out = ((CFG, PAIRS, "") if a.set == "main"
                        else (RQ, RQ_PAIRS, "rq_"))
    tag = a.coin.replace(":", "_")
    te = [f.stem.split("=")[1] for f in sorted((BULK / tag).glob("dt=*.parquet"))[59:]]

    def ser(cfg):
        p = DATA / f"inv_days_{tag}_{cfg}.csv"
        if not p.exists():
            return None
        D = pl.read_csv(p).filter(pl.col("dt").is_in(te)).sort("dt")
        return (D["dt"].to_list(), D["total_bp"].to_numpy().astype(float),
                float(D["n_pair"].sum()) / D.height)

    rows, S = [], {}
    for lab, cfg in cfgs:
        r = ser(cfg)
        if r is None:
            continue
        S[cfg] = dict(zip(r[0], r[1]))
        m, se = nw(r[1])
        lo, hi = boot(r[1])
        rows.append({"label": lab, "cfg": cfg, "pairs_day": r[2], "daily": m,
                     "se": se, "t": m / se, "lo": lo, "hi": hi,
                     "pos_days": int((r[1] > 0).sum()), "days": r[1].size})
    pl.DataFrame(rows).write_csv(DATA / f"dailypnl_{out}{tag}.csv")
    print("★ 日次総額 bp(標本外 39 日・1 建玉 = 1 単位)")
    for r in rows:
        print(f"  {r['label']:26s} 組/日 {r['pairs_day']:>8,.0f}  "
              f"{r['daily']:>+11.2f} ± {r['se']:>8.2f} (t={r['t']:+5.2f})  "
              f"95% [{r['lo']:+10.2f}, {r['hi']:+10.2f}]  正 {r['pos_days']}/{r['days']}")

    pr = []
    print("\n★ 同じ日で対にした差")
    for lab, x, y in pairs:
        if x not in S or y not in S:
            continue
        ks = sorted(set(S[x]) & set(S[y]))
        d = np.array([S[x][k] - S[y][k] for k in ks])
        m, se = nw(d)
        lo, hi = boot(d, seed=11)
        pr.append({"label": lab, "diff": m, "se": se, "t": m / se, "lo": lo,
                   "hi": hi, "pos": int((d > 0).sum()), "n": d.size})
        print(f"  {lab:30s} {m:>+9.2f} ± {se:>6.2f} (t={m/se:+5.2f})  "
              f"95% [{lo:+8.2f}, {hi:+8.2f}]  正 {int((d>0).sum())}/{d.size}")
    pl.DataFrame(pr).write_csv(DATA / f"dailypnl_pairs_{out}{tag}.csv")

    if a.set != "main":
        # 格子探索なので Bonferroni の閾値を併記する(自己精査 C10)
        k = len(pairs)
        z = float(norm.ppf(0.025 / k))
        print()
        print(f"格子 {k} 通り。Bonferroni(両側 5%)の |t| 閾値 = "
              f"{abs(z):.3f}")
        print(f"書き出し {DATA}/dailypnl_{out}{tag}.csv ほか")
        return
    # ---- ドル換算 ----
    mids, deps = [], []
    for dt in te[::5]:
        d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                      .filter(pl.col("dt") == dt).collect())[0]
        mids.append(np.median(0.5 * (d["best_bid"].to_numpy()
                                     + d["best_ask"].to_numpy())))
        deps.append(np.median(0.5 * (d["bid_sz"].to_numpy()
                                     + d["ask_sz"].to_numpy())))
    mid, dep = float(np.median(mids)), float(np.median(deps))
    base = [r for r in rows if r["cfg"] == "q1_imp1_gatedby_q1_imp1"][0]["daily"]
    sz = []
    for s in (0.001, 0.01, 0.1, 0.5, 1.0, 100 / (base / 1e4 * mid)):
        sz.append({"size": s, "notional": s * mid, "usd": base / 1e4 * s * mid,
                   "share_of_touch": s / dep})
    pl.DataFrame(sz).write_csv(DATA / f"dailypnl_size_{tag}.csv")
    print(f"\n★ ドル換算(中央 mid ${mid:,.2f} / 最良気配の中央 {dep:.3f} 枚)")
    for r in sz:
        print(f"  {r['size']:>7.3f} 枚 名目 ${r['notional']:>9,.2f}  "
              f"日次 ${r['usd']:>9,.2f}  最良気配の {100*r['share_of_touch']:>6.1f}%")
    print(f"\n書き出し {DATA}/dailypnl_{tag}.csv ほか")


if __name__ == "__main__":
    main()
