r"""会場間乖離 d の検定 — 時間帯・遅延感度・日クラスタ bootstrap。

    uv run python scripts/xv_analyze.py --coin xyz:MU
    uv run python scripts/xv_analyze.py --coin xyz:MU --thr 2.0

出力: data/xv_test_<coin>.csv / 標準出力の表

=============================================================================
★判定の設計(自己精査 B)
=============================================================================
対照を 0 に取らない。3 つに分けて (c) で判定する:

  (a) 何もしない                 … 発注しない。往復損益は定義上 0、組数も 0
  (b) 門を通した後の 1 組あたり   … これが正でも、元が黒字なら勝手に通る
  (c) **門あり − 門なし**         … これが施策の正味

さらに**単価と総額の両方**を出す。単価だけ見ると「引きまくる方策」が
必ず勝つ(発注をほとんど止めれば 1 組あたりは良くなる)。

★標準誤差は**日でクラスタ**する。1 日の中の組は強く相関しているので、
  組を独立として扱うと誤差が 1 桁小さく出る。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SESS = ["00-08 夜", "08-13.5 プレ", "13.5-20 現物", "20-24 アフター"]
RNG = np.random.default_rng(20260914)
NB = 4000


def boot_day(days, vals, cnts=None, nb=NB):
    """日をブロックとして復元抽出。組数で重みづけた平均(= 総額の平均)。"""
    u = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in u}
    out = np.empty(nb)
    for b in range(nb):
        pick = RNG.choice(u, size=u.size, replace=True)
        sel = np.concatenate([idx[d] for d in pick])
        out[b] = np.average(vals[sel], weights=None if cnts is None
                            else cnts[sel])
    return out


def summarize(S, thr, label):
    """門(side·d > thr)の有無で、1 組あたりと日次総額を比べる。"""
    S = S.with_columns((pl.col("side") * pl.col("d_bp")).alias("sd"))
    F = S.filter(pl.col("rt_pnl").is_not_nan() & pl.col("d_bp").is_not_nan())
    if F.height < 1000:
        return None
    g = (F.group_by("dt").agg([
        pl.col("rt_pnl").mean().alias("all_mean"),
        pl.len().alias("all_n"),
        pl.col("rt_pnl").sum().alias("all_sum"),
        pl.col("rt_pnl").filter(pl.col("sd") > thr).mean().alias("g_mean"),
        pl.col("rt_pnl").filter(pl.col("sd") > thr).len().alias("g_n"),
        pl.col("rt_pnl").filter(pl.col("sd") > thr).sum().alias("g_sum"),
    ]).sort("dt"))
    d = g["dt"].to_numpy()
    an, gn = g["all_n"].to_numpy().astype(float), g["g_n"].to_numpy().astype(float)
    am, gm = g["all_mean"].to_numpy(), g["g_mean"].to_numpy()
    as_, gs = g["all_sum"].to_numpy(), g["g_sum"].to_numpy()
    ok = np.isfinite(gm) & (gn > 0)
    # (b) 門あり 1 組あたり / (c) 門あり − 門なし
    b_unit = boot_day(d[ok], gm[ok], gn[ok])
    c_unit = boot_day(d[ok], (gm - am)[ok], gn[ok])
    # 総額は「日次合計」の日平均。組数の違いがそのまま入る
    b_tot = boot_day(d[ok], gs[ok])
    a_tot = boot_day(d[ok], as_[ok])
    return {
        "label": label, "thr": thr, "days": int(ok.sum()),
        "n_pair_all": int(an.sum()), "n_pair_gate": int(gn.sum()),
        "keep_pct": 100 * gn.sum() / max(an.sum(), 1),
        "unit_all": float(np.average(am[ok], weights=an[ok])),
        "unit_gate": float(np.average(gm[ok], weights=gn[ok])),
        "unit_gate_lo": float(np.quantile(b_unit, 0.025)),
        "unit_gate_hi": float(np.quantile(b_unit, 0.975)),
        "unit_diff": float(np.mean(c_unit)),
        "unit_diff_lo": float(np.quantile(c_unit, 0.025)),
        "unit_diff_hi": float(np.quantile(c_unit, 0.975)),
        "tot_all_day": float(np.mean(a_tot)),
        "tot_gate_day": float(np.mean(b_tot)),
        "tot_gate_lo": float(np.quantile(b_tot, 0.025)),
        "tot_gate_hi": float(np.quantile(b_tot, 0.975)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--thr", type=float, default=2.0, help="門の閾値 (bp)")
    ap.add_argument("--src", default="_q1_d",
                    help="発注候補の出どころ。_q1_d か _q1_imp1_d")
    ap.add_argument("--eval-days", type=int, default=39,
                    help="末尾 N 日だけで判定(門の学習期間を外す)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    fs = sorted(glob.glob(str(DATA / f"xv_sig_{tag}_*_d*.parquet")))
    fs = [f for f in fs if a.src in os.path.basename(f)]
    print(f"見つかった信号ファイル {len(fs)} 本\n")

    # ---- 1. 遅延感度とプラセボ ----
    rows = []
    print("【1】遅延 Δ ごとの十分位勾配と、門(side·d > "
          f"{a.thr:g}bp)の効果")
    print(f"{'版':>16s}{'D10−D1':>9s}{'門なし':>9s}{'門あり':>9s}"
          f"{'差(c)':>9s}{'95% 区間':>20s}{'残す%':>8s}{'日次総額':>10s}")
    for f in fs:
        m = re.search(r"_d(\d+)(_shift(\d+)s)?\.parquet$", f)
        dly, sh = int(m.group(1)), (int(m.group(3)) if m.group(3) else 0)
        lab = (f"Δ={dly}ms" if not sh else f"プラセボ {sh//3600}h")
        S = pl.read_parquet(f)
        r = summarize(S, a.thr, lab)
        if r is None:
            continue
        # 十分位勾配(全期間の分位で切る。記述用)
        X = S.with_columns((pl.col("side") * pl.col("d_bp")).alias("sd"))
        X = X.filter(pl.col("rt_pnl").is_not_nan() & pl.col("sd").is_not_nan())
        q = np.nanquantile(X["sd"].to_numpy(), np.arange(1, 10) / 10)
        dec = np.digitize(X["sd"].to_numpy(), q)
        rt = X["rt_pnl"].to_numpy()
        g10 = float(np.mean(rt[dec == 9]))
        g1 = float(np.mean(rt[dec == 0]))
        r["grad"] = g10 - g1
        r["delay_ms"], r["shift_s"] = dly, sh
        rows.append(r)
        print(f"{lab:>16s}{r['grad']:>9.3f}{r['unit_all']:>9.3f}"
              f"{r['unit_gate']:>9.3f}{r['unit_diff']:>9.3f}"
              f"  [{r['unit_diff_lo']:+.3f},{r['unit_diff_hi']:+.3f}]"
              f"{r['keep_pct']:>8.1f}{r['tot_gate_day']:>10.1f}")
    if rows:
        pl.DataFrame(rows).write_csv(DATA / f"xv_test_{tag}.csv")

    # ---- 2. 時間帯別(既定の Δ=130ms)----
    f0 = str(DATA / f"xv_sig_{tag}{a.src.rstrip('d').rstrip('_')}_d130.parquet")
    if os.path.exists(f0):
        S = pl.read_parquet(f0).with_columns(
            (pl.col("side") * pl.col("d_bp")).alias("sd"))
        print(f"\n【2】時間帯別(Δ=130ms、門 side·d > {a.thr:g}bp)")
        print(f"{'時間帯':>16s}{'組(門なし)':>12s}{'門なし':>9s}"
              f"{'門あり':>9s}{'差(c)':>9s}{'95% 区間':>20s}{'残す%':>8s}")
        for i, nm in enumerate(SESS):
            r = summarize(S.filter(pl.col("sess") == i), a.thr, nm)
            if r is None:
                continue
            print(f"{nm:>16s}{r['n_pair_all']:>12,}{r['unit_all']:>9.3f}"
                  f"{r['unit_gate']:>9.3f}{r['unit_diff']:>9.3f}"
                  f"  [{r['unit_diff_lo']:+.3f},{r['unit_diff_hi']:+.3f}]"
                  f"{r['keep_pct']:>8.1f}")

        # ---- 3. 閾値の格子(多重比較の規模を明示)----
        print(f"\n【3】閾値の格子(Δ=130ms)。**{len([0.5,1,2,3,5,8])} 通り**を全部出す")
        print(f"{'閾値 bp':>8s}{'組':>10s}{'残す%':>8s}{'門あり':>9s}"
              f"{'差(c)':>9s}{'95% 区間':>20s}{'日次総額':>10s}")
        for t in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
            r = summarize(S, t, f"thr{t}")
            if r is None:
                continue
            print(f"{t:>8.1f}{r['n_pair_gate']:>10,}{r['keep_pct']:>8.1f}"
                  f"{r['unit_gate']:>9.3f}{r['unit_diff']:>9.3f}"
                  f"  [{r['unit_diff_lo']:+.3f},{r['unit_diff_hi']:+.3f}]"
                  f"{r['tot_gate_day']:>10.1f}")
        print("\n★★『何もしない』の日次総額は定義上 0。上の『日次総額』が負なら、"
              "\n   その門は**発注しないほうがまし**である(自己精査 B5/B6)。")


if __name__ == "__main__":
    main()
