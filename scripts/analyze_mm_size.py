"""執行できる数量 — メイカーの取り分を金額で出す(7 銘柄)。

    uv run python scripts/analyze_mm_size.py

入力: data/inv_days_xyz_<coin>_q1_sz*.csv / 同 _gatedby_mmg / 同 _lat130
      data/inv_lots_xyz_<coin>_q1_sz*.parquet(金額換算に使う m_in)
      data/mmgate_fit_xyz_<coin>.csv(評価期間の日付)
出力: data/mm_size_curve.csv   銘柄 × 設定 × 数量
      data/mm_size_peak.csv    設定ごとの最大値と、そのときの数量

## なぜ数量が本体なのか

幽霊注文(数量 0)の 1 組あたり bp は**上界**でしかない。板の最後尾に無限小を
置いて「前の行列がはけた瞬間に埋まる」としているので、
**約定するのは必ず「前の行列を食い尽くすほど大きな流れが来たとき」**である。
数量を入れると、その流れがさらに自分のぶんだけ大きくなければ埋まらない。
つまり**残る約定はより攻撃的な(=逆選択の強い)ものに偏る**。

## 2 つの約定モデル

* **全量約定(all-or-nothing)**: `--size S` の実装。前の行列 + S が流れて
  初めて S 全量が約定したとみなす。**保守側**(部分約定を捨てている)。
* **部分約定**: 自分の注文の j 番目の単位は「前の行列 + j」が流れれば埋まる。
  よって数量 S の注文の 1 日あたり損益は

  ```math
  \\mathrm{PnL}(S)=\\int_0^{S} g(j)\\,dj,\\qquad
  g(j)=\\sum_{\\text{要求 }Q_0+j\\text{ で埋まった組}}\\frac{\\mathrm{pnl}_{bp}}{10^4}\\,\\mathrm{mid}
  ```

  `g` は掃引した数量の点でしか測っていないので台形則で積む。**上側の見積り**。

真の値はこの 2 つの間にある。両方を出して、**どちらでも黒字か**を見る。

## 時間契約

数量は発注時点で決める量なので先読みは無い。門を使う設定は、門の
**評価期間(後ろ 40% の日)だけ**で集計する。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
CFG = [("門なし", "", 0), ("門なし+130ms", "", 130),
       ("門あり", "_gatedby_mmg", 0), ("門あり+130ms", "_gatedby_mmg", 130)]


def ev_days(tag: str, base: pl.DataFrame) -> set:
    f = DATA / f"mmgate_fit_{tag}.csv"
    if f.exists():
        return set(pl.read_csv(f).filter(pl.col("train") == 0)["dt"].to_list())
    d = base["dt"].to_list()
    return set(d[int(round(len(d) * 0.60)):])


def one(tag: str, sfx: str, ev: set,
        mid: float = float("nan")):
    """(相殺ぶんの 1 枚あたり日次ドル, 組/日, 1 組 bp, 強制決済ぶん, 強制/日)。"""
    f = DATA / f"inv_days_{tag}{sfx}.csv"
    lp = DATA / f"inv_lots_{tag}{sfx}.parquet"
    if not f.exists() or not lp.exists():
        return None
    E = pl.read_csv(f).filter(pl.col("dt").is_in(list(ev)))
    if E.height < 5:
        return None
    L = pl.read_parquet(lp).filter(pl.col("dt").is_in(list(ev)))
    # 古い実行の inv_lots には m_in が無い。その場合は中央 mid で代用する
    # (bp は建玉時の mid で割ってあるので近似になる)。
    mm = (L["m_in"].to_numpy() if "m_in" in L.columns
          else np.full(L.height, mid))
    usd = L["pnl"].to_numpy() / 1e4 * mm
    fc = L["forced"].to_numpy() > 0
    # ★ 強制決済(日の終わりに残った在庫をテイカーで畳む)は**メイカーの
    #   取り分ではない**。数量を上げるほど約定が減り、在庫が 1 日居座って
    #   強制決済されるので、この成分が損益を支配しうる。実測でも
    #   xyz:AMD の $100,000 では 21 件の強制決済(1 件 +110bp)が
    #   161 組の相殺(1 組 −0.81bp)を打ち消して符号を反転させていた。
    #   これは値動きの賭けであってメイカーではないので、必ず分けて出す。
    g = float(usd[~fc].sum()) / E.height
    gf = float(usd[fc].sum()) / E.height
    n = E["n_pair"].to_numpy().astype(float)
    bp = float(np.nansum(E["pair_pnl_mean"].to_numpy() * n) / max(n.sum(), 1))
    # 日ごとの 1 枚あたりドル(相殺ぶんのみ)。有意性の判定に使う
    day = (pl.DataFrame({"dt": L["dt"], "u": usd, "fc": fc})
           .filter(~pl.col("fc")).group_by("dt").agg(u=pl.col("u").sum()))
    mp = {d: v for d, v in zip(day["dt"], day["u"])}
    ser = np.array([mp.get(d, 0.0) for d in E["dt"].to_list()])
    return g, float(n.mean()), bp, gf, int(fc.sum()) / E.height, ser


def main() -> None:
    rows, peaks = [], []
    for c in COINS:
        tag = f"xyz_{c}"
        base = pl.read_csv(DATA / f"inv_days_{tag}_q1.csv")
        ev = ev_days(tag, base)
        mid = float((pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                     .select(((pl.col("best_bid") + pl.col("best_ask")) / 2)
                             .alias("m")).collect()["m"]).median())
        for lab, gsfx, lat in CFG:
            lsfx = f"_lat{lat}" if lat else ""
            z = one(tag, f"_q1{lsfx}{gsfx}", ev, mid)   # 数量 0(幽霊)
            if z is None:
                continue
            pts = [(0.0,) + tuple(z)]
            for f in sorted(DATA.glob(f"inv_days_{tag}_q1_sz*{gsfx}.csv")):
                st = f.stem
                if bool(re.search(r"_lat\d+", st)) != bool(lat):
                    continue
                if lat and f"_lat{lat}" not in st:
                    continue
                if gsfx and not st.endswith(gsfx):
                    continue
                if not gsfx and "gatedby" in st:
                    continue
                S = float(re.search(r"_sz([0-9.]+)", st).group(1))
                r = one(tag, st.replace(f"inv_days_{tag}", ""), ev, mid)
                if r:
                    pts.append((S,) + tuple(r))
            pts.sort()
            if len(pts) < 2:
                continue
            xs = np.array([p[0] for p in pts])
            gs = np.array([p[1] for p in pts])
            cum = np.concatenate(
                [[0.0], np.cumsum(np.diff(xs) * (gs[1:] + gs[:-1]) / 2)])
            for i, (S, g, npair, bp, gf, nfc, ser) in enumerate(pts):
                v = ser * S
                nd_ = v.size
                se = (float(np.std(v, ddof=1) / np.sqrt(nd_))
                      if nd_ > 2 else np.nan)
                ps = (float(stats.binomtest(int((v > 0).sum()), nd_, 0.5).pvalue)
                      if nd_ > 2 else np.nan)
                rows.append({"coin": f"xyz:{c}", "cfg": lab, "size_unit": S,
                             "size_usd": S * mid, "pairs_day": npair,
                             "pair_bp": bp, "aon_usd_day": g * S,
                             "aon_se_day": se,
                             "aon_t": (g * S) / se if se and se > 0 else np.nan,
                             "p_sign": ps, "p_day_pos": float((v > 0).mean()),
                             "worst_day_usd": float(v.min()) if nd_ else np.nan,
                             "forced_usd_day": gf * S, "forced_day": nfc,
                             "partial_usd_day": cum[i],
                             "partial_usd_year": cum[i] * 252,
                             "n_days": len(ev)})
            k = int(np.argmax(cum))
            ka = int(np.argmax([p[1] * p[0] for p in pts]))
            vb = pts[ka][6] * xs[ka]
            seb = (float(np.std(vb, ddof=1) / np.sqrt(vb.size))
                   if vb.size > 2 else np.nan)
            peaks.append({"coin": f"xyz:{c}", "cfg": lab,
                          "best_size_usd": xs[k] * mid,
                          "best_partial_usd_day": cum[k],
                          "best_partial_usd_year": cum[k] * 252,
                          "best_aon_size_usd": xs[ka] * mid,
                          "best_aon_usd_day": pts[ka][1] * pts[ka][0],
                          "best_aon_usd_year": pts[ka][1] * pts[ka][0] * 252,
                          "best_aon_t": ((pts[ka][1] * pts[ka][0]) / seb
                                         if seb and seb > 0 else np.nan),
                          "best_aon_p_sign": (
                              float(stats.binomtest(int((vb > 0).sum()),
                                                    vb.size, 0.5).pvalue)
                              if vb.size > 2 else np.nan),
                          "n_days": len(ev)})
    R = pl.DataFrame(rows)
    R.write_csv(DATA / "mm_size_curve.csv")
    P = pl.DataFrame(peaks)
    P.write_csv(DATA / "mm_size_peak.csv")

    print("===== 数量ごとの取り分(評価期間・門ありは門の評価期間)=====")
    for c in COINS:
        q = R.filter(pl.col("coin") == f"xyz:{c}")
        if not q.height:
            continue
        print(f"\n  xyz:{c}")
        print(f"    {'設定':<14}{'数量':>11}{'組/日':>8}{'1 組 bp':>9}"
              f"{'全量 $/日':>11}{'部分 $/日':>11}{'部分 $/年':>12}"
              f"{'強制 $/日':>11}{'強制/日':>8}")
        for lab, _, _ in CFG:
            for r in q.filter(pl.col("cfg") == lab).sort("size_usd").iter_rows(
                    named=True):
                nm = "幽霊(0)" if r["size_unit"] == 0 else f"${r['size_usd']:,.0f}"
                print(f"    {lab:<14}{nm:>11}{r['pairs_day']:>8,.0f}"
                      f"{r['pair_bp']:>9.2f}{r['aon_usd_day']:>11,.1f}"
                      f"{r['partial_usd_day']:>11,.1f}"
                      f"{r['partial_usd_year']:>12,.0f}"
                      f"{r['forced_usd_day']:>11,.1f}{r['forced_day']:>8.1f}")

    print("\n===== 最良の数量と、そのときの年間の取り分 =====")
    print(f"{'銘柄':<12}{'設定':<14}{'最良数量':>11}{'部分 $/年':>12}"
          f"{'全量 最良数量':>13}{'全量 $/年':>12}{'t':>7}{'符号 p':>9}{'日数':>6}")
    for r in P.sort("best_partial_usd_year", descending=True).iter_rows(named=True):
        print(f"{r['coin']:<12}{r['cfg']:<14}${r['best_size_usd']:>10,.0f}"
              f"{r['best_partial_usd_year']:>12,.0f}"
              f"${r['best_aon_size_usd']:>12,.0f}"
              f"{r['best_aon_usd_year']:>12,.0f}{r['best_aon_t']:>7.2f}"
              f"{r['best_aon_p_sign']:>9.2g}{r['n_days']:>6}")
    print("\n★ 0 は「クオートを出さない」。年間の取り分が 0 に届かない設定は、")
    print("  何もしないほうが良いということ。設備・回線・資金の費用は含めていない。")


if __name__ == "__main__":
    main()
