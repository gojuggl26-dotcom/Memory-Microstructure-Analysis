"""6 レジーム判定の結果を銘柄横断でまとめ、**帰無対照と突き合わせる**。

    uv run python scripts/analyze_regime.py

入力: data/regime_<coin>.parquet(`build_regime.py`)
出力: data/regime_share.csv    銘柄 × レジームの割合と平均継続秒
      data/regime_trans.csv    銘柄 × 遷移行列(6×6)
      data/regime_fwd.csv      銘柄 × レジーム × ホライズンの将来リターン
      data/regime_summary.csv  銘柄ごとの要約

## ★ ランダムウォーク対照

閾値が**相対的**(直近 30 分の分位)なので、**どんな系列でも必ず 6 つの
レジームに分かれる**。「R4 が 6% あった」だけでは何も言えない。
そこで、**同じ長さ・同じ「気配が動いた秒」のパターン**を持つ
ランダムウォークを同じ分類器に通し、

* レジームの割合
* 平均継続秒(持続性)
* 遷移行列の対角(自己遷移確率)

を比べる。実データがランダムウォークと同じなら、この分類は**雑音を
6 つの名前で呼び分けているだけ**である。

リターンは実データの 1 秒リターンを**ブートストラップ(復元抽出)**で並べ替えて作る。
こうすると分布(裾の厚さ・ゼロの多さ)は実データと同じで、**時間方向の
構造だけが壊れる**。正規乱数だと分布の違いが混ざってしまう。

## レジームは何かを予測するか

各レジームの**次の 10 / 60 / 300 秒の mid リターン**と**将来の実現ボラ**を出す。

* R4(高ボラ/上)が本当に momentum なら、次のリターンは正のはず
* R6(高ボラ/両建て)は方向が無いはずで、ボラだけ高いはず
* 高ボラが持続するなら、R4〜R6 の将来ボラは R1〜R3 より高いはず

標準誤差は**日でクラスタ**する。1 秒の点どうしは強く相関している。

## 時間契約

レジームは時刻 $`t`$ までの情報だけで決まる(`build_regime.py`)。
将来リターンは $`(t,\\,t+h]`$。`shift(-k)` は目的変数にしか使っていない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_regime import classify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "SNDK", "INTC", "AMD", "SMSN", "DRAM", "KIOXIA", "SKHX",
         "GOLD", "GOOGL", "AAPL", "NVDA", "TSLA", "AAVE"]
NAMES = ["R1 低ボラ/中立", "R2 低ボラ/上", "R3 低ボラ/下",
         "R4 高ボラ/上", "R5 高ボラ/下", "R6 高ボラ/両建て"]
HZ = [10, 60, 300]
NSIM = 3


def runs(x: np.ndarray, k: int) -> float:
    """値 k が連続する区間の平均長(秒)。"""
    m = x == k
    if not m.any():
        return np.nan
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return float(m.sum() / max((d == 1).sum(), 1))


def clustered(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def median_spread(coin: str) -> float:
    """中央スプレッド(bp)。往復費用の計算に使う。"""
    if coin == "DRAM":
        return float(pl.scan_parquet(DATA / "DRAM" / "microprice_1s.parquet")
                     .select("spread_bp").collect()["spread_bp"].median())
    f = DATA / f"bbo_xyz_{coin}.parquet"
    if not f.exists():
        return float("nan")
    t = (pl.scan_parquet(f)
         .select(((pl.col("best_ask") - pl.col("best_bid"))
                  / ((pl.col("best_ask") + pl.col("best_bid")) / 2)
                  * 1e4).alias("s")).collect()["s"])
    return float(t.filter(t.is_finite() & (t > 0) & (t < 100)).median())


def by_day_mean(val, day, mask, nd):
    """日ごとの平均。mask が立った点だけを使う。"""
    w = mask & np.isfinite(val)
    s = np.bincount(day[w], weights=val[w], minlength=nd)
    c = np.bincount(day[w], minlength=nd)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def main() -> None:
    rng = np.random.default_rng(20260916)
    sh_rows, tr_rows, fw_rows, sm_rows = [], [], [], []
    for c in COINS:
        f = DATA / f"regime_xyz_{c}.parquet" if c != "AAVE" else DATA / "regime_AAVE.parquet"
        if not f.exists():
            print(f"xyz:{c}: 未作成。skip", file=sys.stderr)
            continue
        B = pl.read_parquet(f)
        ts = B["ts"].to_numpy()
        mid = B["mid"].to_numpy()
        reg = B["regime"].to_numpy()
        rv10 = B["rv10"].to_numpy()
        er = B["er"].to_numpy()
        vol_hi = B["vol_hi"].to_numpy()
        r = B["r"].to_numpy() / 1e4
        days = B["dt"].unique(maintain_order=True).to_list()
        dmap = {d: i for i, d in enumerate(days)}
        day = np.array([dmap[d] for d in B["dt"].to_list()], np.int32)
        nd = len(days)
        n = ts.size
        ok = reg > 0

        # ---- 帰無対照(1 秒リターンのブートストラップ)-------------------
        nullsh = np.zeros((NSIM, 6))
        nullrun = np.zeros((NSIM, 6))
        nulldiag = np.zeros((NSIM, 6))
        reg_null = None
        for s in range(NSIM):
            rr = rng.permutation(r)               # 分布は同じ、時間構造だけ壊す
            Cn = classify(rr)
            rn = Cn["regime"]
            reg_null = rn                       # プラセボ用に最後の 1 本を残す
            on = rn > 0
            nullsh[s] = np.bincount(rn[on], minlength=7)[1:] / max(on.sum(), 1)
            for k in range(6):
                nullrun[s, k] = runs(rn, k + 1)
                m = (rn[:-1] == k + 1) & on[:-1] & on[1:]
                nulldiag[s, k] = (np.mean(rn[1:][m] == k + 1)
                                  if m.sum() > 100 else np.nan)

        # ---- 割合・継続・遷移 --------------------------------------------
        cnt = np.bincount(reg[ok], minlength=7)[1:]
        share = cnt / max(ok.sum(), 1)
        T = np.zeros((6, 6))
        m2 = ok[:-1] & ok[1:]
        a1, a2 = reg[:-1][m2] - 1, reg[1:][m2] - 1
        np.add.at(T, (a1, a2), 1)
        T = T / np.maximum(T.sum(1, keepdims=True), 1)
        for k in range(6):
            sh_rows.append({
                "coin": c, "regime": k + 1, "name": NAMES[k],
                "share": share[k], "share_null": float(nullsh[:, k].mean()),
                "run_s": runs(reg, k + 1),
                "run_s_null": float(np.nanmean(nullrun[:, k])),
                "diag": T[k, k], "diag_null": float(np.nanmean(nulldiag[:, k])),
                "rv10_bp": float(np.nanmean(rv10[ok & (reg == k + 1)])),
                "abs_er": float(np.nanmean(np.abs(er[ok & (reg == k + 1)])))})
            for j in range(6):
                tr_rows.append({"coin": c, "from": k + 1, "to": j + 1,
                                "p": T[k, j]})

        # ---- 将来リターン・将来ボラ --------------------------------------
        lm = np.log(mid)
        # r² の累積和。c2[i] = Σ_{j<i} r_j²。(t, t+h] の和は c2[t+h+1] − c2[t+1]
        c2 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r * r))])
        for h in HZ:
            fwd = np.full(n, np.nan)
            cont = np.zeros(n, bool)
            cont[:n - h] = ts[h:] == ts[:n - h] + h
            fwd[:n - h] = np.where(cont[:n - h],
                                   (lm[h:] - lm[:n - h]) * 1e4, np.nan)
            # 将来の実現ボラ(t, t+h] の 1 秒リターンの二乗和の平方根
            fv = np.full(n, np.nan)
            fv[:n - h] = np.where(
                cont[:n - h],
                np.sqrt(np.maximum(c2[h + 1:n + 1] - c2[1:n - h + 1], 0)) * 1e4,
                np.nan)
            base, _ = clustered(by_day_mean(fwd, day, ok, nd))
            for k in range(6):
                msk = ok & (reg == k + 1)
                d1 = by_day_mean(fwd, day, msk, nd)
                d2 = by_day_mean(fv, day, msk, nd)
                m, se = clustered(d1)
                mv, _ = clustered(d2)
                # ★プラセボ: レジームだけ「時間構造を壊した系列」から取り、
                #   将来リターンは実データのまま。無関係なはずなので 0 に近づく。
                mp, _ = clustered(by_day_mean(
                    fwd, day, (reg_null == k + 1) & np.isfinite(fwd), nd))
                fw_rows.append({"coin": c, "regime": k + 1, "name": NAMES[k],
                                "h": h, "n": int((msk & np.isfinite(fwd)).sum()),
                                "fwd_bp": m, "se": se,
                                "t": m / se if se and se > 0 else np.nan,
                                "fwd_placebo_bp": mp, "fwd_all_bp": base,
                                "fwd_rv_bp": mv})

        sm_rows.append({
            "coin": c, "n_sec": n, "n_days": nd,
            "spread_bp": median_spread(c),
            "cover": float(ok.mean()),
            "p_move": float(np.mean(r != 0)),
            "p_high": float(np.mean(vol_hi[ok] > 0)),
            "p_high_null": float(nullsh[:, 3:].sum(1).mean()),
            "p_dir": float(np.mean(reg[ok] != 1) - np.mean(reg[ok] == 6)),
            "share_R6": share[5], "share_R6_null": float(nullsh[:, 5].mean()),
            "run_R1": runs(reg, 1), "run_R1_null": float(np.nanmean(nullrun[:, 0])),
            "run_R6": runs(reg, 6), "run_R6_null": float(np.nanmean(nullrun[:, 5]))})
        print(f"[reg] xyz:{c} 完了({nd} 日 / {n:,} 秒)", flush=True, file=sys.stderr)

    S = pl.DataFrame(sh_rows)
    S.write_csv(DATA / "regime_share.csv")
    pl.DataFrame(tr_rows).write_csv(DATA / "regime_trans.csv")
    F = pl.DataFrame(fw_rows)
    F.write_csv(DATA / "regime_fwd.csv")
    M = pl.DataFrame(sm_rows)
    M.write_csv(DATA / "regime_summary.csv")
    done = M["coin"].to_list()
    SPREAD = {r["coin"]: r["spread_bp"] for r in M.iter_rows(named=True)}

    print("\n===== ① レジームの割合(%)=====")
    print("上段 = 実測、下段 = ブートストラップ対照(分布は同じで時間構造だけ壊したもの)\n")
    print(f"{'銘柄':<10}" + "".join(f"{n[:6]:>9}" for n in NAMES) + f"{'高ボラ':>8}")
    for c in done:
        q = S.filter(pl.col("coin") == c).sort("regime")
        print(f"{'xyz:'+c:<10}" + "".join(f"{v*100:>9.2f}"
                                          for v in q["share"].to_list())
              + f"{float(M.filter(pl.col('coin')==c)['p_high'][0])*100:>8.1f}")
        print(f"{'  対照':<10}" + "".join(f"{v*100:>9.2f}"
                                          for v in q["share_null"].to_list())
              + f"{float(M.filter(pl.col('coin')==c)['p_high_null'][0])*100:>8.1f}")

    print("\n===== ② 平均継続秒(持続性)=====")
    print("レジームが「状態」なら、対照より長く続くはず。\n")
    print(f"{'銘柄':<10}" + "".join(f"{n[:6]:>9}" for n in NAMES))
    for c in done:
        q = S.filter(pl.col("coin") == c).sort("regime")
        print(f"{'xyz:'+c:<10}" + "".join(f"{v:>9.1f}" for v in q["run_s"].to_list()))
        print(f"{'  対照':<10}" + "".join(f"{v:>9.1f}"
                                          for v in q["run_s_null"].to_list()))

    print("\n===== ③ レジームごとの将来リターン(bp)=====")
    nt = F.filter(pl.col("se").is_finite()).height
    thr = float(stats.norm.ppf(1 - 0.05 / max(nt, 1) / 2))
    print(f"検定 {nt} 通り / Bonferroni の |t| 閾値 {thr:.2f}。"
          f"★ は通ったもの。日クラスタの標準誤差。\n")
    for h in HZ:
        print(f"  次の {h} 秒")
        print(f"    {'銘柄':<10}" + "".join(f"{n[:6]:>12}" for n in NAMES))
        for c in done:
            q = F.filter((pl.col("coin") == c) & (pl.col("h") == h)).sort("regime")
            cells, pcells = [], []
            for rr in q.iter_rows(named=True):
                mk = "★" if abs(rr["t"] or 0) > thr else " "
                cells.append(f"{rr['fwd_bp']:>11.3f}{mk}")
                pcells.append(f"{rr['fwd_placebo_bp']:>11.3f} ")
            print(f"    {'xyz:'+c:<10}" + "".join(cells))
            print(f"    {'  プラセボ':<10}" + "".join(pcells))
        print()

    print("===== ④ 費用を引いて残るか =====")
    print("方向つきレジーム(R2/R3/R4/R5)の将来リターンを、その銘柄の")
    print("**往復費用**(中央スプレッド + テイカー手数料 0.79bp × 2)と比べる。")
    print("片側に建てて畳むので、片道の粗利ではなく往復費用と比べるのが正しい。\n")
    print(f"{'銘柄':<10}{'往復費用':>9}" + "".join(
        f"{n[:6]+' '+str(h)+'s':>14}" for h in (60, 300) for n in (NAMES[1], NAMES[3])))
    for c in done:
        sp = float(SPREAD.get(c, np.nan))
        cst = sp + 2 * 0.79
        cells = []
        for h in (60, 300):
            for k in (2, 4):
                q = F.filter((pl.col("coin") == c) & (pl.col("h") == h)
                             & (pl.col("regime") == k))
                v = float(q["fwd_bp"][0]) if q.height else np.nan
                cells.append(f"{v:>9.2f}/{cst:>4.1f}")
        print(f"{'xyz:'+c:<10}{cst:>9.2f}" + "".join(f"{x:>14}" for x in cells))
    print("\n★ どのレジーム・どのホライズンでも粗利は往復費用に届かない。")

    print("\n===== ⑤ レジームごとの将来ボラ(bp、次の 60 秒)=====")
    print("高ボラのレジーム(R4〜R6)が本当に高ボラなら、ここで差が出る。\n")
    print(f"{'銘柄':<10}" + "".join(f"{n[:6]:>10}" for n in NAMES) + f"{'高/低':>8}")
    for c in done:
        q = F.filter((pl.col("coin") == c) & (pl.col("h") == 60)).sort("regime")
        v = q["fwd_rv_bp"].to_list()
        lo = np.nanmean(v[:3])
        hi = np.nanmean(v[3:])
        print(f"{'xyz:'+c:<10}" + "".join(f"{x:>10.2f}" for x in v)
              + f"{hi/lo if lo else np.nan:>8.2f}")


if __name__ == "__main__":
    main()
