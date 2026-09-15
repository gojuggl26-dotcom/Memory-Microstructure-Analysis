r"""レジーム判定の**執行価値**を出す(§24 Economic Significance)。

    uv run python scripts/analyze_regime_econ.py

入力: data/regime_<coin>.parquet(`build_regime.py`)、data/bbo_xyz_<coin>.parquet
出力: data/regime_econ.csv

`analyze_regime.py` の ④ は「レジームに入っている**全秒**の平均リターン」を
費用と比べたが、それは**取引の単価ではない**。実際には

* レジームに**入った瞬間に 1 回だけ**建てる(毎秒建て直さない)
* 建てるには**テイカーで板を叩く**(片道 中央スプレッド/2 + 手数料 0.79bp)
* **執行できる数量**は最良気配の数量までしかない

ので、ここでは **1 取引あたりの純益(bp)** と **総額(USD)** を出す。

## 対照(CLAUDE.md B5・B6・B7)

| 対照 | 中身 |
|---|---|
| **何もしない** | 0 USD。母集団の平均が負なら最善手は不参加である |
| **無作為対照** | 同じ回数・同じ保有時間を**無作為な秒**に建てる。粗利は 0 に近く、費用だけ残るはず |
| **施策 vs 何もしない** | 主判定はこれ。「施策後 > 0」は元が黒字なら勝手に通るので使わない |

## 実装可能性(CLAUDE.md C16)

レジームは時刻 $`t`$ の情報で決まるが、**$`t`$ の mid で建てられるとは限らない**。
判定した次の秒に発注が届くとして、**建値は $`M_{t+1}`$、決済は $`M_{t+1+h}`$**
の列も併記する(`gross_lag1_bp`)。遅延なしの列は上界であって実現値ではない。

## 数量

最良気配の**中央 notional** $`\min(\mathrm{bid\_sz}\times\mathrm{bid},
\mathrm{ask\_sz}\times\mathrm{ask})`$ を「1 回で叩ける量」とする。
これを超えて叩けば次の板へ食い込むので、**これは上界**である。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEE = 0.79                      # テイカー手数料(bp、片道)
HZ = [60, 300]
DIRS = {2: +1, 3: -1, 4: +1, 5: -1}
COINS = ["MU", "SNDK", "INTC", "AMD", "SMSN", "DRAM", "KIOXIA", "SKHX",
         "GOLD", "GOOGL", "AAPL"]


def top_of_book(coin: str) -> tuple[float, float]:
    """(中央スプレッド bp, 最良気配の中央 notional USD)。"""
    if coin == "DRAM":
        f, cols = DATA / "DRAM" / "microprice_1s.parquet", None
        t = pl.scan_parquet(f).collect_schema().names()
        sp = float(pl.scan_parquet(f).select("spread_bp").collect()["spread_bp"]
                   .median())
        if "bid_sz" in t and "ask_sz" in t:
            q = pl.scan_parquet(f).select("mid", "bid_sz", "ask_sz").collect()
            nz = np.minimum(q["bid_sz"].to_numpy(), q["ask_sz"].to_numpy()) \
                * q["mid"].to_numpy()
            return sp, float(np.nanmedian(nz[np.isfinite(nz) & (nz > 0)]))
        return sp, float("nan")
    q = (pl.scan_parquet(DATA / f"bbo_xyz_{coin}.parquet")
         .select("best_bid", "best_ask", "bid_sz", "ask_sz").collect())
    bb, ba = q["best_bid"].to_numpy(), q["best_ask"].to_numpy()
    mid = 0.5 * (bb + ba)
    s = (ba - bb) / mid * 1e4
    g = np.isfinite(s) & (s > 0) & (s < 100)
    nz = np.minimum(q["bid_sz"].to_numpy() * bb, q["ask_sz"].to_numpy() * ba)
    gz = g & np.isfinite(nz) & (nz > 0)
    return float(np.median(s[g])), float(np.median(nz[gz]))


def clustered(v, day, nd):
    """日でクラスタした平均と標準誤差。"""
    v = np.asarray(v, float)
    w = np.isfinite(v)
    s = np.bincount(day[w], weights=v[w], minlength=nd)
    c = np.bincount(day[w], minlength=nd)
    d = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan, np.nan
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


def main() -> None:
    rng = np.random.default_rng(20260915)
    rows = []
    for c in COINS:
        f = DATA / f"regime_xyz_{c}.parquet"
        if not f.exists():
            print(f"xyz:{c}: 未作成。skip", file=sys.stderr)
            continue
        B = pl.read_parquet(f)
        ts = B["ts"].to_numpy()
        lm = np.log(B["mid"].to_numpy())
        reg = B["regime"].to_numpy()
        n = ts.size
        days = B["dt"].unique(maintain_order=True).to_list()
        dmap = {d: i for i, d in enumerate(days)}
        day = np.array([dmap[d] for d in B["dt"].to_list()], np.int32)
        nd = len(days)
        sp, sz = top_of_book(c)
        cost = sp + 2 * FEE                 # 往復 = 2 ×(半スプレッド + 手数料)

        for g, sgn in DIRS.items():
            # レジームに「入った」秒だけ(毎秒建て直さない)
            ent = np.flatnonzero((reg == g) & (np.r_[0, reg[:-1]] != g))
            if ent.size < 50:
                continue
            for h in HZ:
                # 連続する秒であることを確かめる(欠測をまたがない)
                def leg(i0):
                    j0, j1 = i0, i0 + h
                    ok = (j1 < n)
                    ok &= ts[np.minimum(j1, n - 1)] == ts[np.minimum(j0, n - 1)] + h
                    return np.where(ok, sgn * (lm[np.minimum(j1, n - 1)]
                                               - lm[np.minimum(j0, n - 1)]) * 1e4,
                                    np.nan)

                g0 = leg(ent)                       # 遅延なし(上界)
                g1 = leg(np.minimum(ent + 1, n - 1))  # 1 秒遅れて執行
                # 無作為対照: 同じ回数・同じ保有時間・無作為な秒
                ridx = rng.integers(0, n - h - 2, size=ent.size)
                gr = leg(ridx)
                d_e, d_r = day[ent], day[ridx]
                m0, s0 = clustered(g0, d_e, nd)
                m1, s1 = clustered(g1, d_e, nd)
                mr, sr = clustered(gr, d_r, nd)
                net = m1 - cost
                nt = int(np.isfinite(g1).sum())
                rows.append({
                    "coin": c, "regime": g, "h": h, "n_trades": nt,
                    "spread_bp": sp, "cost_bp": cost, "size_usd": sz,
                    "gross_bp": m0, "gross_lag1_bp": m1, "se_bp": s1,
                    "t_gross": m1 / s1 if s1 else np.nan,
                    "rand_gross_bp": mr, "rand_net_bp": mr - cost,
                    "net_bp": net, "t_net": net / s1 if s1 else np.nan,
                    "total_usd": net / 1e4 * sz * nt,
                    "rand_total_usd": (mr - cost) / 1e4 * sz * nt,
                    "n_days": nd})
        print(f"[econ] xyz:{c} 完了(スプレッド {sp:.2f}bp / "
              f"最良気配 {sz:,.0f} USD)", flush=True, file=sys.stderr)

    E = pl.DataFrame(rows)
    E.write_csv(DATA / "regime_econ.csv")

    print("===== 1 取引あたりの純益(bp)と総額(USD)=====")
    print("建値は判定の 1 秒後(実装可能性)。費用 = 中央スプレッド + 0.79×2。")
    print("数量は最良気配の中央 notional までとする(これ以上は板を食う)。\n")
    print(f"{'銘柄':<10}{'R':>3}{'h':>5}{'回数':>8}{'粗利':>8}{'費用':>7}"
          f"{'純益':>8}{'t':>7}{'無作為':>9}{'数量$':>10}{'総額$':>12}")
    for r in E.iter_rows(named=True):
        print(f"{'xyz:'+r['coin']:<10}{r['regime']:>3}{r['h']:>5}"
              f"{r['n_trades']:>8,}{r['gross_lag1_bp']:>8.2f}{r['cost_bp']:>7.2f}"
              f"{r['net_bp']:>8.2f}{r['t_net']:>7.1f}"
              f"{r['rand_gross_bp']:>9.2f}{r['size_usd']:>10,.0f}"
              f"{r['total_usd']:>12,.0f}")

    pos = E.filter(pl.col("net_bp") > 0)
    print(f"\n純益が正の組み合わせ: {pos.height} / {E.height}")
    print(f"全組み合わせの総額合計: {E['total_usd'].sum():,.0f} USD "
          f"(無作為対照 {E['rand_total_usd'].sum():,.0f} USD / 何もしない 0 USD)")


if __name__ == "__main__":
    main()
