r"""Derive — RFQ と CLOB を分離し、メイカーの execution edge と markout を比較する。

=============================================================================
問い
=============================================================================
Derive のオプション出来高の 20〜34%(メイカー行ベース、2026 年は 34%)は
**RFQ 経由**で板を通らない。これまでの MM 採算はこの 2 つを混ぜて測っていた。

  E[PnL | RFQ]  vs  E[PnL | CLOB]

を比べる。ただし単純平均では意味がない。**RFQ の方が 1 約定が 2 倍大きい**
(中央 $7,902 対 $3,952)ので、サイズ・満期までの日数・moneyness・IV が違う。
同じものを比べるために **マッチング**する。

=============================================================================
測る量
=============================================================================
【1】メイカーの execution edge(約定した瞬間の理論値からの乖離)

      Edge_t = mark_t − P_t   (メイカーが買った場合)
               P_t − mark_t   (メイカーが売った場合)

    正なら「理論値より有利な価格で約定した」= 板に置いた対価を得ている。

【2】markout(約定後に理論値がどちらへ動いたか)

      Markout_Δ = side × (mark_{t+Δ} − mark_t)

    side は +1(買い)/ −1(売り)。負なら逆選択されている。
    ★mark の系列は**約定行にしか無い**(板の記録は 2026-09-11 開始)。
      よって Δ は「同一銘柄の次の約定まで」で近似する。
      時間間隔が開くので、**経過時間を併記**しないと比較にならない。

【3】Edge + Markout ≒ そのメイカーの実質的な取り分。
    これを RFQ と CLOB で比べる。

=============================================================================
★マッチングの設計
=============================================================================
**同一銘柄**で突き合わせる。銘柄が同じなら通貨・行使価格・満期が同じなので、
DTE・moneyness・IV 水準はほぼ揃う(残る差は約定時刻と原資産水準だけ)。
その上で:
  - 時間窓 ±7 日
  - log(サイズ)が最も近い CLOB 約定を 1 本選ぶ(復元抽出なし)
  - ★**キャリパー |Δlog(size)| ≤ 0.5**(= サイズ比 1.65 倍以内)。
    これを入れないと、大口の RFQ に見合う CLOB 約定が無いときに
    桁違いに小さい相手とマッチしてしまう。実際、キャリパー無しでは
    名目合計が RFQ $8.02B 対 CLOB $1.76B と 4.6 倍ずれた。
    中央値では釣り合って見えても**裾で壊れる**
釣り合いは DTE・moneyness・IV・サイズ・時刻で検査して開示する。

IV は Black-Scholes(r=0、S=index)を二分法で解く。
★Derive は forward 基準なので厳密には Black-76 だが、
  **マッチングの釣り合い検査に使うだけ**なので近似で足りる。水準の解釈はしない。

出力: E:/Memory-derive/rfq/*.parquet
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "rfq"
BP = 1e4
MATCH_DAYS = 7
MAX_MARKOUT_S = 86400 * 3       # 次の約定まで 3 日を超えたら markout を測らない


# ------------------------------------------------------------------ IV
def bs_price(S, K, T, sig, is_call):
    """r=0 の Black-Scholes。マッチングの釣り合い検査用。"""
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    from math import erf, log, sqrt
    d1 = (log(S / K) + 0.5 * sig * sig * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return (S * N(d1) - K * N(d2)) if is_call else (K * N(-d2) - S * N(-d1))


def implied_vol(px, S, K, T, is_call, lo=1e-3, hi=6.0, it=60):
    intr = max(0.0, (S - K) if is_call else (K - S))
    if T <= 0 or px <= intr + 1e-12 or S <= 0 or K <= 0:
        return np.nan
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, is_call) < px:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def strike_of(ins: str) -> float:
    """★`_` は小数点。float() に直接渡すと桁区切りとして食われる。"""
    return float(ins.split("-")[2].replace("_", "."))


def expiry_ts(ins: str) -> int:
    d = ins.split("-")[1]
    return int(np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}T08:00:00")
               .astype("datetime64[s]").astype(int))


# ------------------------------------------------------------------ 本体
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv-sample", type=int, default=40000,
                    help="IV は重いので釣り合い検査用に抽出して計算する")
    ap.add_argument("--caliper", type=float, default=0.5,
                    help="log サイズ差の上限。0 で無効")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260912)

    T = pl.read_parquet(SRC / "ledger_trades.parquet")
    M = (T.filter((pl.col("role") == "maker") & (pl.col("mark") > 0)
                  & (pl.col("index") > 0) & (pl.col("amt") > 0))
         .sort(["ins", "ts"]))
    print(f"メイカー約定 {M.height:,} 行 / RFQ {int(M['rfq'].sum()):,} "
          f"({M['rfq'].mean()*100:.1f}%)")

    ins = M["ins"].to_numpy()
    ts = M["ts"].to_numpy().astype(np.int64)
    px = M["px"].to_numpy()
    mk = M["mark"].to_numpy()
    idx = M["index"].to_numpy()
    amt = M["amt"].to_numpy()
    buy = (M["dir"].to_numpy() == "buy")
    rfq = M["rfq"].to_numpy()
    side = np.where(buy, 1.0, -1.0)
    notional = amt * idx

    # 【1】execution edge(名目 bp)
    edge = np.where(buy, mk - px, px - mk) * amt / notional * BP

    # 【2】markout: 同一銘柄の次の約定の mark
    mo = np.full(len(ts), np.nan)
    gap = np.full(len(ts), np.nan)
    start = 0
    for i in range(1, len(ins) + 1):
        if i == len(ins) or ins[i] != ins[start]:
            sl = slice(start, i)
            t_ = ts[sl]; m_ = mk[sl]
            # 次の「mark が変わった」約定を探す(同時刻の連続約定を飛ばす)
            nxt = np.full(i - start, -1)
            j = i - start - 1
            for k in range(i - start - 2, -1, -1):
                if t_[k + 1] > t_[k]:
                    j = k + 1
                nxt[k] = j
            ok = nxt >= 0
            kk = np.arange(start, i)
            valid = ok & (nxt > np.arange(i - start))
            g = np.where(valid, (t_[np.clip(nxt, 0, None)] - t_) / 1000.0, np.nan)
            dm = np.where(valid, m_[np.clip(nxt, 0, None)] - m_, np.nan)
            mo[kk] = side[kk] * dm * amt[kk] / notional[kk] * BP
            gap[kk] = g
            start = i
    mo = np.where(gap <= MAX_MARKOUT_S, mo, np.nan)

    D = M.with_columns([
        pl.Series("edge_bp", edge), pl.Series("markout_bp", mo),
        pl.Series("gap_s", gap), pl.Series("notional", notional),
        pl.Series("side", side),
        (pl.col("realized") / pl.Series("notional", notional) * BP)
        .alias("realized_bp"),
    ])
    D.write_parquet(OUT / "maker_edge.parquet")

    def desc(d: pl.DataFrame, col: str) -> str:
        v = d[col].to_numpy()
        v = v[np.isfinite(v)]
        if not len(v):
            return "n/a"
        return (f"n={len(v):>7,} 平均 {v.mean():+8.3f} 中央 {np.median(v):+8.3f} "
                f"p25 {np.percentile(v,25):+8.2f} p75 {np.percentile(v,75):+8.2f}")

    print("\n=== 【1】マッチング前(素の比較。サイズも DTE も揃っていない)===")
    for lab, sub in (("CLOB", D.filter(~pl.col("rfq"))),
                     ("RFQ ", D.filter(pl.col("rfq")))):
        print(f"  {lab} edge[bp]    {desc(sub,'edge_bp')}")
        print(f"       markout[bp] {desc(sub,'markout_bp')}")
        print(f"       realized[bp]{desc(sub,'realized_bp')}")
        g = sub["gap_s"].to_numpy(); g = g[np.isfinite(g)]
        print(f"       次の約定まで 中央 {np.median(g)/60:.1f} 分 / "
              f"1 約定 中央 ${float(sub['notional'].median()):,.0f}")

    # -------------------------------------------------- マッチング
    print("\n=== マッチング(同一銘柄・±7 日・log サイズ最近傍)===")
    ins_idx = defaultdict(list)
    for i, s in enumerate(ins):
        ins_idx[s].append(i)
    win = MATCH_DAYS * 86400 * 1000
    pairs = []
    used = set()
    for s, ids in ins_idx.items():
        ids = np.array(ids)
        r = ids[rfq[ids]]
        c = ids[~rfq[ids]]
        if len(r) == 0 or len(c) == 0:
            continue
        ct = ts[c]
        cl = np.log(np.maximum(notional[c], 1.0))
        for i in r:
            lo = np.searchsorted(ct, ts[i] - win, "left")
            hi = np.searchsorted(ct, ts[i] + win, "right")
            if hi <= lo:
                continue
            cand = c[lo:hi]
            cc = cl[lo:hi]
            d = np.abs(cc - np.log(max(notional[i], 1.0)))
            order = np.argsort(d)
            for o in order:
                if a.caliper > 0 and d[o] > a.caliper:
                    break                     # ★キャリパーの外は使わない
                j = cand[o]
                if j not in used:
                    used.add(j)
                    pairs.append((i, j))
                    break
    print(f"  成立した組 {len(pairs):,} / RFQ 約定 {int(rfq.sum()):,} "
          f"({len(pairs)/max(int(rfq.sum()),1)*100:.0f}%)"
          f"  キャリパー |Δlog size| ≤ {a.caliper}")
    if not pairs:
        return 0
    ri = np.array([p[0] for p in pairs])
    ci = np.array([p[1] for p in pairs])

    # 釣り合いの検査
    exp_ts = np.array([expiry_ts(s) for s in ins])
    dte = (exp_ts - ts / 1000.0) / 86400.0
    K = np.array([strike_of(s) for s in ins])
    mny = np.log(np.maximum(K, 1e-9) / np.maximum(idx, 1e-9))
    hod = ((ts / 1000.0) % 86400) / 3600.0
    print("\n  釣り合い(マッチ後):")
    print(f"  {'covariate':<16}{'RFQ 中央':>12}{'CLOB 中央':>12}{'標準化差':>10}")
    for lab, v in (("log サイズ", np.log(np.maximum(notional, 1.0))),
                   ("満期まで[日]", dte), ("log(K/S)", mny),
                   ("時刻[UTC 時]", hod)):
        x, y = v[ri], v[ci]
        sd = np.sqrt((np.var(x) + np.var(y)) / 2)
        smd = (x.mean() - y.mean()) / sd if sd > 0 else np.nan
        print(f"  {lab:<16}{np.median(x):>12.3f}{np.median(y):>12.3f}{smd:>10.3f}")
    # IV(抽出して計算)
    n = min(a.iv_sample, len(ri))
    sel = rng.choice(len(ri), n, replace=False)
    ivr, ivc = [], []
    for k in sel:
        for arr, i in ((ivr, ri[k]), (ivc, ci[k])):
            is_call = ins[i].split("-")[3] == "C"
            arr.append(implied_vol(mk[i], idx[i], K[i],
                                   max(dte[i], 0.0) / 365.0, is_call))
    ivr = np.array(ivr); ivc = np.array(ivc)
    ok = np.isfinite(ivr) & np.isfinite(ivc)
    if ok.sum() > 100:
        sd = np.sqrt((np.var(ivr[ok]) + np.var(ivc[ok])) / 2)
        print(f"  {'mark IV':<16}{np.median(ivr[ok]):>12.3f}"
              f"{np.median(ivc[ok]):>12.3f}"
              f"{(ivr[ok].mean()-ivc[ok].mean())/sd:>10.3f}  (n={int(ok.sum()):,})")
    print("  ★標準化差 |d| < 0.1 なら釣り合っているとみなす慣例")

    # 対応のある比較
    print("\n=== 【2】マッチ後の対応比較(RFQ − CLOB)===")
    from scipy.stats import wilcoxon
    res = []
    for lab, v in (("edge [bp]", edge), ("markout [bp]", mo),
                   ("edge+markout [bp]", edge + np.nan_to_num(mo)),
                   ("realized [bp]", D["realized_bp"].to_numpy())):
        x, y = v[ri], v[ci]
        ok2 = np.isfinite(x) & np.isfinite(y)
        if ok2.sum() < 100:
            continue
        d = x[ok2] - y[ok2]
        try:
            _, p = wilcoxon(d[:200000])
        except Exception:
            p = np.nan
        res.append({"metric": lab, "n": int(ok2.sum()),
                    "rfq_mean": float(x[ok2].mean()), "clob_mean": float(y[ok2].mean()),
                    "diff_mean": float(d.mean()), "diff_med": float(np.median(d)),
                    "p": float(p)})
        print(f"  {lab:<20} n={ok2.sum():>7,}  RFQ {x[ok2].mean():+8.3f} / "
              f"CLOB {y[ok2].mean():+8.3f}  差 {d.mean():+8.3f} "
              f"(中央 {np.median(d):+7.3f}) p={p:.3g}")
    pl.DataFrame(res).write_parquet(OUT / "matched.parquet")

    # ドル建ての総額(単価だけ見ると規模を見落とす)
    print("\n=== 【3】ドル建て(マッチ標本の合計)===")
    for lab, i_ in (("RFQ", ri), ("CLOB", ci)):
        e = edge[i_] / BP * notional[i_]
        m_ = np.nan_to_num(mo[i_]) / BP * notional[i_]
        print(f"  {lab:<5} 名目 ${notional[i_].sum():>14,.0f} / "
              f"edge ${e.sum():>12,.0f} / markout ${m_.sum():>12,.0f} / "
              f"合計 ${(e+m_).sum():>12,.0f}")
    # ---------------- 【4】RFQ を選ぶのは誰か ----------------
    print("")
    print("=== 【4】RFQ 偏重は特定のメイカーに集中しているか ===")
    WP = (D.group_by("wallet").agg([
        pl.len().alias("n"), pl.col("rfq").mean().alias("rfq_share"),
        pl.col("notional").sum().alias("notional"),
        pl.col("edge_bp").mean().alias("edge"),
        # ★NaN は polars の mean で伝播する。落としてから平均する
        pl.col("markout_bp").drop_nans().mean().alias("markout"),
        pl.col("markout_bp").drop_nans().len().alias("n_mo"),
    ]).filter(pl.col("n") >= 200).sort("notional", descending=True))
    WP.write_parquet(OUT / "wallet_rfq.parquet")
    print(f"  200 約定以上のメイカー {WP.height} 者 / "
          f"RFQ 比率 中央 {float(WP['rfq_share'].median())*100:.1f}%")
    print(f"  {'wallet':<16}{'約定':>8}{'RFQ比':>8}{'edge[bp]':>10}"
          f"{'markout[bp]':>12}{'名目':>16}")
    for r in WP.head(8).iter_rows(named=True):
        print(f"  {r['wallet'][:14]:<16}{r['n']:>8,}{r['rfq_share']*100:>7.0f}%"
              f"{r['edge']:>10.2f}{r['markout']:>12.2f}${r['notional']:>15,.0f}")
    rs = WP["rfq_share"].to_numpy()
    print(f"  RFQ 比率の分布: 10% 未満 {(rs<0.1).sum()} 者 / "
          f"10〜50% {((rs>=0.1)&(rs<0.5)).sum()} 者 / 50% 以上 {(rs>=0.5).sum()} 者")
    from scipy.stats import rankdata
    ok3 = np.isfinite(WP["edge"].to_numpy())
    if ok3.sum() > 20:
        x = rankdata(rs[ok3]); y = rankdata(WP["edge"].to_numpy()[ok3])
        print(f"  ★RFQ 比率 と edge の Spearman = {np.corrcoef(x,y)[0,1]:+.3f}")
        y2 = rankdata(np.nan_to_num(WP["markout"].to_numpy()[ok3]))
        print(f"  ★RFQ 比率 と markout の Spearman = {np.corrcoef(x,y2)[0,1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
