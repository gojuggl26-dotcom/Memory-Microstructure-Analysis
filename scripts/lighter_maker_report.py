r"""Lighter maker 検証の集計(第 3 段)。

    uv run python scripts/lighter_maker_report.py
    uv run python scripts/lighter_maker_report.py --tags main_Standard,main_Premium

入力: E:/Memory-lighter/mk/mk_{TAG}_{SYM}.parquet(lighter_maker_sim.py の出力)
出力: data/lighter_maker_*.csv(リポジトリへ入れる小さい集計だけ)

=============================================================================
★判定の単位
=============================================================================
- 独立性の分母は**約定数ではなく日**。同じ日の全銘柄をまとめて 1 標本にする
- 95% は日単位の cluster bootstrap(同じ日の全銘柄をまとめて復元抽出)
- 日単位の符号検定も併記する
- 主帰無仮説 H0: E[日次純 PnL] <= 0
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

MK = Path("E:/Memory-lighter/mk")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BP = 1e4
B = 10000
HORS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0)
EXPLORE = ("2026-08-18", "2026-09-05")
SEEN_ONCE = ("2026-09-06", "2026-09-07")


def day_str(d: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(int(d) * 86400, dt.UTC).strftime("%Y-%m-%d")


def enrich(S: pl.DataFrame, sym: str) -> pl.DataFrame:
    """発注時点のスプレッドと、約定**直前**の mid を後から付ける。

    ★入口の価格改善は 2 通りある。
      imp_pre  = 約定直前の mid から見た自分の値段の有利さ(= 捕った半スプレッド)
      imp      = 約定時点の mid から見た有利さ(= 自分の約定が板を動かした後)
    前者だけを見ると「メイカーは常に半スプレッド分得」に見える。後者が実態に近い。
    """
    B = pl.read_parquet(Path("E:/Memory-lighter/wf") / f"bbo_{sym}.parquet").sort("recv_ns")
    br = B["recv_ns"].to_numpy()
    bb, ba = B["bid"].to_numpy(), B["ask"].to_numpy()
    del B
    t0 = S["t0"].to_numpy()
    i0 = np.clip(np.searchsorted(br, t0, "right") - 1, 0, br.size - 1)
    mid0 = 0.5 * (bb[i0] + ba[i0])
    add = {"spr0_bp": np.where(mid0 > 0, BP * (ba[i0] - bb[i0]) / mid0, np.nan)}
    side = S["side"].to_numpy()
    px = S["px"].to_numpy()
    tl = S["tl"].to_numpy()
    il = np.clip(np.searchsorted(br, tl, "right") - 1, 0, br.size - 1)
    midl = 0.5 * (bb[il] + ba[il])
    # ★ live 時点の mid から見た「取ったつもりの半スプレッド」
    add["imp_live_bp"] = np.where(tl > 0, BP * side * (midl - px) / px, np.nan)
    add["sprl_bp"] = np.where(midl > 0, BP * (ba[il] - bb[il]) / midl, np.nan)
    for mdl in ("Q1", "Q2", "Q3"):
        if f"tf_{mdl}" not in S.columns:
            continue
        fl = S[f"fil_{mdl}"].to_numpy()
        tf = S[f"tf_{mdl}"].to_numpy()
        # ★「直前」= 受信時刻が厳密に小さい最後の行
        j = np.clip(np.searchsorted(br, np.where(fl, tf, 0), "left") - 1, 0, br.size - 1)
        mpre = 0.5 * (bb[j] + ba[j])
        add[f"imp_pre_{mdl}"] = np.where(fl, BP * side * (mpre - px) / px, np.nan)
        add[f"mid_pre_{mdl}"] = np.where(fl, mpre, np.nan)
    return S.with_columns([pl.Series(k, v) for k, v in add.items()])


def load(tag: str) -> pl.DataFrame:
    fs = sorted(glob.glob(str(MK / f"mk_{tag}_*.parquet")))
    if not fs:
        return pl.DataFrame()
    out = []
    for f in fs:
        sym = Path(f).stem.rsplit("_", 1)[1]
        out.append(enrich(pl.read_parquet(f), sym))
    return pl.concat(out, how="diagonal_relaxed")


def boot_days(v: np.ndarray, seed: int = 11):
    """日を単位にした復元抽出。v は日ごとの合計。"""
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, (B, v.size))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def sign_test(v: np.ndarray):
    """日単位の符号検定(正の日の割合が 1/2 か)。"""
    from scipy.stats import binomtest
    k = int((v > 0).sum())
    n = int(np.isfinite(v).sum())
    if n == 0:
        return 0, 0, float("nan")
    return k, n, float(binomtest(k, n, 0.5, alternative="greater").pvalue)


def summarise(L: pl.DataFrame, mdl: str, pcol: str = None, label: str = "") -> dict:
    """1 構成 × 1 待ち行列モデルの要約。"""
    pcol = pcol or f"pnl_{mdl}"
    sb = L[f"sub_{mdl}"].to_numpy()
    fl = L[f"fil_{mdl}"].to_numpy()
    q = L["q"].to_numpy()
    fq = L[f"fq_{mdl}"].to_numpy()
    pn = L[pcol].to_numpy()
    no = L[f"notional_{mdl}"].to_numpy()
    day = L["day"].to_numpy()
    ndays = len(np.unique(day[sb])) if sb.any() else 0
    o = {"label": label, "model": mdl,
         "quotes": int(sb.sum()), "reject": int(L["reject"].sum()),
         "stale": int(L["stale"].sum()),
         "alone": int(L["alone"].to_numpy()[sb].sum()) if sb.any() else 0,
         "behind": int(L["behind"].to_numpy()[sb].sum()) if sb.any() else 0,
         "fills": int(fl.sum()),
         "fill_rate": float(fl.sum() / max(sb.sum(), 1)),
         "full_fill": int((fl & (fq >= q - 1e-12)).sum()),
         "partial": int((fl & (fq < q - 1e-12)).sum()),
         "cxpend": int(L[f"cxpend_{mdl}"].to_numpy()[fl].sum()) if fl.any() else 0,
         "days": ndays}
    if fl.any():
        tt = (L[f"tf_{mdl}"].to_numpy()[fl] - L["tl"].to_numpy()[fl]) / 1e9
        o["t_fill_med_s"] = float(np.median(tt))
        o["t_fill_p90_s"] = float(np.quantile(tt, 0.9))
        o["fq_per_quote"] = float(fq[sb].sum() / max(sb.sum(), 1))
        o["imp_bp"] = float(np.nanmean(L[f"imp_{mdl}"].to_numpy()))
        if f"imp_pre_{mdl}" in L.columns:
            o["imp_pre_bp"] = float(np.nanmean(L[f"imp_pre_{mdl}"].to_numpy()))
        o["spr0_bp"] = float(np.nanmedian(L["spr0_bp"].to_numpy()))
        # 約定した quote だけの「live 時点の半スプレッド」と「約定時点の改善」
        o["imp_live_bp"] = float(np.nanmean(
            np.where(fl, L["imp_live_bp"].to_numpy(), np.nan)))
        o["spr_fill_bp"] = float(np.nanmedian(
            np.where(fl, L["sprl_bp"].to_numpy(), np.nan)))
        for h in HORS:
            o[f"mk{h:g}"] = float(np.nanmean(L[f"mk{h:g}_{mdl}"].to_numpy()))
            o[f"dr{h:g}"] = float(np.nanmean(L[f"dr{h:g}_{mdl}"].to_numpy()))
    net = float(np.nansum(pn))
    nn = float(np.nansum(no))
    o["net_usd"] = net
    o["notional_usd"] = nn
    o["net_bp"] = BP * net / nn if nn > 0 else float("nan")
    o["ev_per_quote_usd"] = net / max(int(sb.sum()), 1)
    o["fee_usd"] = float(np.nansum(L[f"fee_{mdl}"].to_numpy()))
    # 日単位
    dd = np.unique(day[sb]) if sb.any() else np.array([], np.int64)
    dv = np.array([np.nansum(np.where(sb & (day == d), pn, 0.0)) for d in dd])
    o["pnl_per_day_usd"] = float(dv.mean()) if dv.size else float("nan")
    lo, hi = boot_days(dv)
    o["ci_lo"], o["ci_hi"] = lo, hi
    k, n, p = sign_test(dv)
    o["pos_days"], o["n_days"], o["sign_p"] = k, n, p
    o["max_day_share"] = float(np.max(np.abs(dv)) / max(np.abs(dv).sum(), 1e-12)) \
        if dv.size else float("nan")
    return o


def per_symbol(L: pl.DataFrame, mdl: str) -> pl.DataFrame:
    rows = []
    for sym in sorted(set(L["sym"].to_list())):
        S = L.filter(pl.col("sym") == sym)
        sb = S[f"sub_{mdl}"].to_numpy()
        fl = S[f"fil_{mdl}"].to_numpy()
        pn = S[f"pnl_{mdl}"].to_numpy()
        no = S[f"notional_{mdl}"].to_numpy()
        nn = float(np.nansum(no))
        rows.append({"sym": sym, "quotes": int(sb.sum()), "fills": int(fl.sum()),
                     "fill_rate": float(fl.sum() / max(sb.sum(), 1)),
                     "imp_bp": float(np.nanmean(S[f"imp_{mdl}"].to_numpy())),
                     "mk300_bp": float(np.nanmean(S[f"mk300_{mdl}"].to_numpy())),
                     "fee_usd": float(np.nansum(S[f"fee_{mdl}"].to_numpy())),
                     "net_usd": float(np.nansum(pn)),
                     "net_bp": BP * float(np.nansum(pn)) / nn if nn > 0 else np.nan})
    return pl.DataFrame(rows)


def daily(L: pl.DataFrame, mdl: str) -> pl.DataFrame:
    sb = L[f"sub_{mdl}"].to_numpy()
    fl = L[f"fil_{mdl}"].to_numpy()
    pn = L[f"pnl_{mdl}"].to_numpy()
    no = L[f"notional_{mdl}"].to_numpy()
    day = L["day"].to_numpy()
    rows = []
    for d in np.unique(day[sb]):
        m = sb & (day == d)
        nn = float(np.nansum(np.where(m, no, 0.0)))
        rows.append({"day": day_str(int(d)), "quotes": int(m.sum()),
                     "fills": int((fl & m).sum()),
                     "fill_rate": float((fl & m).sum() / max(m.sum(), 1)),
                     "net_usd": float(np.nansum(np.where(m, pn, 0.0))),
                     "notional_usd": nn,
                     "net_bp": BP * float(np.nansum(np.where(m, pn, 0.0))) / nn
                     if nn > 0 else np.nan})
    return pl.DataFrame(rows)


def spread_cells(tags, mdl="Q1") -> pl.DataFrame:
    """★3 次元表 — 置き場所 × 寿命 × スプレッド区分。

    スプレッド区分は**銘柄内の過去分位**で切る(全期間の分位を使うと
    その日の情報が閾値に入るため)。ここでは銘柄ごとに発注時点スプレッドの
    3 分位(狭い / 中 / 広い)を、**その銘柄の探索期間全体**で引いている。
    ★この分位は当日を含むので、区分そのものは探索的な層別であり、
      取引規則としては使えない。結論に使わず、どこに約定が集まるかを見るだけ。
    """
    rows = []
    for tag in tags:
        L = load(tag)
        if not L.height:
            continue
        for sym in sorted(set(L["sym"].to_list())):
            S = L.filter(pl.col("sym") == sym)
            sp = S["spr0_bp"].to_numpy()
            ok = np.isfinite(sp)
            if ok.sum() < 100:
                continue
            e = np.quantile(sp[ok], [1 / 3, 2 / 3])
            k = np.searchsorted(e, sp, "right")
            sb = S[f"sub_{mdl}"].to_numpy()
            fl = S[f"fil_{mdl}"].to_numpy()
            pn = S[f"pnl_{mdl}"].to_numpy()
            no = S[f"notional_{mdl}"].to_numpy()
            im = S[f"imp_{mdl}"].to_numpy()
            m3 = S[f"mk300_{mdl}"].to_numpy()
            for c in (0, 1, 2):
                m = sb & (k == c) & ok
                rows.append({"label": tag, "sym": sym, "spread_bin": c,
                             "quotes": int(m.sum()),
                             "fills": int((m & fl).sum()),
                             "net_usd": float(np.nansum(np.where(m, pn, 0.0))),
                             "notional_usd": float(np.nansum(np.where(m, no, 0.0))),
                             "imp_sum": float(np.nansum(np.where(m & fl, im, 0.0))),
                             "mk300_sum": float(np.nansum(np.where(m & fl, m3, 0.0)))})
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    tags = a.tags.split(",") if a.tags else \
        sorted({Path(f).stem.split("mk_")[1].rsplit("_", 1)[0]
                for f in glob.glob(str(MK / "mk_*.parquet"))})
    allrows, sym_rows, day_rows = [], [], []
    for tag in tags:
        L = load(tag)
        if not L.height:
            continue
        for mdl in ("Q1", "Q2", "Q3"):
            if f"sub_{mdl}" not in L.columns:
                continue
            r = summarise(L, mdl, label=tag)
            allrows.append(r)
        P = per_symbol(L, "Q1").with_columns(pl.lit(tag).alias("label"))
        sym_rows.append(P)
        D = daily(L, "Q1").with_columns(pl.lit(tag).alias("label"))
        day_rows.append(D)
    if not allrows:
        print("入力なし")
        return
    S = pl.DataFrame(allrows)
    DATA.mkdir(exist_ok=True)
    S.write_csv(DATA / "lighter_maker_summary.csv")
    pl.concat(sym_rows).write_csv(DATA / "lighter_maker_by_symbol.csv")
    pl.concat(day_rows).write_csv(DATA / "lighter_maker_daily.csv")
    gt = [t for t in tags if t.startswith("g_") or t == "main_Standard"]
    if len(gt) > 1:
        spread_cells(gt).write_csv(DATA / "lighter_maker_cells.csv")
    print("★ 構成 × 待ち行列モデル(標本外ではなく探索期間。暫定)")
    hdr = (f"{'構成':26s} {'M':3s} {'quote':>8s} {'約定':>7s} {'約定率':>7s} "
           f"{'入口改善':>8s} {'mk300':>8s} {'純bp':>9s} {'純$/日':>9s} "
           f"{'95%下限':>9s} {'正の日':>7s}")
    print(hdr)
    for r in S.iter_rows(named=True):
        print(f"{r['label']:26s} {r['model']:3s} {r['quotes']:>8,} "
              f"{r['fills']:>7,} {100*r['fill_rate']:>6.2f}% "
              f"{r.get('imp_bp', float('nan')):>+8.3f} "
              f"{r.get('mk300', float('nan')):>+8.3f} "
              f"{r['net_bp']:>+9.3f} {r['pnl_per_day_usd']:>+9.4f} "
              f"{r['ci_lo']:>+9.4f} {r['pos_days']:>3}/{r['n_days']:<3}")
    print(f"\n書き出し {DATA}/lighter_maker_summary.csv ほか")


if __name__ == "__main__":
    main()
