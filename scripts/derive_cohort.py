r"""Derive — 生存バイアスを除いたコホート分析と、参入者から見た経済性。

=============================================================================
問題
=============================================================================
これまでの「メイカー主体 41 ウォレット」は、**全期間を見た後で**
「maker 比 70% 以上・200 約定以上」で選んでいた。これは戦略評価として危険で、
**将来まで生き残って大量に取引したウォレットほど選ばれやすい**。
「MM は儲かるか」の答えが、勝ち残った者を選んだことの反映になってしまう。

=============================================================================
本報告の設計
=============================================================================
【A】分類と評価を時間で完全に分離する
    - 各ウォレットの **最初の 200 約定だけ**で maker 主体かを判定する
      (maker 比 70% 以上)。この 200 約定の損益は**評価に使わない**
    - **201 約定目以降**の損益だけを測る
    → 分類時点で将来を一切見ていないので、生存バイアスが入らない

【B】参入コホート → 将来損益 → 生存時間
    - 参入月(初約定の月)でコホートを作る
    - **Kaplan-Meier** で 3 / 6 / 12 か月後の生存率を出す
    - 撤退前の累積損益の分布を出す
    ★生存の定義(結果を見る前に固定):
      「最終約定から **30 日**取引が無ければ、その最終約定日に撤退したとみなす。
        最終約定が標本終端の 30 日以内なら **打ち切り(censored)**」
    ★打ち切りを無視して平均生存日数を出すと、まだ生きている者を
      「短命」として数えることになり下方に歪む。KM はこれを正しく扱う。

出力: E:/Memory-derive/cohort/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "cohort"
BP = 1e4
CLASSIFY_N = 200            # ★この本数で分類する。以降で評価する
MAKER_TH = 0.70
DEAD_DAYS = 30              # ★これだけ取引が無ければ撤退とみなす


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    T = pl.read_parquet(SRC / "ledger_trades.parquet")
    S = pl.read_parquet(SRC / "ledger_settle.parquet")
    W = pl.read_parquet(SRC / "wallet_map.parquet")
    P = pl.read_parquet(SRC / "ledger_perp.parquet")
    S = (S.join(W, on="sub", how="left")
         .filter(pl.col("wallet").is_not_null())
         .with_columns(
             pl.when(pl.col("expiry") > 0)
             .then(pl.from_epoch(pl.col("expiry"), time_unit="s")
                   .dt.strftime("%Y-%m-%d"))
             .otherwise(pl.col("ins").str.split("-").list.get(1)
                        .str.strptime(pl.Date, "%Y%m%d").dt.strftime("%Y-%m-%d"))
             .alias("day")))
    return T, S, P


def km(durations: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier の生存関数。events=1 が撤退、0 が打ち切り。"""
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    S, out, i, n = 1.0, [], 0, len(d)
    at_risk = n
    times = np.unique(d[e == 1])
    ti = 0
    for g in grid:
        while ti < len(times) and times[ti] <= g:
            t = times[ti]
            nr = int((d >= t).sum())          # そのときのリスク集合
            dd = int(((d == t) & (e == 1)).sum())
            if nr > 0:
                S *= (1.0 - dd / nr)
            ti += 1
        out.append(S)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-n", type=int, default=CLASSIFY_N)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T, S, P = load()
    END = T["day"].max()
    end_i = int(np.datetime64(END).astype("datetime64[D]").astype(int))
    print(f"約定 {T.height:,} 行 / 標本終端 {END}")

    # ------------------------------------------------ 【A】分類と評価の分離
    T = T.sort(["wallet", "ts"]).with_columns(
        pl.col("ts").cum_count().over("wallet").alias("seq"))
    n_all = T.group_by("wallet").agg(pl.len().alias("n_total"))
    # 分類期(最初の N 約定)
    C = (T.filter(pl.col("seq") <= a.classify_n).group_by("wallet").agg([
        (pl.col("role") == "maker").mean().alias("maker_share_in"),
        pl.len().alias("n_in"),
        pl.col("day").min().alias("entry_day"),
        pl.col("day").max().alias("classify_end"),
        pl.col("realized").sum().alias("pnl_in"),
    ]))
    # ★分類期が N 本に満たないウォレットは対象外(分類が確定しない)
    C = C.filter(pl.col("n_in") >= a.classify_n)
    C = C.with_columns((pl.col("maker_share_in") >= MAKER_TH).alias("is_maker"))
    print(f"\n【A】分類できたウォレット {C.height:,}"
          f"(最初の {a.classify_n} 約定が揃った者)")
    print(f"   うちメイカー主体(分類期の maker 比 {MAKER_TH:.0%} 以上) "
          f"{int(C['is_maker'].sum()):,}")

    # 評価期(201 約定目以降)の損益
    ev = (T.filter(pl.col("seq") > a.classify_n).group_by("wallet").agg([
        pl.col("realized").sum().alias("pnl_opt_out"),
        (pl.col("amt") * pl.col("index")).sum().alias("notional_out"),
        pl.len().alias("n_out"),
        pl.col("day").min().alias("eval_start"),
        pl.col("day").max().alias("last_day"),
        (pl.col("role") == "maker").mean().alias("maker_share_out"),
    ]))
    # 満期決済・perp は「評価期の開始日以降」だけを足す
    L = C.join(ev, on="wallet", how="left").join(n_all, on="wallet", how="left")
    L = L.with_columns(pl.col("eval_start").fill_null(pl.col("classify_end")))
    se = (S.join(L.select(["wallet", "eval_start"]), on="wallet", how="inner")
          .filter(pl.col("day") >= pl.col("eval_start"))
          .group_by("wallet").agg(pl.col("pnl").sum().alias("pnl_settle_out")))
    pp = (P.join(L.select(["wallet", "eval_start"]), on="wallet", how="inner")
          .filter(pl.col("day") >= pl.col("eval_start"))
          .group_by("wallet").agg([
              pl.col("realized").sum().alias("pnl_perp_out"),
              pl.col("notional").sum().alias("notional_perp_out")]))
    L = (L.join(se, on="wallet", how="left").join(pp, on="wallet", how="left")
         .fill_null(0.0))
    L = L.with_columns([
        (pl.col("pnl_opt_out") + pl.col("pnl_settle_out")
         + pl.col("pnl_perp_out")).alias("pnl_out"),
        (pl.col("notional_out") + pl.col("notional_perp_out")).alias("no_out"),
    ])
    L.write_parquet(OUT / "cohort_wallets.parquet")

    MK = L.filter(pl.col("is_maker"))
    print(f"\n=== 【A】201 約定目以降の損益(メイカー主体 {MK.height} 者)===")
    tot = float(MK["pnl_out"].sum())
    no = float(MK["no_out"].sum())
    print(f"  合計 ${tot:,.0f} / 名目 ${no:,.0f} = {tot/max(no,1)*BP:+.3f} bp")
    print(f"  黒字 {int((MK['pnl_out'] > 0).sum())}/{MK.height} / "
          f"中央値 ${float(MK['pnl_out'].median()):,.0f}")
    print(f"  内訳: オプション ${float(MK['pnl_opt_out'].sum()):,.0f} / "
          f"満期 ${float(MK['pnl_settle_out'].sum()):,.0f} / "
          f"perp ${float(MK['pnl_perp_out'].sum()):,.0f}")
    v = np.sort(MK["pnl_out"].to_numpy())[::-1]
    if len(v) >= 3:
        print(f"  ★上位 3 者で ${v[:3].sum():,.0f} = {v[:3].sum()/tot*100:.0f}%"
              f" / 上位 3 を除くと ${v[3:].sum():,.0f}")
    print(f"  分類期(最初の {a.classify_n} 約定)の損益は ${float(MK['pnl_in'].sum()):,.0f}"
          "(評価に使っていない)")
    # 分類期の成績が評価期を予測するか
    from scipy.stats import rankdata
    x, y = MK["pnl_in"].to_numpy(), MK["pnl_out"].to_numpy()
    if len(x) > 5:
        rx, ry = rankdata(x), rankdata(y)
        print(f"  ★分類期の損益と評価期の損益の Spearman = "
              f"{np.corrcoef(rx, ry)[0,1]:+.3f}(n={len(x)})")

    # ------------------------------------------------ 【B】生存分析
    ent = (T.group_by("wallet").agg([
        pl.col("day").min().alias("entry"), pl.col("day").max().alias("last"),
        pl.len().alias("n_trade"),
        (pl.col("role") == "maker").mean().alias("maker_share"),
        pl.col("realized").sum().alias("pnl_opt"),
        (pl.col("amt") * pl.col("index")).sum().alias("notional"),
    ]))
    sa = S.group_by("wallet").agg(pl.col("pnl").sum().alias("pnl_settle"))
    pa = P.group_by("wallet").agg(pl.col("realized").sum().alias("pnl_perp"))
    ent = (ent.join(sa, on="wallet", how="left")
           .join(pa, on="wallet", how="left").fill_null(0.0))
    ent = ent.with_columns(
        (pl.col("pnl_opt") + pl.col("pnl_settle") + pl.col("pnl_perp"))
        .alias("pnl_life"))
    ei = np.array([int(np.datetime64(d).astype("datetime64[D]").astype(int))
                   for d in ent["entry"].to_list()])
    li = np.array([int(np.datetime64(d).astype("datetime64[D]").astype(int))
                   for d in ent["last"].to_list()])
    dur = (li - ei).astype(float)
    # ★最終約定が終端の 30 日以内なら「まだ生きている」= 打ち切り
    event = ((end_i - li) > DEAD_DAYS).astype(int)
    ent = ent.with_columns([pl.Series("dur_days", dur),
                            pl.Series("event", event),
                            pl.Series("entry_mon",
                                      [d[:7] for d in ent["entry"].to_list()])])
    ent.write_parquet(OUT / "survival.parquet")

    print(f"\n=== 【B】生存分析(全 {ent.height:,} ウォレット)===")
    print(f"  撤退(event=1) {int(event.sum()):,} / "
          f"まだ活動中(打ち切り) {int((1-event).sum()):,}")
    grid = np.array([30, 90, 180, 365, 540])
    for lab, sub in (("全ウォレット", ent),
                     ("メイカー主体(全期間 70% 以上)",
                      ent.filter(pl.col("maker_share") >= MAKER_TH)),
                     ("200 約定以上", ent.filter(pl.col("n_trade") >= 200))):
        d = sub["dur_days"].to_numpy().astype(float)
        e = sub["event"].to_numpy()
        if len(d) < 10:
            continue
        s = km(d, e, grid)
        print(f"\n  {lab}(n={len(d):,})")
        print("    " + " / ".join(f"{int(g)}日 {v*100:>5.1f}%"
                                  for g, v in zip(grid, s)))
        # 撤退した者の生涯損益
        gone = sub.filter(pl.col("event") == 1)
        if gone.height:
            pv = gone["pnl_life"].to_numpy()
            print(f"    撤退した {gone.height:,} 者の生涯損益: "
                  f"合計 ${pv.sum():,.0f} / 中央 ${np.median(pv):,.0f} / "
                  f"黒字 {(pv>0).sum()}/{len(pv)} ({(pv>0).mean()*100:.0f}%)")
            print(f"    生存日数: 中央 {np.median(gone['dur_days'].to_numpy()):.0f} 日")
    # コホート別
    co = (ent.group_by("entry_mon").agg([
        pl.len().alias("n"), pl.col("event").mean().alias("exit_rate"),
        pl.col("dur_days").median().alias("dur_med"),
        pl.col("pnl_life").median().alias("pnl_med"),
        pl.col("pnl_life").sum().alias("pnl_sum")]).sort("entry_mon"))
    co.write_parquet(OUT / "cohort_month.parquet")
    print("\n  参入月別(先頭 6 / 末尾 3):")
    for r in list(co.iter_rows(named=True))[:6] + list(co.iter_rows(named=True))[-3:]:
        print(f"    {r['entry_mon']} n={r['n']:>5,} 撤退率 {r['exit_rate']*100:>5.1f}% "
              f"生存中央 {r['dur_med']:>5.0f} 日 損益中央 ${r['pnl_med']:>9,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
