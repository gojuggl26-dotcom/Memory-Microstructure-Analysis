r"""Derive オプション — 実現損益による MM 採算の実測検証。

=============================================================================
何を測るか
=============================================================================
取引所が確定させた実現損益を使って、**オプション MM が実際に儲かっているか**を
測る。推定(mark からのエッジ + マークアウト)ではなく、確定値で検証する。

  損益 = オプションの realized_pnl + 満期決済の settle_pnl
       + **perp の realized_pnl(デルタヘッジ)**
       [+ 残存建玉の時価(別建て・未実現)]

★perp を入れる理由: オプション MM はデルタを perp でヘッジする。実測で
  メイカー主体 41 ウォレットのうち 5 つが perp も取引しており、その中に
  **最大の赤字ウォレットと最大の黒字ウォレットの両方**が含まれる。
★★残り 36 は venue 外でヘッジしているか、建玉間で相殺しているか、
  ヘッジしていない。**venue 外のヘッジは原理的に観測できない**ので、
  ここで測れるのは「Derive 上で見える損益」に限られる。これは必ず明示する。

=============================================================================
★測り方の規律(CLAUDE.md の自己精査に対応)
=============================================================================
A-1 費用を全部引いたか
    `realized_pnl` / `settle_pnl` は**手数料差引後**。手数料前の値も併記して、
    手数料が損益をどれだけ食っているかを出す。リベートは含有が不明なので別建て。
A-2 分母は何か
    bp 表示は**名目**(数量 × index)で割る。プレミアムではない。両方出す。
A-3 その指標は問いに答えているか
    「MM が儲かるか」なので、**メイカーとして約定した分**の損益を主に見る。
    ただし建玉は maker/taker 双方の約定で作られるので、
    **ウォレット単位の総損益**と**メイカー約定に限った損益**を分けて出す。
B-5 主判定がベースライン依存で自動成立しないか
    対照は「何もしない = 損益 0」。判定は「総額 > 0 か」。
B-7 単価だけでなく総額も
    1 約定あたり bp と、ドル総額の両方を出す。
C-9 「有意でない」を「効果がない」と書かない
C-11 中央値だけでなく最悪値も
C-12 帰無対照
    日単位ブートストラップ(同じ日の全ウォレットをまとめて再抽出)。

★残存建玉の時価は**未実現**なので主判定には入れない。別欄で開示する。
★満期決済を落とすと符号が変わりうる(オプションは満期で終わる建玉が多い)。

出力: E:/Memory-derive/mm/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("E:/Memory-derive")
OUT = SRC / "mm"
BP = 1e4
NBOOT = 5000


def boot_days(df: pl.DataFrame, val: str, rng, nboot=NBOOT) -> dict:
    """日を単位に復元抽出(同じ日の全ウォレットをまとめて)。"""
    g = df.group_by("day").agg(pl.col(val).sum().alias("v")).sort("day")
    v = g["v"].to_numpy()
    if len(v) < 5:
        return {}
    idx = rng.integers(0, len(v), size=(nboot, len(v)))
    b = v[idx].sum(axis=1)
    return {"n_day": len(v), "sum": float(v.sum()),
            "day_pos": int((v > 0).sum()),
            "lo": float(np.percentile(b, 2.5)),
            "hi": float(np.percentile(b, 97.5)),
            "p_gt0": float((b > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=200,
                    help="この約定数に満たないウォレットは個別表から外す")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260912)

    T = pl.read_parquet(SRC / "ledger_trades.parquet")
    P = (pl.read_parquet(SRC / "ledger_perp.parquet")
         if (SRC / "ledger_perp.parquet").exists() else None)
    S = pl.read_parquet(SRC / "ledger_settle.parquet")
    W = pl.read_parquet(SRC / "wallet_map.parquet")
    O = pl.read_parquet(SRC / "ledger_open.parquet")

    T = T.with_columns([
        (pl.col("amt") * pl.col("index")).alias("notional"),
        (pl.col("amt") * pl.col("px")).alias("premium"),
    ])
    # 決済に wallet を付ける(決済側は subaccount しか持たない)
    S = S.join(W, on="sub", how="left")
    # ★自前補完した行は expiry=0 なので、銘柄名から満期日を復元する
    S = S.with_columns(
        pl.when(pl.col("expiry") > 0)
        .then(pl.from_epoch(pl.col("expiry"), time_unit="s").dt.strftime("%Y-%m-%d"))
        .otherwise(pl.col("ins").str.split("-").list.get(1)
                   .str.strptime(pl.Date, "%Y%m%d").dt.strftime("%Y-%m-%d"))
        .alias("day"))

    # ---------------- ① ウォレット単位の完全な損益 ----------------
    tr = (T.group_by("wallet").agg([
        pl.col("realized").sum().alias("pnl_trade"),
        pl.col("realized_gross").sum().alias("pnl_trade_gross"),
        pl.col("fee").sum().alias("fee"),
        pl.col("rebate").sum().alias("rebate"),
        pl.len().alias("n_rows"),
        (pl.col("role") == "maker").sum().alias("n_maker"),
        (pl.col("role") == "taker").sum().alias("n_taker"),
        pl.col("notional").sum().alias("notional_both"),
        pl.col("rfq").sum().alias("n_rfq"),
        pl.col("day").min().alias("first_day"),
        pl.col("day").max().alias("last_day"),
    ]))
    se = (S.filter(pl.col("wallet").is_not_null()).group_by("wallet").agg([
        pl.col("pnl").sum().alias("pnl_settle"),
        pl.col("pnl_gross").sum().alias("pnl_settle_gross"),
        pl.len().alias("n_settle"),
    ]))
    # ★ledger_perp は (wallet, day, cur, role) で集計済み
    pp = (P.group_by("wallet").agg([
        pl.col("realized").sum().alias("pnl_perp"),
        pl.col("fee").sum().alias("fee_perp"),
        pl.col("n").sum().alias("n_perp"),
        pl.col("notional").sum().alias("notional_perp"),
    ]) if P is not None and P.height else
        pl.DataFrame(schema={"wallet": pl.String, "pnl_perp": pl.Float64,
                             "fee_perp": pl.Float64, "n_perp": pl.UInt32,
                             "notional_perp": pl.Float64}))
    op = (O.join(W, on="sub", how="left")
          .filter(pl.col("wallet").is_not_null())
          .group_by("wallet").agg([
              pl.col("mtm").sum().alias("open_mtm"),
              pl.len().alias("n_open")]))
    L = (tr.join(se, on="wallet", how="left")
           .join(pp, on="wallet", how="left")
           .join(op, on="wallet", how="left")
           .fill_null(0.0))
    L = L.with_columns([
        (pl.col("pnl_trade") + pl.col("pnl_settle")
         + pl.col("pnl_perp")).alias("pnl_realized"),
        (pl.col("pnl_trade") + pl.col("pnl_settle")).alias("pnl_opt_only"),
        (pl.col("pnl_trade_gross") + pl.col("pnl_settle_gross"))
        .alias("pnl_realized_gross"),
        (pl.col("n_maker") / pl.col("n_rows")).alias("maker_share"),
    ])
    L.write_parquet(OUT / "wallet_pnl.parquet")

    tot = float(L["pnl_realized"].sum())
    print("=" * 72)
    print(f"全ウォレット {L.height:,} / 実現損益の総和 ${tot:,.0f}")
    print(f"  内訳: オプション反対売買 ${L['pnl_trade'].sum():,.0f} + "
          f"満期決済 ${L['pnl_settle'].sum():,.0f} + "
          f"perp ${L['pnl_perp'].sum():,.0f}")
    print(f"  手数料前だと ${L['pnl_realized_gross'].sum():,.0f} "
          f"(手数料 ${L['fee'].sum():,.0f} / リベート ${L['rebate'].sum():,.0f})")
    print(f"  ★残存建玉の時価(未実現・主判定に含めない) "
          f"${L['open_mtm'].sum():,.0f}")
    print("  ※ 総和が 0 でないのは、満期決済が入っても取引所・清算基金・"
          "未取得期間の相手方があるため")

    # ---------------- ② メイカー主体のウォレット ----------------
    # ★分類は結果を見る前に決める: メイカー約定が 7 割以上 かつ 約定数が閾値以上
    MK = L.filter((pl.col("maker_share") >= 0.7)
                  & (pl.col("n_rows") >= a.min_trades))
    TK = L.filter((pl.col("maker_share") <= 0.3)
                  & (pl.col("n_rows") >= a.min_trades))
    print("\n" + "=" * 72)
    print(f"メイカー主体(maker 比 70% 以上・{a.min_trades} 約定以上): {MK.height} ウォレット")
    print(f"  実現損益の合計 ${MK['pnl_realized'].sum():,.0f}")
    print(f"  黒字 {int((MK['pnl_realized'] > 0).sum())}/{MK.height}")
    print(f"テイカー主体(maker 比 30% 以下): {TK.height} ウォレット")
    print(f"  実現損益の合計 ${TK['pnl_realized'].sum():,.0f}")
    print(f"  黒字 {int((TK['pnl_realized'] > 0).sum())}/{TK.height}")

    print("\nメイカー主体の上位・下位:")
    cols = ["wallet", "pnl_realized", "pnl_trade", "pnl_settle", "fee",
            "rebate", "n_rows", "maker_share", "notional_both", "open_mtm"]
    for lab, d in (("上位", MK.sort("pnl_realized", descending=True).head(8)),
                   ("下位", MK.sort("pnl_realized").head(5))):
        print(f"  --- {lab} ---")
        for r in d.iter_rows(named=True):
            no = r["notional_both"] / 2          # maker+taker 行で 2 倍なので
            bp = r["pnl_realized"] / max(no, 1) * BP
            print(f"   {r['wallet'][:12]}.. 実現 ${r['pnl_realized']:>12,.0f} "
                  f"(売買 {r['pnl_trade']:>11,.0f} / 満期 {r['pnl_settle']:>11,.0f}"
                  f" / perp {r['pnl_perp']:>11,.0f}) "
                  f"名目 ${no:>13,.0f} = {bp:>+7.2f} bp  約定 {r['n_rows']:>6,} "
                  f"maker {r['maker_share']*100:>3.0f}%")

    # ---------------- ③ 名目あたり(bp)と日ブートストラップ ----------------
    mk_w = set(MK["wallet"].to_list())
    Tm = T.filter(pl.col("wallet").is_in(list(mk_w)))
    daily = (Tm.group_by("day").agg([
        pl.col("realized").sum().alias("pnl"),
        pl.col("realized_gross").sum().alias("pnl_gross"),
        pl.col("notional").sum().alias("notional"),
        pl.col("fee").sum().alias("fee"),
        pl.len().alias("n")]).sort("day"))
    sd = (S.filter(pl.col("wallet").is_in(list(mk_w)))
          .group_by("day").agg(pl.col("pnl").sum().alias("settle")))
    daily = daily.join(sd, on="day", how="left").fill_null(0.0)
    if P is not None and P.height:
        pd_ = (P.filter(pl.col("wallet").is_in(list(mk_w)))
               .group_by("day").agg(pl.col("realized").sum().alias("perp")))
        daily = daily.join(pd_, on="day", how="left").fill_null(0.0)
    else:
        daily = daily.with_columns(pl.lit(0.0).alias("perp"))
    daily = daily.with_columns(
        (pl.col("pnl") + pl.col("settle") + pl.col("perp")).alias("total"))
    daily.write_parquet(OUT / "daily_maker.parquet")

    print("\n" + "=" * 72)
    print("メイカー主体ウォレットの日次損益(オプション売買 + 満期決済 + perp)")
    no = float(daily["notional"].sum()) / 2
    tt = float(daily["total"].sum())
    print(f"  日数 {daily.height} / 総額 ${tt:,.0f} / 名目 ${no:,.0f} "
          f"= {tt/max(no,1)*BP:+.3f} bp")
    print(f"  黒字の日 {int((daily['total'] > 0).sum())}/{daily.height}")
    print(f"  ★「何もしない」(=0)との比較: {'勝ち' if tt > 0 else '負け'}")
    st = boot_days(daily.select(["day", "total"]), "total", rng)
    if st:
        print(f"  日ブートストラップ 95%CI [${st['lo']:,.0f}, ${st['hi']:,.0f}] "
              f"/ P(総額>0) = {st['p_gt0']:.3f}")
    q = np.percentile(daily["total"].to_numpy(), [5, 25, 50, 75, 95])
    print(f"  日次の分位: p5 ${q[0]:,.0f} / 中央 ${q[2]:,.0f} / p95 ${q[4]:,.0f}")
    print(f"  最悪の日 ${daily['total'].min():,.0f} / 最良 ${daily['total'].max():,.0f}")

    # ---------------- ④ 満期決済を落とすとどうなるか ----------------
    only_tr = float(daily["pnl"].sum())
    print("\n  ★満期決済を無視した場合: "
          f"${only_tr:,.0f} ({only_tr/max(no,1)*BP:+.3f} bp) "
          f"-> 符号は {'変わらない' if (only_tr > 0) == (tt > 0) else '★反転する'}")
    print(f"  ★手数料前: ${float(daily['pnl_gross'].sum()) + float(daily['settle'].sum()):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
