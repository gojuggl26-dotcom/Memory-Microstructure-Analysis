r"""LST の市場価格と償還価値の乖離を出す。

    uv run python scripts/lst_discount_analysis.py

入力: data/staking/lst_market_prices.parquet   (chainlink_lst_prices.py)
      data/staking/lst_onchain_rates.parquet   (onchain_lst_rates.py)
出力: data/staking/lst_discount.parquet
      reports/staking/discount.md

    乖離 = 市場価格 / 償還価値 - 1

負なら「市場が償還価値より安く付けている」= 買って償還を待てば取れる差。

=============================================================================
★ リベース型とレート型で「償還価値」が違う - ここを間違えると全部壊れる
=============================================================================
onchain_lst_rates.py が出す rate は **1 シェアあたりの ETH** である。
これがそのまま償還価値になるのはレート型だけで、リベース型では違う。

| Chainlink フィード | 償還価値                          | 理由                               |
|--------------------|-----------------------------------|------------------------------------|
| stETH/ETH          | **常に 1.0**                      | リベース型。1 stETH は定義上 1 ETH |
|                    |                                   | 分。増えるのは数量であってレートで |
|                    |                                   | はない(1.2439 は wstETH 相当)      |
| rETH/ETH           | rETH の rate (totalEth/rethSupply)| レート型                           |
| cbETH/ETH          | cbETH の rate                     | レート型                           |
| weETH/ETH          | eETH の rate                      | weETH は eETH の**シェア**ラッパー |
|                    | (totalEthLocked/totalEEthShares)   | なので eETH のシェア単価がそのまま |

stETH に 1.2439 を当ててしまうと「市場価格が償還価値の 80%」という
ありえない乖離が出る。2026-09-14 の実測で健全性を確認できる:

    stETH 市場 0.999837 / 償還 1.0        -> -0.016%
    rETH  市場 1.168527 / 償還 1.171722   -> -0.273%
    cbETH 市場 1.139148 / 償還 1.139396   -> -0.022%
    weETH 市場 1.103388 / 償還 1.103636   -> -0.022%

いずれも「市場がわずかに安い」で、符号も桁も妥当である。

=============================================================================
★ 結合は必ず backward - 未来の償還レートを使わない
=============================================================================
市場価格の各時点に対し、**その時点までに確定している最新の償還レート**を
当てる(join_asof の strategy="backward")。nearest や forward を使うと、
まだ起きていないリベースの結果で過去の乖離を評価することになる。

償還レートの更新は 1 日 1 回程度、市場価格は数時間おきなので、backward だと
最大 1 日ぶん古いレートを使うことになる。これは「その時点で市場参加者が
知りえた情報」と一致しており、正しい。
"""
from __future__ import annotations

import datetime as dt
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MKT = ROOT / "data" / "staking" / "lst_market_prices.parquet"
RED = ROOT / "data" / "staking" / "lst_onchain_rates.parquet"
OUT = ROOT / "data" / "staking" / "lst_discount.parquet"
MD = ROOT / "reports" / "staking" / "discount.md"

# フィード -> (償還レートの出どころ, 固定値)
# 固定値が入っているものはリベース型で、償還価値は定数 1.0。
PAIR = {
    "stETH/ETH": (None, 1.0),
    "rETH/ETH":  ("rETH", None),
    "cbETH/ETH": ("cbETH", None),
    "weETH/ETH": ("eETH", None),
}

BIG = -0.01      # これより下を「大きな割安」として拾う(-1%)


def main() -> None:
    mkt = pl.read_parquet(MKT).filter(pl.col("ts").is_not_null())
    red = pl.read_parquet(RED).filter(pl.col("ts").is_not_null())
    print(f"市場価格 {mkt.height:,} 行 / 償還レート {red.height:,} 行")

    frames = []
    for feed, (src, fixed) in PAIR.items():
        m = mkt.filter(pl.col("feed") == feed).sort("ts")
        if m.height == 0:
            print(f"  {feed}: 市場価格なし。飛ばす")
            continue
        if fixed is not None:
            j = m.with_columns(pl.lit(fixed).alias("redeem"),
                               pl.lit("定数(リベース型)").alias("redeem_src"))
        else:
            r = (red.filter(pl.col("lst") == src)
                    .select("ts", pl.col("rate").alias("redeem"))
                    .sort("ts"))
            if r.height == 0:
                print(f"  {feed}: 償還レート({src})なし。飛ばす")
                continue
            # ★ backward。未来のレートを引かない
            j = m.join_asof(r, on="ts", strategy="backward")
            j = j.with_columns(pl.lit(src).alias("redeem_src"))
        j = (j.filter(pl.col("redeem").is_not_null() & (pl.col("redeem") > 0))
              .with_columns((pl.col("price") / pl.col("redeem") - 1.0).alias("disc")))
        frames.append(j)
        print(f"  {feed}: {j.height:,} 行 結合")

    d = pl.concat(frames, how="vertical_relaxed").sort("feed", "ts")
    d = d.with_columns(
        pl.from_epoch("ts").dt.replace_time_zone("UTC").alias("dt_utc")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT, compression="zstd")
    print(f"\n[write] {OUT}  {d.height:,} 行\n")

    # ---- レポート ---------------------------------------------------------
    L = ["# LST の市場価格と償還価値の乖離", "",
         "`乖離 = 市場価格 / 償還価値 - 1`。負なら市場が償還価値より安い。", "",
         "市場価格は Chainlink の `<LST>/ETH` フィード(全 phase)、",
         "償還価値はプロトコルのイベントログから復元した 1 シェアあたり ETH。",
         "結合は `join_asof(strategy=\"backward\")` で、**その時点までに確定している",
         "レートだけ**を使っている。", "",
         "★ stETH はリベース型なので償還価値は定数 1.0 である",
         "(1.2439 は wstETH 相当のレートで、stETH トークンの対 ETH 価値ではない)。", ""]

    # 図は scripts/lst_discount_figs.py が生成する(この後に走らせること)
    L += ["![乖離の推移](fig/discount_ts.png)", "",
          "![乖離の分布](fig/discount_dist.png)", "",
          "## 要約", "",
          "| フィード | 件数 | 期間 | 乖離 中央値 | 5%点 | 最安 | 最安の日 |",
          "|---|---:|---|---:|---:|---:|---|"]
    for f in d["feed"].unique(maintain_order=True):
        s = d.filter(pl.col("feed") == f)
        a = dt.datetime.fromtimestamp(s["ts"].min(), dt.timezone.utc).date()
        b = dt.datetime.fromtimestamp(s["ts"].max(), dt.timezone.utc).date()
        lo = s.sort("disc").row(0, named=True)
        when = dt.datetime.fromtimestamp(lo["ts"], dt.timezone.utc).date()
        L.append(f"| {f} | {s.height:,} | {a} 〜 {b} | "
                 f"{s['disc'].median()*100:.3f}% | {s['disc'].quantile(0.05)*100:.3f}% | "
                 f"**{lo['disc']*100:.2f}%** | {when} |")
    L.append("")

    L += [f"## {BIG*100:.0f}% を超える割安がどれだけあったか", "",
          "| フィード | 該当ラウンド | 全体に占める割合 | 該当した日数 |", "|---|---:|---:|---:|"]
    for f in d["feed"].unique(maintain_order=True):
        s = d.filter(pl.col("feed") == f)
        h = s.filter(pl.col("disc") < BIG)
        days = h.select(pl.col("dt_utc").dt.date()).n_unique() if h.height else 0
        L.append(f"| {f} | {h.height:,} | {100*h.height/s.height:.2f}% | {days} |")
    L.append("")

    # ---- 持続時間 ---------------------------------------------------------
    # ★ 同じ「-5%」でも、7 日続いた乖離と 3 分で消えた乖離はまったく別物である。
    #   実測: stETH の -6.50%(2022-06-18)は前後 7 日ずっと -5.6〜-6.5% だったが、
    #   rETH の -5.11%(2025-12-18 05:41)は 05:40〜05:42 の 3 分だけで、
    #   その前後は -0.09% だった。後者は事実上取りに行けない。
    #   閾値を下回る点が連続した区間をまとめ、その長さを出す。
    L += [f"## {BIG*100:.0f}% を下回った局面の持続時間", "",
          "同じ深さの乖離でも、**数日続いたもの**と**数分で消えたもの**では",
          "取りに行けるかがまるで違う。閾値を下回る点が連続した区間を 1 局面として",
          "まとめ、その長さを出した。", "",
          "区間の長さは「閾値を下回っていた最初の観測から最後の観測まで」で、",
          "観測はチェーンリンクの更新時点である。更新は不規則(数分〜1 日)なので、",
          "**長さは下限の目安**であって正確な継続時間ではない。", "",
          "| フィード | 局面数 | うち 1 時間以上 | うち 1 日以上 | 最長 | 最長局面の最安 | 最長局面の開始 |",
          "|---|---:|---:|---:|---:|---:|---|"]
    episodes: list[dict] = []
    for f in d["feed"].unique(maintain_order=True):
        s = d.filter(pl.col("feed") == f).sort("ts")
        cur: list[dict] = []
        eps: list[dict] = []
        for r in s.iter_rows(named=True):
            if r["disc"] < BIG:
                cur.append(r)
            elif cur:
                eps.append({"feed": f, "t0": cur[0]["ts"], "t1": cur[-1]["ts"],
                            "n": len(cur), "worst": min(x["disc"] for x in cur)})
                cur = []
        if cur:
            eps.append({"feed": f, "t0": cur[0]["ts"], "t1": cur[-1]["ts"],
                        "n": len(cur), "worst": min(x["disc"] for x in cur)})
        for e in eps:
            e["hours"] = (e["t1"] - e["t0"]) / 3600.0
        episodes += eps
        if not eps:
            L.append(f"| {f} | 0 | 0 | 0 | — | — | — |")
            continue
        lg = max(eps, key=lambda e: e["hours"])
        L.append(f"| {f} | {len(eps)} | {sum(1 for e in eps if e['hours'] >= 1)} | "
                 f"{sum(1 for e in eps if e['hours'] >= 24)} | {lg['hours']:.1f}h | "
                 f"{lg['worst']*100:.2f}% | "
                 f"{dt.datetime.fromtimestamp(lg['t0'], dt.timezone.utc):%Y-%m-%d} |")
    L.append("")
    if episodes:
        L += ["### 長かった局面 上位 10", "",
              "| フィード | 開始(UTC) | 終了(UTC) | 長さ | 観測点 | 最安 |",
              "|---|---|---|---:|---:|---:|"]
        for e in sorted(episodes, key=lambda x: -x["hours"])[:10]:
            L.append(f"| {e['feed']} | "
                     f"{dt.datetime.fromtimestamp(e['t0'], dt.timezone.utc):%Y-%m-%d %H:%M} | "
                     f"{dt.datetime.fromtimestamp(e['t1'], dt.timezone.utc):%Y-%m-%d %H:%M} | "
                     f"{e['hours']:.1f}h | {e['n']} | {e['worst']*100:.2f}% |")
        L.append("")
        L += ["### 深いが短かった局面（最安 上位 10）", "",
              "深さだけで選ぶとこちらが並ぶ。**長さの列を見ること。**", "",
              "| フィード | 開始(UTC) | 長さ | 観測点 | 最安 |", "|---|---|---:|---:|---:|"]
        for e in sorted(episodes, key=lambda x: x["worst"])[:10]:
            L.append(f"| {e['feed']} | "
                     f"{dt.datetime.fromtimestamp(e['t0'], dt.timezone.utc):%Y-%m-%d %H:%M} | "
                     f"{e['hours']:.1f}h | {e['n']} | {e['worst']*100:.2f}% |")
        L.append("")

    L += ["## 最も割安だった日(フィードごとに上位 5 日)", ""]
    for f in d["feed"].unique(maintain_order=True):
        s = (d.filter(pl.col("feed") == f)
               .with_columns(pl.col("dt_utc").dt.date().alias("d"))
               .group_by("d").agg(pl.col("disc").min().alias("worst"),
                                  pl.col("price").mean().alias("px"),
                                  pl.col("redeem").mean().alias("rd"))
               .sort("worst").head(5))
        L += [f"### {f}", "", "| 日 | その日の最安乖離 | 市場価格 | 償還価値 |", "|---|---:|---:|---:|"]
        for r in s.iter_rows(named=True):
            L.append(f"| {r['d']} | {r['worst']*100:.2f}% | {r['px']:.6f} | {r['rd']:.6f} |")
        L.append("")

    L += ["## 読むときの注意", "",
          "- **これは執行可能な利益ではない。** 乖離はガス・スワップ手数料・",
          "  スリッページ・償還待ちの期間リスクを引く前の値である。",
          "  `DEFI_RESEARCH_THEMES.md` §24 の `Economic Significance` を出すには、",
          "  その乖離で**実際に約定できた数量**を板から見る必要がある",
          "- Chainlink は更新に閾値と時間の条件があり、**小さな変動は記録されない**。",
          "  よってここの乖離は実際の瞬間値より粗い。日中の一時的な大きい乖離は",
          "  取りこぼしている可能性がある",
          "- 償還価値は「いま償還したら受け取れる ETH」であって、",
          "  **即座に償還できるとは限らない**。Lido の出金待ち行列、Rocket Pool の",
          "  流動性上限などがあり、待ち時間そのものがリスクである", ""]

    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[write] {MD}")
    print()
    for f in d["feed"].unique(maintain_order=True):
        s = d.filter(pl.col("feed") == f)
        print(f"  {f:<12} 中央値 {s['disc'].median()*100:>7.3f}%  "
              f"最安 {s['disc'].min()*100:>7.2f}%  n={s.height:,}")


if __name__ == "__main__":
    main()
