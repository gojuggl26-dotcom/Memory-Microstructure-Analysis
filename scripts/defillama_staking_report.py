r"""data/staking/candidates.parquet から ETH ステーキングプロトコル一覧を作る。

    uv run python scripts/defillama_staking_report.py

出力: reports/staking/README.md

=============================================================================
★ TVL を素朴に足してはいけない
=============================================================================
この 85 件は DefiLlama の 5 カテゴリ(Liquid Staking / Staking Pool /
Restaking / Liquid Restaking / Restaked BTC)を横断して拾ったもので、
**同じ ETH が複数のプロトコルで数えられている**。

  * DVT(SSV / Obol)は他プロトコルのバリデータを運用する基盤である。
    SSV の 13.1B は Lido などの預り分と重複する
  * Restaking / Liquid Restaking は LST を預かる。EigenCloud の 6.6B は
    stETH や weETH として既に Liquid Staking 側で数えられている

よって合計は**層ごとに出し、層をまたいで足さない**。

★ ETH かどうかはシンボル文字列で判定しない
DefiLlama の Liquid Staking カテゴリには SOL / BNB / AVAX / CRO / LINK の
ステーキングが混ざる。判定は fetch 側で tokensInUsd / tokens から
**実測した単価**が ETH 価格の [0.40, 2.00] 倍に入るかで行い、その結果が
`eth_share`(預り資産のうち ETH 系が占める割合)に入っている。
単価が出ないものは null = 判定不能とし、**黙って除外しない**。
"""
from __future__ import annotations

import datetime as dt
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "staking" / "candidates.parquet"
OUT = ROOT / "reports" / "staking" / "README.md"

# 取得日。DefiLlama の /protocols は現在値しか返さないので、この数字は
# この日の断面であって再取得しても再現しない。
FETCHED = "2026-09-13"

# eth_share の層。境界は「ほぼ全部 ETH」「混在」「ETH ではない」を分けるだけの
# 便宜的なもので、理論的な根拠はない。境界付近の銘柄は個別に確認すること。
TIERS = [
    ("A", "ETH 主体",   lambda c: c >= 0.90),
    ("B", "一部 ETH",   lambda c: (c >= 0.01) & (c < 0.90)),
    ("C", "ETH ではない", lambda c: c < 0.01),
]

# 他プロトコルの預り分と重複する層(合計から外す)
DVT_SLUGS = {"ssv-network", "obol"}


def fmt_usd(v: float) -> str:
    if v is None:
        return "—"
    if v >= 1e9:
        return f"${v/1e9:,.2f}B"
    if v >= 1e6:
        return f"${v/1e6:,.1f}M"
    if v >= 1e3:
        return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"


def fmt_listed(ts: int | None) -> str:
    if ts is None:
        return "—"
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m")


def table(d: pl.DataFrame) -> list[str]:
    if d.height == 0:
        return ["", "(該当なし)", ""]
    out = ["", "| # | プロトコル | カテゴリ | トークン | TVL | ETH 比率 | 展開チェーン数 | 登録 |",
           "|---:|---|---|---|---:|---:|---:|---|"]
    for i, r in enumerate(d.iter_rows(named=True), 1):
        sym = r["symbol"] if r["symbol"] and r["symbol"] != "-" else "—"
        share = "—" if r["eth_share"] is None else f"{r['eth_share']*100:.1f}%"
        name = r["name"]
        if r["url"]:
            name = f"[{name}]({r['url']})"
        out.append(f"| {i} | {name} | {r['category']} | {sym} | {fmt_usd(r['tvl_usd'])} | "
                   f"{share} | {r['n_chains']} | {fmt_listed(r['listed_at'])} |")
    out.append("")
    return out


def main() -> None:
    d = pl.read_parquet(SRC).sort("tvl_usd", descending=True)
    L: list[str] = []
    A = L.append

    A("# ETH ステーキングプロトコル一覧")
    A("")
    A(f"出典 DefiLlama / 取得日 **{FETCHED}** / 対象 **{d.height} プロトコル**")
    A("")
    A("取得は `scripts/defillama_staking_fetch.py`、この一覧は "
      "`scripts/defillama_staking_report.py` が生成する。")
    A("DefiLlama の `/protocols` は現在値しか返さないので、ここの数字は"
      "**この日の断面**であって再取得しても再現しない。")
    A("生データを版管理に入れてあるのはそのためである(`data/staking/`)。")
    A("")

    # ---- 索引 ---------------------------------------------------------------
    A("## この一覧と対になる資料")
    A("")
    A("| 資料 | 中身 | 生成 |")
    A("|---|---|---|")
    A("| [利回り履歴](yields.md) | DefiLlama の日次 APY。45 プール・2022-05 以降 | "
      "`defillama_yield_history.py` → `yield_history_report.py` |")
    A("| [市場価格と償還価値の乖離](discount.md) | 主要 LST 4 本。"
      "Chainlink の市場価格とオンチェーンの償還レートの差 | "
      "`chainlink_lst_prices.py` + `onchain_lst_rates.py` → `lst_discount_analysis.py` |")
    A("")

    # ---- 読み方の前提 -------------------------------------------------------
    A("## 読み方の前提(ここを外すと数字が意味を失う)")
    A("")
    A("### 1. TVL を層をまたいで足してはいけない")
    A("")
    A("同じ ETH が複数のプロトコルで数えられている。")
    A("")
    A("- **DVT**(SSV / Obol)は他プロトコルのバリデータを運用する基盤である。"
      "SSV の TVL は Lido などの預り分と重複する")
    A("- **Restaking / Liquid Restaking** は LST を預かる。"
      "EigenCloud の預りは stETH や weETH として既に Liquid Staking 側で数えられている")
    A("")
    A("### 2. ETH かどうかはシンボル文字列で判定していない")
    A("")
    A("DefiLlama の Liquid Staking カテゴリには SOL / BNB / AVAX / CRO / LINK の"
      "ステーキングが混ざる。Ethereum 上で発行されているだけで原資産が ETH でない"
      "ものも多い(Lido は WMATIC も持つ)。")
    A("判定は `tokensInUsd / tokens` から**実測した単価**が ETH 価格の "
      "[0.40, 2.00] 倍に入るかで行い、その割合が `ETH 比率` 列である。")
    A("単価が出ないものは**判定不能**として別掲し、黙って除外していない。")
    A("")
    A("### 3. TVL は預り資産であって収益ではない")
    A("")
    A("手数料・収益は `data/staking/fees_eth_*.json`、利回りは "
      "`data/staking/yields.json` に落としてあるが、この一覧には結合していない。")
    A("")

    # ---- 層別の内訳 ---------------------------------------------------------
    known = d.filter(pl.col("eth_share").is_not_null())
    unknown = d.filter(pl.col("eth_share").is_null())

    A("## 内訳")
    A("")
    A("| 層 | 定義 | 件数 | TVL 合計 |")
    A("|---|---|---:|---:|")
    tier_frames: dict[str, pl.DataFrame] = {}
    for key, label, pred in TIERS:
        sub = known.filter(pred(pl.col("eth_share")))
        tier_frames[key] = sub
        A(f"| {key} | {label} | {sub.height} | {fmt_usd(sub['tvl_usd'].sum())} |")
    A(f"| D | 判定不能 | {unknown.height} | {fmt_usd(unknown['tvl_usd'].sum())} |")
    A(f"| | **計** | **{d.height}** | {fmt_usd(d['tvl_usd'].sum())} |")
    A("")

    # ---- 二重計上を除いた ETH の規模 -----------------------------------------
    a = tier_frames["A"]
    dvt = a.filter(pl.col("slug").is_in(DVT_SLUGS))
    rest = a.filter(pl.col("category").str.contains("Restak"))
    base = a.filter(~pl.col("slug").is_in(DVT_SLUGS) & ~pl.col("category").str.contains("Restak"))
    A("### 層 A から重複を除く")
    A("")
    A("| 区分 | 件数 | TVL | 扱い |")
    A("|---|---:|---:|---|")
    A(f"| 素の預り(LST・ステーキングプール) | {base.height} | {fmt_usd(base['tvl_usd'].sum())} | **これが ETH の実体規模に最も近い** |")
    A(f"| DVT(他プロトコルのバリデータを運用) | {dvt.height} | {fmt_usd(dvt['tvl_usd'].sum())} | 重複。足さない |")
    A(f"| Restaking 系(LST を預かる) | {rest.height} | {fmt_usd(rest['tvl_usd'].sum())} | 重複。足さない |")
    A("")
    A("つまり「ETH ステーキングの総額」として引用してよいのは "
      f"**{fmt_usd(base['tvl_usd'].sum())}** であって、"
      f"85 件の素朴な合計 {fmt_usd(d['tvl_usd'].sum())} ではない。")
    A("")

    # ---- 各層の一覧 ---------------------------------------------------------
    for key, label, _ in TIERS:
        sub = tier_frames[key]
        A(f"## 層 {key} — {label}({sub.height} 件)")
        if key == "A":
            A("")
            A("預り資産のほぼ全部が ETH 系。**このプロジェクトで対象にするならここ。**")
        elif key == "B":
            A("")
            A("ETH と他資産が混在する。ETH 部分だけを見たいなら按分が要る。")
        elif key == "C":
            A("")
            A("ETH 系の預りが検出されなかった。カテゴリ上は拾われたが"
              "**他チェーンのステーキング**である。対象外だが、除外した証拠として残す。")
        L.extend(table(sub))

    A(f"## 層 D — 判定不能({unknown.height} 件)")
    A("")
    A("`tokensInUsd / tokens` から単価が出せず、ETH かどうかを判定できなかった。")
    A("TVL が極小のものが大半だが、**Lombard LBTC だけは TVL が大きい**"
      "(BTC 系なので ETH ではないと思われるが、実測で確かめていない)。")
    A("黙って除外せず、ここに出しておく。")
    L.extend(table(unknown))

    A("## 未着手")
    A("")
    A("- 手数料・収益(`fees_eth_*.json`)と利回り(`yields.json`)をこの一覧に結合していない。"
      "結合すれば「預りの規模」ではなく「取り分の大きさ」で並べ替えられる")
    A("- 層 D の 9 件を一次ソース(各プロトコルのコントラクト)で確かめていない")
    A("- 親での名寄せをしていない。6 組が同じ親を持つ(Swell が LST と LRT に分かれる等)が、"
      "そのほとんどは Restaking 側を層 A の合計から外した時点で解消しており、"
      "残る影響は $136M 未満(素の預り $43.05B の 0.3% 未満)である")
    A("- TVL は断面のみ。時系列は DefiLlama の有料 API が要る")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[write] {OUT}  ({len('\n'.join(L)):,} 文字)")
    print(f"  層A {tier_frames['A'].height} / 層B {tier_frames['B'].height} / "
          f"層C {tier_frames['C'].height} / 層D {unknown.height}")
    print(f"  素の預り合計 = {fmt_usd(base['tvl_usd'].sum())}")


if __name__ == "__main__":
    main()
