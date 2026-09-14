r"""利回り履歴(DefiLlama)を要約して一覧にする。

    uv run python scripts/yield_history_report.py

入力: data/staking/yield_history.parquet  (defillama_yield_history.py)
      data/staking/yields.json            (プールの素性)
出力: reports/staking/yields.md

=============================================================================
★ プールを project+symbol で束ねてはいけない
=============================================================================
同じ project と symbol で**別のプールが複数ある**。実測:

    binance-staked-eth / WBETH   2 プール(Ethereum と BSC)
    ether.fi-stake     / WEETH   4 プール(Ethereum / Linea / Base / Scroll)

これを group_by で束ねると日数が二重三重に数えられる。実際 WBETH の
「日数」が 2,374 日と出たが、収録開始が 2023-06-16 なので物理的にありえない
(2 プール × 1,187 日)。**集計の主キーは pool(UUID)である。**

=============================================================================
★ ETH かどうかは underlyingTokens で判定する(シンボル文字列で判定しない)
=============================================================================
45 プール全件に underlyingTokens が入っている。symbol に "ETH" が入るかで
判定すると、ANKRBNB / VKSM / MATICX のような他チェーンのステーキングを
取りこぼすか誤って拾う。

判定できなかったプールは**捨てずに別掲し、その underlyingTokens を出す**。
そうしておけば、後から ETH 系だと分かったときに ETH_TOKENS に足すだけで済む。

=============================================================================
★ APY に明らかな異常値がある
=============================================================================
nodedao/NETH に apy = 25,806% の日がある(1,140 日中 9 日)。TVL は 4,000 万ドルで
正常なので、DefiLlama 側の利回り計算の不良である。

**黙って落とすと「異常値が無かった」ことになってしまう**ので、
異常値は除外した統計と、異常値そのものの一覧を両方出す。
"""
from __future__ import annotations

import json
import pathlib

import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "staking" / "yield_history.parquet"
POOLS = ROOT / "data" / "staking" / "yields.json"
MD = ROOT / "reports" / "staking" / "yields.md"

# ETH 系の原資産。小文字で比較する。
ETH_TOKENS = {
    "0x0000000000000000000000000000000000000000",   # native ETH
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",   # WETH
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",   # stETH
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",   # wstETH
    "0xae78736cd615f374d3085123a210448e74fc6393",   # rETH
    "0xbe9895146f7af43049ca1c1ae358b0541ea49704",   # cbETH
    "0x35fa164735182de50811e8e2e824cfb9b6118ac2",   # eETH
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee",   # weETH
    "0xa35b1b31ce002fbf2058d22f30f95d405200a15b",   # ETHx
    "0x856c4efb76c1d1ae02e20ceb03a2a6a08b0b8dc3",   # OETH
}

APY_ABSURD = 100.0      # これを超える apy は異常として別扱い


def main() -> None:
    h = pl.read_parquet(HIST)
    pools = json.loads(POOLS.read_text(encoding="utf-8"))

    meta = {}
    for p in pools:
        ut = [t.lower() for t in (p.get("underlyingTokens") or [])]
        meta[p["pool"]] = {
            "is_eth": any(t in ETH_TOKENS for t in ut),
            "tokens": ut,
            "tvl": p["tvlUsd"],
        }
    eth_pools = {k for k, v in meta.items() if v["is_eth"]}
    print(f"全 {h['pool'].n_unique()} プール中 ETH 系と判定 {len(eth_pools)}")

    h = h.with_columns(pl.col("pool").is_in(eth_pools).alias("is_eth"))
    absurd = h.filter(pl.col("apy") > APY_ABSURD)
    clean = h.filter(pl.col("apy").is_null() | (pl.col("apy") <= APY_ABSURD))

    L = ["# LST / ステーキングの利回り履歴", "",
         f"出典 DefiLlama `yields.llama.fi/chart/<pool>`(無料)。",
         f"全 **{h.height:,} 行 / {h['pool'].n_unique()} プール / "
         f"{h['date'].min()} 〜 {h['date'].max()}**。", "",
         "取得は `scripts/defillama_yield_history.py`、この要約は "
         "`scripts/yield_history_report.py` が作る。", "",
         "## 読み方の前提", "",
         "- **集計の主キーはプール(UUID)である。** 同じ project+symbol で別チェーンに",
         "  複数プールがある(WBETH は Ethereum と BSC、WEETH は 4 チェーン)。",
         "  束ねると日数が二重に数えられる",
         "- **ETH かどうかは `underlyingTokens` で判定した。** symbol の文字列では判定",
         "  していない",
         f"- **開始日は DefiLlama の収録開始日であって、プロトコルの開始日ではない。**",
         "  Lido は 2020-12 から動いているが履歴は 2022-05-03 から。それ以前が要るなら",
         "  オンチェーンから取る(`scripts/onchain_lst_rates.py`)",
         f"- apy が {APY_ABSURD:.0f}% を超える行は異常値として統計から外し、末尾に列挙した", ""]

    # ---- ETH 系 -----------------------------------------------------------
    for flag, title in [(True, "ETH 系"), (False, "ETH 系ではない / 判定できなかった")]:
        sub = clean.filter(pl.col("is_eth") == flag)
        if sub.height == 0:
            continue
        g = (sub.group_by("pool", "project", "symbol", "chain")
                .agg(pl.len().alias("n"),
                     pl.col("date").min().alias("first"),
                     pl.col("date").max().alias("last"),
                     pl.col("apy").median().alias("med"),
                     pl.col("apy").quantile(0.05).alias("q05"),
                     pl.col("apy").quantile(0.95).alias("q95"),
                     pl.col("apy").std().alias("sd"),
                     pl.col("tvlUsd").last().alias("tvl"))
                .sort("tvl", descending=True))
        L += [f"## {title}({g.height} プール)", "",
              "| プロジェクト | シンボル | チェーン | 日数 | 期間 | APY 中央値 | 5%点 | 95%点 | 標準偏差 | TVL |",
              "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in g.iter_rows(named=True):
            tv = (f"${r['tvl']/1e9:.2f}B" if r["tvl"] >= 1e9
                  else f"${r['tvl']/1e6:,.1f}M" if r["tvl"] >= 1e6
                  else f"${r['tvl']/1e3:,.0f}K")
            def f(x):
                return "—" if x is None else f"{x:.2f}%"
            L.append(f"| {r['project']} | {r['symbol']} | {r['chain']} | {r['n']:,} | "
                     f"{r['first']}〜{r['last']} | {f(r['med'])} | {f(r['q05'])} | "
                     f"{f(r['q95'])} | {f(r['sd'])} | {tv} |")
        L.append("")
        if not flag:
            L += ["判定できなかったプールの `underlyingTokens`:", ""]
            for r in g.iter_rows(named=True):
                tk = ", ".join(meta[r["pool"]]["tokens"][:4]) or "(なし)"
                L.append(f"- `{r['project']}/{r['symbol']}/{r['chain']}` … {tk}")
            L.append("")

    # ---- 異常値 -----------------------------------------------------------
    L += [f"## 異常値(apy > {APY_ABSURD:.0f}%)", ""]
    if absurd.height == 0:
        L += ["該当なし。", ""]
    else:
        g = (absurd.group_by("project", "symbol", "chain")
                   .agg(pl.len().alias("n"), pl.col("apy").max().alias("mx"))
                   .sort("mx", descending=True))
        L += ["統計から外した。DefiLlama 側の利回り計算の不良と思われる"
              "(TVL は正常な値のまま大きくなっている)。", "",
              "| プロジェクト | シンボル | チェーン | 該当日数 | 最大 apy |", "|---|---|---|---:|---:|"]
        for r in g.iter_rows(named=True):
            L.append(f"| {r['project']} | {r['symbol']} | {r['chain']} | "
                     f"{r['n']} | {r['mx']:,.0f}% |")
        L.append("")
        L += ["最も大きい 10 件:", "",
              "| 日 | プロジェクト | シンボル | apy | TVL |", "|---|---|---|---:|---:|"]
        for r in absurd.sort("apy", descending=True).head(10).iter_rows(named=True):
            L.append(f"| {r['date']} | {r['project']} | {r['symbol']} | "
                     f"{r['apy']:,.0f}% | ${r['tvlUsd']/1e6:,.1f}M |")
        L.append("")

    # ---- pricePerShare の有無 ---------------------------------------------
    pps = (h.group_by("project", "symbol", "chain")
            .agg(pl.len().alias("n"),
                 pl.col("pricePerShare").is_not_null().sum().alias("has"))
            .with_columns((pl.col("has") / pl.col("n")).alias("r"))
            .sort("r"))
    none_pps = pps.filter(pl.col("has") == 0)
    L += ["## pricePerShare が無いプール", "",
          f"{none_pps.height}/{pps.height} プールで `pricePerShare` が全期間 null。",
          "これが無いと DefiLlama の apy 定義をそのまま使うしかなく、"
          "**自分で利回りを再計算できない**。",
          "Lido と Rocket Pool はどちらも null なので、両者の利回りを厳密に",
          "扱いたければオンチェーンから取ること。", ""]

    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[write] {MD}")
    print(f"  ETH 系 {clean.filter(pl.col('is_eth'))['pool'].n_unique()} プール / "
          f"非ETH・判定不能 {clean.filter(~pl.col('is_eth'))['pool'].n_unique()} プール")
    print(f"  異常値 {absurd.height} 行 (apy > {APY_ABSURD:.0f}%)")
    print(f"  pricePerShare が全期間 null: {none_pps.height} プール")


if __name__ == "__main__":
    main()
