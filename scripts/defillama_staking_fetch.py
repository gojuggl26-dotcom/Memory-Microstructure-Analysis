r"""DefiLlama から ETH ステーキング系プロトコルを網羅取得する。

    uv run python scripts/defillama_staking_fetch.py
    uv run python scripts/defillama_staking_fetch.py --refresh

出力(すべて data/staking/):
    protocols_raw.json   … /protocols の生
    detail/<slug>.json   … /protocol/<slug> の抜粋(履歴は捨てて最新断面のみ)
    fees_eth.json        … /overview/fees/ethereum(手数料・収益)
    yields.json          … yields.llama.fi/pools の該当分
    candidates.parquet   … 候補一覧(分類つき)

=============================================================================
★何を「ETH ステーキング」と呼ぶか — シンボル文字列で判定しない
=============================================================================
DefiLlama の Liquid Staking カテゴリには SOL / BNB / AVAX / CRO / LINK の
ステーキングも入っている。`chains` に Ethereum が入っていても、**Ethereum 上で
発行されているだけで原資産が ETH でない**ものが多い(例: Lido は WMATIC も持つ)。

シンボルに "ETH" が入るかで判定すると取りこぼす(`ankrETH` は拾えるが
`WBETH` `cbETH` は拾えても `LsETH` の亜種や独自名は落ちる)し、
誤って拾う(`ETHFI` はガバナンストークン)。

そこで **実測した単価**で判定する:

    単価 = tokensInUsd[token] / tokens[token]      … DefiLlama が両方くれる
    ETH 族 = 単価が ETH 価格の [0.40, 2.00] 倍に入る

LST は報酬を積むので ETH より高い(rETH ≒ 1.1、weETH ≒ 1.07)。
逆に割れることは通常ないが、値崩れした LST を落とさないよう下限は緩めに取る。
**単価が出ない(tokens が空の)プロトコルは「判定不能」として別に出す。
黙って除外しない。**

★この判定は**現在の断面**に対するもの。過去に別の資産を持っていた可能性は
  見ていない(この一覧は「今どうなっているか」の棚卸しである)。

=============================================================================
★二重計上について(先に書いておく)
=============================================================================
- **DVT(SSV / Obol)は他プロトコルのバリデータを数えている。**
  Lido の一部は SSV 上で動くので、合計すると二重になる
- **Restaking(EigenLayer)は LST を預かる。**Lido の stETH が EigenLayer にも
  計上される。LRT(ether.fi / Renzo / Kelp)はさらにその上に乗る
- DefiLlama 自身も `parentProtocol` / `doublecounted` を持つが、
  **カテゴリ間の重なりまでは打ち消さない**
合計を出すときは必ず層を分けること。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "staking"
DET = OUT / "detail"
UA = {"User-Agent": "Mozilla/5.0"}
PACE = 0.25

# ステーキングに関係しうるカテゴリ(広めに取って後で絞る)
CATS = {"Liquid Staking", "Liquid Restaking", "Restaking", "Staking Pool",
        "Staking Rental", "Restaked BTC"}
ETH_LO, ETH_HI = 0.40, 2.00        # ETH 価格に対する単価の許容比


def get(url: str, tries: int = 5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 402:
                return {"_paywalled": True}
            if e.code in (429, 502, 503, 504) and i < tries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            if e.code == 404:
                return None
            raise
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return None


def eth_price() -> float:
    j = get("https://coins.llama.fi/prices/current/coingecko:ethereum")
    return float(j["coins"]["coingecko:ethereum"]["price"])


def slim(j: dict) -> dict:
    """履歴を捨てて最新断面だけ残す(1 件 2MB → 数 KB)。"""
    out = {k: j.get(k) for k in (
        "id", "name", "slug", "symbol", "url", "category", "chain", "chains",
        "description", "methodology", "parentProtocol", "misrepresentedTokens",
        "gecko_id", "cmcId", "mcap", "audits", "audit_links", "openSource",
        "github", "twitter", "listedAt", "hallmarks", "treasury")}
    out["currentChainTvls"] = j.get("currentChainTvls") or {}
    out["tvl_now"] = (j.get("tvl") or [{}])[-1].get("totalLiquidityUSD")
    ct = (j.get("chainTvls") or {}).get("Ethereum") or {}
    for k in ("tokens", "tokensInUsd"):
        arr = ct.get(k) or []
        out[f"eth_{k}"] = arr[-1] if arr else None
    out["raises_usd"] = sum((r.get("amount") or 0) for r in (j.get("raises") or []))
    out["n_raises"] = len(j.get("raises") or [])
    out["hacks_usd"] = sum((h.get("amount") or 0) for h in (j.get("hacks") or []))
    out["n_hacks"] = len(j.get("hacks") or [])
    return out


def classify(d: dict, px: float):
    """Ethereum 上の token 内訳から、ETH 族が占める割合を出す。"""
    tk = (d.get("eth_tokens") or {}).get("tokens") or {}
    us = (d.get("eth_tokensInUsd") or {}).get("tokens") or {}
    if not tk or not us:
        return None, None, ""
    tot = sum(v for v in us.values() if v and v > 0)
    if tot <= 0:
        return None, None, ""
    eth_usd, parts = 0.0, []
    for sym, amt in tk.items():
        u = us.get(sym)
        if not u or not amt or amt <= 0:
            continue
        unit = u / amt
        is_eth = ETH_LO * px <= unit <= ETH_HI * px
        if is_eth:
            eth_usd += u
        parts.append((sym, u, unit, is_eth))
    parts.sort(key=lambda x: -x[1])
    top = " / ".join(f"{s}{'*' if e else ''} {v/tot*100:.0f}%"
                     for s, v, _, e in parts[:4])
    return eth_usd / tot, tot, top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="detail/*.json を取り直す")
    a = ap.parse_args()
    DET.mkdir(parents=True, exist_ok=True)

    px = eth_price()
    print(f"ETH 価格 ${px:,.2f}(coins.llama.fi)", flush=True)

    P = get("https://api.llama.fi/protocols")
    (OUT / "protocols_raw.json").write_text(json.dumps(P), encoding="utf-8")
    print(f"/protocols {len(P):,} 件", flush=True)

    cand = [x for x in P if x.get("category") in CATS
            and "Ethereum" in (x.get("chains") or [])]
    cand.sort(key=lambda x: -(x.get("tvl") or 0))
    print(f"候補(ステーキング系 × Ethereum 展開)= {len(cand)} 件", flush=True)

    rows = []
    for i, c in enumerate(cand, 1):
        f = DET / f"{c['slug']}.json"
        if a.refresh or not f.exists():
            j = get(f"https://api.llama.fi/protocol/{c['slug']}")
            time.sleep(PACE)
            if not j:
                print(f"  [取得不可] {c['slug']}")
                continue
            f.write_text(json.dumps(slim(j)), encoding="utf-8")
        d = json.loads(f.read_text(encoding="utf-8"))
        share, tot, top = classify(d, px)
        rows.append({
            "name": c["name"], "slug": c["slug"], "category": c["category"],
            "symbol": c.get("symbol"), "tvl_usd": c.get("tvl"),
            "tvl_eth_chain": (d.get("currentChainTvls") or {}).get("Ethereum"),
            "eth_share": share, "eth_tokens_usd": tot, "token_mix": top,
            "n_chains": len(c.get("chains") or []),
            "chains": ",".join((c.get("chains") or [])[:6]),
            "parent": d.get("parentProtocol"),
            "mcap": c.get("mcap"), "listed_at": c.get("listedAt"),
            "raises_usd": d.get("raises_usd"), "n_hacks": d.get("n_hacks"),
            "hacks_usd": d.get("hacks_usd"),
            "misrepresented": bool(c.get("misrepresentedTokens")),
            "url": c.get("url"),
            "change_7d": c.get("change_7d"),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(cand)}", flush=True)

    D = pl.DataFrame(rows, infer_schema_length=None)
    D.write_parquet(OUT / "candidates.parquet")
    n_eth = D.filter(pl.col("eth_share") >= 0.5).height
    n_unk = D.filter(pl.col("eth_share").is_null()).height
    print(f"\n候補 {D.height} 件 → ETH 族が過半 {n_eth} 件 / "
          f"判定不能(token 内訳なし){n_unk} 件")

    # ---- 手数料・収益(Ethereum)----
    for kind in ("dailyFees", "dailyRevenue", "dailyHoldersRevenue"):
        j = get("https://api.llama.fi/overview/fees/ethereum"
                f"?excludeTotalDataChart=true"
                f"&excludeTotalDataChartBreakdown=true&dataType={kind}")
        (OUT / f"fees_eth_{kind}.json").write_text(json.dumps(j),
                                                   encoding="utf-8")
        n = len((j or {}).get("protocols") or [])
        print(f"fees/ethereum {kind}: {n} プロトコル", flush=True)
        time.sleep(PACE)

    # ---- 利回り(現在値のみ無料)----
    Y = get("https://yields.llama.fi/pools")
    pools = (Y or {}).get("data") or []
    keep = {r["slug"] for r in rows}
    sel = [p for p in pools if (p.get("project") in keep)]
    (OUT / "yields.json").write_text(json.dumps(sel), encoding="utf-8")
    print(f"yields: 全 {len(pools):,} プール中 該当 {len(sel):,}")
    print(f"\n書き出し {OUT}")


if __name__ == "__main__":
    main()
