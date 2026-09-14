r"""LST の「市場価格 / ETH」を Chainlink の全 phase から全期間取る。

    uv run python scripts/chainlink_lst_prices.py

出力: data/staking/lst_market_prices.parquet
      1 行 = 1 ラウンド。feed / phase / aggregator / block / ts / price

=============================================================================
★ なぜ DefiLlama の coins API ではなく Chainlink なのか
=============================================================================
DefiLlama の `coins.llama.fi/chart` は stETH と WETH の **USD 価格を別々に**
返す。両者の打刻はコインごとにずれる(実測: stETH 00:00:50 / WETH 00:00:39、
日によっては数分)。この 2 本を日付で寄せて比を取ると、ボラの高い日に
**存在しない乖離**が出る。

実際に踏んだ誤り: DefiLlama 経由では 2022-06-22 の stETH/ETH が -11.04% と
出たが、同じ日を Chainlink の stETH/ETH フィードで見ると **-5.32%** だった。
+3.9% という「stETH が ETH より高い」不自然な日も出ていた。これは打刻ずれの
ノイズであって市場の出来事ではない。

Chainlink の `<LST>/ETH` フィードは**比率そのもの**を 1 本の値として出すので、
この問題が原理的に起きない。

=============================================================================
★ このフィードは市場価格であって償還レートではない(検証済み)
=============================================================================
両者を混同すると「乖離がほぼゼロ」という偽の結論になる。

2026-09-14 に実測して確認した: stETH/ETH フィードは 2022-06-18 に **0.935019
(-6.50%)** を記録している。償還レートなら stETH はリベース型なので常に 1.0 の
はずで、そうなっていない。よってこれは二次市場の価格である。

償還レート側は scripts/onchain_lst_rates.py が別途取る。
**乖離 = 市場価格 / 償還レート - 1** がこの 2 本を突き合わせる目的である。

=============================================================================
★ phase をまたぐ
=============================================================================
Chainlink の proxy は裏のアグリゲータを差し替える。差し替えのたびに phaseId が
上がり、**古いアグリゲータのログは新しいアグリゲータには無い**。現在の
アグリゲータだけ見ると 2022 年のデペグが丸ごと落ちる。
`phaseAggregators(uint16)` で 1..phaseId を全部列挙すること。

★ answer は topics[1] に入る
`AnswerUpdated(int256 indexed current, uint256 indexed roundId, uint256 updatedAt)`
は current が indexed なので **data ではなく topics[1]**。int256 なので
2**255 以上は負数として戻す(価格が負になることは無いはずだが、黙って
巨大な正数として扱うと気づけない)。
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import time
import urllib.request

import polars as pl
from eth_utils import keccak

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "staking" / "lst_market_prices.parquet"

RPC = "https://gateway.tenderly.co/public/mainnet"
CHUNK = 50_000
PAUSE = 0.08

# proxy アドレス。裏のアグリゲータはここから引くのでハードコードしない。
FEEDS = {
    "stETH/ETH": "0x86392dC19c0b719886221c78AB11eb8Cf5c52812",
    "rETH/ETH":  "0x536218f9E9Eb48863970252233c8F271f554C2d0",
    "cbETH/ETH": "0xF017fcB346A1885194689bA23Eff2fE6fA5C483b",
    "weETH/ETH": "0x5c9C449BbC9a6075A2c061dF312a35fd1E05fF22",
}

# 走査開始。各フィードの開設より前でよい(空振りするだけ)。
# stETH/ETH が最古なのでそこに合わせ、全フィードを 1 回の getLogs でまとめて引く。
SINCE = 12_200_000      # 2021-04


def rpc(method: str, params: list, retry: int = 5):
    wait = 2.0
    for attempt in range(retry + 1):
        try:
            req = urllib.request.Request(
                RPC, method="POST",
                data=json.dumps({"jsonrpc": "2.0", "id": 1,
                                 "method": method, "params": params}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                o = json.loads(r.read())
            if "error" in o:
                raise RuntimeError(o["error"])
            return o["result"]
        except Exception as e:                                   # noqa: BLE001
            if attempt == retry:
                raise
            print(f"      RPC 再試行 {attempt + 1}/{retry} ({str(e)[:60]})", flush=True)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("unreachable")


def sel(sig: str) -> str:
    return "0x" + keccak(text=sig)[:4].hex()


def call(to: str, data: str) -> str:
    return rpc("eth_call", [{"to": to, "data": data}, "latest"])


def main() -> None:
    head = int(rpc("eth_blockNumber", []), 16)
    print(f"最新ブロック {head:,}\n")

    # --- 全 phase のアグリゲータを列挙 -------------------------------------
    owner: dict[str, tuple[str, int, int]] = {}   # aggregator(小文字) -> (feed, phase, decimals)
    for feed, proxy in FEEDS.items():
        dec = int(call(proxy, sel("decimals()")), 16)
        ph = int(call(proxy, sel("phaseId()")), 16)
        print(f"{feed:<12} decimals={dec}  phase={ph}")
        for i in range(1, ph + 1):
            d = sel("phaseAggregators(uint16)") + hex(i)[2:].rjust(64, "0")
            a = "0x" + call(proxy, d)[-40:]
            if int(a, 16) == 0:
                print(f"   phase {i}: (未設定)")
                continue
            owner[a.lower()] = (feed, i, dec)
            print(f"   phase {i}: {a}")
    print(f"\nアグリゲータ {len(owner)} 個をまとめて走査する\n")

    topic = "0x" + keccak(text="AnswerUpdated(int256,uint256,uint256)").hex()
    addrs = list(owner)
    rows: list[dict] = []
    frm = SINCE
    n_calls = 0
    span = head - SINCE

    while frm <= head:
        to = min(frm + CHUNK - 1, head)
        logs = rpc("eth_getLogs", [{"address": addrs, "topics": [topic],
                                    "fromBlock": hex(frm), "toBlock": hex(to)}])
        n_calls += 1
        for lg in logs:
            a = lg["address"].lower()
            if a not in owner:
                continue
            feed, phase, dec = owner[a]
            cur = int(lg["topics"][1], 16)
            if cur >= 2 ** 255:          # int256 の負数。起きないはずだが黙認しない
                cur -= 2 ** 256
            rows.append({
                "feed": feed, "phase": phase, "aggregator": a,
                "block": int(lg["blockNumber"], 16),
                "ts": int(lg["blockTimestamp"], 16) if lg.get("blockTimestamp") else None,
                "round_id": int(lg["topics"][2], 16),
                "price": cur / 10 ** dec,
            })
        if n_calls % 50 == 0:
            print(f"  {100 * (frm - SINCE) / span:5.1f}%  {len(rows):,} 件", flush=True)
        frm = to + 1
        time.sleep(PAUSE)

    if not rows:
        raise SystemExit("1 件も取れなかった")

    d = pl.DataFrame(rows).sort("feed", "block")
    neg = int((d["price"] <= 0).sum())
    if neg:
        print(f"★ 価格が 0 以下の行が {neg} 件ある。解釈を確認すること")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT, compression="zstd")

    print(f"\n[write] {OUT}  {d.height:,} 行 / RPC {n_calls} 回\n")
    print(f'{"フィード":<12}{"件数":>8}  {"期間":<26}{"最安":>10}{"最高":>10}  phase')
    for f in d["feed"].unique(maintain_order=True):
        s = d.filter(pl.col("feed") == f)
        a = dt.datetime.fromtimestamp(s["ts"].min(), dt.timezone.utc).date()
        b = dt.datetime.fromtimestamp(s["ts"].max(), dt.timezone.utc).date()
        phs = ",".join(str(x) for x in sorted(s["phase"].unique()))
        print(f'{f:<12}{s.height:>8,}  {a} 〜 {b}   '
              f'{s["price"].min():>9.6f}{s["price"].max():>10.6f}  {phs}')


if __name__ == "__main__":
    main()
