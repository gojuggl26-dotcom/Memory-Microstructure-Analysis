r"""主要 LST の対 ETH レートを、オンチェーンのイベントから全期間取る。

    uv run python scripts/onchain_lst_rates.py
    uv run python scripts/onchain_lst_rates.py --only stETH rETH

出力: data/staking/lst_onchain_rates.parquet
      1 行 = 1 レート更新。lst / label / kind / block / ts / rate / apr_pct

=============================================================================
なぜオンチェーンから取るのか
=============================================================================
DefiLlama の yields chart(scripts/defillama_yield_history.py)は無料で日次
履歴を返すが、**収録開始日より前が無い**。Lido は 2020-12 から動いているのに
履歴は 2022-05-03 から、Rocket Pool は 2021-11 稼働で履歴は 2023-01-16 から。
加えて pricePerShare が半分以上 null なので、DefiLlama の apy 定義に依存せず
自分でレートを再計算することができない。

イベントログなら **デプロイ時点から全期間**、かつ**厳密に**取れる。
archive node も要らない(eth_getLogs はログのインデックスを引くだけ)。

=============================================================================
★ 引数が indexed だと data から抜ける - 語の位置を憶測で書かないこと
=============================================================================
実際に踏んだ誤り: Rocket Pool の

    event BalancesUpdated(uint256 block, uint256 slotTimestamp, uint256 totalEth,
                          uint256 stakingEth, uint256 rethSupply, uint256 time)

は 6 引数だが **block が indexed** なので data は 5 語しかない。6 語として
読むと totalEth/rethSupply のつもりで stakingEth/time を割ることになり、
レートが 2e14 という無意味な値になった。**topics 数を必ず確認すること。**
(topics[0] は署名なので、indexed 引数の数 = len(topics) - 1)

下の n_data_words は実測値である。最初の 1 件で照合し、違ったら止める。

=============================================================================
LST は 2 種類ある
=============================================================================
* リベース型(Lido stETH / ether.fi eETH): 保有数量が増える。
  イベントに総 ETH と総シェアが入るので、その比が 1 シェアあたりの ETH。
* レート型(rETH / cbETH / wBETH): 1 トークンあたりの ETH が増える。
  更新イベントに新レートがそのまま入る。

どちらも「1 単位あたり何 ETH か」に正規化して同じ rate 列に入れる。

2026-09-14 の実測値(健全性の目安。大きく外れたら解釈を疑う):
    stETH 1.243946 / rETH 1.171722 / cbETH 1.139396 / wBETH 1.106404 / eETH 1.103636
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time
import urllib.error
import urllib.request

import polars as pl
from eth_utils import keccak

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "staking" / "lst_onchain_rates.parquet"

RPC = "https://gateway.tenderly.co/public/mainnet"
CHUNK = 50_000          # defi-rates/chains.py の実測値(ethereum の公開枠)
PAUSE = 0.08

ROCKET_STORAGE = "0x1d8f8f00cfa6758d7bE78336684788Fb0ee0Fa46"


def t0(sig: str) -> str:
    return "0x" + keccak(text=sig).hex()


SPECS: dict[str, dict] = {
    "stETH": dict(
        label="Lido stETH", kind="rebase",
        addr="0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
        sig="TokenRebased(uint256,uint256,uint256,uint256,uint256,uint256,uint256)",
        n_data_words=6,                 # reportTimestamp が indexed
        rate=lambda w: w[4] / w[3],     # postTotalEther / postTotalShares
        since=11_473_216,               # stETH デプロイ(2020-12-18)
    ),
    # ★ rocketNetworkBalances の BalancesUpdated は使わない。
    #   Rocket Pool はコントラクトをアップグレードしており、rocketStorage が返すのは
    #   **現行アドレスだけ**である。そこを見ると 2026-02 以降の 209 件しか取れず、
    #   2021-10〜2026-02 が丸ごと落ちる(実測済み)。
    #   rETH トークンのアドレスは不変なので、トークン側の mint/burn から取る。
    #   burn/mint は交換レートそのもので実行されるため ethAmount/amount がレートになる。
    #   (実測 2026-09-14: トークン側 1.171714 / rocketNetworkBalances 1.171722 で一致)
    "rETH": dict(
        label="Rocket Pool rETH", kind="rate",
        addr="0xae78736Cd615f374D3085123A210448E74Fc6393",
        sig=["TokensBurned(address,uint256,uint256,uint256)",
             "TokensMinted(address,uint256,uint256,uint256)"],
        n_data_words=3,                 # from/to が indexed なので 3 語
        rate=lambda w: w[1] / w[0],     # ethAmount / amount
        since=13_325_304,               # rETH の最初の mint は 2021-10-02(実測)
        chunk=20_000,                   # burn が 300 件/日 あるので範囲を狭める
    ),
    "cbETH": dict(
        label="Coinbase cbETH", kind="rate",
        addr="0xBe9895146f7AF43049ca1c1AE358B0541Ea49704",
        sig="ExchangeRateUpdated(address,uint256)",
        n_data_words=1,                 # oracle が indexed
        rate=lambda w: w[0] / 1e18,
        since=15_223_000,               # 2022-08
    ),
    "wBETH": dict(
        label="Binance wBETH", kind="rate",
        addr="0xa2E3356610840701BDf5611a53974510Ae27E2e1",
        sig="ExchangeRateUpdated(address,uint256)",
        n_data_words=1,
        rate=lambda w: w[0] / 1e18,
        since=17_100_000,               # 2023-04
    ),
    "eETH": dict(
        label="ether.fi eETH", kind="rebase",
        addr="0x308861A430be4cce5502d0A12724771Fc6DaF216",   # LiquidityPool
        sig="Rebase(uint256,uint256)",
        n_data_words=2,
        rate=lambda w: w[0] / w[1],     # totalEthLocked / totalEEthShares
        since=18_450_000,               # 2023-11
    ),
}


def rpc(method: str, params: list, retry: int = 5):
    wait = 2.0
    for attempt in range(retry + 1):
        req = urllib.request.Request(
            RPC, method="POST",
            data=json.dumps({"jsonrpc": "2.0", "id": 1,
                             "method": method, "params": params}).encode(),
            headers={"Content-Type": "application/json"})
        try:
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


def words(hexdata: str) -> list[int]:
    d = hexdata[2:]
    return [int(d[64 * i: 64 * (i + 1)], 16) for i in range(len(d) // 64)]


def resolve_rocket(name: str) -> str:
    """Rocket Pool のアドレスはハードコードしない。rocketStorage から引く。"""
    k = "0x" + keccak(b"contract.address" + name.encode()).hex()
    sel = "0x" + keccak(text="getAddress(bytes32)")[:4].hex()
    res = rpc("eth_call", [{"to": ROCKET_STORAGE, "data": sel + k[2:]}, "latest"])
    return "0x" + res[-40:]


def fetch(key: str, spec: dict, head: int) -> pl.DataFrame:
    addr = spec["addr"] or resolve_rocket(spec["resolve"])
    # sig は 1 本でも複数でもよい。複数なら topic0 の OR として引く。
    sigs = spec["sig"] if isinstance(spec["sig"], list) else [spec["sig"]]
    topic = [t0(s) for s in sigs]
    chunk = spec.get("chunk", CHUNK)
    rows: list[dict] = []
    frm = spec["since"]
    span = head - spec["since"]
    n_calls = 0
    checked = False

    while frm <= head:
        to = min(frm + chunk - 1, head)
        logs = rpc("eth_getLogs", [{"address": addr, "topics": [topic],
                                    "fromBlock": hex(frm), "toBlock": hex(to)}])
        n_calls += 1
        for lg in logs:
            w = words(lg["data"])
            if not checked:
                # 語数を実測と突き合わせる。違ったら黙って続けず止める。
                if len(w) != spec["n_data_words"]:
                    raise SystemExit(
                        f"{key}: data の語数が想定と違う "
                        f"(実測 {len(w)} / 想定 {spec['n_data_words']}, "
                        f"topics={len(lg['topics'])})。indexed 引数を確認すること。")
                checked = True
            rows.append({
                "lst": key,
                "label": spec["label"],
                "kind": spec["kind"],
                "block": int(lg["blockNumber"], 16),
                "ts": int(lg["blockTimestamp"], 16) if lg.get("blockTimestamp") else None,
                "rate": float(spec["rate"](w)),
            })
        if n_calls % 50 == 0:
            print(f"    {key}: {100 * (frm - spec['since']) / span:5.1f}%  "
                  f"{len(rows):,} 件", flush=True)
        frm = to + 1
        time.sleep(PAUSE)

    if not rows:
        print(f"    {key}: 0 件", flush=True)
        return pl.DataFrame()

    d = pl.DataFrame(rows).sort("block")
    # 直前の点からの実現 APR。ts が無い点は落とさず APR だけ null にする。
    d = d.with_columns(
        pl.col("rate").shift(1).alias("_pr"),
        pl.col("ts").shift(1).alias("_pt"),
    ).with_columns(
        pl.when(pl.col("_pt").is_not_null() & (pl.col("ts") > pl.col("_pt")))
        .then((pl.col("rate") / pl.col("_pr") - 1.0)
              * (365 * 86400) / (pl.col("ts") - pl.col("_pt")) * 100.0)
        .otherwise(None).alias("apr_pct")
    ).drop("_pr", "_pt")
    print(f"    {key}: RPC {n_calls} 回 / {d.height:,} 件", flush=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    head = int(rpc("eth_blockNumber", []), 16)
    print(f"最新ブロック {head:,}\n")

    frames = []
    for k in (a.only or list(SPECS)):
        spec = SPECS[k]
        print(f"[{spec['label']}] since block {spec['since']:,}", flush=True)
        d = fetch(k, spec, head)
        if d.height:
            frames.append(d)
        print(flush=True)

    if not frames:
        raise SystemExit("1 件も取れなかった")

    # --only で一部だけ取り直したときは、取らなかった銘柄を既存ファイルから残す。
    # (これをしないと --only rETH のたびに他の 4 本が消える)
    if a.only and OUT.exists():
        old = pl.read_parquet(OUT).filter(~pl.col("lst").is_in(a.only))
        if old.height:
            print(f"既存ファイルから {old.height:,} 行を保持 "
                  f"({', '.join(old['lst'].unique())})")
            frames.append(old)

    all_d = pl.concat(frames, how="vertical_relaxed").sort("lst", "block")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_d.write_parquet(OUT, compression="zstd")

    print(f"[write] {OUT}  {all_d.height:,} 行\n")
    print(f'{"LST":<8}{"件数":>8}  {"期間":<26}{"レート":<26}{"APR中央値":>10}')
    for k in all_d["lst"].unique(maintain_order=True):
        s = all_d.filter(pl.col("lst") == k)
        f = dt.datetime.fromtimestamp(s["ts"].min(), dt.timezone.utc).date()
        l = dt.datetime.fromtimestamp(s["ts"].max(), dt.timezone.utc).date()
        apr = s["apr_pct"].drop_nulls()
        med = f"{apr.median():.3f}%" if apr.len() else "—"
        print(f'{k:<8}{s.height:>8,}  {str(f)} 〜 {str(l)}   '
              f'{s["rate"].min():.6f} → {s["rate"].max():.6f}  {med:>10}')


if __name__ == "__main__":
    main()
