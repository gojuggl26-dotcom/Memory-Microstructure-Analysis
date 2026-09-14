# ETH ステーキングプロトコル一覧

出典 DefiLlama / 取得日 **2026-09-13** / 対象 **85 プロトコル**

取得は `scripts/defillama_staking_fetch.py`、この一覧は `scripts/defillama_staking_report.py` が生成する。
DefiLlama の `/protocols` は現在値しか返さないので、ここの数字は**この日の断面**であって再取得しても再現しない。
生データを版管理に入れてあるのはそのためである(`data/staking/`)。

## この一覧と対になる資料

| 資料 | 中身 | 生成 |
|---|---|---|
| [利回り履歴](yields.md) | DefiLlama の日次 APY。45 プール・2022-05 以降 | `defillama_yield_history.py` → `yield_history_report.py` |
| [市場価格と償還価値の乖離](discount.md) | 主要 LST 4 本。Chainlink の市場価格とオンチェーンの償還レートの差 | `chainlink_lst_prices.py` + `onchain_lst_rates.py` → `lst_discount_analysis.py` |

## 読み方の前提(ここを外すと数字が意味を失う)

### 1. TVL を層をまたいで足してはいけない

同じ ETH が複数のプロトコルで数えられている。

- **DVT**(SSV / Obol)は他プロトコルのバリデータを運用する基盤である。SSV の TVL は Lido などの預り分と重複する
- **Restaking / Liquid Restaking** は LST を預かる。EigenCloud の預りは stETH や weETH として既に Liquid Staking 側で数えられている

### 2. ETH かどうかはシンボル文字列で判定していない

DefiLlama の Liquid Staking カテゴリには SOL / BNB / AVAX / CRO / LINK のステーキングが混ざる。Ethereum 上で発行されているだけで原資産が ETH でないものも多い(Lido は WMATIC も持つ)。
判定は `tokensInUsd / tokens` から**実測した単価**が ETH 価格の [0.40, 2.00] 倍に入るかで行い、その割合が `ETH 比率` 列である。
単価が出ないものは**判定不能**として別掲し、黙って除外していない。

### 3. TVL は預り資産であって収益ではない

手数料・収益は `data/staking/fees_eth_*.json`、利回りは `data/staking/yields.json` に落としてあるが、この一覧には結合していない。

## 内訳

| 層 | 定義 | 件数 | TVL 合計 |
|---|---|---:|---:|
| A | ETH 主体 | 56 | $64.77B |
| B | 一部 ETH | 5 | $70.3M |
| C | ETH ではない | 15 | $162.3M |
| D | 判定不能 | 9 | $678.9M |
| | **計** | **85** | $65.68B |

### 層 A から重複を除く

| 区分 | 件数 | TVL | 扱い |
|---|---:|---:|---|
| 素の預り(LST・ステーキングプール) | 34 | $43.05B | **これが ETH の実体規模に最も近い** |
| DVT(他プロトコルのバリデータを運用) | 2 | $13.75B | 重複。足さない |
| Restaking 系(LST を預かる) | 20 | $7.96B | 重複。足さない |

つまり「ETH ステーキングの総額」として引用してよいのは **$43.05B** であって、85 件の素朴な合計 $65.68B ではない。

## 層 A — ETH 主体(56 件)

預り資産のほぼ全部が ETH 系。**このプロジェクトで対象にするならここ。**

| # | プロトコル | カテゴリ | トークン | TVL | ETH 比率 | 展開チェーン数 | 登録 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | [Lido](https://lido.fi/) | Liquid Staking | LDO | $24.38B | 100.0% | 5 | — |
| 2 | [SSV Network](https://ssv.network/) | Staking Pool | SSV | $13.11B | 100.0% | 1 | — |
| 3 | [Binance staked ETH](https://www.binance.com/en/wbeth) | Liquid Staking | — | $9.29B | 100.0% | 2 | 2023-04 |
| 4 | [EigenCloud](https://www.eigencloud.xyz) | Restaking | EIGEN | $6.60B | 99.9% | 1 | 2023-06 |
| 5 | [ether.fi Stake](https://www.ether.fi) | Liquid Staking | ETHFI | $4.61B | 99.5% | 5 | 2023-03 |
| 6 | [Rocket Pool](https://rocketpool.net) | Liquid Staking | RPL | $1.30B | 100.0% | 1 | 2021-11 |
| 7 | [Kelp](https://kelpdao.xyz/restake/?utm_source=0x798fF1e6D7AFd28c333eE6eBe03125d30ec6eF10) | Liquid Restaking | KERNEL | $1.11B | 100.0% | 1 | 2023-12 |
| 8 | [StakeWise V3](https://stakewise.io/) | Liquid Staking | SWISE | $945.7M | 100.0% | 2 | — |
| 9 | [Liquid Collective](https://liquidcollective.io) | Liquid Staking | — | $799.5M | 100.0% | 1 | 2023-08 |
| 10 | [Obol](https://obol.org/) | Staking Pool | OBOL | $640.4M | 100.0% | 1 | 2025-11 |
| 11 | [mETH Protocol](https://www.methprotocol.xyz/) | Liquid Staking | COOK | $606.3M | 100.0% | 1 | 2023-12 |
| 12 | [Coinbase Wrapped Staked ETH](https://www.coinbase.com/price/coinbase-wrapped-staked-eth) | Liquid Staking | — | $475.4M | 100.0% | 1 | 2022-10 |
| 13 | [Stader](https://staderlabs.com) | Liquid Staking | SD | $251.9M | 100.0% | 6 | 2021-12 |
| 14 | [Frax Ether](https://frax.com/) | Liquid Staking | FRAX | $126.7M | 100.0% | 2 | 2022-10 |
| 15 | [Renzo](https://app.renzoprotocol.com/restake) | Liquid Restaking | REZ | $114.3M | 100.0% | 13 | 2023-12 |
| 16 | [Origin Ether](https://www.originprotocol.com/oeth) | Liquid Staking | OGN | $61.2M | 100.0% | 3 | 2023-05 |
| 17 | [Mantle Restaking](https://www.methprotocol.xyz/) | Liquid Restaking | — | $41.7M | 100.0% | 1 | 2024-11 |
| 18 | [NodeDAO](https://www.nodedao.com) | Liquid Staking | — | $35.8M | 100.0% | 1 | 2023-05 |
| 19 | [Swell Liquid Staking](https://app.swellnetwork.io/) | Liquid Staking | SWELL | $31.1M | 100.0% | 1 | 2023-04 |
| 20 | [Swell Liquid Restaking](https://app.swellnetwork.io/restake) | Liquid Restaking | SWELL | $29.9M | 100.0% | 1 | 2024-01 |
| 21 | [Puffer Stake](https://www.puffer.fi/restake) | Liquid Restaking | PUFFER | $29.7M | 100.0% | 1 | 2024-02 |
| 22 | [Meta Pool ETH](https://metapool.app) | Liquid Staking | MPDAO | $26.5M | 100.0% | 1 | 2023-08 |
| 23 | [Ankr](https://www.ankr.com/) | Liquid Staking | ANKR | $26.2M | 99.6% | 8 | — |
| 24 | [Bedrock uniETH](https://bedrock.rockx.com/unieth) | Liquid Restaking | BR | $25.7M | 100.0% | 1 | 2023-04 |
| 25 | [StakeStone STONE](https://stakestone.io/) | Liquid Staking | STO | $20.4M | 100.0% | 1 | 2023-09 |
| 26 | [GETH](https://guarda.com/staking/ethereum-staking/) | Liquid Staking | GETH | $19.1M | 100.0% | 1 | 2023-01 |
| 27 | [Bifrost Liquid Staking](https://app.bifrost.io/?channelId=17) | Liquid Staking | BNC | $12.7M | 100.0% | 4 | 2022-05 |
| 28 | [InceptionLRT (Isolated Restaking)](https://www.inceptionlrt.com/) | Liquid Restaking | — | $7.0M | 100.0% | 1 | 2024-01 |
| 29 | [Dinero (pxETH)](https://dinero.xyz) | Liquid Staking | — | $5.4M | 100.0% | 1 | 2023-12 |
| 30 | [Stakingverse](https://stakingverse.io) | Liquid Staking | — | $4.9M | 100.0% | 2 | 2025-04 |
| 31 | [Stafi](https://www.stafi.io/) | Liquid Staking | FIS | $4.8M | 100.0% | 5 | — |
| 32 | [Eigenpie](https://www.eigenlayer.magpiexyz.io) | Liquid Restaking | — | $4.4M | 94.8% | 2 | 2024-01 |
| 33 | ClayStack ETH | Liquid Restaking | — | $2.5M | 100.0% | 1 | 2022-05 |
| 34 | [CRETH2](https://classic.cream.finance/eth2/) | Liquid Staking | CREAM | $1.7M | 100.0% | 1 | 2023-05 |
| 35 | Hord | Liquid Staking | HORD | $361K | 100.0% | 1 | 2023-02 |
| 36 | [Goldsand by InshAllah](https://goldsand.fi/) | Liquid Staking | — | $350K | 100.0% | 1 | 2024-12 |
| 37 | Euclid Finance | Liquid Restaking | — | $297K | 100.0% | 1 | 2024-03 |
| 38 | [TokenPocket](https://dapp.tokenpocket.pro/StakeVault/#/) | Staking Pool | TPT | $252K | 100.0% | 1 | 2023-04 |
| 39 | [Bracket LST](https://app.bracket.fi/) | Liquid Staking | — | $245K | 100.0% | 1 | 2025-07 |
| 40 | [Lido Impact Staking](https://impactstake.com) | Liquid Staking | — | $231K | 100.0% | 1 | 2025-05 |
| 41 | [MEV Protocol](https://mev.io/) | Liquid Staking | — | $124K | 100.0% | 1 | 2023-10 |
| 42 | [Stakehouse](https://blockswap.network/) | Liquid Staking | BSN | $101K | 100.0% | 1 | 2023-04 |
| 43 | [Affine Restaking](https://app.affinedefi.com/restake) | Liquid Restaking | — | $62K | 100.0% | 2 | 2024-06 |
| 44 | Restake Finance | Liquid Restaking | RSTK | $59K | 100.0% | 1 | 2024-02 |
| 45 | [ShardingDAO](https://shardingdao.com/) | Staking Pool | — | $59K | 100.0% | 1 | 2024-11 |
| 46 | [Kernel Protocol](https://kernelprotocol.com) | Liquid Restaking | — | $52K | 96.1% | 1 | 2024-07 |
| 47 | [Prime Staked ETH](https://app.primestaked.com/#/restake) | Liquid Restaking | — | $50K | 100.0% | 1 | 2024-02 |
| 48 | [GenesisLRT (Native Restaking)](https://www.inceptionlrt.com/) | Liquid Restaking | — | $47K | 100.0% | 1 | 2024-01 |
| 49 | Aqua Patina | Liquid Restaking | — | $19K | 100.0% | 1 | 2024-10 |
| 50 | [NEOPIN Liquid](https://app.neopin.io) | Liquid Staking | NPT | $9K | 100.0% | 2 | 2023-06 |
| 51 | [Tranchess Ether](https://tranchess.com/liquid-staking) | Liquid Staking | CHESS | $8K | 100.0% | 1 | 2023-05 |
| 52 | Dunes | Liquid Restaking | — | $5K | 90.9% | 1 | 2024-06 |
| 53 | Aspida | Liquid Restaking | — | $3K | 100.0% | 1 | 2024-01 |
| 54 | Zero-G Finance | Liquid Restaking | — | $3K | 100.0% | 4 | 2024-04 |
| 55 | [OpenGPU](https://opengpu.network/) | Staking Pool | oGPU | $319 | 100.0% | 1 | 2024-12 |
| 56 | LST Optimizer | Liquid Staking | — | $260 | 100.0% | 1 | 2023-11 |

## 層 B — 一部 ETH(5 件)

ETH と他資産が混在する。ETH 部分だけを見たいなら按分が要る。

| # | プロトコル | カテゴリ | トークン | TVL | ETH 比率 | 展開チェーン数 | 登録 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | [Veno Finance](https://veno.finance/) | Liquid Staking | VNO | $35.7M | 6.9% | 4 | 2023-01 |
| 2 | [Mellow Restaking](https://mellow.finance/) | Liquid Restaking | — | $24.3M | 61.0% | 5 | 2024-06 |
| 3 | [OpenGDP Shared Security](https://opengdp.network/) | Restaking | — | $8.1M | 88.0% | 7 | 2024-04 |
| 4 | [King Protocol](https://kingprotocol.org/) | Liquid Restaking | — | $1.3M | 23.1% | 1 | 2025-02 |
| 5 | [Allstake](https://allstake.org/) | Restaking | — | $910K | 59.4% | 3 | 2024-06 |

## 層 C — ETH ではない(15 件)

ETH 系の預りが検出されなかった。カテゴリ上は拾われたが**他チェーンのステーキング**である。対象外だが、除外した証拠として残す。

| # | プロトコル | カテゴリ | トークン | TVL | ETH 比率 | 展開チェーン数 | 登録 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | [stake.link liquid](https://stake.link/staking-pools) | Liquid Staking | SDL | $82.1M | 0.0% | 1 | 2022-12 |
| 2 | [SolvBTC LSTs](https://solv.finance) | Restaked BTC | — | $74.5M | 0.0% | 2 | 2024-10 |
| 3 | [TruStake](https://app.truyields.com) | Liquid Staking | — | $4.3M | 0.0% | 5 | 2023-07 |
| 4 | [Accumulated Finance Liquid Staking](https://accumulated.finance/stake) | Liquid Staking | — | $633K | 0.0% | 11 | 2023-12 |
| 5 | [Tenderize V2](https://tenderize.me) | Liquid Staking | — | $267K | 0.0% | 2 | 2024-02 |
| 6 | [Pell Network](https://pell.network/) | Restaking | PELL | $222K | 0.0% | 20 | 2024-05 |
| 7 | BakerFi | Liquid Staking | — | $78K | 0.0% | 3 | 2024-02 |
| 8 | [Deq](https://deq.fi) | Liquid Staking | — | $64K | 0.0% | 1 | 2024-07 |
| 9 | ClayStack Matic | Liquid Staking | — | $53K | 0.0% | 1 | 2024-02 |
| 10 | zLot | Liquid Staking | — | $29K | 0.0% | 1 | — |
| 11 | [K9 Finance DAO](https://www.k9finance.com) | Liquid Staking | KNINE | $14K | 0.0% | 2 | 2024-09 |
| 12 | [Tenderize V1](https://tenderize.me) | Liquid Staking | — | $9K | 0.0% | 2 | 2022-05 |
| 13 | [NF3 APE](https://parallel.fi/) | Liquid Staking | — | $4K | 0.0% | 1 | 2023-03 |
| 14 | [Rivus DAO](https://stake.rivusdao.xyz) | Liquid Staking | RIVUS | $2K | 0.0% | 1 | 2024-05 |
| 15 | [TruFin Legacy Vaults](https://app.truyields.com) | Liquid Staking | — | $9 | 0.0% | 1 | 2023-04 |

## 層 D — 判定不能(9 件)

`tokensInUsd / tokens` から単価が出せず、ETH かどうかを判定できなかった。
TVL が極小のものが大半だが、**Lombard LBTC だけは TVL が大きい**(BTC 系なので ETH ではないと思われるが、実測で確かめていない)。
黙って除外せず、ここに出しておく。

| # | プロトコル | カテゴリ | トークン | TVL | ETH 比率 | 展開チェーン数 | 登録 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | [Lombard LBTC](https://www.lombard.finance/app/stake/?referrer=ssybnl) | Restaked BTC | BARD | $677.8M | — | 2 | 2024-09 |
| 2 | [Neemo Finance](https://neemo.finance/) | Liquid Restaking | — | $1.1M | — | 2 | 2025-01 |
| 3 | VaultLayer | Liquid Staking | — | $256 | — | 9 | 2025-06 |
| 4 | [SharedStake](https://www.sharedstake.org/) | Liquid Staking | SGT | $0 | — | 1 | — |
| 5 | [StakeHound](https://stakehound.com/) | Liquid Staking | — | $0 | — | 1 | — |
| 6 | [ARPA Staking](https://staking.arpanetwork.io/en-US/stake?action=Stake) | Staking Pool | ARPA | $0 | — | 1 | 2023-07 |
| 7 | Gracy Staking | Staking Pool | GRACY | $0 | — | 2 | 2023-12 |
| 8 | [WBROCK Staking](https://www.bit-rock.io/) | Staking Pool | — | $0 | — | 2 | 2024-06 |
| 9 | [Tokamak Network Staking](https://toki.tokamak.network/) | Staking Pool | TON | $0 | — | 1 | 2026-06 |

## 未着手

- 手数料・収益(`fees_eth_*.json`)と利回り(`yields.json`)をこの一覧に結合していない。結合すれば「預りの規模」ではなく「取り分の大きさ」で並べ替えられる
- 層 D の 9 件を一次ソース(各プロトコルのコントラクト)で確かめていない
- 親での名寄せをしていない。6 組が同じ親を持つ(Swell が LST と LRT に分かれる等)が、そのほとんどは Restaking 側を層 A の合計から外した時点で解消しており、残る影響は $136M 未満(素の預り $43.05B の 0.3% 未満)である
- TVL は断面のみ。時系列は DefiLlama の有料 API が要る
