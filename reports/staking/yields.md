# LST / ステーキングの利回り履歴

出典 DefiLlama `yields.llama.fi/chart/<pool>`(無料)。
全 **38,998 行 / 45 プール / 2022-05-03 〜 2026-09-14**。

取得は `scripts/defillama_yield_history.py`、この要約は `scripts/yield_history_report.py` が作る。

## 読み方の前提

- **集計の主キーはプール(UUID)である。** 同じ project+symbol で別チェーンに
  複数プールがある(WBETH は Ethereum と BSC、WEETH は 4 チェーン)。
  束ねると日数が二重に数えられる
- **ETH かどうかは `underlyingTokens` で判定した。** symbol の文字列では判定
  していない
- **開始日は DefiLlama の収録開始日であって、プロトコルの開始日ではない。**
  Lido は 2020-12 から動いているが履歴は 2022-05-03 から。それ以前が要るなら
  オンチェーンから取る(`scripts/onchain_lst_rates.py`)
- apy が 100% を超える行は異常値として統計から外し、末尾に列挙した

## ETH 系(31 プール)

| プロジェクト | シンボル | チェーン | 日数 | 期間 | APY 中央値 | 5%点 | 95%点 | 標準偏差 | TVL |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| lido | STETH | Ethereum | 1,596 | 2022-05-03〜2026-09-14 | 3.12% | 2.31% | 5.30% | 1.05% | $24.34B |
| binance-staked-eth | WBETH | Ethereum | 1,187 | 2023-06-16〜2026-09-14 | 2.75% | 2.37% | 4.09% | 0.60% | $8.85B |
| ether.fi-stake | WEETH | Ethereum | 832 | 2024-06-05〜2026-09-14 | 2.77% | 1.94% | 3.96% | 0.71% | $5.43B |
| rocket-pool | RETH | Ethereum | 1,338 | 2023-01-16〜2026-09-14 | 2.69% | 2.06% | 4.55% | 0.75% | $1.32B |
| kelp | RSETH | Ethereum | 854 | 2024-05-07〜2026-09-14 | 2.72% | 1.93% | 3.79% | 0.65% | $1.10B |
| liquid-collective | LSETH | Ethereum | 963 | 2024-01-26〜2026-09-14 | 2.80% | 0.00% | 3.85% | 0.95% | $807.5M |
| meth-protocol | METH | Ethereum | 1,002 | 2023-12-18〜2026-09-14 | 2.89% | 1.13% | 7.07% | 1.72% | $613.3M |
| binance-staked-eth | WBETH | BSC | 1,187 | 2023-06-16〜2026-09-14 | 2.75% | 2.37% | 4.09% | 0.60% | $532.0M |
| coinbase-wrapped-staked-eth | CBETH | Ethereum | 1,337 | 2023-01-16〜2026-09-14 | 2.72% | 2.34% | 4.28% | 0.83% | $480.3M |
| stakewise-v3 | OSETH | Ethereum | 995 | 2023-12-25〜2026-09-14 | 2.82% | 2.13% | 3.66% | 0.52% | $401.5M |
| stader | ETHX | Ethereum | 1,132 | 2023-08-10〜2026-09-14 | 2.67% | 0.00% | 4.32% | 1.52% | $201.4M |
| ether.fi-stake | WEETH | Linea | 131 | 2026-05-07〜2026-09-14 | 2.38% | 2.10% | 2.93% | 0.34% | $189.3M |
| renzo | EZETH | Ethereum | 641 | 2024-12-13〜2026-09-14 | 2.87% | 1.81% | 4.68% | 0.96% | $112.8M |
| stakewise-v3 | ETH | Ethereum | 34 | 2026-08-12〜2026-09-14 | 2.33% | 2.22% | 2.43% | 0.13% | $106.4M |
| frax-ether | SFRXETH | Ethereum | 539 | 2025-03-25〜2026-09-14 | 2.98% | 2.57% | 3.87% | 0.61% | $93.2M |
| origin-ether | OETH | Ethereum | 1,189 | 2023-05-29〜2026-09-14 | 3.13% | 2.21% | 7.76% | 1.93% | $57.4M |
| ether.fi-stake | WEETH | Base | 131 | 2026-05-07〜2026-09-14 | 2.38% | 2.10% | 2.93% | 0.34% | $53.1M |
| swell-liquid-staking | SWETH | Ethereum | 1,143 | 2023-07-17〜2026-09-14 | 3.15% | 2.50% | 3.86% | 0.39% | $35.4M |
| swell-liquid-restaking | RSWETH | Ethereum | 729 | 2024-09-08〜2026-09-14 | 2.67% | 1.53% | 5.22% | 1.21% | $32.5M |
| puffer-stake | PUFETH | Ethereum | 167 | 2026-04-01〜2026-09-14 | 2.30% | 0.01% | 3.19% | 1.08% | $30.0M |
| meta-pool-eth | MPETH | Ethereum | 950 | 2024-02-08〜2026-09-14 | 0.27% | 0.00% | 6.44% | 2.90% | $29.3M |
| bedrock-unieth | UNIETH | Ethereum | 1,003 | 2023-12-11〜2026-09-14 | 3.04% | 2.36% | 4.13% | 0.53% | $25.9M |
| nodedao | RNETH | Ethereum | 210 | 2026-02-17〜2026-09-14 | 2.43% | 2.12% | 2.52% | 0.28% | $24.2M |
| origin-ether | SUPEROETHB | Base | 723 | 2024-09-11〜2026-09-14 | 4.05% | 2.42% | 11.39% | 2.99% | $22.1M |
| ankr | ANKRETH | Ethereum | 1,555 | 2022-06-08〜2026-09-14 | 3.23% | 2.12% | 4.92% | 1.17% | $21.4M |
| geth | GETH | Ethereum | 1,074 | 2023-01-20〜2026-09-14 | 0.00% | 0.00% | 2.29% | 0.85% | $19.3M |
| nodedao | NETH | Ethereum | 1,131 | 2023-08-01〜2026-09-14 | 0.00% | 0.00% | 4.70% | 2.20% | $12.0M |
| bifrost-liquid-staking | VETH | Ethereum | 228 | 2026-01-30〜2026-09-14 | 2.69% | 2.33% | 14.04% | 4.21% | $1.5M |
| meta-pool-eth | SPETH | Ethereum | 210 | 2026-02-17〜2026-09-14 | 2.30% | 0.23% | 20.49% | 7.17% | $1.2M |
| ether.fi-stake | WEETH | Scroll | 131 | 2026-05-07〜2026-09-14 | 2.38% | 2.10% | 2.93% | 0.34% | $613K |
| prime-staked-eth | PRIMEETH | Ethereum | 873 | 2024-04-23〜2026-09-14 | 3.53% | 3.13% | 3.86% | 0.32% | $50K |

## ETH 系ではない / 判定できなかった(14 プール)

| プロジェクト | シンボル | チェーン | 日数 | 期間 | APY 中央値 | 5%点 | 95%点 | 標準偏差 | TVL |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| lombard-lbtc | LBTC | Ethereum | 245 | 2026-01-13〜2026-09-14 | 0.35% | 0.22% | 0.41% | 0.07% | $685.9M |
| stake.link-liquid | STLINK | Ethereum | 1,322 | 2022-12-22〜2026-09-07 | 5.29% | 4.64% | 8.65% | 1.56% | $94.9M |
| ether.fi-stake | EBTC | Ethereum | 239 | 2026-01-19〜2026-09-14 | 0.36% | 0.22% | 2.90% | 0.76% | $20.6M |
| bifrost-liquid-staking | VDOT | Polkadot | 1,257 | 2023-02-24〜2026-09-14 | 16.28% | 4.33% | 26.30% | 6.63% | $8.7M |
| ankr | ANKRFLOWEVM | Flow | 135 | 2026-05-03〜2026-09-14 | 7.90% | 7.90% | 7.90% | 0.00% | $6.0M |
| stader | MATICX | Polygon | 945 | 2024-02-13〜2026-09-14 | 2.31% | 0.00% | 14.11% | 4.83% | $3.9M |
| stake.link-liquid | STPOL | Ethereum | 362 | 2025-09-01〜2026-09-07 | 3.63% | 3.34% | 8.97% | 2.43% | $1.4M |
| bifrost-liquid-staking | VKSM | Kusama | 1,257 | 2023-02-24〜2026-09-14 | 15.77% | 12.10% | 27.08% | 4.63% | $598K |
| ankr | ANKRBNB | BSC | 1,542 | 2022-06-08〜2026-09-14 | 1.53% | 0.43% | 5.48% | 3.12% | $535K |
| bifrost-liquid-staking | VASTR | Astar | 935 | 2024-01-12〜2026-09-14 | 14.48% | 9.00% | 30.35% | 6.23% | $442K |
| arpa-staking | ARPA | Ethereum | 1,168 | 2023-07-05〜2026-09-14 | 23.32% | 3.55% | 40.69% | 12.21% | $235K |
| bifrost-liquid-staking | VBNC | Bifrost | 1,257 | 2023-02-24〜2026-09-14 | 5.72% | 2.19% | 13.21% | 3.40% | $217K |
| ankr | ANKRAVAX | Avalanche | 1,555 | 2022-06-08〜2026-09-14 | 6.24% | 3.26% | 12.02% | 3.09% | $120K |
| ankr | ANKRMATIC | Polygon | 1,555 | 2022-06-08〜2026-09-14 | 4.01% | 2.40% | 8.95% | 2.67% | $82K |

判定できなかったプールの `underlyingTokens`:

- `lombard-lbtc/LBTC/Ethereum` … 0x2260fac5e5542a773aa44fbcfedf7c193bc2c599
- `stake.link-liquid/STLINK/Ethereum` … 0x514910771af9ca656af840dff83e8264ecf986ca
- `ether.fi-stake/EBTC/Ethereum` … 0x8236a87084f8b84306f72007f36f2618a5634494, 0x2260fac5e5542a773aa44fbcfedf7c193bc2c599
- `bifrost-liquid-staking/VDOT/Polkadot` … coingecko:polkadot
- `ankr/ANKRFLOWEVM/Flow` … 0xd3bf53dac106a0290b0483ecbc89d40fcc961f3e
- `stader/MATICX/Polygon` … coingecko:polygon-ecosystem-token
- `stake.link-liquid/STPOL/Ethereum` … 0x455e53cbb86018ac2b8092fdcd39d8444affc3f6
- `bifrost-liquid-staking/VKSM/Kusama` … coingecko:kusama
- `ankr/ANKRBNB/BSC` … 0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c
- `bifrost-liquid-staking/VASTR/Astar` … 0xaeaaf0e2c81af264101b9129c00f4440ccf0f720
- `arpa-staking/ARPA/Ethereum` … 0xba50933c268f567bdc86e1ac131be072c6b0b71a
- `bifrost-liquid-staking/VBNC/Bifrost` … coingecko:bifrost-native-coin
- `ankr/ANKRAVAX/Avalanche` … 0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7
- `ankr/ANKRMATIC/Polygon` … 0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270

## 異常値(apy > 100%)

統計から外した。DefiLlama 側の利回り計算の不良と思われる(TVL は正常な値のまま大きくなっている)。

| プロジェクト | シンボル | チェーン | 該当日数 | 最大 apy |
|---|---|---|---:|---:|
| nodedao | NETH | Ethereum | 9 | 25,807% |

最も大きい 10 件:

| 日 | プロジェクト | シンボル | apy | TVL |
|---|---|---|---:|---:|
| 2026-01-04 | nodedao | NETH | 25,807% | $40.4M |
| 2025-06-10 | nodedao | NETH | 24,513% | $36.1M |
| 2025-02-03 | nodedao | NETH | 9,239% | $37.1M |
| 2025-04-29 | nodedao | NETH | 8,960% | $23.1M |
| 2024-12-16 | nodedao | NETH | 3,061% | $51.8M |
| 2024-05-21 | nodedao | NETH | 1,102% | $48.9M |
| 2025-06-27 | nodedao | NETH | 431% | $31.2M |
| 2025-03-19 | nodedao | NETH | 182% | $26.4M |
| 2025-07-21 | nodedao | NETH | 112% | $48.5M |

## pricePerShare が無いプール

27/45 プールで `pricePerShare` が全期間 null。
これが無いと DefiLlama の apy 定義をそのまま使うしかなく、**自分で利回りを再計算できない**。
Lido と Rocket Pool はどちらも null なので、両者の利回りを厳密に
扱いたければオンチェーンから取ること。
