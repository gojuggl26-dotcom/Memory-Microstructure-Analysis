# Boros のデータ源 — 実測による確認(2026-08-23)

**結論: 取得できる。認証不要の公開 REST API があり、板・約定・OHLCV・原資産 APR の履歴まで揃う。**
UI のスクリーンショットと API の板が**全レベル一致**することを確認済み。

---

## 1. API

| | |
|---|---|
| ベース URL | `https://api-boros.pendle.finance/apis/v1` |
| OpenAPI 仕様 | `https://api-boros.pendle.finance/apis/docs-json`(78 エンドポイント) |
| 認証 | **読み取り系は不要**(実測)。発注系のみ署名が要る |
| レート制限 | Computing Unit (CU) 方式、IP ごとの予算 |
| 注意 | **User-Agent を見て 403 を返す**。Python の urllib 既定 UA は弾かれた。curl は通る |

### 使うエンドポイント

| パス | 内容 | 必須パラメータ |
|---|---|---|
| `GET /v1/markets` | 全市場(ページング、`resumeToken`) | — |
| `GET /v1/markets/by-ids` | 個別市場 | ids |
| **`GET /v1/markets/order-book`** | **板** | `marketId`, `tickSize` ∈ {0.0001, 0.001, 0.01, 0.1}, `includeAmm` |
| **`GET /v1/markets/trades`** | **約定** | `marketId`, `limit`, `resumeToken` |
| `GET /v1/markets/ohlcv` | ローソク足 | `marketId`, `timeFrame`, start/end |
| **`GET /v1/markets/historical-underlying-apr`** | **原資産 APR の履歴** | `assetSymbol`, `exchange`, `timeFrame` |
| `GET /v1/indicators` / `/export` | 指標(underlying APR・future premium・fear&greed)。CSV 出力あり | — |
| **`GET /v1/incentives/maker-incentives/campaigns/{marketId}`** | **メイカー報酬の campaign** | marketId |
| `POST /v1/funding-rate/settlement-summary` | 決済履歴 | — |
| `GET /v1/trade-volume/rolling` | ローリング出来高 | — |

> **★注意**: `/v2/markets/order-books`(複数形)はドキュメント記載だが **404**。
> 正しくは `/v1/markets/order-book`(単数)。404 の本文に内部プレフィクス
> `/open-api-v2/v1/` が露出しており、これで気づいた。

## 2. 市場の在庫(2026-08-23 時点)

**194 市場 / 7 プラットフォーム**:

| プラットフォーム | 市場数 |
|---|---:|
| Hyperliquid | 89 |
| Binance | 50 |
| OKX | 20 |
| Lighter | 11 |
| Gate | 9 |
| Bybit | 7 |
| **KuCoin** | **8** |

KuCoin の 8 市場(`midApr` / `floatingApr` / OI / 24h 出来高、単位は原資産):

| marketId | symbol | 満期 | mid APR | floating | OI | 24h Vol |
|---:|---|---|---:|---:|---:|---:|
| 151 | KUCOIN-BTCUSDT-26JUN2026 | 2026-06-26 | 1.70% | 1.42% | 13.48 | 8.58 |
| 150 | KUCOIN-ETHUSDT-26JUN2026 | 2026-06-26 | 3.75% | 3.50% | 7,210.24 | 960.85 |
| 163 | KUCOIN-BTCUSDT-31JUL2026 | 2026-07-31 | 6.98% | 7.23% | 130.72 | 57.28 |
| 159 | KUCOIN-ETHUSDT-31JUL2026 | 2026-07-31 | 5.19% | 5.15% | 6,847.91 | 2,204.07 |
| **174** | **KUCOIN-BTCUSDT-28AUG2026** | **2026-08-28** | **8.48%** | **10.95%** | **131.23** | **45.60** |
| 182 | KUCOIN-ETHUSDT-28AUG2026 | 2026-08-28 | 8.16% | 10.95% | 1,738.17 | 0.00 |
| 175 | KUCOIN-BTCUSDT-25SEP2026 | 2026-09-25 | 6.19% | 10.95% | 7.46 | 10.32 |
| 195 | KUCOIN-ETHUSDT-25SEP2026 | 2026-09-25 | 6.34% | 10.95% | 100.00 | 20.00 |

## 3. 板の構造(marketId 174 で検証)

```
GET /v1/markets/order-book?marketId=174&tickSize=0.001&includeAmm=true
→ {"long": {"ia": [...], "sz": [...]}, "short": {"ia": [...], "sz": [...]}, "syncStatus": {...}}
```

- **`ia`** = implied APR の tick(整数)。`tickSize=0.001` のとき **1 tick = 0.1%**。
  例: 89 → 8.9%
- **`sz`** = 数量、**1e18 スケールの文字列**(wei 形式)。
  例: `"70683407407407400"` → 0.0707 YU
- **`long`(下側)= 買い相当、`short`(上側)= 売り相当**。
  UI では short が上に赤、long が下に緑で表示される
- `syncStatus` に `blockNumber` と `timestamp` が入る(**ブロック単位の時刻**)

### UI との照合(全レベル一致を確認)

| side | tick(API) | 表示 APR | size(API, /1e18) | 画面の Size |
|---|---:|---:|---:|---:|
| short | 89 | 8.9% | 0.070683 | 0.0707 |
| short | 90 | 9.0% | 0.000145 | 0.0001 |
| short | 103 | 10.3% | 28.250704 | 28.2507 |
| short | 122 | 12.2% | 36.363150 | 36.3631 |
| long | 80 | 8.0% | 0.073426 | 0.0734 |
| long | 78 | 7.8% | 16.707393 | 16.7074 |
| long | 59 | 5.9% | 40.108241 | 40.1082 |

**8 + 10 レベルすべてが一致。**

### 約定

```
GET /v1/markets/trades?marketId=174&limit=5
→ {"results": [{"size":…, "rate":…, "txHash":…, "blockTimestamp":…}], "resumeToken": …}
```

**★重要**: **複数の約定が同一 `txHash` を共有する**。
例: `0x1803a5…` の 1 トランザクションで rate 0.08687 / 0.08361 / 0.08209 の 3 件が約定している。
これは**1 つの成行注文が複数の価格レベルを掃いた**ことを意味し、
`DRAM-microprice` 報告 47 の `tid` 構造と同型である。

> **`txHash` で束ねればテイカー注文の単位が復元でき、掃いた深さと方向が測れる。**
> Boros には CLOB があるので、DRAM で作った道具(テイカー分類・板再構成・逆選択の測定)が
> **そのまま移植できる可能性が高い。**

## 4. 契約パラメータ(`/v1/markets` の中身)

1 市場あたり以下が取れる。マイクロストラクチャーに直接効くもの:

| 項目 | 例(marketId 2) | 意味 |
|---|---|---|
| `imData.tickStep` | 2 | ネイティブの tick 幅 |
| `imData.maturity` | unix 秒 | 満期 |
| `imData.marginFloor` | 0.0600 | 証拠金の下限 |
| `config.takerFee` | `5e14`(/1e18 = **0.05% = 5bp**) | テイカー手数料 |
| `config.hardOICap` / `softOICap` | 686 / 325.85 | **OI 上限**(容量の天井が明示されている) |
| `config.maxRateDeviationFactorBase1e4` | 1500 | 価格帯の制限 |
| `config.liqSettings` | base/slope/feeRate | 清算パラメータ |
| `extConfig.paymentPeriod` | **28800 秒 = 8 時間** | **決済(ファンディング授受)の周期** |
| `extConfig.ammAddress` / `ammId` | あり | **板とは別に AMM が併存する** |
| `metadata.maxLeverage` | 3.2 | |
| `data.dailyVolatility` | | 日次ボラ |

## 5. ★分析上の最大の注意点 — 板は報酬で維持されている

UI に **「SHORT RATE 500% APR」「LONG RATE 216% APR」「Incentivized Range 7.66%–9.78%」**
と表示される。API 側にも `/v1/incentives/maker-incentives/campaigns/{marketId}` がある。

**つまりメイカーは指値を置くこと自体に PENDLE 報酬を得ている。**
実際、板の数量は

```
0.0707 / 0.0001 / 3.8037 / 0.0689 / 5.7381 / 28.2507 / 0.0652 / 36.3631
```

のように、**極小(0.0001〜0.08)と大口(28〜40)が交互に並ぶ**。
極小のものは**報酬獲得のための「置いているだけ」の注文**である可能性が高い。

> **含意**: スプレッド・厚み・注文寿命といった指標を素朴に解釈すると誤る。
> 「この板がどれだけ流動的か」を測る前に、
> **報酬目的の注文と真の流動性を分離する**必要がある。
> DRAM 案件で「見せ玉の代理として超短命注文比率」を使ったのと同種の問題だが、
> ここでは**報酬設計が既知**なので、incentivized range の内外で分けるという
> 直接的な切り分けができる。

## 6. 次にやること

> **★訂正(2026-08-23)**: 下表の 1 で「板は録画しないと後から取れない」と書いたのは**誤り**。
> 実際には **(a) オンチェーンのイベントログから遡って完全復元できる**(チェーンに永続するので
> 録り逃しの概念が無い)、**(b) 公式アーカイブ `historical-data.boros.finance` に 1 時間断面がある**。
> 既に別セッションが `E:\Boros-history` に 188 市場ぶんを構築済みで、
> **全 185,239 断面で復元板がアーカイブと完全一致**することを検証している。
> → [market163_scouting.md](market163_scouting.md)

| 順 | 内容 | 判定基準 |
|---|---|---|
| ~~1~~ | ~~板・約定のスナップショット収集を開始~~ → **不要**(上記訂正)。既存の `E:\Boros-history` を使う | 済 |
| 2 | `txHash` で約定を束ね、**テイカー注文単位**を復元。掃いた深さ・方向を測る | 報告 47 と同型の分類ができるか |
| 3 | **incentivized range の内外**で注文を分離し、真の流動性を推定 | 極小注文の比率が range 内で有意に高いか |
| 4 | **implied APR と floating APR の乖離**(174 では 8.48% vs 10.95%)の時系列。`historical-underlying-apr` で原資産側が取れる | 乖離に予測力があるか。**満期への収束を統制すること**(METHODOLOGY §2-1) |
| 5 | DRAM 案件の funding 実測(Hyperliquid、99 日・年率 +28.6%)と接続。**Boros には Hyperliquid 市場が 89 件ある** | 同一原資産で L4 と Boros を突き合わせられるか |

---

*先行案件: `DRAM-microprice`。§3 の `txHash` 束ねと §5 の流動性分離は、
そちらの報告 47(テイカー分類)と報告 41(特徴量棚卸し)の手法が直接使える。*
