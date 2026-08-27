# Memory-Microstructure-Analysis

メモリ半導体関連 perp(Hyperliquid HIP-3 / builder `xyz`)の市場マイクロストラクチャー分析。

-入手方法: ArtemisのL4板データをAWSのEC2経由で取得。

<img src="photo/artemis-logo.png" alt="Artemis" width="96">

Artemis URL  https://about.artemis.ai/

## 1. スコープ(初期設定 — 確定したらこの節を書き換える)

このレポジトリ内で取り扱う銘柄は以下の六銘柄である。(現時点)

| coin | 対象 | 備考 |
|---|---|---|
| `xyz:DRAM` | DRAM 価格指数 | 2026-05-04 15:33 UTC 初イベント。板が空からの観測で孤児ゼロ |
| `xyz:KIOXIA` | キオクシア | |
| `xyz:MU` | Micron | |
| `xyz:SKHX` | SK hynix | |
| `xyz:SMSN` | Samsung | |
| `xyz:SNDK` | SanDisk | |

窓: **2026-05-04 〜 2026-08-10(99 日)**。

単独銘柄(`xyz:DRAM`)の全期間マイクロプライス分析は別リポジトリ
[`DRAM-microprice`](https://github.com/gojuggl26-dotcom/DRAM-microprice) に既にある。
本リポジトリの分析テーマは 銘柄横断(共通因子・リード/ラグ・流動性の相対、メモリ・セクター内でのフロー伝播)に置く。

## 2. データ源

| 層 | 場所 | 内容 |
|---|---|---|
| L1 | `s3://$WORK_BUCKET/l1/` | 正規化イベントログ(14 銘柄 × 99 日、240.3 GiB) |
| L2 | `s3://$WORK_BUCKET/l2/` | 注文ライフサイクル・板スナップショット(1s)・トリガー密度・book_px ほか 5 表(12 銘柄 × 99 日。MU のみ 08-10 が未完成) |
| fills | `s3://$WORK_BUCKET/fills/` | node_fills(14 銘柄 × 99 日) |
| L3 | `s3://$WORK_BUCKET/l3/` | バー粒度特徴量 4 バー種(12 銘柄。L2 からローカル再生成可) |

生成の詳細と再現手順は `hl-l4-pipeline`(`C:\Users\ii562\hl-l4-pipeline`)を参照。銘柄別の在庫は `reports/mu_inventory_report.md` と `scripts/inventory_s3.py` で確認できる。

## 3. ディレクトリ構成

```
scripts/   分析スクリプト(1 スクリプト = 1 目的、再実行で同じ結果になること)
data/      ローカル作業データ。大きな parquet は .gitignore 済み(スクリプトで再生成する)
charts/    図
reports/   分析レポート(*_report.md)
```


