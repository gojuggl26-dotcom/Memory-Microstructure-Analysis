# Memory-Microstructure-Analysis

メモリ半導体関連 perp(Hyperliquid HIP-3 / builder `xyz`)の市場マイクロストラクチャー分析。

作業ディレクトリ: `C:\Users\ii562\Downloads\Memory`
リモート: `gojuggl26-dotcom/Memory-Microstructure-Analysis`(private)

## 1. スコープ(初期設定 — 確定したらこの節を書き換える)

対象は L2(注文ライフサイクル + 板)まで再構成済みの **6 銘柄**。
これは `hl-l4-pipeline` の WORK_BUCKET に L2 と fills が揃っている銘柄集合と一致する。

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
本リポジトリの付加価値は **銘柄横断**(共通因子・リード/ラグ・流動性の相対、
メモリ・セクター内でのフロー伝播)に置く。

## 2. データ源

| 層 | 場所 | 内容 |
|---|---|---|
| L1 | `s3://$WORK_BUCKET/l1/` | 正規化イベントログ(14 銘柄 × 99 日、240.3 GiB) |
| L2 | `s3://$WORK_BUCKET/l2/` | 注文ライフサイクル・板スナップショット(1s)・トリガー密度・book_px ほか 5 表(6 銘柄 132.6 GiB) |
| fills | `s3://$WORK_BUCKET/fills/` | node_fills(6 銘柄 4.7 GiB) |
| L3 | `s3://$WORK_BUCKET/l3/` | バー粒度特徴量。現状 DRAM のみ(L2 からローカル再生成可) |

生成の詳細と再現手順は `hl-l4-pipeline`(`C:\Users\ii562\hl-l4-pipeline`)を参照。
**本リポジトリは L2 を入力とし、Artemis raw には触れない。**

## 3. ディレクトリ構成

```
scripts/   分析スクリプト(1 スクリプト = 1 目的、再実行で同じ結果になること)
data/      ローカル作業データ。大きな parquet は .gitignore 済み(スクリプトで再生成する)
charts/    図
reports/   分析レポート(*_report.md)
```

## 4. 規範(親の `C:\Users\ii562\CLAUDE.md` を継承)

- **ルックアヘッド厳禁**: 説明変数が確定する時刻 ≤ 目的変数の期間の開始時刻。
  asof 結合は backward、`shift(-k)` は目的変数専用、標準化パラメータを全標本から作らない。
  標本外評価は日単位の時間ブロック分割で行う(行シャッフルの k-fold は漏洩する)。
- **`is_crossed` は必ず除外**する。少数の異常行が係数を桁で動かした実例がある。
- **fills の時刻はミリ秒精度**で板イベント(ns)より最大 ~1ms 早い。
  サブミリ秒の因果順序を主張しない。
- 各スクリプトの docstring に「x が確定する時刻」と「y の期間」を明記する。
- 報告前に CLAUDE.md「出力前の自己精査」A〜D の全項目を通す
  (費用控除・分母・対照としての「何もしない」・検出力・多重比較・最悪値・実装可能性)。
