# Memory-Microstructure-Analysis

<p>
  <a href="https://hyperliquid.xyz/"><img src="photo/hyperliquid-logo.png" alt="Hyperliquid" height="76"></a>
  &nbsp;&nbsp;
  <a href="https://trade.xyz/"><img src="photo/tradexyz-logo.png" alt="Trade.xyz" height="76"></a>
  &nbsp;&nbsp;
  <a href="https://about.artemis.ai/"><img src="photo/artemis-logo.png" alt="Artemis" height="76"></a>
  &nbsp;&nbsp;
  <a href="https://www.python.org/"><img src="photo/python-logo.png" alt="Python" height="76"></a>
</p>

メモリ半導体に連動する無期限先物(perpetual futures、以下 perp)を対象に、
注文一本ごとの記録まで遡って市場のミクロ構造を調べるリポジトリです。

対象の銘柄は、分散型取引所 Hyperliquid 上で Trade.xyz が配備したものです。
その板情報を Artemis が公開しており、AWS の EC2 を経由して取得しています。
分析はすべて Python で書かれており、各スクリプトは再実行すれば同じ結果を再現します。

| | 役割 | 公式サイト |
|---|---|---|
| Hyperliquid | 銘柄が上場している分散型取引所 | https://hyperliquid.xyz/ |
| Trade.xyz | 本リポジトリが扱う銘柄を配備した発行者 | https://trade.xyz/ |
| Artemis | 板情報の公開元 | https://about.artemis.ai/ |
| Python | 分析に使用している言語 | https://www.python.org/ |

---

## 目次

分析の成果物は銘柄ごとに `reports/<銘柄コード>/` へまとめています。各フォルダの
**分析索引ページ**に、その銘柄のレポート・図・数値データがすべて並んでいます。
まずは下の索引から入ってください。

| 銘柄 | 分析索引 | 現在のレポート数 |
|---|---|---:|
| `xyz:MU`(マイクロン) | **[分析索引を開く](reports/MU/README.md)** | 5 |
| `xyz:DRAM` / `xyz:KIOXIA` / `xyz:SKHX` / `xyz:SMSN` / `xyz:SNDK` | 未着手 | 0 |

新しい銘柄を分析したら、`reports/<銘柄コード>/README.md` を作ってこの表に 1 行足してください。

---

## 1. このリポジトリの目的

同じメモリ半導体という産業に連動する複数の銘柄を **横断して** 比べることに主眼を置いています。
具体的には、銘柄間の共通要因、値動きの先行と遅行の関係、流動性の相対的な厚み、
そして一つの銘柄で起きた注文の流れが他の銘柄へどう伝わるかを調べる。

## 2. 対象銘柄

現時点では次の 6 銘柄を扱います。いずれも Hyperliquid 上で
[Trade.xyz](https://trade.xyz/) が配備した銘柄で、正式な表記は `xyz:` で始まります。

| 銘柄コード | 対象 | 分析索引 | 備考 |
|---|---|---|---|
| `xyz:DRAM` | DRAM 価格指数 | 未着手 | 2026 年 5 月 4 日 15 時 33 分(UTC)が最初の記録 |
| `xyz:KIOXIA` | キオクシア | 未着手 | |
| `xyz:MU` | マイクロン・テクノロジー | [reports/MU/README.md](reports/MU/README.md) | |
| `xyz:SKHX` | SK ハイニックス | 未着手 | |
| `xyz:SMSN` | サムスン電子 | 未着手 | |
| `xyz:SNDK` | サンディスク | 未着手 | |

標本期間は **2026 年 5 月 4 日から 8 月 10 日までの 99 日間** です。

## 3. データの入手方法と階層

Artemis が公開する保管庫から生データを読み出し、段階的に加工しています。
生データそのものは保存せず、加工後のものだけを作業用バケットに置いています。

| 階層 | 内容 | 保有状況 |
|---|---|---|
| L1 | 正規化した注文イベントの記録。板に出た注文、取り消し、約定が時系列に並ぶ | 14 銘柄 × 99 日 |
| fills | 約定の記録。1 つの取引につき買い手と売り手の 2 行が入る | 14 銘柄 × 99 日 |
| L2 | 注文の一生(発注から消滅まで)と板の状態の復元。5 種類の表からなる | 12 銘柄 × 99 日 |
| L3 | 一定時間ごと、あるいは一定約定数ごとに集計した特徴量 | 12 銘柄 × 4 種類のバー |

生成の手順と詳細は、パイプライン側のリポジトリ `hl-l4-pipeline` にあります。
銘柄ごとの在庫は `scripts/inventory_s3.py` で確認できます。棚卸しの結果は
各銘柄の分析索引([xyz:MU の索引](reports/MU/README.md) など)に置いています。

## 4. ディレクトリ構成

| 場所 | 用途 |
|---|---|
| `scripts/` | 分析スクリプト。1 つのスクリプトにつき 1 つの目的 |
| `reports/<銘柄コード>/` | 銘柄別の分析レポート。`README.md` が分析索引、それ以外は `*_report.md` |
| `charts/` | スクリプトが生成した図 |
| `data/` | 作業用データ。容量の大きい parquet ファイルは版管理から除外し、スクリプトで再生成する |
| `photo/` | ロゴなど、生成物ではない画像 |

## 5. スクリプトと再現手順

上から順に実行すると、レポートの図と数値がすべて再現できます。

| 手順 | スクリプト | 役割 |
|---|---|---|
| 1 | `inventory_s3.py` | 作業用バケットの中身を一度だけ列挙して手元に保存する |
| 2 | `inventory_rows.py` | ファイルの末尾情報だけを読み、日ごとの行数を数える |
| 3 | `fetch_fills.py` | 約定記録から必要な列だけを取り出して手元に落とす |
| 4 | `build_oi_volume.py` | 約定記録から建玉と出来高の日次系列を組み立てる |
| 5 | `plot_oi_volume.py` | 建玉と出来高の図を描く |
| 6 | `plot_price.py` | 日足の図を描く |
| 7 | `regress_oi_volume.py` | 回転率を出し、翌日の建玉を当日の出来高に回帰して散布図を描く |
| 8 | `plot_intraday.py` | 立会日と休場日に分けた日内プロファイルを描く |
| 9 | `build_volume_side.py` | 出来高をテイカーの向きで買いと売りに分け、帰無対照の標準偏差も求める |
| 10 | `plot_volume_side.py` | 買い・売りの積み上げと売買差の図を描く |
| 11 | `plot_intraday_day.py` | 指定した 1 日の 1 時間ごとの値動きと出来高内訳を描く |

補助: `palette_check.py` は図の配色を計算で検査します(目視しない)。
dataviz の検証器の Python 移植で、明度帯・彩度下限・色覚特性下での分離・
背景との対比を測ります。

実行例:

```bash
uv run python scripts/inventory_s3.py --refresh --coin xyz:MU
uv run python scripts/fetch_fills.py --coin xyz:MU
uv run python scripts/build_oi_volume.py --coin xyz:MU
uv run python scripts/plot_oi_volume.py --coin xyz:MU --unit usd
uv run python scripts/plot_price.py --coin xyz:MU
uv run python scripts/regress_oi_volume.py --coin xyz:MU
uv run python scripts/plot_intraday.py --coin xyz:MU
uv run python scripts/build_volume_side.py --coin xyz:MU
uv run python scripts/plot_volume_side.py --coin xyz:MU
uv run python scripts/plot_intraday_day.py --coin xyz:MU --date 2026-06-24 \
    --event 20:00 --event-label "FQ3 決算発表(米国引け後)"
```

依存は `pyproject.toml` に宣言してあります。初回だけ `uv sync` を実行してください。

## 6. 用語

perp・建玉・出来高・テイカーとメイカー・回転率・立会日と休場日など、
各レポートで共通して使う用語は **[用語辞書(GLOSSARY.md)](GLOSSARY.md)** に
まとめています。

## 7. 記述の方針

- 図と表は、初めて読む人が説明なしで意味を取れることを優先します。
- 数式は LaTeX 形式で書きます。
- 推定値には必ず不確かさの幅と、その根拠となる検証を添えます。
- 将来の情報を説明変数に混ぜないこと(先読みの禁止)を最優先の制約とします。
