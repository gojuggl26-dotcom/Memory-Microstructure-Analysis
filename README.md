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
| `xyz:MU`(マイクロン) | **[分析索引を開く](reports/MU/README.md)** | 3 |
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

| 銘柄コード | 対象 | 備考 |
|---|---|---|
| `xyz:DRAM` | DRAM 価格指数 | 2026 年 5 月 4 日 15 時 33 分(UTC)が最初の記録 |
| `xyz:KIOXIA` | キオクシア | |
| `xyz:MU` | マイクロン・テクノロジー | |
| `xyz:SKHX` | SK ハイニックス | |
| `xyz:SMSN` | サムスン電子 | |
| `xyz:SNDK` | サンディスク | |

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

実行例:

```bash
uv run python scripts/inventory_s3.py --refresh --coin xyz:MU
uv run python scripts/fetch_fills.py --coin xyz:MU
uv run python scripts/build_oi_volume.py --coin xyz:MU
uv run python scripts/plot_oi_volume.py --coin xyz:MU --unit usd
uv run python scripts/plot_price.py --coin xyz:MU
uv run python scripts/regress_oi_volume.py --coin xyz:MU
uv run python scripts/plot_intraday.py --coin xyz:MU
```

## 6. 用語

**perp(無期限先物)**
満期のない先物です。現物価格との乖離は、保有者どうしが定期的に支払う調達コストによって
調整されます。

**建玉(OI, open interest)**
ある時点で決済されずに残っている契約の総量です。買い持ちの合計と売り持ちの合計は
必ず一致するため、片側だけを数えます。参加者 $`u`$ の保有量を $`q_u(t)`$ と書くと、

```math
\mathrm{OI}(t) \;=\; \sum_{u} \max\bigl(q_u(t),\, 0\bigr)
```

と定義されます。本リポジトリでは建玉の記録が入手できないため、約定履歴から
$`q_u(t)`$ を復元してこの式で求めています。復元の手順と検証結果は、
各銘柄の分析索引([xyz:MU の索引](reports/MU/README.md) など)から辿れるレポートにあります。

**出来高**
一定期間に成立した取引数量です。約定記録には 1 つの取引につき 2 行(買い手と売り手)が
入るため、価格を提示した側ではなく取りに行った側だけを数えて二重計上を避けています。

**回転率(turnover)**
1 日の出来高がその日の平均建玉の何倍にあたるかを表す指標です。分子と分母がどちらも
枚数なので単位を持ちません。

```math
\mathrm{Turnover}_t = \frac{\mathrm{Volume}_t}{\mathrm{OI}_t}
```

**日足**
1 日を 1 本にまとめた値動きの表示です。始値、高値、安値、終値の 4 つの値からなります。

**立会日と休場日**
本リポジトリでは、原資産である米国株が取引される日を立会日、週末と祝日を休場日と
呼びます。判定にはニューヨーク証券取引所の取引カレンダーを用います。
perp 自体は 24 時間 365 日動きますが、原資産の価格が動くのは立会日だけです。

**金額の表記**
図と表では `M` を 100 万、`B` を 10 億の意味で用います。通貨はすべて米ドルです。

## 7. 記述の方針

- 図と表は、初めて読む人が説明なしで意味を取れることを優先します。
- 数式は LaTeX 形式で書きます。
- 推定値には必ず不確かさの幅と、その根拠となる検証を添えます。
- 将来の情報を説明変数に混ぜないこと(先読みの禁止)を最優先の制約とします。
