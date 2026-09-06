# Memory-Microstructure-Analysis

<p>
  <a href="https://www.python.org/"><img src="photo/python-logo.png" alt="Python" height="76"></a>
</p>

無期限先物(perpetual futures、以下 perp)まわりの市場を、
**注文一本ごとの記録まで遡って**調べるリポジトリです。対象はメモリ半導体に
連動する perp(Hyperliquid)と、perp のファンディングレートそのものを取引する板
(Pendle Boros)です。

板に出た注文、その取り消し、約定のすべてが時刻つきで残っているため、
日足や出来高だけでは見えない「板がどう作られ、どう崩れるか」を直接観察できます。
分析はすべて Python で書かれており、同じスクリプトを実行すれば同じ図と数値が出ます。

**入口は取引所ごとに分かれています。** 板の作り・記録の粒度・データの入手経路は
取引所ごとに違い、同じ指標でも意味が変わります。結論をひとまとめにせず、
取引所単位で並べます。

---

## 取引所

| 取引所 | 対象 | 記録の粒度 | 状態 | 入口 |
|---|---|---|---|---|
| **[Hyperliquid](https://hyperliquid.xyz/)** | [Trade.xyz](https://trade.xyz/) が配備したメモリ半導体 perp 6 銘柄 + 対照 1 銘柄 | L4(注文 1 本ごと) | 進行中 | **[Hyperliquid の分析](reports/hyperliquid/README.md)** |
| **[Pendle Boros](https://boros.pendle.finance/)** | 各取引所の perp ファンディングレートを implied APR 建てで取引する板・全 188 市場 | 注文イベント(Arbitrum オンチェーンログ) | 進行中 | **[Pendle Boros の分析](reports/pendle/README.md)** |
| **[Lighter](https://lighter.xyz/)** | メモリ半導体 5 + 米国株 5 + 金属 2 + H100 の 13 銘柄。Binance 同名 12 銘柄とのリードラグ比較つき | L2(50ms バッチ差分)+ 約定 + BBO | 進行中 | **[Lighter の分析](reports/lighter/README.md)** |
| その他の DEX | 未定 | — | 未着手 | — |

新しい取引所を足すときは `reports/<取引所名>/README.md` を作り、この表に 1 行足します。
書き方の規則は [MAINTENANCE.txt](MAINTENANCE.txt) にあります。

## どこから読むか

| 知りたいこと | 行き先 |
|---|---|
| Hyperliquid の結論だけ知りたい | [Hyperliquid の分析 — 何が判ったか](reports/hyperliquid/README.md#何が判ったか) |
| Pendle Boros の結論だけ知りたい | [Pendle Boros の分析 — 主な結果](reports/pendle/README.md#主な結果) |
| 銘柄ごとの分析を読みたい | [`xyz:MU` 分析索引](reports/hyperliquid/MU/README.md) / [`xyz:DRAM` 分析索引](reports/hyperliquid/DRAM/README.md) / [`xyz:INTC` 分析索引](reports/hyperliquid/INTC/README.md) |
| 用語の意味を調べたい | [用語辞書](GLOSSARY.md) |
| 「予測できた」の定義を知りたい | [予測の定義](reports/predicting_definition.md) |
| 数字を読むときの前提を知りたい | 次節 [結果を読むときの前提](#結果を読むときの前提) |
| 自分の手元で再現したい | [再現手順](#再現手順) |

---

## 結果を読むときの前提

本リポジトリのレポートは、以下を守って書かれています。数字を読むときの前提として
先に共有しておきます。

- **将来の情報を説明変数に混ぜない。** 説明変数が確定する時刻は、必ず目的変数の
  期間の開始時刻以前です。各スクリプトの docstring に「x が確定する時刻」と
  「y の期間」を書いてあり、レポート本文にも同じ表を載せています。

- **「予測」と「同時点の関係」を書き分ける。** 同じ窓の中で測った関係は、
  どれだけ強くても予測ではありません。判定の手順は
  [予測の定義](reports/predicting_definition.md) に一本化してあります。

- **推定値には不確かさの幅と帰無対照を必ず添える。** 有意性だけを示して終わりに
  しません。多重比較をしたときは探索した全件数と補正後の閾値を書きます。

- **費用を引いた後の値で採否を語る。** 手数料・スプレッド・スリッページを引くと
  結論が反転することが実際にありました。

- **図は本文なしで読めるように作る。** 軸・単位・標本数・期間は図の中にあります。
  配色は目視ではなく計算で検査しています(`scripts/palette_check.py`)。

## ディレクトリ構成

| 場所 | 用途 |
|---|---|
| `scripts/` | 分析スクリプト。1 つのスクリプトにつき 1 つの目的 |
| `reports/<取引所名>/` | 取引所ごとの入口。`README.md` にその取引所の結論と索引 |
| `reports/<銘柄コード>/` | 銘柄別のレポート。`README.md` が分析索引、それ以外は `*_report.md` |
| `charts/` | スクリプトが生成した図 |
| `data/` | 作業用データ。容量の大きい parquet は版管理から除外し、スクリプトで再生成する |
| `photo/` | ロゴなど、生成物ではない画像 |
| `GLOSSARY.md` | 全レポート共通の用語辞書 |
| `MAINTENANCE.txt` | リポジトリに手を入れる人向けの規則(命名・索引の更新・配色検査など) |

## 再現手順

依存は `pyproject.toml` に宣言してあります。初回だけ `uv sync` を実行してください。
そのあとの実行順は取引所ごとに違うので、**その取引所の入口**を見てください
(例: [Hyperliquid の再現手順](reports/hyperliquid/README.md#5-再現手順))。
銘柄のレポートをすべて再現する完全な実行順は、その銘柄の分析索引の末尾にあります。

スクリプトは接頭辞で役割が判るようになっています。

| 接頭辞 | 役割 | 例 |
|---|---|---|
| `inventory_` | 作業用バケットの中身と行数を棚卸しする | `inventory_s3.py`, `inventory_rows.py` |
| `fetch_` | 必要な列だけを手元に落とす | `fetch_fills.py`, `fetch_bbo.py`, `fetch_l1.py`, `fetch_l1_extra.py` |
| `build_` | 特徴量・集計・検定結果を算出して `data/` に書く | `build_obi_ofi.py`, `build_sign_persistence.py` |
| `plot_` | `data/` を読んで `charts/` に図を書く | `plot_obi_ofi.py`, `plot_microprice.py` |
| `regress_` | 回帰して散布図を描く | `regress_oi_volume.py` |
| `boros_` | Pendle Boros の分析一式(取得・算出・作図。作図は `_plot` 接尾辞) | `boros_pooled_all.py`, `boros_pooled_plot.py` |

このほかに `palette_check.py` があり、図の配色が色覚特性下でも判別できるかを
計算で検査します。

各スクリプトの docstring に、目的・入出力・実行例・時間契約(説明変数が確定する
時刻と目的変数の期間)が書いてあります。


## 用語

perp・建玉・出来高・テイカーとメイカー・回転率・立会日と休場日・OBI・OFI・
Book Slope・HAC など、各レポートで共通して使う用語は
**[用語辞書(GLOSSARY.md)](GLOSSARY.md)** にまとめています。
