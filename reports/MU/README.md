# xyz:MU 分析索引

<p>
  <a href="https://www.micron.com/"><img src="../../photo/micron-logo.png" alt="Micron Technology" height="76"></a>
</p>

| 項目 | 内容 |
|---|---|
| 銘柄コード | `xyz:MU` |
| 原資産 | マイクロン・テクノロジー(Micron Technology, Inc.)。米国ナスダック上場、ティッカー MU |
| 事業 | DRAM と NAND フラッシュメモリの設計・製造 |
| 商品の種別 | 無期限先物(perp)。満期がなく、24 時間 365 日取引される |
| 取引所 | [Hyperliquid](https://hyperliquid.xyz/)([Trade.xyz](https://trade.xyz/) が配備) |
| 標本期間 | 2026 年 5 月 4 日 〜 8 月 10 日(99 日間) |
| 公式サイト | https://www.micron.com/ |

## レポート一覧


| レポート | 内容 |
|---|---|
| [データ保有状況](mu_inventory_report.md) | 作業用バケットに Micron 銘柄のどのデータが何日分あるかの棚卸し。欠けている 1 日とその復旧方法。 |
| [日次平均建玉と出来高](mu_oi_volume_report.md) | 建玉(OI)と出来高の日次推移。米国市場の休場日には建玉がほとんど動かないことを示す。 |
| [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) | 回転率の算出、当日の出来高と翌日の建玉の回帰(OLS と GLS)、立会日と休場日に分けた日内の出来高と建玉変化。 |

## 図

| 図 | 内容 | 掲載レポート |
|---|---|---|
| [日足チャート](../../charts/xyz_MU_price_daily.png) | 標本期間 99 日分の日足(始値・高値・安値・終値) | [日次平均建玉と出来高](mu_oi_volume_report.md) |
| [建玉と出来高(ドル建て)](../../charts/xyz_MU_oi_volume_usd.png) | 日次平均建玉と日次出来高を名目ドルで表示 | [日次平均建玉と出来高](mu_oi_volume_report.md) |
| [建玉と出来高(枚数)](../../charts/xyz_MU_oi_volume_contracts.png) | 同じ内容を契約枚数で表示 | [日次平均建玉と出来高](mu_oi_volume_report.md) |
| [出来高と翌日建玉の散布図](../../charts/xyz_MU_scatter_oi_volume.png) | 立会日と休場日それぞれの OLS と GLS の当てはめ線つき | [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) |
| [日内プロファイル(2 行 2 列)](../../charts/xyz_MU_intraday_2x2.png) | 立会日と休場日の日内出来高・建玉変化を縦軸共通で比較 | [同上](mu_turnover_intraday_report.md) |
| [立会日の日内出来高](../../charts/xyz_MU_intraday_volume_open.png) | 30 分ごと、68 日の平均 | [同上](mu_turnover_intraday_report.md) |
| [立会日の日内 建玉変化](../../charts/xyz_MU_intraday_oichange_open.png) | 30 分ごと、68 日の平均 | [同上](mu_turnover_intraday_report.md) |
| [休場日の日内出来高](../../charts/xyz_MU_intraday_volume_closed.png) | 30 分ごと、31 日の平均 | [同上](mu_turnover_intraday_report.md) |
| [休場日の日内 建玉変化](../../charts/xyz_MU_intraday_oichange_closed.png) | 30 分ごと、31 日の平均 | [同上](mu_turnover_intraday_report.md) |

## 数値データ

| ファイル | 内容 | 生成スクリプト |
|---|---|---|
| [daily_oi_volume_xyz_MU.csv](../../data/daily_oi_volume_xyz_MU.csv) | 日次の建玉・出来高・取引数・参加者数(99 行) | `build_oi_volume.py` |
| [daily_ohlc_xyz_MU.csv](../../data/daily_ohlc_xyz_MU.csv) | 日次 4 本値(99 行) | `plot_price.py` |
| [daily_turnover_xyz_MU.csv](../../data/daily_turnover_xyz_MU.csv) | 日次回転率(99 行) | `regress_oi_volume.py` |
| [intraday_profile_xyz_MU.csv](../../data/intraday_profile_xyz_MU.csv) | 30 分ごとの日内プロファイル(96 行) | `plot_intraday.py` |

## 再現手順

```bash
uv run python scripts/inventory_s3.py --refresh --coin xyz:MU
uv run python scripts/fetch_fills.py --coin xyz:MU
uv run python scripts/build_oi_volume.py --coin xyz:MU
uv run python scripts/plot_oi_volume.py --coin xyz:MU --unit usd
uv run python scripts/plot_price.py --coin xyz:MU
uv run python scripts/regress_oi_volume.py --coin xyz:MU
uv run python scripts/plot_intraday.py --coin xyz:MU
```

リポジトリ全体の目次は [../../README.md](../../README.md) にあります。
