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
| [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) | 回転率の算出、当日の出来高と翌日の建玉の回帰(OLS と GLS)、立会日と休場日に分けた日内の出来高と建玉変化。5 日・1 日・1 時間の 3 尺度で実現分散と回転率を並べ、同じ向きに動くことを示す(第 5〜10 節。同時点の関係であって予測力ではない)。 |
| [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md) | テイカー側で出来高を買い・売りに分解し、HAC 帰無対照で偏りを検定。標本期間の主なニュースとの対応づけ、および 6 月 24 日決算発表という「発表時刻が事前に判っている」1 件だけの厳密な検証。 |
| [MicroPrice と midprice の差 → 将来 mid の上昇確率](mu_microprice_report.md) | イベントごとに MicroPrice と mid を算出し、その差の帯 × 予測ホライズン(1〜100 イベント)で上昇確率の行列を立会日・閉場日別に作る。生の確率を 0.5 と比べてはいけない理由(同値が 4〜5 割)を示し、同値を除くと中央帯がちょうど 50% になることを確認。 |

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
| [出来高の買い・売り分解](../../charts/xyz_MU_volume_side.png) | 日次出来高の積み上げ棒(買い・売り)と、その差に対する HAC 帰無対照の帯 | [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md) |
| [決算発表日(2026-06-24)の日中](../../charts/xyz_MU_intraday_2026-06-24.png) | 1 時間ごとの価格と買い・売り出来高。20:00 UTC(米国引け後)の決算発表を縦線で表示 | [同上](mu_volume_side_news_report.md) |
| [5 日窓の日次分散(99 日間)](../../charts/xyz_MU_variance_5d_over_99d.png) | 直前 5 日で測った日次分散と同じ窓の回転率 | [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) |
| [1 日窓の日次分散(1 週間)](../../charts/xyz_MU_variance_1d_over_1w.png) | その日だけで測った日次分散と回転率。土日の落差が出る | [同上](mu_turnover_intraday_report.md) |
| [1 時間窓の分散(1 日)](../../charts/xyz_MU_variance_1h_over_1d.png) | その 1 時間だけで測った分散と回転率。寄付きに集中する | [同上](mu_turnover_intraday_report.md) |
| [寄付き前後 6 時間の出来高と相対スプレッド](../../charts/xyz_MU_open_window_volume_spread.png) | 1 分刻み。買い・売りの積み上げと、時間加重の相対スプレッド | [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) 第 11 節 |
| [寄付き前後 6 時間の分散](../../charts/xyz_MU_open_window_variance.png) | 算出窓 5 分、1 分あたり。中値リターンから算出 | [同上](mu_turnover_intraday_report.md) 第 11 節 |
| [MicroPrice の確率推移行列](../../charts/xyz_MU_microprice_matrix.png) | 差の帯 × ホライズンの上昇確率を立会日・閉場日で並べた行列。色は無条件との差 | [MicroPrice と midprice の差](mu_microprice_report.md) |

## 数値データ

| ファイル | 内容 | 生成スクリプト |
|---|---|---|
| [daily_oi_volume_xyz_MU.csv](../../data/daily_oi_volume_xyz_MU.csv) | 日次の建玉・出来高・取引数・参加者数(99 行) | `build_oi_volume.py` |
| [daily_ohlc_xyz_MU.csv](../../data/daily_ohlc_xyz_MU.csv) | 日次 4 本値(99 行) | `plot_price.py` |
| [daily_turnover_xyz_MU.csv](../../data/daily_turnover_xyz_MU.csv) | 日次回転率(99 行) | `regress_oi_volume.py` |
| [intraday_profile_xyz_MU.csv](../../data/intraday_profile_xyz_MU.csv) | 30 分ごとの日内プロファイル(96 行) | `plot_intraday.py` |
| [daily_volume_side_xyz_MU.csv](../../data/daily_volume_side_xyz_MU.csv) | 日次の買い・売り出来高、買い比率、帰無対照の z 値(99 行) | `build_volume_side.py` |
| [variance_daily_xyz_MU.csv](../../data/variance_daily_xyz_MU.csv) | 日次の実現分散(1 日窓・5 日窓)と回転率(99 行) | `build_variance.py` |
| [variance_hourly_xyz_MU.csv](../../data/variance_hourly_xyz_MU.csv) | 1 時間ごとの実現分散と回転率(2,376 行) | `build_variance.py` |
| [open_window_xyz_MU.csv](../../data/open_window_xyz_MU.csv) | 寄付き前後 6 時間の 1 分ごとの出来高・スプレッド・分散(360 行) | `build_open_window.py` |
| microprice_cells_xyz_MU.parquet | 日区分 × 差の帯 × ホライズンの上昇確率・区間・帰無対照(容量のため版管理外) | `build_microprice.py` |

## 再現手順

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
uv run python scripts/build_variance.py --coin xyz:MU
uv run python scripts/build_open_window.py --coin xyz:MU
uv run python scripts/fetch_bbo.py --coin xyz:MU
uv run python scripts/build_microprice.py --coin xyz:MU
uv run python scripts/plot_microprice.py --coin xyz:MU
```

リポジトリ全体の目次は [../../README.md](../../README.md) にあります。
