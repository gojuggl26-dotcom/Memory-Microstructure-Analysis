# `xyz:KIOXIA` 分析索引

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

| 項目 | 内容 |
|---|---|
| 銘柄コード | `xyz:KIOXIA` |
| 原資産 | キオクシア(Kioxia)。NAND フラッシュメモリ大手、東証上場 |
| 商品の種別 | 無期限先物(perp)。24 時間 365 日取引 |
| 取引所 | [Hyperliquid](https://hyperliquid.xyz/)([Trade.xyz](https://trade.xyz/) が配備) |
| 標本期間 | **2026 年 6 月 25 日 〜 8 月 10 日(47 日)** — 板の稼働開始が他銘柄より 52 日遅い |
| 規模 | 日次出来高 \$410 万、取引 13,798 件/日、建玉 \$343 万 |

**7 銘柄で唯一、板が事実上まだ機能していない。**
スプレッド中央値 13.9bp(67 ティック)、最良気配 0.19 枚、約定は
10 秒に 0 件台。fills は 5 月 4 日からあるが、**bbo(板)は 99 ファイル中
52 が空**で、板の記録は 6 月 25 日から始まる。

---

## レポート

| レポート | 内容 |
|---|---|
| [MU の分析一式を KIOXIA で走らせる](kioxia_report.md) | 17 分析 + 特徴量 199 本 × 7 ホライズン。**予測力の山が 10 秒(0.359)** — 歪んだ気配が何十秒も放置される板。Book Slope は唯一立たない(t=1.8)。 |
| [7 銘柄の横並び](../cross_coin_report.md) | 板の姿は 20 倍違っても「効き方」の相関は +0.95〜1.00。 |

---

## 図

**約定から見た素性**

![xyz:KIOXIA 素性](../../../charts/xyz_KIOXIA_profile.png)

**特徴量 199 本 × 7 ホライズンの予測力**

![xyz:KIOXIA 特徴量の予測力](../../../charts/xyz_KIOXIA_xfeat.png)

**建玉と出来高** / **買い売りの内訳**

![xyz:KIOXIA 建玉と出来高](../../../charts/xyz_KIOXIA_oi_volume_usd.png)

![xyz:KIOXIA 買い売り](../../../charts/xyz_KIOXIA_volume_side.png)

**MicroPrice 帯 × ホライズンの上昇確率** / **OBI・OFI の行列**

![xyz:KIOXIA microprice](../../../charts/xyz_KIOXIA_microprice_matrix.png)

![xyz:KIOXIA obi ofi](../../../charts/xyz_KIOXIA_obi_ofi_matrix.png)

**Book Slope 回帰** / **キャンセル率の傾き**

![xyz:KIOXIA book slope](../../../charts/xyz_KIOXIA_book_slope.png)

![xyz:KIOXIA cancel rate](../../../charts/xyz_KIOXIA_cancel_rate.png)

**符号の持続** / **テイカー向きの推移確率**

![xyz:KIOXIA 符号の持続](../../../charts/xyz_KIOXIA_sign_persistence.png)

![xyz:KIOXIA 向きの推移](../../../charts/xyz_KIOXIA_sign_chain_matrix.png)

**OBI・OFI の自己相関(200ms)** / **100ms 分散と将来リターン**

![xyz:KIOXIA acf](../../../charts/xyz_KIOXIA_acf_200ms.png)

![xyz:KIOXIA var100](../../../charts/xyz_KIOXIA_var100.png)

**スプレッド・注文量・約定量の関係** / **深さと逆選択**

![xyz:KIOXIA spread flow](../../../charts/xyz_KIOXIA_spread_flow_corr.png)

![xyz:KIOXIA depth](../../../charts/xyz_KIOXIA_depth.png)

**markout の分解** / **流動性ショックからの回復**

![xyz:KIOXIA markout](../../../charts/xyz_KIOXIA_markout.png)

![xyz:KIOXIA resilience](../../../charts/xyz_KIOXIA_resilience_curve.png)

**ボラティリティの姿** / **分散と回転率** / **注文サイズ** / **日足**

![xyz:KIOXIA vol signature](../../../charts/xyz_KIOXIA_vol_signature.png)

![xyz:KIOXIA variance](../../../charts/xyz_KIOXIA_variance_1d_over_1w.png)

![xyz:KIOXIA order size](../../../charts/xyz_KIOXIA_order_size_dist.png)

![xyz:KIOXIA price](../../../charts/xyz_KIOXIA_price_daily.png)

---

## 数値データ

| ファイル | 中身 |
|---|---|
| [profile_xyz_KIOXIA.csv](../../../data/profile_xyz_KIOXIA.csv) | 日次の出来高・件数・価格・参加者 |
| [suite_headline_xyz_KIOXIA.json](../../../data/suite_headline_xyz_KIOXIA.json) | 見出しの数字 |
| [xfeat_pred_xyz_KIOXIA.csv](../../../data/xfeat_pred_xyz_KIOXIA.csv) | 199 本 × 7 ホライズンの予測力 |
| [xfeat_daily_xyz_KIOXIA.csv](../../../data/xfeat_daily_xyz_KIOXIA.csv) | 上位の日次 r と符号一致 |
| [order_size_stats_xyz_KIOXIA.csv](../../../data/order_size_stats_xyz_KIOXIA.csv) | 時間帯別の注文サイズ |
| [daily_volume_side_xyz_KIOXIA.csv](../../../data/daily_volume_side_xyz_KIOXIA.csv) | 買い売り内訳と HAC z |

## 再現手順

```bash
uv run python scripts/mirror_to_local.py --coin xyz:KIOXIA
uv run python scripts/coin_profile.py    --coin xyz:KIOXIA --ref xyz:MU
sh scripts/run_mu_suite.sh xyz:KIOXIA
uv run python scripts/build_xfeat.py --coin xyz:KIOXIA
uv run python scripts/fit_xfeat.py   --coin xyz:KIOXIA --stage pool
uv run python scripts/fit_xfeat.py   --coin xyz:KIOXIA --stage daily
uv run python scripts/plot_xfeat.py  --coin xyz:KIOXIA
```