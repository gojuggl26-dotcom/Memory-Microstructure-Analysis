# `xyz:SMSN` 分析索引

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

| 項目 | 内容 |
|---|---|
| 銘柄コード | `xyz:SMSN` |
| 原資産 | サムスン電子(Samsung Electronics)。メモリ半導体の最大手、韓国 KRX 上場 |
| 商品の種別 | 無期限先物(perp)。24 時間 365 日取引 |
| 取引所 | [Hyperliquid](https://hyperliquid.xyz/)([Trade.xyz](https://trade.xyz/) が配備) |
| 標本期間 | 2026 年 5 月 4 日 〜 8 月 10 日(99 日) |
| 規模 | 日次出来高 \$1,764 万、取引 36,888 件/日、建玉 \$1,170 万 |

原資産は世界最大のメモリメーカーだが、この perp の板は**中規模で遅め**
(スプレッド 3.65bp、約定 10 秒に 1 件)。
スプレッドが狭いときに約定が集中する傾向(−0.364)は 7 銘柄で最強。

---

## レポート

| レポート | 内容 |
|---|---|
| [MU の分析一式を SMSN で走らせる](smsn_report.md) | 17 分析 + 特徴量 199 本 × 7 ホライズン。1 秒の最大 \|r\| 0.231。corr(スプレッド, 約定量) −0.364 は 7 銘柄で最強。 |
| [7 銘柄の横並び](../cross_coin_report.md) | 板の姿は 20 倍違っても「効き方」の相関は +0.95〜1.00。 |

---

## 図

**約定から見た素性**

![xyz:SMSN 素性](../../../charts/xyz_SMSN_profile.png)

**特徴量 199 本 × 7 ホライズンの予測力**

![xyz:SMSN 特徴量の予測力](../../../charts/xyz_SMSN_xfeat.png)

**建玉と出来高** / **買い売りの内訳**

![xyz:SMSN 建玉と出来高](../../../charts/xyz_SMSN_oi_volume_usd.png)

![xyz:SMSN 買い売り](../../../charts/xyz_SMSN_volume_side.png)

**MicroPrice 帯 × ホライズンの上昇確率** / **OBI・OFI の行列**

![xyz:SMSN microprice](../../../charts/xyz_SMSN_microprice_matrix.png)

![xyz:SMSN obi ofi](../../../charts/xyz_SMSN_obi_ofi_matrix.png)

**Book Slope 回帰** / **キャンセル率の傾き**

![xyz:SMSN book slope](../../../charts/xyz_SMSN_book_slope.png)

![xyz:SMSN cancel rate](../../../charts/xyz_SMSN_cancel_rate.png)

**符号の持続** / **テイカー向きの推移確率**

![xyz:SMSN 符号の持続](../../../charts/xyz_SMSN_sign_persistence.png)

![xyz:SMSN 向きの推移](../../../charts/xyz_SMSN_sign_chain_matrix.png)

**OBI・OFI の自己相関(200ms)** / **100ms 分散と将来リターン**

![xyz:SMSN acf](../../../charts/xyz_SMSN_acf_200ms.png)

![xyz:SMSN var100](../../../charts/xyz_SMSN_var100.png)

**スプレッド・注文量・約定量の関係** / **深さと逆選択**

![xyz:SMSN spread flow](../../../charts/xyz_SMSN_spread_flow_corr.png)

![xyz:SMSN depth](../../../charts/xyz_SMSN_depth.png)

**markout の分解** / **流動性ショックからの回復**

![xyz:SMSN markout](../../../charts/xyz_SMSN_markout.png)

![xyz:SMSN resilience](../../../charts/xyz_SMSN_resilience_curve.png)

**ボラティリティの姿** / **分散と回転率** / **注文サイズ** / **日足**

![xyz:SMSN vol signature](../../../charts/xyz_SMSN_vol_signature.png)

![xyz:SMSN variance](../../../charts/xyz_SMSN_variance_1d_over_1w.png)

![xyz:SMSN order size](../../../charts/xyz_SMSN_order_size_dist.png)

![xyz:SMSN price](../../../charts/xyz_SMSN_price_daily.png)

---

## 数値データ

| ファイル | 中身 |
|---|---|
| [profile_xyz_SMSN.csv](../../../data/profile_xyz_SMSN.csv) | 日次の出来高・件数・価格・参加者 |
| [suite_headline_xyz_SMSN.json](../../../data/suite_headline_xyz_SMSN.json) | 見出しの数字 |
| [xfeat_pred_xyz_SMSN.csv](../../../data/xfeat_pred_xyz_SMSN.csv) | 199 本 × 7 ホライズンの予測力 |
| [xfeat_daily_xyz_SMSN.csv](../../../data/xfeat_daily_xyz_SMSN.csv) | 上位の日次 r と符号一致 |
| [order_size_stats_xyz_SMSN.csv](../../../data/order_size_stats_xyz_SMSN.csv) | 時間帯別の注文サイズ |
| [daily_volume_side_xyz_SMSN.csv](../../../data/daily_volume_side_xyz_SMSN.csv) | 買い売り内訳と HAC z |

## 再現手順

```bash
uv run python scripts/mirror_to_local.py --coin xyz:SMSN
uv run python scripts/coin_profile.py    --coin xyz:SMSN --ref xyz:MU
sh scripts/run_mu_suite.sh xyz:SMSN
uv run python scripts/build_xfeat.py --coin xyz:SMSN
uv run python scripts/fit_xfeat.py   --coin xyz:SMSN --stage pool
uv run python scripts/fit_xfeat.py   --coin xyz:SMSN --stage daily
uv run python scripts/plot_xfeat.py  --coin xyz:SMSN
```