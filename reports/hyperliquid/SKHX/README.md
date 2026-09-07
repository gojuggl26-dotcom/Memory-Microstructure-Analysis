# `xyz:SKHX` 分析索引

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

| 項目 | 内容 |
|---|---|
| 銘柄コード | `xyz:SKHX` |
| 原資産 | SK ハイニックス(SK hynix)。DRAM と NAND の大手、韓国 KRX 上場 |
| 商品の種別 | 無期限先物(perp)。24 時間 365 日取引 |
| 取引所 | [Hyperliquid](https://hyperliquid.xyz/)([Trade.xyz](https://trade.xyz/) が配備) |
| 標本期間 | 2026 年 5 月 4 日 〜 8 月 10 日(99 日) |
| 規模 | 取引 **113,633 件/日**(7 銘柄で最多)、建玉 **\$1.04 億**(最大)、出来高 \$8,833 万/日 |

**取引回数と建玉は最大なのに、最良気配は 0.64 枚しかない。**
細かい注文が高速に流れる板で、原資産が KRX(米国と別の時間帯)に
上場している唯一の銘柄でもある。

---

## レポート

| レポート | 内容 |
|---|---|
| [MU の分析一式を SKHX で走らせる](skhx_report.md) | 17 分析 + 特徴量 199 本 × 7 ホライズン。OBI 符号持続の超過 +0.287 は 7 銘柄で最大、前半後半の r 相関 0.884 は最小(構造がいちばん動く)。 |
| [7 銘柄の横並び](../cross_coin_report.md) | 板の姿は 20 倍違っても「効き方」の相関は +0.95〜1.00。 |

---

## 図

**約定から見た素性**

![xyz:SKHX 素性](../../../charts/xyz_SKHX_profile.png)

**特徴量 199 本 × 7 ホライズンの予測力**

![xyz:SKHX 特徴量の予測力](../../../charts/xyz_SKHX_xfeat.png)

**建玉と出来高** / **買い売りの内訳**

![xyz:SKHX 建玉と出来高](../../../charts/xyz_SKHX_oi_volume_usd.png)

![xyz:SKHX 買い売り](../../../charts/xyz_SKHX_volume_side.png)

**MicroPrice 帯 × ホライズンの上昇確率** / **OBI・OFI の行列**

![xyz:SKHX microprice](../../../charts/xyz_SKHX_microprice_matrix.png)

![xyz:SKHX obi ofi](../../../charts/xyz_SKHX_obi_ofi_matrix.png)

**Book Slope 回帰** / **キャンセル率の傾き**

![xyz:SKHX book slope](../../../charts/xyz_SKHX_book_slope.png)

![xyz:SKHX cancel rate](../../../charts/xyz_SKHX_cancel_rate.png)

**符号の持続** / **テイカー向きの推移確率**

![xyz:SKHX 符号の持続](../../../charts/xyz_SKHX_sign_persistence.png)

![xyz:SKHX 向きの推移](../../../charts/xyz_SKHX_sign_chain_matrix.png)

**OBI・OFI の自己相関(200ms)** / **100ms 分散と将来リターン**

![xyz:SKHX acf](../../../charts/xyz_SKHX_acf_200ms.png)

![xyz:SKHX var100](../../../charts/xyz_SKHX_var100.png)

**スプレッド・注文量・約定量の関係** / **深さと逆選択**

![xyz:SKHX spread flow](../../../charts/xyz_SKHX_spread_flow_corr.png)

![xyz:SKHX depth](../../../charts/xyz_SKHX_depth.png)

**markout の分解** / **流動性ショックからの回復**

![xyz:SKHX markout](../../../charts/xyz_SKHX_markout.png)

![xyz:SKHX resilience](../../../charts/xyz_SKHX_resilience_curve.png)

**ボラティリティの姿** / **分散と回転率** / **注文サイズ** / **日足**

![xyz:SKHX vol signature](../../../charts/xyz_SKHX_vol_signature.png)

![xyz:SKHX variance](../../../charts/xyz_SKHX_variance_1d_over_1w.png)

![xyz:SKHX order size](../../../charts/xyz_SKHX_order_size_dist.png)

![xyz:SKHX price](../../../charts/xyz_SKHX_price_daily.png)

---

## 数値データ

| ファイル | 中身 |
|---|---|
| [profile_xyz_SKHX.csv](../../../data/profile_xyz_SKHX.csv) | 日次の出来高・件数・価格・参加者 |
| [suite_headline_xyz_SKHX.json](../../../data/suite_headline_xyz_SKHX.json) | 見出しの数字 |
| [xfeat_pred_xyz_SKHX.csv](../../../data/xfeat_pred_xyz_SKHX.csv) | 199 本 × 7 ホライズンの予測力 |
| [xfeat_daily_xyz_SKHX.csv](../../../data/xfeat_daily_xyz_SKHX.csv) | 上位の日次 r と符号一致 |
| [order_size_stats_xyz_SKHX.csv](../../../data/order_size_stats_xyz_SKHX.csv) | 時間帯別の注文サイズ |
| [daily_volume_side_xyz_SKHX.csv](../../../data/daily_volume_side_xyz_SKHX.csv) | 買い売り内訳と HAC z |

## 再現手順

```bash
uv run python scripts/mirror_to_local.py --coin xyz:SKHX
uv run python scripts/coin_profile.py    --coin xyz:SKHX --ref xyz:MU
sh scripts/run_mu_suite.sh xyz:SKHX
uv run python scripts/build_xfeat.py --coin xyz:SKHX
uv run python scripts/fit_xfeat.py   --coin xyz:SKHX --stage pool
uv run python scripts/fit_xfeat.py   --coin xyz:SKHX --stage daily
uv run python scripts/plot_xfeat.py  --coin xyz:SKHX
```