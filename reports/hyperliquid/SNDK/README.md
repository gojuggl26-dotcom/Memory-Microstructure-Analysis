# `xyz:SNDK` 分析索引

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

| 項目 | 内容 |
|---|---|
| 銘柄コード | `xyz:SNDK` |
| 原資産 | サンディスク(SanDisk)。NAND フラッシュメモリの大手、米国ナスダック上場 |
| 商品の種別 | 無期限先物(perp)。24 時間 365 日取引 |
| 取引所 | [Hyperliquid](https://hyperliquid.xyz/)([Trade.xyz](https://trade.xyz/) が配備) |
| 標本期間 | 2026 年 5 月 4 日 〜 8 月 10 日(99 日) |
| 規模 | 日次出来高の中央値 **\$1.11 億**(7 銘柄中 2 位)、建玉 \$3,463 万、取引 63,775 件/日 |

**回転が速い投機の板。** 出来高 ÷ 建玉が 3.2 回/日と 7 銘柄で最も高く、
清算も 49 件/日と最多。それでいてスプレッド中央値 1.30bp・実効半スプレッド
0.85bp は MU と並んで最も狭く、**予測力の減衰は 7 銘柄で最速**
(1 秒 0.167 → 10 秒 0.065)。回転が速い市場ほど歪みがすぐ消える。

---

## レポート

| レポート | 内容 |
|---|---|
| [MU の分析一式を SNDK で走らせる](sndk_report.md) | 17 分析 + 特徴量 199 本 × 7 ホライズン。OBI/OFI 帯の端(0.23)は 7 銘柄で最大、しかし δ の逆張り(−0.161)は最弱。 |
| [7 銘柄の横並び](../cross_coin_report.md) | 板の姿は 20 倍違っても「効き方」の相関は +0.95〜1.00。 |
| [7 銘柄 × 84 本の十分位分析](../decile_all_report.md)(7 銘柄共通) | 最良気配と約定だけで作れる 84 本 × 9 ホライズン(100ms〜60 秒) × mid/micro。十分位の境目は**前日の分布**、標準誤差は日でクラスタ、帰無対照は 1 営業日ずらし。有意 55.3%(プラセボ 0.0%)。MU との効き方の順位相関 **0.977** は 21 ペアで最高(中央スプレッドが 1.3 / 1.1bp と近い)。最良は `obi1 → mid_60s` の片側 0.77bp で、費用 2.89bp の 27%。 ★費用は銘柄ごとに違い、判定は \|D10 − D1\| ではなく**片側 max(\|D1\|,\|D10\|)** で行う。**7 銘柄 4,530 本の有意なセルのうち費用を越えたものはゼロ**。 |
| [7 銘柄のメイカー検証](../mm_all_report.md)(7 銘柄共通) | 在庫つきメイカー(両側に指値・\|q\|≤1 で FIFO 相殺・BBO 追随)を `xyz:MU` と同じ設定で当て、1 組あたり損益を恒等式で分解する。入口の門・無作為の門(帰無対照)・発注遅延 130 ms・執行できる数量を順に入れる。半スプレッド 2.143bp に対し劣化 2.121bp(**99%**)でちょうど相殺し、約定時の edge は 0.022bp しか残らない。 ★7 銘柄のどれも、費用を引く前ですら**出す価値が無い**。 |

---

## 図

**約定から見た素性** — 価格・出来高・参加者・時刻分布。

![xyz:SNDK 素性](../../../charts/xyz_SNDK_profile.png)

**特徴量 199 本 × 7 ホライズンの予測力** — 分類別到達点・上位・日次符号。

![xyz:SNDK 特徴量の予測力](../../../charts/xyz_SNDK_xfeat.png)

**建玉と出来高** / **買い売りの内訳**

![xyz:SNDK 建玉と出来高](../../../charts/xyz_SNDK_oi_volume_usd.png)

![xyz:SNDK 買い売り](../../../charts/xyz_SNDK_volume_side.png)

**MicroPrice 帯 × ホライズンの上昇確率** / **OBI・OFI の行列**

![xyz:SNDK microprice](../../../charts/xyz_SNDK_microprice_matrix.png)

![xyz:SNDK obi ofi](../../../charts/xyz_SNDK_obi_ofi_matrix.png)

**Book Slope 回帰** / **キャンセル率の傾き**

![xyz:SNDK book slope](../../../charts/xyz_SNDK_book_slope.png)

![xyz:SNDK cancel rate](../../../charts/xyz_SNDK_cancel_rate.png)

**符号の持続** / **テイカー向きの推移確率**

![xyz:SNDK 符号の持続](../../../charts/xyz_SNDK_sign_persistence.png)

![xyz:SNDK 向きの推移](../../../charts/xyz_SNDK_sign_chain_matrix.png)

**OBI・OFI の自己相関(200ms)** / **100ms 分散と将来リターン**

![xyz:SNDK acf](../../../charts/xyz_SNDK_acf_200ms.png)

![xyz:SNDK var100](../../../charts/xyz_SNDK_var100.png)

**スプレッド・注文量・約定量の関係** / **深さと逆選択**

![xyz:SNDK spread flow](../../../charts/xyz_SNDK_spread_flow_corr.png)

![xyz:SNDK depth](../../../charts/xyz_SNDK_depth.png)

**markout の分解** / **流動性ショックからの回復**

![xyz:SNDK markout](../../../charts/xyz_SNDK_markout.png)

![xyz:SNDK resilience](../../../charts/xyz_SNDK_resilience_curve.png)

**ボラティリティの姿(シグネチャ・推移・指標間)**

![xyz:SNDK vol signature](../../../charts/xyz_SNDK_vol_signature.png)

**分散と回転率** / **注文サイズの分布** / **日足**

![xyz:SNDK variance](../../../charts/xyz_SNDK_variance_1d_over_1w.png)

![xyz:SNDK order size](../../../charts/xyz_SNDK_order_size_dist.png)

![xyz:SNDK price](../../../charts/xyz_SNDK_price_daily.png)

---

## 数値データ

| ファイル | 中身 |
|---|---|
| [mmgate_fit_xyz_SNDK.csv](../../../data/mmgate_fit_xyz_SNDK.csv) | 入口の門の学習・評価の別と通過率 |
| [inv_days_xyz_SNDK_q1.csv](../../../data/inv_days_xyz_SNDK_q1.csv) | 在庫つきメイカーの日次(門なし・幽霊注文) |
| [decile_null_xyz_SNDK_bbo.csv](../../../data/decile_null_xyz_SNDK_bbo.csv) | 同じ表のプラセボ(1 営業日ずらし) |
| [decile_sum_xyz_SNDK_bbo.csv](../../../data/decile_sum_xyz_SNDK_bbo.csv) | 84 本 × 18 目的変数の十分位要約(`build_decile.py`) |
| [profile_xyz_SNDK.csv](../../../data/profile_xyz_SNDK.csv) | 日次の出来高・件数・価格・参加者 |
| [suite_headline_xyz_SNDK.json](../../../data/suite_headline_xyz_SNDK.json) | 見出しの数字(レポートが引く値) |
| [xfeat_pred_xyz_SNDK.csv](../../../data/xfeat_pred_xyz_SNDK.csv) | 199 本 × 7 ホライズンの前向き/後ろ向き/帰無対照 |
| [xfeat_daily_xyz_SNDK.csv](../../../data/xfeat_daily_xyz_SNDK.csv) | 上位の日次 r と符号一致 |
| [order_size_stats_xyz_SNDK.csv](../../../data/order_size_stats_xyz_SNDK.csv) | 時間帯別の注文サイズ |
| [daily_volume_side_xyz_SNDK.csv](../../../data/daily_volume_side_xyz_SNDK.csv) | 買い売り内訳と HAC z |

## 再現手順

```bash
uv run python scripts/mirror_to_local.py --coin xyz:SNDK
uv run python scripts/coin_profile.py    --coin xyz:SNDK --ref xyz:MU
sh scripts/run_mu_suite.sh xyz:SNDK
uv run python scripts/build_xfeat.py --coin xyz:SNDK
uv run python scripts/fit_xfeat.py   --coin xyz:SNDK --stage pool
uv run python scripts/fit_xfeat.py   --coin xyz:SNDK --stage daily
uv run python scripts/plot_xfeat.py  --coin xyz:SNDK
uv run python scripts/build_featbbo.py --coin xyz:SNDK                    # 84 本(bbo + fills のみ)
uv run python scripts/build_fwd.py      --coin xyz:SNDK --src featbbo --suffix _bbo
uv run python scripts/build_decile.py   --coin xyz:SNDK --src featbbo --suffix _bbo
uv run python scripts/build_inventory.py --coin xyz:SNDK --qmax 1 --posts
uv run python scripts/build_mmgate.py      --coin xyz:SNDK
uv run python scripts/build_mmgate_null.py --coin xyz:SNDK
uv run python scripts/build_inventory.py --coin xyz:SNDK --qmax 1 --gate data/entrygate_xyz_SNDK_mmg.parquet
```