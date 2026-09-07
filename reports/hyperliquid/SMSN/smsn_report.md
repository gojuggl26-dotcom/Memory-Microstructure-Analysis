# `xyz:SMSN` — MU の分析一式を当てる(17 分析 + 特徴量 199 本 × 7 ホライズン)

[← xyz:SMSN の分析索引](README.md) ／ [7 銘柄の横並び](../cross_coin_report.md)
／ [予測の定義](../../predicting_definition.md)

対象 `xyz:SMSN`(サムスン電子 perp)／ 標本 2026-05-04 〜 2026-08-10 の 99 日
／ 数字の出所は [suite_headline_xyz_SMSN.json](../../../data/suite_headline_xyz_SMSN.json)

## 先に結論

- **原資産は世界最大のメモリメーカーだが、perp の板は中規模。**
  出来高 \$1,764 万/日は AMD と同水準で、MU の 1/9
- **「約定はスプレッドが狭いときに起きる」が 7 銘柄で最も強い**
  (corr = −0.364)。板が広がっている時間に成行を打つ参加者がいない
- δ の 1 秒逆張り −0.206。山は 1 秒(0.231)で、60 秒でも 0.150 残る
  **遅い減衰の群**(AMD・SKHX と同じ)
- OFI 帯の端は +0.081 と 7 銘柄で最弱。フローの情報が薄い

---

## A. この銘柄はどういう市場か

| 量 | 値 |
|---|---|
| 日次出来高(テイカー側、中央値) | \$17,643,827 |
| 日次取引件数(中央値) | 36,888 |
| 日中平均建玉(中央値) | \$11,699,921 |
| テイカー参加者/日(中央値) | 450 |
| 清算 fill/日(中央値) | 11 |
| 価格の中央値 | \$193 |
| 成行 1 本の中央値 / p99 | 0.967 枚 / 38.4 枚 |
| 買いの割合(平均) | 49.14% |
| \|z\|(HAC) > 2 の日 | 18 / 99 |

![xyz:SMSN 素性](../../../charts/xyz_SMSN_profile.png)

## B. 板の状態は将来の値動きを教えてくれるか

| h | 100ms | 500ms | 1s | 5s | 10s | 30s | 60s |
|---|---|---|---|---|---|---|---|
| 実測の最大 \|r\| | 0.111 | 0.221 | **0.231** | 0.209 | 0.193 | 0.165 | 0.150 |
| 帰無対照の最大 | 0.007 | 0.005 | 0.005 | 0.006 | 0.013 | 0.016 | 0.013 |

![xyz:SMSN 特徴量の予測力](../../../charts/xyz_SMSN_xfeat.png)

- 山は 1 秒(`delta_bp__z` −0.231)。減衰が遅く 60 秒でも 0.150
- mid 建ての δ は +0.063(符号反転は 7 銘柄共通)
- 前半後半の r 相関 0.962

| 指標 | 最大エッジ | セル | 帰無対照 |
|---|---|---|---|
| MicroPrice 乖離 | +0.137 | 閉場日 / > +20bp / k=5 | 0.012 |
| OBI | +0.115 | 立会日 / −1.00〜−0.75 / k=5 | — |
| OFI | +0.081 | 閉場日 / +1〜+2σ / k=1 | — |

![xyz:SMSN microprice](../../../charts/xyz_SMSN_microprice_matrix.png)

![xyz:SMSN obi ofi](../../../charts/xyz_SMSN_obi_ofi_matrix.png)

- Book Slope: β=0.031、HAC t=18.9
- キャンセル率の傾き: 最大差 +0.052bp(立会日/10 秒)、帰無対照 −0.009bp。
  片道費用(実効半スプレッド 2.07bp)の 1/40

![xyz:SMSN book slope](../../../charts/xyz_SMSN_book_slope.png)

![xyz:SMSN cancel rate](../../../charts/xyz_SMSN_cancel_rate.png)

## C. 注文フローはどれだけ自分自身を引きずるか

| 量 | 値 |
|---|---|
| OBI 符号持続の超過(k=1) | +0.231 |
| OBI の lag1(200ms 格子) | +0.877 |
| OFI の lag1(200ms 格子) | −0.035(無記憶) |

![xyz:SMSN 符号の持続](../../../charts/xyz_SMSN_sign_persistence.png)

![xyz:SMSN acf](../../../charts/xyz_SMSN_acf_200ms.png)

![xyz:SMSN sign chain](../../../charts/xyz_SMSN_sign_chain_matrix.png)

![xyz:SMSN var100](../../../charts/xyz_SMSN_var100.png)

## D. 板の中の量どうし

- corr(スプレッド, 約定量) = **−0.364(7 銘柄で最強)**
- corr(最良数量, 約定量) = +0.269

![xyz:SMSN spread flow](../../../charts/xyz_SMSN_spread_flow_corr.png)

![xyz:SMSN markout](../../../charts/xyz_SMSN_markout.png)

![xyz:SMSN resilience](../../../charts/xyz_SMSN_resilience_curve.png)

## 限界

1. bbo(最良 1 段)+ fills だけの物差し。キュー・寿命・口座の層は無い
2. 費用は引いていない。エッジ表は格子の最大値(多重比較あり)
3. MU と同一期間・同一相場つき。原資産は KRX 上場(SKHX と同じ)

数字の再現は [README の再現手順](README.md#再現手順) を参照。
