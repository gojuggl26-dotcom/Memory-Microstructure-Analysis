# `xyz:SKHX` — MU の分析一式を当てる(17 分析 + 特徴量 199 本 × 7 ホライズン)

[← xyz:SKHX の分析索引](README.md) ／ [7 銘柄の横並び](../cross_coin_report.md)
／ [予測の定義](../../predicting_definition.md)

対象 `xyz:SKHX`(SK ハイニックス perp)／ 標本 2026-05-04 〜 2026-08-10 の 99 日
／ 数字の出所は [suite_headline_xyz_SKHX.json](../../../data/suite_headline_xyz_SKHX.json)

## 先に結論

- **建玉 \$1.04 億は 7 銘柄で最大、取引 113,633 件/日も最多。**
  それでいて最良気配は 0.64 枚と薄く、細かい注文が高速で流れる板
- **OBI の符号持続(+0.287)と OBI 帯の端(+0.192)は 7 銘柄の上位。**
  板の偏りの「粘り」がいちばん強い
- **前半後半で割った r の相関は 0.884 と 7 銘柄で最小。**
  効く構造そのものは同じだが、強さの入れ替わりが最も大きい
- δ の 1 秒逆張り −0.209、mid 建てで **+0.114(7 銘柄で最大)**。
  mid が microprice に追いつく動きが最も強い

---

## A. この銘柄はどういう市場か

| 量 | 値 |
|---|---|
| 日次出来高(テイカー側、中央値) | \$88,334,939 |
| 日次取引件数(中央値) | **113,633** |
| 日中平均建玉(中央値) | **\$104,016,209** |
| テイカー参加者/日(中央値) | 1,232 |
| 清算 fill/日(中央値) | 42 |
| 価格の中央値 | \$1,316 |
| 成行 1 本の中央値 / p99 | 0.100 枚 / 9.8 枚 |
| 買いの割合(平均) | 49.96% |
| \|z\|(HAC) > 2 の日 | 29 / 99 |

![xyz:SKHX 素性](../../../charts/xyz_SKHX_profile.png)

回転(出来高 ÷ 建玉)は 0.85 回/日で、SNDK(3.2)とは対照的に
**持ち高を張る参加者の板**。原資産の取引所が KRX なので、他 6 銘柄
(米国上場)と立会時間の重なり方が異なる点に注意。

## B. 板の状態は将来の値動きを教えてくれるか

| h | 100ms | 500ms | 1s | 5s | 10s | 30s | 60s |
|---|---|---|---|---|---|---|---|
| 実測の最大 \|r\| | 0.102 | 0.201 | **0.210** | 0.187 | 0.168 | 0.130 | 0.111 |
| 帰無対照の最大 | 0.010 | 0.008 | 0.013 | 0.010 | 0.017 | 0.013 | 0.011 |

![xyz:SKHX 特徴量の予測力](../../../charts/xyz_SKHX_xfeat.png)

- 山は 1 秒(`delta_bp__dev` −0.210)。60 秒でも 0.111 残り、
  減衰は MU・SNDK より明確に遅い
- 上位は δ 系と OBI×スプレッドの交互作用。日次符号一致は 8〜10 割

| 指標 | 最大エッジ | セル | 帰無対照 |
|---|---|---|---|
| MicroPrice 乖離 | +0.136 | 立会日 / +1〜+5bp / k=1 | 0.028 |
| OBI | +0.192 | 立会日 / +0.75〜+1.00 / k=1 | — |
| OFI | +0.194 | 閉場日 / < −2σ / k=5 | — |

![xyz:SKHX microprice](../../../charts/xyz_SKHX_microprice_matrix.png)

![xyz:SKHX obi ofi](../../../charts/xyz_SKHX_obi_ofi_matrix.png)

- Book Slope: β=0.045、HAC t=12.4
- キャンセル率の傾き: 最大差 +0.084bp(立会日/10 秒)、帰無対照 −0.005bp。
  片道費用(実効半スプレッド 1.32bp)の 1/16

![xyz:SKHX book slope](../../../charts/xyz_SKHX_book_slope.png)

![xyz:SKHX cancel rate](../../../charts/xyz_SKHX_cancel_rate.png)

## C. 注文フローはどれだけ自分自身を引きずるか

| 量 | 値 |
|---|---|
| OBI 符号持続の超過(k=1) | **+0.287(7 銘柄で最大)** |
| OBI の lag1(200ms 格子) | +0.827 |
| OFI の lag1(200ms 格子) | +0.004(無記憶) |

![xyz:SKHX 符号の持続](../../../charts/xyz_SKHX_sign_persistence.png)

![xyz:SKHX acf](../../../charts/xyz_SKHX_acf_200ms.png)

![xyz:SKHX sign chain](../../../charts/xyz_SKHX_sign_chain_matrix.png)

![xyz:SKHX var100](../../../charts/xyz_SKHX_var100.png)

## D. 板の中の量どうし

- corr(スプレッド, 約定量) = −0.144
- corr(最良数量, 約定量) = +0.301

![xyz:SKHX spread flow](../../../charts/xyz_SKHX_spread_flow_corr.png)

![xyz:SKHX markout](../../../charts/xyz_SKHX_markout.png)

![xyz:SKHX resilience](../../../charts/xyz_SKHX_resilience_curve.png)

## 限界

1. bbo(最良 1 段)+ fills だけの物差し。キュー・寿命・口座の層は無い
2. 費用は引いていない。エッジ表は格子の最大値(多重比較あり)
3. MU と同一期間・同一相場つき。原資産の立会時間だけ他銘柄と異なる

数字の再現は [README の再現手順](README.md#再現手順) を参照。
