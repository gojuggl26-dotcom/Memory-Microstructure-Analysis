# xyz:MU 出来高の買い・売り内訳と、標本期間のニュース

[← xyz:MU 分析索引](README.md)

対象は `xyz:MU`、標本期間 2026 年 5 月 4 日から 8 月 10 日までの 99 日間。

再現: `scripts/build_volume_side.py` → `scripts/plot_volume_side.py` /
`scripts/plot_intraday_day.py`

---

## 1. 「買いと売りに分ける」とは何を分けるのか

取引には必ず買い手と売り手が 1 人ずついる。したがって「出来高のうち買いが何割か」
という問いは、そのままでは **常に 50% が答えになってしまい意味を持たない**。

意味を持つのは **どちらが板を取りに行ったか**(アグレッサー、テイカー)である。

| | 意味 | 価格への向き |
|---|---|---|
| テイカー買い | 売り板に当てて買った | 押し上げる |
| テイカー売り | 買い板に当てて売った | 押し下げる |

`node_fills` では `crossed = True` の行がテイカー側にあたる。1 取引につき
`crossed` の行はちょうど 1 行なので、これで二重計上せずに向きが決まる。
実測でも `crossed` 行は 11,174,755 件で、既報の取引数と一致する。

**メイカー側で分けても新しい情報は無い。** メイカー買いはテイカー売りの相手方であり、
枚数は厳密に一致する(実測: テイカー買い 5,764,051 件 = メイカー売り 5,764,051 件)。
分解の軸はアグレッサーしか存在しない。

---

## 2. 図

![xyz:MU 日次出来高を買いと売りに分解した積み上げ棒グラフと、その差(買い − 売り)の図](../../charts/xyz_MU_volume_side.png)

上段の 2 色を足すと、元の日次出来高の図と一致する。
下段が売買差で、灰色の帯は帰無対照(§4)。

**上段だけでは偏りは読めない。** 買い比率の中央値が 49.64% なので、
積み上げの境目はほぼ真ん中に来る。偏りは下段でしか見えない。

---

## 3. 結果

### 全期間

| | 金額 | 枚数 | 件数 | 構成比 |
|---|---:|---:|---:|---:|
| テイカー買い | 10.775B USD | 11,713,416 | 5,764,051 | **49.53%** |
| テイカー売り | 10.981B USD | 11,962,067 | 5,410,704 | **50.47%** |
| 合計 | 21.755B USD | 23,675,483 | 11,174,755 | 100% |

99 日を通したネットは **−205.8M USD**(出来高合計の −0.95%)。
枚数合計 23,675,483 は
[日次平均建玉と出来高](mu_oi_volume_report.md) の値と一致する。

**買いの件数のほうが多いのに、金額では売りが上回る。**
1 件あたりの平均は買い 1,869 USD に対し売り 2,029 USD で、
売り側のほうが 1 件が大きい。

### 日次の買い比率(金額ベース)

| | 値 |
|---|---|
| 最小 | 39.64%(2026-05-10) |
| 中央値 | 49.64% |
| 平均 | 49.61% |
| 最大 | 65.35%(2026-05-09) |
| 50% を上回った日 | 45 / 99 日 |

**★最大・最小の日は、いずれも標本中で最も薄い日である。**
5 月 9 日は 99 日中で出来高が最も少ない日(9.7M USD)で、
買い比率 65.35% といっても金額のネットは **+2.98M USD** にすぎない。
5 月 10 日も薄いほうから 7 番目である。
**比率だけを見て「この日は極端に買われた」と読んではいけない。**
金額と併せて見ること。

### 立会日と休場日

| | 日数 | 買い比率 | 帰無対照の帯の外に出た日 |
|---|---:|---:|---:|
| 立会日 | 68 | 49.49% | 25 |
| 休場日 | 31 | 50.18% | 3 |

偏りは立会日に集中している。休場日は 31 日のうち 3 日しか帯の外に出ない。
これは[前レポート](mu_turnover_intraday_report.md)の「原市場が閉じている間は
新規の建て落ちがほとんど起きない」という結果と整合する。

---

## 4. 帰無対照 —— その偏りは「たまたま」ではないのか

各取引の向きを **独立なコイン投げ**(確率 1/2、数量はそのまま)に置き換えたとき、
1 日の売買差がどこまで散らばるかを計算した。

```math
x_i = \pm(\text{その取引の名目額}), \qquad
\mathrm{Var}\Bigl(\sum_i x_i\Bigr) = \sum_i x_i^2
```

ただし **実際の取引の向きには強い自己相関がある。** 大口が注文を分割すれば、
同じ向きの約定が連続する。iid の帰無はこれを無視するので分散を過小評価する。
そこで Bartlett 核による HAC 分散(100 次まで)も求めた。

```math
\mathrm{Var}_{\mathrm{HAC}} = \sum_i x_i^2
 + 2\sum_{k=1}^{K}\Bigl(1-\frac{k}{K+1}\Bigr)\sum_i x_i x_{i+k}
```

| 帰無 | \|z\| 中央値 | \|z\| 最大 | \|z\| > 1.96 の日数 |
|---|---:|---:|---:|
| iid(自己相関を無視) | 2.91 | 14.64 | **66 / 99** |
| HAC(自己相関を考慮) | 1.24 | 6.03 | **28 / 99** |

**iid の帰無を使うと 99 日中 66 日が「有意」になってしまう。**
これは市場の性質ではなく帰無の作り方の誤りで、
自己相関を織り込むと 28 日に落ちる。図の帯と本文の判定は HAC のほうを使っている。

偶然なら 5% × 99 日 ≒ 5 日なので、**28 日は偶然では説明がつかない。**
日々の売買差には実体がある。

---

## 5. 偏りが大きかった日

| 日付 | ネット | 買い比率 | z(HAC) | 出来高 | その日の値動き |
|---|---:|---:|---:|---:|---:|
| 2026-07-30 | **+52.6M** | 54.10% | +4.45 | 641.9M | **+20.54%** |
| 2026-07-23 | +48.7M | 55.06% | **+6.03** | 480.5M | +0.31% |
| 2026-07-21 | +24.9M | 52.69% | +2.45 | 461.3M | +12.26% |
| 2026-06-08 | +23.1M | 54.02% | +2.97 | 288.1M | +7.27% |
| 2026-07-28 | −24.6M | 48.09% | −2.46 | 643.0M | −6.18% |
| 2026-05-28 | −28.5M | 44.24% | −4.56 | 247.4M | +3.40% |
| 2026-06-10 | −30.1M | 44.06% | −5.26 | 253.9M | −4.71% |
| 2026-06-09 | −36.5M | 45.17% | −3.95 | 378.2M | −3.79% |
| 2026-07-29 | **−53.3M** | 46.85% | −4.30 | 846.2M | **−12.38%** |

最大の買い越しと最大の売り越しが **隣り合う 2 日**(7 月 29 日と 30 日)に来ている。
7 月 29 日は標本中で最も出来高が多い日でもある。

### 売買差と値動きの関係

| | 相関 | $`p`$ | $`n`$ |
|---|---:|---:|---:|
| **同時点**(その日の売買差 と その日のリターン) | **+0.322** | 0.0012 | 98 |
| **予測**(その日の売買差 と 翌日のリターン) | −0.178 | 0.079 | 98 |

**この 2 つは別物なので必ず区別して読むこと。** 同時点の +0.322 は
「テイカーが買い越した日は価格も上がっていた」という当たり前の同時性であって、
予測力ではない。

予測の側(時間契約を満たす向き)は **−0.178 で、5% では有意でない**
(Spearman でも $`p`$ = 0.119)。符号は反転(前日買い越し → 翌日下落)だが、
**「効果が無い」という意味ではなく、99 日ではばらつきに対して検出力が足りない**
という意味である。2 つの相関を見ているので、多重比較としても弱い。

---

## 6. 標本期間の Micron 関連ニュース

以下は、上の日次系列で大きく動いた日について報道を探した結果である。
**日付は UTC 日**(perp の集計単位)で、括弧内が報道された米国株の値動き。

| 日付(UTC) | perp の値動き | 出来事 |
|---|---:|---|
| 05-05 | +15.63% | 時価総額が初めて 7,000 億ドルを突破。メモリ株ラリーが加速 |
| 05-08 | +16.17% | (特定の材料は見つからず。5/11 時点で直近 15 営業日中 11 日上昇と報じられる地合い) |
| 05-26 | +14.71% | 時価総額 **1 兆ドル** を突破。30 日で +54% |
| 06-04 | −8.06% | Broadcom の慎重な AI 見通しをきっかけに半導体全体が下落(株 −8%) |
| 06-05 | −12.34% | 半導体 ETF (SOXX) が −10% と 2020 年 3 月以来の下落。5 月雇用統計が予想を大きく上回り 10 年債利回りが 4.5% 超へ。チップ株の時価総額が計 1 兆ドル消失(株 −13%) |
| 06-11 | +13.03% | (特定の材料は見つからず) |
| 06-22 | +8.83% | 決算前の買い上がり |
| 06-23 | −14.36% | **韓国がレバレッジ ETF への投資規制を打ち出し**、半導体が売られる(株 −13%) |
| **06-24** | **+13.11%** | **FQ3 2026 決算(米国引け後)**。売上 414.6 億ドル(前年同期 93.0 億ドル)、粗利率 84.9%、EPS 24.46 ドル、Q4 見通し 500 億ドル。時間外で株 +15% |
| 07-01 | −12.56% | 四半期明けの反落。株 −11%、時価総額 1,380 億ドル消失。SMH −5% |
| 07-07 | −4.82% | メモリ株が高値から −20% 超え、**弱気相場入り** |
| 07-09 | +4.49% | 米国内の追加投資(数十億ドル規模)を発表 |
| 07-13 | −4.84% | 韓国 KIS が SK ハイニックスの Q2 利益予想を市場予想比 −8% に。HBM4 の出荷遅れが理由(MU/SNDK/WDC が −6%) |
| 07-21 | +12.26% | メモリ主導の反発。株 +12%、SanDisk +14%、SMH +4%。韓国の輸出統計が好調 |
| 07-24 | −7.98% | 韓国 KOSPI の下落が波及。SK ハイニックス −6%、MU −6%、SanDisk −9% |
| 07-28 | −6.18% | **SK ハイニックスの Q2 決算が市場予想を下回る**。売上・利益は伸びたが期待に届かず |
| 07-29 | −12.38% | 売りが深まる。FRB は金利据え置き。半導体全体で時価総額 1 兆ドル超が消失 |
| **07-30** | **+20.54%** | **Samsung が四半期の営業利益・売上で過去最高**を発表し、**チップ不足が 2028 年まで続く**と見通す。Microsoft と Lam Research の決算も好感。SOXX +8%、株 +18% |
| 07-31 | −10.37% | 反発の巻き戻し |

7 月の月間騰落率は、報道された株の **−28.7%** に対し本データの perp は **−29.7%**。

### 見つからなかった日

大きく動いた日(|日次| ≥ 7%)は 20 日ある。そのうち **日付を特定した報道を
見つけられたのは 11 日**で、残りは業界全体の地合いとしてしか説明できないか、
何も見つからなかった(05-08、06-08、06-11、07-14 の 4 日)。
**見つからなかった日を隠さずに書いておく。**

### ★この対応づけは「検定」ではない

**大きく動いた日を先に選んでから、その日のニュースを探した。**
半導体セクターには毎日何かしらの報道があるので、
この手順では **どんな日にも「材料」が見つかってしまう**(ベースレートの問題)。
したがってこの表は仮説の提示であって、「ニュースが価格を動かした」ことの検証ではない。

検証として意味を持つのは、**発表時刻が事前に判っている材料** だけである。
本標本ではそれは 6 月 24 日の決算のみで、次節でその 1 件だけを厳密に検証する。

---

## 7. 決算が板に着弾する瞬間(2026-06-24)

Micron の FQ3 決算は **米国市場の引け後**に発表された。
引けは 16:00 ET = **20:00 UTC**(標本期間は全日が夏時間)。
株式はこの材料を翌営業日の寄りまで価格に反映できないが、
perp は 24 時間動いているのでその場で反映できる。

![2026-06-24 の 1 時間ごとの値動きと、買い・売りに分けた出来高。20:00 UTC に決算発表の縦線](../../charts/xyz_MU_intraday_2026-06-24.png)

| 時刻(UTC) | 終値 | 出来高 |
|---|---:|---:|
| 18:00 | 1,015.10 | 41.1M |
| 19:00 | 1,048.10 | 74.1M |
| **20:00** | **1,201.20** | **198.7M** |
| 21:00 | 1,191.40 | 63.5M |

- 米国の立会時間(13:30–20:00 UTC)には価格はむしろ **下げていた**(1,099 → 1,015)。
- **20:00 UTC の 1 時間で 1,048.10 → 1,201.20(+14.6%)。**
- その 1 時間の出来高 198.7M USD は、**その日の他のどの時間の 2.7 倍**。

発表時刻が事前に判っている材料が、その時刻ちょうどの 1 時間に集中している。
**これは事後の対応づけではなく、時刻を指定した検証である。**

---

## 8. 板データが本当に MU 株を追っているかの検証

前節までの対応づけは、この perp が実際に Micron 株を追っていることを前提にしている。
これを 2 通りで確かめた。

### (a) 値動きの一致(9 件)

| 日付 | 報道された株 | 本データの perp |
|---|---:|---:|
| 06-04 | −8% | −8.06% |
| 06-05 | −13% | −12.34% |
| 06-23 | −13% | −14.36% |
| 07-01 | −11% | −12.56% |
| 07-13 | −6% | −4.84% |
| 07-21 | +12% | +12.26% |
| 07-24 | −6% | −7.98% |
| 07-30 | +18% | +20.54% |
| 7 月の月間 | −28.7% | −29.7% |

### (b) ★水準の一致(時刻を指定した 1 件)

7 月 29 日の報道に「**11:45 a.m. EDT に 776.13 ドル**」という記述がある。
11:45 EDT = **15:45 UTC**。本データの同じ時間帯(15:00–16:00 UTC)の
最後の約定価格は **775.71 ドル**で、**差は 0.05%** である。

これで水準も追随していることが確かめられた。
なお、集計サイト由来の時価総額(7,000 億ドル / 1 兆ドル)から株価を逆算する検証は
**行っていない**。発行済株式数の値によって 8% 以上ぶれ、互いに矛盾する数字も
出回っていたためである。

---

## 9. 限界

1. **§6 の対応づけは検定ではない**(§6 末尾)。厳密な検証は §7 の 1 件のみ。
2. 買い比率の極端な日は薄い日に出る。**比率と金額を必ず併せて見ること**(§3)。
3. 売買差の予測力は −0.178 で有意ではない($`p`$ = 0.079)。
   99 日では検出力が足りないだけの可能性があり、**「効果が無い」とは言えない**。
4. 帰無対照は「向きを入れ替える」ものであって、数量の分布は実測のまま固定している。
   数量そのものが日によって偏る効果は対照に含まれていない。
5. UTC 日で集計しているため、**米国の 1 立会日は 2 つの UTC 日にまたがらない**が、
   引け後(20:00–24:00 UTC)の 4 時間は同じ UTC 日に含まれる。
   決算のような引け後の材料は、株より **1 営業日早く** この系列に現れる(§7)。
6. **`liquidation` 列は 11,174,755 行中 7,695,784 行(68.9%)が null**(取得元の
   スキーマ揺れ)。清算と明示された約定は金額で 0.262% だが、
   **これは下限であって実際の清算比率ではない**。清算を除外した集計はしていない。
7. 1 銘柄の記述統計である。銘柄横断の比較は他の 5 銘柄で同じ手順を実行してから行う。

---

## 出典

- [Micron Technology, Inc. Reports Record Results for the Third Quarter of Fiscal 2026 — GlobeNewswire (2026-06-24)](https://www.globenewswire.com/news-release/2026/06/24/3317151/14450/en/micron-technology-inc-reports-record-results-for-the-third-quarter-of-fiscal-2026.html)
- [Micron (MU) earnings report Q3 2026 — CNBC (2026-06-24)](https://www.cnbc.com/2026/06/24/micron-mu-earnings-report-q3-2026.html)
- [Micron zooms past \$700 billion market cap as rally in memory stocks accelerates — CNBC (2026-05-05)](https://www.cnbc.com/2026/05/05/micron-zooms-past-700-billion-market-cap-rally-in-memory-stocks-.html)
- [Micron shares are rising again despite weak overall market — CNBC (2026-05-11)](https://www.cnbc.com/2026/05/11/micron-shares-are-rising-again-despite-weak-overall-market-why-memory-chip-rally-seems-unstoppable.html)
- [Micron Tumbles 13% As South Korean ETF Warning Fuels Chip Sell-Off — Forbes (2026-06-23)](https://www.forbes.com/sites/antoniopequenoiv/2026/06/23/micron-tumbles-13-as-south-korean-etf-warning-fuels-chip-sell-off/)
- [Micron erases weeks of 2026 rally in shocking move — Yahoo Finance](https://finance.yahoo.com/markets/stocks/articles/micron-erases-weeks-2026-rally-163717129.html)
- [Chip stocks that notched record rallies in second quarter start Q3 with a dud — CNBC (2026-07-01)](https://www.cnbc.com/2026/07/01/chip-stocks-notched-record-rallies-in-second-quarter-start-q3-with-dud.html)
- [Micron stock pops after announcing billions more in U.S. investment — CNBC (2026-07-09)](https://www.cnbc.com/2026/07/09/micron-stock-us-chipmaking.html)
- [Micron, SanDisk, Western Digital Fall 6% as SK Hynix's Weak Outlook Rattles Memory Stocks — 24/7 Wall St. (2026-07-13)](https://247wallst.com/investing/2026/07/13/micron-sandisk-western-digital-fall-6-as-sk-hynixs-weak-outlook-rattles-memory-stocks/)
- [Memory Chips Just Fell Into a Bear Market — The Motley Fool (2026-07-19)](https://www.fool.com/investing/2026/07/19/memory-chips-just-fell-into-a-bear-market-micron-i/)
- [Stock Market Today, July 21: Micron Surges 12% as Semiconductor Strength Lifts Nasdaq — The Motley Fool (2026-07-21)](https://www.fool.com/coverage/stock-market-today/2026/07/21/stock-market-today-july-21-micron-surges-12-as-semiconductor-strength-lifts-nasdaq/)
- [SK Hynix and Micron Sink 6%, SanDisk Drops 9% as Korea Chip Selloff Hits U.S. Memory Stocks — 24/7 Wall St. (2026-07-24)](https://247wallst.com/investing/2026/07/24/sk-hynix-and-micron-sink-6-sandisk-drops-9-as-korea-chip-selloff-hits-u-s-memory-stocks/)
- [AMD, Intel and Micron extend losses as chip stocks get clobbered — CNBC (2026-07-28)](https://www.cnbc.com/2026/07/28/sk-hynix-plunges-semiconductor-selloff-deepens-samsung-softbank.html)
- [Why Micron Stock Dropped Again Today — The Motley Fool (2026-07-29)](https://www.fool.com/investing/2026/07/29/why-micron-stock-dropped-again-today/)
- [Chip stocks shed more than \$1 trillion as selloff hits companies powering AI boom — CNBC (2026-07-29)](https://www.cnbc.com/2026/07/29/chip-selloff-sk-hynix-samsung-softbank.html)
- [AMD and Micron surge, Lam Research climbs 17% as chip stocks rip higher — CNBC (2026-07-30)](https://www.cnbc.com/2026/07/30/chip-stock-rally-lam-research-micron-amd.html)
- [Why Is Micron Stock (MU) Suddenly Jumping Today — July 30, 2026? — TipRanks](https://www.tipranks.com/news/why-is-micron-stock-mu-suddenly-jumping-today-july-30-2026)
- [Why Micron Stock Plummeted 28.7% in July But Is Rebounding in August — The Motley Fool (2026-08-10)](https://www.fool.com/investing/2026/08/10/why-micron-stock-plummeted-287-in-july-but-is-rebo/)
