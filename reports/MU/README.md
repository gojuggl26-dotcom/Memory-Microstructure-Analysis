# `xyz:MU` 分析索引

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

レポートは下で**大きい話から細かい話へ** A→D の順に並べています。
初めて読む場合は A から順に追うと前提が揃います。

各レポートは冒頭に「何を測ったか」と「時間契約」(説明変数が確定する時刻と
目的変数の期間)を置いています。用語は [用語辞書](../../GLOSSARY.md)、
「予測できた」の定義は [予測の定義](../predicting_definition.md) にあります。

---

## レポート

### A. この銘柄はどういう市場か

まず商品としての素性を掴みます。どれだけ取引され、誰がどの時間に動き、
1 本の注文はどのくらいの大きさかを見ます。

| レポート | 内容 |
|---|---|
| [データ保有状況](mu_inventory_report.md) | 作業用バケットに Micron 銘柄のどのデータが何日分あるかの棚卸し。欠けている 1 日とその復旧方法。 |
| [日次平均建玉と出来高](mu_oi_volume_report.md) | 建玉(OI)と出来高の日次推移。24 時間取引できる商品にもかかわらず、米国市場の休場日には建玉がほとんど動かないことを示す。 |
| [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md) | 回転率の算出、当日の出来高と翌日の建玉の回帰(OLS と GLS)、立会日と休場日に分けた日内の出来高と建玉変化。5 日・1 日・1 時間の 3 尺度で実現分散と回転率を並べ、同じ向きに動くことを示す(第 5〜10 節。同時点の関係であって予測力ではない)。 |
| [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md) | テイカー側で出来高を買い・売りに分解し、HAC 帰無対照で偏りを検定。標本期間の主なニュースとの対応づけ、および 6 月 24 日決算発表という「発表時刻が事前に判っている」1 件だけの厳密な検証。 |
| [時間帯ごとのオーダーサイズ分布](mu_order_size_report.md) | 開場後・昼・閉場前・深夜の 4 つの 2 時間帯について、成行注文 1 本のサイズ分布を比較。開場後だけが明確に大きく、どの帯でも上位 1% が数量の 1/3 以上を占める。 |
| [板の深さごとの注文到着率](mu_arrival_depth_report.md) | 1 秒あたり何本の注文が mid からどれだけ離れた価格に届くかを、6 つの時間帯 × 買い売り別 × 9 つの距離帯で測る。開場後は閉場日の昼の 21 倍。最頻帯は 2–5 bp だが開場後だけ 10–25 bp へ外側にずれる。買いと売りの非対称は 6 窓を補正すると残らない。 |
| [置いた指値が約定する確率](mu_fill_rate_report.md) | 指値が最終的に約定する確率を、発注の瞬間に確定する 2 条件(同じ側の最良気配から何ティック離れているか / 同じ価格に既に何本並んでいたか)で層別する。効くのは水準よりキュー位置で、先客 1 本で 2.5 分の 1、5〜9 本で 20 分の 1 になる。トリガー注文を板に入れて 99.95% の時間クロスさせた失敗も記録。**10 日分の暫定値**。 |

### B. 板の状態は将来の値動きを教えてくれるか

このリポジトリの中心的な問いです。板から作った説明変数で、将来の価格の向きを
どこまで当てられるか。**4 本とも「有意だが費用に届かない」という結論**に着地します。

| レポート | 内容 |
|---|---|
| [MicroPrice と midprice の差 → 将来 mid の上昇確率](mu_microprice_report.md) | イベントごとに MicroPrice と mid を算出し、その差の帯 × 予測ホライズン(1〜100 イベント)で上昇確率の行列を立会日・閉場日別に作る。生の確率を 0.5 と比べてはいけない理由(同値が 4〜5 割)を示し、同値を除くと中央帯がちょうど 50% になることを確認。 |
| [OBI と OFI から見た上昇確率](mu_obi_ofi_report.md) | 板の残高の偏り(OBI)と流量の偏り(OFI)をイベントごとに算出し、1〜100 イベント先の上昇確率を行列にする。2 つが別の情報を持つことを同時分布で示し、最も有利な帯・ホライズンでも往復のスプレッド(中央値 1.245 bp)に届かないことを示す。MicroPrice の中央帯が実は情報を捨てていたことも指摘。 |
| [Book Slope と将来の log リターン(OLS / GLS)](mu_book_slope_report.md) | 板の傾きをイベントごとに算出し、1〜500 イベント先の log リターンへ回帰。関係が S 字で線形でないこと、重なる窓が t 値を最大 6.3 倍水増しすること、AR(1) の GLS がこの誤差構造には誤設定で「重ならない部分標本」が OLS を支持することを示す。 |
| [キャンセル率の傾きと将来の log リターン](mu_cancel_rate_report.md) | キャンセル率 CR と不均衡 CI を 100ms 刻みで作り、直近 1 秒に当てた直線の傾きが正のとき log リターンが正になるかを 100ms〜60 秒の 8 ホライズンで検証。16 セルすべてで有意だが、効果は片道費用の 1/8 以下で取引としては成立しない。先読みのバグを踏んで効果が半減した経緯も記録。 |
| [100ms 窓の分散は将来のリターンを説明するか](mu_var100_report.md) | 最良気配の買い数量・売り数量・OBI・OFI を 10ms 格子に載せ、100ms ごとの標本分散を 12 のホライズン(10ms〜100s)の将来 log リターンへ回帰。絶対リターンには OFI の分散が最も効く(500ms で r=+0.132)。**生の分散では何も見えず log(1+x) で初めて見える**こと、片側に紐づいた分散は符号つきリターンとも鏡像の関係を持つこと(事前の予測を外した経緯)を含む。 |

### C. 注文フローはどれだけ自分自身を引きずるか

B が「x から y を当てる」話なのに対し、ここは「x が x 自身をどれだけ引きずるか」、
つまり自己相関の話です。**予測力の話ではない**ことに注意してください。

| レポート | 内容 |
|---|---|
| [攻撃的な売買の向きの推移確率行列](mu_sign_chain_report.md) | テイカーの向きが続いた後に次がどちらに来るかの推移確率。約定単位と成行注文単位を分け、入れ替え検定でどの次数まで情報が増えるかを判定。 |
| [特徴量の符号は何イベント先まで持続するか](mu_sign_persistence_report.md) | これまでに算出した符号つき特徴量すべてについて、正のとき k イベント後も正である確率を 3 状態(負 / ちょうど 0 / 正)の推移行列にする(k=1〜5)。OBI・Book Slope・MicroPrice 乖離が符号として厳密に同一であることの証明、素朴な引き算で「ちょうど 0」が壊れる数値の罠、帰無対照が 0 に潰れ切らない理由も含む。 |
| [OBI と OFI の自己相関(200ms 格子)](mu_acf_200ms_report.md) | イベント時刻の量を 200ms の時計に載せ直し、200ms〜5 分のラグで自己相関を測る。**OBI の高い自己相関(0.2s で +0.68)はほぼ標本化の副作用**で、空の格子点の割合との相関が +0.970 であることを示す。OFI は実質的に無記憶。巡回シフトが自己相関の帰無対照にならない理由も。 |

### D. 板の中の量どうしはどう結びついているか

| レポート | 内容 |
|---|---|
| [スプレッド幅・注文量・約定量と OBI / OFI の関係](mu_spread_flow_report.md) | 最良気配が変わるたびに 5 つの量を測り、相関行列と帯別の姿で関係を示す。約定はスプレッドが狭いときに起きる(−0.24)一方、注文量とはほぼ無関係(−0.01)。 |

---

## 図

レポートに載せている図をすべてここに並べます。区分はレポートと同じ A〜D です。
各図の見出しの末尾に、その図を解説しているレポートへのリンクがあります。

### A. この銘柄はどういう市場か

**日足チャート** — 標本期間 99 日分の始値・高値・安値・終値。白背景が米国市場の立会日、薄いオレンジが休場日。解説: [日次平均建玉と出来高](mu_oi_volume_report.md)

![xyz:MU 日足チャート。標本期間 99 日分の始値・高値・安値・終値。白背景が米国市場の立会日、薄いオレンジが休場日](../../charts/xyz_MU_price_daily.png)

**建玉と出来高(ドル建て)** — 日次平均建玉と日次出来高を名目ドルで表示。解説: [日次平均建玉と出来高](mu_oi_volume_report.md)

![xyz:MU 建玉と出来高(ドル建て)。日次平均建玉と日次出来高を名目ドルで表示](../../charts/xyz_MU_oi_volume_usd.png)

**建玉と出来高(枚数)** — 同じ内容を契約枚数で表示。解説: [日次平均建玉と出来高](mu_oi_volume_report.md)

![xyz:MU 建玉と出来高(枚数)。同じ内容を契約枚数で表示](../../charts/xyz_MU_oi_volume_contracts.png)

**出来高と翌日建玉の散布図** — 立会日と休場日それぞれの OLS と GLS の当てはめ線つき。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 出来高と翌日建玉の散布図。立会日と休場日それぞれの OLS と GLS の当てはめ線つき](../../charts/xyz_MU_scatter_oi_volume.png)

**日内プロファイル(2 行 2 列)** — 立会日と休場日の日内出来高・建玉変化を縦軸共通で比較。次の 4 枚をまとめたもの。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 日内プロファイル(2 行 2 列)。立会日と休場日の日内出来高・建玉変化を縦軸共通で比較。次の 4 枚をまとめたもの](../../charts/xyz_MU_intraday_2x2.png)

**立会日の日内出来高** — 30 分ごと、68 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 立会日の日内出来高。30 分ごと、68 日の平均](../../charts/xyz_MU_intraday_volume_open.png)

**立会日の日内 建玉変化** — 30 分ごと、68 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 立会日の日内 建玉変化。30 分ごと、68 日の平均](../../charts/xyz_MU_intraday_oichange_open.png)

**休場日の日内出来高** — 30 分ごと、31 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 休場日の日内出来高。30 分ごと、31 日の平均](../../charts/xyz_MU_intraday_volume_closed.png)

**休場日の日内 建玉変化** — 30 分ごと、31 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 休場日の日内 建玉変化。30 分ごと、31 日の平均](../../charts/xyz_MU_intraday_oichange_closed.png)

**5 日窓の日次分散(99 日間)** — 直前 5 日で測った日次分散と同じ窓の回転率。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 5 日窓の日次分散(99 日間)。直前 5 日で測った日次分散と同じ窓の回転率](../../charts/xyz_MU_variance_5d_over_99d.png)

**1 日窓の日次分散(1 週間)** — その日だけで測った日次分散と回転率。土日の落差が出る。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 1 日窓の日次分散(1 週間)。その日だけで測った日次分散と回転率。土日の落差が出る](../../charts/xyz_MU_variance_1d_over_1w.png)

**1 時間窓の分散(1 日)** — その 1 時間だけで測った分散と回転率。寄付きに集中する。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 1 時間窓の分散(1 日)。その 1 時間だけで測った分散と回転率。寄付きに集中する](../../charts/xyz_MU_variance_1h_over_1d.png)

**寄付き前後 6 時間の出来高と相対スプレッド** — 1 分刻み。買い・売りの積み上げと、時間加重の相対スプレッド。解説: [回転率・出来高と翌日建玉・日内プロファイル 第 11 節](mu_turnover_intraday_report.md)

![xyz:MU 寄付き前後 6 時間の出来高と相対スプレッド。1 分刻み。買い・売りの積み上げと、時間加重の相対スプレッド](../../charts/xyz_MU_open_window_volume_spread.png)

**寄付き前後 6 時間の分散** — 算出窓 5 分、1 分あたり。中値リターンから算出。解説: [回転率・出来高と翌日建玉・日内プロファイル 第 11 節](mu_turnover_intraday_report.md)

![xyz:MU 寄付き前後 6 時間の分散。算出窓 5 分、1 分あたり。中値リターンから算出](../../charts/xyz_MU_open_window_variance.png)

**出来高の買い・売り分解** — 日次出来高の積み上げ棒(買い・売り)と、その差に対する HAC 帰無対照の帯。解説: [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md)

![xyz:MU 出来高の買い・売り分解。日次出来高の積み上げ棒(買い・売り)と、その差に対する HAC 帰無対照の帯](../../charts/xyz_MU_volume_side.png)

**決算発表日(2026-06-24)の日中** — 1 時間ごとの価格と買い・売り出来高。20:00 UTC(米国引け後)の決算発表を縦線で表示。解説: [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md)

![xyz:MU 決算発表日(2026-06-24)の日中。1 時間ごとの価格と買い・売り出来高。20:00 UTC(米国引け後)の決算発表を縦線で表示](../../charts/xyz_MU_intraday_2026-06-24.png)

**時間帯ごとのオーダーサイズ分布** — 4 つの時間帯の密度と裾の重さ(両対数)。解説: [時間帯ごとのオーダーサイズ分布](mu_order_size_report.md)

![xyz:MU 時間帯ごとのオーダーサイズ分布。4 つの時間帯の密度と裾の重さ(両対数)](../../charts/xyz_MU_order_size_dist.png)

**板の深さごとの注文到着率** — 時間帯ごとの合計到着率(買い売り別・比つき)と、深さ帯ごとの形 6 枚(共通の対数目盛)。解説: [板の深さごとの注文到着率](mu_arrival_depth_report.md)

![xyz:MU 板の深さごとの注文到着率。時間帯ごとの合計到着率(買い売り別・比つき)と、深さ帯ごとの形 6 枚(共通の対数目盛)](../../charts/xyz_MU_arrival_depth.png)

**置いた指値が約定する確率** — ティック水準 × キュー位置の周辺分布と同時分布、出来高三分位での形、出来高との比例性。解説: [置いた指値が約定する確率](mu_fill_rate_report.md)

![xyz:MU 置いた指値が約定する確率。ティック水準 × キュー位置の周辺分布と同時分布、出来高三分位での形、出来高との比例性](../../charts/xyz_MU_fill_rate.png)

### B. 板の状態と将来の値動き

**MicroPrice の確率推移行列** — 差の帯 × ホライズンの上昇確率を立会日・閉場日で並べた行列。色は無条件との差。解説: [MicroPrice と midprice の差](mu_microprice_report.md)

![xyz:MU MicroPrice の確率推移行列。差の帯 × ホライズンの上昇確率を立会日・閉場日で並べた行列。色は無条件との差](../../charts/xyz_MU_microprice_matrix.png)

**OBI / OFI の確率推移行列** — OBI と OFI それぞれの帯 × ホライズンの上昇確率、両者の同時分布、期待値動きとスプレッドの比較。解説: [OBI と OFI から見た上昇確率](mu_obi_ofi_report.md)

![xyz:MU OBI / OFI の確率推移行列。OBI と OFI それぞれの帯 × ホライズンの上昇確率、両者の同時分布、期待値動きとスプレッドの比較](../../charts/xyz_MU_obi_ofi_matrix.png)

**Book Slope と log リターン** — 帯ごとの平均 log リターン(S 字)、β のホライズン依存(OLS / GLS / 重ならない部分標本)、t 値の水増し、決定係数の比較。解説: [Book Slope と将来の log リターン](mu_book_slope_report.md)

![xyz:MU Book Slope と log リターン。帯ごとの平均 log リターン(S 字)、β のホライズン依存(OLS / GLS / 重ならない部分標本)、t 値の水増し、決定係数の比較](../../charts/xyz_MU_book_slope.png)

**キャンセル率の傾き** — CI の数列と当てた直線の実例、傾きの十分位ごとの用量反応、効果のホライズン依存と帰無対照、片道費用との比較。解説: [キャンセル率の傾きと将来の log リターン](mu_cancel_rate_report.md)

![xyz:MU キャンセル率の傾き。CI の数列と当てた直線の実例、傾きの十分位ごとの用量反応、効果のホライズン依存と帰無対照、片道費用との比較](../../charts/xyz_MU_cancel_rate.png)

**100ms 窓の分散と将来リターン** — 絶対リターンへの相関、符号つきリターンへの相関(買い売りが鏡像)、t 値の水増し、生の分散と log(1+x) の違い。解説: [100ms 窓の分散は将来のリターンを説明するか](mu_var100_report.md)

![xyz:MU 100ms 窓の分散と将来リターン。絶対リターンへの相関、符号つきリターンへの相関、t 値の水増し、生の分散と log(1+x) の違い](../../charts/xyz_MU_var100.png)

### C. 注文フローの持続性

**向きの continuation 確率** — 同じ向きが n 回続いた後にまた同じ向きが来る割合。解説: [攻撃的な売買の向きの推移確率行列](mu_sign_chain_report.md)

![xyz:MU 向きの continuation 確率。同じ向きが n 回続いた後にまた同じ向きが来る割合](../../charts/xyz_MU_sign_chain_continuation.png)

**3 次の推移確率行列** — 直前 3 本の向きごとに次が買いになる確率。解説: [攻撃的な売買の向きの推移確率行列](mu_sign_chain_report.md)

![xyz:MU 3 次の推移確率行列。直前 3 本の向きごとに次が買いになる確率](../../charts/xyz_MU_sign_chain_matrix.png)

**符号の持続性** — 6 つの特徴量 × k=1〜5 の持続性ヒートマップ、減衰曲線、帰無対照との比較、OBI の 3×3 推移行列。解説: [特徴量の符号は何イベント先まで持続するか](mu_sign_persistence_report.md)

![xyz:MU 符号の持続性。6 つの特徴量 × k=1〜5 の持続性ヒートマップ、減衰曲線、帰無対照との比較、OBI の 3×3 推移行列](../../charts/xyz_MU_sign_persistence.png)

**OBI と OFI の自己相関(200ms 格子)** — ラグ 200ms〜5 分の自己相関、空の格子点の割合との関係、OFI の符号が日によって変わること。解説: [OBI と OFI の自己相関(200ms 格子)](mu_acf_200ms_report.md)

![xyz:MU OBI と OFI の自己相関。ラグ 200ms〜5 分の自己相関、空の格子点の割合との関係、OFI の符号が日によって変わること](../../charts/xyz_MU_acf_200ms.png)

### D. 板の中の量どうしの関係

**5 つの量の相関行列** — スプレッド幅・注文量・約定量・OBI・OFI のスピアマン順位相関。解説: [スプレッド幅・注文量・約定量と OBI / OFI の関係](mu_spread_flow_report.md)

![xyz:MU 5 つの量の相関行列。スプレッド幅・注文量・約定量・OBI・OFI のスピアマン順位相関](../../charts/xyz_MU_spread_flow_corr.png)

**OBI / OFI の帯ごとの姿** — 帯ごとのスプレッド幅・注文量・約定量・約定の発生率(2 行 4 列)。解説: [スプレッド幅・注文量・約定量と OBI / OFI の関係](mu_spread_flow_report.md)

![xyz:MU OBI / OFI の帯ごとの姿。帯ごとのスプレッド幅・注文量・約定量・約定の発生率(2 行 4 列)](../../charts/xyz_MU_spread_flow_bins.png)

---

## 数値データ

リンクのあるものは版管理下にあります。parquet は容量のため版管理外で、
下表の生成スクリプトを実行すれば作り直せます。

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
| [sign_chain_xyz_MU.csv](../../data/sign_chain_xyz_MU.csv) | 次数ごとの G 統計量・帰無分布・効果量(28 行 + 見出し) | `build_sign_chain.py` |
| [spread_flow_bins_xyz_MU.csv](../../data/spread_flow_bins_xyz_MU.csv) | OBI / OFI の帯ごとの件数・スプレッド・注文量・約定量・発生率(18 行) | `build_spread_flow.py` |
| [spread_flow_corr_xyz_MU.csv](../../data/spread_flow_corr_xyz_MU.csv) | 7 変数のスピアマン順位相関行列(7 行) | `build_spread_flow.py` |
| [order_size_stats_xyz_MU.csv](../../data/order_size_stats_xyz_MU.csv) | 時間帯 × 区分ごとのオーダーサイズの要約統計(16 行) | `build_order_size.py` |
| [order_size_hist_xyz_MU.csv](../../data/order_size_hist_xyz_MU.csv) | 対数階級のヒストグラム(140 行) | `build_order_size.py` |
| [arrival_depth_xyz_MU.csv](../../data/arrival_depth_xyz_MU.csv) | 窓 × 側 × 深さ帯の到着率の平均・中央・p10・p90(144 行) | `build_arrival_depth.py` |
| arrival_depth_xyz_MU.parquet | 日 × 窓 × 側 × 深さ帯の生計数と数量(9,000 行) | `build_arrival_depth.py` |
| [book_slope_fits_xyz_MU.csv](../../data/book_slope_fits_xyz_MU.csv) | 説明変数 × 日区分 × ホライズンの OLS / HAC / GLS / 重ならない部分標本の推定(44 行) | `build_book_slope.py` |
| [acf_200ms_xyz_MU.csv](../../data/acf_200ms_xyz_MU.csv) | 変数 × 日区分 × 156 ラグの自己相関・区間・帰無対照(624 行) | `build_acf_200ms.py` |
| [acf_200ms_daily_lag1_xyz_MU.csv](../../data/acf_200ms_daily_lag1_xyz_MU.csv) | 日ごとの 1 ラグ自己相関と空の格子点の割合(98 行) | `build_acf_200ms.py` |
| [var100_ols_xyz_MU.csv](../../data/var100_ols_xyz_MU.csv) | 特徴量 × 層 × 変換 × 目的変数 × 日区分 × ホライズン × 標本の OLS 推定(2,304 行) | `build_var100.py` |
| microprice_cells_xyz_MU.parquet | 日区分 × 差の帯 × ホライズンの上昇確率・区間・帰無対照 | `build_microprice.py` |
| obi_ofi_cells_xyz_MU.parquet | 説明変数 × 日区分 × 帯 × ホライズンの上昇確率・期待値動き・区間・帰無対照 | `build_obi_ofi.py` |
| obi_ofi_joint_xyz_MU.parquet | OBI × OFI の同時分布(k=1/10/100) | `build_obi_ofi.py` |
| book_slope_bins_xyz_MU.parquet | 帯ごとの平均 log リターン(図示用) | `build_book_slope.py` |
| cancel_rate_cells_xyz_MU.parquet | 日区分 × ホライズンの平均 log リターン・区間・帰無対照 | `build_cancel_rate.py` |
| cancel_rate_bins_xyz_MU.parquet | 傾きの十分位 × ホライズンの平均 log リターン | `build_cancel_rate.py` |
| sign_persist_counts_xyz_MU.parquet | 日 × 特徴量 × k × (符号, 次の符号)の生計数 | `build_sign_persistence.py` |
| sign_persist_cells_xyz_MU.parquet | 日区分 × 特徴量 × k の集計・区間・帰無対照 | `build_sign_persistence.py` |
| sign_persist_ident_xyz_MU.parquet | 恒等式の検算結果(日ごとの不一致件数) | `build_sign_persistence.py` |

## 再現手順

上から順に実行すると、このページの図と数値がすべて再現できます。
初回のみ、リポジトリの直下で `uv sync` を実行してください。

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
uv run python scripts/build_sign_chain.py --coin xyz:MU
uv run python scripts/build_spread_flow.py --coin xyz:MU
uv run python scripts/plot_spread_flow.py --coin xyz:MU
uv run python scripts/build_order_size.py --coin xyz:MU
uv run python scripts/plot_order_size.py --coin xyz:MU
uv run python scripts/fetch_bbo.py --coin xyz:MU
uv run python scripts/build_microprice.py --coin xyz:MU
uv run python scripts/plot_microprice.py --coin xyz:MU
uv run python scripts/build_obi_ofi.py --coin xyz:MU
uv run python scripts/plot_obi_ofi.py --coin xyz:MU
uv run python scripts/build_book_slope.py --coin xyz:MU
uv run python scripts/plot_book_slope.py --coin xyz:MU
uv run python scripts/build_cancel_rate.py --coin xyz:MU
uv run python scripts/plot_cancel_rate.py --coin xyz:MU
uv run python scripts/build_sign_persistence.py --coin xyz:MU
uv run python scripts/plot_sign_persistence.py --coin xyz:MU
uv run python scripts/build_arrival_depth.py --coin xyz:MU
uv run python scripts/plot_arrival_depth.py --coin xyz:MU
uv run python scripts/build_acf_200ms.py --coin xyz:MU
uv run python scripts/plot_acf_200ms.py --coin xyz:MU
uv run python scripts/build_var100.py --coin xyz:MU
uv run python scripts/plot_var100.py --coin xyz:MU
uv run python scripts/fetch_l1.py --coin xyz:MU
uv run python scripts/build_fill_rate.py --coin xyz:MU
uv run python scripts/plot_fill_rate.py --coin xyz:MU
```

---

リポジトリ全体の目次は [../../README.md](../../README.md) にあります。
