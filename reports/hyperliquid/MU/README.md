# `xyz:MU` 分析索引

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

<p>
  <a href="https://www.micron.com/"><img src="../../../photo/micron-logo.png" alt="Micron Technology" height="76"></a>
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
目的変数の期間)を置いています。用語は [用語辞書](../../../GLOSSARY.md)、
「予測できた」の定義は [予測の定義](../../predicting_definition.md) にあります。

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
| [このウォレットはマーケットメイカーか](mu_wallet_mm_report.md) | 板を厚く出している 5 者を 13 基準(両建て・連続在席・BBO 参加・対称性・更新頻度・取消率・生存時間・補充・建玉感応・板厚寄与・カバレッジ・幅と量の安定性)で採点。**板に出す金額の大きさと MM らしさは一致しない** — 最大の出し手は mid から 73.8bp 離れ、±10bp に置くのは 2.3% だけ。 |
| [置いた指値が約定する確率](mu_fill_rate_report.md) | 指値が最終的に約定する確率を、発注の瞬間に確定する 2 条件(同じ側の最良気配から何ティック離れているか / 同じ価格に既に何本並んでいたか)で層別する。効くのは水準よりキュー位置で、先客 1 本で 2.5 分の 1、5〜9 本で 20 分の 1 になる。トリガー注文を板に入れて 99.95% の時間クロスさせた失敗も記録。**10 日分の暫定値**。 |
| [仮想成行 Q に対する板の応答と脆さ](mu_impact_fragility_report.md) | 板のラダー全体を 5 秒ごとに組み直し、仮想の成行 Q(0.5 / 5 / 50 契約)に対する sweep levels・price impact・slippage・marginal depth・marginal impact と、板の脆さ 5 種(fragility・その偏り・depth-at-risk・gap 調整)を出す。Impact ∝ Q^0.25、板は毎秒 10.5% 入れ替わる。**Impact の非対称は 5 秒先の向きに −0.070 で、マイクロプライス(+0.064)と OFI(+0.049)を上回り、しかも後ろ向き相関がほぼ 0**。ただし成行で取るには往復スプレッドに届かない。 |
| [見せかけの板を疑う 18 指標](mu_manipulation_report.md) | spoofing / layering の教科書的な形 18 通りを注文 1 本ごとに測り、口座別に集計する。**母集団のベースラインが高すぎて、ほとんど何も選り分けない** — 最良気配に届いた注文の 87% は約定せず引かれ、取消率は 98.9%。fleeting liquidity は本数で 54% だが表示時間で数えると 1.3%(41 倍差)。決定的な検証(大口を出したあと自分が反対側で売り抜けたか)では**署名が出ないどころか符号が逆**で、反対側で売った層では値段がその注文と逆へ 3.50 bp 動いていた。口座は内部連番のみで扱う。 |
| [複数の口座は同時に動くか — 同期の 8 指標](mu_sync_report.md) | 上位 60 口座(全注文の 87.7%)1,445 ペアについて、ブロック粒度の共起と秒粒度の相関を帰無との比で測る。**同時性は本物**(分層化した期待値の 1.43 倍、取消は 1.70 倍)だが、**向きは同期しない**(符号つき出し入れの超過相関は +0.0085 しかない)。さらに**同期の 9 割は「全員が同じ市場に反応している」で説明でき**、両者を除いた市場を統制すると超過は +0.137 → +0.014 へ落ちる。ただし 1,445 ペア中 **18〜19 ペアだけは残差相関が 0.5 を超える**(帰無では 0 件)。 |
| [指値注文の生存率・取消率・約定率(ハザード)](mu_hazard_report.md) | 板に置かれた指値 5.5 億本を「寿命」を持つ個体として扱い、生存率・原因別ハザード(取消/約定)・累積発生確率を出す。発注時に判る 6 条件(前に並んだ数量・自分の数量・最良からの距離・口座の累計本数・ボラティリティ・OFI)で層別。最終的に約定するのは 0.915% だけで、最も効くのは口座(81 倍)。OFI は最終約定率では効かないように見えて、0.1 秒後の約定ハザードでは 3.6 倍開く。**全 98 日**。 |
| [束の間の注文(fleeting order)は板の何割を占めるか](mu_fleeting_report.md) | 板に置かれてすぐ約定せずに取り消される指値を、8 つの閾値・側・最良からの距離・注文数量・口座で数える。2 秒以内に消えるのは数量の 66.4%、最良気配のそばに限れば 90.0%。大口(上位 1%)だけは 36.1% と半分近く、200ms 以内に消えるのは 1.7% しかない。口座ごとの FLR のばらつきは二項の帰無対照の 47 倍で、**全体 71.8% と口座の中央値 22.5% が分母の違いだけで 3 倍ずれる**ことも示す。**全 98 日**。 |
| [板の数量は何者の口座に集まっているか](mu_wallet_conc_report.md) | 1 秒ごとに板を復元し、数量を置いている口座のシェアから 14 の集中度指標を出す。板全体には 313 者が居るのに **最良気配を持つのは常に 3.17 者**で、そこにある数量は板の 0.14% しかない。touch の HHI 0.644 に対し deep は 0.081。再構成した板が壊れていても指標は整合して見えるため、**2 度にわたり板が単調に膨らんだ**経緯(繰越注文の口座欠落 / 終端が来ない注文)と、bbo との突合で気づいた顛末も記録。**全 98 日**。 |
| [ウォレットの行動クラスタ](mu_wallet_clusters_report.md) | 注文の作り方を 7 領域の分布(数量・距離・生存時間・時刻・取消・価格の置き方・最良気配)で表し、Jensen–Shannon 距離から類似度・クラスタ・2 次元の埋め込みを作る。380 者・72,010 組。**分かれるのは「どこに置くか」で、「いくらで置くか」では分かれない**(size-profile の分離幅は −0.005)。事前宣言した規則では潰れた分割しか出ず、手法を変えた経緯も記録。 |
| [メッセージの流量(quote stuffing / message activity)](mu_msg_activity_report.md) | 板を作らず、流れてくるメッセージ 13.3 億通そのものを数える。中央 111 通/秒、最繁の秒は 4,448 通。約定 1 件あたり 103 通・新規発注 47 本で、7 通に 1 通は拒否。★**「イベント間隔」はチェーンのブロック周期 67.3ms に量子化されており**(99 日を通して 67.18〜67.50ms)、間隔・分散・CV・burstiness は参加者の速さではなく「動きのあったブロックの間引き」を測っている。群れは Fano factor(1 秒窓で 172、ポアソンなら 1)と 1 時間先まで残る自己相関に出る。**全 99 日**。 |
| [最良気配近くのメイカーの混み具合と競争](mu_maker_crowd_report.md) | 最良から 10 ティック以内に居る口座だけを取り出し、混み具合と競争を 9 指標で測る。中央 10.6 者居ても **実効は 3.35 者**で、上位 1 者が数量の 50.6% を持つ。★**帯を 0 → 25 ティックに広げると人数は 8.4 倍になるのに実効は 3.2 倍止まりで、均等度(MCI)は 0.659 → 0.261 へ下がる**(奥のメイカーは競争の実体になっていない)。顔ぶれは 1 分で 6 割入れ替わり、1 価格には 1.54 者しか居ない。帯 0 の人数が[口座の集中度]の BBO wallet count と 98 日すべてで一致(相関 1.00000)。**全 98 日**。 |
| [最良気配の入れ替わり(BBO turnover)](mu_bbo_turnover_report.md) | 最良価格が保たれる時間は中央 343ms しかなく 1 秒格子では測れないので、格子を使わず区間の交差だけで ns 精度で測る。置換 1.32 回/秒、1 秒後も同じ価格である確率は 0.40。★**注文は毎秒 2.87 本入れ替わるのに数量は 0.15 回転/秒しか入れ替わらず、19 倍の開きがある**(入れ替わっているのは小さい注文で、厚みを作る注文は長く居座る)。口座が最良を保つ時間は 363ms で、1 本の注文が居る 335ms とほぼ変わらない。**全 98 日**。 |

### B. 板の状態は将来の値動きを教えてくれるか

このリポジトリの中心的な問いです。板から作った説明変数で、将来の価格の向きを
どこまで当てられるか。**5 本とも「有意だが費用に届かない」という結論**に着地します。

| レポート | 内容 |
|---|---|
| [MicroPrice と midprice の差 → 将来 mid の上昇確率](mu_microprice_report.md) | イベントごとに MicroPrice と mid を算出し、その差の帯 × 予測ホライズン(1〜100 イベント)で上昇確率の行列を立会日・閉場日別に作る。生の確率を 0.5 と比べてはいけない理由(同値が 4〜5 割)を示し、同値を除くと中央帯がちょうど 50% になることを確認。 |
| [OBI と OFI から見た上昇確率](mu_obi_ofi_report.md) | 板の残高の偏り(OBI)と流量の偏り(OFI)をイベントごとに算出し、1〜100 イベント先の上昇確率を行列にする。2 つが別の情報を持つことを同時分布で示し、最も有利な帯・ホライズンでも往復のスプレッド(中央値 1.245 bp)に届かないことを示す。MicroPrice の中央帯が実は情報を捨てていたことも指摘。 |
| [Book Slope と将来の log リターン(OLS / GLS)](mu_book_slope_report.md) | 板の傾きをイベントごとに算出し、1〜500 イベント先の log リターンへ回帰。関係が S 字で線形でないこと、重なる窓が t 値を最大 6.3 倍水増しすること、AR(1) の GLS がこの誤差構造には誤設定で「重ならない部分標本」が OLS を支持することを示す。 |
| [キャンセル率の傾きと将来の log リターン](mu_cancel_rate_report.md) | キャンセル率 CR と不均衡 CI を 100ms 刻みで作り、直近 1 秒に当てた直線の傾きが正のとき log リターンが正になるかを 100ms〜60 秒の 8 ホライズンで検証。16 セルすべてで有意だが、効果は片道費用の 1/8 以下で取引としては成立しない。先読みのバグを踏んで効果が半減した経緯も記録。 |
| [板の入れ替わり(churn)7 種と将来 log リターン](mu_churn_report.md) | 板に入った量と出ていった量を 100ms 窓で 7 通り(全体 / 買い / 売り / 最良気配以上 / それより外側 / 偏り / 本数)測り、11 の予測ホライズンへ回帰。前向きの相関が後ろ向きを下回るのは 77 セル中 77 セル。向きの予測力は $`\|r\| \le 0.0154`$ で実質ゼロ。立会日と閉場日を混ぜると相関が両方より大きくなる罠も示す。 |
| [100ms 窓の分散は将来のリターンを説明するか](mu_var100_report.md) | 最良気配の買い数量・売り数量・OBI・OFI を 10ms 格子に載せ、100ms ごとの標本分散を 12 のホライズン(10ms〜100s)の将来 log リターンへ回帰。絶対リターンには OFI の分散が最も効く(500ms で r=+0.132)。**生の分散では何も見えず log(1+x) で初めて見える**こと、片側に紐づいた分散は符号つきリターンとも鏡像の関係を持つこと(事前の予測を外した経緯)を含む。 |
| [板の弾力性 — 流動性ショックからの回復](mu_resilience_report.md) | 最良気配の数量が直前 1 秒の平均に対して 50% 以上失われた瞬間を「ショック」とし(818 万件)、D(t) = D0 + (D_shock − D0)e^{−κt} で回復を測る 12 指標。**指数模型が平均経路には当てはまらない**(戻る件と戻らない件の混合)こと、1 秒後の回復度 R が上昇確率に 20.6pp の単調な用量反応を持つこと、買い側と売り側で弾力性に差が無いことを示す。 |
| [ティック水準別 OBI と将来 log リターンの回帰](mu_obi_levels_report.md) | 最良気配から 1〜10 ティックの各水準について OBI を作り、100ms〜50 秒の 9 ホライズンで log リターンへ回帰。板は l1 の注文イベントから組み直した。279 格子すべてが Bonferroni 後も有意だが、**生の傾きは水準 1 が最大でも 1σ で測ると水準 2 が最大**で、累積の傾きの伸びは情報の増加ではなく ばらつきの縮小である(4 ティックで頭打ち)。bbo の 53 行の壊れた記録が標準偏差を支配していた事故と、その掃除も記録。 |

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

![xyz:MU 日足チャート。標本期間 99 日分の始値・高値・安値・終値。白背景が米国市場の立会日、薄いオレンジが休場日](../../../charts/xyz_MU_price_daily.png)

**建玉と出来高(ドル建て)** — 日次平均建玉と日次出来高を名目ドルで表示。解説: [日次平均建玉と出来高](mu_oi_volume_report.md)

![xyz:MU 建玉と出来高(ドル建て)。日次平均建玉と日次出来高を名目ドルで表示](../../../charts/xyz_MU_oi_volume_usd.png)

**建玉と出来高(枚数)** — 同じ内容を契約枚数で表示。解説: [日次平均建玉と出来高](mu_oi_volume_report.md)

![xyz:MU 建玉と出来高(枚数)。同じ内容を契約枚数で表示](../../../charts/xyz_MU_oi_volume_contracts.png)

**出来高と翌日建玉の散布図** — 立会日と休場日それぞれの OLS と GLS の当てはめ線つき。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 出来高と翌日建玉の散布図。立会日と休場日それぞれの OLS と GLS の当てはめ線つき](../../../charts/xyz_MU_scatter_oi_volume.png)

**日内プロファイル(2 行 2 列)** — 立会日と休場日の日内出来高・建玉変化を縦軸共通で比較。次の 4 枚をまとめたもの。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 日内プロファイル(2 行 2 列)。立会日と休場日の日内出来高・建玉変化を縦軸共通で比較。次の 4 枚をまとめたもの](../../../charts/xyz_MU_intraday_2x2.png)

**立会日の日内出来高** — 30 分ごと、68 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 立会日の日内出来高。30 分ごと、68 日の平均](../../../charts/xyz_MU_intraday_volume_open.png)

**立会日の日内 建玉変化** — 30 分ごと、68 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 立会日の日内 建玉変化。30 分ごと、68 日の平均](../../../charts/xyz_MU_intraday_oichange_open.png)

**休場日の日内出来高** — 30 分ごと、31 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 休場日の日内出来高。30 分ごと、31 日の平均](../../../charts/xyz_MU_intraday_volume_closed.png)

**休場日の日内 建玉変化** — 30 分ごと、31 日の平均。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 休場日の日内 建玉変化。30 分ごと、31 日の平均](../../../charts/xyz_MU_intraday_oichange_closed.png)

**5 日窓の日次分散(99 日間)** — 直前 5 日で測った日次分散と同じ窓の回転率。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 5 日窓の日次分散(99 日間)。直前 5 日で測った日次分散と同じ窓の回転率](../../../charts/xyz_MU_variance_5d_over_99d.png)

**1 日窓の日次分散(1 週間)** — その日だけで測った日次分散と回転率。土日の落差が出る。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 1 日窓の日次分散(1 週間)。その日だけで測った日次分散と回転率。土日の落差が出る](../../../charts/xyz_MU_variance_1d_over_1w.png)

**1 時間窓の分散(1 日)** — その 1 時間だけで測った分散と回転率。寄付きに集中する。解説: [回転率・出来高と翌日建玉・日内プロファイル](mu_turnover_intraday_report.md)

![xyz:MU 1 時間窓の分散(1 日)。その 1 時間だけで測った分散と回転率。寄付きに集中する](../../../charts/xyz_MU_variance_1h_over_1d.png)

**寄付き前後 6 時間の出来高と相対スプレッド** — 1 分刻み。買い・売りの積み上げと、時間加重の相対スプレッド。解説: [回転率・出来高と翌日建玉・日内プロファイル 第 11 節](mu_turnover_intraday_report.md)

![xyz:MU 寄付き前後 6 時間の出来高と相対スプレッド。1 分刻み。買い・売りの積み上げと、時間加重の相対スプレッド](../../../charts/xyz_MU_open_window_volume_spread.png)

**寄付き前後 6 時間の分散** — 算出窓 5 分、1 分あたり。中値リターンから算出。解説: [回転率・出来高と翌日建玉・日内プロファイル 第 11 節](mu_turnover_intraday_report.md)

![xyz:MU 寄付き前後 6 時間の分散。算出窓 5 分、1 分あたり。中値リターンから算出](../../../charts/xyz_MU_open_window_variance.png)

**出来高の買い・売り分解** — 日次出来高の積み上げ棒(買い・売り)と、その差に対する HAC 帰無対照の帯。解説: [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md)

![xyz:MU 出来高の買い・売り分解。日次出来高の積み上げ棒(買い・売り)と、その差に対する HAC 帰無対照の帯](../../../charts/xyz_MU_volume_side.png)

**決算発表日(2026-06-24)の日中** — 1 時間ごとの価格と買い・売り出来高。20:00 UTC(米国引け後)の決算発表を縦線で表示。解説: [出来高の買い・売り内訳とニュース](mu_volume_side_news_report.md)

![xyz:MU 決算発表日(2026-06-24)の日中。1 時間ごとの価格と買い・売り出来高。20:00 UTC(米国引け後)の決算発表を縦線で表示](../../../charts/xyz_MU_intraday_2026-06-24.png)

**時間帯ごとのオーダーサイズ分布** — 4 つの時間帯の密度と裾の重さ(両対数)。解説: [時間帯ごとのオーダーサイズ分布](mu_order_size_report.md)

![xyz:MU 時間帯ごとのオーダーサイズ分布。4 つの時間帯の密度と裾の重さ(両対数)](../../../charts/xyz_MU_order_size_dist.png)

**板の深さごとの注文到着率** — 時間帯ごとの合計到着率(買い売り別・比つき)と、深さ帯ごとの形 6 枚(共通の対数目盛)。解説: [板の深さごとの注文到着率](mu_arrival_depth_report.md)

![xyz:MU 板の深さごとの注文到着率。時間帯ごとの合計到着率(買い売り別・比つき)と、深さ帯ごとの形 6 枚(共通の対数目盛)](../../../charts/xyz_MU_arrival_depth.png)

**置いた指値が約定する確率** — ティック水準 × キュー位置の周辺分布と同時分布、出来高三分位での形、出来高との比例性。解説: [置いた指値が約定する確率](mu_fill_rate_report.md)

![xyz:MU 置いた指値が約定する確率。ティック水準 × キュー位置の周辺分布と同時分布、出来高三分位での形、出来高との比例性](../../../charts/xyz_MU_fill_rate.png)

**束の間の注文(fleeting order)** — 日次の本数と数量、閾値ごとの FLR、側別・距離帯別・数量帯別、口座ごとの分布と二項の帰無対照、口座の集中。解説: [束の間の注文は板の何割を占めるか](mu_fleeting_report.md)

![xyz:MU 束の間の注文。日次の本数と数量、閾値ごとの fleeting-liquidity ratio、bid/ask 別、最良気配からの距離帯別、注文数量帯別、口座ごとの分布と帰無対照、口座の集中](../../../charts/xyz_MU_fleeting.png)

**口座の集中度(wallet concentration)** — 口座数、最良気配を持つ口座数、HHI と実効口座数、ジニ係数、上位 k のシェア、集中の偏り、touch と deep、日内の推移。解説: [板の数量は何者の口座に集まっているか](mu_wallet_conc_report.md)

![xyz:MU 口座の集中度。板に数量を置いている口座数、最良気配を持つ口座数、wallet HHI と実効口座数、Gini 係数、上位 k のシェア、concentration imbalance、touch と deep の HHI、日内の推移](../../../charts/xyz_MU_wallet_conc.png)

**メッセージの流量(quote stuffing / message activity)** — 秒あたりの通数、種別の内訳、1 ブロックの詰まり方、ブロック間隔の分布、burstiness と memory の平面、窓長ごとの Fano factor、activity entropy、自己相関。解説: [メッセージの流量](mu_msg_activity_report.md)

![xyz:MU メッセージの流量。秒あたりの通数、NEW/REMOVE/UPDATE/REJECTED の内訳、1 ブロックのメッセージ数、ブロック間隔の分布とチェーンの周期、burstiness と memory、窓長ごとの Fano factor、activity entropy、activity autocorrelation](../../../charts/xyz_MU_msg_activity.png)

**メイカーの混み具合と競争(maker crowd / competition)** — 近傍のメイカー数、HHI と実効メイカー数、競争指数、帯を広げたときの変わり方、入替・参入・退出、quote overlap と clustering、立会日と閉場日。解説: [最良気配近くのメイカーの混み具合と競争](mu_maker_crowd_report.md)

![xyz:MU メイカーの混み具合と競争。最良から 10 ティック以内のメイカー数、maker concentration と effective maker count、maker competition index、帯を広げたときの変化、turnover / entry / exit、quote overlap と maker clustering、立会日と閉場日の比較](../../../charts/xyz_MU_maker_crowd.png)

**最良気配の入れ替わり(BBO turnover)** — 置換回数、BBO lifetime の分布、order / wallet / size の 3 つの回転率、best-price と touch の持続、所有時間の分布、立会日と閉場日、被覆率の検算。解説: [最良気配の入れ替わり](mu_bbo_turnover_report.md)

![xyz:MU 最良気配の入れ替わり。BBO replacement count、BBO lifetime の分布、order / wallet / size turnover、best-price persistence と touch persistence、touch ownership duration の分布、3 つの時間の日次推移、立会日と閉場日、再構成が最良を覆えているかの検算](../../../charts/xyz_MU_bbo_turnover.png)

**指値注文の生存率** — 6 条件それぞれの層別生存率 S(τ)。横軸は経過時間(対数)。解説: [指値注文の生存率・取消率・約定率](mu_hazard_report.md)

![xyz:MU 指値注文の生存率 S(τ)。置かれてからの経過時間 τ に対して、まだ板に残っている割合を 6 条件で層別](../../../charts/xyz_MU_hazard_survival.png)

**取消ハザード** — τ まで生きた注文が次の瞬間に取り消される率(毎秒)。解説: [同上](mu_hazard_report.md)

![xyz:MU 指値注文の取消ハザード。τ まで生きた注文が次の瞬間に取り消される率(毎秒)](../../../charts/xyz_MU_hazard_cancel.png)

**約定ハザード** — τ まで生きた注文が次の瞬間に約定する率(毎秒)。解説: [同上](mu_hazard_report.md)

![xyz:MU 指値注文の約定ハザード。τ まで生きた注文が次の瞬間に約定する率(毎秒)](../../../charts/xyz_MU_hazard_fill.png)

**ハザードの検算** — 競合リスクの内訳、無作為 8 層の帰無対照、距離を揃えた対照。解説: [同上](mu_hazard_report.md)

![xyz:MU 指値注文のハザードの検算。競合リスクの内訳、無作為に 8 層へ振った帰無対照、最良から 10 ティック以内に絞った対照](../../../charts/xyz_MU_hazard_controls.png)

**仮想成行 Q に対する板の応答** — Impact(Q)・Slippage(Q)・削る価格水準の数・marginal 2 種・買い売りの非対称。解説: [仮想成行 Q に対する板の応答と脆さ](mu_impact_fragility_report.md)

![xyz:MU 仮想成行 Q に対する板の応答。Impact(Q)、Slippage(Q)、削る価格水準の数、marginal impact、marginal depth、買いと売りの非対称](../../../charts/xyz_MU_impact_curve.png)

**板の脆さ** — 1 秒あたりの入れ替わり率・その偏り・depth-at-risk・gap 調整後の上乗せ。解説: [同上](mu_impact_fragility_report.md)

![xyz:MU 板の脆さ。1 秒あたりの入れ替わり率、その偏り、depth-at-risk、gap 調整後の上乗せ、深さの分布、脆さとスプレッドの関係](../../../charts/xyz_MU_impact_fragility.png)

**板の応答指標はシグナルになるか** — 向き・大きさ・実現ボラティリティへの前向き相関と、後ろ向き・帰無対照・偏相関の比較。解説: [同上](mu_impact_fragility_report.md)

![xyz:MU 板の応答指標のシグナルとしての強さ。向き・大きさ・実現ボラティリティに対する前向き相関の行列と、後ろ向き・帰無対照・偏相関の比較](../../../charts/xyz_MU_impact_signal.png)

**ウォレットの MM らしさ 13 基準の採点表** — 5 者 × 13 基準の採点表、位置取り(距離と最良気配率)、最も MM らしい 1 者の日ごとの在席。解説: [このウォレットはマーケットメイカーか](mu_wallet_mm_report.md)

![xyz:MU ウォレットの MM らしさ。5 者 × 13 基準の採点表、位置取り、日ごとの在席](../../../charts/xyz_MU_wallet_mm.png)

**ウォレットの行動クラスタ** — 行動の埋め込み(MDS)、どの行動領域で分かれているか、クラスタごとの生存時間と距離の分布。解説: [ウォレットの行動クラスタ](mu_wallet_clusters_report.md)

![xyz:MU ウォレットの行動クラスタ。埋め込み、領域ごとの分離、クラスタごとの生存時間と距離の分布](../../../charts/xyz_MU_wallet_clusters.png)

**見せかけの板 — 母集団のベースライン** — fleeting liquidity・fake depth persistence・cancel-before-touch・注文の集中。解説: [見せかけの板を疑う 18 指標](mu_manipulation_report.md)

![xyz:MU 見せかけの板を疑う指標の母集団ベースライン。fleeting liquidity、fake depth persistence、cancel-before-touch、大口の閾値、注文の集中、発注前後の mid の動き](../../../charts/xyz_MU_manip_base.png)

**見せかけの板 — 口座別の分布と 2 つの合成スコア**。解説: [同上](mu_manipulation_report.md)

![xyz:MU 見せかけの板を疑う指標の口座別分布。12 指標の累積分布と、spoof score / layering score の散布図](../../../charts/xyz_MU_manip_wallets.png)

**見せかけの板 — 検証**。大口を出したあと自分が何をしたか。解説: [同上](mu_manipulation_report.md)

![xyz:MU 見せかけの板の検証。スコア十分位ごとの押した分と並べ替え検定、大口を出したあと自分が何をしたか、スコアの分布と活動量との関係](../../../charts/xyz_MU_manip_check.png)

**同期 — ブロック粒度** 同じブロックに一緒に動くか。期待値は分ごとに層化。解説: [複数の口座は同時に動くか](mu_sync_report.md)

![xyz:MU ブロック粒度の同時性。発注と取消の共起を分層化した期待値で割った比の分布、比の行列、活動量との関係](../../../charts/xyz_MU_sync_block.png)

**同期 — 秒粒度の相関** 観測と帰無(±300 秒ずらし)。解説: [同上](mu_sync_report.md)

![xyz:MU 秒粒度の相関。発注本数・取消本数・符号つき出し入れ・発注価格の距離について、観測と帰無の分布](../../../charts/xyz_MU_sync_corr.png)

**同期 — 共通成分と herding**。解説: [同上](mu_sync_report.md)

![xyz:MU 共通成分と herding。自分を除いた市場合計への決定係数、相関行列の固有値、herding score の分布と規模との関係](../../../charts/xyz_MU_sync_common.png)

### B. 板の状態と将来の値動き

**MicroPrice の確率推移行列** — 差の帯 × ホライズンの上昇確率を立会日・閉場日で並べた行列。色は無条件との差。解説: [MicroPrice と midprice の差](mu_microprice_report.md)

![xyz:MU MicroPrice の確率推移行列。差の帯 × ホライズンの上昇確率を立会日・閉場日で並べた行列。色は無条件との差](../../../charts/xyz_MU_microprice_matrix.png)

**OBI / OFI の確率推移行列** — OBI と OFI それぞれの帯 × ホライズンの上昇確率、両者の同時分布、期待値動きとスプレッドの比較。解説: [OBI と OFI から見た上昇確率](mu_obi_ofi_report.md)

![xyz:MU OBI / OFI の確率推移行列。OBI と OFI それぞれの帯 × ホライズンの上昇確率、両者の同時分布、期待値動きとスプレッドの比較](../../../charts/xyz_MU_obi_ofi_matrix.png)

**Book Slope と log リターン** — 帯ごとの平均 log リターン(S 字)、β のホライズン依存(OLS / GLS / 重ならない部分標本)、t 値の水増し、決定係数の比較。解説: [Book Slope と将来の log リターン](mu_book_slope_report.md)

![xyz:MU Book Slope と log リターン。帯ごとの平均 log リターン(S 字)、β のホライズン依存(OLS / GLS / 重ならない部分標本)、t 値の水増し、決定係数の比較](../../../charts/xyz_MU_book_slope.png)

**キャンセル率の傾き** — CI の数列と当てた直線の実例、傾きの十分位ごとの用量反応、効果のホライズン依存と帰無対照、片道費用との比較。解説: [キャンセル率の傾きと将来の log リターン](mu_cancel_rate_report.md)

![xyz:MU キャンセル率の傾き。CI の数列と当てた直線の実例、傾きの十分位ごとの用量反応、効果のホライズン依存と帰無対照、片道費用との比較](../../../charts/xyz_MU_cancel_rate.png)

**100ms 窓の分散と将来リターン** — 絶対リターンへの相関、符号つきリターンへの相関(買い売りが鏡像)、t 値の水増し、生の分散と log(1+x) の違い。解説: [100ms 窓の分散は将来のリターンを説明するか](mu_var100_report.md)

![xyz:MU 100ms 窓の分散と将来リターン。絶対リターンへの相関、符号つきリターンへの相関、t 値の水増し、生の分散と log(1+x) の違い](../../../charts/xyz_MU_var100.png)

**ティック水準別 OBI の回帰係数** — 水準ごとの傾き、90 格子のヒートマップ(水準ごと / 累積)、1σ あたりの効果と水準の占有率、用量反応、板の再構成の検算。解説: [ティック水準別 OBI と将来 log リターンの回帰](mu_obi_levels_report.md)

![xyz:MU ティック水準別 OBI の回帰係数。水準ごとの傾き、90 格子のヒートマップ、1σ あたりの効果と占有率、用量反応、再構成の検算](../../../charts/xyz_MU_obi_levels.png)

**板の入れ替わり(churn)と将来 log リターン** — 特徴量ごとの相関のホライズン依存 7 枚(前向き / 後ろ向き / 向きの予測)と、符号つきリターンとの相関の一覧 2 枚。解説: [板の入れ替わり(churn)7 種と将来 log リターン](mu_churn_report.md)

![xyz:MU churn 7 種と将来 log リターンの OLS。特徴量ごとの相関のホライズン依存と、符号つきリターンとの相関の一覧](../../../charts/xyz_MU_churn_ols.png)

**板の弾力性 — 回復曲線と指標の分布** — 買い売り別の平均回復経路と指数当てはめ、12 指標の分位、日ごとの κ。解説: [板の弾力性](mu_resilience_report.md)

![xyz:MU 板の弾力性の回復曲線。買い売り別の平均回復経路と指数当てはめ、12 指標の分位、日ごとの κ](../../../charts/xyz_MU_resilience_curve.png)

**弾力性の指標 × ホライズンの回帰** — 絶対リターンと符号つきリターンへの相関、生と log(1+x) の違い。解説: [板の弾力性](mu_resilience_report.md)

![xyz:MU 弾力性の指標と将来リターンの回帰。絶対リターンと符号つきリターンへの相関、生と log(1+x) の違い](../../../charts/xyz_MU_resilience_ols.png)

**弾力性の帯ごとの上昇確率** — 4 指標の帯 × ホライズンの P(上昇|動いた) の無条件値からの差。解説: [板の弾力性](mu_resilience_report.md)

![xyz:MU 弾力性の帯ごとの上昇確率。4 指標の帯 × ホライズンの無条件値からの差](../../../charts/xyz_MU_resilience_matrix.png)

### C. 注文フローの持続性

**向きの continuation 確率** — 同じ向きが n 回続いた後にまた同じ向きが来る割合。解説: [攻撃的な売買の向きの推移確率行列](mu_sign_chain_report.md)

![xyz:MU 向きの continuation 確率。同じ向きが n 回続いた後にまた同じ向きが来る割合](../../../charts/xyz_MU_sign_chain_continuation.png)

**3 次の推移確率行列** — 直前 3 本の向きごとに次が買いになる確率。解説: [攻撃的な売買の向きの推移確率行列](mu_sign_chain_report.md)

![xyz:MU 3 次の推移確率行列。直前 3 本の向きごとに次が買いになる確率](../../../charts/xyz_MU_sign_chain_matrix.png)

**符号の持続性** — 6 つの特徴量 × k=1〜5 の持続性ヒートマップ、減衰曲線、帰無対照との比較、OBI の 3×3 推移行列。解説: [特徴量の符号は何イベント先まで持続するか](mu_sign_persistence_report.md)

![xyz:MU 符号の持続性。6 つの特徴量 × k=1〜5 の持続性ヒートマップ、減衰曲線、帰無対照との比較、OBI の 3×3 推移行列](../../../charts/xyz_MU_sign_persistence.png)

**OBI と OFI の自己相関(200ms 格子)** — ラグ 200ms〜5 分の自己相関、空の格子点の割合との関係、OFI の符号が日によって変わること。解説: [OBI と OFI の自己相関(200ms 格子)](mu_acf_200ms_report.md)

![xyz:MU OBI と OFI の自己相関。ラグ 200ms〜5 分の自己相関、空の格子点の割合との関係、OFI の符号が日によって変わること](../../../charts/xyz_MU_acf_200ms.png)

### D. 板の中の量どうしの関係

**5 つの量の相関行列** — スプレッド幅・注文量・約定量・OBI・OFI のスピアマン順位相関。解説: [スプレッド幅・注文量・約定量と OBI / OFI の関係](mu_spread_flow_report.md)

![xyz:MU 5 つの量の相関行列。スプレッド幅・注文量・約定量・OBI・OFI のスピアマン順位相関](../../../charts/xyz_MU_spread_flow_corr.png)

**OBI / OFI の帯ごとの姿** — 帯ごとのスプレッド幅・注文量・約定量・約定の発生率(2 行 4 列)。解説: [スプレッド幅・注文量・約定量と OBI / OFI の関係](mu_spread_flow_report.md)

![xyz:MU OBI / OFI の帯ごとの姿。帯ごとのスプレッド幅・注文量・約定量・約定の発生率(2 行 4 列)](../../../charts/xyz_MU_spread_flow_bins.png)

---

## 数値データ

リンクのあるものは版管理下にあります。parquet は容量のため版管理外で、
下表の生成スクリプトを実行すれば作り直せます。

| ファイル | 内容 | 生成スクリプト |
|---|---|---|
| [daily_oi_volume_xyz_MU.csv](../../../data/daily_oi_volume_xyz_MU.csv) | 日次の建玉・出来高・取引数・参加者数(99 行) | `build_oi_volume.py` |
| [daily_ohlc_xyz_MU.csv](../../../data/daily_ohlc_xyz_MU.csv) | 日次 4 本値(99 行) | `plot_price.py` |
| [daily_turnover_xyz_MU.csv](../../../data/daily_turnover_xyz_MU.csv) | 日次回転率(99 行) | `regress_oi_volume.py` |
| [intraday_profile_xyz_MU.csv](../../../data/intraday_profile_xyz_MU.csv) | 30 分ごとの日内プロファイル(96 行) | `plot_intraday.py` |
| [daily_volume_side_xyz_MU.csv](../../../data/daily_volume_side_xyz_MU.csv) | 日次の買い・売り出来高、買い比率、帰無対照の z 値(99 行) | `build_volume_side.py` |
| [variance_daily_xyz_MU.csv](../../../data/variance_daily_xyz_MU.csv) | 日次の実現分散(1 日窓・5 日窓)と回転率(99 行) | `build_variance.py` |
| [variance_hourly_xyz_MU.csv](../../../data/variance_hourly_xyz_MU.csv) | 1 時間ごとの実現分散と回転率(2,376 行) | `build_variance.py` |
| [open_window_xyz_MU.csv](../../../data/open_window_xyz_MU.csv) | 寄付き前後 6 時間の 1 分ごとの出来高・スプレッド・分散(360 行) | `build_open_window.py` |
| [sign_chain_xyz_MU.csv](../../../data/sign_chain_xyz_MU.csv) | 次数ごとの G 統計量・帰無分布・効果量(28 行 + 見出し) | `build_sign_chain.py` |
| [spread_flow_bins_xyz_MU.csv](../../../data/spread_flow_bins_xyz_MU.csv) | OBI / OFI の帯ごとの件数・スプレッド・注文量・約定量・発生率(18 行) | `build_spread_flow.py` |
| [spread_flow_corr_xyz_MU.csv](../../../data/spread_flow_corr_xyz_MU.csv) | 7 変数のスピアマン順位相関行列(7 行) | `build_spread_flow.py` |
| [order_size_stats_xyz_MU.csv](../../../data/order_size_stats_xyz_MU.csv) | 時間帯 × 区分ごとのオーダーサイズの要約統計(16 行) | `build_order_size.py` |
| [order_size_hist_xyz_MU.csv](../../../data/order_size_hist_xyz_MU.csv) | 対数階級のヒストグラム(140 行) | `build_order_size.py` |
| [arrival_depth_xyz_MU.csv](../../../data/arrival_depth_xyz_MU.csv) | 窓 × 側 × 深さ帯の到着率の平均・中央・p10・p90(144 行) | `build_arrival_depth.py` |
| arrival_depth_xyz_MU.parquet | 日 × 窓 × 側 × 深さ帯の生計数と数量(9,000 行) | `build_arrival_depth.py` |
| [churn_ols_xyz_MU.csv](../../../data/churn_ols_xyz_MU.csv) | 特徴量 × 日区分 × ホライズン × 標本の OLS 推定(6,864 行) | `build_churn.py` |
| churn_xyz_MU/dt=*.parquet | 100ms 窓 × 7 特徴量(98 日 × 864,000 窓) | `build_churn.py` |
| [book_slope_fits_xyz_MU.csv](../../../data/book_slope_fits_xyz_MU.csv) | 説明変数 × 日区分 × ホライズンの OLS / HAC / GLS / 重ならない部分標本の推定(44 行) | `build_book_slope.py` |
| [acf_200ms_xyz_MU.csv](../../../data/acf_200ms_xyz_MU.csv) | 変数 × 日区分 × 156 ラグの自己相関・区間・帰無対照(624 行) | `build_acf_200ms.py` |
| [acf_200ms_daily_lag1_xyz_MU.csv](../../../data/acf_200ms_daily_lag1_xyz_MU.csv) | 日ごとの 1 ラグ自己相関と空の格子点の割合(98 行) | `build_acf_200ms.py` |
| [var100_ols_xyz_MU.csv](../../../data/var100_ols_xyz_MU.csv) | 特徴量 × 層 × 変換 × 目的変数 × 日区分 × ホライズン × 標本の OLS 推定(2,304 行) | `build_var100.py` |
| [resil_ols_xyz_MU.csv](../../../data/resil_ols_xyz_MU.csv) | 弾力性の指標 × 側 × 日区分 × 変換 × 目的変数 × ホライズンの OLS(2,880 行) | `analyze_resilience.py` |
| [resil_trans_xyz_MU.csv](../../../data/resil_trans_xyz_MU.csv) | 弾力性の帯 × ホライズンの上昇確率と日単位ブートストラップの区間 | `analyze_resilience.py` |
| [resil_daily_xyz_MU.csv](../../../data/resil_daily_xyz_MU.csv) | 日ごとの κ_bid / κ_ask / 非対称(98 行) | `analyze_resilience.py` |
| [wallet_mm_xyz_MU.csv](../../../data/wallet_mm_xyz_MU.csv) | ウォレット × 13 の MM 基準(日ごとの中央値) | `build_wallet_mm.py` |
| [wallet_mm_daily_xyz_MU.csv](../../../data/wallet_mm_daily_xyz_MU.csv) | ウォレット × 日 の内訳 | `build_wallet_mm.py` |
| microprice_cells_xyz_MU.parquet | 日区分 × 差の帯 × ホライズンの上昇確率・区間・帰無対照 | `build_microprice.py` |
| obi_ofi_cells_xyz_MU.parquet | 説明変数 × 日区分 × 帯 × ホライズンの上昇確率・期待値動き・区間・帰無対照 | `build_obi_ofi.py` |
| obi_ofi_joint_xyz_MU.parquet | OBI × OFI の同時分布(k=1/10/100) | `build_obi_ofi.py` |
| book_slope_bins_xyz_MU.parquet | 帯ごとの平均 log リターン(図示用) | `build_book_slope.py` |
| cancel_rate_cells_xyz_MU.parquet | 日区分 × ホライズンの平均 log リターン・区間・帰無対照 | `build_cancel_rate.py` |
| cancel_rate_bins_xyz_MU.parquet | 傾きの十分位 × ホライズンの平均 log リターン | `build_cancel_rate.py` |
| obi_levels_fit_xyz_MU.csv / .parquet | 日区分 × 定義 × 水準 × ホライズンの傾き・区間・σx・相関(1,674 行) | `fit_obi_levels.py` |
| obi_levels_dose_xyz_MU.parquet | 水準 × ホライズン × OBI の帯ごとの平均 log リターン(8,370 行) | `fit_obi_levels.py` |
| obi_levels_days/xyz_MU/ | 日ごとの回帰の累積和(ブロックブートストラップの素) | `build_obi_levels.py` |
| obi_levels_meta/xyz_MU/ | 日ごとの検算(bbo との一致率・水準の占有率・孤児) | `build_obi_levels.py` |
| fleeting_cells_xyz_MU.parquet | 日 × 閾値 × 側 × 距離 × 数量帯の計数 | `build_fleeting.py` |
| fleeting_daily_xyz_MU.parquet | 日 × 閾値の合計と大口の内訳 | `build_fleeting.py` |
| fleeting_wallet_xyz_MU.parquet | 口座別の本数・数量と束の間の内訳(14,125 者) | `build_fleeting.py` |
| [fleeting_meta_xyz_MU.csv](../../../data/fleeting_meta_xyz_MU.csv) | 日ごとの検算と大口の閾値(98 行) | `build_fleeting.py` |
| [wallet_conc_daily_xyz_MU.csv](../../../data/wallet_conc_daily_xyz_MU.csv) | 日 × 14 指標の平均と 5 分位、検算(98 行 × 112 列) | `build_wallet_conc.py` |
| wallet_conc_hourly_xyz_MU.parquet | 日 × 時 × 指標の平均(2,352 行) | `build_wallet_conc.py` |
| sign_persist_counts_xyz_MU.parquet | 日 × 特徴量 × k × (符号, 次の符号)の生計数 | `build_sign_persistence.py` |
| sign_persist_cells_xyz_MU.parquet | 日区分 × 特徴量 × k の集計・区間・帰無対照 | `build_sign_persistence.py` |
| sign_persist_ident_xyz_MU.parquet | 恒等式の検算結果(日ごとの不一致件数) | `build_sign_persistence.py` |
| [msg_activity_daily_xyz_MU.csv](../../../data/msg_activity_daily_xyz_MU.csv) | 日 × 14 指標と検算(99 行 × 46 列) | `build_msg_activity.py` |
| msg_activity_curves_xyz_MU.parquet | 日 × 曲線(Fano / ACF / 間隔分布 / 時刻別) | `build_msg_activity.py` |
| msg_activity_burst_xyz_MU.parquet | 日ごとの最繁 10 秒とその口座集中(990 行) | `build_msg_activity.py` |
| [maker_crowd_daily_xyz_MU.csv](../../../data/maker_crowd_daily_xyz_MU.csv) | 日 × 9 指標の平均と 5 分位、検算(98 行) | `build_maker_crowd.py` |
| maker_crowd_curves_xyz_MU.parquet | 日 × 帯 K(0/5/10/25)と時間差 Δ の曲線 | `build_maker_crowd.py` |
| [bbo_turnover_daily_xyz_MU.csv](../../../data/bbo_turnover_daily_xyz_MU.csv) | 日 × 9 指標と検算(98 行) | `build_bbo_turnover.py` |
| bbo_turnover_curves_xyz_MU.parquet | 日 × 持続曲線と寿命 / 所有時間の分布 | `build_bbo_turnover.py` |

| [manip_daily_xyz_MU.csv](../../../data/manip_daily_xyz_MU.csv) | 見せかけの板の指標の市場全体ベースライン(98 行) | `build_manip.py` |
| [manip_scores_xyz_MU.csv](../../../data/manip_scores_xyz_MU.csv) | 口座ごとの 18 指標と spoof / layering スコア(729 行) | `plot_manip.py` |
| [manip_meta_xyz_MU.csv](../../../data/manip_meta_xyz_MU.csv) | 見せかけの板の日ごとの検算(98 行) | `build_manip.py` |
| manip_wallet_xyz_MU.parquet | 口座 × 日 の集計(69,448 行、版管理外) | `build_manip.py` |
| manip_episodes_xyz_MU.parquet | 最良の近くに出した大口 1 本ごと(307,599 行、版管理外) | `build_manip.py` |

| [sync_pairs_xyz_MU.csv](../../../data/sync_pairs_xyz_MU.csv) | 口座ペアごとの共起比・相関・偏相関と帰無(1,445 行) | `plot_sync.py` |
| [sync_wallet_xyz_MU.csv](../../../data/sync_wallet_xyz_MU.csv) | 口座ごとの共通成分と herding score(60 行) | `plot_sync.py` |
| [sync_meta_xyz_MU.csv](../../../data/sync_meta_xyz_MU.csv) | 同期の日ごとの検算(98 行) | `build_sync.py` |
| sync_raw_xyz_MU.npz | 日 × 口座 × 口座 の十分統計(版管理外) | `build_sync.py` |
| [wallet_clusters_xyz_MU.csv](../../../data/wallet_clusters_xyz_MU.csv) | ウォレット × クラスタ ID・埋め込み座標・平均類似度(380 行) | `build_wallet_clusters.py` |

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
uv run python scripts/build_churn.py --coin xyz:MU
uv run python scripts/plot_churn.py --coin xyz:MU --day 立会日
uv run python scripts/build_acf_200ms.py --coin xyz:MU
uv run python scripts/plot_acf_200ms.py --coin xyz:MU
uv run python scripts/build_var100.py --coin xyz:MU
uv run python scripts/plot_var100.py --coin xyz:MU
uv run python scripts/fetch_l1.py --coin xyz:MU
uv run python scripts/build_fill_rate.py --coin xyz:MU
uv run python scripts/plot_fill_rate.py --coin xyz:MU
uv run python scripts/build_obi_levels.py --coin xyz:MU
uv run python scripts/fit_obi_levels.py --coin xyz:MU
uv run python scripts/plot_obi_levels.py --coin xyz:MU
uv run python scripts/fetch_l1_extra.py --coin xyz:MU
uv run python scripts/build_hazard.py --coin xyz:MU
uv run python scripts/plot_hazard.py --coin xyz:MU
uv run python scripts/build_fleeting.py --coin xyz:MU
uv run python scripts/plot_fleeting.py --coin xyz:MU
uv run python scripts/build_wallet_conc.py --coin xyz:MU
uv run python scripts/plot_wallet_conc.py --coin xyz:MU
uv run python scripts/build_msg_activity.py --coin xyz:MU
uv run python scripts/plot_msg_activity.py --coin xyz:MU
uv run python scripts/build_maker_crowd.py --coin xyz:MU
uv run python scripts/plot_maker_crowd.py --coin xyz:MU
uv run python scripts/build_bbo_turnover.py --coin xyz:MU
uv run python scripts/plot_bbo_turnover.py --coin xyz:MU
uv run python scripts/build_impact.py --coin xyz:MU
uv run python scripts/build_impact_signal.py --coin xyz:MU
uv run python scripts/plot_impact.py --coin xyz:MU
uv run python scripts/build_manip.py --coin xyz:MU
uv run python scripts/plot_manip.py --coin xyz:MU
uv run python scripts/build_sync.py --coin xyz:MU
uv run python scripts/plot_sync.py --coin xyz:MU
```

---

リポジトリ全体の目次は [../../../README.md](../../../README.md) にあります。
