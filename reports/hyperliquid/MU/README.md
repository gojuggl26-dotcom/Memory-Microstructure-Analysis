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
| [口座ごとの約定のされ方 11 指標](mu_wallet_fill_report.md) | 板に指値を置いた口座から見て、置いた注文がどれだけ約定に至ったかを 11 通りで測る(fill rate / fill probability / fill-to-order / fill-to-cancel / maker execution share / volume executed÷quoted / average time to fill / partial-fill frequency / fill size / queue-position-at-fill / adverse selection after fill)。**うち 4 つは分子と分母の取り方が違うだけの近い量**で、口座間の順位相関は +0.95 以上。**約定の 87.6% は「前に誰もいない」状態で起きる**(キュー位置は効かない)。**メイカーは約定 10 秒後に中央値 +1.36bp 不利へ動き、時刻をずらす帰無対照では +0.009bp と消える**。ただし**大口ほど逆選択が小さい**(上位 20 者 +0.52bp 対 その他 +1.41bp、ρ=−0.46)。約定量の数え方で 3 割変わる落とし穴と node_fills による 99.0% の照合つき。**全 98 日**。 |
| [置いた指値が約定する確率](mu_fill_rate_report.md) | 指値が最終的に約定する確率を、発注の瞬間に確定する 2 条件(同じ側の最良気配から何ティック離れているか / 同じ価格に既に何本並んでいたか)で層別する。効くのは水準よりキュー位置で、先客 1 本で 2.5 分の 1、5〜9 本で 20 分の 1 になる。トリガー注文を板に入れて 99.95% の時間クロスさせた失敗も記録。**10 日分の暫定値**。 |
| [仮想成行 Q に対する板の応答と脆さ](mu_impact_fragility_report.md) | 板のラダー全体を 5 秒ごとに組み直し、仮想の成行 Q(0.5 / 5 / 50 契約)に対する sweep levels・price impact・slippage・marginal depth・marginal impact と、板の脆さ 5 種(fragility・その偏り・depth-at-risk・gap 調整)を出す。Impact ∝ Q^0.25、板は毎秒 10.5% 入れ替わる。**Impact の非対称は 5 秒先の向きに −0.070 で、マイクロプライス(+0.064)と OFI(+0.049)を上回り、しかも後ろ向き相関がほぼ 0**。ただし成行で取るには往復スプレッドに届かない。 |
| [見せかけの板を疑う 18 指標](mu_manipulation_report.md) | spoofing / layering の教科書的な形 18 通りを注文 1 本ごとに測り、口座別に集計する。**母集団のベースラインが高すぎて、ほとんど何も選り分けない** — 最良気配に届いた注文の 87% は約定せず引かれ、取消率は 98.9%。fleeting liquidity は本数で 54% だが表示時間で数えると 1.3%(41 倍差)。決定的な検証(大口を出したあと自分が反対側で売り抜けたか)では**署名が出ないどころか符号が逆**で、反対側で売った層では値段がその注文と逆へ 3.50 bp 動いていた。口座は内部連番のみで扱う。 |
| [複数の口座は同時に動くか — 同期の 8 指標](mu_sync_report.md) | 上位 60 口座(全注文の 87.7%)1,445 ペアについて、ブロック粒度の共起と秒粒度の相関を帰無との比で測る。**同時性は本物**(分層化した期待値の 1.43 倍、取消は 1.70 倍)だが、**向きは同期しない**(符号つき出し入れの超過相関は +0.0085 しかない)。さらに**同期の 9 割は「全員が同じ市場に反応している」で説明でき**、両者を除いた市場を統制すると超過は +0.137 → +0.014 へ落ちる。ただし 1,445 ペア中 **18〜19 ペアだけは残差相関が 0.5 を超える**(帰無では 0 件)。 |
| [隠れ数量(iceberg)と注文分割の代理指標 8 つ](mu_iceberg_report.md) | 「削られたから足す」(約定の後)と「値段を付け直す」(取消の後)を分けて、隠れ数量の痕跡を 8 通りに測る。**市場全体では痕跡が出ない** — 1 秒以内の出し直しは約定後 12.99% 対 取消後 12.79% でほぼ同じ、約定のあった補充の 96.1% は見せた最大の数量以下しか約定していない。距離帯を揃えても 9 帯中 8 帯で約定後のほうが低く、唯一の例外がスプレッドの内側(比 1.18)。395 口座のうち 3 条件すべてを満たすのは **1 口座**(全注文の 0.034%)。 |
| [OBI・OFI・攻撃的注文の五分位と予測ホライズン](mu_prediction_report.md) | 3 指標を前日の分布で五分位に切り、100ms〜60s の 8 ホライズンの前向きリターンを 1.41 億観測で測る。3 つとも t は 3.3〜20.9 で有意だが意味は別物 — **OBI の予測力はまるごと「mid が既知の microprice に追いつく」動きで、microprice で測ると全ホライズンで符号が負**(1 秒で t=−6.1)。OFI と攻撃的注文は microprice 自体を動かすので本物。攻撃的注文の**件数**は向きを当てず、絶対リターンを 100ms で 4.3 倍に分ける。ただし**板を叩くと 32 通りすべて −0.96〜−1.54 bp の赤字**。 |
| [絞った標本の burstiness と自己相関](mu_burstiness_report.md) | Q1/Q5 の標本が時間的に固まっているかを測り、標準誤差の正しさを検証する。**特徴量は強く自己相関**(OBI のラグ 1 秒 0.460、分散の膨張率 100 倍)だが**リターンはほぼ白色**(0.040、膨張率 1.3 倍)。選ばれ方は明確に塊で、連は独立の想定より 1.9〜8.1 倍長く Fano 因子は 1 時間窓で 105〜393。**独立とみなした標準誤差は最大 15.8 倍過小**。さらに日どうしも独立でなく(ラグ 7 で +0.703 の週次の山)、**予測レポートの OBI の結論を 1 つ訂正した**。 |
| [信号を検出した瞬間に BBO へメイカー注文を置いたら](mu_maker_report.md) | 6 通り(3 指標 × Q1/Q5)について、その瞬間に最良気配へ指値を置いた場合の約定率を待ち行列モデルで算出し、待ち行列がはける速さの十分位ごとに将来リターンを見る。10 秒の約定率は 11.9〜31.3%。**約定したときの損益は 8 ホライズンすべて負**で、受け取る半スプレッド 0.6〜0.9 bp を約定後の動き −2.2〜−3.9 bp が必ず上回る。**信号の向きに置くと逆選択は悪化する**(10 秒で OBI −3.24 対 対照 Q3 −2.21)。十分位では **mid と microprice でリターンの向きが逆**になり、OFI は microprice で見ると約定しやすい D1 が最も強い(+0.206)。 |
| [信号 × 待ち行列時間の 2 次元ヒートマップ](mu_heatmap_report.md) | 信号の十分位(縦)× 待ち行列がはける時間の十分位(横)の 10×10 を特徴量ごとに。**2 軸はほぼ直交する** — 約定できるかは横軸だけ(OFI で横 46.9pp 対 縦 5.0pp)、情報があるかは縦軸だけ(microprice で縦 0.601bp 対 横 0.066bp)。ただし **mid で測ると横軸に偽の勾配が出る**(OFI の比 1.21 → microprice で 9.04)。**300 セルすべてで約定時の損益は負**、最良でも −1.20 bp。 |
| [約定時刻から測った逆選択 AS と、クオートの期待値 EV](mu_ev_report.md) | **AS(S,D) = E[r(τ→τ+h) | Fill, S, D]** を実際の約定時刻 τ から測り、EV = P(Fill)×[Spread/2 − Fee ± AS] を全セルで作る。**クオートすべきセルは 1 つも無い**(日次 Newey-West で検査した 1,717 セル中 0 件)。理由は 1 行 — **自分を食った約定の即時の影響 −0.54 bp だけで、受け取る半スプレッド +0.58 bp がほぼ消える**。h=100ms まで縮めても負。測定の誤りを 2 つ修正(約定時刻ちょうどの板は約定後・区分累積和の桁落ち)。 |
| [SelectionPenalty(S,D) — 約定という条件が alpha を壊す量](mu_penalty_report.md) | **E[r|Fill,S,D] − E[r|S,D]** を同じ窓で作る。**alpha は約定という条件だけで平均 16 倍(最大 58 倍)の大きさで消える**。誤差を付けた 1,717 セルすべてが負で、1,715 が Bonferroni を突破、正のセルは 0。最も破壊されるのは**自分がクオートしたい向きに信号が出ている状態**(OBI S10 で 無条件 +0.560 → 約定時 −4.135 = **−4.694 bp**)。mid でも microprice でも同じ値になる稀な指標。 |
| [約定の 3 分解と toxic な約定の事前兆候](mu_markout_report.md) | SelectionPenalty(窓 T→T+h)と AS(窓 τ→τ+h)が**別物である**ことを恒等式で検算したうえで、**A = r(T→τ)** と **B = r(τ→τ+h)** に分ける。**A はクオートの寿命でほぼ決まり**(100ms で −0.007、10 秒で −1.618 bp)、**B は寿命に依らない**(−1.21〜−1.50)。実際の優位がゼロを切るのは寿命 2〜5 秒。toxic な約定を事前に見分けられるかは **AUC 最良 0.452** でほぼ不可能。 |
| [約定 δ ms 前の標本外 AUC と実際に取り消せる速さ](mu_latency_report.md) | δ = 500/250/100/50/20/10 ms で各特徴量と 10 変数ロジットの**標本外** AUC を出し、実測の取り消し遅延と並べる。AUC は **50 ms で 0.590 に飽和**するが、**50 ms より内側は板が動いていない**(50→20ms で 99.6%、20→10ms で 100% 同一)。ブロックは **65 ms 周期**、次の板更新までの待ちは中央値 **185 ms**、**50 ms 以内に取り消せている注文は 0.01%**。使える δ=250ms の AUC 0.577 で上位半分を捨てても markout の改善は +0.267 bp、黒字化に必要な +1.031 bp に届かない。 |
| [遅延を織り込んだ取り消しの backtest](mu_cancel_bt_report.md) | 板が更新されるたびにスコアを計算し、閾値超えで取り消しを出す。取り消しは **L ミリ秒後**に有効になり、約定がそれより早ければ間に合わない。**スコアは本物**(無作為対照との差は L=130ms で +0.231 bp)だが、**「何もしない」との差は +0.098 bp しかなく L=300ms で消える**。実測遅延で到達できる最良の markout は **−0.850**(基準 −1.001)で、−0.6 には届かない。1 発注あたりの期待値は取り消しを増やすほど**ゼロに近づくが超えない**(block bootstrap 95% 区間が 22 通りすべてで 0 を跨がない)。 |
| [BBO から 10 ティック奥までの P(Fill) × PnL_fill](mu_depth_report.md) | 最良気配から k=0..10 ティック奥に置いた場合。**奥へ置いても実際の優位は増えない**(名目の k ティックは、そこまで価格が落ちないと約定しないので丸ごと消える)。h ≥ 1 秒はどの深さも負。88 通り中**正で有意なのは 1 通りだけ** — **最良気配・待ち行列の先頭・保有 100 ミリ秒**(EV +0.036 bp、t=4.15、前半後半とも正)。待ち行列の優先権の価値がここに出ている。 |
| [仮想発注候補テーブル(Step 1)](mu_quotes_table_report.md) | 今後の分析の**基礎データ**。1 行 = 1 つの発注候補で、候補時点は BBO 更新ごと、1 時点につき買い/売りの 2 行。**98 日 7,034 万行 × 54 列**(説明変数 45 / ラベル 9)。規則は **X_t は t までに観測可能な値だけ**で、これを独立再計算(相対差 5×10⁻⁸ 以下)と板の時刻の検定(t 以前のセル開始時点と一致 77〜88% 対 t 時点 0.1〜1.2%)で確かめた。無条件に最良気配へ置くと **1 秒 net PnL 平均 −3.756 bp、98 日中 97 日が負**。期間内で標本の性質が変わる(候補数 1.65 倍・スプレッド 2.34→0.98 bp)ので**まとめて平均してはいけない**。NaN と null の混在で偽陽性を大量に出した経緯も記録。 |
| [250ms 約定ハザード × 条件つき損益](mu_fillpnl_report.md) | 60 秒 1 本の Hazard をやめ、**P(Fill 250ms) と E[PnL_1s|Fill] を別々に当てはめて** 予測値の 10×10 を見る。**EV>0 の島はある** — 55/100 が正、29 が Bonferroni を超え、無作為対照は 0/100。前後半の相関 +0.948、買い売り対称、遅延 130 ms でも 10 セル残る。ただし **EV は +0.0042 bp/候補・+0.346 bp/約定と小さく、mid の markout であって往復ではない**(手仕舞いに半スプレッド 0.74 bp を払えば消える)。構造は **符号は予測損益 Q が決め、大きさは約定確率 H が決める** — H は情報軸ではなく倍率。解釈可能な 3 変数版では島が 0 セルになるのが最大の留保。 |
| [microprice 再検証・往復 backtest・モデル比較・門の重ね合わせ](mu_roundtrip_report.md) | 前報の島を 4 通りに潰しにかかった。**① microprice でも消えない**(1 約定 +0.346 → +0.498 bp、セル間相関 +0.96)。**③ 共通分母をほどいた特徴量(FlowBalance + LiquidityScale)でも OLS/Ridge/ElasticNet/GAM の 4 つとも残る** — Ridge が選ぶ罰則は **λ=0** で共線性は解けた。しかし **② 手仕舞いを入れると 100 セル中 陽性 0**(Taker −1.117 / Hybrid10 −1.499 / Hybrid60 −2.493 bp/約定)。受動的手仕舞いの +1.264 bp は**閉じた分だけを見た罠**(10 秒で閉じるのは 26%)。**④ 学習期間で領域を固定し OBI・OFI の門を足すと −0.0145 → −0.0056 bp/候補まで縮むがゼロを超えない**(t=−8.7)。★**スプレッドは markout を 16 倍にするが往復は平ら**、**OBI は markout では平らだが往復を 3 倍改善する** — markout と往復では効く変数が違う。 |
| [指値注文の生存率・取消率・約定率(ハザード)](mu_hazard_report.md) | 板に置かれた指値 5.5 億本を「寿命」を持つ個体として扱い、生存率・原因別ハザード(取消/約定)・累積発生確率を出す。発注時に判る 6 条件(前に並んだ数量・自分の数量・最良からの距離・口座の累計本数・ボラティリティ・OFI)で層別。最終的に約定するのは 0.915% だけで、最も効くのは口座(81 倍)。OFI は最終約定率では効かないように見えて、0.1 秒後の約定ハザードでは 3.6 倍開く。**全 98 日**。 |
| [束の間の注文(fleeting order)は板の何割を占めるか](mu_fleeting_report.md) | 板に置かれてすぐ約定せずに取り消される指値を、8 つの閾値・側・最良からの距離・注文数量・口座で数える。2 秒以内に消えるのは数量の 66.4%、最良気配のそばに限れば 90.0%。大口(上位 1%)だけは 36.1% と半分近く、200ms 以内に消えるのは 1.7% しかない。口座ごとの FLR のばらつきは二項の帰無対照の 47 倍で、**全体 71.8% と口座の中央値 22.5% が分母の違いだけで 3 倍ずれる**ことも示す。**全 98 日**。 |
| [板の数量は何者の口座に集まっているか](mu_wallet_conc_report.md) | 1 秒ごとに板を復元し、数量を置いている口座のシェアから 14 の集中度指標を出す。板全体には 313 者が居るのに **最良気配を持つのは常に 3.17 者**で、そこにある数量は板の 0.14% しかない。touch の HHI 0.644 に対し deep は 0.081。再構成した板が壊れていても指標は整合して見えるため、**2 度にわたり板が単調に膨らんだ**経緯(繰越注文の口座欠落 / 終端が来ない注文)と、bbo との突合で気づいた顛末も記録。**全 98 日**。 |
| [ウォレットの行動クラスタ](mu_wallet_clusters_report.md) | 注文の作り方を 7 領域の分布(数量・距離・生存時間・時刻・取消・価格の置き方・最良気配)で表し、Jensen–Shannon 距離から類似度・クラスタ・2 次元の埋め込みを作る。380 者・72,010 組。**分かれるのは「どこに置くか」で、「いくらで置くか」では分かれない**(size-profile の分離幅は −0.005)。事前宣言した規則では潰れた分割しか出ず、手法を変えた経緯も記録。 |
| [メッセージの流量(quote stuffing / message activity)](mu_msg_activity_report.md) | 板を作らず、流れてくるメッセージ 13.3 億通そのものを数える。中央 111 通/秒、最繁の秒は 4,448 通。約定 1 件あたり 103 通・新規発注 47 本で、7 通に 1 通は拒否。★**「イベント間隔」はチェーンのブロック周期 67.3ms に量子化されており**(99 日を通して 67.18〜67.50ms)、間隔・分散・CV・burstiness は参加者の速さではなく「動きのあったブロックの間引き」を測っている。群れは Fano factor(1 秒窓で 172、ポアソンなら 1)と 1 時間先まで残る自己相関に出る。**全 99 日**。 |
| [最良気配近くのメイカーの混み具合と競争](mu_maker_crowd_report.md) | 最良から 10 ティック以内に居る口座だけを取り出し、混み具合と競争を 9 指標で測る。中央 10.6 者居ても **実効は 3.35 者**で、上位 1 者が数量の 50.6% を持つ。★**帯を 0 → 25 ティックに広げると人数は 8.4 倍になるのに実効は 3.2 倍止まりで、均等度(MCI)は 0.659 → 0.261 へ下がる**(奥のメイカーは競争の実体になっていない)。顔ぶれは 1 分で 6 割入れ替わり、1 価格には 1.54 者しか居ない。帯 0 の人数が[口座の集中度]の BBO wallet count と 98 日すべてで一致(相関 1.00000)。**全 98 日**。 |
| [最良気配の入れ替わり(BBO turnover)](mu_bbo_turnover_report.md) | 最良価格が保たれる時間は中央 343ms しかなく 1 秒格子では測れないので、格子を使わず区間の交差だけで ns 精度で測る。置換 1.32 回/秒、1 秒後も同じ価格である確率は 0.40。★**注文は毎秒 2.87 本入れ替わるのに数量は 0.15 回転/秒しか入れ替わらず、19 倍の開きがある**(入れ替わっているのは小さい注文で、厚みを作る注文は長く居座る)。口座が最良を保つ時間は 363ms で、1 本の注文が居る 335ms とほぼ変わらない。**全 98 日**。 |
| [9 系列の長期記憶と自己相関](mu_longmem_report.md) | OFI・depth・spread・order arrivals・cancellations・order size・liquidity churn・queue size・wallet activity の 9 系列を 1 秒格子に載せ、ACF / PACF / Hurst / DFA / GPH / Local Whittle の 8 推定量で測る。★**系列を並べ替えた帰無対照で R/S の Hurst だけ +0.055 上振れする**(他 3 つは不偏)ので補正せずに読んではいけない。★**4 推定量から出した d は到着系で 1.5 倍食い違い、1 つの値に決まらない**。OFI は ρ(1)=0.03 とほぼ無相関なのに d=0.12 の記憶を持つ。日内周期を除いても d は平均 −0.010 しか動かない。**全 98 日**。 |

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
| [板と注文イベントのエントロピー 13 種](mu_entropy_report.md) | 板の散らばり具合を 13 通り(価格水準別の数量・本数、注文数量、口座、滞留時間、最良の待ち行列、イベントの種別・側・向き、Δエントロピー、エントロピー率、条件付き、遷移)測り、6 地平 × 2 目的変数の 156 セルへ当てる。**Bonferroni を通った 49 セルは全部がボラティリティで、向きは 0 セル**。さらに前向き < 後ろ向き・活動量の統制で大半が消え、**4 段の検査を全部通ったのは156 セル中 1 つ**(Δエントロピーの 1 秒)。順位相関では見えなかった **U 字**を十分位表で見つけて結論を訂正した経緯も記録。**全 98 日**。 |

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

**9 系列の長期記憶と自己相関** — ACF、PACF、4 推定量から出した d、帰無対照で見た推定量の偏り、DFA のスケーリング、日内周期の除去、lag-k の一覧、推定量どうしの一致。解説: [9 系列の長期記憶と自己相関](mu_longmem_report.md)

![xyz:MU 9 系列の長期記憶と自己相関。ACF と PACF、Hurst / DFA / GPH / Local Whittle から出した分数階差 d、系列を並べ替えた帰無対照で見た推定量自身の偏り、DFA のスケーリング、日内周期を除いた場合との比較、lag-k の自己相関、推定量どうしの一致](../../../charts/xyz_MU_longmem.png)

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

**隠れ数量 — 補充の形** 削られたから足すのか、値段を付け直すのか。解説: [隠れ数量と注文分割の代理指標](mu_iceberg_report.md)

![xyz:MU 補充の形。1 秒以内に同じ値段へ出し直した割合、出し直すまでの時間の累積分布、距離帯ごとの比較、子注文の数の分布、同じ数量だった割合](../../../charts/xyz_MU_ice_refill.png)

**隠れ数量 — 代理指標と回転**。解説: [同上](mu_iceberg_report.md)

![xyz:MU 隠れ数量の代理と回転。約定量を見せた最大の数量で割った累積分布、補充した数量の比、価格水準の回転、最頻数量の割合、口座ごとの回転、約定後と取消後の散布図](../../../charts/xyz_MU_ice_hidden.png)

**口座ごとの約定のされ方** — 4 つの約定割合の重なり、出した量と約定量、待ち時間と約定の大きさ、約定時のキュー位置、約定後の逆選択。解説: [口座ごとの約定のされ方 11 指標](mu_wallet_fill_report.md)

![xyz:MU 口座ごとの約定のされ方 11 指標。約定割合 4 種の分布と相互相関、出した数量と約定した数量の散布、待ち時間と約定の大きさ、約定した瞬間に先頭だった割合、メイカー約定シェア上位 20 口座の約定後 10 秒の逆選択](../../../charts/xyz_MU_wallet_fill.png)

**予測 — 五分位ごとの前向きリターン** 3 指標 × 8 ホライズン。解説: [OBI・OFI・攻撃的注文の五分位と予測ホライズン](mu_prediction_report.md)

![xyz:MU OBI・OFI・攻撃的注文の符号つき数量・件数について、五分位ごとの前向きリターンを 100ms から 60s の 8 ホライズンで並べた 4 枚組](../../../charts/xyz_MU_pred_quintile.png)

**予測 — 新しい情報か、費用を引くと残るか**。解説: [同上](mu_prediction_report.md)

![xyz:MU mid で測った Q5−Q1 と microprice で測った Q5−Q1 の比較、往復スプレッドを引いた純益、帰無対照、符号つき t 値の 6 枚組](../../../charts/xyz_MU_pred_micro.png)

**予測 — Q1 と Q5 のときのリターン分布 6 通り**。解説: [同上](mu_prediction_report.md)

![xyz:MU OBI・OFI・攻撃的注文の符号つき数量について、Q1 と Q5 それぞれのリターン分布を 4 ホライズン重ねた 6 枚組](../../../charts/xyz_MU_pred_dist.png)

**予測 — Q1 と Q5 の違いの大きさ**。解説: [同上](mu_prediction_report.md)

![xyz:MU Q1 と Q5 の両側裾確率、効果量のホライズン依存、上がった割合、外れ値を丸めたときの頑健性の 6 枚組](../../../charts/xyz_MU_pred_dist_cmp.png)

**標本は時間的に固まっているか** 自己相関・連の長さ・Fano 因子・ブロック頑健な標準誤差。解説: [burstiness と自己相関](mu_burstiness_report.md)

![xyz:MU 特徴量とリターンの自己相関、Q5 の連の長さ、Fano 因子、ブロック幅ごとの標準誤差、日次系列の自己相関、補正前後の t 値の 6 枚組](../../../charts/xyz_MU_burst.png)

**メイカー注文の反実仮想 — 約定率と逆選択**。解説: [信号を検出した瞬間に BBO へメイカー注文を置いたら](mu_maker_report.md)

![xyz:MU 6 通りの約定率、対照との比、約定したときの損益、逆選択の分解、1 発注あたりの期待損益、約定率の上限下限の 6 枚組](../../../charts/xyz_MU_maker_fill.png)

**メイカー注文の反実仮想 — 待ち行列の十分位**。解説: [同上](mu_maker_report.md)

![xyz:MU 待ち行列がはける推定時間の十分位ごとの約定率と、mid と microprice で測った将来リターンの 6 枚組](../../../charts/xyz_MU_maker_decile.png)

**信号 × 待ち行列時間の 2 次元** 特徴量ごとに 1 枚。解説: [信号 × 待ち行列時間の 2 次元ヒートマップ](mu_heatmap_report.md)

![xyz:MU OBI の信号十分位 × 待ち行列十分位。約定率(買い・売り)、観測数、mid と microprice で測った将来リターン、約定時の損益の 6 枚組](../../../charts/xyz_MU_heat_obi.png)

![xyz:MU OFI の信号十分位 × 待ち行列十分位。約定率は横軸だけ、microprice のリターンは縦軸だけに勾配が出る 6 枚組](../../../charts/xyz_MU_heat_ofi_10s.png)

![xyz:MU 攻撃的注文の符号つき数量の信号十分位 × 待ち行列十分位の 6 枚組](../../../charts/xyz_MU_heat_ai_net_10s.png)

**約定時刻から測った AS と EV** どのセルならクオートすべきか。解説: [約定時刻から測った逆選択 AS と EV](mu_ev_report.md)

![xyz:MU OFI の信号十分位 × 待ち行列十分位。約定確率、約定時刻から測った逆選択 AS、約定 2ms 前の半スプレッド、EV_bid、EV_ask、1 約定あたりの中身の 6 枚組。EV は全セル負](../../../charts/xyz_MU_ev_ofi_10s.png)

![xyz:MU OBI についての同じ 6 枚組](../../../charts/xyz_MU_ev_obi.png)

![xyz:MU 攻撃的注文の符号つき数量についての同じ 6 枚組](../../../charts/xyz_MU_ev_ai_net_10s.png)

**SelectionPenalty(S,D)** 約定条件で alpha がどれだけ壊れるか。解説: [SelectionPenalty](mu_penalty_report.md)

![xyz:MU OBI の信号十分位 × 待ち行列十分位。無条件リターン、約定条件つきリターン、SelectionPenalty(買い・売り)、microprice 版、P(Fill) の 6 枚組。penalty は全セル負](../../../charts/xyz_MU_penalty_obi.png)

![xyz:MU OFI についての同じ 6 枚組](../../../charts/xyz_MU_penalty_ofi_10s.png)

![xyz:MU 攻撃的注文の符号つき数量についての同じ 6 枚組](../../../charts/xyz_MU_penalty_ai_net_10s.png)

**BBO から 10 ティック奥までの P(Fill) × PnL_fill**。解説: [深い水準](mu_depth_report.md)

![xyz:MU 最良気配から k ティック奥に置いた場合の約定率、約定時損益、EV、損益の内訳、待ち時間、信号十分位との組み合わせの 6 枚組](../../../charts/xyz_MU_depth.png)

**約定の 3 分解と toxic の事前兆候**。解説: [3 分解と事前兆候](mu_markout_report.md)

![xyz:MU クオートの寿命ごとの損益分解、A と B の寿命依存、toxic 五分位ごとの攻撃的流量と OFI の推移、群間の開き、事前に toxic を当てる AUC の 6 枚組](../../../charts/xyz_MU_markout.png)

**標本外 AUC と取り消し遅延**。解説: [δ ms 前の AUC と取り消せる速さ](mu_latency_report.md)

![xyz:MU 約定 500/250/100/50/20/10 ms 前の標本外 AUC、ブロック間隔の分布、参加者が実際に取り消せている速さの 3 枚組。取り消しが間に合わない領域を赤帯で示す](../../../charts/xyz_MU_latency.png)

**遅延を織り込んだ取り消しの backtest**。解説: [cancel backtest](mu_cancel_bt_report.md)

![xyz:MU 発火率ごとの markout、遅延による効果の減衰、無作為対照との差、約定率、1 約定あたり損益、1 発注あたり期待値の 6 枚組](../../../charts/xyz_MU_cancel_bt.png)

**取り消し backtest の block bootstrap 95% 区間**。解説: [同上](mu_cancel_bt_report.md)

![xyz:MU 標本外日次 EV 系列の巡回移動ブロック bootstrap による 95% 区間。EV・何もしないとの差・無作為との差の 3 枚組](../../../charts/xyz_MU_cancel_ci.png)

**仮想発注候補テーブルの日次推移**。解説: [Step 1 候補テーブル](mu_quotes_table_report.md)

![xyz:MU 仮想発注候補テーブルの日次推移。候補行数・60 秒約定率・スプレッド中央値・1 秒 net PnL 平均の 4 面](../../../charts/xyz_MU_quotes_daily.png)

**候補テーブルの説明変数とラベルの分布**。解説: [同上](mu_quotes_table_report.md)

![xyz:MU 候補テーブルの分布。上段は t 時点で観測可能な説明変数 4 つ、下段は未来を使うラベル 4 つ](../../../charts/xyz_MU_quotes_dist.png)

**250ms 約定ハザード × 条件つき損益の 10×10**。解説: [EV>0 の島はあるか](mu_fillpnl_report.md)

![xyz:MU 予測約定確率の十分位 × 予測損益の十分位。実測 P(Fill 250ms)、約定あたり損益、候補あたり EV、候補数の 4 枚組](../../../charts/xyz_MU_fillpnl.png)

**島の頑健性**。解説: [同上](mu_fillpnl_report.md)

![xyz:MU 遅延 0/65/130 ms の EV、評価期間の前後半の一致、3 変数版、セル平均スプレッドとの関係の 6 枚組](../../../charts/xyz_MU_fillpnl_rb.png)

**microprice 再検証と往復 backtest**。解説: [往復で島は消える](mu_roundtrip_report.md)

![xyz:MU microprice 基準の EV、mid との対応、手仕舞い 3 通りの比較、受動的手仕舞いの約定曲線、Taker exit の内訳、往復 EV の 6 枚組](../../../charts/xyz_MU_rt_micro.png)

**モデル比較と門の重ね合わせ**。解説: [同上](mu_roundtrip_report.md)

![xyz:MU 共通分母をほどいた 4 推定量の比較と、領域内で OBI・OFI・スプレッドの門を順に足したときの EV の 6 枚組](../../../charts/xyz_MU_rt_gates.png)

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

**エントロピー 13 種は何を捉えているか** — 3 系統の分布、実効的な個数 2^H、日ごとの平均、13 種 + 統制 2 の順位相関。解説: [板と注文イベントのエントロピー 13 種](mu_entropy_report.md)

![xyz:MU 板と注文イベントのエントロピー 13 種。板・イベント・動的の 3 系統の分布、実効的な個数 2^H、日ごとの平均の推移、13 種と統制 2 変数の順位相関](../../../charts/xyz_MU_entropy.png)

**エントロピーは将来の値動きを予測するか** — 156 セル全件の符号検定、前向き 対 後ろ向き、帰無対照、活動量の統制、U 字の十分位。解説: [同上](mu_entropy_report.md)

![xyz:MU エントロピー 13 種の予測力。将来ボラティリティと向きへの符号検定の行列、予測か後始末かの散布図、帰無対照、活動量を統制した偏相関、Δエントロピーの十分位が描く U 字](../../../charts/xyz_MU_entropy_predict.png)

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
| [longmem_daily_xyz_MU.csv](../../../data/longmem_daily_xyz_MU.csv) | 日 × 9 系列 × 3 種別(生 / 周期除去 / 帰無対照)× 8 推定量(2,646 行) | `build_longmem.py` |
| longmem_curves_xyz_MU.parquet | 日 × 系列 × (ACF / PACF / DFA / R/S)の曲線 | `build_longmem.py` |
| longmem_profile_xyz_MU.parquet | 98 日から作った時刻(秒)ごとのプロファイル | `build_longmem.py` |

| [manip_daily_xyz_MU.csv](../../../data/manip_daily_xyz_MU.csv) | 見せかけの板の指標の市場全体ベースライン(98 行) | `build_manip.py` |
| [manip_scores_xyz_MU.csv](../../../data/manip_scores_xyz_MU.csv) | 口座ごとの 18 指標と spoof / layering スコア(729 行) | `plot_manip.py` |
| [manip_meta_xyz_MU.csv](../../../data/manip_meta_xyz_MU.csv) | 見せかけの板の日ごとの検算(98 行) | `build_manip.py` |
| manip_wallet_xyz_MU.parquet | 口座 × 日 の集計(69,448 行、版管理外) | `build_manip.py` |
| manip_episodes_xyz_MU.parquet | 最良の近くに出した大口 1 本ごと(307,599 行、版管理外) | `build_manip.py` |
| [entropy_fit_xyz_MU.csv](../../../data/entropy_fit_xyz_MU.csv) | 156 セルの全結果(効果量・符号検定・帰無対照・統制・後ろ向き) | `fit_entropy.py` |
| [entropy_corr_xyz_MU.csv](../../../data/entropy_corr_xyz_MU.csv) | 13 種 + 統制 2 の順位相関行列 | `fit_entropy.py` |
| [entropy_daily_xyz_MU.csv](../../../data/entropy_daily_xyz_MU.csv) | 日ごとの平均(98 行) | `fit_entropy.py` |
| [entropy_abschg_xyz_MU.csv](../../../data/entropy_abschg_xyz_MU.csv) | \|Δh_depth\| の追試(6 地平) | `fit_entropy.py` |
| [entropy_decile_xyz_MU.csv](../../../data/entropy_decile_xyz_MU.csv) | Δh_depth の十分位ごとの将来 \|r\|(U 字の数値) | `fit_entropy.py` |
| [entropy_meta_xyz_MU.csv](../../../data/entropy_meta_xyz_MU.csv) | 日ごとの検算(格子数・欠測・薄い格子・繰越) | `build_entropy.py` |
| entropy_xyz_MU.parquet | 1 秒格子 × 13 特徴量 + 統制 2 + 将来リターン(846 万行、版管理外) | `build_entropy.py` |
| [wfmetrics_xyz_MU.csv](../../../data/wfmetrics_xyz_MU.csv) | 口座 × 11 指標(395 口座) | `fit_wallet_fill.py` |
| [wfmetrics_corr_xyz_MU.csv](../../../data/wfmetrics_corr_xyz_MU.csv) | 11 指標どうしの順位相関 | `fit_wallet_fill.py` |
| [wfill_meta_xyz_MU.csv](../../../data/wfill_meta_xyz_MU.csv) | 日ごとの検算(事象数・約定数・孤児・打ち切り) | `build_wallet_fill.py` |
| wfill_wallet_xyz_MU.parquet | 口座ごとの集計(14,877 者、版管理外) | `build_wallet_fill.py` |
| wfill_fills_xyz_MU.parquet | 約定 1 件ごとの記録(765.9 万件、キュー位置と逆選択つき、版管理外) | `build_wallet_fill.py` |

| [sync_pairs_xyz_MU.csv](../../../data/sync_pairs_xyz_MU.csv) | 口座ペアごとの共起比・相関・偏相関と帰無(1,445 行) | `plot_sync.py` |
| [sync_wallet_xyz_MU.csv](../../../data/sync_wallet_xyz_MU.csv) | 口座ごとの共通成分と herding score(60 行) | `plot_sync.py` |
| [sync_meta_xyz_MU.csv](../../../data/sync_meta_xyz_MU.csv) | 同期の日ごとの検算(98 行) | `build_sync.py` |
| sync_raw_xyz_MU.npz | 日 × 口座 × 口座 の十分統計(版管理外) | `build_sync.py` |
| [wallet_clusters_xyz_MU.csv](../../../data/wallet_clusters_xyz_MU.csv) | ウォレット × クラスタ ID・埋め込み座標・平均類似度(380 行) | `build_wallet_clusters.py` |

| [ice_daily_xyz_MU.csv](../../../data/ice_daily_xyz_MU.csv) | 隠れ数量の代理指標の市場全体の日次(98 行) | `build_iceberg.py` |
| [ice_band_xyz_MU.csv](../../../data/ice_band_xyz_MU.csv) | 距離帯 × 日 の補充率と回転(882 行) | `build_iceberg.py` |
| [ice_hist_xyz_MU.csv](../../../data/ice_hist_xyz_MU.csv) | 子注文数・出し直しまでの時間・隠れ倍率の分布 | `build_iceberg.py` |
| [ice_wallet_summary_xyz_MU.csv](../../../data/ice_wallet_summary_xyz_MU.csv) | 口座ごとの 8 指標(395 行) | `plot_iceberg.py` |
| [ice_meta_xyz_MU.csv](../../../data/ice_meta_xyz_MU.csv) | 隠れ数量の日ごとの検算(98 行) | `build_iceberg.py` |
| ice_wallet_xyz_MU.parquet | 口座 × 日 の集計(版管理外) | `build_iceberg.py` |

| pred_cells_xyz_MU.parquet | 日 × 指標 × ホライズン × 五分位 の十分統計量(17,720 セル・版管理外) | `build_pred.py` |
| [pred_edges_xyz_MU.csv](../../../data/pred_edges_xyz_MU.csv) | 日ごとの五分位の境目(686 行) | `build_pred.py` |
| [pred_summary_xyz_MU.csv](../../../data/pred_summary_xyz_MU.csv) | 指標 × ホライズンの要約(32 行) | `plot_pred.py` |
| pred_hist_xyz_MU.parquet | リターン分布のヒストグラム(対数等比 243 ビン・33,437 行・版管理外) | `build_pred.py` |
| [pred_dist_stats_xyz_MU.csv](../../../data/pred_dist_stats_xyz_MU.csv) | 6 分布 × 8 ホライズンの要約統計(48 行) | `plot_pred_dist.py` |
| [burst_acf_xyz_MU.csv](../../../data/burst_acf_xyz_MU.csv) | 自己相関 5 系列 × ラグ 1〜1800 秒 | `build_burst.py` |
| [burst_summary_xyz_MU.csv](../../../data/burst_summary_xyz_MU.csv) | burstiness・記憶・Fano・Gini(6 行) | `build_burst.py` |
| [burst_block_xyz_MU.csv](../../../data/burst_block_xyz_MU.csv) | ブロック幅ごとの標準誤差(60 行) | `build_burst.py` |
| [burst_run_xyz_MU.csv](../../../data/burst_run_xyz_MU.csv) | Q1/Q5 に入り続けた連の長さの分布 | `build_burst.py` |
| [burst_tvalues_xyz_MU.csv](../../../data/burst_tvalues_xyz_MU.csv) | 日次と Newey-West の t 値(64 行・ラグ 5/10/14/21) | `plot_burst.py` |
| maker_cells_xyz_MU.parquet | 日 × 指標 × 分位 × 側 × ホライズン × 十分位(133,664 セル・版管理外) | `build_maker.py` |
| [maker_summary_xyz_MU.csv](../../../data/maker_summary_xyz_MU.csv) | 約定率・損益・逆選択の要約 | `plot_maker.py` |
| [maker_decile_xyz_MU.csv](../../../data/maker_decile_xyz_MU.csv) | 十分位ごとの約定率・リターン・損益 | `plot_maker.py` |
| [maker_edges_xyz_MU.csv](../../../data/maker_edges_xyz_MU.csv) | 日ごとの十分位の境目(196 行) | `build_maker.py` |
| [heat_pooled_xyz_MU.csv](../../../data/heat_pooled_xyz_MU.csv) | 信号 × 待ち行列の 10×10(4,800 セル) | `build_heat.py` |
| heat_daily_xyz_MU.parquet | 同・日ごと(103,144 行・h=1s/10s・版管理外) | `build_heat.py` |
| [heat_summary_xyz_MU.csv](../../../data/heat_summary_xyz_MU.csv) | 軸ごとの勾配の大きさ | `plot_heat.py` |
| [ev_pooled_xyz_MU.csv](../../../data/ev_pooled_xyz_MU.csv) | AS と EV の材料(4,800 セル) | `build_ev.py` |
| ev_daily_xyz_MU.parquet | 同・日ごと(154,716 行・版管理外) | `build_ev.py` |
| [ev_summary_xyz_MU.csv](../../../data/ev_summary_xyz_MU.csv) | EV と日次 Newey-West 誤差 | `plot_ev.py` |
| [ev_positive_xyz_MU.csv](../../../data/ev_positive_xyz_MU.csv) | 判定を通ったセル(0 件) | `plot_ev.py` |
| [penalty_summary_xyz_MU.csv](../../../data/penalty_summary_xyz_MU.csv) | SelectionPenalty と日次 Newey-West 誤差 | `plot_penalty.py` |
| [depth_pooled_xyz_MU.csv](../../../data/depth_pooled_xyz_MU.csv) | 水準 × 側 × ホライズン(176 行) | `build_depth.py` |
| [depth_signal_xyz_MU.csv](../../../data/depth_signal_xyz_MU.csv) | 同・信号十分位つき(5,280 行) | `build_depth.py` |
| depth_daily_xyz_MU.parquet | 同・日ごと(8,624 行・版管理外) | `build_depth.py` |
| [markout_decomp_xyz_MU.csv](../../../data/markout_decomp_xyz_MU.csv) | 日 × 側 × クオート寿命の 3 分解 | `build_markout.py` |
| [markout_toxic_xyz_MU.csv](../../../data/markout_toxic_xyz_MU.csv) | toxic 五分位 × 事前時点の状態(3,920 行) | `build_markout.py` |
| [markout_auc_xyz_MU.csv](../../../data/markout_auc_xyz_MU.csv) | 事前指標の判別力(3,528 行) | `build_markout.py` |
| [markout_summary_xyz_MU.csv](../../../data/markout_summary_xyz_MU.csv) | 寿命ごとの A・B と誤差 | `plot_markout.py` |
| latency_features_xyz_MU.parquet | 約定 δ ms 前の特徴量(352.8 万行・版管理外) | `build_latency.py` |
| [latency_auc_xyz_MU.csv](../../../data/latency_auc_xyz_MU.csv) | 11 モデル × 6 時点の標本外 AUC | `plot_latency.py` |
| cancel_bt_xyz_MU.parquet | 取り消し backtest の日次結果(12,740 行・版管理外) | `build_cancel_bt.py` |
| cancel_bt_xyz_MU_hq2.parquet | 同・クオート寿命 2 秒(版管理外) | `build_cancel_bt.py` |
| [cancel_bt_summary_xyz_MU.csv](../../../data/cancel_bt_summary_xyz_MU.csv) | 発火率 × 遅延の要約 | `plot_cancel_bt.py` |
| [cancel_ci_xyz_MU.csv](../../../data/cancel_ci_xyz_MU.csv) | block bootstrap 95% 区間(ブロック長 4 通り) | `plot_cancel_ci.py` |
| quotes_sub_xyz_MU/ | 仮想発注候補テーブルの 1/20 間引き版(351.7 万行 × 54 列・版管理外) | `build_quotes.py` |
| [quotes_day_xyz_MU.csv](../../../data/quotes_day_xyz_MU.csv) | 候補テーブルの日次集計(98 日) | `check_quotes.py` |
| [quotes_stats_xyz_MU.csv](../../../data/quotes_stats_xyz_MU.csv) | 列ごとの欠損率と分位(dt/ts/sec/side を除く 41 列) | `check_quotes.py` |
| [quotes_check_xyz_MU.json](../../../data/quotes_check_xyz_MU.json) | 整合性検査 8 項目の結果 | `check_quotes.py` |
| [quotes_audit_xyz_MU.csv](../../../data/quotes_audit_xyz_MU.csv) | ルックアヘッド監査(3 日 × A/B/C) | `audit_quotes.py` |
| [fillpnl_cells_xyz_MU.csv](../../../data/fillpnl_cells_xyz_MU.csv) | 10×10 セルの実測(遅延 0 ms) | `build_fillpnl.py` |
| [fillpnl_cells_xyz_MU_lag65.csv](../../../data/fillpnl_cells_xyz_MU_lag65.csv) | 同・遅延 65 ms | `build_fillpnl.py` |
| [fillpnl_cells_xyz_MU_lag130.csv](../../../data/fillpnl_cells_xyz_MU_lag130.csv) | 同・遅延 130 ms | `build_fillpnl.py` |
| [fillpnl_cells_xyz_MU_simple.csv](../../../data/fillpnl_cells_xyz_MU_simple.csv) | 同・3 変数版 | `build_fillpnl.py` |
| [fillpnl_summary_xyz_MU.csv](../../../data/fillpnl_summary_xyz_MU.csv) | 遅延ごとの要約 | `plot_fillpnl.py` |
| [fillpnl_meta_xyz_MU.json](../../../data/fillpnl_meta_xyz_MU.json) | 2 モデルの係数 | `build_fillpnl.py` |
| [fillpnl_models_xyz_MU.csv](../../../data/fillpnl_models_xyz_MU.csv) | OLS/Ridge/ENet/GAM の標本外比較 | `build_fillpnl_models.py` |
| [gates_stages_xyz_MU.csv](../../../data/gates_stages_xyz_MU.csv) | 門を順に足したときの EV | `build_gates.py` |
| [gates_quintiles_xyz_MU.csv](../../../data/gates_quintiles_xyz_MU.csv) | 領域内の五分位(markout と往復) | `build_gates.py` |
| quotes_mu_sub_xyz_MU/ | microprice ラベルの 1/20 間引き版(版管理外) | `build_mu_labels.py` |
| quotes_rt_sub_xyz_MU/ | 往復損益の 1/20 間引き版(版管理外) | `build_roundtrip.py` |

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
uv run python scripts/build_longmem.py --coin xyz:MU
uv run python scripts/plot_longmem.py --coin xyz:MU
uv run python scripts/build_impact.py --coin xyz:MU
uv run python scripts/build_impact_signal.py --coin xyz:MU
uv run python scripts/plot_impact.py --coin xyz:MU
uv run python scripts/build_manip.py --coin xyz:MU
uv run python scripts/plot_manip.py --coin xyz:MU
uv run python scripts/build_entropy.py --selftest
uv run python scripts/build_entropy.py --coin xyz:MU
uv run python scripts/build_entropy.py --coin xyz:MU --merge
uv run python scripts/fit_entropy.py --coin xyz:MU
uv run python scripts/plot_entropy.py --coin xyz:MU
uv run python scripts/build_wallet_fill.py --coin xyz:MU
uv run python scripts/build_wallet_fill.py --coin xyz:MU --merge
uv run python scripts/fit_wallet_fill.py --coin xyz:MU
uv run python scripts/plot_wallet_fill.py --coin xyz:MU
uv run python scripts/build_sync.py --coin xyz:MU
uv run python scripts/plot_sync.py --coin xyz:MU
uv run python scripts/build_iceberg.py --coin xyz:MU
uv run python scripts/plot_iceberg.py --coin xyz:MU
uv run python scripts/build_pred.py --coin xyz:MU
uv run python scripts/plot_pred.py --coin xyz:MU
uv run python scripts/plot_pred_dist.py --coin xyz:MU
uv run python scripts/build_burst.py --coin xyz:MU
uv run python scripts/plot_burst.py --coin xyz:MU
uv run python scripts/build_maker.py --coin xyz:MU
uv run python scripts/plot_maker.py --coin xyz:MU
uv run python scripts/build_heat.py --coin xyz:MU
uv run python scripts/plot_heat.py --coin xyz:MU
uv run python scripts/build_ev.py --coin xyz:MU
uv run python scripts/plot_ev.py --coin xyz:MU
uv run python scripts/plot_penalty.py --coin xyz:MU
uv run python scripts/build_depth.py --coin xyz:MU
uv run python scripts/plot_depth.py --coin xyz:MU
uv run python scripts/build_markout.py --coin xyz:MU
uv run python scripts/plot_markout.py --coin xyz:MU
uv run python scripts/build_latency.py --coin xyz:MU
uv run python scripts/plot_latency.py --coin xyz:MU
uv run python scripts/build_cancel_bt.py --coin xyz:MU
uv run python scripts/build_cancel_bt.py --coin xyz:MU --hq 2.0
uv run python scripts/plot_cancel_bt.py --coin xyz:MU
uv run python scripts/plot_cancel_ci.py --coin xyz:MU
uv run python scripts/build_quotes.py --coin xyz:MU
uv run python scripts/fix_quotes_nan.py --coin xyz:MU
uv run python scripts/check_quotes.py --coin xyz:MU
uv run python scripts/audit_quotes.py --coin xyz:MU
uv run python scripts/plot_quotes.py --coin xyz:MU
uv run python scripts/build_fillpnl.py --coin xyz:MU
uv run python scripts/build_fillpnl.py --coin xyz:MU --lag 0.065
uv run python scripts/build_fillpnl.py --coin xyz:MU --lag 0.130
uv run python scripts/build_fillpnl.py --coin xyz:MU --simple
uv run python scripts/plot_fillpnl.py --coin xyz:MU
uv run python scripts/build_mu_labels.py --coin xyz:MU
uv run python scripts/build_roundtrip.py --coin xyz:MU
uv run python scripts/build_fillpnl.py --coin xyz:MU --target mu
uv run python scripts/build_fillpnl_models.py --coin xyz:MU
uv run python scripts/build_gates.py --coin xyz:MU
uv run python scripts/plot_rt.py --coin xyz:MU
```

---

リポジトリ全体の目次は [../../../README.md](../../../README.md) にあります。
