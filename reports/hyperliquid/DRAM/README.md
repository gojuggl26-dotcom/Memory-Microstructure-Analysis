# xyz:DRAM マイクロストラクチャー分析(全期間・最高解像度)

[← ホーム](../../../README.md) / [Hyperliquid の分析](../README.md)

Hyperliquid の HIP-3 市場 xyz:DRAM / USDC(発行者 Trade.xyz、market_id 100065)を、
上場から 99 日間、板と約定のイベント 1 件単位で再構成し、特徴量を算出した。

-入手方法: ArtemisのL4板データをAWSのEC2経由で取得。

<img src="../../../photo/artemis-logo.png" alt="Artemis" width="64">

Artemis URL  https://about.artemis.ai/

- 期間: 2026-05-04 15:53:28 UTC(板が両側そろった最初の瞬間)〜 2026-08-10 23:59:59 UTC
- 解像度: ns。最良気配 `(best_bid, best_ask, bid_sz, ask_sz)` が変化した瞬間ごと
- 規模: 板イベント 2,347 万 / 板レベル差分由来 4,180 万 / 注文 4.47 億 / 約定 596 万

まず読むもの → [METHODOLOGY.md](METHODOLOGY.md)(「予測力」の定義と限界。全レポート共通)

---

## 1. 何を算出したか

| # | 内容 | 規模 | レポート |
|---|---|---|---|
| 1 | マイクロプライス 全期間・全解像度 | 23,470,241 点 | [microprice_report.md](microprice_report.md) |
| 2 | OBI(L=1..10) 毎ティック | 41,803,773 点 × 10 | [obi_report.md](obi_report.md) |
| 3 | 4 系列(スプレッド / 板インバランス / VOI / 約定インバランス) | 23,470,241 点 | [features_report.md](features_report.md) |
| 4 | 価格リターンの自己相関 6 サンプリング × 3 価格系列 × ラグ 20 | 84.96 億グリッド点 | [acf_report.md](acf_report.md) |
| 5 | 回帰 $r_{t+i} = \alpha + \beta m_t + \varepsilon$ 40 地平 × 2 仕様 + 統計監査 | 23,458,514 点 | [regression_report.md](regression_report.md) |
| 6 | 注文フロー強度 $\lambda_{\mathrm{cancel}}$ / $\lambda_{\mathrm{new}}$(Bid・Ask 別) | 4.47 億注文 | [flow_report.md](flow_report.md) |
| 7 | キャンセル 件数・数量(Bid/Ask 別・6 窓) | 3.40 億件 | [cancel_report.md](cancel_report.md) |
| 8 | QI(注文のキュー内位置) $Q_{\mathrm{ahead}}$ / $Q_{\mathrm{behind}}$ | 3.44 億注文 | [queue_qi_report.md](queue_qi_report.md) |
| 9 | Lasso 回帰 全 25 特徴量 $(t-1) \to r_t$・標本外検証 | 8,466,661 行 | [lasso_report.md](lasso_report.md) |
| 10 | 分散と相対 MSE の関係 33 分割・6 解像度 | 33 ブロック | [variance_report.md](variance_report.md) |
| 11 | 1 分刻みの相対 MSE(粒度依存の検証) | 119,169 分 | [minute_report.md](minute_report.md) |
| 12 | 1 分フォールドの Lasso 9 変数・学習量の閾値 | 125,499 分 | [minute_lasso_report.md](minute_lasso_report.md) |
| 13 | 新特徴量 11 個の追加(板の傾き・参加者集中・棄却率) | 8,451,761 行 | [panel_v2_report.md](panel_v2_report.md) |
| 14 | book_slope の最適粒度(深さ帯 × 予測地平のスイープ) | 864 万点 × 10 日 | [book_slope_grain_report.md](book_slope_grain_report.md) |
| 15 | $\beta_A$ / $\beta_B$ の推定と t 検定(1s・25bp・3 系統の推論) | 8,451,761 行 | [slope_regression_report.md](slope_regression_report.md) |
| 16 | HFT 戦略化のロードマップ(手数料・逆選択の実測) | メイカー約定 154 万件 | [hft_roadmap.md](hft_roadmap.md) |
| 17 | キャンセル猶予時間の実測(反応時間 $\delta$ を与えた選別・プラセボつき) | メイカー約定 150 万件 | [cancel_latency_report.md](cancel_latency_report.md) |
| 18 | 約定確率モデル + イベント駆動バックテスタ | 3,718 万注文 / 10 日 | [fill_prob_backtest_report.md](fill_prob_backtest_report.md) |
| 19 | 閾値最適化・気配改善・在庫制約・再較正 | 3,718 万注文 / 10 日 | [gate_optimization_report.md](gate_optimization_report.md) |
| 20 | book_slope 予測モデル 全 17 段階(Lasso→NW→WF→Bootstrap→FDR→コスト) | 699 万行 / 91 日 | [slope_model_report.md](slope_model_report.md) |
| 21 | キュー順位ごとの約定確率・約定後の価格変動・分布 | 2.1 億注文 / 336 万約定 | [queue_position_report.md](queue_position_report.md) |
| 22 | 前の行列は約定で消えたか取消で消えたか(逆選択の機構) | 103 万約定 / 27 日 | [queue_clearing_report.md](queue_clearing_report.md) |
| 23 | 前方取消フローによる引き規則(段階① オフライン検証) | 16.4 万約定 / 27 日 | [pull_rule_report.md](pull_rule_report.md) |
| 24 | 同上を深さ × 売買 × 置き場所で層別(16 層 × 60 セル) | 27 日 | [pull_rule_strata_report.md](pull_rule_strata_report.md) |
| 25 | 絶対量閾値 × 99 日 × 体制別(段階① 確定) | 69 万約定 / 84 日 | [pull_rule_99_report.md](pull_rule_99_report.md) |
| 26 | 規模(容量)の測定 — 機会・置ける量・競合 | 20 日 / 成熟期 | [capacity_report.md](capacity_report.md) |
| 27 | 短期価格予測との組合せ | 41 日 | [combine_signals_report.md](combine_signals_report.md) |
| 28 | OBI のレベル別分解 — 各段の情報量 | 91 日 | [obi_levels_report.md](obi_levels_report.md) |
| 29 | MLOBI — レベル別重み付けした多段不均衡 | 評価 71 日 | [mlobi_report.md](mlobi_report.md) |
| 30 | 加法スプライン回帰 — 各レベル OBI の非線形性 | 評価 71 日 | [mlobi_spline_report.md](mlobi_spline_report.md) |
| 31 | L10 book_slope スプライン × 7 地平 | 評価 30 日 | [slope_spline_report.md](slope_spline_report.md) |
| 32 | バックテスタ v3 — 両側同時への統合 | 26 日 | [backtester_v3_report.md](backtester_v3_report.md) |
| 33 | 規模の測定(部分約定つき) | 30 日 | [capacity_v3_report.md](capacity_v3_report.md) |
| 34 | テスト資金 100 USD の $E[\mathrm{PnL}]$ と APY | 30 日 + 最悪日 | [test_capital_report.md](test_capital_report.md) |
| 35 | ★分散・テールリスクと統計監査 | 30 日 | [tail_risk_audit_report.md](tail_risk_audit_report.md) |
| 36 | ★確率微分方程式による 1,000 本シミュレーション | 30 日で較正 | [sde_report.md](sde_report.md) |
| 37 | ★トレード頻度を落とすべきか(建て直し規則の測定) | 30 日 × 18 構成 | [rest_policy_report.md](rest_policy_report.md) |
| 38 | ★Hyperliquid HFT の現実的制約の全数調査 | 公式仕様 + live API + 実測 30 日 | [hft_constraints_report.md](hft_constraints_report.md) |
| 39 | Lighter で実装できるか(A / B / C の 3 案) | 仕様 + live API 235 市場 | [lighter_feasibility_report.md](lighter_feasibility_report.md) |
| 40 | ★既存戦略を棄却し遅延前提で作り直すべきか | 30 日 × 8 + 初期 15 日 | [latency_first_report.md](latency_first_report.md) |
| 41 | まだ算出していない特徴量の一覧 | 24 日 + 全列照合 | [feature_gaps_report.md](feature_gaps_report.md) |
| 42 | ★TWAP サンドイッチの経済的有意性 / funding × 短期リターン OLS | TWAP 4,150 件 / OLS 169 万観測 | [twap_funding_report.md](twap_funding_report.md) |
| 43 | ★期間体制 5 通り × OBI 5 種のスプライン回帰(knot は分位配置 + 日単位 GroupKFold) | 98 日 845 万行 × 25 セル | [obi_spline_regime_report.md](obi_spline_regime_report.md) |
| 44 | ★HL と Binance の DRAM リードラグ | 85 日重複 / 810 万約定 | [leadlag_binance_report.md](leadlag_binance_report.md) |
| 45 | ★Hyperliquid の最低遅延とテールレイテンシ(実測) | live + 99 日 5,355 万本 | [latency_report.md](latency_report.md) |
| 46 | ★東京 AWS 移行判断(実測 3 点) | 30 日 × 16 セル + live | [colocation_decision_report.md](colocation_decision_report.md) |
| 47 | ★全期間・全約定のテイカー分類(Aggressive Buy/Sell、真値) | 99 日 / 596 万取引 | [taker_classification_report.md](taker_classification_report.md) |

各報告の結論と数値は §2、訂正の経緯は §3 にまとめてある。

### 主な数式

マイクロプライス

$$
\mathrm{MP} = \frac{p^{b} q^{a} + p^{a} q^{b}}{q^{b} + q^{a}}
            = m + \left(I - \tfrac{1}{2}\right) s ,
\qquad I = \frac{q^{b}}{q^{b} + q^{a}}
$$

$p^{b}, p^{a}$ は最良買い・売り気配、$q^{b}, q^{a}$ はその数量、$m$ は mid、$s$ はスプレッド。

板インバランス

$$
\mathrm{OBI}(L) = \frac{\sum_{i \le L} q^{b}_{i} - \sum_{i \le L} q^{a}_{i}}
                      {\sum_{i \le L} q^{b}_{i} + \sum_{i \le L} q^{a}_{i}} ,
\qquad
\mathrm{OBI}(1) = 2I - 1 \iff \mathrm{MP} = m + \tfrac{1}{2} \mathrm{OBI}(1) s
$$

約定インバランス

$$
\mathrm{OI} = \frac{V^{\mathrm{buy}} - V^{\mathrm{sell}}}{V^{\mathrm{buy}} + V^{\mathrm{sell}}}
$$

VOI

$$
\mathrm{VOI}_{t} = \Delta Q^{b}_{t} - \Delta Q^{a}_{t}
$$

$\Delta Q$ は $t-1$ と $t$ の最良気配の価格と数量から決める。

キュー位置

$$
\mathrm{QI} = \frac{Q_{\mathrm{ahead}} - Q_{\mathrm{behind}}}{Q_{\mathrm{ahead}} + Q_{\mathrm{behind}}}
$$

フロー強度

$$
\lambda^{\mathrm{side}}_{\mathrm{cancel}} = \frac{V^{\mathrm{side}}_{\mathrm{cancel}}}{\Delta t} ,
\qquad
\lambda^{\mathrm{side}}_{\mathrm{new}} = \frac{V^{\mathrm{side}}_{\mathrm{new}}}{\Delta t}
$$

リターン

$$
r_{t+1} = \log P_{t+1} - \log P_{t}
$$

---

## 2. 主な結果

市場の姿 — 上場直後と成熟後で別の市場と言ってよいほど違う。

| | 2026-05-04(初日) | 2026-08-10(最終日) |
|---|---|---|
| スプレッド(時間加重) | 142.7 bp | 0.46 bp |
| BBO 更新 | 4,930 回/日 | 345,789 回/日 |
| 全期間 | 更新間隔 中央値 136 ms、1 パーセンタイル 222 µs | |

予測力の一覧(1 秒先の中値変化との相関、日次推定の中央値。定義は METHODOLOGY.md)

| 指標 | 粒度 | 相関 | 符号の一貫性 |
|---|---|---|---|
| 約定インバランス OI | 1 秒窓(次窓) | +0.077 | 93 / 97 日 |
| OBI(L=2)(スプレッド<1bp) | イベント | +0.132 | 88 / 97 日 |
| OBI(L=1)(全体) | イベント | +0.072 | 98 / 99 日 |
| 板インバランス depth_one_level | イベント | +0.079 | 96 / 98 日 |
| VOI(t−1/t 最良気配) | イベント | +0.064 | 92 / 98 日 |
| キャンセル件数差(ask−bid) | 100ms 窓 | +0.036 | 85 / 99 日 |
| 新規注文強度差 | 1 秒窓 | +0.017 | 72 / 99 日 |
| マイクロプライス乖離 m | イベント | +0.032(プール) | 日次 β が 94 / 99 日で同符号 |

どれも単独では執行コストに届かない。 予測幅は 0.015〜0.24 bp、
ハーフスプレッドは平均 1.66 bp(`regression_report.md` §6)。
用途はパッシブ執行の傾け・逆選択回避・他シグナルとの合成。

板の傾き `book_slope_diff` が最強の単一特徴量(全 99 日で相関 −0.135、
98/98 日で同符号、既存 25 変数への $R^2 = 0.05$ = 95% が新情報)。
これを加えると標本外の日次相関は 0.203 → 0.243(+20%)([panel_v2_report.md](panel_v2_report.md))。

特徴量を組み合わせると単独の 1.35 倍(Lasso、標本外 28 日):
25 変数のうち 9 変数が残り、日次相関の中央値 0.192(28 日すべて正)。
最良の単一特徴量 `obi_2` は 0.142。プラセボ(x を日内でずらす)は 0.000 で、
ルックアヘッドが無いことを確認済み。立ち上げ期を学習に含めると逆に成績が落ちる
(市場の体制が変わっているため)。

構造的な発見

- 板の階層で最適な深さが変わる: スプレッドが狭いときは L=2 前後、
  10bp 超では L=10 が最良。1 段目だけでは足りない局面がある
- OBI の情報は 1 段目に集中([obi_levels_report.md](obi_levels_report.md)):
  レベル別 $R^2$ は 1 段目を 1 として 2 段目 0.305 / 5 段目 0.054 / 10 段目 0.026。
  重複を除いた増分では 2 段目 0.177 / 10 段目 0.012。10 段全部でも 1.54 倍。
  累積 OBI(L) は深くするほど下がる(1.000 → 0.467)= 1 段目の信号が希釈される。
  ただし 10 段目でも符号は 66/91 日で一致し増分は 91/91 日で正 —「小さいが本物」。
  深い板を使いたいなら OBI ではなく `book_slope`(形 vs 左右の比)
- キャンセルは束で入る: 1ms と 10ms で窓数がほぼ同一(41,110,727 vs 41,109,343)。
  約定も同じ構造で、1ms まで刻んでも情報は増えない
- 中値は継続、マイクロプライスは反転: 同じ板から作った 2 系列で ρ₁ の符号が逆
  (1ms で +0.013 / −0.012、99 日中 99 日・98 日で符号一致)
- キュー位置は約定率を 5.2 倍動かす: 先頭 1.579% → 1k–2.5k の後ろ 0.302%
- 比より差: `cancel_ratio` は符号が消える(47/99 日)が、差 `ask − bid` は残る(80/99 日)。
  比は水準を捨てるため

戦略化の検証(成熟期 10 日・[gate_optimization_report.md](gate_optimization_report.md))

- 効果の本体は「どこに置くか」であって「いつ引くか」ではない。
  スプレッドが広いとき最良気配の内側 1 ティックに置くと +1,521〜+2,162 bp/日、
  6 通りのゲート全部で正、8〜9/10 日で正(符号検定 p = 0.011)。
  理由は逆選択 — 最良気配の行列の最後尾は「板が薙ぎ払われるとき」しか約定しない
- シグナルで引くゲートは確立していない。在庫上限 5 でだけ効き(6/10 日、符号検定 p = 0.377)、
  上限 2 と 20 では逆効果。36 通りの格子から出た偶然と判断した。
  立てた仮説(在庫制約が内点最適を作る)は反証された
- 在庫制約は必須。入れないと建玉 133 単位・標準偏差 35,629 の方向性の賭けになる
  (日次シャープ 0.49)。マーケットメイクではない
- 約定確率の再較正は有効。等調回帰で ECE −52% / MCE −69%、AUC は 0.859 のまま不変。
  ただし学習期→検証期で約定率の水準が +9.2% 動くので、較正関数は継続更新が要る
- 方策選択は標本外で 8/8 日黒字(+1,624 bp/日)。選択バイアスは −11%。
  ただし選ばれた方策は 8 回すべて「気配改善あり」で、ゲートは毎回変わった

book_slope 予測モデル(91 日・[slope_model_report.md](slope_model_report.md))

- OOS $R^2 = +0.0447$(ウォークフォワード 28 日、27/28 日で正、符号検定 p<0.0001)。
  プラセボ(説明変数を日内で +1 時間ずらす)は −0.00001 で完全に消える
- 予測力の 9 割は book_slope 系。slope 系のみ +0.0399 に対し統制 12 個で +0.0127。
  `book_slope_diff` 単独でも `m_bp` の 9 倍($R^2$ 0.0105 対 0.0011)
- Newey-West と日次推論で結論が割れる。NW は 25 変数すべてを FDR 通過させるが、
  日次では 6 個が落ちる(符号一致 29〜38/63 日 = コイン投げ)。日次を信じるべき
- 損益分岐コストは片道 0.2975 bp(無選別)〜 1.0119 bp(上位 1%)。
  実測のテイカーコスト 0.579 bp を超えるのは上位 5% の予測だけ(SR 0.53)
- 検証期間で候補を選ぶと全部入りが勝つ。OOS で良かった簡素モデルは採用できない
  (結果を見てからの選別になるため)

キュー順位(28 日・2.2 億注文・[queue_position_report.md](queue_position_report.md))

- 約定確率は単調減少(最良気配で先頭 7.45% → q>1000 で 2.32% の 3.2 倍)。
  ただし置き場所の効果 29 倍(内側 23.70% 対 1-5bp 外 0.82%)のほうが圧倒的に大きい
- 逆選択は非単調。先頭 −0.088 / 中間層(101-1k)+0.218〜+0.245 / q>1000 −0.134。
  「後ろほど不利」という事前予想は外れた。符号検定でも成立(19/24 日など)
- 交絡ではない。板の厚み・スプレッドで層別しても 6 層すべてで中間層が先頭を上回る
- 平均の差は裾から来る。中央値はどこも 0.00 で、深いキューほど分散が大きい
  (1.89 → 2.98)。中間層は上振れが厚いだけで、シャープで見れば優位は薄まる
- ★この市場ではキューが約定でなく取消で蒸発する。前に 1,634 単位あっても
  約定待ちは中央値 0.5 秒。順位を待つコストがほとんど無い
- 逆選択は 1 秒スケールで顕在化(100ms ではほぼゼロ、60 秒では全バケット負)
- 発注 1 件あたりの期待値では差がほぼ消える(先頭 +0.0108 / 中間層 +0.0115)。
  単価と回数のトレードオフが均衡している

行列の消え方(27 日・103 万約定・[queue_clearing_report.md](queue_clearing_report.md))

- ★「前が取消で消えた後の約定」が最も毒性が高い(−0.296 bp、27/27 日で負)。
  「大口が食い破った」ほう(−0.092 bp)より 3.2 倍悪い。事前予測と逆だった
- しかもこれが最多。前の行列の 77.2% は取消で消え、約定の 62.9% がこの形
- 解釈: 取消そのものが情報。前のメイカーが察知して引き、逃げ遅れた者が掴まされる。
  この区分は粗利も最低(0.112)で、約定時点で既に中値が寄ってきている
- 偏回帰で `clear_ratio` は `q_ahead` の 2.2 倍(+0.1202 / 27 日全部 vs +0.0536 / 22 日)。
  報告 21 のキュー順位の非単調性は、相当部分が「消え方」の代理だった
- この指標は予測に使えない(約定後にしか計算できない)。実務には
  「直近 N ms の前方取消量」というリアルタイム代理指標が要る → 報告 23 で検証

引き規則の検証(27 日・[pull_rule_report.md](pull_rule_report.md))

- 上の代理指標を実装し、格子 180 通りを全件評価した(3 シグナル × 4 窓 × 5 閾値 × 3 反応時間)
- 1 約定あたりは明確に改善(+0.18 bp、25/27 日、20 セル全部で正)。
  ランダム対照を 22〜23/27 日で突破(偶然なら 1.35 日)。選別は情報を持つ
- 反応時間 200ms でも劣化しない。実装可能な範囲
- ★しかし総額では「そもそも出さない」に勝てない。
  出し続ける −922.9 bp/日 → 引く +4.6 bp/日 → 出さない 0。14/27 日、p=0.50
- 原因は構造的。約定するには前の行列が消えている必要があるので
  シグナルはほぼ必ず発火し(引いた率 73〜88%)、判別力は
  「発火が約定より δ 以上前だったか」という時間差だけから来ている
- 機構の解釈は部分的にしか支持されない。`exec` 単独は無力(取消成分は必要)だが、
  `total` が `cancel` を上回るため「取消は情報」だけでは説明できない

層別の再検証(16 層 × 60 セル・[pull_rule_strata_report.md](pull_rule_strata_report.md))

- 規則なしで黒字の層はゼロ。報告 22 §4 の「深い行列は黒字」は 1 約定あたりの値で、
  総額に直すと有意性が残らない(100<q≤400 で −4.4 bp/日、11/24 日、p=0.73)
- 引く規則は 5 層で「出さない」を上回る。最良は
  `最良気配 × 前に 100〜400 単位 × 両側` で +88.4 bp/日(18/24 日、p=0.0113)、
  引いた率 55% — 16 層で唯一「極端に引かない」層
- ★深さで発火率が両極に振れる。q≤20 で 91%、q>400 で 6%。
  シグナルを `flow/q_ahead` と深さで正規化した副作用
- Bid と Ask で結論が割れる。Ask は有意(p=0.0053)だが Bid は p=0.105。
  方向は一致するので検出力不足だが、事前基準では未成立
- ★多重比較(960 通り)を厳密に当てるとどの層も残らない。
  一方ランダム対照は 14〜17/27 日で突破(偶然なら 1.35 日)。
  「選別は情報を持つが、多重比較に耐える形で総額の優位に変換できていない」が正確な状態
  → 報告 25 で閾値を絶対量に変え 99 日へ拡張し、確定

確定版(84 日・69 万約定・[pull_rule_99_report.md](pull_rule_99_report.md))

- ★事前基準を満たす層が 1 つ。`最良気配 × 前に 400 単位超` で
  `exec/10ms/abs5`(前方の約定が直近 10ms で 5 単位超なら引く)→
  引く価値 +25.5 bp/日(手数料控除後、32/41 日、p=0.0002)、
  引いた率 13%。セルを固定した頑健性検定で両体制 × 両サイドの 7 通りすべてが有意。
  Bid(p=0.0009)・Ask(p=0.0401)の両サイド、立ち上げ期・成熟期の両体制で成立、
  Bonferroni(0.0031)も両側と Bid が通過
- ★主判定に欠陥があった。「残り合計 > 0」はベースラインが黒字の層で自動的に成立する。
  比較対照を (a) 出す価値 / (b) 引いて出す価値 / (c) 引く価値 に分け、(c) を主判定に変更
- ★結論が 2 回動いた。同じ層の「出す価値」が 13 日 +142.1 → 27 日 −4.4 → 84 日 +93.3。
  27 日での「誤りだった」という訂正自体が検出力不足による誤りだった。
  体制別に見ると立ち上げ期の現象(p=0.0008)で、成熟期では有意でない(p=0.072)
- 絶対量閾値で発火率が実務帯に入った(割合版は 55〜91% に張り付き → 絶対量で 2〜66%)
- 品質検査を機械化。`q_ahead_median > 1000` で 2026-08-10 を自動検出(中央値 7,541)
- ★実際に効いているのは「取消」ではなく「約定」の検知。同じ層で
  `exec` +25.5(32/41 日、p=0.0002)に対し `cancel` は +16.8(23/41 日、p=0.266)で有意でない。
  規則の中身は「前で誰かが食べ始めたら逃げる」であり、
  報告 22 の「取消そのものが情報」とは別の現象を捉えている可能性が高い

### 2026-08-26 図示した 4 件(データは既存・作図のみ追加)

これまで本文の表だけだった 4 つの結果に図を付けた。数値は既存の `data/DRAM/*.json` から
そのまま描いており、再計算していない。

| 図 | 結果 | 何が見えるか |
|---|---|---|
| [46](../../../charts/xyz_DRAM_46_taker_classification.png) | テイカー分類(99 日 596 万取引) | 件数は買い越し 50.74% なのに想定元本は売り越し 49.11% と符号が逆転する。原因は約定サイズの非対称(日次比の中央値 1.069)。符号の自己相関はラグ 20 でも 0.127 と長期記憶。tick rule の正答率 66.46% に対し L4 の真値は 100% |
| [47](../../../charts/xyz_DRAM_47_latency_tail.png) | レイテンシの裾 | 更新間隔は中央値 106ms でも p99 は 2,055ms。価格が 1 ティック動く直前に反応できなかった時間は中央値 118ms・p99 1,208ms で、報告 44 の信号地平 500ms を 100 回に 1 回超える |
| [48](../../../charts/xyz_DRAM_48_leadlag_binance.png) | Binance → HL リードラグ | CCF の山は −0.25 秒(Binance 先行)、85 日中 83 日で同符号。プラセボは平坦(最大 0.0006)。3 つの時間解像度すべてで再現 |
| [49](../../../charts/xyz_DRAM_49_obi_spline_regime.png) | OBI × 期間体制 | 線形 R² が体制 1 → 5 で 23 倍(0.00087 → 0.01958)。半スプレッドが 8.14bp → 0.29bp と縮むのと同時に起きており、非線形性は最初から在ったのではなく現れた |

図を作る過程で気づいた粗も直した: 図 48 は当初 5000ms 刻みを重ねていたが、
±10 秒窓でも点が 5 個しか入らず曲線として読めないうえ縦軸を歪めるので外した。
図 46 のサイズ非対称は当初「全期間プールの平均比 6.7%」を出していたが、
報告本文の検定は日次比の中央値に基づくため 6.9% に揃えた。

---

## 3. 訂正の履歴(重要)

分析の途中で自分の誤りを 4 件見つけて訂正した。数値を引用する際は最新版を使うこと。

| 日付 | 内容 | 影響 |
|---|---|---|
| 08-14 | クロス状態(全体の 0.05%)の混入 | $\beta$ が 16 倍変わっていた(1s で 0.517 → 0.033)。<br>二乗和の 47% を 1 日が占めていた。`regression_report.md` §2 |
| 08-14 | 経済的有意性を体制平均の $\beta$ で判定 | 狭スプレッド帯では $\lvert\beta\rvert \gt 1$ で前提が崩れる。<br>「0 件」→ 体制別に再計算。§6 |
| 08-14 | 検定統計量の記号と参照分布 | 時刻添字 t と衝突。p 値は正規から算出していた<br>→ `z` と `t₉₈` に分離、クラスター小標本補正を追加。§1 |
| 08-14 | OBI の相関だけプール値だった | 日次中央値に統一。最良の段数が変わった<br>(<1bp で L=1 → L=2、3–10bp で L=5 → L=3) |
| 08-14 | 1 秒グリッドは無汚染、と書いていた | 誤り。`enforce_uncrossed` の resync が秒境界に載る。<br>0.176% の行が $\sum y^{2}$ の 79.3% を占めていた。`variance_report.md` §2 |
| 08-15 | `queue_qi` の 2026-08-10 が壊れていた | `ts_close` に null がある日は polars の `to_numpy()` が float64 を返し、<br>ns 時刻(~1.79e18)が 2^53 の 199 倍で精度を失う。`q_ahead_open` が 1,135 倍に膨張。<br>99 日中この 1 日だけ。報告 21 を 28→27 日で再計算(結論は不変)。`queue_clearing_report.md` §6-2 |
| 08-15 | 整合性検査を中央値だけで報告していた | 27 日が正常なら中央値は正常のままで、1 日の破綻(0.006)を見逃した。<br>最悪値を併記するよう修正。検査を作っても集計の仕方で無効になりうる |
| 08-16 | ★メイカー手数料を控除していなかった | 指摘を受けて発覚。`pull_rule` / `queue_adverse` / `queue_clearing` が<br>`net = gross + adv` で 0.088bp/約定 を引いていなかった(`backtester_v2` と<br>`slope_model_fit` は正しく控除済みで影響なし)。効き方が逆向き —<br>「出す価値」は約定回数の多い層ほど削られ(+224.1→−37.9 など符号反転)、<br>「引く価値」は引いた分が浮くので増える(+22.0→+25.5)。<br>報告 21・22・25 の「正味」列を全て差し替え。`pull_rule_99_report.md` §1 |
| 08-16 | 主判定がベースライン依存で自動成立していた | 「施策後 > 0」は元が黒字の層で勝手に通る。実際は施策が損なのに合格していた。<br>対照を (a) 出す価値 /(b) 引いて出す価値 /(c) 引く価値 に分け (c) を主判定に |

統計的正当性の監査(プラセボ・順位相関・標本外・異質性・有効標本サイズ)は
`regression_report.md` §9 にまとめてある。プラセボは通過(説明変数を日内でずらすと $\beta \approx 0$)、
外れ値依存もなし(順位相関で 95/99 日が同符号)。一方で
「単一の $\beta$ が存在する」という主張は支持されない($I^2 = 99.7$ %、次日の予測区間が 0 を跨ぐ)。

---

## 4. リポジトリの中身

| パス | 中身 |
|---|---|
| `*_report.md` | レポート 46 本 |
| `METHODOLOGY.md` | 予測力の算出方法(全レポート共通) |
| `charts/` | 図 49 点 |
| `data/DRAM/oi_grid/step=*/dt=*/` | 窓ごとの約定インバランス(1ms〜60s の 6 通り・94MB) |
| `data/DRAM/microprice_1m.csv` / `data/DRAM/obi_1m.{csv,parquet}` | 1 分グリッド(各 14 万行) |
| `data/DRAM/*_daily.csv` | 日次サマリ(99 行 × 6 種) |
| `data/DRAM/*.json` | レポート中の全数値の出所 |
| `scripts/` | 生成コード 132 本 |

### 入っていないもの(サイズのため。すべて再生成可能)

| パス | サイズ / 行数 | 復元 |
|---|---|---|
| `data/DRAM/microprice/` | 888 MB / 23,470,241 | `scripts/build_microprice_dram.py` |
| `data/DRAM/microprice_1s.parquet` | 210 MB / 8,496,221 | 同上 |
| `data/DRAM/obi/` | 3.1 GB / 41,803,773 | `scripts/build_obi.py` |
| `data/DRAM/features/` | 680 MB / 23,470,241 | `scripts/build_features.py` |
| `data/DRAM/flow_grid/` | 1.5 GB | `scripts/build_flow_grid.py` |
| `data/DRAM/cancel_grid/` | 2.7 GB | `scripts/build_cancel_grid.py` |
| `data/DRAM/queue_qi/` | 2.2 GB / 343,703,060 | `scripts/build_queue_qi.py` |

GitHub の 1 ファイル 100MB 制限のため除外している。手元
`C:\Users\ii562\Downloads\Memory\data\DRAM\` には実体がある(2026-08-29 の統合で移設)。

---

## 5. 再現手順

```bash
# 入力(作業バケットの L2 テーブル)
aws s3 sync s3://$WORK_BUCKET/l2/bbo/     data/l2_v99/bbo/     --profile hl-artemis-ro --region us-east-1  # 430MB
aws s3 sync s3://$WORK_BUCKET/l2/book_px/ data/l2_v99/book_px/ --profile hl-artemis-ro --region us-east-1  # 3.7GB
# lifecycle(8.5GB)と fills は同バケットの l2/lifecycle, fills/ から

uv run python scripts/build_microprice_dram.py <出力先>   # 4 分   マイクロプライス
uv run python scripts/microprice_report.py           #        診断 + 図 1-7
uv run python scripts/build_obi.py <出力先>          # 6 分   OBI(1..10)
uv run python scripts/obi_analyze.py                 #        図 8-11
uv run python scripts/build_features.py <出力先>     # 6 分   4 系列
uv run python scripts/features_summary.py
uv run python scripts/build_oi_grid.py <出力先>      # 2 分   窓ごとの OI
uv run python scripts/oi_grid_summary.py
uv run python scripts/regress_micro.py               # 15 分  回帰 40 地平
uv run python scripts/audit_regression.py            # 10 分  統計監査
uv run python scripts/regress_charts.py              #        図 12-15
uv run python scripts/audit_chart.py                 #        図 16
uv run python scripts/return_acf.py                  # 25 分  自己相関
uv run python scripts/acf_charts.py                  #        図 17-19
uv run python scripts/build_flow_grid.py <出力先>    # 20 分  フロー強度
uv run python scripts/flow_grid_summary.py
uv run python scripts/build_queue_qi.py <出力先>     # 12 分  キュー位置
uv run python scripts/queue_qi_summary.py && uv run python scripts/queue_qi_chart.py   # 図 20
uv run python scripts/build_cancel_grid.py <出力先>  # 15 分  キャンセル 6 窓
uv run python scripts/cancel_grid_summary.py && uv run python scripts/cancel_grid_chart.py  # 図 21
uv run python scripts/build_panel.py <出力先>        # 25 分  1 秒パネル(全特徴量)
uv run python scripts/lasso_analysis.py              # 15 分  Lasso + プラセボ
uv run python scripts/lasso_univariate_oos.py && uv run python scripts/lasso_chart.py       # 図 22

# 2026-08-26 追加(既存 data/*.json から作図するだけ。再計算は不要)
uv run python scripts/taker_chart.py        # 図 46  テイカー分類
uv run python scripts/latency_chart.py      # 図 47  レイテンシの裾
uv run python scripts/leadlag_chart.py      # 図 48  Binance リードラグ
uv run python scripts/obi_spline_chart.py   # 図 49  OBI スプライン × 期間体制
```

ルックアヘッドバイアスは厳禁(`CLAUDE.md` に規定)。特徴量が確定する時刻は
目的変数の期間の開始時刻以下、asof は backward のみ、標本外評価は時間ブロック分割、
前処理の統計量は訓練期間だけから推定する。`lasso_report.md` §1 に本件での適用を記載。

---

## 6. ★ クロス状態は必ず除外すること

板の再構成には `best_ask < best_bid` となる一過性の状態が全ティックの 0.05%(11,628 件)含まれる
(`is_crossed` 列で判別)。件数は僅かだが mid が板の外に出て $\lvert \mathrm{MP} - m \rvert$ が最大 7,349 bp に飛ぶため、
混ぜたまま回帰すると二乗和の 47% を 1 日が占め、係数が 16 倍変わる。
本リポジトリの集計はすべて除外済み。詳細は `regression_report.md` §2。

【2026-08-14 訂正】 初版は「クロス状態は 1 秒境界を跨がないので 1 秒グリッドは
構造的に汚染されない」と書いていたが不正確だった。クロス状態そのものは境界を跨がないが、
その解消(`enforce_uncrossed` の resync)がちょうど秒境界に載る(`ts mod 1s == 999999999`)。
1 秒グリッドはその点を拾うため、片側が退去した直後の異常な mid が入る。
全ティックの 0.020%(4,792 件)だが、1 秒パネルでは 0.176% の行が $\sum y^{2}$ の 79.3% を占め、
うち 2 行だけで約 71%。1 秒グリッドを使う分析では必ず除くこと(`variance_report.md` §2)。

---

## 7. データの出所について

指示は「`Hyperliquid-L4data-DRAM-USDC-v1` リポジトリ内の xyz:DRAM データを使う」だったが、
あのリポジトリにデータは入っていない。`.gitignore` が `data/DRAM/` `wal/` `snapshots/`
`*.parquet` を除外しており、実体はライブ収集機のローカルにしかない
(手元のクローンにあるのは空の `data/DRAM/manifest.db` = 0 行のみ)。

そのため、同一市場 xyz:DRAM の板を Artemis 公開バケット(`node_order_statuses` /
`node_fills`)から注文単位で再構成した L2(hl-l4-pipeline v99、99 日分)を入力に使った。

| | ライブ収集(あのリポジトリ) | 本分析(Artemis 経由) |
|---|---|---|
| 取得元 | WebSocket `book_diffs` を直接記録 | 取引所ノードのイベントログを再生 |
| 時刻 | 受信時刻 + 取引所時刻 | 取引所イベント時刻(UTC ns) |
| 期間 | 収集開始以降 | 2026-05-04(上場)〜 2026-08-10 |

ライブ収集側のデータが手に入れば、同じスクリプトが同じ形の入力
(`ts, best_bid, best_ask, bid_sz, ask_sz`)で走るので、突合と延長ができる。

---

計算に要した AWS 費用: EC2 \$25.05 | S3 ≈\$1.87 | 累計 ≈\$26.92 / 予算 \$60
(egress は L2 テーブルの取得 4.15GB のみ。EC2 は停止中)
