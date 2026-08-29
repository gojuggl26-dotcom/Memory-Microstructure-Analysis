# 報告 38 の戦略は Lighter で実装できるか

方法: 公式 API docs + live API(2026-08-18 直接取得、235 市場)+ WebSocket 実接続によるストリーム粒度の確認 + SDK ソース確認。バックテストは行っていない(理由は §4 — 材料が存在しない)
作成日: 2026-08-18

---

## 0. 結論

| 戦略 | 判定 | 一言 |
|---|---|---|
| A. 置き型 MM(k=2 + Q≥5 + modify) | △ 実装は完全に可能、採算の根拠がない | Premium 口座で機械的には全部できる。しかし maker 手数料 0.40bp が、我々が DRAM で実測した達成可能スプレッド獲得(0.27〜0.44bp)と同額 — 数字を移植する限り赤字 |
| B. 上場初期ライフサイクル MM | △〜○ 唯一、制度と噛み合う | 月 7〜26 銘柄の新規上場が実在。初期の板は低頻度なので、ゼロ手数料 Standard 口座の人工遅延 200ms・低レート枠でも足りる見込み。ただし初日スプレッドの実測値がゼロ件 — 録画してから |
| C. トリガー密度・情報系 | × 不可能 | 公開データに注文単位の情報が存在しない(実測で確認)。トリガー注文・キュー位置・ウォレットはすべて不可視 |

メタ結論: 我々の優位は戦略アイデアではなく「L4 で測ってから動く能力」であり、
Lighter はその測定基盤を没収する。zk で「マッチングの正しさ」は検証できるが、
「板の中身」は Hyperliquid のノード配布(L4)より遥かに粗い。
移るなら、38 本のレポートで作った定量的裏付けをほぼ全部捨てて始めることになる。
唯一の例外が B — 録画(BBO+trades)→ 測定 → 参入の順で 1〜2 週で新規に立ち上げられる。

---

## 1. Lighter の制約構造 — Hyperliquid と正反対の思想

Hyperliquid は速度をオークションで売る(優先手数料)。Lighter は速度を会員制で売る:

| | Standard(既定) | Premium(opt-in) |
|---|---|---|
| maker / taker 手数料 | 0 / 0 | 0.0040% / 0.0280%(LIT 50 万枚ステークで −30% → 0.0028%/0.0196%) |
| 人工遅延: maker / cancel | 200ms | 0ms |
| 人工遅延: taker | 300ms | 140〜150ms |
| 発注系 tx 枠 | 60(重み付き)/分 — 読み方により実質 10〜150 tx/分 | 4,000/分(LIT ステークで最大 40,000/分) |

- 出典は公式 rate-limits ページ + 複数レビューの一致。sendTx は weight 6、バッチ(≤15 tx)も weight 6
- ほか: pending 16 件/市場、active 1,000 件/市場、WS は板 50ms バッチ・クライアント 200 msg/分
- 構造的な含意: Standard メイカーの取消(200ms)は Premium テイカー(150ms)に設計上必ず負ける。
  ゼロ手数料の見返りに「stale 気配を食われる側」に置かれる。我々の DRAM 実測(遅延 100〜300ms 帯で
  E[PnL] ≤ 0、報告 35)がそのまま当てはまる帯である
- シーケンサは単一・FIFO・フランクフルト(第三者情報。soft-confirm 5〜15ms)。
  優先手数料はなく、Premium 内の競争は純粋な速度勝負(コロケ先が東京 → フランクフルトに変わる)

## 2. 戦略 A(置き型 MM)の判定

機械的には全部ある(SDK ソースで確認): POST_ONLY / IOC / GTT、`modify_order`(建て直し 1 アクション化)、
バッチ 15 件、WS 発注、SL/TP/TWAP。

経済が成立しない(DRAM の実測数値を当てはめた場合):

```
δ*_Lighter(Premium) = f + βσΔ = 0.40(〜0.28) + 0.275 ≈ 0.56〜0.68 bp
達成可能な δ(DRAM 実測)          = 0.27(k=0)〜 0.44(k≥20) bp   → 全域で δ < δ*
δ*_Lighter(Standard) = 0 + βσΔ ≈ 0.275 bp — ただし遅延 200ms と tx 枠 10〜150/分が
  高頻度形を禁止し、許される低頻度形(k≥10 相当)の E[PnL] は DRAM 実測でゼロと区別できない
```

しかも βσΔ=0.275bp は DRAM の値であり、Lighter の主戦場(BTC/ETH 等、スプレッドはより薄く
競合はプロ)ではより厳しい方向に外れると考えるのが自然。「実装できるか」= Yes、
「やる根拠があるか」= No(現状の測定では)。

## 3. 戦略 B(上場初期 MM)の判定 — 唯一の現実的な入口

live API 実測(2026-08-18): 235 市場(perp 227)、新規上場は
2026 年に入って月 7〜26 銘柄(5 月 21、6 月 26、7 月 6、8 月は 13 日までに 7)。
直近の例: 8/13 に KIOXIA・WDC・AXTI・SOXS(半導体・メモリ系 4 銘柄 — DRAM 分析の
ドメイン知識が直接使える領域)。

制度との噛み合わせが A と正反対に良い:

| 制約 | 上場初期にはどうなるか |
|---|---|
| Standard の人工遅延 200ms | 初期の板は低頻度(DRAM 初日: BBO 更新 4,930 回/日 = 17 秒に 1 回)→ 無関係 |
| tx 枠 10〜150/分 | 建て直し頻度が低いので足りる見込み |
| 手数料 | 0/0 — growth mode の消滅リスク(HL の §3-2)に相当するものがない |
| 競合 | ゼロ手数料ゆえ圧縮は HL より速い可能性(未測定) |

ただし判断材料がゼロ件: Lighter の上場初日スプレッドを我々は 1 つも測っていない
(DRAM の 142.7bp は Hyperliquid の値)。次にやるべきことは実装ではなく録画:
WS(板 50ms + 約定 + BBO)を次の新規上場数件で記録し、初日〜初週のスプレッド・
出来高・圧縮速度を測る。録画系は既存 collector の流用で 1〜2 週。数字が出るまで参入しない。

## 4. 戦略 C(情報系)の判定 — データが存在しない

WebSocket に実接続して確認した(2026-08-18)。板チャネルの実ペイロードは

```json
{"channel":"order_book:1","order_book":{"asks":[{"price":"64141.2","size":"0.33516"}, ...]}}
```

価格レベルの集約のみ。注文 ID・注文単位イベント・ウォレットは公開されない。
トリガー注文はシーケンサ内部に保持され、公開ストリームに出ない。
ブロック/トランザクション参照系 API は外部から 403(WAF)。

したがって、この 99 日間の分析の主柱だった
キュー位置(3.44 億注文)・トリガー密度(L4 の (c))・ウォレット集中・注文ライフサイクルは
Lighter では原理的に測定不能。Artemis に相当する歴史アーカイブも見つからない。
C は不可能、A の較正(β・約定確率・キュー価値)も不可能。

## 5. Hyperliquid 対 Lighter — 報告 38 の 7 制約での対照表

| 制約軸 | Hyperliquid(xyz:DRAM) | Lighter |
|---|---|---|
| 遅延 | 実測 884ms(注文往復)、ブロック ~80ms、東京 | soft-confirm 5〜15ms、人工遅延 0〜300ms を階層で付与、フランクフルト |
| 速度の売り方 | オークション(優先手数料、bp 建て) | 会員制(Premium 手数料 + LIT ステーク) |
| レート制限 | 出来高連動(1 リクエスト/1 USDC)— 小口が失格 | 口座階層固定 — Standard が失格、Premium は潤沢 |
| 手数料 | growth mode 0.088bp(30 日ごとに消滅しうる) | Standard 恒久 0 / Premium 0.28〜0.40bp |
| 板の物理 | 取消がテイカーに同ブロック先行 | FIFO・zk で優先順位を証明(操作の余地なし) |
| 単一主体リスク | deployer(オラクル・halt) | シーケンサ運営 = Lighter 社そのもの(集中度はより高い。zk は「不正な約定」を防ぐが「止まる・締め出す」は防がない) |
| データ | L4 全量(ノード配布 + Artemis) | レベル集約 50ms のみ。L4 なし |
| 新規上場 | HIP-3、11 dex、増殖中 | チーム主導、月 7〜26 銘柄 |

## 6. 未確認事項(参入判断の前に要確認)

1. Standard の発注系スループットの正確な値(公式表の読み方で 10〜150 tx/分の幅。実測が要る)
2. 本番シーケンサの所在(フランクフルトは第三者情報・テスト網の値)
3. 日本からの API 取引の規約上の可否(ToS は制裁リスト除外のみ確認。明示的な国別条項は未発見)
4. Premium への昇格手続き(opt-in の具体、最低要件)
5. 上場初日のスプレッド・出来高(→ §3 の録画で測る)
6. 上場アナウンスの事前リード時間(WS の announcements チャネルで観測可能)

## 7. 検定・自己精査

- 本報告に新しい統計的主張はない。数値は (i) 公式仕様、(ii) live API 1 時点(235 市場・created_at)、
  (iii) WS 実接続 1 回、(iv) 既報(35/36/37/38)の実測値の引用のみ
- §2 の δ 比較は DRAM で較正した βσΔ を Lighter に外挿しており、市場が違えば無効。
  「移植すると赤字」は「Lighter で測り直すまで参入根拠がない」の意味であって、
  「Lighter の全市場で MM が赤字」という主張ではない
- 楽観・悲観の向き: §3 の「足りる見込み」は 2 点とも DRAM 初日の低頻度を Lighter に
  外挿した推定。録画実測で置き換えるまで参入判断に使わない

## 8. 出典

- [Rate Limits(公式)](https://apidocs.lighter.xyz/docs/rate-limits) / [WebSocket reference(公式)](https://apidocs.lighter.xyz/docs/websocket-reference) / [Get Started(公式)](https://apidocs.lighter.xyz/docs/get-started) / [lighter-python SDK](https://github.com/elliottech/lighter-python) / [Terms of Service](https://lighter.xyz/terms)
- 手数料・階層: [Lighterpedia: Fees](https://lighterpedia.com/guides/fees) / [PerpDexGuide: Lighter Fees](https://perpdexguide.com/lighter/fees/) / [CoinCodeCap Review](https://coincodecap.com/lighter-review) / [Hyperliquid vs Lighter](https://hyperliquidguide.com/compare/hyperliquid-vs-lighter)
- アーキテクチャ: [Lighter Whitepaper](https://assets.lighter.xyz/whitepaper.pdf) / [Datawallet: Lighter Explained](https://www.datawallet.com/crypto/lighter-explained) / [kkdemian: LIT 分析(フランクフルト・5〜15ms)](https://www.kkdemian.com/blog/lighter-lit-verifiable-perp-dex-zk-rollup-value-capture-gap)
- live 実測(2026-08-18): `GET https://mainnet.zklighter.elliot.ai/api/v1/orderBooks`(235 市場)/ `wss://mainnet.zklighter.elliot.ai/stream` の order_book:1 実ペイロード

---

計算に要した AWS 費用: EC2(stopped)\$25.12 | S3 ≈\$1.50 | 累計 ≈\$26.62 / 予算 \$60(停止閾値 \$48)
(本報告はローカル + 無料の公開 API のみ。EC2 は停止中)
