# Uniswap 分析索引

[← ホーム](../../README.md)

| 項目 | 内容 |
|---|---|
| プロトコル | [Uniswap v3](https://uniswap.org/)。集中流動性型 AMM |
| チェーン | Arbitrum One(主)。Ethereum / Base / Avalanche / BSC は生ログ取得済み |
| データの出所 | **オンチェーンのログのみ。費用 $0**(公開 RPC、API キー不要)。作業場所は `E:\Uniswap-arb` |
| 状態 | **着手済み・レポート移設待ち** |

---

## この取引所を他と分けて扱う理由

板のある取引所(Hyperliquid / Lighter / Derive / Pendle Boros)と**市場の作りが
根本的に違う**ので、同じ指標でも意味が変わる。

| | CLOB | **Uniswap v3(AMM)** |
|---|---|---|
| 気配 | 観測量。スプレッドは変動する | **スプレッド = 手数料(既知の定数)** |
| 板の数量 | 表示分のみ(隠れ注文は見えない) | **閉形式で厳密**。近似がゼロ |
| 価格の刻み | 取引所が決めるティック | **tickSpacing = 手数料ティアに紐付く**(0.05% なら 10bp) |
| 待ち行列 | 存在する | **存在しない** |
| 板の非対称(OBI) | 主要な予測変数 | **縮退している**(標準偏差が CLOB の 1/15) |
| 参加者の行動 | 口座が見えない | **`Mint`/`Burn` の `owner` で全部見える** |
| 時間の粒度 | ns | **ブロック**(Arbitrum 0.251 秒)。同一ブロック内の順序は MEV が決める |

---

## 現状の成果(`E:\Uniswap-arb` にあるもの。ここへ移設予定)

| 内容 | 場所 |
|---|---|
| 板の再構成(Swap 720 万件と**完全一致・誤差 0**) | `E:/Uniswap-arb/README.md` |
| mid price の厳密生成(深さ 4,328 断面で相対誤差 0) | 同 §6 |
| **手数料ティア間のリードラグ** | `E:/Uniswap-arb/reports/leadlag_fee.md` |
| **AMM → CLOB 型 L2 の変換** | `E:/Uniswap-arb/reports/l2_from_amm.md` |
| 5 チェーン・14 プールの生ログ(395,079 件) | `E:/Uniswap-arb/data/mc_logs/` |

---

## レポート

*(移設後にここへ並べる)*
