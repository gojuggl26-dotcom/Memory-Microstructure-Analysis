# DeFi Protocol Fundamentals 分析索引

[← ホーム](../../README.md)

| 項目 | 内容 |
|---|---|
| 対象 | Uniswap / Aave / Morpho ほか、このリポジトリで扱う DeFi プロトコル |
| 視点 | プロトコルを**企業に近い視点**で評価する(研究方針書 §17) |
| 状態 | **未着手** |

---

## この対象を他と分けて扱う理由

`reports/{uniswap, aave, morpho, ...}` の各索引は**個々のプロトコルの市場
マイクロストラクチャー**(板・金利曲線・清算など)を扱う。ここではその内側では
なく、**プロトコルという事業そのもの**を横断して評価する。したがって単一
プロトコルの深掘りではなく、複数プロトコル間の比較が主軸になる。

---

## 分析の柱(研究方針書 §17)

| 柱 | 内容 |
|---|---|
| **17.1 Revenue** | 実際に獲得する fee(見かけの出来高ではなく実収益) |
| **17.2 Capital Efficiency** | Revenue/TVL、Volume/TVL |
| **17.3 Revenue Quality** | インセンティブ依存の volume と organic volume の分離 |
| **17.4 Concentration Risk** | Revenue / Whale / Chain / Asset の集中度 |
| **17.5 Treasury** | runway と財務の持続可能性 |

---

## 既存分析との接続

- **Uniswap**: 手数料ティア別の出来高(`reports/uniswap/README.md`)は
  §17.1 Revenue の入力になる。段の厚み・リードラグは §17.2 の前段
- **Aave / Morpho**: 金利構造(`reports/aave/README.md`、
  `reports/morpho/README.md`)は §17.2 Capital Efficiency と
  §17.4 Concentration Risk(資産・チェーンごとの利用率の偏り)に直結する

---

## レポート

*(未着手)*
