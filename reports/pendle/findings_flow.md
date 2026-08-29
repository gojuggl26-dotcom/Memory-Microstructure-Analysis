# 注文流入の Pressure — 予測できることと執行できることは別

[← Pendle Boros の分析に戻る](README.md)

反対側の最良数量で正規化した注文流入(Pressure)が将来の mid 変化を予測するかを調べた 2 つの結果。「有意である」ことは「取れる」ことを意味しない、という限界も示す。詳細は [pressure_report.md](pressure_report.md) / [pressure_ew_report.md](pressure_ew_report.md)。

---

### ⑤ 注文流入の Pressure — 予測できることと執行できることは別

反対側の最良数量で正規化した注文流入の差を、イベント時間で

$$
\mathrm{Pressure}^{s}_{t} = \frac{Q^{s}_{t} - Q^{s}_{t-1}}{Q^{\bar{s},\mathrm{best}}_{t}}
$$

と定義し、 $k = 1 \dots 30$ イベント先の mid 変化に当てた(1,018 万イベント / 42 セル)。

| | 値 |
|---|---|
| 片側総量 short | **全 7 地平で予測どおり負**( $z = -3.9$ 〜 $-7.4$ ) |
| GLS | **OLS と質的に逆の結論**を出す(説明変数が先決変数のため) |
| ★ $k=1$ の窓 | **75.5% が同一ブロック内**(= その時点では執行できない) |

![Pressure](../../charts/boros_pressure.png)

**執行不能な窓を落とすと符号が反転するセルがある。**「有意である」ことは
「取れる」ことを意味しない。

### ⑥ 最良気配を最も重くすると有効標本が 5.1 倍になる

最良からの tick 距離 $d$ に重み $\exp(-d/\tau)$ を掛けて流入量を測り直した。
$\tau$ を入れるとゼロ膨張が解け、使える標本が **5.1 倍**に増える。

| 側 | 最適な $\tau$ | 意味 |
|---|---|---|
| long | $\tau \to 0$ | 最良気配に集中 |
| short | $\tau \to \infty$ | 板全体 |
| **diff** | **$\tau = 64$ に内点最適** | 両端より強い |

3 指標(符号検定・Pearson・Spearman)が揃って一致するのは
$\tau = 64$ の diff の $k = 1 \dots 3$ だけだった。

![指数減衰 Pressure](../../charts/boros_pressure_ew.png)
