"""日次出来高を「買い」と「売り」に分解する。

## どちら側を数えるのかという問題

取引には必ず買い手と売り手が 1 人ずついる。したがって「出来高のうち買いが何割か」
という問いは、そのままでは常に 50% が答えになってしまい意味を持たない。
意味を持つのは **どちらが板を取りに行ったか**(アグレッサー、テイカー)である。

    テイカー買い = 売り板に当てて買った   → 価格を押し上げる向きの取引
    テイカー売り = 買い板に当てて売った   → 価格を押し下げる向きの取引

node_fills では `crossed = True` の行がテイカー側にあたる。1 取引につき
`crossed` の行はちょうど 1 行なので、これで二重計上せずに向きが決まる。

**メイカー側で分解しても新しい情報は無い**(メイカー買い = テイカー売りの相手方で、
枚数は厳密に一致する)。分解の軸はアグレッサーしか無い。

## 帰無対照

日次の売買差(net = 買い − 売り)が「たまたま」の大きさなのかを判定するため、
各取引の向きを独立なコイン投げに置き換えた場合の標準偏差を計算する。

    x_i = ±(その取引の名目額)
    iid の帰無: Var(Σx) = Σ x_i²

ただし **実際の取引の向きには強い自己相関がある**(大口が注文を分割するため)。
iid の帰無はこれを無視するので分散を過小評価し、ほぼ全ての日を「有意」にしてしまう。
そこで Bartlett 核による HAC 分散も併せて求め、こちらを主たる基準にする。

    Var_HAC(Σx) = Σ x_i² + 2 Σ_{k=1..K} (1 − k/(K+1)) Σ_i x_i x_{i+k}

x が確定する時刻 / y の期間: 該当なし(実測量の分解であって予測ではない)。
同時点の値どうしを比べているだけで、将来の情報は一切使っていない。

    uv run python scripts/build_volume_side.py --coin xyz:MU
出力: data/daily_volume_side_<coin>.parquet, data/daily_volume_side_<coin>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

HAC_LAGS = 100      # 取引の向きの自己相関を吸収する遅れの数


def hac_sd(x: np.ndarray, lags: int = HAC_LAGS) -> float:
    """Σx の標準偏差を Bartlett 核の HAC で求める。x は符号つきの名目額。"""
    x = x - x.mean()                       # 平均 0 のまわりの分散を見る
    v = float(x @ x)
    k = min(lags, len(x) - 1)
    for j in range(1, k + 1):
        v += 2.0 * (1.0 - j / (k + 1.0)) * float(x[:-j] @ x[j:])
    return float(np.sqrt(max(v, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet")
        .filter(pl.col("crossed"))                       # テイカー側の 1 行だけ
        .with_columns(d=pl.col("ts").dt.date(),
                      notional=pl.col("px") * pl.col("sz"),
                      buy=pl.col("side") == "B")
        .sort("ts")
    )

    # 1 取引 1 行になっていることの検算。crossed 行数 = 取引数
    n_tr = f.height
    print(f"[検算] crossed 行 {n_tr:,} 件(= 取引数)")

    daily = (
        f.group_by("d")
        .agg(
            buy_sz=pl.col("sz").filter(pl.col("buy")).sum(),
            sell_sz=pl.col("sz").filter(~pl.col("buy")).sum(),
            buy_usd=pl.col("notional").filter(pl.col("buy")).sum(),
            sell_usd=pl.col("notional").filter(~pl.col("buy")).sum(),
            n_buy=pl.col("buy").sum(),
            n_sell=(~pl.col("buy")).sum(),
            vwap=pl.col("notional").sum() / pl.col("sz").sum(),
            px_close=pl.col("px").last(),
        )
        .sort("d")
        .with_columns(
            volume_sz=pl.col("buy_sz") + pl.col("sell_sz"),
            volume_usd=pl.col("buy_usd") + pl.col("sell_usd"),
            net_sz=pl.col("buy_sz") - pl.col("sell_sz"),
            net_usd=pl.col("buy_usd") - pl.col("sell_usd"),
        )
        .with_columns(
            buy_share=pl.col("buy_usd") / pl.col("volume_usd") * 100,
            net_share=(pl.col("buy_usd") - pl.col("sell_usd")) / pl.col("volume_usd") * 100,
        )
    )

    # --- 帰無対照: 向きをコイン投げに置き換えたときの net の散らばり --------------
    sd_iid, sd_hac = [], []
    for day in daily["d"].to_list():
        g = f.filter(pl.col("d") == day)
        x = np.where(g["buy"].to_numpy(), 1.0, -1.0) * g["notional"].to_numpy()
        sd_iid.append(float(np.sqrt((x ** 2).sum())))
        sd_hac.append(hac_sd(x))

    daily = daily.with_columns(
        sd_null_iid=pl.Series(sd_iid), sd_null_hac=pl.Series(sd_hac)
    ).with_columns(
        z_iid=pl.col("net_usd") / pl.col("sd_null_iid"),
        z_hac=pl.col("net_usd") / pl.col("sd_null_hac"),
    )

    # --- 要約 -----------------------------------------------------------------
    tb, ts_ = daily["buy_usd"].sum(), daily["sell_usd"].sum()
    tot = tb + ts_
    print(f"\n=== 全期間 {daily.height} 日 ===")
    print(f"  買い(テイカー) {tb/1e9:7.3f}B USD / {daily['buy_sz'].sum():12,.0f} 枚 / "
          f"{daily['n_buy'].sum():10,} 件  = 金額の {tb/tot*100:.2f}%")
    print(f"  売り(テイカー) {ts_/1e9:7.3f}B USD / {daily['sell_sz'].sum():12,.0f} 枚 / "
          f"{daily['n_sell'].sum():10,} 件  = 金額の {ts_/tot*100:.2f}%")
    print(f"  合計           {tot/1e9:7.3f}B USD / {daily['volume_sz'].sum():12,.0f} 枚")
    print(f"  期間ネット     {(tb-ts_)/1e6:+.1f}M USD(= 合計の {(tb-ts_)/tot*100:+.2f}%)")

    s = daily["buy_share"]
    print(f"\n=== 日次の買い比率(金額ベース、%)===")
    print(f"  最小 {s.min():.2f}%({daily.filter(pl.col('buy_share') == s.min())['d'][0]}) / "
          f"中央値 {s.median():.2f}% / 平均 {s.mean():.2f}% / "
          f"最大 {s.max():.2f}%({daily.filter(pl.col('buy_share') == s.max())['d'][0]})")
    print(f"  50% を上回った日 {int((s > 50).sum())} / {daily.height} 日")

    print(f"\n=== 帰無対照(向きが独立なコイン投げなら net はどれだけ散らばるか)===")
    for col, lab in (("z_iid", "iid       "), ("z_hac", "HAC(自己相関を考慮)")):
        z = daily[col]
        print(f"  {lab}: |z| 中央値 {z.abs().median():7.2f} / 最大 {z.abs().max():8.2f} / "
              f"|z| > 1.96 の日 {int((z.abs() > 1.96).sum()):2d} / {daily.height}")
    print("  ※ iid の帰無は取引の向きの自己相関(注文分割)を無視するため分散を過小評価する。"
          "\n     判定には HAC のほうを使うこと。")

    print(f"\n=== 売り越しが大きい日 ===")
    for r in daily.sort("net_usd").head(5).iter_rows(named=True):
        print(f"  {r['d']}  net {r['net_usd']/1e6:+8.1f}M  買い比率 {r['buy_share']:5.2f}%  "
              f"z_hac {r['z_hac']:+6.2f}  出来高 {r['volume_usd']/1e6:6.1f}M")
    print(f"=== 買い越しが大きい日 ===")
    for r in daily.sort("net_usd", descending=True).head(5).iter_rows(named=True):
        print(f"  {r['d']}  net {r['net_usd']/1e6:+8.1f}M  買い比率 {r['buy_share']:5.2f}%  "
              f"z_hac {r['z_hac']:+6.2f}  出来高 {r['volume_usd']/1e6:6.1f}M")

    # --- 同時点の関係(予測ではない。必ず区別して報告する)------------------------
    ret = daily["px_close"].log().diff().to_numpy()[1:] * 100
    ns = daily["net_share"].to_numpy()[1:]
    print(f"\n=== 同時点の関係(その日の売買差とその日の終値リターン)===")
    print(f"  相関 {np.corrcoef(ns, ret)[0,1]:+.3f}(n={len(ns)})")
    print("  ※ これは同時性であって予測力ではない。同じ日のうちに起きた 2 つの量を並べただけ。")
    fwd = daily["px_close"].log().diff().to_numpy()[1:]         # r_{t+1}
    print(f"  参考: net_share_t と r_(t+1) の相関 "
          f"{np.corrcoef(daily['net_share'].to_numpy()[:-1], fwd * 100)[0,1]:+.3f}"
          "(こちらは時間契約を満たす予測の関係)")

    out = ROOT / "data" / f"daily_volume_side_{tag}.parquet"
    daily.write_parquet(out)
    daily.select("d", "buy_sz", "sell_sz", "buy_usd", "sell_usd", "n_buy", "n_sell",
                 "volume_usd", "net_usd", "buy_share", "z_hac").write_csv(
        ROOT / "data" / f"daily_volume_side_{tag}.csv")
    print(f"\n[out] {daily.height} 日 -> {out}")


if __name__ == "__main__":
    main()
