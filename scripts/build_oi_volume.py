"""日次平均 OI(建玉)と出来高を node_fills から再構成する。

WORK_BUCKET に建玉のスナップショット(perp_and_spot_balances)が無いため、
fills の `startPosition` からユーザーごとの建玉推移を復元して合計する。

## 復元の手順

1 行の fill だけで、その約定前後の建玉が判る:

    pos_before = startPosition
    pos_after  = startPosition + (+sz if side=="B" else -sz)

`dir`(Open Long / Close Short / Long > Short …)と突き合わせて 22,349,510 行すべてで
一致することを確認済み(検証 V1)。

同一ユーザー・同一 ns に複数の fill があると行の並び順が定まらないので、
ns 単位のグループにまとめ、グループの入口と出口の建玉を次で決める:

    Σd > 0(買い越し) → entry = min(startPosition), exit = max(pos_after)
    Σd ≤ 0(売り越し) → entry = max(startPosition), exit = min(pos_after)

多重グループの 99.05% は売買が片方向で、この規則は `exit == entry + Σd` を
98.5% で満たす(検証 V2)。売買混在のグループ(0.95%)は exit = entry + Σd とする。

OI(t) = Σ_user max(position, 0) は、グループごとの増分の累積で求める:

    dL = max(exit, 0) - max(前のグループの exit, 0)

ユーザーの初回グループでは「前の exit」をそのグループの entry とし、
窓開始の OI は Σ_user max(初回グループの entry, 0) とする。

## 推定の限界(報告に必ず併記する)

- **窓期間中に一度も約定しなかったユーザーの建玉は観測できない。** 観測ユーザーの
  ネット建玉が 0 でない分だけ未観測者がいる(窓開始で +2,934 枚 = 当時の OI の 6%)。
- ロング側とショング側を独立に復元すると、本来一致するはずの値が 0.4〜11% ずれる。
  約定を伴わない建玉の飛び(全遷移の 0.25%)が原因。両者の平均を推定値、
  差の半分を不確かさとして出力する(`oi_mean` と `oi_halfgap`)。

x が確定する時刻 / y の期間: 該当なし(実測量の再構成であって予測ではない)。

    uv run python scripts/build_oi_volume.py --coin xyz:MU
出力: data/daily_oi_volume_<coin>.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

POS = lambda c: pl.max_horizontal(pl.col(c), pl.lit(0.0))   # noqa: E731  ロング側
NEG = lambda c: pl.max_horizontal(-pl.col(c), pl.lit(0.0))  # noqa: E731  ショート側


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    f = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet")
        .with_columns(signed=pl.when(pl.col("side") == "B").then(pl.col("sz")).otherwise(-pl.col("sz")))
        .with_columns(pa=pl.col("startPosition") + pl.col("signed"))
    )

    # --- ns グループの入口/出口 -------------------------------------------------
    g = (
        f.group_by("user", "ts")
        .agg(
            n=pl.len(),
            n_buy=(pl.col("side") == "B").sum(),
            d=pl.col("signed").sum(),
            sp_min=pl.col("startPosition").min(),
            sp_max=pl.col("startPosition").max(),
            pa_min=pl.col("pa").min(),
            pa_max=pl.col("pa").max(),
        )
        .with_columns(uni=(pl.col("n_buy") == 0) | (pl.col("n_buy") == pl.col("n")))
        .with_columns(entry=pl.when(pl.col("d") > 0).then(pl.col("sp_min")).otherwise(pl.col("sp_max")))
        .with_columns(
            exit=pl.when(pl.col("uni") & (pl.col("d") > 0)).then(pl.col("pa_max"))
            .when(pl.col("uni") & (pl.col("d") <= 0)).then(pl.col("pa_min"))
            .otherwise(pl.col("entry") + pl.col("d"))
        )
        .sort("user", "ts")
        .with_columns(prev=pl.col("exit").shift(1).over("user"))
    )

    # 検証: 約定を伴わない建玉の飛び
    tel = g.filter(pl.col("prev").is_not_null()).with_columns(e=(pl.col("prev") - pl.col("entry")).abs())
    n_tel = int((tel["e"] > 1e-6).sum())
    print(f"[V3] 建玉の飛び {n_tel:,} / {tel.height:,} 遷移 = {n_tel/tel.height*100:.4f}%  "
          f"Σ|飛び| = {tel.filter(pl.col('e')>1e-6)['e'].sum():,.0f} 枚")

    g = g.with_columns(prev=pl.col("prev").fill_null(pl.col("entry")))
    g = g.with_columns(dL=POS("exit") - POS("prev"), dS=NEG("exit") - NEG("prev"))

    first = g.group_by("user").agg(e0=pl.col("entry").first())
    oi0_l = first.select(POS("e0").sum()).item()
    oi0_s = first.select(NEG("e0").sum()).item()
    print(f"[V4] 窓開始 OI: long {oi0_l:,.0f} / short {oi0_s:,.0f} / ネット {oi0_l-oi0_s:+,.0f} "
          f"(= 未観測ユーザーの建玉。0 に近いほど良い)")

    # --- OI の階段関数。日境界を挿入してどの区間も日を跨がないようにする -----------
    ev = g.group_by("ts").agg(dL=pl.col("dL").sum(), dS=pl.col("dS").sum())
    days = f["ts"].dt.date().unique().sort().to_list()
    bounds = pl.DataFrame(
        {"ts": [dt.datetime.combine(d, dt.time()) for d in days]
               + [dt.datetime.combine(days[-1] + dt.timedelta(days=1), dt.time())]}
    ).with_columns(ts=pl.col("ts").cast(pl.Datetime("ns")), dL=pl.lit(0.0), dS=pl.lit(0.0))
    ev = (
        pl.concat([ev, bounds])
        .group_by("ts").agg(dL=pl.col("dL").sum(), dS=pl.col("dS").sum())
        .sort("ts")
        .with_columns(oi_l=oi0_l + pl.col("dL").cum_sum(), oi_s=oi0_s + pl.col("dS").cum_sum())
        .with_columns(oi=(pl.col("oi_l") + pl.col("oi_s")) / 2, halfgap=(pl.col("oi_l") - pl.col("oi_s")).abs() / 2)
        .with_columns(dur=(pl.col("ts").shift(-1) - pl.col("ts")).dt.total_nanoseconds().cast(pl.Float64),
                      d=pl.col("ts").dt.date())
        .drop_nulls("dur")
    )
    wmean = lambda c: (pl.col(c) * pl.col("dur")).sum() / pl.col("dur").sum()  # noqa: E731
    daily_oi = ev.group_by("d").agg(
        oi_mean=wmean("oi"), oi_halfgap=wmean("halfgap"),
        oi_min=pl.col("oi").min(), oi_max=pl.col("oi").max(), oi_close=pl.col("oi").last(),
        covered_s=pl.col("dur").sum() / 1e9,
    )

    # --- 出来高: 1 取引 2 行なのでテイカー側だけを数える ---------------------------
    tk = f.filter(pl.col("crossed")).with_columns(
        d=pl.col("ts").dt.date(), notional=pl.col("px") * pl.col("sz")
    )
    daily_v = tk.group_by("d").agg(
        volume=pl.col("sz").sum(), volume_usd=pl.col("notional").sum(),
        n_trades=pl.len(), n_users=pl.col("user").n_unique(),
        vwap=pl.col("notional").sum() / pl.col("sz").sum(),
        px_close=pl.col("px").sort_by("ts").last(),
    )

    daily = (
        daily_oi.join(daily_v, on="d", how="full", coalesce=True).sort("d")
        .with_columns(oi_mean_usd=pl.col("oi_mean") * pl.col("vwap"),
                      oi_halfgap_usd=pl.col("oi_halfgap") * pl.col("vwap"))
    )
    assert daily["covered_s"].min() > 86399.9, "1 日 86,400 秒を覆えていない日がある"
    out = ROOT / "data" / f"daily_oi_volume_{tag}.parquet"
    daily.write_parquet(out)
    print(f"[out] {daily.height} 日 -> {out}")
    with pl.Config(tbl_rows=8, tbl_width_chars=200):
        print(daily.select("d", "oi_mean", "oi_halfgap", "oi_mean_usd", "volume", "volume_usd", "n_trades", "vwap"))


if __name__ == "__main__":
    main()
