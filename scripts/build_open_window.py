"""米国市場の寄付きを挟む 6 時間を、1 分刻みで見る。

## 窓

ニューヨーク証券取引所の寄付きは 13:30 UTC(標本期間は全日が米国東部夏時間)。
その **3 時間前から 3 時間後まで**、10:30-16:30 UTC の 360 分を対象とする。
立会日 68 日それぞれで 1 分ごとの値を出し、**日をまたいで平均**した profile を描く。

## 出来高

約定記録のテイカー側(`crossed`)だけを数え、買い(`side == "B"`)と売りに分ける。
取引には必ず買い手と売り手がいるので、意味を持つ分解軸は
「どちらが板を取りに行ったか」しかない(GLOSSARY.md 参照)。

## 相対スプレッド

L2 の `bbo` 表に入っている `spread_bp`(= (ask - bid) / mid の 1 万分率)を使う。
板は約定のたびに変わる階段関数なので、**各更新が有効だった時間で重みづけ**して
1 分ごとの平均を取る。区間が分の境界をまたぐ場合は開始時刻の分に寄せている
(1 分あたりの更新は中央値で数十件あるため、この近似の影響は小さい)。

## 分散

**算出窓 5 分、1 分あたりに換算した実現分散**。1 分足の中値リターン
$r_i = \\ln(m_i / m_{i-1})$ について

    Var(t) = (1/5) * Σ_{i=t-4..t} r_i^2

中値(mid)を使うのは、板の内側で価格が buy/sell を往復することによる
見かけの分散(bid-ask bounce)を避けるためである。既存の
`build_variance.py` は約定値を使っているので、水準は直接比較できない。

x が確定する時刻 / y の期間: 該当なし(実測量の集計であって予測ではない)。
5 分窓は後ろ向き(その分を含む直前 5 分)なので、未来の情報は入らない。

    uv run python scripts/build_open_window.py --coin xyz:MU
出力: data/open_window_<coin>.csv と charts/<coin>_open_window_*.png
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import exchange_calendars as xc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BUY_C, SELL_C = "#2a78d6", "#eb6834"      # 検証済みパレットの 1 番と 2 番
SPREAD_C, VAR_C = "#1baf7a", "#4a3aa7"    # 3 番(アクア)と 7 番(すみれ)
PRE_BG = "#fadfc9"                        # 寄付き前(原市場が閉じている時間)
OPEN_MIN = 13 * 60 + 30                   # 13:30 UTC
HALF_H = 3                                # 前後それぞれ 3 時間
VAR_WIN = 5                               # 分散の算出窓(分)


def fetch_bbo(coin: str, days: list) -> pl.DataFrame:
    """必要な列だけを落として bbo を読む。ts は int64(ns)。"""
    tag = coin.replace(":", "_")
    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    want = {str(d) for d in days}
    keys = (
        inv.filter(
            (pl.col("coin") == coin) & (pl.col("layer") == "l2") & (pl.col("table") == "bbo")
            & (pl.col("leaf") == "part-000.parquet") & pl.col("dt").is_in(list(want))
        )
        .sort("dt").select("dt", "key").rows()
    )
    bucket = os.environ.get("WORK_BUCKET") or exit("WORK_BUCKET 未設定")
    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(access_key=cr.access_key, secret_key=cr.secret_key,
                           session_token=cr.token, region="us-east-1")

    lo, hi = (OPEN_MIN - HALF_H * 60), (OPEN_MIN + HALF_H * 60)

    def one(item):
        dt_, key = item
        t = pq.read_table(f"{bucket}/{key}", columns=["ts", "mid", "spread_bp"], filesystem=fs)
        d = pl.from_arrow(t).with_columns(ts=pl.col("ts").cast(pl.Datetime("ns")))
        return d.with_columns(
            mo=(pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)),
            d=pl.col("ts").dt.date(),
        ).filter((pl.col("mo") >= lo) & (pl.col("mo") < hi))

    with ThreadPoolExecutor(12) as ex:
        parts = list(ex.map(one, keys))
    out = pl.concat(parts).sort("ts")
    print(f"[bbo] {len(keys)} 日 / 窓内 {out.height:,} 行", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    lo, hi = (OPEN_MIN - HALF_H * 60), (OPEN_MIN + HALF_H * 60)

    # --- 立会日の一覧 --------------------------------------------------------
    f = pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet",
                        columns=["ts", "sz", "side", "crossed"]).filter(pl.col("crossed"))
    all_days = f["ts"].dt.date().unique().sort().to_list()
    cal = xc.get_calendar("XNYS")
    sess = {x.date() for x in cal.sessions_in_range(str(all_days[0]), str(all_days[-1]))}
    days = [d for d in all_days if d in sess]

    # 板(bbo)が完成している日だけを対象にする。MU は 2026-08-10 の L2 が
    # 未完成なので(reports/MU/mu_inventory_report.md)、出来高・スプレッド・分散を
    # 同じ日集合の上で比べるために、ここで揃えておく。
    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    done = set(
        inv.filter((pl.col("coin") == a.coin) & (pl.col("layer") == "l2")
                   & (pl.col("table") == "bbo") & (pl.col("leaf") == "_SUCCESS"))["dt"].to_list()
    )
    dropped = [d for d in days if str(d) not in done]
    days = [d for d in days if str(d) in done]
    print(f"[窓] {lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d} UTC の {hi - lo} 分 "
          f"× 立会日 {len(days)} 日"
          + (f"(板が未完成のため除外: {', '.join(str(d) for d in dropped)})" if dropped else ""))

    # --- 出来高(買い / 売り)-------------------------------------------------
    fv = (
        f.with_columns(
            d=pl.col("ts").dt.date(),
            mo=(pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)),
        )
        .filter(pl.col("d").is_in(days) & (pl.col("mo") >= lo) & (pl.col("mo") < hi))
        .group_by("d", "mo")
        .agg(buy=pl.col("sz").filter(pl.col("side") == "B").sum(),
             sell=pl.col("sz").filter(pl.col("side") == "A").sum())
    )
    grid = pl.DataFrame({"d": [d for d in days for _ in range(hi - lo)],
                         "mo": list(range(lo, hi)) * len(days)}).with_columns(
        pl.col("mo").cast(pl.Int32))
    fv = grid.join(fv, on=["d", "mo"], how="left").with_columns(
        buy=pl.col("buy").fill_null(0.0), sell=pl.col("sell").fill_null(0.0))
    assert fv.height == len(days) * (hi - lo), fv.height

    # --- 相対スプレッド(時間加重)と中値 ---------------------------------------
    b = fetch_bbo(a.coin, days)

    # ★クロス(best_ask < best_bid)の行を先に落とす。全体の 0.04% しか無いが
    # 最悪 −2,910 bp に達し、時間加重平均を 1 行で支配しうる。
    # 時間の重みを計算する前に落とすことで、直前の正常な気配がその区間を引き継ぐ。
    n0 = b.height
    n_cross = int((b["spread_bp"] < 0).sum())
    n_lock = int((b["spread_bp"] == 0).sum())
    b = b.filter(pl.col("spread_bp") >= 0)
    print(f"[クロス除外] {n_cross:,} 行({n_cross / n0 * 100:.4f}%)を除外。"
          f"ロック(spread = 0)は {n_lock:,} 行で、こちらは実在しうる状態として残す。")

    b = b.with_columns(
        dur=(pl.col("ts").shift(-1) - pl.col("ts")).dt.total_nanoseconds().cast(pl.Float64).over("d")
    ).drop_nulls("dur").filter(pl.col("dur") > 0)
    sp = b.group_by("d", "mo").agg(
        spread_bp=(pl.col("spread_bp") * pl.col("dur")).sum() / pl.col("dur").sum(),
        mid=pl.col("mid").last(),
    )

    # --- 5 分窓の分散(1 分あたり)---------------------------------------------
    m = grid.join(sp, on=["d", "mo"], how="left").sort("d", "mo")
    m = m.with_columns(mid=pl.col("mid").fill_null(strategy="forward").over("d"))
    m = m.with_columns(r=(pl.col("mid").log() - pl.col("mid").log().shift(1)).over("d"))
    m = m.with_columns(r2=pl.col("r") ** 2)
    m = m.with_columns(var5=pl.col("r2").rolling_sum(VAR_WIN).over("d") / VAR_WIN * 1e4)

    # --- 日をまたいで平均 -----------------------------------------------------
    prof = (
        fv.join(m.select("d", "mo", "spread_bp", "var5"), on=["d", "mo"], how="left")
        .group_by("mo")
        .agg(buy=pl.col("buy").mean(), sell=pl.col("sell").mean(),
             spread_bp=pl.col("spread_bp").mean(),
             spread_q1=pl.col("spread_bp").quantile(0.25),
             spread_q3=pl.col("spread_bp").quantile(0.75),
             var5=pl.col("var5").mean(),
             var5_q1=pl.col("var5").quantile(0.25),
             var5_q3=pl.col("var5").quantile(0.75),
             n=pl.len())
        .sort("mo")
        .with_columns(vol=pl.col("buy") + pl.col("sell"),
                      rel=(pl.col("mo") - OPEN_MIN))
    )
    prof.write_csv(ROOT / "data" / f"open_window_{tag}.csv")

    pre = prof.filter(pl.col("rel") < 0)
    post = prof.filter(pl.col("rel") >= 0)
    print(f"\n=== 寄付き前 3 時間 vs 後 3 時間(立会日 {len(days)} 日の平均)===")
    print(f"  出来高      前 {pre['vol'].sum():9,.0f} 枚 / 後 {post['vol'].sum():9,.0f} 枚"
          f"  → 後は前の {post['vol'].sum() / pre['vol'].sum():.1f} 倍")
    print(f"  相対スプレッド 前 {pre['spread_bp'].mean():6.2f} bp / 後 {post['spread_bp'].mean():6.2f} bp"
          f"  → 後は前の {post['spread_bp'].mean() / pre['spread_bp'].mean():.2f} 倍")
    print(f"  5 分窓の分散  前 {pre['var5'].mean():7.4f} / 後 {post['var5'].mean():7.4f} (%)^2/分"
          f"  → 後は前の {post['var5'].mean() / pre['var5'].mean():.1f} 倍")
    top = prof.sort("vol", descending=True).head(3)
    print("\n  出来高が多い分 上位 3:")
    for r in top.iter_rows(named=True):
        print(f"    {r['mo'] // 60:02d}:{r['mo'] % 60:02d} UTC(寄付き {r['rel']:+d} 分)"
              f"  {r['vol']:8,.0f} 枚  スプレッド {r['spread_bp']:5.2f} bp")
    mn = prof.sort("spread_bp").head(1).to_dicts()[0]
    mx = prof.sort("spread_bp", descending=True).head(1).to_dicts()[0]
    print(f"\n  スプレッド 最小 {mn['spread_bp']:.2f} bp(寄付き {mn['rel']:+d} 分) / "
          f"最大 {mx['spread_bp']:.2f} bp(寄付き {mx['rel']:+d} 分)")
    print(f"  買いの比率(窓全体)= {prof['buy'].sum() / prof['vol'].sum() * 100:.2f}%")

    # --- 作図 ---------------------------------------------------------------
    mpl.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "axes.unicode_minus": False, "text.parse_math": False,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    x = prof["rel"].to_numpy()

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-HALF_H * 60, 0, color=PRE_BG, lw=0, zorder=0)
        ax.axvline(0, color=BASELINE, lw=1.2, zorder=2)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
        ax.set_xlim(-HALF_H * 60, HALF_H * 60)
        ax.set_xticks(range(-180, 181, 30))

    # 図 1: 出来高(買い・売り)と 相対スプレッド
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True, dpi=170,
                                 gridspec_kw={"hspace": 0.18, "height_ratios": [1.25, 1]})
    buy, sell = prof["buy"].to_numpy(), prof["sell"].to_numpy()
    a1.bar(x, buy, width=1.0, color=BUY_C, zorder=3, linewidth=0, label="テイカー買い")
    a1.bar(x, sell, width=1.0, bottom=buy, color=SELL_C, zorder=3, linewidth=0, label="テイカー売り")
    style(a1)
    a1.set_ylabel("1 分あたりの出来高(枚)", color=INK2, fontsize=10)
    a1.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, ncol=2)
    a1.set_title(f"{a.coin} 寄付きを挟む 6 時間の 1 分ごとの出来高と相対スプレッド",
                 loc="left", color=INK, fontsize=13, pad=44, weight="bold")
    a1.text(0, 1.085,
            f"立会日 {len(days)} 日の平均。横軸はニューヨーク寄付き(13:30 UTC)からの経過分。"
            "薄いオレンジは寄付き前(原市場が閉じている時間)。",
            transform=a1.transAxes, color=INK2, fontsize=9.5)
    a1.text(0, 1.040,
            "縦軸が 2 つある図は目盛りの合わせ方が恣意的になるため作らない。"
            "出来高とスプレッドは横軸を共有した上下 2 段で並べている。",
            transform=a1.transAxes, color=INK2, fontsize=9.5)

    a2.fill_between(x, prof["spread_q1"].to_numpy(), prof["spread_q3"].to_numpy(),
                    color=SPREAD_C, alpha=0.16, lw=0, zorder=2)
    a2.plot(x, prof["spread_bp"].to_numpy(), color=SPREAD_C, lw=2.0, zorder=3,
            solid_capstyle="round")
    style(a2)
    a2.set_ylabel("相対スプレッド(bp)", color=INK2, fontsize=10)
    a2.set_ylim(0, float(prof["spread_q3"].max()) * 1.15)
    a2.set_xlabel("ニューヨーク寄付き(13:30 UTC)からの経過分", color=INK2, fontsize=10)
    a2.set_title("相対スプレッド(帯は日ごとのばらつき、第 1〜第 3 四分位)",
                 loc="left", color=INK, fontsize=11.5, pad=8, weight="bold")
    fig.text(0.005, 0.008,
             "出所: Hyperliquid L4 (Artemis) node_fills と L2 bbo / 窓 2026-05-04〜08-10 の立会日",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.855, bottom=0.085)
    out1 = ROOT / "charts" / f"{tag}_open_window_volume_spread.png"
    fig.savefig(out1)
    plt.close(fig)
    print(f"\n[chart] {out1}")

    # 図 2: 5 分窓の分散
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=170)
    ax.fill_between(x, prof["var5_q1"].to_numpy(), prof["var5_q3"].to_numpy(),
                    color=VAR_C, alpha=0.14, lw=0, zorder=2)
    ax.plot(x, prof["var5"].to_numpy(), color=VAR_C, lw=2.0, zorder=3, solid_capstyle="round")
    style(ax)
    ax.set_ylim(0, float(prof["var5_q3"].max()) * 1.15)
    ax.set_ylabel("1 分あたりの分散((%)^2)", color=INK2, fontsize=10)
    ax.set_xlabel("ニューヨーク寄付き(13:30 UTC)からの経過分", color=INK2, fontsize=10)
    ax.set_title(f"{a.coin} 寄付きを挟む 6 時間の分散(算出窓 5 分、1 分あたり)",
                 loc="left", color=INK, fontsize=13, pad=30, weight="bold")
    ax.text(0, 1.055,
            f"立会日 {len(days)} 日の平均。中値の 1 分足リターンを直前 5 分ぶん二乗和して 5 で割った値。"
            "帯は日ごとのばらつき(第 1〜第 3 四分位)。",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    fig.text(0.005, 0.012,
             "出所: Hyperliquid L4 (Artemis) L2 bbo の中値 / 窓 2026-05-04〜08-10 の立会日",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.845, bottom=0.125)
    out2 = ROOT / "charts" / f"{tag}_open_window_variance.png"
    fig.savefig(out2)
    print(f"[chart] {out2}")


if __name__ == "__main__":
    main()
