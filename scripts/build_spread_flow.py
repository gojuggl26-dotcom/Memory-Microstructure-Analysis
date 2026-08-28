"""イベントごとにスプレッド幅・注文量・約定量を出し、OBI / OFI との関係を測る。

## 1 イベント = 最良気配が変わった 1 行(l2/bbo)

99 日で 35,188,572 イベント。各イベントで次の 5 つを持たせる。

| 量 | 定義 | 単位 |
|---|---|---|
| スプレッド幅 | (best_ask − best_bid) / mid × 10⁴ | bp |
| 注文量 | bid_sz + ask_sz(最良気配の合計数量) | 枚 |
| 約定量 | 直前のイベントからこのイベントまでに約定した数量 | 枚 |
| OBI | (bid_sz − ask_sz) / (bid_sz + ask_sz) | 無次元 |
| OFI | Cont–Kukanov–Stoikov。直前 5,000 イベントの標準偏差で割って無次元化 | 無次元 |

OBI と OFI の定義・正規化・帯の切り方は `build_obi_ofi.py` と同一にしてある
(同じ銘柄で 2 つのレポートが違う定義を使わないため)。

## 約定量の割り当て(先読みをしないための向き)

約定はイベントを**引き起こす**側なので、割り当ての向きを間違えると未来を見てしまう。
ここでは **イベント t の約定量 = 区間 (t−1, t] に起きた約定の合計** とする。
つまり各約定を「その約定以後で最初に来るイベント」に寄せる。こうすると
イベント t の 5 つの量はすべて時刻 t までの情報だけで決まる。

## スプレッドは相対値を主に使う

標本期間で価格が 545 → 1,250 ドルと 2.3 倍動くので、絶対幅(ドル)は期間内で
比較できない。相対幅(bp)を主に使い、絶対幅も併せて出す。

## クロスの除外

best_ask < best_bid のイベント 16,826 件(0.048%)を除く。件数は僅かだが
スプレッドが負になり、平均や相関を 1 行で歪めうる(CLAUDE.md の既知の事故型)。
ロック(bid = ask、2,360 件)は実在しうる状態として残す。

## これは同時点の関係である

5 つの量はすべて同じイベントで測っている。**予測ではない**
(reports/predicting_definition.md 第 2 節)。相関は同時性の記述としてのみ読む。

x が確定する時刻 / y の期間: 該当なし(同時点の記述統計)。

    uv run python scripts/build_spread_flow.py --coin xyz:MU
出力: data/spread_flow_bins_<coin>.csv, data/spread_flow_corr_<coin>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

ROLL_W = 5_000                                    # build_obi_ofi.py と同じ
OBI_EDGES = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
OBI_LABELS = ["−1.00〜−0.75", "−0.75〜−0.50", "−0.50〜−0.25", "−0.25〜0",
              "ちょうど 0", "0〜+0.25", "+0.25〜+0.50", "+0.50〜+0.75", "+0.75〜+1.00"]
OFI_EDGES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
OFI_LABELS = ["< −2σ", "−2〜−1σ", "−1〜−0.5σ", "−0.5〜0σ",
              "ちょうど 0", "0〜+0.5σ", "+0.5〜+1σ", "+1〜+2σ", "> +2σ"]
NB = len(OBI_LABELS)
SUB_FRAC = 0.03                                   # 相関と分位に使う抜き取り率

VARS = ["spread_bp", "qsize", "fillsz", "obi", "abs_obi", "ofi_z", "abs_ofi_z"]
VAR_JA = {"spread_bp": "スプレッド幅(bp)", "qsize": "注文量(枚)", "fillsz": "約定量(枚)",
          "obi": "OBI", "abs_obi": "|OBI|", "ofi_z": "OFI(正規化)", "abs_ofi_z": "|OFI|"}


def assign_bins(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """帯番号。ちょうど 0 は専用の帯 4。無効値は −1(build_obi_ofi.py と同じ規則)。"""
    ok = np.isfinite(x)
    b = np.digitize(np.nan_to_num(x), edges)
    out = np.where(b < 4, b, np.where(b == 4, 5, b + 1))
    out = np.where(ok & (x == 0.0), 4, out)
    return np.where(ok, out, -1).astype(np.int64)


def one_day(bbo: pl.DataFrame, fills: pl.DataFrame) -> pl.DataFrame:
    """1 日分のイベント表を作る。"""
    d = bbo.sort("ts").with_columns(
        mid=(pl.col("best_bid") + pl.col("best_ask")) / 2,
        obi=(pl.col("bid_sz") - pl.col("ask_sz")) / (pl.col("bid_sz") + pl.col("ask_sz")),
        qsize=pl.col("bid_sz") + pl.col("ask_sz"),
        spread_abs=pl.col("best_ask") - pl.col("best_bid"),
    ).with_columns(spread_bp=pl.col("spread_abs") / pl.col("mid") * 1e4)

    pb, pa = pl.col("best_bid"), pl.col("best_ask")
    qb, qa = pl.col("bid_sz"), pl.col("ask_sz")
    d = d.with_columns(
        ofi=(
            pl.when(pb >= pb.shift(1)).then(qb).otherwise(0.0)
            - pl.when(pb <= pb.shift(1)).then(qb.shift(1)).otherwise(0.0)
            - pl.when(pa <= pa.shift(1)).then(qa).otherwise(0.0)
            + pl.when(pa >= pa.shift(1)).then(qa.shift(1)).otherwise(0.0)
        )
    )
    d = d.with_columns(
        scale=pl.col("ofi").rolling_std(ROLL_W, min_samples=ROLL_W // 2).shift(1)
    ).with_columns(
        ofi_z=pl.when(pl.col("scale") > 0).then(pl.col("ofi") / pl.col("scale")).otherwise(None)
    )

    # 約定量: 各約定を「その約定以後で最初に来るイベント」へ寄せる。
    # こうするとイベント t には区間 (t−1, t] の約定だけが入り、未来は入らない。
    if fills.height:
        f = fills.sort("ts").join_asof(
            d.select("ts").with_columns(ev=pl.col("ts")), on="ts", strategy="forward")
        agg = f.drop_nulls("ev").group_by("ev").agg(fillsz=pl.col("sz").sum())
        d = d.join(agg, left_on="ts", right_on="ev", how="left")
    else:
        d = d.with_columns(fillsz=pl.lit(None, dtype=pl.Float64))
    return d.with_columns(fillsz=pl.col("fillsz").fill_null(0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(0)

    bbo = pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet").with_columns(
        ts=pl.col("ts").cast(pl.Datetime("ns")))
    n_all = bbo.height
    n_cross = int((bbo["best_ask"] < bbo["best_bid"]).sum())
    bbo = bbo.filter(pl.col("best_ask") >= pl.col("best_bid"))
    print(f"[bbo] {n_all:,} イベント / クロス除外 {n_cross:,}"
          f"({n_cross / n_all * 100:.3f}%) → {bbo.height:,}", file=sys.stderr)

    fl = (
        pl.read_parquet(ROOT / "data" / f"fills_{tag}.parquet", columns=["ts", "sz", "crossed"])
        .filter(pl.col("crossed")).with_columns(dt=pl.col("ts").dt.date().cast(pl.String))
    )

    days = bbo["dt"].unique().sort().to_list()
    subs, nsum = [], 0
    # 帯ごとの合計(全イベント)。行 = 帯、列 = [件数, spread, qsize, fillsz]
    # 列 = [件数, spread 合計, qsize 合計, fillsz 合計, 約定を伴った件数]
    acc = {f: np.zeros((NB, 5)) for f in ("OBI", "OFI_z")}

    for dt_ in days:
        d = one_day(bbo.filter(pl.col("dt") == dt_),
                    fl.filter(pl.col("dt") == dt_).select("ts", "sz"))
        arr = {v: d[v].to_numpy().astype(np.float64) for v in
               ("spread_bp", "spread_abs", "qsize", "fillsz", "obi", "ofi_z")}
        arr["abs_obi"] = np.abs(arr["obi"])
        arr["abs_ofi_z"] = np.abs(arr["ofi_z"])
        nsum += len(arr["obi"])

        for feat, edges in (("OBI", OBI_EDGES), ("OFI_z", OFI_EDGES)):
            b = assign_bins(arr["obi"] if feat == "OBI" else arr["ofi_z"], edges)
            ok = b >= 0
            np.add.at(acc[feat][:, 0], b[ok], 1.0)
            for j, v in enumerate(("spread_bp", "qsize", "fillsz"), start=1):
                np.add.at(acc[feat][:, j], b[ok], arr[v][ok])
            np.add.at(acc[feat][:, 4], b[ok], (arr["fillsz"][ok] > 0).astype(float))

        m = rng.random(len(arr["obi"])) < SUB_FRAC
        subs.append(pl.DataFrame({v: arr[v][m] for v in VARS + ["spread_abs"]}))

    S = pl.concat(subs).drop_nulls()
    S.write_parquet(ROOT / "data" / f"spread_flow_sample_{tag}.parquet")
    print(f"[抜き取り] 相関・分位に使う {S.height:,} 行(全 {nsum:,} の "
          f"{S.height / nsum * 100:.2f}%)", file=sys.stderr)

    # --- 抜き取りが全体を代表しているかの検算 --------------------------------
    print("\n=== 抜き取りの妥当性(全イベントの帯別平均と突き合わせ)===")
    for feat in ("OBI", "OFI_z"):
        full = acc[feat][:, 1].sum() / acc[feat][:, 0].sum()
        print(f"  {feat} 帯の全イベント加重平均スプレッド {full:.4f} bp")
    print(f"  抜き取りの平均スプレッド {S['spread_bp'].mean():.4f} bp")

    # --- 記述統計 -------------------------------------------------------------
    print("\n=== 1 イベントあたりの分布(抜き取り)===")
    print(f"{'量':<16}{'中央値':>12}{'平均':>12}{'第1四分位':>12}{'第3四分位':>12}{'99%点':>12}")
    for v in ("spread_bp", "spread_abs", "qsize", "fillsz"):
        c = S[v]
        lab = {"spread_abs": "スプレッド幅(ドル)"}.get(v, VAR_JA.get(v, v))
        print(f"{lab:<16}{c.median():>12.4f}{c.mean():>12.4f}"
              f"{c.quantile(0.25):>12.4f}{c.quantile(0.75):>12.4f}{c.quantile(0.99):>12.4f}")
    print(f"  約定を伴ったイベントの割合 {(S['fillsz'] > 0).mean() * 100:.2f}%")

    # --- 相関行列(Spearman)---------------------------------------------------
    X = np.vstack([S[v].to_numpy() for v in VARS])
    R = np.zeros((len(VARS), len(VARS)))
    ranks = np.vstack([np.argsort(np.argsort(x)).astype(np.float64) for x in X])
    for i in range(len(VARS)):
        for j in range(len(VARS)):
            R[i, j] = np.corrcoef(ranks[i], ranks[j])[0, 1]
    pl.DataFrame({"var": VARS, **{VARS[j]: R[:, j] for j in range(len(VARS))}}).write_csv(
        ROOT / "data" / f"spread_flow_corr_{tag}.csv")
    print("\n=== スピアマン順位相関(抜き取り、同時点)===")
    print(f"{'':<16}" + "".join(f"{VAR_JA[v][:8]:>11}" for v in VARS))
    for i, v in enumerate(VARS):
        print(f"{VAR_JA[v][:14]:<16}" + "".join(f"{R[i, j]:>11.3f}" for j in range(len(VARS))))

    # --- 帯別の代表値 ----------------------------------------------------------
    rows = []
    for feat, labels in (("OBI", OBI_LABELS), ("OFI_z", OFI_LABELS)):
        key = "obi" if feat == "OBI" else "ofi_z"
        edges = OBI_EDGES if feat == "OBI" else OFI_EDGES
        b = assign_bins(S[key].to_numpy(), edges)
        for i, lab in enumerate(labels):
            m = b == i
            n_full = acc[feat][i, 0]
            row = {"feat": feat, "bin_i": i, "bin": lab, "n_full": int(n_full),
                   "n_sub": int(m.sum()),
                   "fill_rate": acc[feat][i, 4] / n_full if n_full else float("nan")}
            for j, v in enumerate(("spread_bp", "qsize", "fillsz"), start=1):
                row[f"{v}_mean_full"] = acc[feat][i, j] / n_full if n_full else float("nan")
                s = S[v].to_numpy()[m]
                row[f"{v}_med"] = float(np.median(s)) if m.sum() else float("nan")
                row[f"{v}_q1"] = float(np.quantile(s, 0.25)) if m.sum() else float("nan")
                row[f"{v}_q3"] = float(np.quantile(s, 0.75)) if m.sum() else float("nan")
            rows.append(row)
    B = pl.DataFrame(rows)
    B.write_csv(ROOT / "data" / f"spread_flow_bins_{tag}.csv")

    for feat in ("OBI", "OFI_z"):
        print(f"\n=== {feat} の帯ごと(中央値。件数は全イベント)===")
        print(f"{'帯':<16}{'件数':>13}{'スプレッド bp':>14}{'注文量 枚':>12}"
              f"{'約定量 平均':>13}{'約定の発生率':>13}")
        for r in B.filter(pl.col("feat") == feat).iter_rows(named=True):
            print(f"{r['bin']:<16}{r['n_full']:>13,}{r['spread_bp_med']:>14.3f}"
                  f"{r['qsize_med']:>12.3f}{r['fillsz_mean_full']:>13.4f}"
                  f"{r['fill_rate'] * 100:>12.2f}%")
    print(f"\n[out] data/spread_flow_bins_{tag}.csv, data/spread_flow_corr_{tag}.csv")


if __name__ == "__main__":
    main()
