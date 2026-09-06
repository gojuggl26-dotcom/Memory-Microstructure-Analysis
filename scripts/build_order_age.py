"""5 つの時間帯ごとに「注文年齢」(板に置かれてから消えるまでの時間)の分布を出す。

## 注文年齢に何を使ったか

L2 の `lifecycle` 表の `lifetime_ns` = `ts_close - ts_open`。
1 注文 1 行で、板に載った瞬間から終端イベント(約定完了・取消・拒否)までの
経過時間そのものである。約定記録から推定した代理変数ではない。

**この表は手元にある。** `build_order_size.py` の docstring には
「lifecycle は DEEP_ARCHIVE にあり読めない」と書いてあるが、それは
2026-08-28 時点で誤りで、8 銘柄ぶんが次の場所に落ちている:

    E:/hlpipe/l2/coin=xyz%3A<COIN>/lifecycle/dt=YYYY-MM-DD/     (7 銘柄)
    C:/Users/ii562/hl-l4-pipeline/data/l2_v99/lifecycle/        (xyz:DRAM)

## どの注文を数えたか

各注文を **発注時刻 `ts_open` が入る時間帯**に割り当てる。
「その時間帯に置かれた注文は、どれだけ生きたか」を見る指標であって、
「その時間帯に板に載っていた注文の滞在時間」ではない(別の量なので混同しない)。

除外するもの:

| 除外 | 理由 |
|---|---|
| `is_rejected` | 板に載らずに拒否された。年齢の概念がない |
| `is_orphan` | `open` を観測していないので `ts_open` が無い |
| `is_censored` | 標本期間の終端でまだ生きている。年齢は下限値でしかない |
| `tif` ∈ {Ioc, FrontendMarket, LiquidationMarket} | 板に滞留しない種別。年齢は定義上ほぼ 0 で、混ぜると分布が 0 の棘に潰れる |

残るのは `tif` ∈ {Alo, Gtc} の**板に置かれた指値注文**。除外した各群の件数も
CSV に出すので、割合はそちらで確認できる。

## 時間帯

「開場」「閉場」はニューヨーク証券取引所の 9:30 / 16:00。標本期間 2026-05-04〜08-10 は
全日が米国東部夏時間(EDT = UTC−4)なので 13:30 / 20:00 UTC にあたる。

**「昼 12 時」「夜 12 時」は米国東部時間として読んだ。** UTC の 12 時と読むと
「昼 12 時の 1 時間」(12:00–13:00 UTC)が「開場前 1 時間」(12:30–13:30 UTC)と
30 分重なってしまい、5 つの窓が独立しない。東部時間で読むと 1 日を過不足なく
5 点で刻める。念のため UTC 読みの 2 窓も `_alt` として併せて集計してある。

| 呼び名 | UTC | 米国東部時間 |
|---|---|---|
| 開場前 1 時間 | 12:30–13:30 | 8:30–9:30 |
| 開場後 1 時間 | 13:30–14:30 | 9:30–10:30 |
| 昼 12 時から 1 時間 | 16:00–17:00 | 12:00–13:00 |
| 閉場前 1 時間 | 19:00–20:00 | 15:00–16:00 |
| 夜 12 時から 1 時間 | 04:00–05:00 | 0:00–1:00 |
| (参考)昼 12 時 UTC | 12:00–13:00 | 8:00–9:00 |
| (参考)夜 12 時 UTC | 00:00–01:00 | 20:00–21:00(前日の引け後) |

## 立会日だけを使う

「開場前」「閉場前」は原市場が開いている日にしか意味がないので、
`exchange_calendars` の XNYS で立会日を判定し、**半日立会も除く**。
5 つの窓すべてで同じ日集合を使わないと窓どうしを比べられないため、
窓ごとに日を変えることはしない。除外した日は実行時に表示する。

## 実装

97GB の parquet を全部メモリに載せられないので、日ごとに読んで
**対数階級のヒストグラムに足し込む**。階級は 1 桁 200 分割なので、
階級内の相対幅は 10^(1/200) − 1 = 1.16%。ここから読む分位点の誤差はその半分以下で、
本文で使う桁には影響しない。閾値以下の割合(1ms 未満など)はヒストグラムを
経由せず厳密に数えている。

x が確定する時刻 / y の期間: 該当なし(実測量の記述統計であって予測ではない)。

    uv run python scripts/build_order_age.py --coin xyz:MU
    uv run python scripts/build_order_age.py --all
出力: data/order_age_stats_<coin>.csv, data/order_age_hist_<coin>.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

# (呼び名, 開始分, 終了分) すべて UTC の分。終了は含まない
WINDOWS = [
    ("開場前 1 時間", 12 * 60 + 30, 13 * 60 + 30),
    ("開場後 1 時間", 13 * 60 + 30, 14 * 60 + 30),
    ("昼 12 時から 1 時間", 16 * 60, 17 * 60),
    ("閉場前 1 時間", 19 * 60, 20 * 60),
    ("夜 12 時から 1 時間", 4 * 60, 5 * 60),
]
# UTC 読みをした場合の 2 窓。参考値として同時に集計する
WINDOWS_ALT = [
    ("(参考)昼 12 時 UTC", 12 * 60, 13 * 60),
    ("(参考)夜 12 時 UTC", 0, 60),
]
ALL_WINDOWS = WINDOWS + WINDOWS_ALT
WNAMES = [w[0] for w in ALL_WINDOWS]

RESTING_TIF = ["Alo", "Gtc"]          # 板に滞留する種別
LO, HI, BPD = -9, 7, 200              # log10 秒の下限・上限・1 桁あたりの階級数
NB = (HI - LO) * BPD                  # 3200 階級
EDGES = np.logspace(LO, HI, NB + 1)
# 厳密に数える閾値(秒)
THRESH = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 60.0, 600.0, 3600.0]

COINS = ["xyz:MU", "xyz:SNDK", "xyz:SKHX", "xyz:SMSN",
         "xyz:KIOXIA", "xyz:INTC", "xyz:AMD", "xyz:DRAM"]


def lifecycle_dir(coin: str) -> Path:
    """銘柄名から手元の lifecycle ディレクトリを引く。"""
    if coin == "xyz:DRAM":
        return Path("C:/Users/ii562/hl-l4-pipeline/data/l2_v99/lifecycle")
    return Path("E:/hlpipe/l2") / f"coin={coin.replace(':', '%3A')}" / "lifecycle"


def sessions(days: list[str]) -> tuple[set, set]:
    """(全日立会の日, 半日立会の日) を返す。半日立会は使わない。"""
    cal = xc.get_calendar("XNYS")
    ss = cal.sessions_in_range(days[0], days[-1])
    early = {x.date() for x in cal.early_closes.intersection(ss)}
    return {x.date() for x in ss} - early, early


def window_expr() -> pl.Expr:
    """UTC の分から窓の名前を作る。どの窓にも入らなければ null。"""
    mo = (pl.col("ts_open").cast(pl.Datetime("ns")).dt.hour().cast(pl.Int32) * 60
          + pl.col("ts_open").cast(pl.Datetime("ns")).dt.minute().cast(pl.Int32))
    e = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for name, lo, hi in ALL_WINDOWS:
        e = e.when((mo >= lo) & (mo < hi)).then(pl.lit(name))
    return e.otherwise(pl.lit(None, dtype=pl.String))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    targets = COINS if a.all else [a.coin]
    if not targets or targets == [None]:
        raise SystemExit("--coin か --all を指定すること")
    for c in targets:
        run(c)


def run(coin: str) -> None:
    tag = coin.replace(":", "_")
    d = lifecycle_dir(coin)
    parts = sorted(p for p in d.glob("dt=*") if (p / "_SUCCESS").exists())
    if not parts:
        raise SystemExit(f"lifecycle が無い: {d}")
    days = [p.name[3:] for p in parts]
    keep, early = sessions(days)
    print(f"\n=== {coin} ===  {len(days)} 日 ({days[0]}〜{days[-1]})")
    print(f"[立会日] 全日立会 {len(keep)} 日 / 半日立会 {len(early)} 日"
          f"{' ' + ', '.join(str(x) for x in sorted(early)) if early else ''}")

    # 蓄積器。窓 × 階級
    hist = {w: np.zeros(NB, dtype=np.int64) for w in WNAMES}
    acc = {w: {"n": 0, "sum_s": 0.0, "zero": 0, "max_s": 0.0,
               **{f"lt_{t}": 0 for t in THRESH}} for w in WNAMES}
    # 除外の内訳(窓に入る行だけを対象に数える)
    drop = {w: {"rejected": 0, "orphan": 0, "censored": 0, "nonresting": 0,
                "other_tif": 0} for w in WNAMES}

    t0 = time.time()
    for i, p in enumerate(parts):
        f = sorted(p.glob("part-*.parquet"))
        # ファイルは 1 日 1 回だけ読む。窓と立会日で絞ったあとは小さいので
        # メモリに置いて 3 種類の集計を回す(再スキャンより速い)
        lf = (
            pl.scan_parquet(f)
            .select("ts_open", "lifetime_ns", "tif",
                    "is_rejected", "is_orphan", "is_censored")
            # ts_open が無い行(孤児)は窓に割り当てられないので先に落とす
            .filter(pl.col("ts_open").is_not_null())
            .with_columns(w=window_expr())
            .filter(pl.col("w").is_not_null())
            .with_columns(
                d=(pl.col("ts_open").cast(pl.Datetime("ns")) -
                   pl.duration(hours=4)).dt.date())
            .filter(pl.col("d").is_in(list(keep)))
            .collect()
            .lazy()
        )
        # 除外の内訳
        dd = lf.group_by("w").agg(
            rejected=pl.col("is_rejected").sum(),
            orphan=pl.col("is_orphan").sum(),
            censored=pl.col("is_censored").sum(),
            nonresting=(~pl.col("is_rejected") & ~pl.col("is_orphan")
                        & ~pl.col("is_censored")
                        & pl.col("tif").is_in(["Ioc", "FrontendMarket",
                                               "LiquidationMarket"])).sum(),
            other_tif=(~pl.col("is_rejected") & ~pl.col("is_orphan")
                       & ~pl.col("is_censored")
                       & ~pl.col("tif").is_in(RESTING_TIF + ["Ioc", "FrontendMarket",
                                                            "LiquidationMarket"])).sum(),
        ).collect()
        for r in dd.iter_rows(named=True):
            for k in drop[r["w"]]:
                drop[r["w"]][k] += int(r[k])

        keep_lf = lf.filter(
            ~pl.col("is_rejected") & ~pl.col("is_orphan") & ~pl.col("is_censored")
            & pl.col("tif").is_in(RESTING_TIF)
            & pl.col("lifetime_ns").is_not_null()
        ).with_columns(s=pl.col("lifetime_ns") / 1e9)

        # 集計 1: 厳密な統計量
        agg = keep_lf.group_by("w").agg(
            n=pl.len(), sum_s=pl.col("s").sum(), max_s=pl.col("s").max(),
            zero=(pl.col("lifetime_ns") == 0).sum(),
            **{f"lt_{t}": (pl.col("s") < t).sum() for t in THRESH},
        ).collect()
        for r in agg.iter_rows(named=True):
            A = acc[r["w"]]
            A["n"] += int(r["n"]); A["sum_s"] += float(r["sum_s"])
            A["zero"] += int(r["zero"])
            A["max_s"] = max(A["max_s"], float(r["max_s"] or 0.0))
            for t in THRESH:
                A[f"lt_{t}"] += int(r[f"lt_{t}"])

        # 集計 2: 対数ヒストグラム(0 は階級に入らないので別勘定)
        hb = keep_lf.filter(pl.col("lifetime_ns") > 0).with_columns(
            b=(((pl.col("s").log10() - LO) * BPD).floor()
               .clip(0, NB - 1).cast(pl.Int32))
        ).group_by("w", "b").agg(c=pl.len()).collect()
        for r in hb.iter_rows(named=True):
            hist[r["w"]][r["b"]] += r["c"]

        if (i + 1) % 20 == 0 or i + 1 == len(parts):
            print(f"  {i + 1:>3}/{len(parts)} 日  {time.time() - t0:6.1f}s")

    # ---- 出力 ----
    rows, hrows = [], []
    for w in WNAMES:
        A, D = acc[w], drop[w]
        n = A["n"]
        if n == 0:
            print(f"  [警告] {w}: 対象 0 件")
            continue
        h = hist[w]
        # 0 の塊 + 階級の度数 = 総数。ここがずれたら階級の外に落ちた行がある
        assert A["zero"] + int(h.sum()) == n, (
            f"{w}: 度数の合計 {A['zero'] + int(h.sum()):,} != 総数 {n:,}")
        q = quantiles(h, A["zero"], n)
        rows.append({
            "window": w, "n": n,
            "mean_s": A["sum_s"] / n, "max_s": A["max_s"],
            **{f"p{int(k * 100) if k >= 0.01 else k}": v for k, v in q.items()},
            "share_zero": A["zero"] / n,
            **{f"share_lt_{t}": A[f"lt_{t}"] / n for t in THRESH},
            "n_rejected": D["rejected"], "n_orphan": D["orphan"],
            "n_censored": D["censored"], "n_nonresting": D["nonresting"],
            "n_other_tif": D["other_tif"],
        })
        for b in np.nonzero(h)[0]:
            hrows.append({"window": w, "lo": float(EDGES[b]),
                          "hi": float(EDGES[b + 1]), "count": int(h[b])})

    S = pl.DataFrame(rows)
    S.write_csv(ROOT / "data" / f"order_age_stats_{tag}.csv")
    pl.DataFrame(hrows).write_csv(ROOT / "data" / f"order_age_hist_{tag}.csv")

    print(f"\n{'時間帯':<24}{'本数':>12}{'中央値':>11}{'第1四分位':>11}"
          f"{'第3四分位':>11}{'95%点':>11}{'平均':>11}{'1秒未満':>9}")
    for r in S.iter_rows(named=True):
        print(f"{r['window']:<24}{r['n']:>12,}{fmt(r['p50']):>11}{fmt(r['p25']):>11}"
              f"{fmt(r['p75']):>11}{fmt(r['p95']):>11}{fmt(r['mean_s']):>11}"
              f"{r['share_lt_1.0'] * 100:>8.1f}%")
    print(f"\n[out] data/order_age_stats_{tag}.csv, data/order_age_hist_{tag}.csv")


def quantiles(h: np.ndarray, n_zero: int, n: int) -> dict:
    """対数階級の度数から分位点を読む。0 の塊は最左に置く。

    階級内は対数一様と仮定して線形に内挿する。階級の相対幅が 1.16% なので、
    ここから来る誤差は分位点の値の 0.6% 未満。
    """
    cum = np.concatenate([[n_zero], n_zero + np.cumsum(h)])   # cum[i] = 階級 i 未満の件数
    out = {}
    for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        t = p * n
        if t <= n_zero:
            out[p] = 0.0
            continue
        i = int(np.searchsorted(cum, t, side="left")) - 1
        i = min(max(i, 0), len(h) - 1)
        c0, c1 = cum[i], cum[i + 1]
        f = 0.5 if c1 == c0 else (t - c0) / (c1 - c0)
        out[p] = float(10 ** (np.log10(EDGES[i]) + f * (np.log10(EDGES[i + 1])
                                                        - np.log10(EDGES[i]))))
    return out


def fmt(s: float) -> str:
    """秒を読みやすい単位で。"""
    if s == 0:
        return "0"
    if s < 1e-3:
        return f"{s * 1e6:.0f}us"
    if s < 1:
        return f"{s * 1e3:.1f}ms"
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


if __name__ == "__main__":
    main()
