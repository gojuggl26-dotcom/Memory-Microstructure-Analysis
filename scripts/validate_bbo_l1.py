"""l1 から組み直した bbo が、l2 の bbo とどれだけ一致するかを日ごとに測る。

    uv run python scripts/validate_bbo_l1.py --coin xyz:MU

出力: data/bbo_l1_check_<coin>.csv と charts/<coin>_bbo_l1_check.png

★なぜ要るか
------------
xyz:INTC の `l2/bbo` は DEEP_ARCHIVE にあり読めないので、板は `l1` から
組み直すしかない(`build_bbo_l1.py`)。その組み直しがどれだけ信用できるかは、
**両方そろっている xyz:MU** で測るしかない。ここで測った誤差が、そのまま
INTC の特徴量に乗る誤差の目安になる。

測る量は 1 秒格子(後ろ向き、厳密に t 以前の最後の行)で

  * best_bid / best_ask の完全一致率と 1 ティック以内の率
  * mid の差(bp)の分布
  * スプレッドの中央値の差
  * 最良の数量の中央値の差と、OBI1 の相関・符号一致率

x が確定する時刻 / y の期間: 該当なし(2 つの再構成の突合であって予測ではない)。
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from _chartstyle import C1, C2, C3, CM, D, np as _np, plt, save  # noqa: F401

NG = 86_400


def sample(df: pl.DataFrame, tg: np.ndarray):
    ts = df["ts"].cast(pl.Int64).to_numpy()
    j = np.searchsorted(ts, tg, side="right") - 1
    ok = j >= 0
    j = np.maximum(j, 0)
    return tuple(np.where(ok, df[c].to_numpy()[j], np.nan)
                 for c in ("best_bid", "best_ask", "bid_sz", "ask_sz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    real = pl.read_parquet(D / f"bbo_{tag}.parquet")
    mine = pl.read_parquet(D / f"bbo_{tag}_l1.parquet")
    days = sorted(set(real["dt"].unique().to_list())
                  & set(mine["dt"].unique().to_list()))
    print(f"共通 {len(days)} 日")

    rows = []
    for d in days:
        lo = int(dt.datetime.fromisoformat(d + "T00:00:00+00:00").timestamp()) * 10 ** 9
        tg = lo + np.arange(NG, dtype=np.int64) * 10 ** 9
        rb, ra, rqb, rqa = sample(real.filter(pl.col("dt") == d).sort("ts"), tg)
        mb, ma, mqb, mqa = sample(mine.filter(pl.col("dt") == d).sort("ts"), tg)
        m = np.isfinite(rb) & np.isfinite(mb) & (ra > rb)
        if m.sum() < 1000:
            continue
        rm = (rb + ra) / 2
        mm = (mb + ma) / 2
        tick = np.where(rm >= 1000, 0.1, 0.01)
        ro = np.where(rqb + rqa > 0, (rqb - rqa) / np.maximum(rqb + rqa, 1e-12), np.nan)
        mo = np.where(mqb + mqa > 0, (mqb - mqa) / np.maximum(mqb + mqa, 1e-12), np.nan)
        k = m & np.isfinite(ro) & np.isfinite(mo)
        dmid = (mm - rm) / rm * 1e4
        rows.append({
            "dt": d, "n": int(m.sum()),
            "bid_exact": float(np.mean(np.abs(mb - rb)[m] < 1e-9)),
            "ask_exact": float(np.mean(np.abs(ma - ra)[m] < 1e-9)),
            "bid_1tick": float(np.mean(np.abs(mb - rb)[m] <= tick[m] * 1.001)),
            "ask_1tick": float(np.mean(np.abs(ma - ra)[m] <= tick[m] * 1.001)),
            "mid_abs_p50": float(np.nanmedian(np.abs(dmid[m]))),
            "mid_abs_p90": float(np.nanpercentile(np.abs(dmid[m]), 90)),
            "mid_abs_p99": float(np.nanpercentile(np.abs(dmid[m]), 99)),
            "mid_abs_max": float(np.nanmax(np.abs(dmid[m]))),
            # ★中央値だけ見ると「完全一致」に見えるが、稀に大きく外れる瞬間がある。
            #   裾を必ず一緒に出す
            "mid_gt_1bp": float(np.mean(np.abs(dmid[m]) > 1.0)),
            "mid_gt_10bp": float(np.mean(np.abs(dmid[m]) > 10.0)),
            "obi_gt_02": float(np.mean(np.abs(mo - ro)[k] > 0.2)),
            "spread_real": float(np.nanmedian((ra - rb)[m] / rm[m] * 1e4)),
            "spread_mine": float(np.nanmedian((ma - mb)[m] / mm[m] * 1e4)),
            "szb_real": float(np.nanmedian(rqb[m])), "szb_mine": float(np.nanmedian(mqb[m])),
            "sza_real": float(np.nanmedian(rqa[m])), "sza_mine": float(np.nanmedian(mqa[m])),
            "obi_corr": float(np.corrcoef(ro[k], mo[k])[0, 1]),
            "obi_sign": float(np.mean(np.sign(ro[k]) == np.sign(mo[k]))),
            "cross_mine": float(np.mean((mb >= ma)[np.isfinite(mb)])),
        })
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{len(days)} ({d})", flush=True)

    t = pl.DataFrame(rows)
    t.write_csv(D / f"bbo_l1_check_{tag}.csv")
    q = t.select(pl.col(["bid_exact", "ask_exact", "bid_1tick", "ask_1tick",
                         "mid_abs_p90", "mid_gt_1bp", "mid_gt_10bp",
                         "obi_corr", "obi_sign", "obi_gt_02"]).median())
    print("\n中央値:", q.to_dicts()[0])
    print("最悪日:", t.sort("ask_exact").head(3).select(
        "dt", "bid_exact", "ask_exact", "mid_abs_p90", "obi_corr").to_dicts())

    x = np.arange(t.height)
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 6.2))
    a0 = ax[0, 0]
    a0.plot(x, t["bid_exact"] * 100, color=C1, lw=1.1, label="best_bid")
    a0.plot(x, t["ask_exact"] * 100, color=C2, lw=1.1, label="best_ask")
    a0.plot(x, t["bid_1tick"] * 100, color=C1, lw=0.8, ls=":", label="best_bid (1tick 以内)")
    a0.plot(x, t["ask_1tick"] * 100, color=C2, lw=0.8, ls=":", label="best_ask (1tick 以内)")
    a0.set_title("A. 最良気配が l2 の bbo と一致した割合 (1 秒格子)")
    a0.set_ylabel("%")
    a0.set_ylim(80, 100.5)
    a0.legend(fontsize=7, frameon=False)

    a1 = ax[0, 1]
    a1.plot(x, t["mid_abs_p50"], color=C3, lw=1.1, label="中央値")
    a1.plot(x, t["mid_abs_p90"], color=C1, lw=1.1, label="90% 点")
    a1.set_yscale("symlog", linthresh=1e-4)
    a1.set_title("B. mid のずれ |Δ| (bp)")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[1, 0]
    a2.plot(x, t["spread_real"], color=CM, lw=1.4, label="l2 の bbo")
    a2.plot(x, t["spread_mine"], color=C2, lw=1.0, label="l1 から再構成")
    a2.set_title("C. スプレッドの日中央値 (bp)")
    a2.set_xlabel("日 (標本の通し番号)")
    a2.legend(fontsize=7, frameon=False)

    a3 = ax[1, 1]
    a3.plot(x, t["obi_corr"], color=C1, lw=1.1, label="OBI1 の相関")
    a3.plot(x, t["obi_sign"], color=C3, lw=1.1, label="OBI1 の符号一致率")
    a3.set_ylim(0, 1.02)
    a3.set_title("D. 最良気配の数量から作った OBI の一致")
    a3.set_xlabel("日 (標本の通し番号)")
    a3.legend(fontsize=7, frameon=False)
    save(fig, f"{tag}_bbo_l1_check.png",
         f"{a.coin}: l1 から組み直した板と l2 の bbo の突合 ({t.height} 日)")


if __name__ == "__main__":
    main()
