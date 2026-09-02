"""口座ごとの 11 指標を組み立て、指標どうしの重なりと分布を測る。

【対象の選び方】
指標は割合なので、分母が小さい口座では暴れる。**結果を見る前に**
「置いた本数が MIN_ORDERS 本以上」という閾値だけで選ぶ
(約定した口座を選ぶと生存バイアスになる)。

【★ 4 つの「約定割合」は近い量である】
fill rate / fill probability / volume executed÷quoted / fill-to-order ratio は
分子と分母の取り方が違うだけで同じことを測っている。独立な 4 つの発見が
あるかのように並べないため、**実測した相関を出す**(出力 wfmetrics_corr)。

【出力】
    data/wfmetrics_<coin>.csv        口座 × 11 指標(閾値を満たす口座)
    data/wfmetrics_corr_<coin>.csv   11 指標どうしの順位相関
    data/wfmetrics_summary_<coin>.json 見出しの数値と検算

実行例:
    uv run python scripts/fit_wallet_fill.py --coin xyz:MU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SZ_LOT = 0.001
MIN_ORDERS = 10_000          # ★結果を見る前に決めた閾値
ADV_H = (1, 10, 60)
NS = 1_000_000_000

METRICS = ("fill_rate", "fill_prob", "fill_to_order", "fill_to_cancel",
           "maker_share", "vol_exec_quoted", "avg_ttf_s", "partial_freq",
           "fill_size", "queue_at_fill", "adverse_10s")
JP = {"fill_rate": "fill rate", "fill_prob": "fill probability",
      "fill_to_order": "fill-to-order ratio", "fill_to_cancel": "fill-to-cancel ratio",
      "maker_share": "maker execution share", "vol_exec_quoted": "volume exec / quoted",
      "avg_ttf_s": "average time to fill", "partial_freq": "partial-fill frequency",
      "fill_size": "fill size", "queue_at_fill": "queue-position-at-fill",
      "adverse_10s": "adverse selection after fill"}


def placebo_adverse(f: pl.DataFrame, tag: str, shift_s: int = 3600):
    """★帰無対照 — 約定した時刻を日の中でずらして同じ量を測る。

    「約定の直後に不利へ動く」が本当に**約定の瞬間に固有**かを確かめる。
    約定時刻を +shift_s ずらし(側はそのまま)、同じ式で符号つきの値動きを出す。
    ずらしても同じだけ出るなら、それは逆選択ではなく相場の傾き(や日内の形)である。
    """
    bb = pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet").sort(["dt", "ts"])
    out = []
    for dt, s in f.group_by("dt"):
        d0 = dt[0] if isinstance(dt, tuple) else dt
        b = bb.filter(pl.col("dt") == d0)
        if not b.height:
            continue
        bts = b["ts"].cast(pl.Int64).to_numpy()
        bmid = (b["best_bid"].to_numpy() + b["best_ask"].to_numpy()) / 2.0
        good = np.isfinite(bmid) & (b["best_ask"].to_numpy()
                                    > b["best_bid"].to_numpy())
        bts, bmid = bts[good], bmid[good]
        if bts.size < 2:
            continue

        def mid_at(t):
            j = np.searchsorted(bts, t, side="left") - 1
            return np.where(j >= 0, bmid[np.clip(j, 0, bmid.size - 1)], np.nan)

        t = s["ts"].to_numpy()
        lo, hi = bts[0], bts[-1]
        tp = lo + (t + shift_s * NS - lo) % max(hi - lo, 1)   # 日の中で巡回
        m0, m1 = mid_at(tp), mid_at(tp + 10 * NS)
        sgn = np.where(s["is_bid"].to_numpy(), -1.0, 1.0)
        out.append(pl.DataFrame({"wid": s["wid"],
                                 "adv10_pl": sgn * (m1 - m0) / m0 * 1e4}))
    if not out:
        return None
    return (pl.concat(out).filter((pl.col("wid") >= 0)
                                  & pl.col("adv10_pl").is_finite())
            .group_by("wid").agg(adv10_placebo=pl.col("adv10_pl").mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--min-orders", type=int, default=MIN_ORDERS)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    w = pl.read_parquet(ROOT / "data" / f"wfill_wallet_{tag}.parquet")
    f = pl.read_parquet(ROOT / "data" / f"wfill_fills_{tag}.parquet")
    tot_fill_v = float(w["v_fill"].sum())
    print(f"[load] 口座 {w.height:,} / 約定 {f.height:,} 件", file=sys.stderr)

    # ---- 約定 1 件ごとの記録から、口座別の中央値などを作る ----
    fa = (f.filter(pl.col("wid") >= 0)
          .with_columns(ttf_s=pl.col("ttf_ns") / NS,
                        fill_c=pl.col("fill_sz") * SZ_LOT,
                        qa_c=pl.when(pl.col("qa_fill") >= 0)
                              .then(pl.col("qa_fill") * SZ_LOT).otherwise(None))
          .group_by("wid").agg(
              med_ttf_s=pl.col("ttf_s").median(),
              med_fill_c=pl.col("fill_c").median(),
              mean_fill_c=pl.col("fill_c").mean(),
              med_qa_c=pl.col("qa_c").median(),
              share_q0=(pl.col("qa_c") == 0).mean(),
              # ★NaN は mean/median に伝播する。bbo が無い瞬間の約定が
              #   766 万件中 10 件あり、それだけで口座の平均が NaN に化けた
              **{f"adv{h}": pl.col(f"adv{h}").filter(
                  pl.col(f"adv{h}").is_finite()).mean() for h in ADV_H},
              **{f"adv{h}_med": pl.col(f"adv{h}").filter(
                  pl.col(f"adv{h}").is_finite()).median() for h in ADV_H},
              n_adv_bad=(~pl.col("adv10").is_finite()).sum(),
              n_fillev_chk=pl.len()))

    pla = placebo_adverse(f, tag)
    d = (w.filter(pl.col("n_ord") >= a.min_orders)
         .join(fa, on="wid", how="left")
         .join(pla if pla is not None else
               pl.DataFrame({"wid": [], "adv10_placebo": []},
                            schema={"wid": pl.Int64, "adv10_placebo": pl.Float64}),
               on="wid", how="left")
         .with_columns(
             # ① 本数ベースの約定率
             fill_rate=pl.col("n_fill") / pl.col("n_ord"),
             # ② 数量で重みづけた約定確率
             #    Σ(1 枚でも約定した注文の当初数量) / Σ(置いた当初数量)
             #    ①が本数、②が数量。大口ほど②に効く。⑥との違いは
             #    「約定した注文を丸ごと数える」か「約定した枚数だけ数える」か
             fill_prob=pl.col("v_ord_filled") / pl.col("v_ord"),
             # ③ 全量約定だけを約定と数える
             fill_to_order=pl.col("n_full") / pl.col("n_ord"),
             # ④ 約定した本数 / 1 枚も約定せず取り消した本数
             fill_to_cancel=pl.col("n_fill")
             / pl.when(pl.col("n_cxl") > 0).then(pl.col("n_cxl")).otherwise(None),
             # ⑤ 市場全体のメイカー約定数量に占める割合
             maker_share=pl.col("v_fill") / tot_fill_v,
             # ⑥ 置いた数量のうち実際に約定した割合
             vol_exec_quoted=pl.col("v_fill") / pl.col("v_ord"),
             # ⑦ 置いてから最初の約定までの平均(秒)
             avg_ttf_s=pl.col("sum_ttf_ns") / pl.col("n_fill") / NS,
             # ⑧ 約定したうち全量に至らなかった割合
             partial_freq=pl.col("n_part")
             / pl.when(pl.col("n_fill") > 0).then(pl.col("n_fill")).otherwise(None),
             # ⑨ 約定 1 件あたりの数量(契約)
             fill_size=pl.col("v_fill") * SZ_LOT
             / pl.when(pl.col("n_fillev") > 0).then(pl.col("n_fillev")).otherwise(None),
             # ⑩ 約定した瞬間に前にいた数量(契約、中央値)
             queue_at_fill=pl.col("med_qa_c"),
             # ⑪ 約定後 10 秒の不利方向の値動き(bp、平均)
             adverse_10s=pl.col("adv10"),
             # 参考
             # 約定した注文が平均してどれだけ埋まったか(⑥ ÷ ②)
             fill_completeness=(pl.col("v_fill")
                                / pl.when(pl.col("v_ord_filled") > 0)
                                    .then(pl.col("v_ord_filled")).otherwise(None)),
             qplace_c=pl.col("sum_qa_place") / pl.col("n_ord") * SZ_LOT,
             v_ord_c=pl.col("v_ord") * SZ_LOT,
             v_fill_c=pl.col("v_fill") * SZ_LOT))

    d = d.sort("maker_share", descending=True)
    keep = (["wid", "user", "n_ord", "n_fill", "n_full", "n_part", "n_cxl",
             "v_ord_c", "v_fill_c", "n_fillev", "qplace_c", "share_q0",
             "med_ttf_s", "med_fill_c", "fill_completeness",
             "adv1", "adv60", "adv10_med", "adv10_placebo", "n_adv_bad"]
            + list(METRICS))
    keep = [c for c in keep if c in d.columns]
    d.select(keep).write_csv(ROOT / "data" / f"wfmetrics_{tag}.csv")

    # ---- 11 指標どうしの順位相関(近い量がどれだけ重なるか) ----
    M = np.column_stack([d[m].to_numpy().astype(np.float64) for m in METRICS])
    ok = np.isfinite(M).all(axis=1)
    R = np.corrcoef(np.column_stack(
        [stats.rankdata(M[ok, j]) for j in range(M.shape[1])]).T)
    pl.DataFrame({"metric": list(METRICS),
                  **{m: R[:, j] for j, m in enumerate(METRICS)}}) \
      .write_csv(ROOT / "data" / f"wfmetrics_corr_{tag}.csv")

    i_fr, i_fp = METRICS.index("fill_rate"), METRICS.index("fill_to_order")
    i_vq = METRICS.index("vol_exec_quoted")
    summ = {
        "n_wallet_all": w.height, "n_wallet_kept": d.height,
        "min_orders": a.min_orders,
        "n_orders_kept": int(d["n_ord"].sum()),
        "share_orders_kept": float(d["n_ord"].sum() / w["n_ord"].sum()),
        "share_fillvol_kept": float(d["v_fill_c"].sum() * 1000 / tot_fill_v),
        "corr_fillrate_filltoorder": float(R[i_fr, i_fp]),
        "corr_fillrate_volexec": float(R[i_fr, i_vq]),
        "corr_filltoorder_volexec": float(R[i_fp, i_vq]),
        "n_corr_obs": int(ok.sum()),
        "top1_maker_share": float(d["maker_share"].max()),
        "top5_maker_share": float(d["maker_share"].head(5).sum()),
        "hhi_maker": float((d["maker_share"] ** 2).sum()),
        "market_fill_rate": float(w["n_fill"].sum() / w["n_ord"].sum()),
        "market_vol_exec_quoted": float(w["v_fill"].sum() / w["v_ord"].sum()),
        # 逆選択: 約定量で重みづけた平均と、帰無対照
        "adv10_volw": float(np.average(
            d["adverse_10s"].to_numpy()[np.isfinite(d["adverse_10s"].to_numpy())],
            weights=d["maker_share"].to_numpy()[
                np.isfinite(d["adverse_10s"].to_numpy())])),
        "adv10_share_positive": float(
            (d["adverse_10s"].to_numpy()[
                np.isfinite(d["adverse_10s"].to_numpy())] > 0).mean()),
        "adv10_placebo_med": float(np.nanmedian(d["adv10_placebo"].to_numpy())),
        "adv10_placebo_share_positive": float(np.nanmean(
            d["adv10_placebo"].to_numpy()[
                np.isfinite(d["adv10_placebo"].to_numpy())] > 0)),
    }
    # 実測 対 帰無対照 の対応のある検定と、規模との関係
    av = d["adverse_10s"].to_numpy()
    pv = d["adv10_placebo"].to_numpy()
    ms = d["maker_share"].to_numpy()
    ok2 = np.isfinite(av) & np.isfinite(pv)
    if ok2.sum() > 10:
        wt = stats.wilcoxon(av[ok2], pv[ok2])
        rs = stats.spearmanr(ms[ok2], av[ok2])
        big = np.argsort(-ms)[:20]
        oth = np.argsort(-ms)[20:]
        summ.update(
            adv10_vs_placebo_share=float((av[ok2] > pv[ok2]).mean()),
            adv10_vs_placebo_p=float(wt.pvalue), adv10_n_pair=int(ok2.sum()),
            adv10_vs_makershare_rho=float(rs.statistic),
            adv10_vs_makershare_p=float(rs.pvalue),
            adv10_top20_med=float(np.nanmedian(av[big])),
            adv10_rest_med=float(np.nanmedian(av[oth])),
            spread_bp=1.245)
    for m in METRICS:
        v = d[m].to_numpy().astype(np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            summ[f"{m}_q"] = [float(x) for x in np.percentile(v, [10, 50, 90])]
    (ROOT / "data" / f"wfmetrics_summary_{tag}.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fit] 閾値 {a.min_orders:,} 本以上 -> {d.height} 口座 "
          f"(注文の {100*summ['share_orders_kept']:.1f}% / "
          f"約定量の {100*summ['share_fillvol_kept']:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
