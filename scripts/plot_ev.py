"""AS(S,D) と EV(S,D) のヒートマップ — どのセルならクオートすべきか。

    uv run python scripts/plot_ev.py --coin xyz:MU
出力: charts/<coin>_ev_<feat>.png    特徴量ごとに 1 枚
      data/ev_summary_<coin>.csv     セルごとの EV と日次 Newey-West 誤差
      data/ev_positive_<coin>.csv    EV>0 と判定できたセルの一覧

【読み方】
EV = P(約定) x [ Spread/2 - Fee + (買いなら +AS / 売りなら -AS) ]
Spread/2 は**約定 2ms 前**のスプレッドの半分。約定時刻ちょうどの板はもう
約定後で、スプレッドが 3 倍に開いて見えるので使ってはいけない。
AS は約定時刻 tau から tau+h までの mid のリターン (符号なし)。

誤差は**日ごとにセル値を作ってから Newey-West(14 日)**で取る。プールの
標準誤差は最大 16 倍過小になる (burstiness の報告)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_maker import MAKER_FEE_BP  # noqa: E402
from plot_heat import DIV, SEQ, draw  # noqa: E402
from plot_vol import INK, INK2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = 10
NW_LAGS = 14
MIN_FILL = 200          # これ未満の約定しかないセルは判定しない
LAB = {"obi": "OBI 板の不均衡", "ofi_10s": "OFI 注文流の不均衡 (10 秒)",
       "ai_net_10s": "攻撃的注文の符号つき数量 (10 秒)"}
SNAME = {1: "買い指値 (bid)", -1: "売り指値 (ask)"}


def nwse(v, q=NW_LAGS):
    v = v[np.isfinite(v)]
    n = v.size
    if n < q + 3:
        return np.nan, np.nan, n
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, q + 1):
        g += 2.0 * (1.0 - lg / (q + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return float(v.mean()), float(np.sqrt(max(g, 0.0) / n)), n


def cellmap(P, f, sd, h, col):
    d = P.filter((pl.col("feat") == f) & (pl.col("side") == sd)
                 & (pl.col("h") == h))
    a = np.full((NB, NB), np.nan)
    for r in d.iter_rows(named=True):
        a[r["s_dec"], r["q_dec"]] = r[col]
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--h", type=float, default=10.0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    P = pl.read_csv(D / f"ev_pooled_{tag}.csv")
    Y = pl.read_parquet(D / f"ev_daily_{tag}.parquet")

    P = P.with_columns(
        p_fill=pl.col("n_fill") / pl.col("n"),
        AS=pl.col("sum_as") / pl.col("n_fill"),
        half_spread=pl.col("sum_hs") / pl.col("n_fill"),
        half_spread_post=pl.col("sum_hs_post") / pl.col("n_fill"),
        impact=pl.col("sum_imp") / pl.col("n_fill"),
        edge=pl.col("sum_edge") / pl.col("n_fill"),
        wait_s=pl.col("sum_wait") / pl.col("n_fill"))
    P = P.with_columns(
        core=pl.col("half_spread") - MAKER_FEE_BP
        + pl.col("side") * pl.col("AS"),
        core_realized=pl.col("edge") - MAKER_FEE_BP
        + pl.col("side") * pl.col("AS"))
    P = P.with_columns(EV=pl.col("p_fill") * pl.col("core"),
                       EV_realized=pl.col("p_fill") * pl.col("core_realized"))

    # 日ごとにセル値を作り、Newey-West で誤差を取る
    hs_daily = sorted(Y["h"].unique().to_list())
    Y = Y.filter(pl.col("n_fill") > 0).with_columns(
        core_d=pl.col("sum_hs") / pl.col("n_fill") - MAKER_FEE_BP
        + pl.col("side") * pl.col("sum_as") / pl.col("n_fill"))
    rows = []
    for (f, sd, h, s, dd), grp in Y.group_by(
            ["feat", "side", "h", "s_dec", "q_dec"], maintain_order=True):
        m, se, nd = nwse(grp["core_d"].to_numpy())
        rows.append({"feat": f, "side": sd, "h": h, "s_dec": s, "q_dec": dd,
                     "core_daily": m, "core_se": se, "n_day": nd,
                     "core_t": m / se if se and se > 0 else np.nan})
    Nw = pl.DataFrame(rows)
    S = P.join(Nw, on=["feat", "side", "h", "s_dec", "q_dec"], how="left")
    S.write_csv(D / f"ev_summary_{tag}.csv")

    ok = S.filter((pl.col("n_fill") >= MIN_FILL) & (pl.col("EV") > 0)
                  & (pl.col("core_t") > 3.29)).sort("EV", descending=True)
    ok.write_csv(D / f"ev_positive_{tag}.csv")
    tot = S.filter(pl.col("h").is_in(hs_daily))
    print(f"{a.coin}: EV>0 かつ日次 NW の t>3.29 のセル {ok.height} 件 "
          f"(検査対象 {tot.filter(pl.col('n_fill') >= MIN_FILL).height} セル)")
    if ok.height:
        with pl.Config(tbl_rows=20, tbl_width_chars=160):
            print(ok.select("feat", "side", "h", "s_dec", "q_dec", "n", "n_fill",
                            "p_fill", "half_spread", "AS", "core", "EV",
                            "core_t").head(15))

    ch = ROOT / "charts"
    for f in LAB:
        fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
        fig.suptitle(f"{a.coin}  {LAB[f]} — AS(S,D) と EV(S,D)・h = {a.h:g} 秒"
                     f"・縦 S1..S10 × 横 D1..D10・97 日",
                     color=INK, fontsize=13, y=0.985)
        draw(axes[0, 0], cellmap(P, f, 1, a.h, "p_fill") * 100,
             "(a) P(Fill) % — 買い指値", SEQ, False, "{:.0f}")
        draw(axes[0, 1], cellmap(P, f, 1, a.h, "AS"),
             "(b) AS = E[r(tau→tau+h) | Fill] bp — 買い指値", DIV, True,
             "{:+.2f}")
        draw(axes[0, 2], cellmap(P, f, 1, a.h, "half_spread"),
             "(c) Spread/2 (約定 2ms 前) bp — 買い指値", SEQ, False, "{:.2f}")
        draw(axes[1, 0], cellmap(P, f, 1, a.h, "EV"),
             "(d) EV_bid bp / 1 発注", DIV, True, "{:+.2f}")
        draw(axes[1, 1], cellmap(P, f, -1, a.h, "EV"),
             "(e) EV_ask bp / 1 発注", DIV, True, "{:+.2f}")
        draw(axes[1, 2], cellmap(P, f, 1, a.h, "core"),
             "(f) 1 約定あたりの中身 Spread/2 − Fee + AS bp", DIV, True,
             "{:+.2f}")
        fig.tight_layout(rect=(0, 0.005, 1, 0.962))
        out = ch / f"{tag}_ev_{f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  {out.name}")


if __name__ == "__main__":
    main()
