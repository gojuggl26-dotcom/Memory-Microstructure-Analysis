"""SelectionPenalty(S,D) = E[r | Fill, S, D] − E[r | S, D] のヒートマップ。

    uv run python scripts/plot_penalty.py --coin xyz:MU
出力: charts/<coin>_penalty_<feat>.png
      data/penalty_summary_<coin>.csv

【読み方】
「その状態で、約定という条件を付けるとリターンがどれだけ壊れるか」。
両方とも**同じ窓 T→T+h** で測る。約定時刻 tau 起点にすると約定しなかった
側が定義できないので比較にならない。

買い指値なら penalty が負 = 不利。売り指値は符号が逆になるので、
図では **quoter の損益の向き** に揃えてある (どちらも負が不利)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_heat import DIV, SEQ, draw  # noqa: E402
from plot_vol import INK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = 10
NW_LAGS = 14
LAB = {"obi": "OBI 板の不均衡", "ofi_10s": "OFI 注文流の不均衡 (10 秒)",
       "ai_net_10s": "攻撃的注文の符号つき数量 (10 秒)"}


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


def grid(H, f, sd, h, col):
    d = H.filter((pl.col("feat") == f) & (pl.col("side") == sd)
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
    H = pl.read_csv(D / f"heat_pooled_{tag}.csv")
    Y = pl.read_parquet(D / f"heat_daily_{tag}.parquet").filter(
        pl.col("n_fill") > 0)

    H = H.with_columns(
        uncond=pl.col("sum_ret") / pl.col("n"),
        filled=pl.col("sum_ret_fill") / pl.col("n_fill"),
        uncond_mic=pl.col("sum_retc") / pl.col("n"),
        filled_mic=pl.col("sum_retc_fill") / pl.col("n_fill"),
        p_fill=pl.col("n_fill") / pl.col("n"))
    H = H.with_columns(
        penalty=pl.col("filled") - pl.col("uncond"),
        penalty_mic=pl.col("filled_mic") - pl.col("uncond_mic"))
    # quoter の損益の向きに揃える (どちらの側も負が不利)
    H = H.with_columns(
        penalty_signed=pl.col("side") * pl.col("penalty"),
        penalty_mic_signed=pl.col("side") * pl.col("penalty_mic"),
        alpha_signed=pl.col("side") * pl.col("uncond"))

    Y = Y.with_columns(pen_d=pl.col("side")
                       * (pl.col("sum_ret_fill") / pl.col("n_fill")
                          - pl.col("sum_ret") / pl.col("n")))
    rows = []
    for (f, sd, h, s, dd), grp in Y.group_by(
            ["feat", "side", "h", "s_dec", "q_dec"], maintain_order=True):
        m, se, nd = nwse(grp["pen_d"].to_numpy())
        rows.append({"feat": f, "side": sd, "h": h, "s_dec": s, "q_dec": dd,
                     "pen_daily": m, "pen_se": se, "n_day": nd,
                     "pen_t": m / se if se and se > 0 else np.nan})
    S = H.join(pl.DataFrame(rows), on=["feat", "side", "h", "s_dec", "q_dec"],
               how="left")
    S.write_csv(D / f"penalty_summary_{tag}.csv")

    T = S.filter(pl.col("pen_t").is_not_null() & (pl.col("n_fill") >= 200))
    print(f"{a.coin}: 日次 NW で誤差を付けたセル {T.height}")
    print(f"  penalty < 0 (quoter に不利): {T.filter(pl.col('penalty_signed') < 0).height}")
    print(f"  そのうち t < −3.29 (Bonferroni {T.height} 検定): "
          f"{T.filter(pl.col('pen_t') < -3.29).height}")
    print(f"  penalty > 0 かつ t > +3.29: {T.filter(pl.col('pen_t') > 3.29).height}")

    ch = ROOT / "charts"
    for f in LAB:
        fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
        fig.suptitle(f"{a.coin}  {LAB[f]} — SelectionPenalty(S,D)"
                     f"・h = {a.h:g} 秒・縦 S1..S10 × 横 D1..D10・97 日",
                     color=INK, fontsize=13, y=0.985)
        draw(axes[0, 0], grid(H, f, 1, a.h, "uncond"),
             "(a) 無条件 E[r | S,D] bp (符号なし)", DIV, True, "{:+.2f}")
        draw(axes[0, 1], grid(H, f, 1, a.h, "filled"),
             "(b) 約定条件つき E[r | Fill, S,D] bp — 買い指値", DIV, True,
             "{:+.2f}")
        draw(axes[0, 2], grid(H, f, 1, a.h, "penalty"),
             "(c) ★ SelectionPenalty bp — 買い指値", DIV, True, "{:+.2f}")
        draw(axes[1, 0], grid(H, f, -1, a.h, "penalty_signed"),
             "(d) SelectionPenalty bp — 売り指値 (損益の向きに揃えた)", DIV,
             True, "{:+.2f}")
        draw(axes[1, 1], grid(H, f, 1, a.h, "penalty_mic"),
             "(e) 同じものを microprice で測る — 買い指値", DIV, True,
             "{:+.2f}")
        draw(axes[1, 2], grid(H, f, 1, a.h, "p_fill") * 100,
             "(f) P(Fill) % — 買い指値", SEQ, False, "{:.0f}")
        fig.tight_layout(rect=(0, 0.005, 1, 0.962))
        out = ch / f"{tag}_penalty_{f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  {out.name}")


if __name__ == "__main__":
    main()
