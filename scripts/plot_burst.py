"""絞った標本の burstiness と自己相関を描く。

    uv run python scripts/plot_burst.py --coin xyz:MU
出力: charts/<coin>_burst.png
      data/burst_tvalues_<coin>.csv   日次 SE と Newey-West SE の比較

【読み方】
標本が時間的に固まっているかどうかは、**標準誤差の正しさ**に直結する。
独立とみなした標準誤差は最大 15.8 倍過小である。さらに日ごとの値にも
自己相関があるので、日でまとめるだけでも足りず、日次系列に
Newey-West を当てる必要がある。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
FEATS = ["obi", "ofi_10s", "ai_net_10s"]
ALLF = FEATS + ["ai_rate_10s"]
FC = {"obi": "#b5322f", "ofi_10s": "#1f5fa8", "ai_net_10s": "#7a52c9",
      "ai_rate_10s": "#1f8a5e"}
LAB = {"obi": "OBI", "ofi_10s": "OFI(10s)", "ai_net_10s": "攻撃的数量(10s)",
       "ai_rate_10s": "攻撃的件数(10s)"}
# 日次系列は週次の周期を持つ (ラグ 7 で自己相関 +0.70)。週末は原資産の米国株が
# 動かないためで、5 次では周期を跨げない。2 周期ぶんの 14 日を既定にする。
NW_LAGS = 14
BONF = 3.29


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def daily_series(C, f, h, col):
    s = (C.filter((pl.col("feat") == f) & (pl.col("h") == h)
                  & pl.col("q").is_in([0, 4])).sort("dt", "q"))
    d = s.group_by("dt", maintain_order=True).agg(
        v=(pl.col(col) / pl.col("n")).last() - (pl.col(col) / pl.col("n")).first())
    v = d["v"].to_numpy()
    return v[np.isfinite(v)]


def nwse(v, q=NW_LAGS):
    """日次系列の長期分散から標準誤差を出す (Bartlett 重み)。"""
    n = v.size
    e = v - v.mean()
    g = float((e * e).sum() / n)
    for lg in range(1, q + 1):
        g += 2.0 * (1.0 - lg / (q + 1.0)) * float((e[lg:] * e[:-lg]).sum() / n)
    return (float(v.mean()), float(np.sqrt(max(g, 0.0) / n)),
            float(np.sqrt(e.var(ddof=1) / n)), n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    A = pl.read_csv(D / f"burst_acf_{tag}.csv")
    S = pl.read_csv(D / f"burst_summary_{tag}.csv")
    B = pl.read_csv(D / f"burst_block_{tag}.csv")
    R = pl.read_csv(D / f"burst_run_{tag}.csv")
    C = pl.read_parquet(D / f"pred_cells_{tag}.parquet")

    tv = []
    for f in ALLF:
        for col, pn in (("sum_r", "mid"), ("sum_rmic", "micro")):
            for h in HS:
                v = daily_series(C, f, h, col)
                m, se_n, se_d, n = nwse(v)
                tv.append({"feat": f, "price": pn, "h": h, "mean": m, "n_day": n,
                           "se_daily": se_d, "t_daily": m / se_d,
                           "se_nw": se_n, "t_nw": m / se_n,
                           **{f"t_nw{q}": m / nwse(v, q)[1] for q in (5, 10, 21)},
                           "ratio": se_n / se_d,
                           "crosses_bonferroni":
                           (abs(m / se_d) > BONF) != (abs(m / se_n) > BONF)})
    T = pl.DataFrame(tv)
    T.write_csv(D / f"burst_tvalues_{tag}.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{a.coin}  絞った標本は時間的に固まっているか — 自己相関と "
                 f"burstiness (85〜97 日)", color=INK, fontsize=13, y=0.985)

    ax = axes[0, 0]
    for s, c, ls in (("obi", FC["obi"], "-"), ("ofi_10s", FC["ofi_10s"], "-"),
                     ("ai_net_10s", FC["ai_net_10s"], "-"),
                     ("ret_1s", "#0b0b0b", "--"), ("absret_1s", "#eb6834", "--")):
        d = A.filter(pl.col("series") == s).sort("lag_s")
        nm = {"ret_1s": "1 秒リターン",
              "absret_1s": "1 秒リターンの絶対値"}.get(s, LAB.get(s, s))
        ax.plot(d["lag_s"].to_numpy(), d["acf"].to_numpy(), color=c, lw=1.8,
                ls=ls, label=nm)
    ax.axhline(0, color=BASELINE, lw=1.0)
    style(ax, "ラグ (秒)", "自己相関", logx=True)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("(a) 特徴量は強く相関、リターンはほぼ白色", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper right")

    ax = axes[0, 1]
    for f in FEATS:
        d = R.filter((pl.col("feat") == f) & (pl.col("q") == 4)).sort("bin")
        lo = d["lo"].to_numpy().astype(float)
        c = d["count"].to_numpy().astype(float)
        surv = 1.0 - np.concatenate([[0.0], np.cumsum(c)[:-1]]) / c.sum()
        x = np.maximum(lo, 1)
        ax.plot(x, surv, color=FC[f], lw=2.0, label=LAB[f])
        pm = float(S.filter((pl.col("feat") == f) & (pl.col("q") == 4))
                   ["n_sel"][0]) / (85 * 86401)
        ax.plot(x, pm ** (x - 1), color=FC[f], lw=1.0, ls=":", alpha=0.8)
    ax.text(0.03, 0.06, "点線 = 独立なら (幾何分布)", color=INK2, fontsize=8,
            transform=ax.transAxes)
    style(ax, "Q5 に入り続けた長さ (秒)", "それ以上続く割合", logx=True,
          logy=True)
    ax.set_ylim(1e-6, 1.4)
    ax.set_title("(b) 連は独立の想定より 1.9〜8.1 倍長い", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper right")

    ax = axes[0, 2]
    ws = [60, 300, 900, 3600]
    for f in FEATS:
        r = S.filter((pl.col("feat") == f) & (pl.col("q") == 4)).row(0, named=True)
        ax.plot(ws, [r[f"fano_{w}s"] for w in ws], color=FC[f], lw=2.0,
                marker="o", ms=5, label=LAB[f])
    ax.axhline(1.0, color=INK, lw=1.4, zorder=3)
    ax.text(62, 1.15, "ポアソン = 1", color=INK, fontsize=8)
    style(ax, "窓の幅 (秒)", "Fano 因子 = 件数の 分散/平均", logx=True, logy=True)
    ax.set_title("(c) 窓を広げるほど過分散 — 塊で来ている", color=INK,
                 fontsize=10, loc="left")
    legend(ax, loc="upper left")

    ax = axes[1, 0]
    for f in FEATS:
        for h, ls, al in ((1.0, "-", 1.0), (10.0, "--", 0.6)):
            d = B.filter((pl.col("feat") == f) & (pl.col("h") == h)
                         & (pl.col("block_s") <= 86400)).sort("block_s")
            ax.plot(d["block_s"].to_numpy(), d["se"].to_numpy(), color=FC[f],
                    lw=1.9, ls=ls, alpha=al, marker="o", ms=3.5,
                    label=f"{LAB[f]} h={h:g}s")
    style(ax, "ブロックの幅 (秒)", "Q5−Q1 の標準誤差 (bp)", logx=True, logy=True)
    ax.set_title("(d) 独立とみなすと最大 15.8 倍過小", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="upper left", ncol=2)

    ax = axes[1, 1]
    lags = np.arange(1, 11)
    w = 0.26
    se = 0.1
    for i, f in enumerate(FEATS):
        v = daily_series(C, f, 1.0, "sum_r")
        ac = [float(np.corrcoef(v[:-k], v[k:])[0, 1]) for k in lags]
        ax.bar(lags + (i - 1) * w, ac, width=w, color=FC[f], lw=0, label=LAB[f])
        se = 1.0 / np.sqrt(v.size)
    for sgn in (1, -1):
        ax.axhline(sgn * 2 * se, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(10.4, 2 * se, "±2SE ", color=MUTED, fontsize=8, ha="right",
            va="bottom")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(lags)
    style(ax, "ラグ (日)", "日ごとの Q5−Q1 の自己相関")
    ax.set_title("(e) 日どうしも独立でない — 日でまとめるだけでは足りない",
                 color=INK, fontsize=10, loc="left")
    legend(ax, loc="upper right")

    ax = axes[1, 2]
    yy = np.arange(len(ALLF))
    for j, (h, mk) in enumerate(((1.0, "o"), (10.0, "s"), (60.0, "^"))):
        for k, (col, fill) in enumerate((("t_daily", "none"), ("t_nw", "full"))):
            v = [float(T.filter((pl.col("feat") == f) & (pl.col("price") == "micro")
                                & (pl.col("h") == h))[col][0]) for f in ALLF]
            ax.plot(v, yy + (j - 1) * 0.22, mk, ms=7,
                    color=["#b5322f", "#1f5fa8", "#7a52c9"][j],
                    fillstyle=fill, mew=1.4, ls="none",
                    label=(f"h={h:g}s " + ("補正後" if k else "日次のみ")))
    ax.axvline(0, color=INK, lw=1.2, zorder=3)
    for v in (-BONF, BONF):
        ax.axvline(v, color="#eb6834", lw=1.0, ls="--", zorder=1)
    ax.text(BONF + 0.3, len(ALLF) - 0.6, "Bonferroni 3.29", color="#eb6834",
            fontsize=8, va="top")
    ax.set_yticks(yy, [LAB[f] for f in ALLF], fontsize=8.5)
    style(ax, "microprice で測った Q5−Q1 の t 値", "")
    ax.set_title("(f) 中抜き = 日次のみ / 塗り = Newey-West(14 日) 補正後",
                 color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower right", ncol=2)

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    out = ROOT / "charts" / f"{tag}_burst.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")
    n = int(T["crosses_bonferroni"].sum())
    print(f"Bonferroni 閾値をまたぐセル {n} / {T.height}")


if __name__ == "__main__":
    main()
