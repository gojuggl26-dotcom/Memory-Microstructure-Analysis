"""約定の 3 分解と、toxic な約定の事前兆候を描く。

    uv run python scripts/plot_markout.py --coin xyz:MU
出力: charts/<coin>_markout.png / data/markout_summary_<coin>.csv

【読み方】
A = r(T→τ) は**クオートの寿命でほぼ決まる**。速く出し直せば消える。
B = r(τ→τ+h) は寿命を変えてもほとんど動かない。置いた以上は避けられない。
SelectionPenalty は「同じ窓での無条件との差」で、A・B とは別の量である。
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
from build_markout import H_FILL, H_QUOTE  # noqa: E402
from plot_burst import nwse  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GC = ["#7d1f1d", "#b5322f", "#9a9890", "#7fa8dd", "#1f5fa8"]
VN = {"aggr": "攻撃的流量 (直前 1 秒)", "ofi": "OFI (直前 1 秒)",
      "obi": "OBI (自分側が厚いと +)", "depth": "自分側の厚み",
      "qdepl": "待ち行列の消化", "drift": "T からの値動き"}
VC = {"aggr": "#b5322f", "ofi": "#1f5fa8", "obi": "#7a52c9",
      "depth": "#1f8a5e", "qdepl": "#c48a12", "drift": "#eb6834"}


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--side", type=int, default=1)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    D = ROOT / "data"
    C = pl.read_csv(D / f"markout_decomp_{tag}.csv").filter(
        pl.col("side") == a.side)
    T = pl.read_csv(D / f"markout_toxic_{tag}.csv").filter(
        pl.col("side") == a.side)
    U = pl.read_csv(D / f"markout_auc_{tag}.csv").filter(
        pl.col("side") == a.side)
    sd = a.side

    rows = []
    # tau は H_FILL 秒までしか探していないので、それより長い寿命は
    # 同じ値の重複になる。誤解を招くので落とす。
    for hq in [x for x in H_QUOTE if x <= H_FILL]:
        n = float(C[f"nq_{hq:g}"].sum())
        nb = float(C[f"nB_{hq:g}"].sum())
        if n == 0 or nb == 0:
            continue
        A = sd * float(C[f"A_{hq:g}"].sum()) / n
        sp = float(C[f"spr_{hq:g}"].sum()) / n
        B = sd * float(C[f"B_{hq:g}"].sum()) / nb
        vA = sd * (C[f"A_{hq:g}"].to_numpy() / np.maximum(C[f"nq_{hq:g}"].to_numpy(), 1))
        vB = sd * (C[f"B_{hq:g}"].to_numpy() / np.maximum(C[f"nB_{hq:g}"].to_numpy(), 1))
        _, seA, _, _ = nwse(vA)
        _, seB, _, _ = nwse(vB)
        rows.append({"h_quote": hq, "n_fill": int(n), "half_spread_T": sp,
                     "A": A, "A_se": seA, "B": B, "B_se": seB,
                     "edge": sp + A, "total": sp + A + B - MAKER_FEE_BP})
    S = pl.DataFrame(rows)
    S.write_csv(D / f"markout_summary_{tag}.csv")
    with pl.Config(tbl_width_chars=150):
        print(S)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.6))
    fig.suptitle(f"{a.coin}  約定の 3 分解と toxic な約定の事前兆候 — "
                 f"{'買い' if sd == 1 else '売り'}指値・97 日",
                 color=INK, fontsize=13, y=0.985)
    x = np.arange(S.height)
    lab = [f"{h:g}s" if h >= 1 else f"{h*1000:.0f}ms" for h in S["h_quote"]]

    ax = axes[0, 0]
    sp = S["half_spread_T"].to_numpy()
    A = S["A"].to_numpy()
    B = S["B"].to_numpy()
    ax.bar(x, sp, width=0.6, color="#1f5fa8", lw=0, label="半スプレッド (発注時)")
    ax.bar(x, np.full_like(sp, -MAKER_FEE_BP), width=0.6, bottom=sp,
           color="#9a9890", lw=0, label="手数料")
    ax.bar(x, A, width=0.6, bottom=sp - MAKER_FEE_BP, color="#eb6834", lw=0,
           label="A: 置いてから約定するまで")
    ax.bar(x, B, width=0.6, bottom=sp - MAKER_FEE_BP + A, color="#b5322f", lw=0,
           label="B: 約定してから 10 秒")
    ax.plot(x, S["total"].to_numpy(), color=INK, lw=1.8, marker="o", ms=4,
            label="合計")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, lab, fontsize=8.5)
    style(ax, "クオートの寿命", "bp / 1 約定")
    ax.set_title("(a) 寿命ごとの分解", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7, loc="lower left")

    ax = axes[0, 1]
    for v, se, c, nm in ((A, S["A_se"].to_numpy(), "#eb6834", "A: T→τ"),
                         (B, S["B_se"].to_numpy(), "#b5322f", "B: τ→τ+10s")):
        ax.fill_between(x, v - se, v + se, color=c, alpha=0.18, lw=0)
        ax.plot(x, v, color=c, lw=2.2, marker="o", ms=4.5, label=nm)
    ax.plot(x, S["edge"].to_numpy(), color="#1f5fa8", lw=1.6, ls="--",
            marker="s", ms=4, label="実際の優位 = 半スプ + A")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, lab, fontsize=8.5)
    style(ax, "クオートの寿命", "bp / 1 約定")
    ax.set_title("(b) ★ A は寿命で消える、B は消えない", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="lower left")

    G = (T.group_by("delta", "grp")
         .agg(n=pl.col("n").sum(),
              **{c: (pl.col(c) * pl.col("n")).sum() for c in VN})
         .with_columns(**{c: pl.col(c) / pl.col("n") for c in VN}))
    ds = sorted(T["delta"].unique().to_list(), reverse=True)
    xs = np.arange(len(ds))
    for j, (v, ax) in enumerate(zip(("aggr", "ofi"), (axes[0, 2], axes[1, 0]))):
        for g in range(5):
            y = [float(G.filter((pl.col("delta") == d) & (pl.col("grp") == g))[v][0])
                 for d in ds]
            ax.plot(xs, y, color=GC[g], lw=2.0, marker="o", ms=4.5,
                    label=f"G{g+1}" + (" 最も toxic" if g == 0 else
                                       (" 最も無害" if g == 4 else "")))
        ax.set_xticks(xs, [("約定時" if d == 0 else f"{d:g} 秒前") for d in ds],
                      fontsize=8.5)
        ax.axhline(0, color=BASELINE, lw=1.0)
        style(ax, "", VN[v])
        ax.set_title(f"({'c' if j == 0 else 'd'}) {VN[v]} — 約定後 markout の五分位",
                     color=INK, fontsize=10, loc="left")
        legend(ax, fs=7.5, loc="best")

    ax = axes[1, 1]
    for v in ("obi", "depth", "qdepl", "drift"):
        y = []
        for d in ds:
            g1 = float(G.filter((pl.col("delta") == d) & (pl.col("grp") == 0))[v][0])
            g5 = float(G.filter((pl.col("delta") == d) & (pl.col("grp") == 4))[v][0])
            s_ = np.std([float(G.filter((pl.col("delta") == d)
                                        & (pl.col("grp") == g))[v][0])
                         for g in range(5)])
            y.append((g1 - g5) / s_ if s_ > 0 else np.nan)
        ax.plot(xs, y, color=VC[v], lw=2.0, marker="o", ms=4.5, label=VN[v])
    for v, ax2 in (("aggr", ax), ("ofi", ax)):
        y = []
        for d in ds:
            g1 = float(G.filter((pl.col("delta") == d) & (pl.col("grp") == 0))[v][0])
            g5 = float(G.filter((pl.col("delta") == d) & (pl.col("grp") == 4))[v][0])
            s_ = np.std([float(G.filter((pl.col("delta") == d)
                                        & (pl.col("grp") == g))[v][0])
                         for g in range(5)])
            y.append((g1 - g5) / s_ if s_ > 0 else np.nan)
        ax2.plot(xs, y, color=VC[v], lw=1.4, ls="--", marker="s", ms=3.5,
                 label=VN[v])
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(xs, [("約定時" if d == 0 else f"{d:g} 秒前") for d in ds],
                  fontsize=8.5)
    style(ax, "", "(G1 − G5) ÷ 5 群の標準偏差")
    ax.set_title("(e) toxic と無害の開き (群間の散らばりで割った)", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=6.5, loc="best", ncol=2)

    ax = axes[1, 2]
    vv = ["ofi", "depth", "qdepl", "obi", "drift", "aggr"]
    w = 0.26
    for j, dl in enumerate((0.5, 1.0, 2.0)):
        m, se = [], []
        for v in vv:
            r = U.filter((pl.col("var") == v) & (pl.col("delta") == dl)).sort("dt")
            mm, ss, _, _ = nwse(r["auc"].to_numpy())
            m.append(mm)
            se.append(ss)
        ax.barh(np.arange(len(vv)) + (j - 1) * w, np.array(m) - 0.5, height=w,
                xerr=se, color=["#b5322f", "#1f5fa8", "#7a52c9"][j], lw=0,
                error_kw=dict(lw=0.8, ecolor=MUTED), label=f"{dl:g} 秒前")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(vv)), [VN[v] for v in vv], fontsize=8)
    style(ax, "AUC − 0.5 (0 = 当てられない)", "")
    ax.set_title("(f) 事前に toxic を当てられるか — ほぼ当てられない", color=INK,
                 fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="lower right")

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    out = ROOT / "charts" / f"{tag}_markout.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


if __name__ == "__main__":
    main()
