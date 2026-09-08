r"""ウォークフォワードの図 — 分割の設計・構成の分布・日ブートストラップ・未見期間。

★見出しの断定は実測値から組み立てる(結果を見る前に書かない)。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
SRC = Path("E:/Memory-lighter/wf")
EXPLORE_END = "2026-09-05"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
CE, CO, CG = "#2563eb", "#dc2626", "#9ca3af"


def daily(df):
    if df is None or df.height == 0:
        return [], np.array([]), np.array([])
    g = (df.group_by("day").agg([pl.col("pnl_sum").sum(),
                                 pl.col("n_trade").sum()]).sort("day"))
    return g["day"].to_list(), g["pnl_sum"].to_numpy(), g["n_trade"].to_numpy()


def main() -> int:
    L = pl.concat([pl.read_parquet(f) for f in
                   sorted(glob.glob(str(SRC / "ledger_*.parquet")))],
                  how="diagonal_relaxed").filter(pl.col("day") <= EXPLORE_END)
    G = pl.read_parquet(SRC / "wf_summary.parquet")
    S = pl.read_parquet(SRC / "wf_selected.parquet")
    S2 = pl.read_parquet(SRC / "wf_selected_global.parquet")
    cfg = json.loads((ROOT / "config" / "lighter_frozen.json").read_text("utf-8"))
    ofs = sorted(glob.glob(str(SRC / "oos_*.parquet")))
    O = (pl.concat([pl.read_parquet(f) for f in ofs], how="diagonal_relaxed")
         if ofs else None)

    fig = plt.figure(figsize=(19.0, 11.8))
    gs = fig.add_gridspec(2, 4, hspace=0.46, wspace=0.30)

    # ---------- ① 分割の設計 ----------
    ax = fig.add_subplot(gs[0, 0])
    days = sorted(L["day"].unique().to_list())
    folds = sorted(L["fold"].unique().to_list())[:10]
    base = np.datetime64(days[0])
    for i, f in enumerate(folds):
        g = L.filter(pl.col("fold") == f)
        if g.height == 0:
            continue
        x0 = (np.datetime64(g["train_from"][0]) - base).astype(int)
        x1 = (np.datetime64(g["train_to"][0]) - base).astype(int)
        xt = (np.datetime64(g["test_day"][0]) - base).astype(int)
        ax.barh(i, x1 - x0 + 1, left=x0, height=0.7, color=CE, alpha=0.85)
        ax.barh(i, xt - x1 - 1, left=x1 + 1, height=0.7, color="#f59e0b", alpha=.9)
        ax.barh(i, 1, left=xt, height=0.7, color=CO, alpha=0.9)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"fold {f}" for f in folds], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel(f"{days[0]} からの日数", fontsize=8.5)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CE, "#f59e0b", CO)]
    ax.legend(hs, ["学習 7 日", "検証 2 日", "テスト 1 日"], fontsize=7.5,
              frameon=False, loc="lower right")
    ax.set_title("① 分割(1 日ずつ前進)\n各ブロック末尾 300 秒はパージ",
                 fontsize=10, loc="left")

    # ---------- ② 63 構成の分布 ----------
    ax = fig.add_subplot(gs[0, 1])
    v = G["pnl_sum"].to_numpy()
    ax.hist(v, bins=25, color="#7c3aed", alpha=0.8)
    ax.axvline(0, color="#111827", lw=1.4)
    npos = int((v > 0).sum())
    fz = G.filter((pl.col("signal") == cfg["model"])
                  & (pl.col("q") == cfg["threshold_rule"]["q"])
                  & (pl.col("h") == cfg["exit"]["seconds"]))
    if fz.height:
        ax.axvline(fz["pnl_sum"][0], color=CO, lw=2.0, ls="--",
                   label=f"凍結した構成\n{fz['pnl_sum'][0]:+,.0f}")
        ax.legend(fontsize=7.5, frameon=False)
    ax.set_xlabel("テスト日の純損益総額[bp]", fontsize=8.5)
    ax.set_ylabel("構成の数", fontsize=8.5)
    ax.set_title(f"★② 試した {G.height} 構成 — 純が正は {npos}/{G.height}\n"
                 "最良だけを見ないための台帳", fontsize=10, loc="left")

    # ---------- ③ 探索期間の累積 ----------
    ax = fig.add_subplot(gs[0, 2])
    dd = []
    for df, lab, col in ((S, "(A) 銘柄×fold で選択", CE),
                         (S2, "(A2) fold ごと共通で選択", "#059669")):
        dd, p, _ = daily(df)
        if len(dd):
            ax.plot(range(len(dd)), np.cumsum(p), "o-", color=col, lw=1.8,
                    ms=3.5, label=f"{lab}\n計 {p.sum():+,.0f}")
    ax.axhline(0, color="#111827", lw=1.2)
    if len(dd):
        st = max(1, len(dd) // 5)
        ax.set_xticks(range(0, len(dd), st))
        ax.set_xticklabels([dd[i][5:] for i in range(0, len(dd), st)],
                           fontsize=7.5)
    ax.set_ylabel("累積純損益[bp]", fontsize=8.5)
    ax.legend(fontsize=7.2, frameon=False)
    ax.set_title("③ 探索期間の累積\n★既に全部見た期間なので探索的",
                 fontsize=10, loc="left")

    # ---------- ④ 日単位ブートストラップ ----------
    ax = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(11)
    for df, lab, col in ((S, "(A)", CE), (S2, "(A2)", "#059669")):
        _, p, _ = daily(df)
        if len(p) < 3:
            continue
        bt = p[rng.integers(0, len(p), size=(5000, len(p)))].sum(axis=1)
        ax.hist(bt, bins=50, alpha=0.55, color=col,
                label=f"{lab} 95%CI\n[{np.percentile(bt,2.5):+,.0f}, "
                      f"{np.percentile(bt,97.5):+,.0f}]")
    ax.axvline(0, color="#111827", lw=1.4)
    ax.set_xlabel("日を単位に復元抽出した総額[bp]", fontsize=8.5)
    ax.legend(fontsize=7.2, frameon=False)
    ax.set_title("④ 日単位ブートストラップ\n同じ日の全銘柄をまとめて再抽出",
                 fontsize=10, loc="left")

    # ---------- ⑤ 未見期間 日別(純 = 粗 − 費)----------
    ax = fig.add_subplot(gs[1, 0])
    if O is not None and O.height:
        g = (O.group_by("day").agg([pl.col("pnl_sum").sum(),
                                    pl.col("gross_sum").sum(),
                                    pl.col("cost_sum").sum(),
                                    pl.col("n_trade").sum()]).sort("day"))
        dl = g["day"].to_list()
        x = np.arange(len(dl))
        ax.bar(x - 0.25, g["gross_sum"].to_numpy(), 0.24, color="#059669",
               label="粗利")
        ax.bar(x, -g["cost_sum"].to_numpy(), 0.24, color=CG, label="費用")
        ax.bar(x + 0.25, g["pnl_sum"].to_numpy(), 0.24, color=CO, label="純損益")
        ax.axhline(0, color="#111827", lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels([d[5:] for d in dl], fontsize=8.5)
        ax.set_ylabel("bp", fontsize=8.5)
        ax.legend(fontsize=7.5, frameon=False)
        tot = float(O["pnl_sum"].sum()); n = int(O["n_trade"].sum())
        ax.set_title(f"★⑤ 未見期間(1 回だけ評価)\n純 {tot:+,.0f} bp / "
                     f"{n:,} 取引 = {tot/n:+.3f} bp/取引", fontsize=10, loc="left")

    # ---------- ⑥ 未見期間 銘柄別 ----------
    ax = fig.add_subplot(gs[1, 1])
    if O is not None and O.height:
        gs2 = (O.group_by("symbol").agg([pl.col("pnl_sum").sum(),
                                         pl.col("gross_sum").sum()])
               .sort("pnl_sum"))
        y = np.arange(gs2.height)
        ax.barh(y + 0.19, gs2["gross_sum"].to_numpy(), 0.36, color="#059669",
                label="粗利")
        ax.barh(y - 0.19, gs2["pnl_sum"].to_numpy(), 0.36,
                color=[CO if x2 < 0 else CE for x2 in gs2["pnl_sum"].to_numpy()],
                label="純損益")
        ax.set_yticks(y)
        ax.set_yticklabels(gs2["symbol"].to_list(), fontsize=7.4)
        ax.axvline(0, color="#111827", lw=1.2)
        ng = int((gs2["gross_sum"].to_numpy() > 0).sum())
        npo = int((gs2["pnl_sum"].to_numpy() > 0).sum())
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.set_xlabel("未見期間の総額[bp]", fontsize=8.5)
        ax.set_title(f"★⑥ 未見期間の銘柄別\n粗利が正 {ng}/{gs2.height} ・ "
                     f"純が正 {npo}/{gs2.height}", fontsize=10, loc="left")

    # ---------- ⑦ 損益分岐スプレッド vs 実際 ----------
    ax = fig.add_subplot(gs[1, 2])
    if O is not None and O.height:
        gb = O.group_by("symbol").agg([pl.col("gross_sum").sum(),
                                       pl.col("cost_sum").sum(),
                                       pl.col("n_trade").sum()])
        be = (gb["gross_sum"] / gb["n_trade"]).to_numpy()
        ac = (gb["cost_sum"] / gb["n_trade"]).to_numpy()
        syms = gb["symbol"].to_list()
        ax.scatter(be, ac, s=46, c=[CE if b > a else CO
                                    for b, a in zip(be, ac)], zorder=3)
        for s_, b, a in zip(syms, be, ac):
            ax.annotate(s_, (b, a), fontsize=6.6, xytext=(4, 3),
                        textcoords="offset points")
        lim = max(np.nanmax(be), np.nanmax(ac)) * 1.1
        ax.plot([-0.3, lim], [-0.3, lim], "--", color="#111827", lw=1.2)
        ax.set_xlabel("損益分岐スプレッド = 粗利/取引 [bp]", fontsize=8.5)
        ax.set_ylabel("実際に払った費用/取引 [bp]", fontsize=8.5)
        nb = int((be > ac).sum())
        ax.set_title(f"★⑦ 対角線の下 = 黒字({nb}/{len(be)} 銘柄)\n"
                     "青 = 費用が損益分岐を下回った銘柄", fontsize=10, loc="left")

    # ---------- ⑧ 探索 vs 未見 の対比 ----------
    ax = fig.add_subplot(gs[1, 3])
    labs, gross, cost, net = [], [], [], []
    te = L.filter(pl.col("split") == "test")
    g3 = te.group_by(["signal", "q", "h"]).agg([pl.col("gross_sum").sum(),
                                                pl.col("n_trade").sum()])
    labs.append("探索\n全 63 構成\nの中央値")
    gross.append(float(np.median((g3["gross_sum"] / g3["n_trade"]).to_numpy())))
    cost.append(float(S2["cost_sum"].sum() / S2["n_trade"].sum()))
    net.append(gross[-1] - cost[-1])
    for lab, df in (("探索\n(A2 選択後)", S2), ("★未見\n(凍結)", O)):
        if df is None or df.height == 0:
            continue
        n = int(df["n_trade"].sum())
        labs.append(lab)
        gross.append(float(df["gross_sum"].sum() / n))
        cost.append(float(df["cost_sum"].sum() / n))
        net.append(float(df["pnl_sum"].sum() / n))
    x = np.arange(len(labs))
    ax.bar(x - 0.25, gross, 0.24, color="#059669", label="粗利")
    ax.bar(x, cost, 0.24, color=CG, label="費用")
    ax.bar(x + 0.25, net, 0.24, color=CO, label="純損益")
    ax.axhline(0, color="#111827", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.6)
    ax.set_ylabel("1 取引あたり [bp]", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("★⑧ 選択は粗利を上げるが費用を超えない\n"
                 "未見期間でも同じ構造", fontsize=10, loc="left")

    fig.suptitle("Lighter — ウォークフォワード検証(学習 7 日 → 検証 2 日 → テスト 1 日、"
                 "境界パージ 300 秒、費用は実測スプレッド往復、テイカー前提)",
                 fontsize=13.5, y=0.985)
    out = CHARTS / "lighter_wf.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
