r"""Derive — 実現損益による MM 採算の図。

★見出しの断定は実測値から組み立てる(結果を見る前に書かない)。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
SRC = Path("E:/Memory-derive")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
CPOS, CNEG, CG = "#2563eb", "#dc2626", "#9ca3af"
COPT, CSET, CPERP = "#7c3aed", "#0891b2", "#16a34a"


def main() -> int:
    CHARTS.mkdir(exist_ok=True)
    L = pl.read_parquet(SRC / "mm" / "wallet_pnl.parquet")
    MK = (L.filter((pl.col("maker_share") >= 0.7) & (pl.col("n_rows") >= 200))
          .sort("pnl_realized", descending=True))
    d = pl.read_parquet(SRC / "mm" / "daily_maker.parquet").sort("day")
    tot = float(MK["pnl_realized"].sum())

    fig = plt.figure(figsize=(18.6, 11.4))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.27)

    # ---------- ① オプション単独 vs perp 込み ----------
    ax = fig.add_subplot(gs[0, 0])
    opt = float(MK["pnl_opt_only"].sum())
    perp = float(MK["pnl_perp"].sum())
    bars = [("オプション\n単独", opt), ("perp\nヘッジ", perp),
            ("合計", opt + perp)]
    ax.bar([b[0] for b in bars], [b[1] / 1e6 for b in bars],
           color=[CNEG if b[1] < 0 else CPOS for b in bars], width=0.6)
    ax.axhline(0, color="#111827", lw=1.4)
    for i, (_, v) in enumerate(bars):
        ax.text(i, v / 1e6, f"{v/1e6:+.2f}M", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10.5)
    ax.set_ylabel("実現損益 [百万ドル]")
    ax.set_title("★① perp ヘッジを入れると符号が反転する\n"
                 "メイカー主体 41 ウォレット・2024-01〜2026-09",
                 fontsize=11, loc="left")

    # ---------- ② 損益の内訳 ----------
    ax = fig.add_subplot(gs[0, 1])
    parts = [("オプション\n反対売買", float(MK["pnl_trade"].sum()), COPT),
             ("満期決済", float(MK["pnl_settle"].sum()), CSET),
             ("perp", float(MK["pnl_perp"].sum()), CPERP),
             ("残存建玉\n(未実現)", float(MK["open_mtm"].sum()), CG)]
    ax.bar([p[0] for p in parts], [p[1] / 1e6 for p in parts],
           color=[p[2] for p in parts], width=0.6)
    ax.axhline(0, color="#111827", lw=1.4)
    for i, (_, v, _c) in enumerate(parts):
        ax.text(i, v / 1e6, f"{v/1e6:+.1f}M", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9.5)
    ax.set_ylabel("[百万ドル]")
    ax.set_title("② 内訳 — オプション売買は赤字で、perp が支えている\n"
                 "灰は未実現(主判定に含めない)", fontsize=11, loc="left")

    # ---------- ③ ウォレット別(集中度)----------
    ax = fig.add_subplot(gs[0, 2])
    v = MK["pnl_realized"].to_numpy()
    ax.bar(range(len(v)), v / 1e6, color=[CPOS if x > 0 else CNEG for x in v],
           width=0.8)
    ax.axhline(0, color="#111827", lw=1.2)
    ax.set_xlabel("メイカー主体ウォレット(損益の降順)")
    ax.set_ylabel("[百万ドル]")
    top3 = v[:3].sum()
    ax.set_title(f"★③ 上位 3 者で ${top3/1e6:.2f}M = 全体の {top3/tot*100:.0f}%\n"
                 f"黒字 {(v>0).sum()}/{len(v)} ・ 中央値 ${np.median(v):,.0f}",
                 fontsize=11, loc="left")

    # ---------- ④ 月次 ----------
    ax = fig.add_subplot(gs[1, 0])
    m = (d.with_columns(pl.col("day").str.slice(0, 7).alias("mon"))
         .group_by("mon").agg(pl.col("total").sum()).sort("mon"))
    mv = m["total"].to_numpy()
    ax.bar(range(len(mv)), mv / 1e6,
           color=[CPOS if x > 0 else CNEG for x in mv])
    ax.axhline(0, color="#111827", lw=1.2)
    st = max(1, len(mv) // 8)
    ax.set_xticks(range(0, len(mv), st))
    ax.set_xticklabels([m["mon"][i][2:] for i in range(0, len(mv), st)],
                       fontsize=7.5, rotation=45)
    ax.set_ylabel("[百万ドル]")
    ax.set_title(f"④ 月次 — 黒字 {(mv>0).sum()}/{len(mv)} か月\n"
                 "振れ幅が大きく、少数の月が全体を決める", fontsize=11, loc="left")

    # ---------- ⑤ 上位の日を除く感度 ----------
    ax = fig.add_subplot(gs[1, 1])
    dv = d["total"].to_numpy()
    o = np.argsort(-np.abs(dv))
    ks = [0, 1, 3, 5, 10, 20]
    ys = [(dv.sum() - dv[o[:k]].sum()) / 1e6 for k in ks]
    ax.bar(range(len(ks)), ys, color=[CPOS if y > 0 else CNEG for y in ys],
           width=0.6)
    ax.axhline(0, color="#111827", lw=1.4)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k} 日\n除外" for k in ks], fontsize=8.5)
    for i, y in enumerate(ys):
        ax.text(i, y, f"{y:+.1f}M", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=9)
    ax.set_ylabel("[百万ドル]")
    ax.set_title("★⑤ 絶対値の大きい日を除くと符号が反転する\n"
                 "結論は少数の観測に支配されている", fontsize=11, loc="left")

    # ---------- ⑥ 日ブートストラップ ----------
    ax = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(20260912)
    idx = rng.integers(0, len(dv), size=(5000, len(dv)))
    b = dv[idx].sum(axis=1) / 1e6
    ax.hist(b, bins=60, color="#7c3aed", alpha=0.8)
    ax.axvline(0, color="#111827", lw=1.6)
    lo, hi = np.percentile(b, [2.5, 97.5])
    ax.axvline(lo, color=CNEG, ls="--", lw=1.2)
    ax.axvline(hi, color=CNEG, ls="--", lw=1.2)
    ax.set_xlabel("日を単位に復元抽出した合計 [百万ドル]")
    ax.set_ylabel("回数")
    ax.set_title(f"⑥ 日単位ブートストラップ\n95%CI [{lo:+.1f}M, {hi:+.1f}M] / "
                 f"P(合計>0) = {(b>0).mean():.3f}", fontsize=11, loc="left")

    fig.suptitle("Derive オプション — 取引所が確定させた実現損益による MM 採算の検証"
                 "(反対売買 + 満期決済 + perp ヘッジ)", fontsize=13.5, y=0.985)
    out = CHARTS / "derive_mm_pnl.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
