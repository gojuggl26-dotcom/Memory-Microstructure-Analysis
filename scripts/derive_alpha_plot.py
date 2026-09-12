r"""Derive — wallet alpha の「技術 vs 運」分解の図。

★見出しの断定は実測値から組み立てる。
★matplotlib のテキストに markdown の強調記号(アスタリスク2つ)を書かない。
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
A = Path("E:/Memory-derive/alpha")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "font.monospace": ["MS Gothic", "Meiryo", "DejaVu Sans Mono"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
TOP3 = ["0x43a9880dA38446a6Cc589BE9070E90A1E5D17ed7",
        "0x3bf64B8c8ce86abDBcF44Fa5a559E27eb9F63Efb",
        "0x3aAa2bB57a0eFeA8F6f89423A923f7107c33471B"]
LBL = {TOP3[0]: "①0x43a9880d", TOP3[1]: "②0x3bf64B8c", TOP3[2]: "③0x3aAa2bB5"}
C3 = {TOP3[0]: "#2563eb", TOP3[1]: "#dc2626", TOP3[2]: "#16a34a"}


def main() -> int:
    CHARTS.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18.6, 11.6))
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.28)

    # ---------- ① 縮約の効き方(素 vs 縮約後)----------
    ax = fig.add_subplot(gs[0, 0])
    g = pl.read_parquet(A / "alpha_pnl_total_raw.parquet")
    x = g["a"].to_numpy() / 1e3
    y = g["alpha_eb"].to_numpy() / 1e3
    ax.scatter(x, y, s=6, alpha=0.25, color="#9ca3af", linewidths=0)
    for w in TOP3:
        r = g.filter(pl.col("wallet") == w)
        if r.height:
            ax.scatter(r["a"][0] / 1e3, r["alpha_eb"][0] / 1e3, s=90,
                       color=C3[w], zorder=5, label=LBL[w])
    lim = np.percentile(np.abs(x), 99) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="#111827", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim / 2, lim / 2)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xlabel("素の日次アルファ [千ドル/日]")
    ax.set_ylabel("階層的縮約の後 [千ドル/日]")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("★① 縮約は見かけの大勝ちを 0 へ引き戻す\n"
                 "破線が y=x。縦に潰れている分が『運』と判定された量",
                 fontsize=11, loc="left")

    # ---------- ② 縮約係数 B(どれだけ信用されたか)----------
    ax = fig.add_subplot(gs[0, 1])
    names = [("alpha_pnl_total_raw.parquet", "純損益 [$]"),
             ("alpha_pnl_option_side_raw.parquet", "オプション [$]"),
             ("alpha_pnl_perp_raw.parquet", "perp [$]"),
             ("alpha_bp_raw.parquet", "損益/名目 [bp]")]
    xs = np.arange(len(names))
    wdt = 0.25
    for j, w in enumerate(TOP3):
        vals = []
        for f, _ in names:
            gg = pl.read_parquet(A / f)
            r = gg.filter(pl.col("wallet") == w)
            vals.append(float(r["shrink_B"][0]) if r.height else np.nan)
        ax.bar(xs + (j - 1) * wdt, vals, wdt, color=C3[w], label=LBL[w])
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in names], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("縮約係数 B(1 に近いほど素の値を信用)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("② B が小さいほど『運』と判定された\n"
                 "②0x3bf64B8c は純損益で B=0.006 = ほぼ全部が運",
                 fontsize=11, loc="left")

    # ---------- ③ P(alpha > 0) ----------
    ax = fig.add_subplot(gs[0, 2])
    for j, w in enumerate(TOP3):
        vals = []
        for f, _ in names:
            gg = pl.read_parquet(A / f)
            r = gg.filter(pl.col("wallet") == w)
            vals.append(float(r["p_alpha_gt0"][0]) if r.height else np.nan)
        ax.bar(xs + (j - 1) * wdt, vals, wdt, color=C3[w], label=LBL[w])
    ax.axhline(0.95, color="#dc2626", ls="--", lw=1.2)
    ax.axhline(0.5, color="#111827", lw=1.0)
    ax.text(len(names) - 0.5, 0.955, "0.95", fontsize=8, color="#dc2626",
            ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in names], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(alpha > 0)(縮約後の事後確率)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("★③ 0.95 を超えるのは ①の perp だけ\n"
                 "②③ は 0.5 付近 = 0 と区別できない", fontsize=11, loc="left")

    # ---------- ④ ウォークフォワード IC vs 帰無 ----------
    ax = fig.add_subplot(gs[1, 0])
    W = pl.read_parquet(A / "walkforward.parquet")
    tg = ["純損益 [$]", "オプションのみ [$]", "perp のみ [$]", "純損益/名目 [bp]"]
    pos = np.arange(len(tg))
    for j, sp in enumerate(("素", "日固定効果あり")):
        m, nn = [], []
        for t in tg:
            d = W.filter((pl.col("target") == t) & (pl.col("spec") == sp))
            m.append(float(np.nanmedian(d["ic"].to_numpy())) if d.height else np.nan)
            nn.append(float(np.nanmedian(d["ic_null_p95"].to_numpy()))
                      if d.height else np.nan)
        ax.bar(pos + (j - 0.5) * 0.35, m, 0.35,
               color=["#2563eb", "#7c3aed"][j], label=f"実測 ({sp})")
        if j == 0:
            ax.plot(pos, nn, "x", color="#dc2626", ms=10, mew=2,
                    label="帰無の 95 分位")
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([t.replace(" ", "\n") for t in tg], fontsize=8)
    ax.set_ylabel("Spearman rank IC(22 窓の中央値)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("★④ 順位の再現性は帰無を超えない\n"
                 "形成 6 か月 → 評価 6 か月・1 か月ずつ前進", fontsize=11, loc="left")

    # ---------- ⑤ 上位四分位の残存 ----------
    ax = fig.add_subplot(gs[1, 1])
    for j, sp in enumerate(("素", "日固定効果あり")):
        m, nn = [], []
        for t in tg:
            d = W.filter((pl.col("target") == t) & (pl.col("spec") == sp))
            m.append(float(np.nanmean(d["top_q_keep"].to_numpy()))
                     if d.height else np.nan)
            nn.append(float(np.nanmean(d["keep_null_mean"].to_numpy()))
                      if d.height else np.nan)
        ax.bar(pos + (j - 0.5) * 0.35, m, 0.35,
               color=["#2563eb", "#7c3aed"][j], label=f"実測 ({sp})")
        if j == 0:
            ax.plot(pos, nn, "x", color="#dc2626", ms=10, mew=2, label="帰無")
    ax.set_xticks(pos)
    ax.set_xticklabels([t.replace(" ", "\n") for t in tg], fontsize=8)
    ax.set_ylabel("上位四分位に残る確率")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("⑤ 上位層の残存だけは帰無を上回る\n"
                 "全体の順位は再現しないが、上位には持続がある",
                 fontsize=11, loc="left")

    # ---------- ⑥ 上位 3 者の評価期での位置 ----------
    ax = fig.add_subplot(gs[1, 2])
    cols = [c for c in W.columns if c.startswith("pct_")]
    tgts = ["純損益 [$]", "オプションのみ [$]", "perp のみ [$]"]
    for j, (w, c) in enumerate(zip(TOP3, cols)):
        med, cnt = [], []
        for t in tgts:
            d = W.filter((pl.col("target") == t) & (pl.col("spec") == "素"))
            v = d[c].to_numpy()
            v = v[np.isfinite(v)]
            med.append(float(np.median(v)) if len(v) else np.nan)
            cnt.append(len(v))
        b = ax.bar(pos[:3] + (j - 1) * wdt, med, wdt, color=C3[w],
                   label=f"{LBL[w]}({cnt[0]} 窓)")
    ax.axhline(0.75, color="#dc2626", ls="--", lw=1.0)
    ax.axhline(0.5, color="#111827", lw=1.0)
    ax.set_xticks(pos[:3])
    ax.set_xticklabels([t.replace(" ", "\n") for t in tgts], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("評価期での順位(0=最下位, 1=最上位)")
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    ax.set_title("★⑥ 上位 3 者を別期間で追う\n"
                 "②はオプション最下位・perp 最上位で一貫している",
                 fontsize=11, loc="left")

    fig.suptitle("Derive — 上位ウォレットの実績を『技術』と『運』に分解する"
                 "(階層的縮約 + ウォークフォワード)", fontsize=13.5, y=0.985)
    out = CHARTS / "derive_alpha.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
