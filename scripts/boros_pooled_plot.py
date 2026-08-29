"""全 188 市場のプール結果を図にする。

主題は「**標本を 85 倍にして何が残り、何が残らなかったか**」なので、
3 変数を同じ軸で並べ、市場ごとの符号分布を主役にする。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, CHARTS = ROOT / "data", ROOT / "charts"
SRC = Path("E:/Boros-history")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Meiryo", "Noto Sans JP", "Yu Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})
LAB = {"log_dur": "spread duration  log(1+秒)", "spread_pp": "spread (pp)",
       "slope_diff": "book_slope_diff (pp)"}
SHORT = {"log_dur": "spread duration", "spread_pp": "spread", "slope_diff": "book_slope_diff"}
COL = {"log_dur": "#dc2626", "spread_pp": "#f59e0b", "slope_diff": "#2563eb"}


def main() -> int:
    R = json.loads((DATA / "pooled_all.json").read_text(encoding="utf-8"))["results"]
    cfg = {m["marketId"]: m for m in
           json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))}
    xs = ["log_dur", "spread_pp", "slope_diff"]
    H = 3600

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 2.0], hspace=0.40, wspace=0.26)

    # 上段: 当てはめ曲線
    for c, xc in enumerate(xs):
        ax = fig.add_subplot(gs[0, c])
        for h, ls, al in [(600, "-", 0.5), (3600, "--", 1.0)]:
            r = next(z for z in R if z["x"] == xc and z["horizon"] == h)
            xg = np.array(r["grid_x"])
            ax.fill_between(xg, r["ci_lo"], r["ci_hi"], color=COL[xc],
                            alpha=0.10 + 0.06 * al, lw=0)
            ax.plot(xg, r["grid_f"], ls, color=COL[xc], lw=2.0, alpha=al,
                    label=f"{h}s edf={r['edf']:.1f} λ={r['lambda']:.3g}")
            if h == H:
                bx, by = np.array(r["bin_x"]), np.array(r["bin_y"])
                m = np.isfinite(bx) & np.isfinite(by)
                ax.plot(bx[m], by[m], "o", ms=3.6, color=COL[xc], alpha=0.6, mew=0)
        ax.axhline(0, color="#111827", lw=0.9)
        ax.set_xlabel(LAB[xc])
        if c == 0:
            ax.set_ylabel("将来の mid 変化(因果標準化)")
        r3 = next(z for z in R if z["x"] == xc and z["horizon"] == H)
        pos, n = r3["n_pos"], r3["n_mk"]
        z = (pos - n / 2) / np.sqrt(n / 4)
        ok = z > 3
        ax.set_title(f"{'★検出' if ok else '検出できず'}   正の市場 {pos}/{n}  (z={z:.1f})",
                     fontsize=11, loc="left", color="#111827" if ok else "#6b7280")
        ax.legend(fontsize=8, frameon=False)

    # 下段左: 市場ごとの相関のヒストグラム
    ax = fig.add_subplot(gs[1, :2])
    for xc in xs:
        r = next(z for z in R if z["x"] == xc and z["horizon"] == H)
        cs = np.array(list(r["cors_by_market"].values()))
        ax.hist(cs, bins=46, range=(-0.35, 0.35), histtype="step", lw=2.0,
                color=COL[xc],
                label=f"{SHORT[xc]}  中央{r['cor_med']:+.4f}  "
                      f"正 {r['n_pos']}/{r['n_mk']}")
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_xlabel("市場ごとの標本外相関(地平 3600s、その市場を含まない模型で予測)")
    ax.set_ylabel("市場数")
    ax.set_title("★188 市場の符号分布 — slope_diff だけがゼロの右へ丸ごと寄る",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    # 下段右: プラットフォーム別(slope_diff)
    ax = fig.add_subplot(gs[1, 2])
    r = next(z for z in R if z["x"] == "slope_diff" and z["horizon"] == H)
    by_pf: dict[str, list[float]] = {}
    for k, v in r["cors_by_market"].items():
        pf = cfg.get(int(k), {}).get("symbol", "?").split("-")[0]
        by_pf.setdefault(pf, []).append(v)
    items = sorted(by_pf.items(), key=lambda z: -len(z[1]))
    lab = [f"{k}\n{sum(1 for v in vs if v > 0)}/{len(vs)}" for k, vs in items]
    ax.boxplot([vs for _, vs in items], tick_labels=lab, widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", edgecolor="#2563eb"),
               medianprops=dict(color="#1d4ed8", lw=2))
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_ylabel("標本外相関")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title("book_slope_diff — 取引所別\n(下段は 正/全 市場数)",
                 fontsize=10, loc="left")

    fig.suptitle("Boros 全 188 市場 / 1,042 万イベント — spread duration は検出可能か。"
                 "平滑化スプライン min{Σ(y−f(x))²+λ∫f''²}、λ は入れ子 10 分割で機械選択",
                 fontsize=12.5, y=0.985)
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "boros_pooled_all.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"-> {p}")

    # 層別を数字でも出す
    print("\n=== プラットフォーム別(slope_diff, 3600s)===")
    for k, vs in items:
        a = np.array(vs)
        print(f"  {k:<12} 市場 {len(a):>3}  正 {int((a>0).sum()):>3}  "
              f"中央 {np.median(a):+.4f}")
    print("\n=== 参考: spread duration も同じ層別 ===")
    rd = next(z for z in R if z["x"] == "log_dur" and z["horizon"] == H)
    bd: dict[str, list[float]] = {}
    for k, v in rd["cors_by_market"].items():
        bd.setdefault(cfg.get(int(k), {}).get("symbol", "?").split("-")[0], []).append(v)
    for k, _ in items:
        a = np.array(bd.get(k, []))
        if len(a):
            print(f"  {k:<12} 市場 {len(a):>3}  正 {int((a>0).sum()):>3}  "
                  f"中央 {np.median(a):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
