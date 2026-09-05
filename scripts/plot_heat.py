"""信号 × 待ち行列時間の 2 次元ヒートマップ。

    uv run python scripts/plot_heat.py --coin xyz:MU
出力: charts/<coin>_heat_<feat>.png  (特徴量ごとに 1 枚)
      data/heat_summary_<coin>.csv   軸ごとの勾配の大きさ

【読み方】
縦が信号の十分位 (S1 = 最も売り側 / S10 = 最も買い側)、
横が待ち行列がはける推定時間の十分位 (D1 = すぐはける / D10 = はけない)。
リターンは**符号をつけない**値なので、信号が効いていれば縦方向に勾配が出る。
横方向にも勾配が出たら、それは待ち行列の長さ (= 板の厚み) が持つ情報である。
mid と microprice を必ず並べて見ること。板の厚みは microprice に直結するので、
mid だけ見ると「mid が既知の microprice に寄る」分を信号と誤認する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_maker import MAKER_FEE_BP  # noqa: E402
from plot_vol import INK, INK2, SURFACE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = 10
LAB = {"obi": "OBI 板の不均衡", "ofi_10s": "OFI 注文流の不均衡 (10 秒)",
       "ai_net_10s": "攻撃的注文の符号つき数量 (10 秒)"}
# 発散: 赤 - 無彩色 - 青 (虹は使わない。中央は必ず無彩色)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#7d1f1d", "#b5322f", "#e08b7f", "#9a9890", "#7fa8dd", "#1f5fa8",
            "#0f3a6b"])
# 逐次: 単一色相を淡→濃
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f2f5fa", "#cddff2", "#9fc0e6", "#6b9bd6", "#3d78c0", "#1f5fa8",
            "#123a6b"])


def grid(P, f, sd, h, num, den=None, scale=1.0):
    """(信号十分位 x 待ち行列十分位) の 10x10 を作る。"""
    d = P.filter((pl.col("feat") == f) & (pl.col("side") == sd)
                 & (pl.col("h") == h))
    a = np.full((NB, NB), np.nan)
    for r in d.iter_rows(named=True):
        dd = r[den] if den else 1.0
        if den and dd == 0:
            continue
        a[r["s_dec"], r["q_dec"]] = r[num] / dd * scale
    return a


def draw(ax, A, title, cmap, diverging, fmt="{:.2f}", robust=98.0, sub=0.0):
    v = A[np.isfinite(A)] - sub
    if diverging:
        m = float(np.percentile(np.abs(v), robust)) if v.size else 1.0
        vmin, vmax = -m, m
    else:
        vmin = float(np.percentile(v, 100 - robust)) if v.size else 0.0
        vmax = float(np.percentile(v, robust)) if v.size else 1.0
    im = ax.imshow(A - sub, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                   aspect="auto")
    for i in range(NB):
        for j in range(NB):
            if not np.isfinite(A[i, j]):
                continue
            x = A[i, j] - sub
            rel = abs(x) / max(abs(vmin), abs(vmax), 1e-12)
            ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center",
                    fontsize=5.4, color="#ffffff" if rel > 0.62 else INK)
    ax.set_xticks(range(NB), [f"D{i+1}" for i in range(NB)], fontsize=7)
    ax.set_yticks(range(NB), [f"S{i+1}" for i in range(NB)], fontsize=7)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.043, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    cb.outline.set_visible(False)
    return im


def fig_feat(P, f, h, tag, out):
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
    fig.suptitle(f"{tag}  {LAB[f]} — 信号 (縦 S1..S10) × 待ち行列がはける時間 "
                 f"(横 D1..D10)・h = {h:g} 秒・97 日",
                 color=INK, fontsize=13, y=0.985)

    draw(axes[0, 0], grid(P, f, 1, h, "n_fill", "n", 100.0),
         "(a) 約定率 % — 最良買い気配に買い指値", SEQ, False, "{:.0f}")
    draw(axes[0, 1], grid(P, f, -1, h, "n_fill", "n", 100.0),
         "(b) 約定率 % — 最良売り気配に売り指値", SEQ, False, "{:.0f}")
    N = grid(P, f, 1, h, "n")
    draw(axes[0, 2], np.log10(np.maximum(N, 1)),
         "(c) 観測数 (常用対数) — 空きセルが無いことの確認", SEQ, False,
         "{:.1f}")

    draw(axes[1, 0], grid(P, f, 1, h, "sum_ret", "n"),
         "(d) 将来リターン bp — mid で測る (符号なし)", DIV, True, "{:+.2f}")
    draw(axes[1, 1], grid(P, f, 1, h, "sum_retc", "n"),
         "(e) 将来リターン bp — microprice で測る", DIV, True, "{:+.2f}")
    draw(axes[1, 2], grid(P, f, 1, h, "sum_pnl", "n_fill") - MAKER_FEE_BP,
         "(f) 約定したときの損益 bp — 買い指値・手数料控除後", DIV, True,
         "{:+.1f}")

    fig.tight_layout(rect=(0, 0.005, 1, 0.962))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def gradients(A):
    """縦 (信号) と横 (待ち行列) の勾配の大きさを返す。"""
    r = np.nanmean(A, axis=1)
    c = np.nanmean(A, axis=0)
    return float(np.nanmax(r) - np.nanmin(r)), float(np.nanmax(c) - np.nanmin(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--h", type=float, default=10.0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    P = pl.read_csv(ROOT / "data" / f"heat_pooled_{tag}.csv")
    ch = ROOT / "charts"
    rows = []
    for f in LAB:
        for h in sorted(P["h"].unique().to_list()):
            for nm, A in (("fill_rate", grid(P, f, 1, h, "n_fill", "n", 100.0)),
                          ("ret_mid", grid(P, f, 1, h, "sum_ret", "n")),
                          ("ret_micro", grid(P, f, 1, h, "sum_retc", "n")),
                          ("pnl", grid(P, f, 1, h, "sum_pnl", "n_fill"))):
                gs, gq = gradients(A)
                rows.append({"feat": f, "h": h, "metric": nm,
                             "grad_signal": gs, "grad_queue": gq,
                             "ratio_signal_over_queue": gs / gq if gq else np.nan,
                             "vmin": float(np.nanmin(A)),
                             "vmax": float(np.nanmax(A))})
        fig_feat(P, f, a.h, a.coin, ch / f"{tag}_heat_{f}.png")
    S = pl.DataFrame(rows)
    S.write_csv(ROOT / "data" / f"heat_summary_{tag}.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=130):
        print(S.filter(pl.col("h") == a.h))


if __name__ == "__main__":
    main()
