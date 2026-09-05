"""取り消し backtest の標本外・日次 EV 系列に block bootstrap をかけ 95% 区間を出す。

    uv run python scripts/plot_cancel_ci.py --coin xyz:MU
出力: charts/<coin>_cancel_ci.png / data/cancel_ci_<coin>.csv

なぜ block bootstrap か
-----------------------
日次の系列は独立ではない。[burstiness の報告](../reports/) で見たとおり
ラグ 1 で +0.674、**ラグ 7 で +0.703**(週次の周期)の自己相関がある。
日を無作為に入れ替える普通の bootstrap は区間を過小に出す。
そこで**巡回移動ブロック bootstrap**(長さ L の連続する日をまとめて抽出)を使う。
標本は評価 39 日しかないので、L = 1 / 3 / 7 / 14 を全部出して感度も見る。

比の推定量の扱い
----------------
EV = Σ(損益) / Σ(発注数) は比の推定量なので、日ごとの EV を平均するのではなく
**再抽出した日の分子と分母をそれぞれ足してから割る**。

対の扱い
--------
「スコア − 何もしない」「スコア − 無作為」の区間を出すときは、
**同じ日の抽出を全条件で共有する**(対応のある bootstrap)。
条件ごとに別々に抽出すると差の分散を過大に見積もる。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cancel_bt import LATENCY_MS, TRIG_RATE  # noqa: E402
from plot_vol import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = 10000
BLOCKS = [1, 3, 7, 14]
BL_MAIN = 7
LC = {0: "#0f3a6b", 65: "#1f5fa8", 130: "#7fa8dd", 200: "#e08b7f",
      300: "#b5322f"}


def legend(ax, fs=8, **kw):
    lg = ax.legend(fontsize=fs, frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   labelcolor=INK2, **kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def block_idx(n, L, B, rng):
    """巡回移動ブロックの抽出添字 (B × n)。"""
    k = int(np.ceil(n / L))
    st = rng.integers(0, n, size=(B, k))
    off = np.arange(L)
    idx = (st[:, :, None] + off[None, None, :]) % n
    return idx.reshape(B, k * L)[:, :n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--hq", type=str, default="")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    suf = f"_hq{a.hq}" if a.hq else ""
    D = ROOT / "data"
    B = pl.read_parquet(D / f"cancel_bt_{tag}{suf}.parquet").filter(
        pl.col("split") == "評価")
    # 両側をまとめる (実際の運用は両側に出すため)
    G = (B.group_by("dt", "rule", "trig_rate", "lat_ms")
         .agg(**{c: pl.col(c).sum() for c in ("n", "n_fill", "sum_pnl", "sum_mk")})
         .sort("dt"))
    days = sorted(G["dt"].unique().to_list())
    nd = len(days)
    di = {d: i for i, d in enumerate(days)}
    print(f"{a.coin}: 評価 {nd} 日。circular moving block bootstrap "
          f"B={NB:,}、ブロック長 {BLOCKS}")

    keys, arr = [], {}
    for (rule, r, L), g in G.group_by(["rule", "trig_rate", "lat_ms"],
                                      maintain_order=True):
        v = np.zeros((4, nd))
        for row in g.iter_rows(named=True):
            i = di[row["dt"]]
            v[0, i], v[1, i] = row["n"], row["n_fill"]
            v[2, i], v[3, i] = row["sum_pnl"], row["sum_mk"]
        keys.append((rule, r, L))
        arr[(rule, r, L)] = v

    rng = np.random.default_rng(20260905)
    IDX = {L: block_idx(nd, L, NB, rng) for L in BLOCKS}

    def boot(v, idx, kind):
        num = v[2 if kind == "ev" else 3][idx]
        den = v[0 if kind == "ev" else 1][idx]
        return num.sum(1) / np.maximum(den.sum(1), 1e-9)

    rows = []
    for L in BLOCKS:
        idx = IDX[L]
        base = {lat: arr[("スコア", 0.0, lat)] for lat in LATENCY_MS}
        bb = {lat: boot(base[lat], idx, "ev") for lat in LATENCY_MS}
        bm = {lat: boot(base[lat], idx, "mk") for lat in LATENCY_MS}
        for (rule, r, lat) in keys:
            v = arr[(rule, r, lat)]
            for kind, ref in (("ev", bb), ("mk", bm)):
                pt = (v[2].sum() / v[0].sum()) if kind == "ev" \
                    else (v[3].sum() / max(v[1].sum(), 1e-9))
                bs = boot(v, idx, kind)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                row = {"block": L, "rule": rule, "trig_rate": r, "lat_ms": lat,
                       "metric": kind, "point": float(pt),
                       "lo95": float(lo), "hi95": float(hi),
                       "se_boot": float(bs.std(ddof=1)),
                       "bias": float(bs.mean() - pt)}
                if r > 0:
                    dv = bs - ref[lat]
                    row["vs_base"] = float(
                        (v[2].sum() / v[0].sum() - base[lat][2].sum()
                         / base[lat][0].sum()) if kind == "ev" else
                        (v[3].sum() / v[1].sum() - base[lat][3].sum()
                         / base[lat][1].sum()))
                    row["vs_base_lo"], row["vs_base_hi"] = \
                        [float(x) for x in np.percentile(dv, [2.5, 97.5])]
                    other = "無作為" if rule == "スコア" else "スコア"
                    if (other, r, lat) in arr:
                        w = arr[(other, r, lat)]
                        dr = bs - boot(w, idx, kind)
                        pw = (w[2].sum() / w[0].sum()) if kind == "ev" \
                            else (w[3].sum() / w[1].sum())
                        row["vs_other"] = float(pt - pw)
                        row["vs_other_lo"], row["vs_other_hi"] = \
                            [float(x) for x in np.percentile(dr, [2.5, 97.5])]
                rows.append(row)
    C = pl.DataFrame(rows)
    C.write_csv(D / f"cancel_ci_{tag}{suf}.csv")

    M = C.filter((pl.col("block") == BL_MAIN) & (pl.col("metric") == "ev")
                 & (pl.col("rule") == "スコア"))
    print("\n=== 1 発注あたり EV の 95% 区間 (ブロック長 7 日)")
    print(f'{"発火率":>7}{"遅延":>6}{"EV":>10}{"95% 区間":>22}{"0 を含むか":>11}')
    for r in TRIG_RATE:
        for lat in (0, 130, 300):
            x = M.filter((pl.col("trig_rate") == r) & (pl.col("lat_ms") == lat))
            if not x.height:
                continue
            x = x.row(0, named=True)
            inc = "含む" if x["lo95"] <= 0 <= x["hi95"] else "含まない"
            print(f'{100*r:>6.0f}%{lat:>6}{x["point"]:>10.4f}'
                  f'  [{x["lo95"]:>8.4f}, {x["hi95"]:>8.4f}]{inc:>11}')

    K = C.filter((pl.col("block") == BL_MAIN) & (pl.col("metric") == "mk")
                 & (pl.col("rule") == "スコア") & (pl.col("trig_rate") > 0))
    print("\n=== markout の差の 95% 区間 (ブロック長 7 日)")
    print(f'{"発火率":>7}{"遅延":>6}{"vs 何もしない":>24}{"vs 無作為":>24}')
    for r in (0.05, 0.10, 0.20, 0.40):
        for lat in (0, 130, 300):
            x = K.filter((pl.col("trig_rate") == r) & (pl.col("lat_ms") == lat))
            if not x.height:
                continue
            x = x.row(0, named=True)
            print(f'{100*r:>6.0f}%{lat:>6}'
                  f'  {x["vs_base"]:>+7.3f} [{x["vs_base_lo"]:>+6.3f},'
                  f'{x["vs_base_hi"]:>+6.3f}]'
                  f'  {x["vs_other"]:>+7.3f} [{x["vs_other_lo"]:>+6.3f},'
                  f'{x["vs_other_hi"]:>+6.3f}]')

    print("\n=== ブロック長への感度 (発火率 20%・L=130ms・markout の差)")
    for L in BLOCKS:
        x = C.filter((pl.col("block") == L) & (pl.col("metric") == "mk")
                     & (pl.col("rule") == "スコア") & (pl.col("trig_rate") == 0.20)
                     & (pl.col("lat_ms") == 130)).row(0, named=True)
        print(f'  ブロック {L:>2} 日: vs 何もしない {x["vs_base"]:+.3f} '
              f'[{x["vs_base_lo"]:+.3f},{x["vs_base_hi"]:+.3f}]   '
              f'vs 無作為 {x["vs_other"]:+.3f} '
              f'[{x["vs_other_lo"]:+.3f},{x["vs_other_hi"]:+.3f}]')

    rr = [r for r in TRIG_RATE if r > 0]
    xr = np.array(rr) * 100
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    fig.suptitle(f"{a.coin}  標本外・日次 EV 系列の block bootstrap 95% 区間"
                 f"(評価 {nd} 日・ブロック長 {BL_MAIN} 日・B={NB:,})",
                 color=INK, fontsize=13, y=0.975)

    ax = axes[0]
    for lat in (0, 130, 300):
        q = M.filter(pl.col("lat_ms") == lat).sort("trig_rate")
        ax.fill_between(np.array(q["trig_rate"]) * 100, q["lo95"], q["hi95"],
                        color=LC[lat], alpha=0.16, lw=0)
        ax.plot(np.array(q["trig_rate"]) * 100, q["point"], color=LC[lat],
                lw=2.2, marker="o", ms=4.5, label=f"L={lat}ms")
    ax.axhline(0, color="#1f8a5e", lw=1.6, ls="--", zorder=3)
    ax.text(xr[-1], 0.008, "何もしない = 0 ", color="#1f8a5e", fontsize=8.5,
            ha="right")
    style(ax, "発火率 (%)", "1 発注あたり EV (bp)")
    ax.set_title("(a) EV は 95% 区間ごと 0 の下にある", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="lower right")

    ax = axes[1]
    for lat in (0, 130, 300):
        q = K.filter(pl.col("lat_ms") == lat).sort("trig_rate")
        ax.fill_between(np.array(q["trig_rate"]) * 100, q["vs_base_lo"],
                        q["vs_base_hi"], color=LC[lat], alpha=0.16, lw=0)
        ax.plot(np.array(q["trig_rate"]) * 100, q["vs_base"], color=LC[lat],
                lw=2.2, marker="o", ms=4.5, label=f"L={lat}ms")
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    style(ax, "発火率 (%)", "markout の改善 vs 何もしない (bp)")
    ax.set_title("(b) L=300ms では 0 を跨ぐ", color=INK, fontsize=10, loc="left")
    legend(ax, fs=7.5, loc="upper left")

    ax = axes[2]
    for lat in (0, 130, 300):
        q = K.filter(pl.col("lat_ms") == lat).sort("trig_rate")
        ax.fill_between(np.array(q["trig_rate"]) * 100, q["vs_other_lo"],
                        q["vs_other_hi"], color=LC[lat], alpha=0.16, lw=0)
        ax.plot(np.array(q["trig_rate"]) * 100, q["vs_other"], color=LC[lat],
                lw=2.2, marker="o", ms=4.5, label=f"L={lat}ms")
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    style(ax, "発火率 (%)", "markout の差 vs 無作為 (bp)")
    ax.set_title("(c) スコアは無作為より確実に良い", color=INK, fontsize=10,
                 loc="left")
    legend(ax, fs=7.5, loc="lower right")

    fig.tight_layout(rect=(0, 0.005, 1, 0.945))
    out = ROOT / "charts" / f"{tag}_cancel_ci{suf}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  {out.name}")


if __name__ == "__main__":
    main()
