"""指値 1 本ごとの記録から、L4 でしか作れない特徴量を集計する。

    uv run python scripts/analyze_orderlife.py --coin xyz:INTC

出力: data/orderlife_summary_<tag>.csv / data/orderlife_grid_<tag>.csv
      charts/<tag>_orderlife.png

測るもの
--------
1. 寿命(生存時間)の分布と、束の間の注文(fleeting)の割合
2. **発注時に判る 2 条件**での約定確率
     * `dist_tick` … 同じ側の最良から何ティック奥か(負 = スプレッドの内側)
     * `ahead_lots` … 同じ価格に既に積まれていた数量(= キューの前の量)
3. 注文の結末の内訳と order-to-trade 比

時間契約
--------
層別に使う `dist_tick` / `ahead_lots` / `spread_tick` / `orig_lots` は
**発注の瞬間に確定**する。目的変数(約定したか・寿命)は発注より後の情報なので、
層別は先読みにならない。

★分母の取り方
-------------
約定確率は **本数基準**と **数量基準**の 2 通りで出す。同じ現象でも
分母が変わると 3 割ずれることがある(`mu_wallet_fill_report.md`)。
`reduce_only` はポジション減少で取引所が残量を縮めるため、残量の減少が
約定を意味しない。数量基準からは必ず除く。

窓の終わりまで決着しなかった注文(右打ち切り)は集計に入れない。件数は報告する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, C4, CM, D, plt, save  # noqa: E402
from build_obi_levels import SZ_LOT  # noqa: E402

DBIN = [-1e9, -0.5, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 1e9]
DLAB = ["内側", "最良", "1–2", "3–5", "6–10", "11–25", "26–50", "51+"]
ABIN = [-1e-9, 1e-9, 1.0, 10.0, 100.0, 1000.0, 1e18]
ALAB = ["先客なし", "≤1", "1–10", "10–100", "100–1k", ">1k"]
FLEET = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((D / f"orderlife_{tag}").glob("dt=*.parquet"))
    if not files:
        sys.exit("orderlife が無い")
    cen = D / f"orderlife_{tag}_censored.parquet"
    n_cen = pl.read_parquet(cen).height if cen.exists() else 0

    n = 0
    life = []
    st_cnt: dict[str, int] = {}
    grid_n = np.zeros((len(DLAB), len(ALAB)))
    grid_f = np.zeros((len(DLAB), len(ALAB)))
    grid_v = np.zeros((len(DLAB), len(ALAB)))
    grid_fv = np.zeros((len(DLAB), len(ALAB)))
    fleet_n = np.zeros(len(FLEET))
    fleet_v = np.zeros(len(FLEET))
    tot_v = 0.0
    fill_v = 0.0
    fill_n = 0
    dist_hist = np.zeros(len(DLAB))
    life_by_d = [[] for _ in DLAB]
    for fp in files:
        d = pl.read_parquet(fp, columns=["dist_tick", "ahead_lots", "orig_lots",
                                         "filled_lots", "life_ns", "status",
                                         "reduce_only", "is_bid"])
        n += d.height
        for k, v in d["status"].value_counts().rows():
            st_cnt[k] = st_cnt.get(k, 0) + v
        dt_ = d["dist_tick"].to_numpy().astype(np.float64)
        dt_ = np.where(dt_ < -900, np.nan, dt_)          # 片側が空の瞬間は除く
        ah = d["ahead_lots"].to_numpy().astype(np.float64) * SZ_LOT
        og = d["orig_lots"].to_numpy().astype(np.float64) * SZ_LOT
        fl = d["filled_lots"].to_numpy().astype(np.float64) * SZ_LOT
        ln = d["life_ns"].to_numpy().astype(np.float64) / 1e9
        ro = d["reduce_only"].to_numpy()
        good = np.isfinite(dt_) & (~ro)
        i = np.clip(np.digitize(dt_, DBIN) - 1, 0, len(DLAB) - 1)
        j = np.clip(np.digitize(ah, ABIN) - 1, 0, len(ALAB) - 1)
        f = fl > 0
        np.add.at(grid_n, (i[good], j[good]), 1.0)
        np.add.at(grid_f, (i[good], j[good]), f[good].astype(float))
        np.add.at(grid_v, (i[good], j[good]), og[good])
        np.add.at(grid_fv, (i[good], j[good]), fl[good])
        np.add.at(dist_hist, i[good], 1.0)
        tot_v += og[good].sum()
        fill_v += fl[good].sum()
        fill_n += int(f[good].sum())
        for k, th in enumerate(FLEET):
            m = good & (ln <= th) & (fl <= 0)
            fleet_n[k] += m.sum()
            fleet_v[k] += og[m].sum()
        life.append(np.random.default_rng(0).choice(
            ln[good], size=min(200_000, int(good.sum())), replace=False))
        for k in range(len(DLAB)):
            m = good & (i == k)
            if m.sum() > 50:
                life_by_d[k].append(np.percentile(ln[m], [10, 50, 90]))
    L = np.concatenate(life)
    ng = grid_n.sum()

    print(f"{a.coin}: 決着した指値 {n:,} 本(右打ち切り {n_cen:,} 本は除外)")
    print(f"  reduce_only を除いた集計対象 {ng:,.0f} 本 / {tot_v:,.0f} 枚")
    print(f"  一度でも約定した本数 {fill_n:,} = {fill_n/ng*100:.3f}%")
    print(f"  約定した数量 {fill_v:,.0f} 枚 = {fill_v/tot_v*100:.3f}%")
    print(f"  寿命 p10 {np.percentile(L,10):.3f}s  中央 {np.percentile(L,50):.3f}s  "
          f"p90 {np.percentile(L,90):.2f}s  p99 {np.percentile(L,99):.1f}s")
    print("  結末:", dict(sorted(st_cnt.items(), key=lambda x: -x[1])))
    print("\n  束の間の注文(約定せずに閾値以内で消えた割合)")
    for k, th in enumerate(FLEET):
        print(f"    {th:>5.2f}s 以内   本数 {fleet_n[k]/ng*100:6.2f}%   "
              f"数量 {fleet_v[k]/tot_v*100:6.2f}%")

    with np.errstate(invalid="ignore", divide="ignore"):
        pn = np.where(grid_n > 200, grid_f / np.maximum(grid_n, 1), np.nan)
        pv = np.where(grid_v > 0, grid_fv / np.maximum(grid_v, 1e-9), np.nan)
    rows = []
    for i, dl in enumerate(DLAB):
        for j, al in enumerate(ALAB):
            rows.append({"dist": dl, "ahead": al, "n": grid_n[i, j],
                         "p_fill_count": pn[i, j], "p_fill_volume": pv[i, j],
                         "vol": grid_v[i, j]})
    pl.DataFrame(rows).write_csv(D / f"orderlife_grid_{tag}.csv")
    pl.DataFrame({"metric": ["n_orders", "n_censored", "p_fill_count",
                             "p_fill_volume", "life_p50", "life_p90"],
                  "value": [ng, n_cen, fill_n / ng, fill_v / tot_v,
                            float(np.percentile(L, 50)), float(np.percentile(L, 90))]}
                 ).write_csv(D / f"orderlife_summary_{tag}.csv")

    print("\n  約定確率(本数基準・行 = 最良からの距離、列 = 前に居た量)")
    print("            " + "".join(f"{s:>10s}" for s in ALAB))
    for i, dl in enumerate(DLAB):
        print(f"    {dl:>8s}" + "".join(
            ("    ―     " if not np.isfinite(pn[i, j]) else f"{pn[i,j]*100:9.3f}%")
            for j in range(len(ALAB))))

    # ---- 図 ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11.8, 6.8))
    a0 = ax[0, 0]
    xs = np.logspace(-3, 4, 120)
    a0.hist(np.clip(L, 1e-3, 1e4), bins=xs, color=C1)
    a0.set_xscale("log")
    a0.set_xlabel("寿命(秒・対数軸)")
    a0.set_ylabel("本数")
    a0.axvline(np.percentile(L, 50), color=C2, lw=1.0)
    a0.text(np.percentile(L, 50) * 1.15, a0.get_ylim()[1] * 0.85,
            f"中央 {np.percentile(L,50):.2f}s", color=C2, fontsize=7)
    a0.set_title("A. 指値が板に居た時間")

    a1 = ax[0, 1]
    a1.plot(FLEET, fleet_n / ng * 100, marker="o", ms=4, color=C1, label="本数基準")
    a1.plot(FLEET, fleet_v / tot_v * 100, marker="s", ms=4, color=C2, label="数量基準")
    a1.set_xscale("log")
    a1.set_xlabel("閾値(秒)")
    a1.set_ylabel("%")
    a1.set_title("B. 約定せずに閾値以内で消えた割合")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[1, 0]
    im = a2.imshow(pn * 100, cmap="viridis", aspect="auto")
    a2.set_xticks(range(len(ALAB)))
    a2.set_xticklabels(ALAB, fontsize=7, rotation=20)
    a2.set_yticks(range(len(DLAB)))
    a2.set_yticklabels(DLAB, fontsize=7)
    a2.set_xlabel("同じ価格に前から居た数量(枚)")
    a2.set_ylabel("同じ側の最良からの距離(ティック)")
    a2.grid(False)
    for i in range(len(DLAB)):
        for j in range(len(ALAB)):
            if np.isfinite(pn[i, j]):
                a2.text(j, i, f"{pn[i,j]*100:.2f}", ha="center", va="center",
                        fontsize=6, color="w")
    fig.colorbar(im, ax=a2, fraction=0.04, label="約定確率 (%)")
    a2.set_title("C. 発注時に判る 2 条件での約定確率")

    a3 = ax[1, 1]
    q = np.array([np.median(np.array(v), axis=0) if v else [np.nan] * 3
                  for v in life_by_d])
    xx = np.arange(len(DLAB))
    a3.plot(xx, q[:, 1], marker="o", ms=4, color=C1, label="中央値")
    a3.fill_between(xx, q[:, 0], q[:, 2], color=C1, alpha=0.18, lw=0,
                    label="10–90% 点")
    a3.set_yscale("log")
    a3.set_xticks(xx)
    a3.set_xticklabels(DLAB, fontsize=7, rotation=20)
    a3.set_xlabel("同じ側の最良からの距離(ティック)")
    a3.set_ylabel("寿命(秒)")
    a3.set_title("D. 置いた場所と寿命")
    a3.legend(fontsize=7, frameon=False)
    save(fig, f"{tag}_orderlife.png",
         f"{a.coin}: 指値 {ng:,.0f} 本の寿命とキュー位置別の約定確率")


if __name__ == "__main__":
    main()
