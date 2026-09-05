"""δ = microprice − mid を「板の偏り」と「スプレッド幅」に分解する。

    uv run python scripts/featlib_delta.py --coin xyz:INTC

出力: data/featlib_delta_<tag>.csv(五分位表)
      data/featlib_delta_coef_<tag>.csv(日ごとの回帰係数)
      charts/<tag>_featlib_delta.png

何を測るか
----------
build_featlib.py の作りから、恒等式

```math
\\delta_{\\rm bp} = \\frac{s_{\\rm bp}}{2}\\,\\mathrm{OBI}_1
```

が成り立つ(`delta1_bp` と `obi_x_spread` は定義上同じ量)。まずこれを
数値で確かめ、そのうえで **δ の予測力が OBI 由来なのかスプレッド由来なのか**を
分ける。片方だけを見ていると「厚みの偏り」と「幅の広さ」を取り違える。

時間契約
--------
`x` は格子点 `T` までの情報だけ。`y` は `(T, T+h]` の microprice の log リターン。
`shift(-k)` は `y` にしか使わない。五分位の閾値は **その日より前の日**の分布から
決める(全標本から作らない)。標準誤差は日単位でクラスタする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import C1, C2, C3, CM, D, plt, save  # noqa: E402

COLS = ["mid", "spread_bp", "obi1", "delta1_bp", "obi_x_spread", "micro1",
        "fwd_micro_1s", "fwd_micro_10s", "fwd_micro_60s",
        "fwd_mid_1s", "fwd_mid_10s", "fwd_mid_60s"]
HS = ["fwd_micro_1s", "fwd_micro_10s", "fwd_micro_60s"]
NQ = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((D / f"featlib_{tag}").glob("dt=*.parquet"))
    if not files:
        sys.exit("featlib が無い")

    id_err = 0.0
    prev_edges = None
    qrows, crows = [], []
    for fp in files:
        d = pl.read_parquet(fp, columns=[c for c in COLS if c])
        dt = fp.stem.split("=")[1]
        sp = d["spread_bp"].to_numpy().astype(np.float64)
        ob = d["obi1"].to_numpy().astype(np.float64)
        de = d["delta1_bp"].to_numpy().astype(np.float64)
        m0 = np.isfinite(sp) & np.isfinite(ob) & np.isfinite(de)
        id_err = max(id_err, float(np.nanmax(np.abs(de[m0] - sp[m0] / 2 * ob[m0]))))

        y = {h: d[h].to_numpy().astype(np.float64) for h in HS}
        # ---- 五分位: 閾値は「前の日まで」の分布から(標本外) --------------
        if prev_edges is not None:
            q = np.searchsorted(prev_edges, de, side="right")
            q = np.clip(q, 0, NQ - 1)
            for k in range(NQ):
                m = m0 & (q == k)
                r = {"dt": dt, "q": k + 1, "n": int(m.sum()),
                     "delta_mid": float(np.nanmedian(de[m])) if m.any() else np.nan}
                for h in HS:
                    mm = m & np.isfinite(y[h])
                    r[h] = float(y[h][mm].mean()) if mm.sum() > 20 else np.nan
                qrows.append(r)
        v = de[m0]
        if v.size > 1000:
            e = np.percentile(v, [20, 40, 60, 80])
            prev_edges = e if prev_edges is None else 0.7 * prev_edges + 0.3 * e

        # ---- 回帰: y ~ obi1 + spread_bp + obi1*spread_bp -------------------
        for h in HS:
            m = m0 & np.isfinite(y[h])
            if m.sum() < 2000:
                continue
            X = np.column_stack([np.ones(m.sum()), ob[m], sp[m], de[m]])
            Xs = X.copy()
            for j in (1, 2, 3):                       # 1σ で測るため標準化
                s = Xs[:, j].std()
                Xs[:, j] = (Xs[:, j] - Xs[:, j].mean()) / (s if s > 1e-12 else 1.0)
            yy = y[h][m]
            b, *_ = np.linalg.lstsq(Xs, yy, rcond=None)
            b1, *_ = np.linalg.lstsq(Xs[:, [0, 1]], yy, rcond=None)     # OBI だけ
            b3, *_ = np.linalg.lstsq(Xs[:, [0, 3]], yy, rcond=None)     # δ だけ
            crows.append({"dt": dt, "h": h, "b_obi": b[1], "b_spread": b[2],
                          "b_delta": b[3], "solo_obi": b1[1], "solo_delta": b3[1],
                          "n": int(m.sum())})

    Q = pl.DataFrame(qrows)
    C = pl.DataFrame(crows)
    Q.write_csv(D / f"featlib_delta_{tag}.csv")
    C.write_csv(D / f"featlib_delta_coef_{tag}.csv")
    print(f"恒等式 δ = (s/2)·OBI1 の最大誤差 {id_err:.3e} bp")

    def clus(df, col):
        v = df[col].to_numpy()
        v = v[np.isfinite(v)]
        return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(v.size)) if v.size > 2 else np.nan

    print("\n五分位ごとの将来 microprice リターン (bp, 日クラスタ t)")
    tbl = []
    for k in range(1, NQ + 1):
        s = Q.filter(pl.col("q") == k)
        row = {"q": k, "n": int(s["n"].sum()),
               "delta_bp": float(s["delta_mid"].median())}
        for h in HS:
            m, t = clus(s, h)
            row[h] = m
            row[h + "_t"] = t
        tbl.append(row)
        print(f"  Q{k} n={row['n']:>10,} δ={row['delta_bp']:+7.3f}bp  " +
              "  ".join(f"{h.replace('fwd_micro_','')}: {row[h]:+7.4f} (t={row[h+'_t']:+6.2f})"
                        for h in HS))
    print("\n回帰係数 (1σ あたり bp, 日クラスタ t)")
    for h in HS:
        s = C.filter(pl.col("h") == h)
        out = []
        for c in ("solo_obi", "solo_delta", "b_obi", "b_spread", "b_delta"):
            m, t = clus(s, c)
            out.append(f"{c} {m:+7.4f}(t={t:+6.2f})")
        print(f"  {h}: " + "  ".join(out))

    # ---- 図 ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.9))
    a0 = ax[0]
    x = [r["delta_bp"] for r in tbl]
    for h, c in zip(HS, (C3, C1, C2)):
        a0.plot(x, [r[h] for r in tbl], marker="o", ms=4, lw=1.2, color=c,
                label=h.replace("fwd_micro_", "h="))
    a0.axhline(0, color=CM, lw=0.8)
    a0.set_xlabel("δ = microprice − mid の五分位中央値 (bp)")
    a0.set_ylabel("将来 microprice リターン (bp)")
    a0.set_title("A. δ の五分位 → 将来リターン")
    a0.legend(fontsize=7, frameon=False)

    a1 = ax[1]
    lab = ["OBI だけ", "δ だけ", "OBI(3 変数)", "spread(3 変数)", "δ(3 変数)"]
    key = ["solo_obi", "solo_delta", "b_obi", "b_spread", "b_delta"]
    w = 0.26
    for i, (h, c) in enumerate(zip(HS, (C3, C1, C2))):
        s = C.filter(pl.col("h") == h)
        v = [clus(s, k)[0] for k in key]
        a1.bar(np.arange(len(key)) + i * w, v, width=w, color=c,
               label=h.replace("fwd_micro_", "h="))
    a1.axhline(0, color=CM, lw=0.8)
    a1.set_xticks(np.arange(len(key)) + w)
    a1.set_xticklabels(lab, fontsize=7, rotation=12)
    a1.set_ylabel("1σ あたり bp")
    a1.set_title("B. δ の予測力は OBI 由来か 幅 由来か")
    a1.legend(fontsize=7, frameon=False)

    a2 = ax[2]
    s = C.filter(pl.col("h") == "fwd_micro_10s")
    a2.plot(np.arange(s.height), s["solo_delta"], color=C1, lw=1.0, label="δ だけ")
    a2.plot(np.arange(s.height), s["b_obi"], color=C3, lw=1.0, label="OBI(3 変数)")
    a2.plot(np.arange(s.height), s["b_spread"], color=C2, lw=1.0, label="spread(3 変数)")
    a2.axhline(0, color=CM, lw=0.8)
    a2.set_xlabel("日 (標本の通し番号)")
    a2.set_ylabel("1σ あたり bp")
    a2.set_title("C. 係数の日ごとの安定性 (h=10s)")
    a2.legend(fontsize=7, frameon=False)
    save(fig, f"{tag}_featlib_delta.png",
         f"{a.coin}: δ = microprice − mid の分解")


if __name__ == "__main__":
    main()
