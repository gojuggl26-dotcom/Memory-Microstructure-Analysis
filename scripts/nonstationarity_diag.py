"""「分散の非定常性が問題なのか」を分離して測る。

候補 3 つ:
  (a) 尺度の非定常性   … 日ごとに σ_y が桁違い。相関は不変なので、正規化で完全に消えるはず
  (b) 関係の非定常性   … 相関そのものが時間で変わる。正規化では消えない
  (c) 目的変数の退化   … 1 秒窓で y が厳密に 0 の割合が高い。相関が測れない

β_d = corr_d × σ_y,d / σ_x,d なので、log|β| の分散を
log|corr| と log σ_y と log σ_x に分解すれば (a) と (b) の寄与が出る。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); PANEL = D / "panel_1s"
FEATS = ["depth_one_level", "obi_2", "voi_sum", "ofi_sum", "oi", "r_prev", "m_bp"]

rows = []
for k, dt in enumerate(sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())):
    df = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet")
    y = df["y"].to_numpy()
    if y.size < 1000:
        continue
    nz = np.abs(y) > 1e-12
    r = {"dt": dt, "day": k, "n": int(y.size), "zero_share": float(1 - nz.mean()),
         "sigma_y": float(y.std())}
    for f in FEATS:
        x = df[f].to_numpy().astype(float)
        r[f"corr_{f}"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else np.nan
        r[f"beta_{f}"] = float(np.cov(x, y)[0, 1] / x.var()) if x.var() > 0 else np.nan
        r[f"sigx_{f}"] = float(x.std())
        # y が動いた窓だけの相関(退化の影響を除く)
        if nz.sum() > 500 and x[nz].std() > 0:
            r[f"corrnz_{f}"] = float(np.corrcoef(x[nz], y[nz])[0, 1])
        else:
            r[f"corrnz_{f}"] = np.nan
    rows.append(r)

d = pl.DataFrame(rows)
d.write_csv(D / "nonstationarity_daily.csv")
out = {"days": d.height}

print(f"{'月':>8} {'ゼロ率':>7} {'σ_y':>7} " + " ".join(f"{f[:12]:>13}" for f in FEATS[:4]))
mon = d.with_columns(pl.col("dt").str.slice(0, 7).alias("m"))
for m in sorted(set(mon["m"].to_list())):
    s = mon.filter(pl.col("m") == m)
    print(f"{m:>8} {s['zero_share'].median():>7.3f} {s['sigma_y'].median():>7.3f} " +
          " ".join(f"{s['corr_'+f].median():>13.4f}" for f in FEATS[:4]))
print(f"{'(y≠0)':>8} {'':>7} {'':>7} " + " ".join(f"{f[:12]:>13}" for f in FEATS[:4]))
for m in sorted(set(mon["m"].to_list())):
    s = mon.filter(pl.col("m") == m)
    print(f"{m:>8} {'':>7} {'':>7} " +
          " ".join(f"{s['corrnz_'+f].median():>13.4f}" for f in FEATS[:4]))

# --- log|β| の分散分解 ---
print("\n=== log|beta| の分散分解(全 98 日) ===")
print(f"{'feature':>18} {'var(log|b|)':>12} {'corr 寄与':>10} {'σ_y 寄与':>10} {'σ_x 寄与':>10} "
      f"{'相関の時間トレンド':>12}")
for f in FEATS:
    b = np.abs(d[f"beta_{f}"].to_numpy()); c = np.abs(d[f"corr_{f}"].to_numpy())
    sy = d["sigma_y"].to_numpy(); sx = d[f"sigx_{f}"].to_numpy()
    m = (b > 0) & (c > 0) & (sx > 0)
    lb, lc, ly, lx = np.log(b[m]), np.log(c[m]), np.log(sy[m]), np.log(sx[m])
    v = lb.var()
    # cov(lb, ·)/var(lb) で寄与を出す(lb = lc + ly − lx なので 3 つの和は 1)
    ctr = np.cov(lb, lc)[0, 1] / v, np.cov(lb, ly)[0, 1] / v, -np.cov(lb, lx)[0, 1] / v
    day = d["day"].to_numpy()[m]
    tr = np.corrcoef(day, c[m])[0, 1]
    print(f"{f:>18} {v:>12.4f} {ctr[0]:>10.3f} {ctr[1]:>10.3f} {ctr[2]:>10.3f} {tr:>12.3f}")
    out[f] = {"var_log_beta": float(v), "share_corr": float(ctr[0]),
              "share_sigma_y": float(ctr[1]), "share_sigma_x": float(ctr[2]),
              "corr_vs_time": float(tr)}
(D / "nonstationarity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
