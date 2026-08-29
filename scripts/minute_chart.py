"""1 分粒度の相対 MSE(図 25)。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CH = Path("C:/Users/ii562/Downloads/Memory/charts")
C1, C2, C3, CM, INK, GRID = "#3b6fd4", "#c2410c", "#0f766e", "#9aa3b2", "#1f2733", "#e3e7ee"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":9,"font.family":"sans-serif",
    "font.sans-serif":["Yu Gothic","Meiryo","MS Gothic","DejaVu Sans"],"axes.unicode_minus":False,
    "axes.edgecolor":GRID,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"figure.autolayout":True})

m = pl.read_parquet(D/"minute_relmse.parquet").filter((pl.col("n")>=30)&(pl.col("n_bad")==0))
r = m["relmse_raw"].to_numpy(); c = m["corr_raw"].to_numpy(); v = m["var_raw"].to_numpy()
ok = np.isfinite(r)&np.isfinite(c)&np.isfinite(v)&(v>0); r,c,v = r[ok],c[ok],v[ok]
k = c + np.sqrt(np.maximum(r-1+c**2, 0))
ed = np.quantile(v, np.linspace(0,1,11))
dec, med, shr, kk, cc = [], [], [], [], []
for i in range(10):
    s = (v>=ed[i]) & ((v<=ed[i+1]) if i==9 else (v<ed[i+1]))
    dec.append(np.sqrt(np.median(v[s]))); med.append(np.median(r[s])); shr.append((r[s]<1).mean())
    kk.append(np.median(k[s])); cc.append(np.median(np.abs(c[s])))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
x = np.arange(1, 11)
axes[0].bar(x, med, color=[C2 if u>1 else C1 for u in med], width=0.65)
axes[0].axhline(1, color=INK, linewidth=1, linestyle="--")
axes[0].set_xticks(x, [f"{d:.2f}" for d in dec], rotation=40, ha="right", fontsize=7.5)
axes[0].set_xlabel("その分の σ_y (bp) — 分散十分位")
axes[0].set_ylabel("相対 MSE の中央値")
axes[0].set_title("静かな分ほどモデルが害になる")
axes[0].text(1.2, med[0]*0.93, f"{med[0]:.2f}", color=C2, fontsize=9)

axes[1].plot(x, kk, marker="o", color=C2, linewidth=2, label="k = 予測SD / 実現SD")
axes[1].plot(x, cc, marker="o", color=C1, linewidth=2, label="|相関|")
axes[1].set_xticks(x, [f"{d:.2f}" for d in dec], rotation=40, ha="right", fontsize=7.5)
axes[1].set_xlabel("その分の σ_y (bp)")
axes[1].set_title("相関は一定・尺度 k だけが動く")
axes[1].legend(frameon=False, fontsize=8)

lams = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
rm = [np.median(1-2*c*(k*l)+(k*l)**2) for l in lams]
sh = [float((1-2*c*(k*l)+(k*l)**2 < 1).mean()) for l in lams]
axes[2].plot(lams, rm, marker="o", color=C1, linewidth=2, label="相対 MSE 中央値")
axes[2].axhline(1, color=INK, linewidth=1, linestyle="--")
ax2 = axes[2].twiny()
ax2.axis("off")
for xx, yy, s2 in zip(lams, rm, sh):
    axes[2].annotate(f"{s2:.0%}", (xx, yy), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7.5, color=CM)
axes[2].invert_xaxis()
axes[2].set_xlabel("予測に掛ける縮小係数 λ")
axes[2].set_ylabel("相対 MSE の中央値")
axes[2].set_title("λ≈0.4 で中央値 0.990(注記は <1 の割合)")
axes[2].legend(frameon=False, fontsize=8)
fig.savefig(CH / "xyz_DRAM_25_minute_relmse.png")
print("ok")
