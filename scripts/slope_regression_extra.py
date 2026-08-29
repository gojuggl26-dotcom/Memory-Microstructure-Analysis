"""追加検定: (1) 対称性 beta_A + beta_B = 0、(2) 経済的大きさ、(3) X 側の退化・共線性。"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL, SLOPE = D/"panel_v2", D/"slope25"
DROP = ["book_slope_bid","book_slope_ask","book_slope_diff"]
W = 0.001
days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
fr=[]
for k,dt in enumerate(days):
    p = pl.read_parquet(PANEL/f"dt={dt}"/"part-000.parquet").drop(DROP)
    s = pl.read_parquet(SLOPE/f"{dt}.parquet").select(["ts","s_bid","s_ask"]).rename({"s_bid":"S_bid","s_ask":"S_ask"})
    fr.append(p.join(s,on="ts",how="inner").with_columns(pl.lit(k).alias("day")))
df = pl.concat(fr)
other=[c for c in df.columns if c not in ("ts","y","day","S_bid","S_ask")]
day=df["day"].to_numpy(); y=df["y"].to_numpy()
y=np.clip(y,np.quantile(y,W),np.quantile(y,1-W))
def prep(A):
    l,h=np.quantile(A,W,axis=0),np.quantile(A,1-W,axis=0); return np.clip(A,l,h)
S=prep(df.select(["S_ask","S_bid"]).to_numpy()); Xo=prep(df.select(other).to_numpy())

# (3) 退化と共線性
sd=Xo.std(0)
deg=[other[i] for i in range(len(other)) if sd[i]==0]
print("分散ゼロの変数:", deg if deg else "なし")
C=np.corrcoef(np.column_stack([S,Xo]),rowvar=False)
nm=["S_ask","S_bid"]+other
pairs=[(nm[i],nm[j],C[i,j]) for i in range(len(nm)) for j in range(i+1,len(nm)) if abs(C[i,j])>0.9]
print("相関 0.9 超の組:")
for a,b,c in sorted(pairs,key=lambda x:-abs(x[2]))[:8]: print(f"   {a} ↔ {b}: {c:+.4f}")
print(f"S_ask ↔ S_bid の相関: {C[0,1]:+.4f}")

# (1) 対称性検定(S のみのモデルで、日次クラスター分散を使う)
A=np.column_stack([np.ones(len(y)),S])
XtX=A.T@A; b=np.linalg.solve(XtX,A.T@y); u=y-A@b; Ainv=np.linalg.inv(XtX)
g=A*u[:,None]; Sc=np.zeros((3,3))
for d0 in np.unique(day):
    gg=g[day==d0].sum(0); Sc+=np.outer(gg,gg)
G=len(np.unique(day)); ssc=G/(G-1)*(len(y)-1)/(len(y)-3)
V=Ainv@Sc@Ainv*ssc
c=np.array([0.,1.,1.])                     # beta_A + beta_B
est=float(c@b); se=math.sqrt(float(c@V@c))
print(f"\n(1) 対称性 β_A + β_B = 0 の検定")
print(f"    β_A = {b[1]:+.6f}   β_B = {b[2]:+.6f}   和 = {est:+.6f}  SE = {se:.6f}  t₉₈ = {est/se:+.2f}")
print(f"    → 和は個々の大きさの {abs(est)/((abs(b[1])+abs(b[2]))/2)*100:.1f}%")

# (2) 経済的大きさ
r=json.loads((D/"slope_regression.json").read_text(encoding="utf-8"))
bf=r["model_full"]["coef"]
sd_s=S.std(0)
sp=df["spread_bp"].to_numpy() if "spread_bp" in df.columns else None
half=float(np.median(sp))/2 if sp is not None else float("nan")
print(f"\n(2) 経済的大きさ(S + X モデル)")
for i,k in enumerate(["S_ask","S_bid"]):
    print(f"    {k}: β={bf[k]['beta']:+.6f} × SD({k})={sd_s[i]:.3f}bp → 1SD あたり {abs(bf[k]['beta'])*sd_s[i]:.4f}bp")
print(f"    ハーフスプレッドの中央値: {half:.4f}bp")
json.dump({"symmetry":{"beta_A":b[1],"beta_B":b[2],"sum":est,"se":se,"t98":est/se},
           "corr_S_ask_S_bid":float(C[0,1]),
           "high_corr_pairs":[[a,b_,float(c_)] for a,b_,c_ in pairs],
           "degenerate":deg,
           "sd_S_ask":float(sd_s[0]),"sd_S_bid":float(sd_s[1]),"half_spread_median_bp":half},
          open(D/"slope_regression_extra.json","w"),indent=1,ensure_ascii=False)
