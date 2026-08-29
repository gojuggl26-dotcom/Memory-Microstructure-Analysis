"""補足: 不利な約定のうち「シグナルが事前に警告していた」割合と、δ 別の回避可能損失。"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); SL = D/"book_slope_grain"
FEE=0.088; BAND="25bp"
LAGS={"0ms":0,"100ms":10**8,"200ms":2*10**8,"500ms":5*10**8,"1s":10**9}
rows=[]
for f in sorted(SL.glob("*.parquet")):
    dt=f.stem
    fs=sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet",recursive=True))
    if not fs: continue
    fl=pl.concat([pl.read_parquet(x,columns=["ts","px","side","crossed","tid"]) for x in fs],how="diagonal_relaxed")
    fl=(fl.filter(~pl.col("crossed")).unique(subset=["tid"],keep="first")
          .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort("t"))
    m=(pl.read_parquet(D/f"microprice/dt={dt}/part-000.parquet",columns=["ts","mid","is_crossed"])
       .filter(~pl.col("is_crossed")).sort("ts"))
    s=pl.read_parquet(f); sig=s[f"bid_{BAND}"].to_numpy()-s[f"ask_{BAND}"].to_numpy(); sts=s["ts"].to_numpy()
    t=fl["t"].to_numpy(); px=fl["px"].to_numpy(); sd=np.where(fl["side"].to_numpy()=="B",1.,-1.)
    mts=m["ts"].to_numpy(); mv=m["mid"].to_numpy()
    j0=np.searchsorted(mts,t,side="right")-1; j1=np.searchsorted(mts,t+10**9,side="right")-1
    ok=(j0>=0)&(j1>=0)&(t+10**9<=mts[-1])
    m0=mv[np.clip(j0,0,mv.size-1)]
    r=sd*(mv[np.clip(j1,0,mv.size-1)]-px)/m0*1e4-FEE
    adv=ok&(r<0)
    row={"dt":dt,"n_fills":int(ok.sum()),"n_adverse":int(adv.sum()),
         "loss_total":float(-r[adv].sum())}
    for nm,lag in LAGS.items():
        k=np.searchsorted(sts,t-lag,side="right")-1
        g=ok&(k>=0)
        a=np.where(g,sd*(-sig[np.clip(k,0,sig.size-1)]),np.nan)
        wa=adv&g&(a<0)                                    # 警告があった不利約定
        row[f"warned_{nm}"]=float(wa.sum()/max(adv.sum(),1))
        row[f"loss_avoided_{nm}"]=float(-r[wa].sum()/max(-r[adv].sum(),1e-9))
        # 警告に従って引いた場合の全体平均(警告のあった約定を除外)
        keep=ok&g&~(a<0)
        row[f"mean_if_pull_{nm}"]=float(r[keep].mean()) if keep.sum()>100 else np.nan
    rows.append(row)
d=pl.DataFrame(rows)
d.write_csv(D/"cancel_latency_extra.csv")
print(f"{'δ':>7} {'警告のあった不利約定':>20} {'回避できる損失':>14} {'警告に従った場合の平均損益':>26}")
out={}
for nm in LAGS:
    w=d[f"warned_{nm}"].median(); l=d[f"loss_avoided_{nm}"].median(); mp=d[f"mean_if_pull_{nm}"].median()
    out[nm]={"warned_share":float(w),"loss_avoided_share":float(l),"mean_if_pull":float(mp)}
    print(f"{nm:>7} {w:>20.3f} {l:>14.3f} {mp:>+26.4f}")
out["baseline_all"]=float(d.select((pl.col("loss_total")*0).sum()).item()) # placeholder
json.dump(out,open(D/"cancel_latency_extra.json","w"),indent=1,ensure_ascii=False)
print(f"\n参考: 不利な約定は全体の {float((d['n_adverse']/d['n_fills']).median()):.3f}")
