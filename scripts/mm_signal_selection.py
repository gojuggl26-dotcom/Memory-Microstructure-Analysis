"""シグナルで約定を選別するとメイカー損益がどう変わるか。

メイカー約定ごとに、約定時点で判っているシグナル(book_slope_diff)を引き当て、
「その約定が自分に有利な向きだったか」でグループ分けして実現損益を比べる。

  予測リターン ∝ −book_slope_diff (相関 −0.135)
  整合度 align = 方向 s × 予測リターン    … 正なら「シグナルと同じ向きに約定した」

★時間契約: シグナルは約定時刻以前の直近 1 秒グリッド点の値のみ(backward)。
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
DAYS = ["2026-07-20","2026-07-21","2026-07-22","2026-07-23","2026-07-24",
        "2026-07-27","2026-07-28","2026-07-29","2026-08-03","2026-08-05"]
FEE = 0.088
rows = []
for dt in DAYS:
    fs = sorted(glob.glob(f'data/fills_v99/dt={dt}/**/*.parquet', recursive=True))
    if not fs: continue
    f = pl.concat([pl.read_parquet(x, columns=["ts","px","sz","side","crossed","tid"])
                   for x in fs], how="diagonal_relaxed")
    f = (f.filter(~pl.col("crossed")).unique(subset=["tid"], keep="first")
          .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort("t"))
    m = (pl.read_parquet(D/f"microprice/dt={dt}/part-000.parquet",
                         columns=["ts","mid","is_crossed"]).filter(~pl.col("is_crossed")).sort("ts"))
    sl = pl.read_parquet(D/f"slope25/{dt}.parquet").select(["ts","s_bid","s_ask"]).sort("ts")
    mts=m["ts"].to_numpy(); mv=m["mid"].to_numpy()
    sts=sl["ts"].to_numpy(); sd=(sl["s_bid"].to_numpy()-sl["s_ask"].to_numpy())
    t=f["t"].to_numpy(); p=f["px"].to_numpy()
    s=np.where(f["side"].to_numpy()=="B",1.0,-1.0)
    j0=np.searchsorted(mts,t,side="right")-1
    js=np.searchsorted(sts,t,side="right")-1              # ★backward のみ
    ok=(j0>=0)&(js>=0)
    m0=mv[np.clip(j0,0,mv.size-1)]
    sig=-sd[np.clip(js,0,sd.size-1)]                      # 予測リターンの向き
    for h,ns in (("1s",10**9),("10s",10**10)):
        j1=np.searchsorted(mts,t+ns,side="right")-1
        k=ok&(t+ns<=mts[-1])&(j1>=0)
        m1=mv[np.clip(j1,0,mv.size-1)]
        r=s*(m1-p)/m0*1e4
        rows.append(pl.DataFrame({"h":[h]*int(k.sum()),"align":(s*sig)[k],
                                  "r":r[k],"sz":f["sz"].to_numpy()[k]}))
df=pl.concat(rows)
out={}
print(f"{'地平':>5} {'整合度十分位':>12} {'件数':>10} {'実現損益':>10} {'手数料後':>10} {'黒字割合':>9}")
for h in ("1s","10s"):
    d=df.filter(pl.col("h")==h)
    a=d["align"].to_numpy(); r=d["r"].to_numpy()
    ed=np.quantile(a,np.linspace(0,1,11))
    res=[]
    for i in range(10):
        sel=(a>=ed[i])&((a<=ed[i+1]) if i==9 else (a<ed[i+1]))
        if sel.sum()<100: continue
        mr=float(r[sel].mean())
        res.append({"decile":i+1,"n":int(sel.sum()),"realized":mr,"net":mr-FEE,
                    "share_pos":float((r[sel]>0).mean())})
        print(f"{h:>5} {i+1:>12} {sel.sum():>10,} {mr:>+10.4f} {mr-FEE:>+10.4f} {(r[sel]>0).mean():>9.3f}")
    # 上位 k% だけ約定する戦略(= 不利な約定を避ける)
    for q in (0.5,0.3,0.1):
        th=np.quantile(a,1-q); sel=a>=th
        mr=float(r[sel].mean())
        out.setdefault(h,{})[f"top{int(q*100)}pct"]={"n":int(sel.sum()),"realized":mr,"net":mr-FEE}
    out.setdefault(h,{})["all"]={"n":int(r.size),"realized":float(r.mean()),"net":float(r.mean())-FEE}
    out[h]["deciles"]=res
    print()
for h in ("1s","10s"):
    print(f"[{h}] 全約定 {out[h]['all']['net']:+.4f} bp / 上位50% {out[h]['top50pct']['net']:+.4f} / "
          f"上位30% {out[h]['top30pct']['net']:+.4f} / 上位10% {out[h]['top10pct']['net']:+.4f}")
(D/"mm_signal_selection.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
