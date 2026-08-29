"""メイカー約定の損益分解: 粗いスプレッド獲得 − 逆選択 − 手数料。

メイカーが価格 p で約定したとき(方向 s = +1 買い / −1 売り)、
    粗い獲得       g   = s × (mid(t) − p)          … 約定時点の中値との差
    逆選択         a(h)= −s × (mid(t+h) − mid(t))  … 約定後に不利へ動いた分
    実現損益       r(h)= g − a(h) = s × (mid(t+h) − p)
すべて mid の bp で測る。ここから手数料を引いたものが正味。
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
DAYS = ["2026-07-20","2026-07-21","2026-07-22","2026-07-23","2026-07-24",
        "2026-07-27","2026-07-28","2026-07-29","2026-08-03","2026-08-05"]
HS = {"1s":10**9, "10s":10**10, "60s":6*10**10, "300s":3*10**11}
acc = {h: [] for h in HS}; g_all=[]; sp_all=[]
for dt in DAYS:
    fs = sorted(glob.glob(f'data/fills_v99/dt={dt}/**/*.parquet', recursive=True))
    if not fs: continue
    f = pl.concat([pl.read_parquet(x, columns=["ts","px","sz","side","crossed","tid"])
                   for x in fs], how="diagonal_relaxed")
    f = (f.filter(~pl.col("crossed")).unique(subset=["tid"], keep="first")   # メイカー行
          .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort("t"))
    m = (pl.read_parquet(D/f"microprice/dt={dt}/part-000.parquet",
                         columns=["ts","mid","spread_bp","is_crossed"])
         .filter(~pl.col("is_crossed")).sort("ts"))
    mts=m["ts"].to_numpy(); mv=m["mid"].to_numpy(); sp=m["spread_bp"].to_numpy()
    t=f["t"].to_numpy(); p=f["px"].to_numpy()
    s=np.where(f["side"].to_numpy()=="B", 1.0, -1.0)     # メイカーが買った=+1
    j0=np.searchsorted(mts,t,side="right")-1
    ok=j0>=0
    m0=np.where(ok, mv[np.clip(j0,0,mv.size-1)], np.nan)
    g=s*(m0-p)/m0*1e4                                    # 粗い獲得(bp)
    g_all.append(g[np.isfinite(g)]); sp_all.append(sp[np.clip(j0,0,sp.size-1)][ok])
    for h,ns in HS.items():
        j1=np.searchsorted(mts,t+ns,side="right")-1
        ok2=ok&(t+ns<=mts[-1])&(j1>=0)
        m1=np.where(ok2, mv[np.clip(j1,0,mv.size-1)], np.nan)
        r=s*(m1-p)/m0*1e4                                # 実現損益(bp)
        acc[h].append(r[np.isfinite(r)])
G=np.concatenate(g_all); SP=np.concatenate(sp_all)
out={"n_maker_fills":int(G.size),
     "gross_capture_bp":{"mean":float(G.mean()),"median":float(np.median(G))},
     "spread_bp_median":float(np.median(SP)),
     "maker_fee_bp":0.088,"taker_fee_bp":0.846}
print(f"メイカー約定 {G.size:,} 件(10 日)")
print(f"粗いスプレッド獲得: 平均 {G.mean():+.4f} bp / 中央値 {np.median(G):+.4f} bp")
print(f"参考: 約定時のスプレッド中央値 {np.median(SP):.4f} bp(半分 = {np.median(SP)/2:.4f} bp)")
print(f"\n{'地平':>6} {'実現損益(平均)':>14} {'逆選択':>10} {'手数料後':>10} {'黒字の割合':>10}")
for h in HS:
    R=np.concatenate(acc[h]); adv=G[:R.size].mean()-R.mean() if R.size else float('nan')
    net=R.mean()-0.088
    out[h]={"realized_bp_mean":float(R.mean()),"realized_bp_median":float(np.median(R)),
            "adverse_bp":float(adv),"net_after_fee_bp":float(net),
            "share_positive":float((R>0).mean()),"n":int(R.size)}
    print(f"{h:>6} {R.mean():>+14.4f} {adv:>+10.4f} {net:>+10.4f} {(R>0).mean():>10.3f}")
(D/"mm_pnl_decomp.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
