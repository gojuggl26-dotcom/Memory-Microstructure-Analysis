"""100 USD の資金が「生き残るか」を検査する。

【なぜ必要か】
  simulate は建玉を日末の中値で評価するだけで、**日中の含み損で証拠金が尽きる
  = 清算**を模擬していない。DRAM は日中レンジ中央値 7.75% / p90 14.8% なので、
  レバレッジを上げると清算が先に来る。

【厳密な最小値の求め方】
  約定と約定の間は建玉が一定なので equity = cash + pos×mid は mid の単調関数。
  よって区間内の最小 equity は、pos>0 なら mid の最小値、pos<0 なら最大値で達する。
  np.minimum.reduceat で区間ごとの極値を取れば**近似でなく厳密**に求まる。
"""
import sys; sys.path.insert(0,'scripts')
import numpy as np, polars as pl
from pathlib import Path
from capacity_v3 import load
from backtester_v3 import load_model, FEE_BP, TICK

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CAP = 100.0

def sim_path(d, qty, inv_limit):
    mp=d["mp"]; mts=mp["ts"].to_numpy()
    bb=mp["best_bid"].to_numpy(); ba=mp["best_ask"].to_numpy(); mid=mp["mid"].to_numpy()
    fl=d["fill"]; f_t=fl["t"].to_numpy(); f_px=fl["px"].to_numpy()
    f_sz=fl["sz"].to_numpy(); f_side=fl["side"].to_numpy()
    tr={}
    for s in ("B","A"):
        m=f_side==("A" if s=="B" else "B"); tr[s]=(f_t[m],f_px[m],f_sz[m])
    iv=[]
    for side,ser in (("B",bb),("A",ba)):
        i=0; n=mts.size
        while i<n:
            p=ser[i]; j=i
            while j+1<n and ser[j+1]==p: j+=1
            iv.append((mts[i], mts[j+1] if j+1<n else mts[-1], side, p, i)); i=j+1
    iv.sort()
    cash=0.0; pos=0.0; ft=[]; fc=[]; fp=[]
    for t0,t_end,side,p,i0 in iv:
        sd=1.0 if side=="B" else -1.0
        if inv_limit>0 and sd*pos>=inv_limit: continue
        px_=p
        if (ba[i0]-bb[i0])>2*TICK*1.5: px_=p+TICK if side=="B" else p-TICK
        k_=int(round(px_*1e6))*2+(side=="B"); lv=d["lvl"].get(k_); t_front=t0
        if px_==p and lv is not None:
            jj=int(np.searchsorted(lv[0],t0,side="right"))-1
            if jj>=0 and lv[1][jj]>t0: t_front=int(lv[1][jj])
        tt,tp,tz=tr[side]
        lo=np.searchsorted(tt,max(t0,t_front)); hi=np.searchsorted(tt,t_end)
        rem=qty
        for x in range(lo,hi):
            if abs(tp[x]-px_)>1e-9: continue
            take=min(rem,float(tz[x]))
            cash += -sd*px_*take - FEE_BP/1e4*px_*take
            pos += sd*take; rem-=take
            ft.append(int(tt[x])); fc.append(cash); fp.append(pos)
            if rem<=1e-9: break
    if not ft: return None
    ft=np.array(ft); fc=np.array(fc); fp=np.array(fp)
    # 各約定から次の約定までの区間で equity の厳密な最小値を取る
    idx=np.searchsorted(mts, ft, side="left")
    idx=np.clip(idx,0,len(mts)-1)
    ends=np.append(idx[1:], len(mts))
    lows=[]
    for a,b,c_,p_ in zip(idx, ends, fc, fp):
        if b<=a: continue
        seg=mid[a:b]
        lows.append(c_ + p_*(seg.min() if p_>0 else seg.max()))
    eq_min=min(lows) if lows else 0.0
    return {"pnl": fc[-1]+fp[-1]*mid[-1], "eq_min": float(eq_min),
            "max_pos_usd": float(np.abs(fp).max()*mid.mean())}

model=load_model()
days=sorted(p.stem for p in (D/"slope_spline").glob("*.npz"))
days=[x for x in days if x not in set(model[4])][:30]
PX=53.7; LEV=[1,3,5,10]
rows=[]
for i,dt in enumerate(days):
    d=load(dt,model)
    if d is None: continue
    for L in LEV:
        q=CAP*L/(5*PX)
        r=sim_path(d,q,5.0*q)
        if r: rows.append({"dt":dt,"lev":L,**r})
    if (i+1)%10==0: print(f"  {i+1}/{len(days)}",flush=True)
df=pl.DataFrame(rows); df.write_csv(D/"liq_100usd.csv")
print(f"\n=== 日中の最大含み損 vs 証拠金 100 USD ({df['dt'].n_unique()} 日) ===")
print(f"{'レバ':>4}{'最悪の含み損':>14}{'中央値':>10}{'証拠金を割った日':>18}{'維持証拠金割れ':>16}")
for L in LEV:
    s=df.filter(pl.col("lev")==L); e=s["eq_min"].to_numpy()
    # 清算: equity が −CAP を割ると証拠金消尽。維持証拠金は概ね建玉の 2〜3%
    blow=int((e <= -CAP).sum())
    mm=int((e <= -CAP*0.97).sum())
    print(f"{L:>3}×{e.min():>14.1f}{np.median(e):>10.1f}{blow:>14} / {len(e):<3}{mm:>12} / {len(e):<3}")
