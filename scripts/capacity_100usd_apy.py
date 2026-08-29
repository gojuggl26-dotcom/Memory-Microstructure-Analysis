"""年次成果の**予測分布**。前版の誤りを修正する。

【前版の誤り】
  ブロックブートストラップで 365 日を復元抽出し合計したが、これは
  「真の日次分布が標本そのものだったら 1 年はどれだけ振れるか」であって、
  **標本 30 日から推定した平均自体の誤差**を含まない。
  結果 p1〜p99 が ±7% という非現実的に狭い区間になった。

【修正】二重ブートストラップ
  ① 30 日をブロック復元抽出して「あり得た母集団」を作る(母数の不確実性)
  ② そこから 365 日を抽出して合計する(経路の不確実性)
  分散は Var = 365σ² + 365²·s²/n となり、後者(母数側)が支配的になる。
"""
import numpy as np, polars as pl, json
from pathlib import Path
from scipy import stats
D=Path("C:/Users/ii562/Downloads/Memory/data/DRAM"); CAP=100.0; BLK=2; NB=20000; YR=365
rng=np.random.default_rng(20260817)
c=pl.read_csv(D/"capacity_100usd.csv")

def blocks(u,n_out,size):
    n=len(u); nb=int(np.ceil(n_out/BLK))
    st=rng.integers(0,n,size=(size,nb))
    idx=(st[:,:,None]+np.arange(BLK)[None,None,:])%n
    return u[idx.reshape(size,-1)[:,:n_out]]

print(f"標本 30 日 / 二重ブートストラップ {NB:,} 回 / 資金 {CAP:.0f} USD")
print(f"{'レバ':>4}{'E[PnL]/日':>11}{'日次sd':>9}{'平均のSE':>10}"
      f"{'E[PnL]/年':>11}{'年sd(旧)':>11}{'年sd(修正)':>12}")
res={}
for L in sorted(c["lev"].unique().to_list()):
    u=c.filter(pl.col("lev")==L)["pnl_usd"].to_numpy(); n=len(u)
    pop=blocks(u,n,NB)                       # ① 母集団の再抽出
    ann=np.empty(NB)
    for i in range(0,NB,2000):               # ② そこから 1 年
        sl=slice(i,min(i+2000,NB))
        p=pop[sl]; k=p.shape[0]
        nb=int(np.ceil(YR/BLK))
        st=rng.integers(0,n,size=(k,nb))
        ix=(st[:,:,None]+np.arange(BLK)[None,None,:])%n
        ann[sl]=np.take_along_axis(p, ix.reshape(k,-1)[:,:YR], axis=1).sum(axis=1)
    res[L]=(u,ann)
    se=u.std(ddof=1)/np.sqrt(n)
    old=u.std(ddof=1)*np.sqrt(YR)
    new=np.sqrt(YR*u.std(ddof=1)**2 + YR**2*se**2)
    print(f"{L:>3}×{u.mean():>11.3f}{u.std(ddof=1):>9.2f}{se:>10.3f}"
          f"{u.mean()*YR:>11.0f}{old:>11.0f}{new:>12.0f}")

print(f"\n=== 年次 APY の予測分布(母数誤差込み・単利)===")
print(f"{'レバ':>4}{'p1':>9}{'p10':>9}{'中央値':>9}{'p70':>9}{'p80':>9}{'p90':>9}{'p99':>9}{'負':>7}")
out={}
for L,(u,a) in res.items():
    q=np.percentile(a/CAP*100,[1,10,50,70,80,90,99])
    print(f"{L:>3}×"+"".join(f"{v:>8.0f}%" for v in q)+f"{(a<0).mean()*100:>6.1f}%")
    out[L]={"apy_mean":float(u.mean()*YR/CAP*100),
            "pct":{f"p{k}":float(np.percentile(a/CAP*100,k)) for k in (1,10,50,70,80,90,99)},
            "prob_loss":float((a<0).mean()),
            "day_mean":float(u.mean()),"day_sd":float(u.std(ddof=1)),
            "p_t":float(stats.ttest_1samp(u,0).pvalue)}
(D/"capacity_100usd.json").write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"\n参考: 旧版(母数誤差なし)の 1× は p1=998% p99=1349% だった → 修正版と比較すること")
