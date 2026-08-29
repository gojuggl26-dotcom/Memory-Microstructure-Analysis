"""キャンセル指標の記述統計と、次窓の中値リターンとの相関。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
STEPS = {"1ms":1_000_000,"10ms":10_000_000,"100ms":100_000_000,
         "1s":1_000_000_000,"10s":10_000_000_000,"60s":60_000_000_000}
SIG = ["cancel_diff_n", "cancel_diff_vol", "n_cancel_bid", "n_cancel_ask",
       "vol_cancel_bid", "vol_cancel_ask"]

def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D/"microprice").iterdir() if p.is_dir())
    desc, daily = {}, {}
    for i, dt in enumerate(days, 1):
        t = (pl.read_parquet(D/f"microprice/dt={dt}/part-000.parquet",
                             columns=["ts","mid","is_crossed"])
             .filter(~pl.col("is_crossed")).sort("ts"))
        ts = t["ts"].to_numpy(); lm = np.log(t["mid"].to_numpy())
        if ts.size < 1000: continue
        for name, step in STEPS.items():
            f = D/f"cancel_grid/step={name}/dt={dt}/part-000.parquet"
            if not f.exists(): continue
            g = pl.read_parquet(f).sort("ts")
            T = g["ts"].to_numpy()
            def mid_at(u):
                j = np.searchsorted(ts, u, side="right") - 1
                return np.where((j >= 0) & (u <= ts[-1]), lm[np.clip(j,0,ts.size-1)], np.nan)
            y = (mid_at(T+2*step) - mid_at(T+step)) * 1e4      # 窓が閉じた後の 1 窓分
            for c in SIG:
                x = g[c].to_numpy().astype(float)
                d = desc.setdefault((name,c), {"n":0,"s":0.0,"ss":0.0})
                d["n"] += x.size; d["s"] += float(x.sum()); d["ss"] += float((x*x).sum())
                k = np.isfinite(x) & np.isfinite(y)
                if k.sum() < 500 or x[k].std() == 0 or y[k].std() == 0: continue
                daily.setdefault((name,c), []).append(float(np.corrcoef(x[k], y[k])[0,1]))
        if i % 20 == 0: print(f"{i}/{len(days)}", flush=True)

    res = {"describe": {}, "predictive_next_window": {}}
    for (name,c), d in desc.items():
        n,s,ss = d["n"], d["s"], d["ss"]
        res["describe"][f"{c}|{name}"] = {"n": n, "mean": s/n, "std": (ss/n-(s/n)**2)**0.5}
    for (name,c), v in daily.items():
        a = np.array(v)
        res["predictive_next_window"][f"{c}|{name}"] = {
            "days": a.size, "median": float(np.median(a)), "q25": float(np.quantile(a,.25)),
            "q75": float(np.quantile(a,.75)), "n_days_negative": int((a<0).sum())}
    (D/"cancel_grid_predictive.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'系列':>16} {'窓':>6} {'平均':>13} {'標準偏差':>13} {'次窓corr':>9} {'符号逆':>8}")
    for c in SIG:
        for name in STEPS:
            k = f"{c}|{name}"
            if k not in res["describe"]: continue
            d = res["describe"][k]; p = res["predictive_next_window"].get(k)
            pm = f"{p['median']:+.4f}" if p else "   n/a"
            pn = f"{p['n_days_negative']}/{p['days']}" if p else "-"
            print(f"{c:>16} {name:>6} {d['mean']:>13.3f} {d['std']:>13.3f} {pm:>9} {pn:>8}")

if __name__ == "__main__":
    main()
