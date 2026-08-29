"""窓幅ごとの OI について、飽和度と予測力を測る。

x = OI(窓 w) は窓が閉じた時点 T+step でしか判らないので、
予測力の検定は **窓が閉じた後の 1 窓分のリターン** に対して行う(先読みなし):
    y_fwd = log mid(T+2·step) − log mid(T+step)
参考に同時点の y_now = log mid(T+step) − log mid(T) も出す。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
STEPS = {"1ms":1_000_000,"10ms":10_000_000,"100ms":100_000_000,
         "1s":1_000_000_000,"10s":10_000_000_000,"60s":60_000_000_000}

def main() -> None:
    days = sorted(p.name.split("=")[1] for p in (D/"microprice").iterdir() if p.is_dir())
    acc = {k: {"now": [], "fwd": [], "sat": [], "n": 0} for k in STEPS}
    for i, dt in enumerate(days, 1):
        t = (pl.read_parquet(D/f"microprice/dt={dt}/part-000.parquet",
                             columns=["ts","mid","is_crossed"])
             .filter(~pl.col("is_crossed")).sort("ts"))
        ts = t["ts"].to_numpy(); lm = np.log(t["mid"].to_numpy())
        if ts.size < 1000: continue
        for name, step in STEPS.items():
            f = D/f"oi_grid/step={name}/dt={dt}/part-000.parquet"
            if not f.exists(): continue
            g = pl.read_parquet(f).sort("ts")
            T = g["ts"].to_numpy(); x = g["order_imbalance"].to_numpy()
            def mid_at(u):
                j = np.searchsorted(ts, u, side="right") - 1
                return np.where((j >= 0) & (u <= ts[-1]), lm[np.clip(j, 0, ts.size-1)], np.nan)
            m0, m1, m2 = mid_at(T), mid_at(T+step), mid_at(T+2*step)
            y_now = (m1-m0)*1e4; y_fwd = (m2-m1)*1e4
            acc[name]["n"] += g.height
            acc[name]["sat"].append(float(np.mean(np.abs(x) > 0.999)))
            for key, y in (("now", y_now), ("fwd", y_fwd)):
                k = np.isfinite(x) & np.isfinite(y)
                if k.sum() < 500 or x[k].std() == 0 or y[k].std() == 0: continue
                acc[name][key].append(float(np.corrcoef(x[k], y[k])[0,1]))
        if i % 20 == 0: print(f"{i}/{len(days)}", flush=True)

    res = {}
    for name in STEPS:
        a = acc[name]
        def q(v):
            v = np.array(v)
            return {"days": v.size, "median": float(np.median(v)), "q25": float(np.quantile(v,.25)),
                    "q75": float(np.quantile(v,.75)), "n_days_negative": int((v<0).sum())} if v.size else None
        res[name] = {"windows": a["n"], "saturated_share": float(np.mean(a["sat"])),
                     "corr_same_window": q(a["now"]), "corr_next_window": q(a["fwd"])}
    (D/"oi_grid_predictive.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'窓':>6} {'窓数':>10} {'飽和率':>7} {'同窓 corr':>10} {'次窓 corr':>10} {'次窓で符号逆の日':>10}")
    for k, v in res.items():
        cn, cf = v["corr_same_window"], v["corr_next_window"]
        print(f"{k:>6} {v['windows']:>10,} {v['saturated_share']:>7.3f} "
              f"{cn['median']:>10.4f} {cf['median']:>10.4f} {cf['n_days_negative']:>10d}")

if __name__ == "__main__":
    main()
