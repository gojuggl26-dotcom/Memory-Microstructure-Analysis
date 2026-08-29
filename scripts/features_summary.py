"""4 系列の記述統計と、次イベント / 1 秒先の中値変化に対する予測力を出す。

監査(regression_report.md §9)の結論に従い、
  - クロス状態は除外
  - プールせず日次で推定し、中央値と負の日数で読む
  - 相関(尺度不変)を主指標にし、回帰係数は補助
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, polars as pl

F = Path("C:/Users/ii562/Downloads/Memory/data/DRAM/features")
OUT = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
COLS = ["bid_ask_spread_bp", "order_imbalance", "depth_one_level", "voi",
        "voi_raw_diff", "ofi_cont"]
SIG = ["order_imbalance", "depth_one_level", "voi", "voi_raw_diff", "ofi_cont"]


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in F.iterdir() if p.is_dir())
    desc = {c: {"n": 0, "s": 0.0, "ss": 0.0, "nz": 0} for c in COLS}
    daily = {(c, h): [] for c in SIG for h in ("ev1", "1s")}
    sign_flip = {"n": 0, "same": 0, "px_chg": 0}
    corr_pairs = {}

    for i, dt in enumerate(days, 1):
        d = pl.read_parquet(F / f"dt={dt}" / "part-000.parquet").filter(~pl.col("is_crossed"))
        if d.height < 5000:
            continue
        ts = d["ts"].to_numpy()
        mid = (d["best_bid"].to_numpy() + d["best_ask"].to_numpy()) / 2
        lm = np.log(mid)
        y_ev = np.full(ts.size, np.nan)
        y_ev[:-1] = (lm[1:] - lm[:-1]) * 1e4                      # 次イベント
        tgt = ts + 1_000_000_000
        j = np.searchsorted(ts, tgt, side="right") - 1
        y_1s = np.where(tgt <= ts[-1], (lm[j] - lm) * 1e4, np.nan)  # 1 秒先

        arr = {c: d[c].to_numpy() for c in COLS}
        for c in COLS:
            v = arr[c][np.isfinite(arr[c])]
            desc[c]["n"] += v.size
            desc[c]["s"] += float(v.sum())
            desc[c]["ss"] += float((v * v).sum())
            desc[c]["nz"] += int((v != 0).sum())

        v, o = arr["voi_raw_diff"], arr["voi"]
        m = np.isfinite(v) & np.isfinite(o)
        sign_flip["n"] += int(m.sum())
        sign_flip["same"] += int((np.sign(v[m]) == np.sign(o[m])).sum())
        bb, ba = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
        sign_flip["px_chg"] += int((np.diff(bb) != 0).sum() + (np.diff(ba) != 0).sum() > 0
                                   and ((np.diff(bb) != 0) | (np.diff(ba) != 0)).sum())

        for c in SIG:
            x = arr[c]
            for h, y in (("ev1", y_ev), ("1s", y_1s)):
                k = np.isfinite(x) & np.isfinite(y)
                if k.sum() < 1000:
                    continue
                xx, yy = x[k], y[k]
                if xx.std() == 0 or yy.std() == 0:
                    continue
                daily[(c, h)].append(float(np.corrcoef(xx, yy)[0, 1]))
        # 系列間の相関
        k = np.ones(ts.size, dtype=bool)
        for a in SIG:
            for b in SIG:
                if a >= b:
                    continue
                kk = np.isfinite(arr[a]) & np.isfinite(arr[b])
                if kk.sum() < 1000:
                    continue
                c_ = float(np.corrcoef(arr[a][kk], arr[b][kk])[0, 1])
                corr_pairs.setdefault(f"{a}|{b}", []).append(c_)
        if i % 20 == 0:
            print(f"{i}/{len(days)}", flush=True)

    res = {"describe": {}, "predictive_daily_corr": {}, "cross_corr_median": {},
           "voi_vs_ofi_sign_agreement": sign_flip["same"] / max(1, sign_flip["n"])}
    for c in COLS:
        n, s, ss = desc[c]["n"], desc[c]["s"], desc[c]["ss"]
        res["describe"][c] = {"n": n, "mean": s / n, "std": (ss / n - (s / n) ** 2) ** 0.5,
                              "nonzero_share": desc[c]["nz"] / n}
    for (c, h), v in daily.items():
        a = np.array(v)
        res["predictive_daily_corr"][f"{c}|{h}"] = {
            "days": a.size, "median": float(np.median(a)), "mean": float(a.mean()),
            "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)),
            "n_days_negative": int((a < 0).sum())}
    for k2, v in corr_pairs.items():
        res["cross_corr_median"][k2] = float(np.median(np.array(v)))
    (OUT / "features_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
