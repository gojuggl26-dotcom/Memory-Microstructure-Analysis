"""99 日を 1 分刻みに割り、1 分ごとの相対 MSE を出す。

フォールド構造は variance_blocks.py と同じ拡大窓:
  3 日ブロック b の評価には**ブロック 0..b−1 だけで学習**した Lasso を使う。
  その予測をブロック b の中の**各 1 分**に分けて、分単位の相対 MSE を計算する。

    相対 MSE(分 m) = Σ_{t∈m}(y_t − ŷ_t)² / Σ_{t∈m} y_t²

  分母はその分自身の二乗和なので、**分ごとの尺度は自動的に消える**。
  つまり「分散が大きい分ほど相対 MSE が悪い/良い」という関係が出れば、
  それは尺度ではなく中身(関係の強さ)の違いである。

1 分 ≒ 60 行しかないので個々の値は非常にノイジー。分布と条件付き平均で読む。
resync アーティファクト(variance_report.md §2)の行数も分ごとに数え、
汚染の有無で比較できるようにする。

★時間契約: 学習は評価ブロックより前のみ。前処理も訓練側だけから推定。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
PANEL = D / "panel_1s"
N_BLOCK = 33
ALPHA = 0.0161
WINSOR = 0.001
MIN_NS = 60_000_000_000


def main() -> None:
    days = sorted(p.name.split("=")[1] for p in PANEL.iterdir() if p.is_dir())
    frames = []
    for k, dt in enumerate(days):
        f = pl.read_parquet(PANEL / f"dt={dt}" / "part-000.parquet").with_columns(
            pl.lit(k).alias("day"))
        fl = PANEL / f"dt={dt}" / "resync_flag.parquet"
        f = (f.join(pl.read_parquet(fl), on="ts", how="left")
              .with_columns(pl.col("is_resync_contaminated").fill_null(False))
             if fl.exists() else f.with_columns(pl.lit(False).alias("is_resync_contaminated")))
        frames.append(f)
    df = pl.concat(frames)
    feats = [c for c in df.columns
             if c not in ("ts", "y", "day", "is_resync_contaminated")]
    day = df["day"].to_numpy()
    ts = df["ts"].to_numpy()
    bad = df["is_resync_contaminated"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float64)
    n_days = int(day.max()) + 1
    blk = np.minimum((day * N_BLOCK) // n_days, N_BLOCK - 1)
    minute = ts // MIN_NS
    print(f"rows={y.size:,} 分の数={np.unique(minute).size:,}")

    sig = np.ones(n_days)
    for d0 in range(n_days):
        m = day == d0
        if m.sum() > 100:
            sig[d0] = max(float(y[m].std()), 1e-6)
    yz = y / np.concatenate([[sig[0]], sig[:-1]])[day]

    sub = np.zeros(y.size, dtype=bool)
    sub[::5] = True
    out = []
    for b in range(5, N_BLOCK):
        tr = (blk < b) & sub
        te = blk == b
        if tr.sum() < 50_000 or te.sum() < 5_000:
            continue
        lo = np.quantile(X[tr], WINSOR, axis=0).astype(np.float32)
        hi = np.quantile(X[tr], 1 - WINSOR, axis=0).astype(np.float32)
        Ztr = np.clip(X[tr], lo, hi)
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd[sd == 0] = 1.0
        Ztr = ((Ztr - mu) / sd).astype(np.float64)
        Zte = ((np.clip(X[te], lo, hi) - mu) / sd).astype(np.float64)
        mi = minute[te]
        uniq, inv = np.unique(mi, return_inverse=True)
        rec = {"minute": uniq, "block": np.full(uniq.size, b),
               "n": np.bincount(inv, minlength=uniq.size),
               "n_bad": np.bincount(inv, weights=bad[te].astype(float),
                                    minlength=uniq.size)}
        for tag, target in (("raw", y - y[tr].mean()), ("volnorm", yz - yz[tr].mean())):
            mdl = Lasso(alpha=ALPHA, max_iter=5000, tol=1e-5).fit(Ztr, target[tr])
            p = mdl.predict(Zte)
            t_ = target[te]
            sse = np.bincount(inv, weights=(t_ - p) ** 2, minlength=uniq.size)
            sst = np.bincount(inv, weights=t_ ** 2, minlength=uniq.size)
            sy = np.bincount(inv, weights=t_, minlength=uniq.size)
            sp = np.bincount(inv, weights=p, minlength=uniq.size)
            spp = np.bincount(inv, weights=p ** 2, minlength=uniq.size)
            syp = np.bincount(inv, weights=t_ * p, minlength=uniq.size)
            n = rec["n"]
            rec[f"relmse_{tag}"] = np.where(sst > 0, sse / np.maximum(sst, 1e-300), np.nan)
            rec[f"var_{tag}"] = sst / np.maximum(n, 1) - (sy / np.maximum(n, 1)) ** 2
            vp = spp / np.maximum(n, 1) - (sp / np.maximum(n, 1)) ** 2
            cv = syp / np.maximum(n, 1) - (sy / np.maximum(n, 1)) * (sp / np.maximum(n, 1))
            rec[f"corr_{tag}"] = np.where((vp > 0) & (rec[f"var_{tag}"] > 0),
                                          cv / np.sqrt(np.maximum(vp * rec[f"var_{tag}"], 1e-300)),
                                          np.nan)
        out.append(pl.DataFrame(rec))
        print(f"  block {b}: 分 {uniq.size:,}", flush=True)

    m = pl.concat(out)
    m.write_parquet(D / "minute_relmse.parquet", compression="zstd")

    # ---- 集計 ----
    def q(v, p):
        return float(np.nanquantile(v, p))
    res = {}
    for clean in (False, True):
        d = m.filter(pl.col("n_bad") == 0) if clean else m
        d = d.filter(pl.col("n") >= 30)
        for tag in ("raw", "volnorm"):
            r = d[f"relmse_{tag}"].to_numpy()
            v = d[f"var_{tag}"].to_numpy()
            ok = np.isfinite(r) & np.isfinite(v) & (v > 0)
            r, v = r[ok], v[ok]
            # 分散十分位ごとの相対 MSE
            edges = np.nanquantile(v, np.linspace(0, 1, 11))
            dec = []
            for i in range(10):
                s = (v >= edges[i]) & (v <= edges[i + 1] if i == 9 else v < edges[i + 1])
                if s.sum() < 50:
                    continue
                dec.append({"decile": i + 1, "n": int(s.sum()),
                            "var_median": float(np.median(v[s])),
                            "relmse_median": float(np.median(r[s])),
                            "share_lt_1": float((r[s] < 1).mean())})
            key = f"{'clean' if clean else 'all'}_{tag}"
            res[key] = {
                "n_minutes": int(r.size),
                "relmse_median": float(np.median(r)),
                "relmse_mean": float(np.mean(r)),
                "relmse_q10": q(r, .1), "relmse_q90": q(r, .9),
                "share_lt_1": float((r < 1).mean()),
                "pooled_relmse": float(np.average(r, weights=v)),
                "corr_logvar_vs_relmse": float(np.corrcoef(np.log(v), r)[0, 1]),
                "by_variance_decile": dec,
            }
    res["minutes_total"] = int(m.height)
    res["minutes_with_artifact"] = int((m["n_bad"] > 0).sum())
    (D / "minute_relmse.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    for key in ("all_raw", "clean_raw", "clean_volnorm"):
        r = res[key]
        print(f"\n[{key}] 分 {r['n_minutes']:,}  中央値 {r['relmse_median']:.4f}  "
              f"平均 {r['relmse_mean']:.4f}  <1 の割合 {r['share_lt_1']:.3f}  "
              f"log分散との相関 {r['corr_logvar_vs_relmse']:+.4f}")
        for x in r["by_variance_decile"]:
            print(f"    十分位{x['decile']:>2} var={x['var_median']:>9.4f} "
                  f"relMSE中央値={x['relmse_median']:.4f} <1 の割合={x['share_lt_1']:.3f}")


if __name__ == "__main__":
    main()
