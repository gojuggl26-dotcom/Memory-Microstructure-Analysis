"""特徴量ライブラリ(build_featlib.py の出力)を棚卸しし、予測力を測る。

    uv run python scripts/analyze_featlib.py --coin xyz:INTC [--ref xyz:MU]

出力
----
  data/featlib_stats_<tag>.csv    特徴量ごとの分類・有限率・分位
  data/featlib_pred_<tag>.csv     特徴量ごとの前向き / 後ろ向き / 帰無対照の相関
  data/featlib_dup_<tag>.csv      |r| > 0.99 の重複ペア
  data/featlib_fam_<tag>.csv      分類ごとの要約

時間契約
--------
説明変数 `x` は格子点 `T` **までの**情報だけで作ってある(build_featlib.py)。
目的変数は

    fwd_mid_h  = log mid(T+h)  − log mid(T)      … 期間 (T, T+h]
    fwd_micro_h= log micro(T+h)− log micro(T)    … 同上

で、`x` の確定時刻 `T` ≤ `y` の期間の開始時刻 `T` を満たす。
比較のために **後ろ向き**リターン `bwd_h = log mid(T) − log mid(T−h)` も作るが、
これは同時点の関係を測るためだけに使い、予測力とは別の欄に書く。

★標準誤差は日単位でクラスタする
--------------------------------
1 秒格子の特徴量は強く自己相関しており、行を独立とみなした標準誤差は
最大 15.8 倍過小になる(`mu_burstiness_report.md`)。ここでは
**1 日を 1 観測**として日ごとの相関を出し、日をまたいだ t 検定で報告する。

★帰無対照は「日の入れ替え」
--------------------------
x を日 d、y を日 d+1 の**同じ秒**から取る。日内の周期(米国市場の開場・閉場)を
両方に残したまま、実際の対応だけを壊すので、巡回シフトより強い帰無になる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chartstyle import D  # noqa: E402

# ---- 12 分類。上から順に当てはめる(先に当たったものが勝つ) ----------------
FAMILY = [
    ("12 攻撃的な約定フロー", ("trade_", "signed_vol", "signed_cnt", "aggr_", "size_wgt_imb")),
    ("10 取消", ("cancel_", "bbo_cancel_rate", "deep_cancel_rate", "t_since_cancel")),
    ("11 指値の到着", ("add_intensity", "bbo_add_int", "deep_add_int", "repl_intensity",
                  "arrival_")),
    ("9 指値フロー", ("add_", "exec_", "net_lof", "repl_", "depl_")),
    ("8 OFI", ("ofi_", "d_ofi")),
    ("7 板の弾力性", ("elast_", "local_elast_", "deep_elast_", "marg_elast_")),
    ("6 板の隙間", ("gap12_", "max_gap_", "mean_gap_", "empty_lv_", "wall_dist_",
                "cliff_", "gap_asym", "med_gap", "cum_gap", "wgt_gap")),
    ("5 板の形", ("slope_", "local_slope_", "deep_slope_", "curv_", "shape_asym",
               "hhi_", "gini_", "entropy_", "liq_conc")),
    ("4 マイクロプライス", ("micro", "delta", "obi_x_spread", "d_micro")),
    # ★build_featlib.py の §2 は板の厚みそのものではなく「偏り」なので、
    #   depth_diff / depth_ratio / log_depth_ratio / d_obi もここに入れる
    ("2 板の偏り OBI", ("obi", "depth_diff", "depth_ratio", "log_depth_ratio",
                   "d_obi")),
    ("3 板の厚み", ("qb", "qa", "cum_b", "cum_a", "depth", "near_depth", "deep_depth",
                "near_deep_ratio", "dollar_depth", "liq_density", "d_depth",
                "depl_rate", "repl_rate")),
    ("1 価格と最良気配", ("best_", "mid", "log_mid", "spread", "tick", "d_bid", "d_ask",
                   "d_mid", "d_spread", "t_since_", "bbo_age")),
]


def family_of(c: str) -> str:
    for name, pre in FAMILY:
        for p in pre:
            if c == p or c.startswith(p):
                return name
    return "0 その他"


def load_day(fp: Path, cols: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(fp, columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = D / f"featlib_{tag}"
    files = sorted(src.glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"{src} が空")
    head = load_day(files[0])
    allc = [c for c in head.columns if c not in ("dt", "sec")]
    feats = [c for c in allc if not c.startswith("fwd_")]
    tgts = [c for c in allc if c.startswith("fwd_")]
    print(f"{a.coin}: {len(files)} 日 / 特徴量 {len(feats)} / 目的変数 {len(tgts)}")

    nf = len(feats)
    # ---- 1 パス目: 分位と有限率、そして相関の材料 --------------------------
    cnt = np.zeros(nf)
    s1 = np.zeros(nf)
    s2 = np.zeros(nf)
    nz = np.zeros(nf)
    qs = {c: [] for c in feats}                 # 日ごとの分位(あとで中央を取る)
    # 目的変数(前向き)と後ろ向きに対する日ごとの相関
    TG = ["fwd_mid_1s", "fwd_micro_1s", "fwd_mid_10s", "fwd_micro_10s",
          "fwd_mid_60s", "fwd_micro_60s"]
    TG = [t for t in TG if t in tgts]
    BW = ["bwd_mid_10s", "bwd_mid_60s"]
    PL = ["pl_micro_10s"]                       # 日入れ替えの帰無対照
    RCOLS = TG + BW + PL
    rday = {k: np.full((len(files), nf), np.nan) for k in RCOLS}
    # 冗長性のための相互相関(全列有限の行だけで積む)
    core = [c for c in feats if c not in
            ("ofi_per_event", "add_cancel_ratio", "cancel_add_ratio",
             "repl_depl_ratio", "bbo_cancel_rate", "deep_cancel_rate",
             "near_deep_ratio", "depth_ratio")]
    ic = [feats.index(c) for c in core]
    nc = len(core)
    Sx = np.zeros(nc)
    Sxx = np.zeros((nc, nc))
    Ncore = 0
    prev = None                                 # 帰無対照用に前日を持ち越す

    for di, fp in enumerate(files):
        d = load_day(fp)
        X = d.select(feats).to_numpy().astype(np.float64)
        n = X.shape[0]
        fin = np.isfinite(X)
        cnt += fin.sum(0)
        Xz = np.where(fin, X, 0.0)
        s1 += Xz.sum(0)
        s2 += (Xz * Xz).sum(0)
        nz += ((np.abs(Xz) < 1e-12) & fin).sum(0)
        for j, c in enumerate(feats):
            v = X[fin[:, j], j]
            if v.size > 100:
                qs[c].append(np.percentile(v, [5, 50, 95]))

        mid = d["mid"].to_numpy().astype(np.float64)
        lm = np.log(np.where(mid > 0, mid, np.nan))
        y = {}
        for t in TG:
            y[t] = d[t].to_numpy().astype(np.float64)
        for h, k in ((2, "bwd_mid_10s"), (12, "bwd_mid_60s")):   # 行間隔 5 秒
            b = np.full(n, np.nan)
            b[h:] = (lm[h:] - lm[:-h]) * 1e4
            y[k] = b
        if prev is not None and "fwd_micro_10s" in d.columns:
            pv = prev
            m = min(n, pv.size)
            z = np.full(n, np.nan)
            z[:m] = pv[:m]
            y["pl_micro_10s"] = z
        else:
            y["pl_micro_10s"] = np.full(n, np.nan)
        prev = d["fwd_micro_10s"].to_numpy().astype(np.float64) if \
            "fwd_micro_10s" in d.columns else None

        for k in RCOLS:
            yk = y[k]
            fy = np.isfinite(yk)
            if fy.sum() < 500:
                continue
            yc = np.where(fy, yk, 0.0)
            sy = yc[fy].sum()
            syy = (yc[fy] ** 2).sum()
            for j in range(nf):
                m = fin[:, j] & fy
                nn = m.sum()
                if nn < 500:
                    continue
                xj = X[:, j]
                sx = xj[m].sum()
                sxx = (xj[m] ** 2).sum()
                sxy = (xj[m] * yk[m]).sum()
                sy_ = yk[m].sum()
                syy_ = (yk[m] ** 2).sum()
                vx = sxx - sx * sx / nn
                vy = syy_ - sy_ * sy_ / nn
                if vx > 1e-18 and vy > 1e-18:
                    rday[k][di, j] = (sxy - sx * sy_ / nn) / np.sqrt(vx * vy)
            del sy, syy

        ok = np.isfinite(X[:, ic]).all(1)
        if ok.sum() > 200:
            C = X[np.ix_(ok, ic)]
            Sx += C.sum(0)
            Sxx += C.T @ C
            Ncore += int(ok.sum())
        if (di + 1) % 10 == 0 or di == 0:
            print(f"  [{di+1}/{len(files)}] {fp.stem}", flush=True)

    # ---- 棚卸し表 ----------------------------------------------------------
    q = np.array([np.median(np.array(qs[c]), axis=0) if qs[c] else [np.nan] * 3
                  for c in feats])
    tot = len(files) * head.height
    st = pl.DataFrame({
        "feature": feats,
        "family": [family_of(c) for c in feats],
        "finite_rate": cnt / tot,
        "zero_rate": np.where(cnt > 0, nz / np.maximum(cnt, 1), np.nan),
        "p05": q[:, 0], "p50": q[:, 1], "p95": q[:, 2],
        "std": np.sqrt(np.maximum(s2 / np.maximum(cnt, 1)
                                  - (s1 / np.maximum(cnt, 1)) ** 2, 0)),
    })
    st.write_csv(D / f"featlib_stats_{tag}.csv")

    # ---- 予測力(日クラスタ) ---------------------------------------------
    nd = len(files)
    half = nd // 2
    rows = []
    for j, c in enumerate(feats):
        row = {"feature": c, "family": family_of(c)}
        for k in RCOLS:
            v = rday[k][:, j]
            v = v[np.isfinite(v)]
            row[k] = float(v.mean()) if v.size else np.nan
            if v.size > 2:
                row[k + "_t"] = float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size)))
                row[k + "_nd"] = int(v.size)
            else:
                row[k + "_t"] = np.nan
                row[k + "_nd"] = 0
        a1 = rday["fwd_micro_10s"][:half, j]
        a2 = rday["fwd_micro_10s"][half:, j]
        row["is_1st"] = float(np.nanmean(a1)) if np.isfinite(a1).any() else np.nan
        row["oos_2nd"] = float(np.nanmean(a2)) if np.isfinite(a2).any() else np.nan
        rows.append(row)
    pr = pl.DataFrame(rows)
    pr.write_csv(D / f"featlib_pred_{tag}.csv")

    # ---- 冗長性 -----------------------------------------------------------
    mu = Sx / max(Ncore, 1)
    Cov = Sxx / max(Ncore, 1) - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(Cov), 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = Cov / np.outer(sd, sd)
    np.save(D / f"featlib_corr_{tag}.npy", R.astype(np.float32))
    with open(D / f"featlib_corr_{tag}.cols", "w", encoding="utf-8") as f:
        f.write("\n".join(core))
    iu = np.triu_indices(nc, 1)
    rv = np.abs(R[iu])
    sel = np.where(rv > 0.99)[0]
    dup = pl.DataFrame({"a": [core[iu[0][i]] for i in sel],
                        "b": [core[iu[1][i]] for i in sel],
                        "r": [float(R[iu[0][i], iu[1][i]]) for i in sel]}
                       ).sort("r", descending=True)
    dup.write_csv(D / f"featlib_dup_{tag}.csv")

    # 単連結クラスタ (|r| > 0.95) で「実質いくつあるか」を数える
    par = list(range(nc))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i in np.where(rv > 0.95)[0]:
        ra_, rb_ = find(iu[0][i]), find(iu[1][i])
        if ra_ != rb_:
            par[ra_] = rb_
    ncl = len({find(i) for i in range(nc)})

    fam = (st.join(pr.select("feature", "fwd_micro_10s", "bwd_mid_10s"), on="feature")
           .group_by("family").agg(
               pl.len().alias("n"),
               pl.col("finite_rate").median().alias("finite_p50"),
               pl.col("fwd_micro_10s").abs().max().alias("max_abs_fwd"),
               pl.col("bwd_mid_10s").abs().max().alias("max_abs_bwd"))
           .sort("family"))
    fam.write_csv(D / f"featlib_fam_{tag}.csv")

    print(f"\n全列有限の行 {Ncore:,} / {tot:,} ({Ncore/tot*100:.1f}%)")
    print(f"|r|>0.99 の重複ペア {dup.height:,} 組 / 単連結 |r|>0.95 で {nc} → {ncl} 群")
    print(fam)
    top = pr.filter(pl.col("fwd_micro_10s").is_not_null()).with_columns(
        ab=pl.col("fwd_micro_10s").abs()).sort("ab", descending=True).head(15)
    print(top.select("feature", "family", "fwd_micro_10s", "fwd_micro_10s_t",
                     "bwd_mid_10s", "pl_micro_10s", "is_1st", "oos_2nd"))


if __name__ == "__main__":
    main()
