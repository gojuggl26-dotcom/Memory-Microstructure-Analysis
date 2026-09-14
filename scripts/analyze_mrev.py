"""1 分足の平均回帰 — 自己相関・回帰速度(半減期)・分散比。

    uv run python scripts/analyze_mrev.py

入力: data/mrev_bars_<coin>.parquet(`build_mrev.py`)
出力: data/mrev_acf.csv     銘柄 × 価格定義 × ラグ の自己相関
      data/mrev_speed.csv   銘柄 × 基準価格 の AR(1) と半減期
      data/mrev_vr.csv      銘柄 × ホライズン の分散比
      data/mrev_summary.csv 銘柄ごとの要約

## 何への平均回帰かを先に決める

$`D_t=P_t-F_t`$ の $`F_t`$ を 4 通り作る。$`P_t`$ は常に **mid**。

| 記号 | $`F_t`$ | 意味 |
|---|---|---|
| `micro` | microprice | 板の偏りで調整した LOB 上の短期均衡 |
| `ewma` | mid の指数移動平均(半減期 7 分) | 時系列的均衡。SMA20 と平均ラグをそろえた |
| **`sma20`** | **直近 20 分の mid の単純移動平均** | **本レポートの主基準(指示)** |
| `model` | OBI・符号つき出来高・直近リターンから予測した mid | 実用的な fair value |

`ewma` の半減期 7 分は、SMA20 の平均ラグ $`(20-1)/2=9.5`$ 分に
EWMA の平均ラグ $`1/\\alpha-1`$ を合わせて選んだ。窓の長さを変えずに
「単純平均か指数平均か」だけを比べるためである。

## ★ ランダムウォーク対照(いちばん重要)

**$`D_t=P_t-\\mathrm{SMA}_{20}(P)_t`$ は、$`P`$ が純粋なランダムウォークでも
自己相関を持つ。** 移動平均の残差だからで、平均回帰の証拠ではない。
したがって $`\\phi`$ を 0 や 1 と比べるのは無意味で、
**同じ欠測パターン・同じ 1 分ボラ・同じティック幅のランダムウォーク**を
発生させて得た $`\\phi_{RW}`$ と比べなければならない。本スクリプトは
銘柄ごとに 200 本の疑似系列を回して帰無分布を作る。

自己相関と分散比についても同じ対照を置く。**ティック丸めを入れる**ので、
離散性が作る偽の負の自己相関もこの帰無に含まれる。

## 約定値で測ってはいけない

買いと売りが交互に来るだけでリターンに強い負の自己相関が出る(bid–ask bounce)。
比較のため約定終値でも同じ計算をして、mid / microprice との差を示す(§1)。

## 標準誤差

すべて**日ごとに推定して日を単位に**まとめる(日クラスタ)。1 分足の点どうしは
同じ日の中で強く相関しているので、全点を独立とみなすと過小になる。

## 時間契約

バー $`T`$ の値は $`T`$ までの情報だけ(`build_mrev.py` が保証)。
`sma20` と `ewma` は**後ろ向き 20 分 / 指数平均**で、中心化移動平均は使わない。
`model` の係数は**前 60% の日だけ**から推定する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "SNDK", "INTC", "AMD", "SMSN", "DRAM", "KIOXIA", "SKHX"]
NMIN = 1440
LAGS = list(range(1, 61))            # 全ラグを出す(表は抜粋、検算は全部使う)
SHOW = [1, 2, 3, 5, 10, 20, 30, 60]
VRK = [2, 5, 10, 20, 30, 60]
SMA = 20
EWMA_HL = 7.0
RTH = (13 * 60 + 30, 20 * 60)      # 米国立会 13:30–20:00 UTC
NSIM = 200
TRAIN_FRAC = 0.60


def rmean(x, w):
    """後ろ向き w 点の単純移動平均(w 点そろわない先頭は NaN)。"""
    c = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(x))])
    n = np.concatenate([[0], np.cumsum(np.isfinite(x))])
    i = np.arange(x.size) + 1
    lo = np.maximum(i - w, 0)
    s, k = c[i] - c[lo], n[i] - n[lo]
    return np.where(k >= w, s / np.maximum(k, 1), np.nan)


def ewma(x, hl):
    a = 1.0 - np.exp(-np.log(2.0) / hl)
    out = np.full(x.size, np.nan)
    s = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = s
            continue
        s = v if not np.isfinite(s) else s + a * (v - s)
        out[i] = s
    return out


def acf_day(r, lags):
    """1 日ぶんのリターン系列の自己相関。欠測は対で落とす。"""
    out = np.full(len(lags), np.nan)
    f = np.isfinite(r)
    if f.sum() < 60:
        return out
    x = r - np.nanmean(r[f])
    for i, k in enumerate(lags):
        m = f[:-k] & f[k:]
        if m.sum() < 30:
            continue
        a, b = x[:-k][m], x[k:][m]
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            out[i] = float((a * b).mean() / (sa * sb))
    return out


def ar1_day(d):
    """D_{t+1} = phi D_t の原点回帰(1 日ぶん)。"""
    f = np.isfinite(d[:-1]) & np.isfinite(d[1:])
    if f.sum() < 60:
        return np.nan, np.nan
    x, y = d[:-1][f], d[1:][f]
    den = float(x @ x)
    if den <= 0:
        return np.nan, np.nan
    phi = float(x @ y) / den
    r2 = 1.0 - float(((y - phi * x) ** 2).sum()) / max(float((y * y).sum()), 1e-18)
    return phi, r2


def vr_day(r, ks):
    """分散比 VR(k) = Var(r^(k)) / (k Var(r))。重なり合う和で作る。"""
    out = np.full(len(ks), np.nan)
    f = np.isfinite(r)
    if f.sum() < 120:
        return out
    x = np.where(f, r, 0.0)
    v1 = float(np.var(r[f], ddof=1))
    if v1 <= 0:
        return out
    c = np.concatenate([[0.0], np.cumsum(x)])
    cf = np.concatenate([[0], np.cumsum(f)])
    for i, k in enumerate(ks):
        s = c[k:] - c[:-k]
        n = cf[k:] - cf[:-k]
        g = n == k                       # 窓の中が全部そろっている場合だけ
        if g.sum() < 60:
            continue
        out[i] = float(np.var(s[g], ddof=1)) / (k * v1)
    return out


def halflife(phi):
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan
    return float(-np.log(2.0) / np.log(phi))


def clustered(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan, v.size
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), v.size


def sim_null(mid, tick, rng, nsim=NSIM):
    """同じ欠測・同じ 1 分ボラ・同じティック幅のランダムウォークの帰無分布。"""
    f = np.isfinite(mid)
    r = np.diff(np.log(mid))
    sd = float(np.nanstd(r[np.isfinite(r)]))
    p0 = float(mid[f][0])
    phis, vrs, ac1 = [], [], []
    for _ in range(nsim):
        w = np.cumsum(rng.standard_normal(mid.size) * sd)
        p = p0 * np.exp(w)
        p = np.round(p / tick) * tick          # ★ ティックに丸める
        p = np.where(f, p, np.nan)
        d = p - rmean(p, SMA)
        phis.append(ar1_day(d)[0])
        lr = np.diff(np.log(p))
        vrs.append(vr_day(lr, VRK))
        ac1.append(acf_day(lr, LAGS))
    return (np.array(phis), np.array(vrs, dtype=float),
            np.array(ac1, dtype=float))


def fair_model(B, days, ntr):
    """OBI・符号つき出来高・直近リターンから次の 1 分の mid リターンを予測する。

    係数は**前 60% の日だけ**から推定する。
    """
    mid = B["mid"].to_numpy()
    obi = B["obi"].to_numpy()
    bv, sv = B["buy_vol"].to_numpy(), B["sell_vol"].to_numpy()
    tot = np.maximum(np.nan_to_num(bv) + np.nan_to_num(sv), 1e-12)
    ti = (np.nan_to_num(bv) - np.nan_to_num(sv)) / tot
    lm = np.log(mid)
    r1 = np.concatenate([[np.nan], np.diff(lm)])
    dayidx = np.repeat(np.arange(len(days)), NMIN)
    # 日の先頭のリターンは前日を跨ぐので落とす
    r1[np.arange(r1.size) % NMIN == 0] = np.nan
    X = np.column_stack([np.nan_to_num(obi), ti, np.nan_to_num(r1) * 1e4])
    y = np.concatenate([np.diff(lm), [np.nan]]) * 1e4
    y[np.arange(y.size) % NMIN == NMIN - 1] = np.nan
    tr = (dayidx < ntr) & np.isfinite(y) & np.isfinite(mid)
    if tr.sum() < 1000:
        return np.full(mid.size, np.nan)
    Z = np.column_stack([np.ones(tr.sum()), X[tr]])
    w, *_ = np.linalg.lstsq(Z, y[tr], rcond=None)
    yhat = w[0] + X @ w[1:]
    return mid * (1.0 + yhat / 1e4)       # 予測どおり動いた先が fair


def main() -> None:
    rng = np.random.default_rng(20260915)
    acf_rows, sp_rows, vr_rows, sm_rows = [], [], [], []
    for c in COINS:
        f = DATA / f"mrev_bars_xyz_{c}.parquet"
        if not f.exists():
            print(f"xyz:{c}: バーが無い。skip", file=sys.stderr)
            continue
        B = pl.read_parquet(f).sort("dt", "minute")
        days = B["dt"].unique(maintain_order=True).to_list()
        nd = len(days)
        ntr = int(round(nd * TRAIN_FRAC))
        mid = B["mid"].to_numpy()
        micro = B["micro"].to_numpy()
        trade = B["trade"].to_numpy()
        minute = B["minute"].to_numpy()
        tick = float(np.nanmedian(np.where(np.nanmedian(mid) >= 1000, 0.1, 0.01)))
        fmodel = fair_model(B, days, ntr)

        # ---- 基準価格 4 通り ------------------------------------------
        FS = {}
        for nm in ("micro", "ewma", "sma20", "model"):
            g = np.full(mid.size, np.nan)
            for i in range(nd):
                sl = slice(i * NMIN, (i + 1) * NMIN)
                m = mid[sl]
                if nm == "micro":
                    g[sl] = micro[sl]
                elif nm == "ewma":
                    g[sl] = ewma(m, EWMA_HL)
                elif nm == "sma20":
                    g[sl] = rmean(m, SMA)
                else:
                    g[sl] = fmodel[sl]
            FS[nm] = g

        # ---- §1 自己相関 -----------------------------------------------
        for pn, px in (("mid", mid), ("micro", micro), ("trade", trade)):
            if not np.isfinite(px).any():
                continue
            per = []
            for i in range(nd):
                sl = slice(i * NMIN, (i + 1) * NMIN)
                per.append(acf_day(np.diff(np.log(px[sl])), LAGS))
            A = np.array(per, float)
            for j, k in enumerate(LAGS):
                m, se, n = clustered(A[:, j])
                acf_rows.append({"coin": f"xyz:{c}", "px": pn, "lag": k,
                                 "rho": m, "se": se, "n_days": n,
                                 "t": m / se if se and se > 0 else np.nan})

        # ---- 帰無(ランダムウォーク)------------------------------------
        nph, nvr, nac = [], [], []
        for i in range(min(nd, 30)):          # 30 日ぶん回せば十分安定する
            sl = slice(i * NMIN, (i + 1) * NMIN)
            if np.isfinite(mid[sl]).sum() < 200:
                continue
            a, b, d = sim_null(mid[sl], tick, rng, nsim=max(NSIM // 30, 5))
            nph.append(a)
            nvr.append(b)
            nac.append(d)
        nph = np.concatenate(nph) if nph else np.array([np.nan])
        nvr = np.concatenate(nvr) if nvr else np.full((1, len(VRK)), np.nan)
        nac = np.concatenate(nac) if nac else np.full((1, len(LAGS)), np.nan)

        # ---- §2 回帰速度 -----------------------------------------------
        for nm, g in FS.items():
            per, r2s, sds = [], [], []
            for i in range(nd):
                sl = slice(i * NMIN, (i + 1) * NMIN)
                d = mid[sl] - g[sl]
                p, r2 = ar1_day(d)
                per.append(p)
                r2s.append(r2)
                sds.append(float(np.nanstd(d / mid[sl] * 1e4)))
            m, se, n = clustered(per)
            phi_rw = float(np.nanmean(nph)) if nm == "sma20" else np.nan
            sp_rows.append({
                "coin": f"xyz:{c}", "fair": nm, "phi": m, "se": se, "n_days": n,
                "half_life_min": halflife(m),
                "phi_rw": phi_rw, "half_life_rw": halflife(phi_rw),
                "z_vs_rw": ((m - phi_rw) / se) if (nm == "sma20" and se) else np.nan,
                "r2": float(np.nanmean(r2s)),
                "sd_bp": float(np.nanmean(sds))})

        # ---- §3 分散比 -------------------------------------------------
        per = []
        for i in range(nd):
            sl = slice(i * NMIN, (i + 1) * NMIN)
            per.append(vr_day(np.diff(np.log(mid[sl])), VRK))
        V = np.array(per, float)
        for j, k in enumerate(VRK):
            m, se, n = clustered(V[:, j])
            rw = float(np.nanmean(nvr[:, j]))
            vr_rows.append({"coin": f"xyz:{c}", "k": k, "vr": m, "se": se,
                            "n_days": n, "vr_rw": rw,
                            "z_vs_rw": (m - rw) / se if se and se > 0 else np.nan,
                            "z_vs_1": (m - 1) / se if se and se > 0 else np.nan})

        # ---- §4 時間帯 -------------------------------------------------
        out = {}
        for lab, msk in (("立会", (minute >= RTH[0]) & (minute < RTH[1])),
                         ("時間外", ~((minute >= RTH[0]) & (minute < RTH[1])))):
            per_p, per_v = [], []
            for i in range(nd):
                sl = slice(i * NMIN, (i + 1) * NMIN)
                mm = np.where(msk[sl], mid[sl], np.nan)
                d = mm - rmean(mm, SMA)
                per_p.append(ar1_day(d)[0])
                per_v.append(vr_day(np.diff(np.log(mm)), [20])[0])
            out[lab] = (clustered(per_p)[0], clustered(per_v)[0],
                        np.array(per_v, float))

        # ---- 執行できるか(D_t から将来リターンを予測して費用と比べる)----
        # y は (t, t+h] の mid リターン、x は t で確定する D_t。先読みは無い。
        eco = {}
        for h in (5, 20, 60):
            per_b = []
            for i in range(nd):
                sl = slice(i * NMIN, (i + 1) * NMIN)
                m = mid[sl]
                d = (m - rmean(m, SMA)) / m * 1e4
                fw = np.full(NMIN, np.nan)
                fw[:-h] = (np.log(m[h:]) - np.log(m[:-h])) * 1e4
                g = np.isfinite(d) & np.isfinite(fw)
                if g.sum() < 100:
                    continue
                x, yv = d[g], fw[g]
                den = float(x @ x)
                if den > 0:
                    per_b.append(float(x @ yv) / den)
            eco[h] = clustered(per_b)
        sdd = [r for r in sp_rows if r["coin"] == f"xyz:{c}"
               and r["fair"] == "sma20"][0]["sd_bp"]

        sm_rows.append({
            "coin": f"xyz:{c}", "n_days": nd,
            "beta5": eco[5][0], "beta20": eco[20][0], "beta60": eco[60][0],
            "t_beta20": (eco[20][0] / eco[20][1]) if eco[20][1] else np.nan,
            "sd_D_bp": sdd, "edge1sd_20_bp": eco[20][0] * sdd,
            "cover": float(np.isfinite(mid).mean()),
            "spread_bp": float(B["spread_bp"].median()),
            "sd_1min_bp": float(np.nanstd(np.diff(np.log(mid))) * 1e4),
            "tick_bp": tick / float(np.nanmedian(mid)) * 1e4,
            "phi_sma20": [r for r in sp_rows if r["coin"] == f"xyz:{c}"
                          and r["fair"] == "sma20"][0]["phi"],
            "phi_rw": float(np.nanmean(nph)),
            "rho1_mid": [r for r in acf_rows if r["coin"] == f"xyz:{c}"
                         and r["px"] == "mid" and r["lag"] == 1][0]["rho"],
            "rho1_rw": float(np.nanmean(nac[:, 0])),
            "phi_rth": out["立会"][0], "phi_oth": out["時間外"][0],
            "vr20_rth": out["立会"][1], "vr20_oth": out["時間外"][1],
            "vr20_diff_z": (lambda d: (np.nanmean(d)
                                       / (np.nanstd(d, ddof=1)
                                          / np.sqrt(np.isfinite(d).sum())))
                            if np.isfinite(d).sum() > 3 else np.nan)(
                out["立会"][2] - out["時間外"][2])})
        print(f"[mrev] xyz:{c} 完了({nd} 日)", flush=True, file=sys.stderr)

    pl.DataFrame(acf_rows).write_csv(DATA / "mrev_acf.csv")
    pl.DataFrame(sp_rows).write_csv(DATA / "mrev_speed.csv")
    pl.DataFrame(vr_rows).write_csv(DATA / "mrev_vr.csv")
    S = pl.DataFrame(sm_rows)
    S.write_csv(DATA / "mrev_summary.csv")

    A = pl.DataFrame(acf_rows)
    print("\n===== ① 1 分リターンの自己相関 ρ(k) =====")
    print("★ 約定値は bid–ask bounce で負に出る。mid / microprice と比べること。\n")
    for pn in ("mid", "micro", "trade"):
        q = A.filter(pl.col("px") == pn)
        if not q.height:
            continue
        print(f"  {pn}")
        print("    銘柄        " + "".join(f"{'k='+str(k):>10}" for k in SHOW))
        for c in COINS:
            r = (q.filter((pl.col("coin") == f"xyz:{c}")
                          & pl.col("lag").is_in(SHOW)).sort("lag"))
            if not r.height:
                continue
            print(f"    xyz:{c:<8}" + "".join(
                f"{v:>10.3f}" for v in r["rho"].to_list()))
    print("\n  ランダムウォーク(同じ欠測・同じボラ・ティック丸め)の ρ(1)")
    print("    " + "  ".join(f"{r['coin']} {r['rho1_rw']:+.3f}"
                             for r in S.iter_rows(named=True)))

    P = pl.DataFrame(sp_rows)
    print("\n===== ② 基準価格からの乖離 D_t の AR(1) と半減期 =====")
    print(f"{'銘柄':<12}{'基準':<8}{'phi':>8}{'se':>7}{'半減期(分)':>11}"
          f"{'R^2':>7}{'sd(bp)':>9}{'RW の phi':>10}{'RW の半減期':>12}{'z':>8}")
    for c in COINS:
        for nm in ("micro", "ewma", "sma20", "model"):
            r = P.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("fair") == nm))
            if not r.height:
                continue
            r = r.to_dicts()[0]
            print(f"xyz:{c:<8}{nm:<8}{r['phi']:>8.4f}{r['se']:>7.4f}"
                  f"{r['half_life_min']:>11.2f}{r['r2']:>7.3f}{r['sd_bp']:>9.2f}"
                  f"{r['phi_rw']:>10.4f}{r['half_life_rw']:>12.2f}"
                  f"{r['z_vs_rw']:>8.2f}")
        print()

    V = pl.DataFrame(vr_rows)
    from scipy import stats
    nt = P.filter(pl.col("fair") == "sma20").height + V.height
    thr = float(stats.norm.ppf(1 - 0.05 / nt / 2))
    print("===== ③ 分散比 VR(k) =====")
    print(f"検定は {nt} 通り(φ の RW 比較 + VR の RW 比較)/ "
          f"Bonferroni の |z| 閾値 {thr:.2f}\n")
    print("    銘柄        " + "".join(f"{'k='+str(k):>9}" for k in VRK))
    for c in COINS:
        q = V.filter(pl.col("coin") == f"xyz:{c}").sort("k")
        if not q.height:
            continue
        print(f"    xyz:{c:<8}" + "".join(f"{v:>9.3f}" for v in q["vr"].to_list()))
        print(f"      RW    " + "".join(f"{v:>9.3f}" for v in q["vr_rw"].to_list()))
        print(f"      z     " + "".join(
            (f"{v:>8.1f}" + ("*" if abs(v) > thr else " "))
            for v in q["z_vs_rw"].to_list()))

    # ★ VR と ρ(k) の整合性検算(同じ標本から独立に作った 2 つが一致するか)
    print("\n  検算: VR(k) ≈ 1 + 2 Σ_{j<k} (1 − j/k) ρ(j)")
    print(f"    {'銘柄':<12}{'VR(20) 実測':>12}{'ρ から再構成':>14}{'差':>8}")
    for c in COINS:
        a = (A.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("px") == "mid"))
             .sort("lag"))
        q = V.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("k") == 20))
        if not a.height or not q.height:
            continue
        # ラグ 1〜19 をすべて測ってあるので補間は不要(厳密な検算)
        lg = np.array(a["lag"].to_list(), float)
        rh = np.array(a["rho"].to_list(), float)
        jj = np.arange(1, 20)
        ri = rh[np.searchsorted(lg, jj)]
        rec = 1 + 2 * float(np.sum((1 - jj / 20) * ri))
        vv = float(q["vr"][0])
        print(f"    xyz:{c:<8}{vv:>12.3f}{rec:>14.3f}{vv-rec:>8.3f}")

    print("\n===== ④ 平均回帰は取引できる大きさか =====")
    print("y = (t, t+h] の mid リターン(bp)を x = D_t(bp、SMA20 基準)へ回帰する。")
    print("β<0 なら乖離が戻る。1 標準偏差ぶん乖離したときに取れる bp を、")
    print("その銘柄の**往復費用**(メイカー検証レポートの実測値)と比べる。\n")
    # 往復費用 = 2 × (中央スプレッド / 2 + テイカー手数料 0.79bp)
    # = 中央スプレッド + 1.58bp。メイカー検証レポートの実測値と一致する。
    print(f"{'銘柄':<12}{'β(5分)':>9}{'β(20分)':>10}{'β(60分)':>10}{'t(20分)':>9}"
          f"{'sd(D)':>8}{'1sd の粗利':>12}{'往復費用':>10}{'粗利−費用':>11}")
    for r in S.iter_rows(named=True):
        cst = r["spread_bp"] + 2 * 0.79
        e = abs(r["edge1sd_20_bp"])          # 逆張りするので符号は取り払う
        print(f"{r['coin']:<12}{r['beta5']:>9.4f}{r['beta20']:>10.4f}"
              f"{r['beta60']:>10.4f}{r['t_beta20']:>9.2f}{r['sd_D_bp']:>8.1f}"
              f"{e:>12.2f}{cst:>10.2f}{e-cst:>11.2f}")
    print("\n★ 「1sd で取れる bp」は**片道の粗利**で、費用を引く前である。")

    print("\n===== ⑤ 立会時間と時間外 =====")
    print("同じ日で立会と時間外の VR(20) を引き算し、その差の系列に日クラスタの t を当てる。")
    print(f"{'銘柄':<12}{'phi 立会':>10}{'phi 時間外':>11}"
          f"{'VR(20) 立会':>13}{'VR(20) 時間外':>14}{'差の z':>9}")
    for r in S.iter_rows(named=True):
        sig = "*" if abs(r["vr20_diff_z"] or 0) > thr else " "
        print(f"{r['coin']:<12}{r['phi_rth']:>10.4f}{r['phi_oth']:>11.4f}"
              f"{r['vr20_rth']:>13.3f}{r['vr20_oth']:>14.3f}"
              f"{r['vr20_diff_z']:>8.2f}{sig}")


if __name__ == "__main__":
    main()
