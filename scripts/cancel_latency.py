"""キャンセル猶予時間の実測 — 反応時間 δ を与えたときに選別の利益が残るか。

【問い】
  不利な約定を避けるには「約定より前に」シグナルを見て板から引く必要がある。
  実際の発注系は反応に時間がかかる(シグナル計算 + 発注往復 + 取引所のブロック)。
  反応時間 δ を仮定したとき、δ だけ古いシグナルでも選別の利益が残るか。

【設計】
  メイカー約定 1 件について
    方向        s   = +1(メイカーが買った) / −1(売った)
    実現損益    r   = s × (mid(t+1s) − p) / mid(t) × 10⁴ − 手数料 0.088bp
    シグナル    g_δ = book_slope_diff を **時刻 t − δ 以前の直近グリッド点**で読む
    整合度      a_δ = s × (−g_δ)        … 予測リターン ∝ −book_slope_diff
  δ ∈ {0, 100ms, 200ms, 500ms, 1s, 2s, 5s} で、
  「a_δ が上位 30% の約定だけを取る」戦略の平均損益を測る。

  δ を伸ばして利益が消える点が、この戦略に許される反応時間の上限になる。

【統計的正当性のための措置】
  (1) 閾値は**前日**の分布から取る(当日の分布を使うと未来を見たことになる)
  (2) 推定は**日ごと**に行い、10 日の中央値・符号一貫性・ブロック・ブートストラップで報告する
      (プールした t は n が大きすぎて意味を持たない: regression_report.md §9)
  (3) **プラセボ**: シグナル系列を 1 時間ずらして同じ手順を回す。
      分布は同じで時点の対応だけ壊れるので、ここで利益が消えなければ見せかけを疑う
  (4) δ ごとの結果を**全部報告**する(良い δ だけ拾わない)
  (5) 参考として「警告時間」= 不利な約定の前にシグナルが不利へ転じてからの経過時間、
      の分布も出す

【限界(必ず併記)】
  ここで数えているのは**他の参加者に実際に起きた約定**である。
  自分が引いたときに「その約定が自分に起きなかった」ことは検証できない。
  したがってこれは**タイミングが物理的に間に合うかの検証**であって、損益の検証ではない。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
SLOPE100 = D / "book_slope_grain"          # 100ms グリッドの深さ帯別 slope(10 日分)
FEE = 0.088
BAND = "25bp"
LAGS = {"0ms": 0, "100ms": 100_000_000, "200ms": 200_000_000, "500ms": 500_000_000,
        "1s": 1_000_000_000, "2s": 2_000_000_000, "5s": 5_000_000_000}
TOPQ = 0.70                                 # 上位 30% を選ぶ
PLACEBO_SHIFT = 3_600_000_000_000           # 1 時間
RNG = np.random.default_rng(20260815)


def load_day(dt: str) -> dict | None:
    f = SLOPE100 / f"{dt}.parquet"
    if not f.exists():
        return None
    fs = sorted(glob.glob(f"data/fills_v99/dt={dt}/**/*.parquet", recursive=True))
    if not fs:
        return None
    fl = pl.concat([pl.read_parquet(x, columns=["ts", "px", "sz", "side", "crossed", "tid"])
                    for x in fs], how="diagonal_relaxed")
    fl = (fl.filter(~pl.col("crossed")).unique(subset=["tid"], keep="first")
            .with_columns(pl.col("ts").cast(pl.Int64).alias("t")).sort("t"))
    m = (pl.read_parquet(D / f"microprice/dt={dt}/part-000.parquet",
                         columns=["ts", "mid", "is_crossed"])
         .filter(~pl.col("is_crossed")).sort("ts"))
    s = pl.read_parquet(f)
    sig = (s[f"bid_{BAND}"].to_numpy() - s[f"ask_{BAND}"].to_numpy())
    return {"t": fl["t"].to_numpy(), "px": fl["px"].to_numpy(),
            "s": np.where(fl["side"].to_numpy() == "B", 1.0, -1.0),
            "mts": m["ts"].to_numpy(), "mv": m["mid"].to_numpy(),
            "sts": s["ts"].to_numpy(), "sig": sig}


def main() -> None:
    days = sorted(p.stem for p in SLOPE100.glob("*.parquet"))
    per_day = []
    warn = []
    for dt in days:
        d = load_day(dt)
        if d is None:
            continue
        t, px, s = d["t"], d["px"], d["s"]
        mts, mv = d["mts"], d["mv"]
        sts, sig = d["sts"], d["sig"]
        j0 = np.searchsorted(mts, t, side="right") - 1
        j1 = np.searchsorted(mts, t + 1_000_000_000, side="right") - 1
        ok = (j0 >= 0) & (j1 >= 0) & (t + 1_000_000_000 <= mts[-1])
        m0 = mv[np.clip(j0, 0, mv.size - 1)]
        r = s * (mv[np.clip(j1, 0, mv.size - 1)] - px) / m0 * 1e4 - FEE
        row = {"dt": dt, "n": int(ok.sum()), "all_mean": float(r[ok].mean())}
        for name, lag in LAGS.items():
            k = np.searchsorted(sts, t - lag, side="right") - 1      # ★backward のみ
            good = ok & (k >= 0)
            a = np.where(good, s * (-sig[np.clip(k, 0, sig.size - 1)]), np.nan)
            row[f"align_{name}"] = a
        # プラセボ: シグナルを 1 時間ずらして読む(分布は同じ・時点対応だけ壊す)
        kp = np.searchsorted(sts, t - PLACEBO_SHIFT, side="right") - 1
        gp = ok & (kp >= 0)
        row["align_placebo"] = np.where(gp, s * (-sig[np.clip(kp, 0, sig.size - 1)]), np.nan)
        row["r"] = r
        row["ok"] = ok
        # 警告時間: 不利な約定について、シグナルが不利へ転じてからの経過
        adv = ok & (r < 0)
        k0 = np.searchsorted(sts, t, side="right") - 1
        al_now = s * (-sig[np.clip(k0, 0, sig.size - 1)])
        for i in np.nonzero(adv & (al_now < 0))[0][:20000]:
            j = k0[i]
            c = 0
            while j > 0 and (s[i] * (-sig[j])) < 0 and c < 600:      # 最大 60 秒遡る
                j -= 1
                c += 1
            warn.append((t[i] - sts[j]) / 1e6)                       # ms
        per_day.append(row)

    # ---- 日ごとに「上位 30% だけ約定」の平均損益を出す(閾値は前日から) ----
    res: dict = {"days": [x["dt"] for x in per_day], "fee_bp": FEE,
                 "band": BAND, "top_quantile": TOPQ}
    keys = list(LAGS) + ["placebo"]
    daily = {k: [] for k in keys}
    daily_all = []
    for i in range(1, len(per_day)):
        cur, prev = per_day[i], per_day[i - 1]
        daily_all.append(cur["all_mean"])
        for k in keys:
            a_prev = prev[f"align_{k}"]
            a_cur = cur[f"align_{k}"]
            m_prev = np.isfinite(a_prev)
            m_cur = np.isfinite(a_cur) & cur["ok"]
            if m_prev.sum() < 1000 or m_cur.sum() < 1000:
                daily[k].append(np.nan)
                continue
            th = np.quantile(a_prev[m_prev], TOPQ)                   # ★前日の閾値
            sel = m_cur & (a_cur >= th)
            daily[k].append(float(cur["r"][sel].mean()) if sel.sum() > 100 else np.nan)

    base = np.array(daily_all)
    res["baseline_all_fills"] = {"median": float(np.median(base)),
                                 "days": int(base.size)}
    for k in keys:
        v = np.array(daily[k], dtype=float)
        g = np.isfinite(v)
        d_ = v[g] - base[g]                                          # 選別の上乗せ
        if d_.size < 3:
            continue
        L = 2
        nb = int(np.ceil(d_.size / L))
        st = RNG.integers(0, d_.size - L + 1, size=(5000, nb))
        boot = np.stack([np.concatenate([d_[x:x + L] for x in row])[:d_.size]
                         for row in st]).mean(1)
        res[k] = {
            "days": int(d_.size),
            "selected_mean_median": float(np.median(v[g])),
            "uplift_median": float(np.median(d_)),
            "uplift_mean": float(d_.mean()),
            "days_negative": int((d_ < 0).sum()),
            "boot_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "boot_p_le0": float((boot <= 0).mean()),
        }
    w = np.array(warn)
    res["warning_time_ms"] = {
        "n": int(w.size),
        **{f"p{int(q*100)}": float(np.quantile(w, q)) for q in (.1, .25, .5, .75, .9)},
    }
    (D / "cancel_latency.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    print(f"対象 {len(per_day)} 日 / 評価は 2 日目以降の {len(base)} 日"
          f"(閾値に前日を使うため)")
    print(f"選別なし(全約定)の平均損益 中央値: {res['baseline_all_fills']['median']:+.4f} bp\n")
    print(f"{'反応時間 δ':>10} {'選別後の損益':>12} {'上乗せ':>10} "
          f"{'負の日':>8} {'BS 95% CI':>22} {'p(≤0)':>8}")
    for k in keys:
        if k not in res:
            continue
        v = res[k]
        ci = f"[{v['boot_ci95'][0]:+.4f},{v['boot_ci95'][1]:+.4f}]"
        nm = "プラセボ" if k == "placebo" else k
        print(f"{nm:>10} {v['selected_mean_median']:>+12.4f} {v['uplift_median']:>+10.4f} "
              f"{v['days_negative']:>3}/{v['days']} {ci:>22} {v['boot_p_le0']:>8.4f}")
    w_ = res["warning_time_ms"]
    print(f"\n警告時間(不利な約定 {w_['n']:,} 件、ms): "
          f"p10={w_['p10']:.0f} p25={w_['p25']:.0f} p50={w_['p50']:.0f} "
          f"p75={w_['p75']:.0f} p90={w_['p90']:.0f}")


if __name__ == "__main__":
    main()
