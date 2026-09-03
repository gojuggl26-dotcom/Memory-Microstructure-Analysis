"""板の応答 11 指標が将来の値動きを教えるかを、5 秒格子で測る。

【何を測るか】
`build_impact.py` が出した指標(時刻 T までの情報だけで決まる)を x、
T **以降**の値動きを y として関係を測る。y は 3 種類:

    dir   将来の log リターン        …… 向きを当てられるか
    mag   将来の |log リターン|      …… 大きさを当てられるか
    vol   将来の実現ボラティリティ    …… 5 秒刻みの二乗和の平方根

【この案件で繰り返し踏んだ落とし穴を、最初から潰しておく】

1. **同時点の関係と予測を混ぜない。** 同じ x について「前向き」corr(x_T, r_{T→T+h})
   と「後ろ向き」corr(x_T, r_{T−h→T}) を必ず並べて出す。板の量は値動きの
   **後始末**であることが多く、後ろ向きのほうが大きければそれは予測力ではない。
2. **重なる窓を使わない。** ホライズン h の検定は h ごとに 1 点だけ取る
   (stride = h)。重ねると t 値が最大 6 倍に水増しされる(mu_book_slope_report)。
3. **帰無対照を 2 つ置く。** (a) x を 1 日**ちょうど**ずらす → 時刻の季節性は
   保たれるので、「時間帯の癖だけで出る分」が測れる。(b) 1 日 + 6 時間ずらす →
   季節性も崩れるので純粋な帰無になる。(b) で相関が残れば計算が間違っている。
4. **ベースラインを引く。** 大きさ・ボラティリティの予測は、直前の
   ボラティリティを入れるだけで勝手に当たる。統制集合
   [直前 5 分の実現ボラティリティ, スプレッド, 深さの偏り] へ x と y の両方を
   回帰した**残差どうし**の相関(偏相関)を本命の判定に使う。
5. **標本外を時間で切る。** 前半で符号と尺度を決め、後半だけで評価する。
   行を混ぜた k 分割は隣の格子点が相関しているので漏れる。
6. **費用と比べる。** 期待できる値動き(bp)を往復のスプレッドと並べる。
   有意であることと、費用を引いて残ることは別である。
7. **既存の物差しと並べる。** 板から向きを当てる特徴量は既にいくつもある。
   最良気配の数量の偏り(= マイクロプライス)と OFI を同じ表に入れ、
   新しい指標がそれらを**超えるか**、あるいは統制しても残るかを見る。

【x が確定する時刻 / y の期間】
x は T まで(板は T 未満のイベント、脆さは [T−30秒, T) の実績)。
y は [T, T+h)。境界は厳密に守り、日を跨ぐリターンは作らない。

    uv run python scripts/build_impact_signal.py --coin xyz:MU
出力: data/impact_signal_<coin>.csv   特徴量 × 目的変数 × ホライズン
      data/impact_curve_<coin>.csv    Impact(Q) の平均形(図示用)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

GRID_S = 5                                   # 格子(秒)
HOR = [1, 3, 12, 60, 180]                    # 格子点。5秒 / 15秒 / 1分 / 5分 / 15分
HLAB = ["5秒", "15秒", "1分", "5分", "15分"]
VOL_LB = 60                                  # 統制に使う直前の窓(格子点 = 5 分)
PLACEBO_SHIFT = 17_280 + 4_320               # 1 日 + 6 時間(季節性も崩す)
PLACEBO_DAY = 17_280                         # 1 日ちょうど(季節性は残る)
OFI_LB = {"ofi_1s": 1, "ofi_60s": 12}        # OFI の窓(格子点)

FEATS = {
    "imp_avg_q5": "価格インパクト(平均、Q=5)",
    "imp_asym_q5": "価格インパクトの非対称(売り−買い、Q=5)",
    "imp_avg_q50": "価格インパクト(平均、Q=50)",
    "slp_avg_q5": "スリッページ(平均、Q=5)",
    "lv_avg_q5": "削る価格水準の数(平均、Q=5)",
    "lv_avg_q50": "削る価格水準の数(平均、Q=50)",
    "mdep_avg_q5": "marginal depth(平均、Q=5)",
    "mimp_avg_q5": "marginal impact(平均、Q=5)",
    "spread_bp": "スプレッド幅",
    "d25_tot": "±25bp の深さ(両側合計)",
    "d25_imb": "±25bp の深さの偏り(買い−売り)",
    "frag_avg": "脆さ(平均)",
    "frag_imb": "脆さの偏り(買い−売り)",
    "dar_tot": "depth-at-risk(両側合計)",
    "dar_imb": "depth-at-risk の偏り",
    "gfr_avg_q5": "gap 調整後の上乗せ(平均、Q=5)",
    "gfr_avg_q50": "gap 調整後の上乗せ(平均、Q=50)",
    "gfr_asym_q5": "gap 調整後の上乗せの非対称(売り−買い)",
    "l1_imb": "【比較】最良気配の数量の偏り(マイクロプライス)",
    "ofi_1s": "【比較】直前 1 秒の OFI",
    "ofi_60s": "【比較】直前 60 秒の OFI",
}
SIGNED = {"imp_asym_q5", "d25_imb", "frag_imb", "dar_imb", "gfr_asym_q5",
          "l1_imb", "ofi_1s", "ofi_60s"}
# 偏相関の統制集合は目的変数で変える。
#   向き   … 既知の板シグナル(最良気配の偏り・OFI)とスプレッドを抜く
#   大きさ … 直前のボラティリティを抜かないと勝手に当たる
CTRL = {"dir": ["l1_imb", "ofi_1s", "spread_bp"],
        "mag": ["rv_past", "spread_bp", "d25_imb"],
        "vol": ["rv_past", "spread_bp", "d25_imb"]}


def bbo_grid(tag: str, d: pl.DataFrame) -> pl.DataFrame:
    """bbo から比較用の特徴量を作り、同じ格子へ**厳密に T 未満**で載せる。

    最良気配の数量の偏り(= マイクロプライスの乖離と単調に対応)と OFI は、
    板から向きを当てる古典的な物差し。新しい指標と同じ表に並べるために、
    `build_impact.py` とまったく同じ後ろ向きの規約で作る。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_obi_levels import clean_bbo
    bb, _ = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    outs = []
    for dt, sub in d.group_by("dt", maintain_order=True):
        dtv = dt[0] if isinstance(dt, tuple) else dt
        b = bb.filter(pl.col("dt") == dtv).sort("ts")
        ts = b["ts"].cast(pl.Int64).to_numpy()
        pb, pa = b["best_bid"].to_numpy(), b["best_ask"].to_numpy()
        qb, qa = b["bid_sz"].to_numpy(), b["ask_sz"].to_numpy()
        # Cont-Kukanov-Stoikov の OFI
        e = np.concatenate([[0.0], (pb[1:] >= pb[:-1]) * qb[1:]
                            - (pb[1:] <= pb[:-1]) * qb[:-1]
                            - ((pa[1:] <= pa[:-1]) * qa[1:]
                               - (pa[1:] >= pa[:-1]) * qa[:-1])])
        ce = np.concatenate([[0.0], np.cumsum(e)])
        tg = sub["ts"].to_numpy()
        j = np.searchsorted(ts, tg, side="left") - 1      # ★ T 未満
        ok = j >= 0
        jj = np.where(ok, j, 0)
        sb, sa = qb[jj], qa[jj]
        with np.errstate(invalid="ignore", divide="ignore"):
            imb = np.where(ok & (sb + sa > 0), (sb - sa) / (sb + sa), np.nan)
        cols = {"ts": sub["ts"], "dt": pl.Series([dtv] * sub.height),
                "l1_imb": imb}
        for nm, lb in OFI_LB.items():
            j1 = np.searchsorted(ts, tg - lb * GRID_S * 1_000_000_000, side="left")
            cols[nm] = np.where(ok, ce[jj + 1] - ce[j1], np.nan)
        outs.append(pl.DataFrame(cols))
    return pl.concat(outs)


def build(tag: str) -> pl.DataFrame:
    d = pl.read_parquet(ROOT / "data" / f"impact_{tag}.parquet").sort("dt", "ts")
    d = d.join(bbo_grid(tag, d), on=["dt", "ts"], how="left")
    e = {}
    for tg in ("q05", "q5", "q50"):
        for nm in ("imp", "slp", "lv", "mdep", "mimp", "gfr"):
            a, b = pl.col(f"{nm}_a_{tg}"), pl.col(f"{nm}_b_{tg}")
            e[f"{nm}_avg_{tg}"] = (a + b) / 2
            e[f"{nm}_asym_{tg}"] = b - a          # 売り側 − 買い側
    d = d.with_columns(**e)
    d = d.with_columns(
        d25_tot=pl.col("d25_b") + pl.col("d25_a"),
        d25_imb=(pl.col("d25_b") - pl.col("d25_a"))
        / (pl.col("d25_b") + pl.col("d25_a")),
        dar_tot=pl.col("dar_b") + pl.col("dar_a"),
        dar_imb=(pl.col("dar_b") - pl.col("dar_a"))
        / (pl.col("dar_b") + pl.col("dar_a")),
        frag_avg=(pl.col("frag_b") + pl.col("frag_a")) / 2,
        lm=pl.col("mid").log(),
    )
    # 日ごとに 1 格子リターン。日を跨ぐ差は作らない
    d = d.with_columns(r1=(pl.col("lm").diff().over("dt") * 1e4))   # bp
    # 直前 5 分の実現ボラティリティ(T までで閉じる)
    d = d.with_columns(
        rv_past=(pl.col("r1").pow(2).rolling_sum(VOL_LB, min_samples=VOL_LB // 2)
                 .over("dt")).sqrt())
    return d


def fwd(lm: np.ndarray, ts: np.ndarray, h: int):
    """[T, T+h) の log リターン(bp)。

    ★bbo が引けなかった格子点は行ごと落ちているので、**行の間隔が格子幅の
      h 倍ちょうど**であることを ts で確かめる。行番号だけで h 個先を見ると、
      抜けた行のぶんだけ実際のホライズンが伸びる(日を跨ぐ判定も兼ねる)。
    """
    n = len(lm)
    r = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    r[:n - h] = (lm[h:] - lm[:n - h]) * 1e4
    ok[:n - h] = (ts[h:] - ts[:n - h]) == h * GRID_S * 1_000_000_000
    return np.where(ok, r, np.nan)


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan, 0
    xs, ys = x[m], y[m]
    if xs.std() == 0 or ys.std() == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(xs, ys)[0, 1]), int(m.sum())


def resid(y, C, m):
    """統制集合 C(列で並べた配列)へ最小二乗で回帰した残差。"""
    A = np.column_stack([np.ones(m.sum())] + [c[m] for c in C])
    b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    out = np.full(len(y), np.nan)
    out[m] = y[m] - A @ b
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    d = build(tag)
    print(f"[load] {d.height:,} 格子点 / {d['dt'].n_unique()} 日", file=sys.stderr)

    lm = d["lm"].to_numpy()
    tsv = d["ts"].to_numpy()
    day = d["dt"].to_numpy()
    r1 = d["r1"].to_numpy()
    days = np.array(sorted(set(day.tolist())))
    cut = days[len(days) // 2]
    is_h1 = day < cut                                   # 前半 = 標本内
    X = {f: d[f].to_numpy().astype(np.float64) for f in FEATS}
    CV = {t: [np.nan_to_num(d[c].to_numpy().astype(np.float64), nan=0.0)
              for c in cs] for t, cs in CTRL.items()}

    rows = []
    for h, hl in zip(HOR, HLAB):
        yf = fwd(lm, tsv, h)                              # 前向き
        yb = np.roll(yf, h)                               # 後ろ向き(同じ量をずらす)
        yb[:h] = np.nan
        ya = np.abs(yf)
        # 将来の実現ボラティリティ(同じ窓の 5 秒リターンの二乗和)
        sq = np.nan_to_num(r1 ** 2, nan=0.0)
        cs = np.concatenate([[0.0], np.cumsum(sq)])
        rv = np.full(len(lm), np.nan)
        if h >= 2:
            v = cs[h:] - cs[:-h]
            rv[:len(v) - 1] = np.sqrt(v[1:])
            rv[np.isnan(yf)] = np.nan
        rvb = np.roll(rv, h)                              # 直前 h の実現ボラティリティ
        rvb[:h] = np.nan
        # 目的変数ごとに「後ろ向き」の相手も同じ変換にする
        BWD = {"dir": yb, "mag": np.abs(yb), "vol": rvb}
        stride = max(h, 1)
        for f in FEATS:
            x = X[f]
            xp = np.roll(x, PLACEBO_SHIFT)                # 帰無対照(季節性も崩す)
            xp[:PLACEBO_SHIFT] = np.nan
            xd = np.roll(x, PLACEBO_DAY)                  # 1 日ちょうど(季節性は残る)
            xd[:PLACEBO_DAY] = np.nan
            for tname, y in (("dir", yf), ("mag", ya), ("vol", rv)):
                if tname == "vol" and h < 2:
                    continue
                sl = slice(None, None, stride)            # 重ならない部分標本
                cf, n = corr(x[sl], y[sl])
                cb, _ = corr(x[sl], BWD[tname][sl])
                cp, _ = corr(xp[sl], y[sl])
                cpd, _ = corr(xd[sl], y[sl])
                # 偏相関(統制集合の残差どうし)
                m = np.isfinite(x) & np.isfinite(y)
                # 自分自身を統制集合に入れると偏相関は定義上 0 になる
                Cf = [c for cn, c in zip(CTRL[tname], CV[tname]) if cn != f]
                for c in Cf:
                    m &= np.isfinite(c)
                m[np.arange(len(m)) % stride != 0] = False
                cpar = np.nan
                if m.sum() > 200:
                    rx = resid(x, Cf, m)
                    ry = resid(y, Cf, m)
                    cpar, _ = corr(rx, ry)
                # 標本外(前半で符号を決め、後半だけで評価)
                m1 = m & is_h1
                m2 = m & ~is_h1
                oos = np.nan
                beta = np.nan
                if m1.sum() > 200 and m2.sum() > 200:
                    s1 = x[m1].std()
                    if s1 > 0:
                        z1 = (x[m1] - x[m1].mean()) / s1
                        b1 = float(np.polyfit(z1, y[m1], 1)[0])   # bp / 1SD
                        z2 = (x[m2] - x[m1].mean()) / s1
                        pred = b1 * z2
                        oos, _ = corr(pred, y[m2])
                        beta = b1
                ci = 1.96 / np.sqrt(max(n - 3, 1))
                rows.append({"feature": f, "label": FEATS[f], "target": tname,
                             "h_grid": h, "horizon": hl, "n": n,
                             "corr_fwd": cf, "corr_bwd": cb, "corr_placebo": cp,
                             "corr_placebo_day": cpd,
                             "corr_partial": cpar, "ci95": ci,
                             "beta_bp_per_sd": beta, "oos_corr": oos,
                             "signed": f in SIGNED})
    R = pl.DataFrame(rows)
    R.write_csv(ROOT / "data" / f"impact_signal_{tag}.csv")

    # Impact(Q) の平均形(図示用)
    cur = []
    for q, tg in ((0.5, "q05"), (5.0, "q5"), (50.0, "q50")):
        for s, lab in (("a", "買い(ask を削る)"), ("b", "売り(bid を削る)")):
            for nm in ("imp", "slp", "lv", "mimp", "mdep", "gfr"):
                v = d[f"{nm}_{s}_{tg}"].to_numpy()
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                cur.append({"q": q, "side": lab, "metric": nm, "n": len(v),
                            "p25": float(np.percentile(v, 25)),
                            "median": float(np.median(v)),
                            "mean": float(v.mean()),
                            "p75": float(np.percentile(v, 75)),
                            "p99": float(np.percentile(v, 99))})
    pl.DataFrame(cur).write_csv(ROOT / "data" / f"impact_curve_{tag}.csv")

    hs = float(np.nanmedian(d["spread_bp"].to_numpy())) / 2
    print(f"\n[基準] スプレッドの中央値 {hs*2:.3f} bp(ハーフ {hs:.3f} bp)",
          file=sys.stderr)
    print("\n=== 前向き相関の絶対値が大きい順(上位 20)===", file=sys.stderr)
    top = (R.filter(pl.col("corr_fwd").is_not_null())
           .with_columns(a=pl.col("corr_fwd").abs()).sort("a", descending=True))
    with pl.Config(tbl_rows=25, tbl_cols=12, fmt_str_lengths=30):
        print(top.head(20).select("feature", "target", "horizon", "n", "corr_fwd",
                                  "corr_bwd", "corr_placebo", "corr_partial",
                                  "oos_corr", "beta_bp_per_sd"), file=sys.stderr)
    print(f"\n-> data/impact_signal_{tag}.csv / impact_curve_{tag}.csv",
          file=sys.stderr)


if __name__ == "__main__":
    main()
