"""xyz:MU(24 時間動く perp)に、原資産 MU の寄り付きに対する価格発見能力があるか。

    uv run python scripts/build_pricedisc.py --coin xyz:MU

入力: data/hlcandle_xyz_MU_1h.csv(公開 API)/ data/cash_{MU,NQF,SMH}_{1h,1d}.csv /
      data/_liqmid_xyz_MU.npz(手元の 1 秒 mid。無ければ bbo から作る)
出力: data/pricedisc_panel_xyz_MU.csv     1 行 = 1 営業日
      data/pricedisc_reg_xyz_MU.csv       入れ子回帰の結果
      data/pricedisc_open_xyz_MU.csv      寄り付き前後 1 秒刻みの perp の道筋
      data/pricedisc_gap_xyz_MU.csv       寄り付きで埋まらなかった差とその後

## 何を「価格発見能力」と呼ぶか

「先に動いた」だけでは足りない。perp が単に**プレマーケットの現物**や
**株価指数先物**をなぞっているだけなら、perp 自身は何も発見していない。
そこで 4 段階で切り分ける。

    T1  寄り付きの瞬間(13:30:00 UTC)に perp は跳ねるか(1 秒刻み)
        跳ねる = 寄り値という新しい情報を perp は持っていなかった
    T2  夜間の perp の動きは寄り値を当てるか。**指数先物とプレマーケットを
        統制したうえで**増分 R² がどれだけ残るか
    T3  寄り付きが織り込まなかった差(gap)は、その後 1 時間の現物を当てるか
        当たる = 寄り付きの板より perp のほうが正しかった
    T4  逆に gap は寄り付き直後の perp を動かすか
        動く = perp が寄り値に合わせに行った(現物が先行)

## 時間契約

* 夜間の窓は全銘柄で **[20:00 UTC(前営業日の引け), 13:00 UTC]** に揃える。
  指数先物の足は正時区切りで 13:30 UTC が取れないため、**perp に有利に
  ならないよう**現物寄り付きの 30 分前で全て切る。
* T3 の説明変数 `gap` は寄り値が出た 13:30:00 時点で確定し、
  目的変数は 13:30 以降。T4 も同じ。`shift(-k)` は目的変数にしか使わない。
* 分位・標準化は一切使っていない(全て生の log リターン)。

## 既知の落とし穴

* 時間外の足の high / low には誤気配が混ざる。**open と close しか使わない**。
* `gap` と「寄り → 1 時間後」のリターンは**寄り値の観測誤差を共有する**ので、
  仮説の向きに見かけの相関が乗る。大きさを見積もって本文に書く
  (寄り付きの板寄せ値の誤差 ~1bp に対し gap の散らばりは ~100bp なので
  見かけの傾きは 1e-4 程度で無視できる)。重ならない「2 時間目」でも検算する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_liq_impact import midgrid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
H = 3600
OPEN_UTC = 13 * H + 30 * 60          # 13:30 UTC = 9:30 ET(標本は全て米国夏時間)
CLOSE_UTC = 20 * H


def bp(a, b):
    return np.log(np.asarray(a, float) / np.asarray(b, float)) * 1e4


def bar(df: pl.DataFrame, day: np.ndarray, sec: int, col: str) -> np.ndarray:
    """その日の `sec` 秒(UTC)に始まる足の `col` を引く。無ければ NaN。"""
    m = dict(zip(df["ts"].to_list(), df[col].to_list()))
    return np.array([np.nan if not np.isfinite(d) else m.get(int(d) + sec, np.nan)
                     for d in day], float)


def ols(y, X, names):
    """定数項つき OLS。HC3 の頑健標準誤差つき。"""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    n, k = A.shape
    XtXi = np.linalg.pinv(A.T @ A)
    h = np.einsum("ij,jk,ik->i", A, XtXi, A)
    S = (A * (r / (1 - h))[:, None]).T @ (A * (r / (1 - h))[:, None])
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(r @ r) / sst
    return {"n": int(n), "r2": r2,
            "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k),
            "coef": dict(zip(["const"] + names, b)),
            "se": dict(zip(["const"] + names, se)),
            "t": dict(zip(["const"] + names, b / se)),
            "rss": float(r @ r), "k": k, "mask": m}


def ftest(small, big):
    q = big["k"] - small["k"]
    df = big["n"] - big["k"]
    f = (small["rss"] - big["rss"]) / q / (big["rss"] / df)
    from math import lgamma, log, exp
    # 生存関数を連分数ではなく数値積分で(依存を増やさない)
    x = np.linspace(0, 1, 20001)[1:-1]
    a, b = df / 2, q / 2
    lb = lgamma(a) + lgamma(b) - lgamma(a + b)
    pdf = np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - lb)
    xv = df / (df + q * f)
    p = float(np.trapezoid(pdf[x <= xv], x[x <= xv]))
    return f, p, q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--under", default="MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(0)

    # ---------- 読み込み -------------------------------------------------
    d1 = pl.read_csv(D / f"cash_{a.under}_1d.csv")
    h1 = pl.read_csv(D / f"cash_{a.under}_1h.csv")
    nq = pl.read_csv(D / "cash_NQF_1h.csv")
    smh = pl.read_csv(D / "cash_SMH_1h.csv")
    hp = pl.read_csv(D / f"hlcandle_{tag}_1h.csv")

    assert set((d1["ts"] % 86400).unique().to_list()) == {OPEN_UTC}, \
        "日足の時刻が 13:30 UTC でない日がある(夏時間の切替?)"
    day = (d1["ts"].to_numpy() // 86400) * 86400          # その日の 00:00 UTC
    d1 = d1.with_columns(day=pl.Series(day))

    # ---------- 1 営業日 1 行のパネル ------------------------------------
    o = d1["open"].to_numpy()
    c = d1["close"].to_numpy()
    prev_c = np.r_[np.nan, c[:-1]]
    prev_day = np.r_[np.nan, day[:-1]]

    def at(df, sec, col, dd=day):
        return bar(df, dd, sec, col)

    # ★寄り値は「13:30 UTC の足の始値」を使う。Yahoo の日足の `open` は
    #   16/95 日で 13:30 足の値幅の外に出ており(5 日は完全に外)、その日の値は
    #   直前のプレマーケットの値と一致する。レギュラーの寄り付きの値ではない。
    bar_open = bar(h1, day, OPEN_UTC, "open")
    o_daily = o
    o = np.where(np.isfinite(bar_open), bar_open, np.nan)

    pan = {
        "day": [str(np.datetime64(int(x), "s"))[:10] for x in day],
        "day_sec": day.astype(float), "prev_day_sec": prev_day.astype(float),
        "cash_prev_close": prev_c, "cash_open": o, "cash_open_daily": o_daily,
        "cash_close": c,
        "cash_h1": at(h1, OPEN_UTC, "close"),          # 13:30 足の終値 = 10:30 ET
        "cash_h2": at(h1, OPEN_UTC + H, "close"),      # 14:30 足の終値 = 11:30 ET
        "cash_bar_open": at(h1, OPEN_UTC, "open"),
        "pre_1300": at(h1, 13 * H, "open"),            # 13:00 UTC のプレマーケット
        "pre_1330": at(h1, 13 * H, "close"),           # 寄り付き直前
        "perp_13": at(hp, 12 * H, "close"),            # 12:00 足の終値 = 13:00 UTC
        "perp_14": at(hp, 13 * H, "close"),
        "perp_prev20": at(hp, 19 * H, "close", prev_day),
        "nq_13": at(nq, 12 * H, "close"),
        "nq_prev20": at(nq, 19 * H, "close", prev_day),
        "smh_1300": at(smh, 13 * H, "open"),
        "cash_vol": d1["volume"].to_numpy(),
    }
    p = pl.DataFrame(pan)
    p = p.with_columns(
        R_cash_on=pl.Series(bp(pan["cash_open"], pan["cash_prev_close"])),
        R_perp_on=pl.Series(bp(pan["perp_13"], pan["perp_prev20"])),
        R_pre_on=pl.Series(bp(pan["pre_1300"], pan["cash_prev_close"])),
        R_nq_on=pl.Series(bp(pan["nq_13"], pan["nq_prev20"])),
        R_smh_on=pl.Series(bp(pan["smh_1300"], pan["cash_prev_close"])),
        R_h1=pl.Series(bp(pan["cash_h1"], pan["cash_open"])),
        R_h2=pl.Series(bp(pan["cash_h2"], pan["cash_h1"])),
        R_day=pl.Series(bp(pan["cash_close"], pan["cash_open"])),
    )
    p = p.filter(pl.col("R_cash_on").is_finite() & pl.col("R_perp_on").is_finite()
                 & pl.col("R_nq_on").is_finite() & pl.col("R_pre_on").is_finite())
    print(f"[panel] {p.height} 営業日 {p['day'][0]} 〜 {p['day'][-1]}", file=sys.stderr)

    # 寄り値の 2 つの出どころが合うか(データの検算)
    dd = bp(p["cash_open_daily"], p["cash_open"])
    nbad = int((np.abs(dd) > 20).sum())
    print(f"[検算] 日足の寄り値 と 13:30 足の始値 の差: 中央 {np.nanmedian(np.abs(dd)):.2f}bp "
          f"/ 90% 点 {np.nanpercentile(np.abs(dd), 90):.2f}bp / "
          f"20bp 超え {nbad} 日 → 13:30 足の始値を採用", file=sys.stderr)
    zz = np.abs(dd) > 20
    print(f"        食い違う日では、日足の寄り値は直前のプレマーケットと "
          f"{np.nanmedian(np.abs(bp(p['cash_open_daily'].to_numpy()[zz], p['pre_1330'].to_numpy()[zz]))):.1f}bp、"
          f"13:30 足の始値とは {np.nanmedian(np.abs(dd[zz])):.1f}bp 離れている", file=sys.stderr)

    # ---------- 手元の 1 秒 mid と公開足の突合 ---------------------------
    mid, t0 = midgrid(tag)
    dsec = p["day_sec"].to_numpy().astype(np.int64)
    # ★前「営業日」の引け。前カレンダー日にすると月曜・休場明けが全部壊れる
    prev_dsec = p["prev_day_sec"].to_numpy()

    def m_at(sec_abs):
        s = np.asarray(sec_abs, float)
        i = np.where(np.isfinite(s), s - t0, -1).astype(np.int64)
        ok = (i >= 0) & (i < mid.size)
        v = np.full(i.size, np.nan)
        v[ok] = mid[i[ok]]
        return v

    m13 = m_at(dsec + 13 * H)
    dv = bp(p["perp_13"].to_numpy(), m13)
    ok = np.isfinite(dv)
    print(f"[検算] 公開足の 13:00 終値 vs 手元の 1 秒 mid: {ok.sum()} 日で "
          f"中央 {np.nanmedian(np.abs(dv[ok])):.2f}bp / 90% 点 "
          f"{np.nanpercentile(np.abs(dv[ok]), 90):.2f}bp / 最大 "
          f"{np.nanmax(np.abs(dv[ok])):.1f}bp", file=sys.stderr)

    p = p.with_columns(
        perp_1329=pl.Series(m_at(dsec + OPEN_UTC - 1)),
        perp_1330=pl.Series(m_at(dsec + OPEN_UTC)),
        perp_1331=pl.Series(m_at(dsec + OPEN_UTC + 60)),
        perp_1335=pl.Series(m_at(dsec + OPEN_UTC + 300)),
        perp_1430=pl.Series(m_at(dsec + OPEN_UTC + H)),
        perp_prev20s=pl.Series(m_at(prev_dsec + CLOSE_UTC)),
    )
    p = p.with_columns(
        R_perp_on_s=(pl.col("perp_1329") / pl.col("perp_prev20s")).log() * 1e4,
        R_perp_open1m=(pl.col("perp_1331") / pl.col("perp_1330")).log() * 1e4,
        R_perp_open5m=(pl.col("perp_1335") / pl.col("perp_1330")).log() * 1e4,
        R_perp_h1=(pl.col("perp_1430") / pl.col("perp_1330")).log() * 1e4,
        R_perp_last30=(pl.col("perp_1329") / pl.col("perp_13")).log() * 1e4,
    )
    p = p.with_columns(
        gap=pl.col("R_cash_on") - pl.col("R_perp_on_s"),
        R_pre_on_s=(pl.col("pre_1330") / pl.col("cash_prev_close")).log() * 1e4)
    p.write_csv(D / f"pricedisc_panel_{tag}.csv")

    # ---------- T1: 寄り付きの瞬間に perp は跳ねるか ---------------------
    W = 900
    prof = np.full((p.height, 2 * W + 1), np.nan)
    for i, ds in enumerate(dsec):
        k = ds + OPEN_UTC - t0
        if k - W >= 0 and k + W < mid.size:
            seg = mid[k - W:k + W + 1]
            prof[i] = np.log(seg / seg[W]) * 1e4
    trad = np.isfinite(prof[:, 0]) & np.isfinite(prof[:, -1])
    # 帰無対照: 非立会日(週末・休場)の同じ時刻
    alld = np.arange(t0 // 86400 * 86400, t0 + mid.size, 86400)
    hol = np.array([d for d in alld if d not in set(dsec.tolist())])
    hp_ = []
    for ds in hol:
        k = ds + OPEN_UTC - t0
        if k - W >= 0 and k + W < mid.size:
            seg = mid[k - W:k + W + 1]
            if np.isfinite(seg).all():
                hp_.append(np.log(seg / seg[W]) * 1e4)
    hp_ = np.array(hp_) if hp_ else np.zeros((0, 2 * W + 1))
    op = pl.DataFrame({
        "sec": np.arange(-W, W + 1),
        "abs_med_trading": np.nanmedian(np.abs(prof[trad]), axis=0),
        "abs_med_holiday": (np.nanmedian(np.abs(hp_), axis=0) if len(hp_)
                            else np.full(2 * W + 1, np.nan)),
        "signed_med_trading": np.nanmedian(prof[trad], axis=0),
    })
    op.write_csv(D / f"pricedisc_open_{tag}.csv")
    print(f"\n[T1] 立会日 {int(trad.sum())} 日 / 非立会日 {len(hp_)} 日")
    for s in (-300, -60, -5, 1, 5, 60, 300, 900):
        a_ = np.nanmedian(np.abs(prof[trad][:, W + s]))
        b_ = np.nanmedian(np.abs(hp_[:, W + s])) if len(hp_) else np.nan
        print(f"   13:30 から {s:+5d} 秒: |Δmid| 中央 立会日 {a_:6.2f}bp / "
              f"非立会日 {b_:6.2f}bp / 比 {a_/b_ if b_ else np.nan:5.2f}")
    # ★同じ立会日の中での比較 — 「1 秒で mid がどれだけ動くか」の平常値
    d1s = np.abs(np.diff(np.log(mid)) * 1e4)
    inus = np.zeros(mid.size - 1, bool)
    for ds in dsec:
        k0, k1 = ds + OPEN_UTC - t0, ds + CLOSE_UTC - t0
        if k0 >= 0 and k1 < d1s.size:
            inus[k0:k1] = True
    base1 = np.nanmedian(d1s[inus])
    jump1 = np.nanmedian(np.abs(prof[trad][:, W + 1]))
    print(f"   参考: 立会時間中の 1 秒の |Δmid| の中央値は {base1:.3f}bp。"
          f"寄り付き直後 1 秒は {jump1:.2f}bp で {jump1/base1:.0f} 倍")

    # ---------- T2: 入れ子回帰 -------------------------------------------
    y = p["R_cash_on"].to_numpy()
    Xnq = p["R_nq_on"].to_numpy()[:, None]
    Xpre = p["R_pre_on"].to_numpy()[:, None]
    Xperp = p["R_perp_on"].to_numpy()[:, None]
    Xperp_s = p["R_perp_on_s"].to_numpy()[:, None]
    models = {
        "M1 指数先物のみ": (np.c_[Xnq], ["nq"]),
        "M2 先物+プレ": (np.c_[Xnq, Xpre], ["nq", "pre"]),
        "M3 先物+perp": (np.c_[Xnq, Xperp], ["nq", "perp"]),
        "M4 先物+プレ+perp": (np.c_[Xnq, Xpre, Xperp], ["nq", "pre", "perp"]),
        "M5 perp のみ": (np.c_[Xperp], ["perp"]),
        "M6 プレのみ": (np.c_[Xpre], ["pre"]),
    }
    res = {k: ols(y, X, n) for k, (X, n) in models.items()}
    rows = []
    for k, r in res.items():
        rows.append({"model": k, "n": r["n"], "r2": r["r2"], "adj_r2": r["adj_r2"],
                     **{f"b_{v}": r["coef"].get(v) for v in ("nq", "pre", "perp")},
                     **{f"t_{v}": r["t"].get(v) for v in ("nq", "pre", "perp")}})
    reg = pl.DataFrame(rows)
    reg.write_csv(D / f"pricedisc_reg_{tag}.csv")
    print("\n[T2] 寄り値の夜間リターンを説明する(全て 20:00 UTC → 13:00 UTC の窓)")
    with pl.Config(tbl_width_chars=200, float_precision=3):
        print(reg)
    for s, b, nm in (("M2 先物+プレ", "M4 先物+プレ+perp", "perp を足す"),
                     ("M3 先物+perp", "M4 先物+プレ+perp", "プレを足す"),
                     ("M1 指数先物のみ", "M2 先物+プレ", "プレを足す(perp 無し)")):
        f, pv, q = ftest(res[s], res[b])
        print(f"   {nm:<22}: ΔR² {res[b]['r2']-res[s]['r2']:+.4f}  "
              f"F({q},{res[b]['n']-res[b]['k']})={f:6.2f}  p={pv:.4g}")

    # 期間で前半・後半に割る
    half = p.height // 2
    for nm, sl in (("前半", slice(0, half)), ("後半", slice(half, p.height))):
        r2a = ols(y[sl], np.c_[Xnq[sl], Xpre[sl]], ["nq", "pre"])
        r2b = ols(y[sl], np.c_[Xnq[sl], Xpre[sl], Xperp[sl]], ["nq", "pre", "perp"])
        print(f"   {nm}({r2b['n']} 日): ΔR² {r2b['r2']-r2a['r2']:+.4f} "
              f"/ perp の係数 {r2b['coef']['perp']:+.3f} (t={r2b['t']['perp']:+.2f})")

    # プラセボ: perp を 1 営業日ずらす
    for sh in (1, -1, 5):
        Xp = np.roll(Xperp, sh, axis=0)
        rA = ols(y, np.c_[Xnq, Xpre], ["nq", "pre"])
        rB = ols(y, np.c_[Xnq, Xpre, Xp], ["nq", "pre", "perp"])
        print(f"   プラセボ(perp を {sh:+d} 営業日ずらす): ΔR² {rB['r2']-rA['r2']:+.4f} "
              f"(t={rB['t']['perp']:+.2f})")

    # ★13:29 でそろえた公平な勝負(1 秒 mid がある日だけ)
    ys = p["R_cash_on"].to_numpy()
    Xps = p["R_perp_on_s"].to_numpy()[:, None]
    Xprs = p["R_pre_on_s"].to_numpy()[:, None]
    sub = (np.isfinite(ys) & np.isfinite(Xps[:, 0]) & np.isfinite(Xprs[:, 0])
           & np.isfinite(Xnq[:, 0]))
    print(f"\n[T2b] 13:29:59 でそろえた比較({int(sub.sum())} 日。"
          f"perp は 1 秒 mid、プレは 13:00 足の終値)")
    mm = {"先物+プレ(13:29)": (np.c_[Xnq, Xprs], ["nq", "pre"]),
          "先物+perp(13:29)": (np.c_[Xnq, Xps], ["nq", "perp"]),
          "先物+プレ+perp(13:29)": (np.c_[Xnq, Xprs, Xps], ["nq", "pre", "perp"])}
    rr = {k: ols(ys[sub], X[sub], n) for k, (X, n) in mm.items()}
    for k, r in rr.items():
        cs = " ".join(f"{v}={r['coef'][v]:+.3f}(t={r['t'][v]:+.2f})"
                      for v in r["coef"] if v != "const")
        print(f"   {k:<24} R²={r['r2']:.4f}  {cs}")
    f1, p1_, q1 = ftest(rr["先物+プレ(13:29)"], rr["先物+プレ+perp(13:29)"])
    f2, p2_, q2 = ftest(rr["先物+perp(13:29)"], rr["先物+プレ+perp(13:29)"])
    print(f"   perp を足す: ΔR² "
          f"{rr['先物+プレ+perp(13:29)']['r2']-rr['先物+プレ(13:29)']['r2']:+.4f} "
          f"F={f1:.2f} p={p1_:.4g}")
    print(f"   プレを足す: ΔR² "
          f"{rr['先物+プレ+perp(13:29)']['r2']-rr['先物+perp(13:29)']['r2']:+.4f} "
          f"F={f2:.2f} p={p2_:.4g}")

    # ★どちらの値段が寄り値に近いか(素朴で一番大事な統計)
    print("\n[T2c] 13:29:59 時点の値段は、寄り値をどれだけ言い当てているか"
          "(|現物の夜間リターン − 各市場の夜間リターン|)")
    sub2 = sub
    err = []
    for nm, v in (("perp(1 秒 mid・13:29:59)", p["R_perp_on_s"].to_numpy()),
                  ("プレマーケット(13:29:59)", p["R_pre_on_s"].to_numpy()),
                  ("プレマーケット(13:00)", p["R_pre_on"].to_numpy()),
                  ("perp(公開足・13:00)", p["R_perp_on"].to_numpy()),
                  ("指数先物だけ(β 調整なし)", p["R_nq_on"].to_numpy()),
                  ("何もしない(前日の引け)", np.zeros(p.height))):
        e = (ys - v)[sub2]
        err.append({"source": nm, "n": int(sub2.sum()),
                    "abs_med_bp": float(np.nanmedian(np.abs(e))),
                    "rmse_bp": float(np.sqrt(np.nanmean(e ** 2)))})
        print(f"   {nm:<26} |誤差| 中央 {np.nanmedian(np.abs(e)):6.1f}bp / "
              f"RMSE {np.sqrt(np.nanmean(e**2)):6.1f}bp")
    pl.DataFrame(err).write_csv(D / f"pricedisc_err_{tag}.csv")
    print(f"   参考: 現物の夜間リターンそのものの散らばり "
          f"{np.nanstd(ys[sub2]):.1f}bp(中央 |値| {np.nanmedian(np.abs(ys[sub2])):.1f}bp)")

    # ---------- T3 / T4: 寄り付きで埋まらなかった差 ----------------------
    g = p["gap"].to_numpy()
    out = []
    for nm, yy in (("現物 寄り→1 時間後", p["R_h1"].to_numpy()),
                   ("現物 1 時間後→2 時間後", p["R_h2"].to_numpy()),
                   ("現物 寄り→引け", p["R_day"].to_numpy()),
                   ("perp 13:30→13:31", p["R_perp_open1m"].to_numpy()),
                   ("perp 13:30→13:35", p["R_perp_open5m"].to_numpy()),
                   ("perp 13:30→14:30", p["R_perp_h1"].to_numpy())):
        r = ols(yy, g[:, None], ["gap"])
        bsr = []
        m = r["mask"]
        for _ in range(2000):
            idx = rng.integers(0, int(m.sum()), int(m.sum()))
            rb = ols(yy[m][idx], g[m][idx, None], ["gap"])
            bsr.append(rb["coef"]["gap"])
        lo, hi = np.percentile(bsr, [2.5, 97.5])
        out.append({"y": nm, "n": r["n"], "beta_gap": r["coef"]["gap"],
                    "t": r["t"]["gap"], "lo": lo, "hi": hi, "r2": r["r2"]})
    t34 = pl.DataFrame(out)
    t34.write_csv(D / f"pricedisc_gap_{tag}.csv")
    # 頑健性: 順位相関と、上下 5% を刈り込んだ回帰
    gy = p["R_perp_open1m"].to_numpy()
    mm2 = np.isfinite(g) & np.isfinite(gy)
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    sp = np.corrcoef(rk(g[mm2]), rk(gy[mm2]))[0, 1]
    lo_, hi_ = np.percentile(g[mm2], [5, 95])
    tr = mm2 & (g > lo_) & (g < hi_)
    rt = ols(gy[tr], g[tr, None], ["gap"])
    print(f"\n   [頑健性] gap → perp 13:30→13:31: 順位相関 {sp:+.3f} / "
          f"上下 5% を刈ると β={rt['coef']['gap']:+.3f}(t={rt['t']['gap']:+.2f}, "
          f"n={rt['n']})")
    # プラセボ: gap を 1 営業日ずらす
    for sh in (1, -1):
        rp = ols(gy, np.roll(g, sh)[:, None], ["gap"])
        print(f"   [プラセボ] gap を {sh:+d} 営業日ずらす: β={rp['coef']['gap']:+.3f} "
              f"(t={rp['t']['gap']:+.2f})")

    # 寄り付き前 5 分の perp の動きは、寄り後 1 時間の現物を当てるか
    p5 = p.with_columns(perp_1325=pl.Series(m_at(dsec + OPEN_UTC - 300)))
    x5 = ((p5["perp_1330"] / p5["perp_1325"]).log() * 1e4).to_numpy()
    for nm, yy in (("現物 寄り→1 時間後", p["R_h1"].to_numpy()),
                   ("現物 1 時間後→2 時間後", p["R_h2"].to_numpy())):
        r = ols(yy, x5[:, None], ["x"])
        rpl = ols(yy, np.roll(x5, 1)[:, None], ["x"])
        print(f"   [T5] 寄り 5 分前までの perp の動き → {nm}: β={r['coef']['x']:+.3f} "
              f"(t={r['t']['x']:+.2f}, R²={r['r2']:.3f}, n={r['n']}) / "
              f"プラセボ β={rpl['coef']['x']:+.3f}(t={rpl['t']['x']:+.2f})")
    # 寄り後 1 時間の同時性(予測ではない)
    m3 = np.isfinite(p["R_perp_h1"].to_numpy()) & np.isfinite(p["R_h1"].to_numpy())
    print(f"   [同時点] perp 13:30→14:30 と 現物 寄り→1 時間後 の相関 "
          f"{np.corrcoef(p['R_perp_h1'].to_numpy()[m3], p['R_h1'].to_numpy()[m3])[0,1]:.4f} "
          f"(n={int(m3.sum())}・予測ではない)")

    # ---------- T6: 現物が閉まっている時間に perp は独自に動くか ----------
    # ★1 秒リターンは使わない。8,462,534 秒のうち 34 秒で mid が一瞬だけ
    #   桁違いに飛んで戻る(例 2026-06-14 10:58:37 に 1009 → 198.7 → 1005)。
    #   これは板が一瞬空いた痕跡であって価格情報ではないが、2 乗和は支配される。
    #   5 分リターンで実現分散を作れば潰れる。
    S = 300
    n5 = (mid.size // S) * S
    m5 = mid[:n5].reshape(-1, S)[:, 0]
    r5 = np.diff(np.log(m5)) * 1e4
    sec = (np.arange(r5.size) * S) + t0
    hr = (sec % 86400) // 3600
    dayk = (sec // 86400) * 86400
    is_trade = np.isin(dayk, dsec)
    rv = []
    for h in range(24):
        for nm, m_ in (("立会日", is_trade), ("非立会日", ~is_trade)):
            k = (hr == h) & m_ & np.isfinite(r5)
            if k.sum() > 60:
                rv.append({"hour": h, "kind": nm,
                           "rv_bp": float(np.sqrt(np.nansum(r5[k] ** 2) / (k.sum() / 12))),
                           "n_5m": int(k.sum())})
    rvt = pl.DataFrame(rv)
    rvt.write_csv(D / f"pricedisc_rv_{tag}.csv")
    piv = rvt.pivot(values="rv_bp", index="hour", on="kind")
    print("\n[T6] perp の 1 時間あたり実現ボラティリティ(bp)— 現物の営業とどう関係するか")
    with pl.Config(tbl_rows=30, float_precision=1):
        print(piv.with_columns(比=(pl.col("立会日") / pl.col("非立会日"))))
    tr = rvt.filter(pl.col("kind") == "立会日")
    seg4 = {"現物がどこでも動かない(00〜08)": range(0, 8),
            "プレマーケット(08〜13)": range(8, 13),
            "レギュラー(13〜20)": range(13, 20),
            "アフター(20〜24)": range(20, 24)}
    tot = float(tr["rv_bp"].pow(2).sum())
    print("   1 日の分散の内訳(立会日・時間の境は正時で近似):")
    for nm, hs in seg4.items():
        v = float(tr.filter(pl.col("hour").is_in(list(hs)))["rv_bp"].pow(2).sum())
        print(f"     {nm:<28} RV {v**0.5:5.0f}bp  分散の {v/tot*100:4.1f}%")

    # ---------- T7: 現物が一切動かない 8 時間に perp は何かを見つけるか ----
    # 米国株の時間外は 20:00-00:00 UTC(アフター)と 08:00-13:30 UTC(プレ)。
    # **00:00-08:00 UTC は現物がどこでも取引されない**。この窓で perp が
    # 見つけた情報が、寄り値に残るかを見る。
    q = p.with_columns(
        cash_0000=pl.Series(bar(h1, p["prev_day_sec"].to_numpy(), 23 * H, "close")),
        cash_0800=pl.Series(bar(h1, dsec, 8 * H, "open")),
        nq_0000=pl.Series(bar(nq, p["prev_day_sec"].to_numpy(), 23 * H, "close")),
        nq_0800=pl.Series(bar(nq, dsec, 7 * H, "close")),
        perp_0000=pl.Series(m_at(dsec.astype(float))),
        perp_0800=pl.Series(m_at(dsec + 8 * H)),
    )
    A = (np.log(q["cash_0000"] / q["cash_prev_close"]) * 1e4).to_numpy()      # アフター
    X0 = (np.log(q["cash_0800"] / q["cash_0000"]) * 1e4).to_numpy()           # ★夜を跨いだ現物の飛び
    Nn = (np.log(q["nq_0800"] / q["nq_0000"]) * 1e4).to_numpy()               # 夜の先物
    Pn = (np.log(q["perp_0800"] / q["perp_0000"]) * 1e4).to_numpy()           # 夜の perp
    Pm = (np.log(q["pre_1330"] / q["cash_0800"]) * 1e4).to_numpy()            # プレ
    Nm = (np.log(q["nq_13"] / q["nq_0800"]) * 1e4).to_numpy()                 # 朝の先物
    yy = q["R_cash_on"].to_numpy()
    # ★入れ子モデルは同じ標本で比べないと F 検定が成立しない。
    #   全変数がそろう日だけに絞る。
    allv = np.c_[A, X0, Nn, Nm, Pn, Pm]
    keep = np.isfinite(yy) & np.all(np.isfinite(allv), axis=1)
    yk, Ak, X0k, Nnk, Nmk, Pnk, Pmk = (v[keep] for v in (yy, A, X0, Nn, Nm, Pn, Pm))
    ms = {
        "S1 現物(アフター+夜の飛び)+先物": (np.c_[Ak, X0k, Nnk, Nmk], ["A", "X0", "Nn", "Nm"]),
        "S2 +夜の perp": (np.c_[Ak, X0k, Nnk, Nmk, Pnk], ["A", "X0", "Nn", "Nm", "Pn"]),
        "S3 S1+プレ": (np.c_[Ak, X0k, Nnk, Nmk, Pmk], ["A", "X0", "Nn", "Nm", "Pm"]),
        "S4 S3+夜の perp": (np.c_[Ak, X0k, Nnk, Nmk, Pmk, Pnk],
                           ["A", "X0", "Nn", "Nm", "Pm", "Pn"]),
    }
    rs = {k: ols(yk, Xm, n) for k, (Xm, n) in ms.items()}
    print("\n[T7] 現物がどこでも動かない 00:00〜08:00 UTC の 8 時間に、"
          f"perp が見つけた分は寄り値に残るか(全変数がそろう {int(keep.sum())} 日)")
    for k, r in rs.items():
        cs = " ".join(f"{v}={r['coef'][v]:+.2f}({r['t'][v]:+.1f})"
                      for v in r["coef"] if v != "const")
        print(f"   {k:<28} n={r['n']} R²={r['r2']:.4f}  {cs}")
    for s, b, nm in (("S1 現物(アフター+夜の飛び)+先物", "S2 +夜の perp", "プレ無しの状態から"),
                     ("S3 S1+プレ", "S4 S3+夜の perp", "プレを入れた状態から")):
        f_, p_, _ = ftest(rs[s], rs[b])
        print(f"   夜の perp を足す({nm:<18}): ΔR² {rs[b]['r2']-rs[s]['r2']:+.4f} "
              f"F={f_:.2f} p={p_:.4g}")
    # プラセボ: 夜の perp を 1 営業日ずらす
    for sh in (1, -1):
        rp = ols(yk, np.c_[Ak, X0k, Nnk, Nmk, Pmk, np.roll(Pnk, sh)],
                 ["A", "X0", "Nn", "Nm", "Pm", "Pn"])
        print(f"   プラセボ(夜の perp を {sh:+d} 営業日ずらす): "
              f"ΔR² {rp['r2']-rs['S3 S1+プレ']['r2']:+.4f} (t={rp['t']['Pn']:+.2f})")
    pl.DataFrame([{"model": k, "n": r["n"], "r2": r["r2"],
                   **{f"b_{v}": r["coef"].get(v) for v in ("A", "X0", "Nn", "Nm", "Pn", "Pm")},
                   **{f"t_{v}": r["t"].get(v) for v in ("A", "X0", "Nn", "Nm", "Pn", "Pm")}}
                  for k, r in rs.items()]).write_csv(D / f"pricedisc_seg_{tag}.csv")
    print(f"\n[T3/T4] gap = 現物の夜間リターン − perp の夜間リターン"
          f"(1 秒 mid・13:29:59 まで)を説明変数にする")
    print(f"   gap の散らばり: 標準偏差 {np.nanstd(g):.1f}bp / "
          f"|gap| の中央値 {np.nanmedian(np.abs(g)):.1f}bp")
    with pl.Config(tbl_width_chars=200, float_precision=3):
        print(t34)


if __name__ == "__main__":
    main()
