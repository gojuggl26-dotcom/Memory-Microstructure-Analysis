"""夜間(米国が閉まっている間)に、xyz:MU の値段を動かしているのはどこか。

    uv run python scripts/build_nightdisc.py --coin xyz:MU

入力: data/_liqmid_xyz_MU.npz(1 秒 mid)/ data/cash_*_1h.csv / data/pricedisc_panel_*.csv
出力: data/night_marks_xyz_MU.csv  30 分刻みの「その時刻に perp が跳ねるか」の全景
      data/night_explain_xyz_MU.csv  夜の値動きをアジア現物がどれだけ説明するか
      data/night_lead_xyz_MU.csv     夜間の時間内リードラグ
      data/night_open_xyz_MU.csv     MU の寄り値への増分寄与(統制を全部入れた版)

## 夜間とは、アジアの取引時間のことである

米国の時間外のうち **00:00〜08:00 UTC の 8 時間は米国現物がどこでも動かない**。
そしてこの 8 時間は、そのままアジアの取引時間である
(標本期間 5〜9 月はアジアに夏時間が無いので時刻は固定)。

    00:00 東京・ソウル 寄り / 01:00 台北 寄り / 02:30 東京 前引け /
    03:30 東京 後場寄り / 05:30 台北 引け / 06:00 東京 引け /
    06:30 ソウル 引け / 07:00 欧州 寄り / 08:00 ロンドン寄り・米プレ開始

メモリ半導体の同業(サムスン電子・SK ハイニックス・キオクシア・TSMC)はここで動く。

## 4 つの検定

    A  1 日 48 個の 30 分刻みすべてで「その瞬間に perp が跳ねるか」を測り、
       市場の寄り引けが背景から浮くかを見る(恣意的な対照時刻を選ばない)
    B  夜の値動き(00:00→08:00 UTC)そのものを、アジア現物がどれだけ説明するか
    C  夜間の 1 時間ごとに perp とアジア現物のどちらが先か(前後のラグを明示)
    D  MU の寄り値への増分寄与。**MU 自身の 00:00→08:00 の飛びを統制に入れる**
       (入れないと「夜の perp が効く」という偽の結論が出る。前レポートで踏んだ罠)

## 時間契約

検定 A は事象時刻の前後を分けて測るだけ。検定 B / C はラグを明示する。
検定 D は説明変数がすべて寄り付き(13:30 UTC)より前。
`shift(-k)` は目的変数にしか使わない。

## 既知の落とし穴

* アジア現物は現地通貨建て。為替は別物として混ざる(6 時間で 10〜30bp 程度)。
* 各市場は独自の休場日を持つ。日付は市場ごとに突合する。
* 1 時間足どうしのリードラグは**足の窓の取り方を 1 時間間違えると符号が入れ替わる**。
  下では「アジアの足 i」= [ts_i, ts_i+1h) と定義し、perp も同じ窓で取る。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_liq_impact import midgrid  # noqa: E402
from build_pricedisc import bar, ftest, ols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
H = 3600
OPEN_UTC = 13 * H + 1800

MARKS = {0: "東京・ソウル 寄り", 1 * H: "台北 寄り", 2 * H + 1800: "東京 前引け",
         3 * H + 1800: "東京 後場寄り", 5 * H + 1800: "台北 引け",
         6 * H: "東京 引け", 6 * H + 1800: "ソウル 引け", 7 * H: "欧州 寄り",
         8 * H: "ロンドン寄り/米プレ", OPEN_UTC: "米国 寄り", 20 * H: "米国 引け"}
ASIA = {"サムスン電子": "005930_KS", "SK ハイニックス": "000660_KS",
        "キオクシア": "285A_T", "東京エレクトロン": "8035_T", "TSMC": "2330_TW"}


def hourly(tag: str):
    t = pl.read_csv(D / f"cash_{tag}_1h.csv").sort("ts")
    return (t["ts"].to_numpy(), t["open"].to_numpy(), t["close"].to_numpy())


def sess_ret(tag: str, dsec: np.ndarray) -> np.ndarray:
    """その市場の当日のセッション内リターン(最初の足の始値 → 最後の足の終値)。"""
    ts, o, c = hourly(tag)
    day = (ts // 86400) * 86400
    out = np.full(dsec.size, np.nan)
    idx = {}
    for i, d in enumerate(day):
        idx.setdefault(int(d), [i, i])[1] = i
    for k, d in enumerate(dsec):
        v = idx.get(int(d))
        if v and o[v[0]] and c[v[1]]:
            out[k] = np.log(c[v[1]] / o[v[0]]) * 1e4
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    mid, t0 = midgrid(tag)
    lg = np.log(mid)

    p = pl.read_csv(D / f"pricedisc_panel_{tag}.csv")
    dsec = p["day_sec"].to_numpy().astype(np.int64)
    prev = p["prev_day_sec"].to_numpy()
    us_days = set(dsec.tolist())
    alld = np.arange((t0 // 86400) * 86400, t0 + mid.size, 86400)
    wk = np.array([d for d in alld if d not in us_days])

    # ---------- A. 1 日 48 個の刻み全部で跳ねを測る -----------------------
    def med(days, sec, d0, d1):
        i = np.asarray(days, np.int64) + sec - t0
        ok = (i + min(d0, d1) >= 0) & (i + max(d0, d1) < lg.size)
        if ok.sum() < 5:
            return np.nan
        j = i[ok]
        return float(np.nanmedian(np.abs(lg[j + d1] - lg[j + d0]) * 1e4))

    rows = []
    for s in range(0, 86400, 1800):
        rows.append({"sec": s, "utc": f"{s//3600:02d}:{s%3600//60:02d}",
                     "mark": MARKS.get(s, ""),
                     "jump1": med(dsec, s, 0, 1), "jump1_wk": med(wk, s, 0, 1),
                     "pre60": med(dsec, s, -60, 0), "post60": med(dsec, s, 0, 60),
                     "pre300": med(dsec, s, -300, 0), "post300": med(dsec, s, 0, 300)})
    mk = pl.DataFrame(rows).with_columns(
        asym60=pl.col("post60") / pl.col("pre60"),
        asym300=pl.col("post300") / pl.col("pre300"))
    mk.write_csv(D / f"night_marks_{tag}.csv")
    base = float(np.nanmedian(np.abs(np.diff(lg)) * 1e4))
    print(f"[A] 1 日 48 個の 30 分刻みで「その瞬間の跳ね」を測る"
          f"(1 秒の |Δmid| の全体中央値は {base:.3f}bp)")
    q = mk.filter(pl.col("mark") != "")
    with pl.Config(tbl_rows=20, tbl_width_chars=170, float_precision=2):
        print(q.select("utc", "mark", "jump1", "jump1_wk", "pre60", "post60",
                       "asym60", "asym300"))
    o_ = mk.filter(pl.col("mark") == "")
    print(f"    市場の刻みでない {o_.height} 点の背景: jump1 中央 "
          f"{float(o_['jump1'].median()):.2f}bp / 90% 点 "
          f"{float(o_['jump1'].quantile(0.9)):.2f}bp / 最大 {float(o_['jump1'].max()):.2f}bp"
          f" | asym60 中央 {float(o_['asym60'].median()):.2f} / 90% 点 "
          f"{float(o_['asym60'].quantile(0.9)):.2f} / 最大 {float(o_['asym60'].max()):.2f}")
    # ★恣意的な対照時刻を選ばず、48 点の中での順位で見る
    print("    48 点の中での順位(1 = 最大):")
    for col in ("jump1", "asym60", "asym300"):
        r = mk.with_columns(rk=pl.col(col).rank("min", descending=True))
        z = r.filter(pl.col("mark") != "").select("utc", "mark", "rk", col)
        txt = " / ".join(f"{a}{int(b)}位" for a, b in zip(z["mark"], z["rk"]))
        print(f"      {col:<8}: {txt}")

    # ---------- B. 夜の値動きそのものをアジアで説明する --------------------
    def win(s0, s1, days=dsec):
        i0 = np.asarray(days, np.int64) + s0 - t0
        i1 = np.asarray(days, np.int64) + s1 - t0
        ok = (i0 >= 0) & (i1 < lg.size) & (i1 >= 0)
        v = np.full(len(i0), np.nan)
        v[ok] = (lg[i1[ok]] - lg[i0[ok]]) * 1e4
        return v

    Pn = win(0, 8 * H)                         # 夜の perp(00:00 → 08:00 UTC)
    KOa = sess_ret("005930_KS", dsec)
    KOb = sess_ret("000660_KS", dsec)
    JPa = sess_ret("285A_T", dsec)
    JPb = sess_ret("8035_T", dsec)
    TW = sess_ret("2330_TW", dsec)
    nq = pl.read_csv(D / "cash_NQF_1h.csv")
    nk = pl.read_csv(D / "cash_NKDF_1h.csv")
    Nn = np.log(bar(nq, dsec, 7 * H, "close")
                / bar(nq, prev, 23 * H, "close")) * 1e4
    Kn = np.log(bar(nk, dsec, 7 * H, "close")
                / bar(nk, prev, 23 * H, "close")) * 1e4

    print("\n[B] 夜の値動き(perp の 00:00→08:00 UTC)を何が説明するか")
    V = np.c_[Pn, KOa, KOb, JPa, JPb, TW, Nn, Kn]
    keep = np.all(np.isfinite(V), axis=1)
    yk = Pn[keep]
    print(f"   全変数がそろう {int(keep.sum())} 営業日 / "
          f"夜の値動きの標準偏差 {np.std(yk):.0f}bp")
    MB = {
        "指数先物だけ(NQ+日経)": (np.c_[Nn, Kn][keep], ["nq", "nk"]),
        "アジアのメモリ現物だけ": (np.c_[KOa, KOb, JPa, JPb, TW][keep],
                        ["ko_ss", "ko_hx", "jp_kx", "jp_tel", "tw_tsmc"]),
        "両方": (np.c_[Nn, Kn, KOa, KOb, JPa, JPb, TW][keep],
               ["nq", "nk", "ko_ss", "ko_hx", "jp_kx", "jp_tel", "tw_tsmc"]),
    }
    RB = {k: ols(yk, X, n) for k, (X, n) in MB.items()}
    for k, r in RB.items():
        cs = " ".join(f"{v}={r['coef'][v]:+.2f}({r['t'][v]:+.1f})"
                      for v in r["coef"] if v != "const")
        print(f"   {k:<22} R²={r['r2']:.4f} adjR²={r['adj_r2']:.4f}  {cs}")
    f_, pv_, q_ = ftest(RB["指数先物だけ(NQ+日経)"], RB["両方"])
    print(f"     指数先物にアジア現物を足す: ΔR² "
          f"{RB['両方']['r2']-RB['指数先物だけ(NQ+日経)']['r2']:+.4f} "
          f"F({q_},{RB['両方']['n']-RB['両方']['k']})={f_:.2f} p={pv_:.4g}")
    f2, pv2, q2 = ftest(RB["アジアのメモリ現物だけ"], RB["両方"])
    print(f"     アジア現物に指数先物を足す: ΔR² "
          f"{RB['両方']['r2']-RB['アジアのメモリ現物だけ']['r2']:+.4f} "
          f"F({q2},{RB['両方']['n']-RB['両方']['k']})={f2:.2f} p={pv2:.4g}")
    # プラセボ: アジアを 1 営業日ずらす
    Z = np.c_[Nn, Kn][keep]
    Zp = np.c_[Z, np.roll(np.c_[KOa, KOb, JPa, JPb, TW][keep], 1, axis=0)]
    rp = ols(yk, Zp, ["nq", "nk", "a", "b", "c", "d", "e"])
    print(f"     プラセボ(アジアを 1 営業日ずらす): ΔR² "
          f"{rp['r2']-RB['指数先物だけ(NQ+日経)']['r2']:+.4f}")
    pl.DataFrame([{"model": k, "n": r["n"], "r2": r["r2"], "adj_r2": r["adj_r2"],
                   **{f"t_{v}": r["t"].get(v) for v in
                      ("nq", "nk", "ko_ss", "ko_hx", "jp_kx", "jp_tel", "tw_tsmc")}}
                  for k, r in RB.items()]).write_csv(D / f"night_explain_{tag}.csv")

    # ---------- C. 夜間の時間内リードラグ(窓をそろえる) ------------------
    print("\n[C] 夜の 1 時間ごと — アジアの足 i =[ts_i, ts_i+1h) と同じ窓で perp を取る")
    lead = []
    NIGHT = dict(ASIA)
    NIGHT["NQ 先物"] = "NQF"
    NIGHT["日経先物"] = "NKDF"
    for nm, tg in NIGHT.items():
        ts, o, c = hourly(tg)
        if tg in ("NQF", "NKDF"):            # 先物は夜の窓だけに絞る
            k = ((ts % 86400) // 3600) < 8
            ts, o, c = ts[k], o[k], c[k]
        day = (ts // 86400) * 86400
        # アジアの足 i のリターン(始値 → 終値。窓は [ts_i, ts_i+1h))
        ra = np.log(c / o) * 1e4
        pa = win(0, H, ts)                       # 同じ窓の perp
        pb = win(-H, 0, ts)                      # 1 つ前の 1 時間の perp
        pc = win(H, 2 * H, ts)                   # 1 つ後の 1 時間の perp
        nxt = np.r_[(day[1:] == day[:-1]) & (ts[1:] - ts[:-1] == H), False]
        prv = np.r_[False, (day[1:] == day[:-1]) & (ts[1:] - ts[:-1] == H)]
        ra_next = np.where(nxt, np.r_[ra[1:], np.nan], np.nan)

        def cc(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            return (float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum()))
        c0, n0 = cc(ra, pa)
        c_pl, n1 = cc(np.where(prv, ra, np.nan), pc)     # perp が 1 時間遅れて追う
        c_al, n2 = cc(np.where(nxt, pa, np.nan), ra_next)  # アジアが 1 時間遅れて追う
        c_pre, n3 = cc(ra, pb)                            # perp が 1 時間先に動く
        lead.append({"market": nm, "n": n0, "corr_same": c0,
                     "corr_perp_lags": c_pl, "corr_asia_lags": c_al,
                     "corr_perp_leads": c_pre})
        print(f"   {nm:<14} n={n0:>4}  同じ 1 時間 {c0:+.3f} | "
              f"perp が次の 1 時間で追う {c_pl:+.3f} | "
              f"アジアが次の 1 時間で追う {c_al:+.3f} | "
              f"perp の前の 1 時間 {c_pre:+.3f}")
    pl.DataFrame(lead).write_csv(D / f"night_lead_{tag}.csv")

    # ---------- D. MU の寄り値への増分寄与(統制を全部入れる) --------------
    print("\n[D] MU の寄り値への増分寄与 — MU 自身の 00:00→08:00 の飛びを統制に入れる")
    h1 = pl.read_csv(D / "cash_MU_1h.csv")
    y = p["R_cash_on"].to_numpy()
    A = np.log(bar(h1, prev, 23 * H, "close") / p["cash_prev_close"].to_numpy()) * 1e4
    X0 = np.log(bar(h1, dsec, 8 * H, "open") / bar(h1, prev, 23 * H, "close")) * 1e4
    Nm = np.log(bar(nq, dsec, 12 * H, "close") / bar(nq, dsec, 7 * H, "close")) * 1e4
    Pm = np.log(p["pre_1330"].to_numpy() / bar(h1, dsec, 8 * H, "open")) * 1e4
    KO = np.nanmean(np.c_[KOa, KOb], axis=1)
    JP = np.nanmean(np.c_[JPa, JPb], axis=1)
    V2 = np.c_[y, A, X0, Nn, Nm, KO, JP, Pn, Pm]
    k2 = np.all(np.isfinite(V2), axis=1)
    yk2, Ak, X0k, Nnk, Nmk, KOk, JPk, Pnk, Pmk = (v[k2] for v in V2.T)
    print(f"   全変数がそろう {int(k2.sum())} 営業日")
    MD = {
        "D1 MU 自身(アフター+夜の飛び)+先物": (np.c_[Ak, X0k, Nnk, Nmk],
                                  ["A", "X0", "Nn", "Nm"]),
        "D2 +アジア現物": (np.c_[Ak, X0k, Nnk, Nmk, KOk, JPk],
                      ["A", "X0", "Nn", "Nm", "KO", "JP"]),
        "D3 +夜の perp": (np.c_[Ak, X0k, Nnk, Nmk, Pnk], ["A", "X0", "Nn", "Nm", "Pn"]),
        "D4 D1+米プレ": (np.c_[Ak, X0k, Nnk, Nmk, Pmk], ["A", "X0", "Nn", "Nm", "Pm"]),
        "D5 D4+アジア+夜の perp": (np.c_[Ak, X0k, Nnk, Nmk, Pmk, KOk, JPk, Pnk],
                            ["A", "X0", "Nn", "Nm", "Pm", "KO", "JP", "Pn"]),
    }
    RD = {k: ols(yk2, X, n) for k, (X, n) in MD.items()}
    for k, r in RD.items():
        cs = " ".join(f"{v}={r['coef'][v]:+.2f}({r['t'][v]:+.1f})"
                      for v in r["coef"] if v != "const")
        print(f"   {k:<26} R²={r['r2']:.4f}  {cs}")
    for s_, b_, nm in (("D1 MU 自身(アフター+夜の飛び)+先物", "D2 +アジア現物", "アジア現物を足す"),
                       ("D1 MU 自身(アフター+夜の飛び)+先物", "D3 +夜の perp", "夜の perp を足す"),
                       ("D4 D1+米プレ", "D5 D4+アジア+夜の perp", "米プレの上に両方を足す")):
        f_, pv_, q_ = ftest(RD[s_], RD[b_])
        print(f"     {nm:<22}: ΔR² {RD[b_]['r2']-RD[s_]['r2']:+.4f} "
              f"F({q_},{RD[b_]['n']-RD[b_]['k']})={f_:6.2f} p={pv_:.4g}")
    pl.DataFrame([{"model": k, "n": r["n"], "r2": r["r2"],
                   **{f"b_{v}": r["coef"].get(v) for v in
                      ("A", "X0", "Nn", "Nm", "KO", "JP", "Pm", "Pn")},
                   **{f"t_{v}": r["t"].get(v) for v in
                      ("A", "X0", "Nn", "Nm", "KO", "JP", "Pm", "Pn")}}
                  for k, r in RD.items()]).write_csv(D / f"night_open_{tag}.csv")


if __name__ == "__main__":
    main()
