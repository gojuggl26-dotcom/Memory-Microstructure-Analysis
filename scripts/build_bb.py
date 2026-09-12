"""ボリンジャーバンドの三本線と OBI / OFI の関係を検証する。

    uv run python scripts/build_bb.py --coin xyz:MU

入力: data/_liqmid_<coin>.npz(1 秒 mid)/ data/feat200_<coin>/dt=*.parquet
出力: data/bb_<coin>.parquet        1 分足のパネル(バンド + OBI/OFI + 前向き)
      data/bb_profile_<coin>.csv    帯位置 z ごとの OBI / OFI の姿
      data/bb_touch_<coin>.csv      バンドに触れた前後の道筋(イベントスタディ)
      data/bb_cross_<coin>.csv      三本線の突破 — 特徴量は閾値になるか

## 設定(指示された画面のとおり)

    期間 20 / ベース MA = SMA / ソース = 終値 / 標準偏差 2 / オフセット 0
    時間足 = チャート(= 1 分)/ 時間足の確定を待つ ✓

    MA_t = mean(close[t-19..t]),  sd_t = std(close[t-19..t]),
    upper = MA + 2 sd,  lower = MA - 2 sd,  z_t = (close_t - MA_t) / sd_t

「確定を待つ」は、評価をすべて**足の終値の時点**で行うことで満たしている。
バンドに当日の終値を含める形(TradingView の既定)と、**前の足までで閉じる形**
(`--confirm-lag 1`)の両方を計算し、結論が変わらないことを確かめる。

## 使うデータ

OBI / OFI は板の量なので、**原資産(Yahoo の 1 分足)では計算できない**。
本検証は perp 側の 1 分足で行う。標本は板がある 2026-05-04 〜 08-09 の 98 日。

    OBI = (qb - qa)/(qb + qa)     状態。足の終値時点の値と、足の中の平均
    OFI = Cont-Kukanov-Stoikov    流量。足の 1 分間の合計(200ms 格子の和)

どちらも `build_acf_200ms.py` が 200ms 格子に載せたもので、格子点 T の値は
**T までの情報だけ**で決まる。

## 時間契約

* バンドも OBI / OFI も、足 t の**終値の時点**で確定する。
* 目的変数は `(t, t+k]` の前向きリターンのみ。`shift(-k)` は目的変数だけに使う。
* OFI の標準化は**後ろ向き 60 分窓**でのみ行う(全標本の平均・分散は使わない)。

## ★この検証で避けた落とし穴

1. **同語反復**: 「上のバンドに近い」は「直前に上がった」とほぼ同義なので、
   OFI が正なのは当たり前。**直前 20 分のリターンで回帰した残差**でも見る。
2. **夜の静けさ**: 板が動かない分の終値は直前の持ち越しになり、sd が潰れて
   偽の「バンド突破」が出る。**板の更新が 1 件以上ある足だけ**を使い、
   さらに立会時間と時間外を分けて出す。
3. **突破の判定が実装可能か**: 突破は足 t の終値で判定し、目的変数は t より後。
   「あとから見ればあの足で抜けていた」という判定はしない。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_liq_impact import midgrid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
MIN = 60
NMIN = 1440
US0, US1 = 13 * 3600 + 1800, 20 * 3600      # 立会時間(UTC・標本は全て米国夏時間)


def minute_bars(mid: np.ndarray, t0: int) -> pl.DataFrame:
    """1 秒 mid から 1 分足を作る。足 t は [60t, 60t+60)。"""
    n = (mid.size // MIN) * MIN
    m = mid[:n].reshape(-1, MIN)
    fin = np.isfinite(m)
    cnt = fin.sum(1)
    with np.errstate(invalid="ignore"):
        op = np.where(cnt > 0, m[:, 0], np.nan)
        cl = np.where(cnt > 0, np.nanmax(np.where(fin, np.arange(MIN), -1), axis=1), -1)
    idx = np.clip(cl.astype(int), 0, MIN - 1)
    close = np.where(cnt > 0, m[np.arange(m.shape[0]), idx], np.nan)
    hi = np.where(cnt > 0, np.nanmax(np.where(fin, m, -np.inf), axis=1), np.nan)
    lo = np.where(cnt > 0, np.nanmin(np.where(fin, m, np.inf), axis=1), np.nan)
    ts = t0 + np.arange(m.shape[0], dtype=np.int64) * MIN
    return pl.DataFrame({"ts": ts, "open": op, "high": hi, "low": lo,
                         "close": close, "n_sec": cnt.astype(np.int32)})


def minute_feats(tag: str) -> pl.DataFrame:
    """200ms 格子の OBI / OFI を 1 分に畳む。OBI は終端の値、OFI は合計。"""
    out = []
    for f in sorted(glob.glob(str(D / f"feat200_{tag}" / "dt=*.parquet"))):
        t = pl.read_parquet(f, columns=["ts", "obi", "ofi", "n_ev"])
        # ts は区間の終端。足 t =[60t, 60t+60) に入るのは終端が (60t, 60t+60]
        t = t.with_columns(m=((pl.col("ts").cast(pl.Int64) - 1) // 10 ** 9 // MIN) * MIN)
        g = (t.group_by("m").agg(pl.col("obi").drop_nulls().last().alias("obi_close"),
                                 pl.col("obi").mean().alias("obi_mean"),
                                 pl.col("ofi").sum().alias("ofi_sum"),
                                 pl.col("ofi").abs().sum().alias("ofi_abs"),
                                 pl.col("n_ev").sum().alias("n_ev"))
             .rename({"m": "ts"}))
        out.append(g)
    return pl.concat(out).unique(subset="ts", keep="last").sort("ts")


def roll(x: np.ndarray, w: int, fn) -> np.ndarray:
    s = pl.Series(x)
    return fn(s, w).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--period", type=int, default=20)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--suffix", default="", help="出力名の後ろに付ける")
    ap.add_argument("--confirm-lag", type=int, default=0,
                    help="1 にすると当日の足を band に入れない(確定待ちの厳密版)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_") + a.suffix
    rng = np.random.default_rng(0)

    mid, t0 = midgrid(a.coin.replace(":", "_"))
    b = minute_bars(mid, t0)
    fe = minute_feats(a.coin.replace(":", "_"))
    p = b.join(fe, on="ts", how="left").sort("ts")

    # ---- ボリンジャーバンド ------------------------------------------------
    c = p["close"].to_numpy()
    L = a.confirm_lag
    s = pl.Series(np.r_[[np.nan] * L, c[:c.size - L]] if L else c)
    ma = s.rolling_mean(a.period, min_samples=a.period).to_numpy()
    sd = s.rolling_std(a.period, min_samples=a.period, ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (c - ma) / sd
    p = p.with_columns(ma=pl.Series(ma), sd=pl.Series(sd), z=pl.Series(z),
                       upper=pl.Series(ma + a.k * sd), lower=pl.Series(ma - a.k * sd),
                       bw_bp=pl.Series(2 * a.k * sd / ma * 1e4))

    # ---- 前向き / 後ろ向きリターンと OFI の後ろ向き標準化 -------------------
    lc = np.log(c)
    def sh(k):
        o = np.full(lc.size, np.nan)
        if k > 0:
            o[:lc.size - k] = lc[k:]
        elif k < 0:
            o[-k:] = lc[:lc.size + k]
        else:
            o[:] = lc
        return o
    of = p["ofi_sum"].to_numpy()
    ofs = pl.Series(of)
    m60 = ofs.rolling_mean(60, min_samples=30).to_numpy()
    s60 = ofs.rolling_std(60, min_samples=30, ddof=0).to_numpy()
    p = p.with_columns(
        ret_1m=pl.Series((lc - sh(-1)) * 1e4),
        ret_5m=pl.Series((lc - sh(-5)) * 1e4),
        ret_20m=pl.Series((lc - sh(-20)) * 1e4),
        fwd_1m=pl.Series((sh(1) - lc) * 1e4),
        fwd_5m=pl.Series((sh(5) - lc) * 1e4),
        fwd_15m=pl.Series((sh(15) - lc) * 1e4),
        ofi_z=pl.Series(np.where(s60 > 0, (of - m60) / s60, np.nan)),
        sec=pl.col("ts") % 86400,
    ).with_columns(us=(pl.col("sec") >= US0) & (pl.col("sec") < US1),
                   dt=pl.from_epoch("ts", time_unit="s").dt.strftime("%Y-%m-%d"))

    # ★板が動いていない足は落とす(持ち越しの終値で偽の突破が出る)
    ok = (pl.col("n_ev") > 0) & pl.col("z").is_finite() & (pl.col("sd") > 0) \
        & (pl.col("n_sec") >= 30)
    p = p.with_columns(usable=ok)
    p.write_parquet(D / f"bb_{tag}.parquet", compression="zstd")
    q = p.filter(ok)
    print(f"[bb] 1 分足 {p.height:,} 本 / 使える足 {q.height:,} 本 "
          f"({q.height/p.height*100:.1f}%) / {q['dt'].n_unique()} 日", file=sys.stderr)
    print(f"[bb] 帯の幅 中央 {float(q['bw_bp'].median()):.0f}bp "
          f"(立会 {float(q.filter(pl.col('us'))['bw_bp'].median()):.0f}bp / "
          f"時間外 {float(q.filter(~pl.col('us'))['bw_bp'].median()):.0f}bp)", file=sys.stderr)
    print(f"[bb] |z|>2 の足 {float((q['z'].abs() > a.k).mean())*100:.2f}% "
          f"(正規なら 4.55%)", file=sys.stderr)

    # ---- A. 帯位置 z ごとの OBI / OFI の姿 --------------------------------
    EDG = np.array([-9, -2.5, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 2.5, 9])
    zz = q["z"].to_numpy()
    g = np.digitize(zz, EDG) - 1
    r20 = q["ret_20m"].to_numpy()
    ofz = q["ofi_z"].to_numpy()
    obi = q["obi_close"].to_numpy()
    # ★同語反復を外す: 直前 20 分のリターンで説明できる分を引いた残差
    def resid(y, x):
        m = np.isfinite(y) & np.isfinite(x)
        A = np.c_[np.ones(m.sum()), x[m]]
        bta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        o = np.full(y.size, np.nan)
        o[m] = y[m] - A @ bta
        return o
    ofz_r = resid(ofz, r20)
    obi_r = resid(obi, r20)
    rows = []
    for i in range(EDG.size - 1):
        k = g == i
        if k.sum() < 200:
            continue
        rows.append({"z_lo": EDG[i], "z_hi": EDG[i + 1], "n": int(k.sum()),
                     "z_mid": float(np.nanmedian(zz[k])),
                     "obi": float(np.nanmean(obi[k])),
                     "obi_resid": float(np.nanmean(obi_r[k])),
                     "ofi_z": float(np.nanmean(ofz[k])),
                     "ofi_z_resid": float(np.nanmean(ofz_r[k])),
                     "abs_ofi_z": float(np.nanmean(np.abs(ofz[k]))),
                     "ret_1m": float(np.nanmean(q["ret_1m"].to_numpy()[k])),
                     "fwd_5m": float(np.nanmean(q["fwd_5m"].to_numpy()[k]))})
    prof = pl.DataFrame(rows)
    prof.write_csv(D / f"bb_profile_{tag}.csv")
    print("\n[A] 帯の位置 z ごとの OBI / OFI(resid = 直前 20 分のリターンを回帰で抜いた残差)")
    with pl.Config(tbl_rows=20, tbl_width_chars=190, float_precision=3):
        print(prof)

    # ---- B. バンドに触れた前後の道筋(1 分リターンを揃えた対照つき)------
    W = 10
    zs = q["z"].to_numpy()
    tsq = q["ts"].to_numpy()
    r1 = q["ret_1m"].to_numpy()
    f5 = q["fwd_5m"].to_numpy()
    f15 = q["fwd_15m"].to_numpy()
    usq = q["us"].to_numpy()
    # OFI 自身のラグ 1 自己相関 — 「減衰」の基準線
    m_ = np.isfinite(ofz[:-1]) & np.isfinite(ofz[1:]) & (tsq[1:] - tsq[:-1] == MIN)
    print(f"\n[B] 参考: 1 分 OFI のラグ 1 自己相関 = "
          f"{np.corrcoef(ofz[:-1][m_], ofz[1:][m_])[0,1]:+.4f}"
          f"(これが「減衰」の下限。バンド固有の減衰はこれを超えた分だけ)")

    # ret_1m の十分位(全標本・立会/時間外別)。層別の対照に使う
    dec = np.full(q.height, -1)
    for su in (True, False):
        k = usq == su
        v = r1[k]
        qq = np.nanquantile(v[np.isfinite(v)], np.linspace(0, 1, 11)[1:-1])
        dec[k] = np.digitize(r1[k], qq) + (10 if su else 0)

    ev_rows = []
    for nm, cond, sgn in (
            ("上バンド到達", (zs >= a.k) & (np.r_[np.nan, zs[:-1]] < a.k), 1.0),
            ("下バンド到達", (zs <= -a.k) & (np.r_[np.nan, zs[:-1]] > -a.k), -1.0),
            ("中間線を上抜け", (zs >= 0) & (np.r_[np.nan, zs[:-1]] < 0), 1.0),
            ("中間線を下抜け", (zs <= 0) & (np.r_[np.nan, zs[:-1]] > 0), -1.0)):
        idx = np.flatnonzero(np.nan_to_num(cond, nan=0).astype(bool))
        idx = idx[(idx >= W) & (idx + W < q.height)]
        idx = idx[np.array([tsq[i + W] - tsq[i - W] == 2 * W * MIN for i in idx])]
        # 対照: 同じ層(ret_1m 十分位 × 立会/時間外)で、帯の内側(|z|<1)に居た足
        inner = np.flatnonzero((np.abs(zs) < 1.0) & (np.arange(q.height) >= W)
                               & (np.arange(q.height) + W < q.height))
        inner = inner[np.array([tsq[i + W] - tsq[i - W] == 2 * W * MIN for i in inner])]
        wcnt = np.bincount(dec[idx] + 1, minlength=22).astype(float)
        wctl = np.bincount(dec[inner] + 1, minlength=22).astype(float)
        wt = np.where(wctl > 0, wcnt / np.maximum(wctl, 1), 0.0)[dec[inner] + 1]
        for k in range(-W, W + 1):
            v = ofz[idx + k] * sgn
            vc = ofz[inner + k] * sgn
            mv = np.isfinite(vc) & (wt > 0)
            ev_rows.append({
                "event": nm, "n": int(idx.size), "rel_min": k,
                "ofi_mean": float(np.nanmean(v)),
                "ofi_ctl": float(np.average(vc[mv], weights=wt[mv])),
                "obi_mean": float(np.nanmean(obi[idx + k] * sgn)),
                "ret_mean": float(np.nanmean(r1[idx + k] * sgn)),
                "ret_ctl": float(np.average((r1[inner + k] * sgn)[mv], weights=wt[mv])),
            })
    tou = pl.DataFrame(ev_rows).with_columns(
        ofi_excess=pl.col("ofi_mean") - pl.col("ofi_ctl"),
        ret_excess=pl.col("ret_mean") - pl.col("ret_ctl"))
    tou.write_csv(D / f"bb_touch_{tag}.csv")
    print("\n[B1] 到達の前後で OFI(到達の向きに符号をそろえた平均)")
    with pl.Config(tbl_rows=25, tbl_width_chars=190, float_precision=3):
        print(tou.pivot(values="ofi_mean", index="rel_min", on="event").sort("rel_min"))
    print("\n[B2] ★同じ 1 分リターンの層で帯の内側に居た足を引いた「超過」")
    with pl.Config(tbl_rows=25, tbl_width_chars=190, float_precision=3):
        print(tou.pivot(values="ofi_excess", index="rel_min", on="event").sort("rel_min"))
    print("\n[B3] 同じく 1 分リターンの超過(到達の向き。正なら順行が続く)")
    with pl.Config(tbl_rows=25, tbl_width_chars=190, float_precision=3):
        print(tou.pivot(values="ret_excess", index="rel_min", on="event").sort("rel_min"))
    print(tou.group_by("event").agg(pl.col("n").first()).sort("event"))

    # ★「減衰」がどれだけバンド固有か — OFI 自身の持続だけで説明できる分と比べる
    rho = float(np.corrcoef(ofz[:-1][m_], ofz[1:][m_])[0, 1])
    print("\n[B4] 到達の次の分の OFI は、ただの持続(AR(1))より速く落ちるか")
    for nm in ("上バンド到達", "下バンド到達", "中間線を上抜け", "中間線を下抜け"):
        s0 = tou.filter((pl.col("event") == nm) & (pl.col("rel_min") == 0))
        s1 = tou.filter((pl.col("event") == nm) & (pl.col("rel_min") == 1))
        v0, v1 = float(s0["ofi_mean"][0]), float(s1["ofi_mean"][0])
        print(f"   {nm:<14} 到達時 {v0:.3f} → 次の分 {v1:.3f}。"
              f"持続だけなら {rho*v0:.3f} のはず → 実測はその {v1/(rho*v0):.2f} 倍")

    # ---- C. 突破するときの特徴量は閾値になるか ------------------------------
    print("\n[C] 三本線の突破 — 突破した足の OFI は「続くか戻るか」を分けるか")
    base_abs = float(np.nanmean(np.abs(ofz)))
    cro = []
    for nm, lvl, up in (("上バンド突破", a.k, True), ("下バンド突破", -a.k, False),
                        ("中間線 上抜け", 0.0, True), ("中間線 下抜け", 0.0, False)):
        prev = np.r_[np.nan, zs[:-1]]
        cond = (zs >= lvl) & (prev < lvl) if up else (zs <= lvl) & (prev > lvl)
        idx = np.flatnonzero(np.nan_to_num(cond, nan=0).astype(bool))
        idx = idx[(idx + 15 < q.height)]
        idx = idx[tsq[idx + 15] - tsq[idx] == 15 * MIN]
        if idx.size < 100:
            continue
        sgn = 1.0 if up else -1.0
        x = ofz[idx] * sgn
        y5 = f5[idx] * sgn
        y15 = f15[idx] * sgn
        m = np.isfinite(x) & np.isfinite(y5) & np.isfinite(y15)
        x, y5, y15, ix = x[m], y5[m], y15[m], idx[m]
        qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        d5 = np.digitize(x, qs)
        for dd in range(5):
            kk = d5 == dd
            cro.append({"event": nm, "n_all": int(x.size), "quintile": dd + 1,
                        "n": int(kk.sum()), "ofi_z_med": float(np.median(x[kk])),
                        "fwd5_mean": float(np.mean(y5[kk])),
                        "fwd15_mean": float(np.mean(y15[kk])),
                        "p_hold": float((y5[kk] > 0).mean())})
        # AUC(OFI で「5 分後が順方向」を当てられるか)と日ブロック bootstrap
        lab = y5 > 0
        r = np.argsort(np.argsort(x)) + 1
        n1, n0 = lab.sum(), (~lab).sum()
        auc = (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        dayid = (tsq[ix] // 86400)
        ud = np.unique(dayid)
        bs = []
        for _ in range(500):
            pick = rng.choice(ud, ud.size, replace=True)
            sel = np.concatenate([np.flatnonzero(dayid == dv) for dv in pick])
            lb, xb = lab[sel], x[sel]
            if lb.sum() in (0, lb.size):
                continue
            rb = np.argsort(np.argsort(xb)) + 1
            a1, a0 = lb.sum(), (~lb).sum()
            bs.append((rb[lb].sum() - a1 * (a1 + 1) / 2) / (a1 * a0))
        xp = np.roll(ofz, NMIN)[ix] * sgn
        mp = np.isfinite(xp)
        rp2 = np.argsort(np.argsort(xp[mp])) + 1
        lp = lab[mp]
        aucp = ((rp2[lp].sum() - lp.sum() * (lp.sum() + 1) / 2)
                / (lp.sum() * (~lp).sum()))
        print(f"   {nm:<12} n={x.size:>5}  |OFI| 平均 {np.mean(np.abs(x)):.3f}"
              f"(全体 {base_abs:.3f} の {np.mean(np.abs(x))/base_abs:.2f} 倍)"
              f"  AUC {auc:.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]"
              f" / プラセボ {aucp:.3f}"
              f"  5 分後が順方向 {lab.mean()*100:.1f}%")
    cr = pl.DataFrame(cro)
    cr.write_csv(D / f"bb_cross_{tag}.csv")
    print("\n   突破した足の OFI 五分位 → 5 分後のリターン(bp・突破の向き)")
    with pl.Config(tbl_rows=30, tbl_width_chars=170, float_precision=3):
        print(cr.pivot(values="fwd5_mean", index="quintile", on="event").sort("quintile"))
        print(cr.pivot(values="p_hold", index="quintile", on="event").sort("quintile"))


if __name__ == "__main__":
    main()
