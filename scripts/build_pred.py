"""OBI / OFI / 攻撃的注文の強度を五分位に分け、予測ホライズンごとのリターンを測る。

    uv run python scripts/build_pred.py --coin xyz:MU

説明変数 (すべて時刻 T の時点で確定する)
----------------------------------------
  obi     板の不均衡  (bid_sz - ask_sz) / (bid_sz + ask_sz)
          T 以前の最後の気配から作る瞬間値
  ofi     注文流の不均衡 (Cont-Kukanov-Stoikov) を (T-1s, T] で合計。枚
  ai_net  攻撃的注文の符号つき数量 = taker 買い - taker 売り。枚
          窓は [T-1.005s, T-0.005s)。**5ms の安全余裕**を取ってあるのは、
          node_fills の時刻がミリ秒精度で板イベント(ns)より最大 ~1ms
          早いことがあるため (CLAUDE.md)。この余裕がないと 100ms
          ホライズンに未来が混じりうる
  ai_rate 攻撃的注文の件数 (符号なしの強度)。同じ窓

なぜ microprice も測るか
------------------------
microprice = mid + (スプレッド/2) x OBI なので、**T 時点で既に判っている**。
「OBI が高いと mid が上がる」の大部分は、mid がその既知の microprice へ
寄っていくだけの機械的な動きでありうる。それなら取引で取れない。
そこで (1) T 時点の microprice - mid と (2) microprice で測った前向き
リターンを同時に記録し、mid のリターンがどこまで行って止まるかを見る。

目的変数
--------
  r(h) = log mid(T+h) - log mid(T)  を bp で。mid は各時刻以前の最後の気配。
  h = 100ms, 500ms, 1s, 3s, 5s, 10s, 30s, 60s

時間契約 (CLAUDE.md の厳禁事項)
------------------------------
1. 説明変数の窓は必ず T で閉じ、目的変数は T から始まる。重なりはない
2. asof は全て後ろ向き (searchsorted の side を明示)
3. **五分位の境目は前日の分布から作る**。全標本から作ると、その日の実現値を
   知らないと決まらない境目になり実装不能な規則になる。前日基準なら各時点で
   計算できる (checklist C-16 の実装可能性)
4. ホライズンごとに歩幅を h にして**窓を重ねない**。重ねると SE が嘘になる
5. 帰無対照として、説明変数を日内で +300 秒だけ円状にずらした版も同時に測る。
   分布と日内の起伏は保ったまま、対応だけが壊れる

費用について
------------
五分位の差 (Q5-Q1) は**粗利**である。実際に取るなら往復でスプレッドを払う。
セルごとに T 時点のスプレッドも記録するので、必ず引いてから読むこと。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_hazard import ofi_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DAY_NS = 86_400_000_000_000
HORIZONS = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
# 説明変数の窓。1 秒だと攻撃的注文は 45.8% の窓が「約定ゼロ」になり、五分位の
# 境目が 4 つとも 0 に縮退して分位そのものが作れない。10 秒を主、1 秒を対照にする。
LOOKBACKS = (1, 10)
FILL_GUARD_NS = 5_000_000          # fills の時刻ずれに対する安全余裕 5ms
PLACEBO_SHIFT = 300                # 帰無対照: 日内で 300 秒ずらす
FEATS = ("obi",) + tuple(f"{b}_{w}s" for w in LOOKBACKS
                         for b in ("ofi", "ai_net", "ai_rate"))
NQ = 5
GRID_N = 86_400                    # 1 秒格子

# リターン分布のビン。ホライズンによって散らばりが 100 倍違う (100ms で平均
# 絶対値 0.14bp、60s で 9.1bp) ので、等間隔ではなく**対数等比**で区切る。
# 0.001bp から 1000bp まで 1 桁あたり 20 本、正負対称。ちょうど 0 は別勘定。
BE = 10.0 ** np.arange(-3.0, 3.0001, 0.05)
EDG = np.concatenate([-BE[::-1], [0.0], BE])
NBIN = EDG.size + 1


def quintile(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """前日の境目で 0..4 に割り当てる。"""
    return np.searchsorted(edges, v, side="right").astype(np.int8)


def usable(edges: np.ndarray | None) -> bool:
    """境目が狭義単調でなければ五分位は作れない (離散変数で起きる)。

    ここで弾かないと、たとえば境目が 4 つとも 0 のときに「Q1 が 14.9% で
    Q5 が 54.9%」といった中身の無い分位表が黙って出てくる。
    """
    return edges is not None and bool(np.all(np.diff(edges) > 0))


def day_features(d: pl.DataFrame, fd: pl.DataFrame | None) -> dict | None:
    """1 日ぶんの 1 秒格子の特徴量と前向きリターンを作る。

    build_pred.py と build_burst.py で定義がずれないよう、ここ 1 か所に置く。
    戻り値の配列はすべて長さ GRID_N+1 で、添字 k が「その日の k 秒目」に対応する。
    """
    ts = d["ts"].cast(pl.Int64).to_numpy()
    pb = d["best_bid"].to_numpy()
    pa = d["best_ask"].to_numpy()
    qb = d["bid_sz"].to_numpy()
    qa = d["ask_sz"].to_numpy()
    mid = 0.5 * (pb + pa)
    lmid = np.log(mid)
    tt = qb + qa
    lmic = np.log(np.where(tt > 0, (qa * pb + qb * pa) / np.where(tt > 0, tt, 1.0), mid))
    spr = (pa - pb) / mid * 1e4
    obi_e = np.where(tt > 0, (qb - qa) / np.where(tt > 0, tt, 1.0), 0.0)
    cofi = np.cumsum(ofi_series({"bid": pb, "ask": pa, "bsz": qb, "asz": qa}))

    d0 = int(ts[0]) // DAY_NS * DAY_NS
    tg = d0 + np.arange(GRID_N + 1, dtype=np.int64) * 10 ** 9
    gi = np.searchsorted(ts, tg, side="right") - 1      # T 以前の最後の気配
    ok = gi >= 0
    gi_s = np.where(ok, gi, 0)

    mg = np.where(ok, lmid[gi_s], np.nan)
    cg = np.where(ok, lmic[gi_s], np.nan)
    mmg = (cg - mg) * 1e4          # T 時点で既知の microprice と mid の差 (bp)
    F = {"obi": np.where(ok, obi_e[gi_s], np.nan)}
    sprg = np.where(ok, spr[gi_s], np.nan)

    if fd is None:
        return None
    ft = fd["ts"].cast(pl.Int64).to_numpy()
    fs = fd["sz"].to_numpy()
    sgn = np.where(fd["side"].to_numpy() == "B", 1.0, -1.0)
    cnet = np.concatenate([[0.0], np.cumsum(fs * sgn)])
    ccnt = np.arange(ft.size + 1, dtype=np.float64)

    for w in LOOKBACKS:
        back = np.roll(gi_s, w)
        okb = ok & np.roll(ok, w)
        okb[:w] = False
        F[f"ofi_{w}s"] = np.where(okb, cofi[gi_s] - cofi[back], np.nan)
        lo = np.searchsorted(ft, tg - w * 10 ** 9 - FILL_GUARD_NS, side="left")
        hi = np.searchsorted(ft, tg - FILL_GUARD_NS, side="left")
        F[f"ai_net_{w}s"] = cnet[hi] - cnet[lo]
        F[f"ai_rate_{w}s"] = ccnt[hi] - ccnt[lo]

    # 目的変数: 各ホライズンの前向きリターン (bp)
    R, RC = {}, {}
    for h in HORIZONS:
        if h < 1.0:
            gj = np.maximum(np.searchsorted(ts, tg + int(h * 1e9), side="right") - 1, 0)
            R[h] = (np.where(ok, lmid[gj], np.nan) - mg) * 1e4
            RC[h] = (np.where(ok, lmic[gj], np.nan) - cg) * 1e4
        else:
            w = int(h)
            R[h] = np.concatenate([mg[w:] - mg[:-w], np.full(w, np.nan)]) * 1e4
            RC[h] = np.concatenate([cg[w:] - cg[:-w], np.full(w, np.nan)]) * 1e4
    return {"tg": tg, "mg": mg, "mmg": mmg, "F": F, "spr": sprg, "R": R,
            "RC": RC, "ok": ok,
            # 生の気配 (反実仮想のメイカー注文の判定に使う)
            "bid": np.where(ok, pb[gi_s], np.nan),
            "ask": np.where(ok, pa[gi_s], np.nan),
            "bsz": np.where(ok, qb[gi_s], np.nan),
            "asz": np.where(ok, qa[gi_s], np.nan)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    bb, nd = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))
    print(f"bbo {bb.height:,} 行 (異常 {nd:,} 行を除外)", flush=True)
    FI = pl.read_parquet(DATA / f"fills_{tag}.parquet",
                         columns=["ts", "sz", "side", "crossed", "dt"]
                         ).filter(pl.col("crossed"))
    print(f"fills(taker) {FI.height:,} 行", flush=True)
    FID = {t[0]: v.sort("ts") for t, v in FI.partition_by("dt", as_dict=True).items()}
    del FI
    days = sorted(bb["dt"].unique().to_list())

    rows, edrows = [], []
    hist: dict[tuple, np.ndarray] = {}
    prev_edges: dict[str, np.ndarray] = {}
    for k, dt in enumerate(days):
        d = bb.filter(pl.col("dt") == dt).sort("ts")
        D = day_features(d, FID.get(dt))
        if D is None:
            print(f"  {dt}: taker fills なし → 飛ばす", flush=True)
            continue
        F, R, RC = D["F"], D["R"], D["RC"]
        mmg, sprg = D["mmg"], D["spr"]

        for f in FEATS:
            v = F[f]
            fin = np.isfinite(v)
            e = np.nanquantile(v[fin], [0.2, 0.4, 0.6, 0.8]) if fin.any() else None
            if e is not None:
                edrows.append({"dt": dt, "feat": f,
                               **{f"e{i+1}": float(e[i]) for i in range(4)}})
            pe = prev_edges.get(f)
            if not usable(pe):
                continue        # 初日、または境目が縮退していて分位が作れない日
            q = quintile(v, pe)
            vp = np.roll(v, PLACEBO_SHIFT)
            qp = quintile(vp, pe)
            for h in HORIZONS:
                r = R[h]
                st = max(1, int(h))          # 歩幅 = h。窓を重ねない
                sl = slice(0, GRID_N + 1, st)
                rr, qq, qqp, ss = r[sl], q[sl], qp[sl], sprg[sl]
                rc, mm = RC[h][sl], mmg[sl]
                fr = np.isfinite(rr)
                m = fr & np.isfinite(v[sl])
                mp = fr & np.isfinite(vp[sl])
                for iq in range(NQ):
                    sel = m & (qq == iq)
                    selp = mp & (qqp == iq)
                    x = rr[sel]
                    key = (f, h, iq)
                    if key not in hist:
                        hist[key] = np.zeros(NBIN, dtype=np.int64)
                    hist[key] += np.bincount(
                        np.searchsorted(EDG, x, side="right"), minlength=NBIN)
                    rows.append({
                        "dt": dt, "feat": f, "h": h, "q": iq,
                        "n": int(x.size), "sum_r": float(x.sum()),
                        "sum_r2": float((x * x).sum()),
                        "sum_abs": float(np.abs(x).sum()),
                        "sum_r3": float((x ** 3).sum()),
                        "sum_r4": float((x ** 4).sum()),
                        "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum()),
                        "n_zero": int((x == 0).sum()),
                        "sum_spr": float(np.nansum(ss[sel])),
                        "sum_mm": float(np.nansum(mm[sel])),
                        "sum_rmic": float(np.nansum(rc[sel])),
                        "n_p": int(selp.sum()),
                        "sum_r_p": float(rr[selp].sum())})
        prev_edges = {f: np.nanquantile(F[f][np.isfinite(F[f])],
                                        [0.2, 0.4, 0.6, 0.8])
                      for f in FEATS if np.isfinite(F[f]).any()}
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{k+1}/{len(days)}] {dt}", flush=True)

    pl.DataFrame(rows).write_parquet(DATA / f"pred_cells_{tag}.parquet")
    hr = []
    for (f, h, iq), v in hist.items():
        nz = np.nonzero(v)[0]
        for b in nz:
            hr.append({"feat": f, "h": h, "q": iq, "bin": int(b),
                       "lo": float(EDG[b - 1]) if b > 0 else -np.inf,
                       "hi": float(EDG[b]) if b < EDG.size else np.inf,
                       "count": int(v[b])})
    pl.DataFrame(hr).write_parquet(DATA / f"pred_hist_{tag}.parquet")
    print(f"書き出し pred_hist({len(hr):,} 行)")
    pl.DataFrame(edrows).write_csv(DATA / f"pred_edges_{tag}.csv")
    print(f"書き出し pred_cells({len(rows):,} セル) / pred_edges({len(edrows)})")


if __name__ == "__main__":
    main()
