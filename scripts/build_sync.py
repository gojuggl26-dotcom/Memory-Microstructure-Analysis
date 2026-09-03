"""複数の口座が同時に動くか — 同期の 8 指標。

【最初に — 同時に動くこと自体は当たり前である】
このデータの `ts` は取引所のブロック時刻で、**同じブロックに平均 18.7 行**が入る
(2026-06-24 実測)。つまりブロック粒度の「同時」は構造的に強制される。加えて
全員が同じ公開情報を見て、同じ時間帯に活動する。素の同時率・素の相関は
**何の主張にもならない**ので、本スクリプトは最初から比との形でしか出さない。

    ブロック粒度 … 観測された共起 ÷ **分ごとに層化した**独立の期待値
                    (各口座の分あたり活動ブロック数と、その分の総ブロック数から
                     解析的に出す。日内の活動プロファイルは層化で吸収される)
    秒粒度       … 観測された相関 ÷ **±300 秒の小さなずらし**で作った帰無
                    (日内の活動の山は保ったまま、秒単位の対応だけを壊す。
                     1 日まるごと巡回させると「同じ時間帯にいる」ことまで
                     壊してしまい、帰無が低く出すぎる)

同期は共謀の証拠ではない。共通の情報・共通の遅延・ブロック構造のいずれでも
同じ形が出る。口座は内部連番 `wid` だけで扱い、アドレスは出さない。

【8 指標】
  1 simultaneous placement rate   同じブロックに両者が**発注**した回数 ÷ 期待値
  2 simultaneous removal rate     同じブロックに両者が**取消**した回数 ÷ 期待値
  3 synchronized quote movement   1 秒ごとの「発注した価格の mid からの平均距離」
                                  の相関(= 一緒に広げ、一緒に詰めるか)
  4 wallet-flow correlation       1 秒ごとの符号つき出し入れ(買い − 売り、契約)
  5 wallet activity correlation   1 秒ごとの発注本数
  6 common-mode cancellation      取消本数のうち市場共通成分で説明できる割合
                                  (自分を除いた市場合計への回帰の決定係数)
  7 common-mode replenishment     発注本数について同じもの
  8 maker herding score           4・5・6・7 の順位を [0,1] にして平均

★6・7 の「市場合計」は**自分を除く**(leave-one-out)。自分を含めると、
  出来高の大きい口座ほど機械的に決定係数が上がる。

【x が確定する時刻 / y の期間】
同期の記述量であって予測の説明変数ではない。相関はすべて**同時点**で測っている
(前向きの主張はしない)。

    uv run python scripts/build_sync.py --coin xyz:MU --days 3
    uv run python scripts/build_sync.py --coin xyz:MU
出力: data/sync_pairs_<coin>.parquet  口座ペアごとの共起と相関
      data/sync_wallet_<coin>.csv     口座ごとの共通成分と herding score
      data/sync_daily_<coin>.csv      日次の市場全体
      data/sync_meta_<coin>.csv       日ごとの検算
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import (CANCELS, DAY_NS, PX_UNIT, RESTING,  # noqa: E402
                              SZ_LOT, TERMINAL, clean_bbo)

ROOT = Path(__file__).resolve().parents[1]

NW = 60                     # 対象の口座数(注文数の上位。全注文の 87.7% を覆う)
SEC = 86_400
MIN_D = 10                  # ペアの統計を出すのに要る「両者が活動した日数」
SHIFT_S = 300               # 帰無のずらし幅(±秒)
N_NULL = 3                  # 1 日あたりの帰無の反復
BLK_CHUNK = 250_000         # ブロック共起を刻む単位(メモリを抑える)
SERIES = ("add", "can", "flow", "dist")


def day_orders(fp: Path, bts, bbid, bask, um: pl.DataFrame, lut: np.ndarray):
    """その日に出て消えた指値を、対象口座に絞って返す。"""
    d = pl.read_parquet(fp)
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & (~pl.col("reduce_only"))
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pt=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  sz=(pl.col("orig_sz") / SZ_LOT).round().cast(pl.Int32),
                  op=(pl.col("status") == "open"),
                  canc=pl.col("status").is_in(CANCELS)))
    op = ev.filter(pl.col("op")).select("oid", "is_bid", "pt", "sz",
                                        t0=pl.col("ts"))
    tm = (ev.filter(~pl.col("op")).select("oid", t1=pl.col("ts"), canc="canc")
          .unique(subset="oid", keep="first"))
    n_open = op.height
    o = op.join(tm, on="oid", how="inner").join(um, on="oid", how="inner")
    del d, ev, op, tm
    # ★口座 → 行番号は配列引きにする。内包表記だと 1,300 万要素の Python リストが
    #   できてメモリを使い切る(実際に落ちた)
    wi = lut[o["wid"].to_numpy()]
    o = o.with_columns(wi=pl.Series(wi)).filter(pl.col("wi") >= 0).drop("oid", "wid")
    wi = o["wi"].to_numpy()
    # 発注時の最良気配(★ts < t0 の最後の行。自分自身は入らない)
    t0 = o["t0"].to_numpy()
    j = np.searchsorted(bts, t0, side="left") - 1
    ok = j >= 0
    jj = np.clip(j, 0, len(bts) - 1)
    mid = (bbid[jj].astype(np.float64) + bask[jj]) / 2.0
    pt = o["pt"].to_numpy().astype(np.float64)
    isb = o["is_bid"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dbp = np.where(ok & (mid > 0),
                       np.where(isb, mid - pt, pt - mid) / mid * 1e4, np.nan)
    o = o.with_columns(dbp=pl.Series(dbp).fill_nan(None))
    del mid, pt, j, jj, ok
    return o, n_open, wi


def cooc(idx_block: np.ndarray, idx_w: np.ndarray, nb: int, n: int):
    """ブロック × 口座 の共起数(n×n)。ブロックを刻んで dense で回す。"""
    C = np.zeros((n, n), np.float64)
    if not len(idx_block):
        return C
    order = np.argsort(idx_block, kind="stable")
    ib, iw = idx_block[order], idx_w[order]
    start = 0
    for lo in range(0, nb, BLK_CHUNK):
        hi = min(lo + BLK_CHUNK, nb)
        end = np.searchsorted(ib, hi, side="left")
        if end > start:
            M = np.zeros((hi - lo, n), np.float32)
            M[ib[start:end] - lo, iw[start:end]] = 1.0
            C += (M.T @ M).astype(np.float64)
            del M
        start = end
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    bb_all, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    bb_all = bb_all.sort("ts")
    have = set(bb_all["dt"].unique().to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]
    # ★3,500 万行の文字列列を持ち続けると 1GB 近い。日ごとの行範囲だけ先に取り、
    #   数値 3 列の numpy にしてから文字列列を捨てる
    bidx = {d: (int(a), int(b)) for d, a, b in
            bb_all.with_row_index("r").group_by("dt")
            .agg(lo=pl.col("r").min(), hi=pl.col("r").max()).iter_rows()}
    BTS = bb_all["ts"].cast(pl.Int64).to_numpy()
    BBID = np.rint(bb_all["best_bid"].to_numpy() / PX_UNIT).astype(np.int32)
    BASK = np.rint(bb_all["best_ask"].to_numpy() / PX_UNIT).astype(np.int32)
    del bb_all
    gc.collect()

    # 対象口座: 注文数の上位 NW(既存の集計を使う)
    W = pl.read_parquet(ROOT / "data" / f"manip_wallet_{tag}.parquet")
    tot = W.group_by("wid").agg(n=pl.col("n_ord").sum()).sort("n", descending=True)
    keep = tot.head(NW)["wid"].to_numpy()
    share = float(tot.head(NW)["n"].sum() / tot["n"].sum())
    lut = np.full(int(tot["wid"].max()) + 1, -1, np.int64)
    lut[keep] = np.arange(len(keep))
    print(f"[load] {len(files)} 日 / 対象 {len(keep)} 口座(全注文の {share:.1%})",
          file=sys.stderr)

    n = len(keep)
    rng = np.random.default_rng(0)
    # 日ごとに貯める。あとで「両者が活動した日」だけで組み直す
    D = {k: [] for k in ("dt", "nb", "obs_add", "exp_add", "obs_can", "exp_can",
                         "act_add", "act_can")}
    for s in SERIES:
        D[f"sxy_{s}"], D[f"sx_{s}"], D[f"sxx_{s}"] = [], [], []
        D[f"null_{s}"] = []
    metas = []

    for fp in files:
        dt = fp.stem.split("=")[1]
        up = ROOT / "data" / f"l1user_{tag}" / f"dt={dt}.parquet"
        if not up.exists() or dt not in bidx:
            continue
        lo_, hi_ = bidx[dt]
        o, n_open, wi = day_orders(fp, BTS[lo_:hi_ + 1], BBID[lo_:hi_ + 1],
                                   BASK[lo_:hi_ + 1], pl.read_parquet(up), lut)
        if not o.height:
            continue
        t00 = (int(o["t0"].min()) // DAY_NS) * DAY_NS
        t0 = o["t0"].to_numpy()
        t1 = o["t1"].to_numpy()
        isc = o["canc"].to_numpy()
        sz = o["sz"].to_numpy().astype(np.float64)
        isb = o["is_bid"].to_numpy()
        dbp = o["dbp"].to_numpy().astype(np.float64)

        # ---- ブロック粒度 -------------------------------------------------
        allts = np.unique(np.concatenate([t0, t1[isc]]))
        nb = len(allts)
        mnb = ((allts - t00) // 60_000_000_000).astype(np.int64)   # 各ブロックの分
        Bm = np.bincount(np.clip(mnb, 0, 1439), minlength=1440).astype(np.float64)
        obs, exp, act = {}, {}, {}
        for nm, tt, mask in (("add", t0, np.ones(len(t0), bool)),
                             ("can", t1, isc)):
            b = np.searchsorted(allts, tt[mask])
            w = wi[mask]
            pair = np.unique(np.stack([b, w], 1), axis=0)     # 同ブロック重複を潰す
            obs[nm] = cooc(pair[:, 0], pair[:, 1], nb, n)
            # 分ごとに層化した独立の期待値  E_ij = Σ_m a_i(m) a_j(m) / B(m)
            mm = np.clip(mnb[pair[:, 0]], 0, 1439)
            A = np.zeros((1440, n), np.float64)
            np.add.at(A, (mm, pair[:, 1]), 1.0)
            act[nm] = A.sum(axis=0)
            At = A / np.sqrt(np.maximum(Bm, 1))[:, None]
            exp[nm] = At.T @ At
            del A, At

        # ---- 秒粒度の 4 系列 ----------------------------------------------
        s0 = np.clip((t0 - t00) // 1_000_000_000, 0, SEC - 1).astype(np.int64)
        s1 = np.clip((t1 - t00) // 1_000_000_000, 0, SEC - 1).astype(np.int64)
        szc = sz * SZ_LOT
        np.negative(szc, out=szc, where=~isb)
        # ★1 系列ずつ作って、統計を取ったら捨てる。4 本同時に持つと
        #   (86400 × 60 × 8B) × 5 = 200MB がピークに乗る
        for s in SERIES:
            Xs = np.zeros((SEC, n))
            if s == "add":
                np.add.at(Xs, (s0, wi), 1.0)
            elif s == "can":
                np.add.at(Xs, (s1[isc], wi[isc]), 1.0)
            elif s == "flow":
                np.add.at(Xs, (s0, wi), szc)
            else:
                Msk = np.zeros((SEC, n))
                okd = np.isfinite(dbp)
                np.add.at(Xs, (s0[okd], wi[okd]), dbp[okd])
                np.add.at(Msk, (s0[okd], wi[okd]), 1.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    Xs = np.where(Msk > 0, Xs / np.maximum(Msk, 1), 0.0)
                # ★出していない秒は 0 のままにせず、その口座のその日の平均からの
                #   ずれにする。0 のままだと「両者とも休んでいる」ことが相関に化ける
                act_s = Msk > 0
                cnt = act_s.sum(axis=0)
                mu = np.where(cnt > 0, Xs.sum(axis=0) / np.maximum(cnt, 1), 0.0)
                Xs = np.where(act_s, Xs - mu[None, :], 0.0)
                del Msk, act_s
            D[f"sxy_{s}"].append(Xs.T @ Xs)
            D[f"sx_{s}"].append(Xs.sum(axis=0))
            D[f"sxx_{s}"].append((Xs ** 2).sum(axis=0))
            # ★帰無: ±SHIFT_S の小さなずらし。日内の山は保ち、秒の対応だけ壊す
            acc = np.zeros((n, n))
            Z = np.empty_like(Xs)
            for _ in range(N_NULL):
                for i in range(n):
                    Z[:, i] = np.roll(Xs[:, i],
                                      int(rng.integers(-SHIFT_S, SHIFT_S + 1)))
                acc += Z.T @ Z
            D[f"null_{s}"].append(acc / N_NULL)
            del Xs, Z, acc

        D["dt"].append(dt)
        D["nb"].append(nb)
        for nm in ("add", "can"):
            D[f"obs_{nm}"].append(obs[nm])
            D[f"exp_{nm}"].append(exp[nm])
            D[f"act_{nm}"].append(act[nm])
        metas.append({"dt": dt, "n_open": n_open, "n_kept": o.height,
                      "n_blocks": nb, "n_wallet_active": int((act["add"] > 0).sum()),
                      "share_kept": o.height / max(n_open, 1)})
        print(f"  {dt} 注文 {o.height:>9,}(対象口座) ブロック {nb:>8,} "
              f"活動口座 {int((act['add'] > 0).sum()):>3}", file=sys.stderr)
        del o, obs, exp, act, t0, t1, isc, sz, isb, dbp, szc, s0, s1, wi
        gc.collect()

    np.savez_compressed(ROOT / "data" / f"sync_raw_{tag}.npz",
                        wid=keep, dt=np.array(D["dt"]),
                        **{k: np.array(v) for k, v in D.items() if k != "dt"})
    pl.DataFrame(metas).write_csv(ROOT / "data" / f"sync_meta_{tag}.csv")
    print(f"\n[out] {len(D['dt'])} 日 × {n} 口座 -> data/sync_raw_{tag}.npz",
          file=sys.stderr)


if __name__ == "__main__":
    main()
