"""束の間の注文(fleeting order)を数え、流動性に占める割合を出す。

【定義】
**束の間の注文** = 板に置かれてから $\\tau$ 以内に、約定せずに取り消された指値。
Hasbrouck and Saar (2009) は 2 秒を閾値に置いている。閾値を 1 つに決め打ちすると
結論がその値に依存するので、本スクリプトは

    0(同一ブロック)/ 100ms / 200ms / 500ms / 1s / 2s / 5s / 10s

の 8 つすべてで数え、2 秒を見出しの値として使う。

    束の間の本数   N_f(τ) = τ 以内に取り消された指値の本数
    束の間の数量   V_f(τ) = その合計数量(契約)
    束の間比率     FLR(τ) = V_f(τ) / V_all      … 数量ベース(見出し)
                            N_f(τ) / N_all      … 本数ベース(併記)

**分母は「その日に板から消えた指値」**であって、板にある残高ではない。
発注日ではなく終端日で数えているのは、日を跨いで生き残った注文を二重に数えず、
かつ寿命が確定したものだけを母集団に入れるためである。日を跨ぐ注文は
1 日あたり 1〜2 万本(全体の 0.1〜1%)しかなく、どれも長寿命なので
束の間ではない。比率を出すときは必ずこの分母を明記する。

層別は次の 4 つ。いずれも**発注の瞬間までに確定している量**だけを使う。

    side    買い / 売り                     → bid FLR / ask FLR
    dist    同じ側の最良気配から何ティック   → touch FLR(最良かそれより内側)
    size    注文数量                        → large-order FLR(上位 1%)
    wallet  発注元の口座                    → wallet FLR

【母集団から外すもの】既存のハザード分析と同じ規則にそろえてある。

    トリガー注文    発火するまで板に載らない
    テイカー        tif が Ioc / FrontendMarket / LiquidationMarket
    reduce_only     取引所がポジション減少に合わせて自動縮小するため終端の意味が曖昧

【終端の判定】
`filled` は残量 0 のときだけ終端で、残量が残っていれば部分約定(継続)である
(ゲート2 D3)。約定して消えた注文は束の間ではない。取り消しだけを数える。

【x が確定する時刻 / y の期間】
本レポートは記述統計であって予測ではない。層別に使う量(側・距離・数量・口座)は
すべて発注時刻 $t_0$ までに確定し、束の間かどうかは $t_0$ より後に決まる。
最良気配は $t_0$ **未満**の最後の bbo 行で取る(同一 ns の行には自分自身の
発注が含まれうるため)。

【口座ごとの比率に付ける対照】
口座によって FLR が違って見えても、注文本数が少ない口座はばらつきが大きいだけ
かもしれない。そこで**各口座の本数を保ったまま、束の間かどうかを全体の比率で
無作為に割り当てた**分布を並べて出す。実測の広がりがこれを超えて初めて
「口座によって癖が違う」と言える。

    uv run python scripts/build_fleeting.py --coin xyz:MU
出力: data/fleeting_cells_<coin>.parquet    … 日 × 側 × 距離 × 数量 × τ の計数
      data/fleeting_daily_<coin>.parquet    … 日 × τ の合計
      data/fleeting_wallet_<coin>.parquet   … 口座 × τ の合計(全期間)
      data/fleeting_meta_<coin>.csv         … 検算
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RESTING = ["Alo", "Gtc"]
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]

# 閾値[ns]。0 は「同じブロックで発注と取消」= 寿命ちょうど 0
TAU = [0, 100_000_000, 200_000_000, 500_000_000, 1_000_000_000,
       2_000_000_000, 5_000_000_000, 10_000_000_000]
TAU_LAB = ["同一ブロック", "100ms", "200ms", "500ms", "1s", "2s", "5s", "10s"]
TAU_HEAD = 5                       # 見出しに使う閾値の添字(2 秒)

# 区切りは既存のハザード分析と同じものを使う(レポート間で読み替えずに済むよう)
D_EDGES = [0, 1, 3, 6, 11, 26, 51, 101, 301, 1001]
D_LAB = ["改善 <0", "0 最良", "1-2", "3-5", "6-10", "11-25", "26-50", "51-100",
         "101-300", "301-1000", "1001+"]
S_EDGES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
S_LAB = ["-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", "100+"]
CALIB_DAYS = 5                     # 「大口」の閾値を測る日数(先頭から)
DAY_NS = 86_400_000_000_000


def tick_of(px: np.ndarray) -> np.ndarray:
    """Hyperliquid の価格は有効数字 5 桁。1000 未満 0.01 / 1000 以上 0.1。"""
    return np.where(px >= 1000.0, 0.1, 0.01)


def day_orders(fp: Path, bb: pl.DataFrame, carry: pl.DataFrame):
    """1 日分の「置かれた指値」を作り、寿命と層別の帯を付ける。"""
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status", "orig_sz",
                                     "remaining_sz", "tif", "reduce_only", "is_trigger"])
    op = (d.filter((pl.col("status") == "open") & (~pl.col("is_trigger"))
                   & (~pl.col("reduce_only")) & pl.col("tif").is_in(RESTING))
          .select(oid=pl.col("oid").cast(pl.Int64), t0=pl.col("ts").cast(pl.Int64),
                  isbid=(pl.col("side") == "B"), px=pl.col("px"),
                  sz=pl.col("orig_sz").cast(pl.Float64)))
    # 終端 = 取消、または残量 0 の filled。残量が残る filled は部分約定で継続する。
    tm = (d.filter(pl.col("status").is_in(CANCELS)
                   | ((pl.col("status") == "filled") & (pl.col("remaining_sz") <= 0)))
          .select(oid=pl.col("oid").cast(pl.Int64), t1=pl.col("ts").cast(pl.Int64),
                  cx=pl.col("status").is_in(CANCELS))
          .group_by("oid").agg(t1=pl.col("t1").min(), cx=pl.col("cx").first()))
    del d

    book = pl.concat([carry, op]) if carry.height else op
    book = book.join(tm, on="oid", how="left")
    nxt = book.filter(pl.col("t1").is_null()).select(op.columns)   # 翌日へ持ち越す
    cur = book.filter(pl.col("t1").is_not_null())

    t0 = cur["t0"].to_numpy()
    life = cur["t1"].to_numpy() - t0
    cx = cur["cx"].to_numpy()
    isbid = cur["isbid"].to_numpy()
    px = cur["px"].to_numpy()
    sz = cur["sz"].to_numpy()

    # ---- 発注時の最良気配。★t0 未満で切る(同一 ns には自分の発注が入りうる)
    bts = bb["ts"].cast(pl.Int64).to_numpy()
    j = np.searchsorted(bts, t0, side="left") - 1
    ok = j >= 0
    j = np.where(ok, j, 0)
    pb = bb["best_bid"].to_numpy()[j]
    pa = bb["best_ask"].to_numpy()[j]
    ok &= np.isfinite(pb) & np.isfinite(pa)
    ref = np.where(isbid, pb, pa)
    tk = tick_of(np.where(isbid, pb, pa))
    dist = np.where(isbid, (ref - px) / tk, (px - ref) / tk)
    dist = np.where(ok, np.rint(dist), np.nan)
    return cur, life, cx, isbid, sz, dist, nxt, int((~ok).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"l1_{tag}").glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"data/l1_{tag} が空。先に fetch_l1.py を実行すること")

    bbo, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 信じられない行を {n_drop:,} 除いた -> {bbo.height:,} 行", file=sys.stderr)
    wal = ROOT / "data" / f"l1user_{tag}"

    # ★「大口」の閾値は本処理に入る前に測る。途中で決めると、それ以前の日の
    #   大口の集計が空のまま書き出されてしまう。
    cal = []
    for fp in files[:CALIB_DAYS]:
        cal.append(pl.read_parquet(fp, columns=["status", "orig_sz", "tif",
                                                "reduce_only", "is_trigger"])
                   .filter((pl.col("status") == "open") & (~pl.col("is_trigger"))
                           & (~pl.col("reduce_only")) & pl.col("tif").is_in(RESTING))
                   ["orig_sz"].to_numpy())
    big = float(np.percentile(np.concatenate(cal), 99))
    del cal
    print(f"[大口] 先頭 {CALIB_DAYS} 日の 99% 点 = {big:.3f} 契約", file=sys.stderr)

    outd = ROOT / "data" / f"fleeting_days_{tag}"
    outd.mkdir(parents=True, exist_ok=True)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "t0": pl.Int64, "isbid": pl.Boolean,
                                 "px": pl.Float64, "sz": pl.Float64})
    # 板は日を跨ぐので途中から始められない。日末の carry を毎日残し、
    # 落ちたらその続きから再開する。
    k0 = 0
    for i, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        if all((outd / f"{w}_{dt}.parquet").exists()
               for w in ("cells", "daily", "wallet", "carry")):
            k0 = i + 1
        else:
            break
    if k0:
        carry = pl.read_parquet(outd / f"carry_{files[k0-1].stem.split('=')[1]}.parquet")
        print(f"[再開] {k0} 日ぶんを飛ばす(繰越 {carry.height:,} 本)", file=sys.stderr)

    NT = len(TAU)
    NC = (len(D_LAB) + 1) * 2 * len(S_LAB)
    for fp in files[k0:]:
        dt = fp.stem.split("=")[1]
        t = time.time()
        bd = bbo.filter(pl.col("dt") == dt).sort("ts")
        if bd.is_empty():
            print(f"  {dt} bbo 無し。飛ばす", file=sys.stderr)
            continue
        cur, life, cx, isbid, sz, dist, carry, n_nobbo = day_orders(fp, bd, carry)

        # ★「その注文が束の間になる最小の閾値」を 1 本の添字にまとめる。
        #   閾値ごとに真偽配列を作って 8 回集計し直すと、1,300 万行の日で
        #   1GB 近い一時配列を 16 本作ることになり、実際にそれで詰まった。
        #   添字にしておけば集計は 1 回で済み、あとから累積和を取ればよい。
        ti = np.where(cx, np.searchsorted(TAU, life, side="left"), NT)
        ti = np.minimum(ti, NT)
        db = np.where(np.isfinite(dist),
                      np.searchsorted(D_EDGES, np.nan_to_num(dist), side="right"), -1)
        sb = np.searchsorted(S_EDGES, sz, side="right")
        key = (db.astype(np.int64) + 1) * (2 * len(S_LAB)) + sb * 2 + isbid.astype(int)

        kk = key * (NT + 1) + ti
        cn = np.bincount(kk, minlength=NC * (NT + 1)).reshape(NC, NT + 1)
        cv = np.bincount(kk, weights=sz, minlength=NC * (NT + 1)).reshape(NC, NT + 1)
        fn = np.cumsum(cn[:, :NT], axis=1)          # τ 以内に取り消された本数
        fv = np.cumsum(cv[:, :NT], axis=1)
        n_all, v_all = cn.sum(1), cv.sum(1)

        rows = []
        for c in np.flatnonzero(n_all):
            dbi, r = divmod(int(c), 2 * len(S_LAB))
            sbi, bi = divmod(r, 2)
            for ti_ in range(NT):
                rows.append({"dt": dt, "tau": TAU_LAB[ti_], "tau_ns": TAU[ti_],
                             "dist": D_LAB[dbi - 1] if dbi else "気配不明",
                             "size": S_LAB[sbi], "side": "買い" if bi else "売り",
                             "n_all": int(n_all[c]), "v_all": float(v_all[c]),
                             "n_f": int(fn[c, ti_]), "v_f": float(fv[c, ti_])})
        pl.DataFrame(rows).write_parquet(outd / f"cells_{dt}.parquet")

        bg = sz >= big
        cnb = np.bincount(ti[bg], minlength=NT + 1)
        cvb = np.bincount(ti[bg], weights=sz[bg], minlength=NT + 1)
        fnb, fvb = np.cumsum(cnb[:NT]), np.cumsum(cvb[:NT])
        cna = np.bincount(ti, minlength=NT + 1)
        cva = np.bincount(ti, weights=sz, minlength=NT + 1)
        fna, fva = np.cumsum(cna[:NT]), np.cumsum(cva[:NT])
        pl.DataFrame([{"dt": dt, "tau": TAU_LAB[j], "tau_ns": TAU[j],
                       "n_all": int(len(cx)), "v_all": float(sz.sum()),
                       "n_f": int(fna[j]), "v_f": float(fva[j]),
                       "n_big": int(bg.sum()), "v_big": float(sz[bg].sum()),
                       "n_big_f": int(fnb[j]), "v_big_f": float(fvb[j])}
                      for j in range(NT)]).write_parquet(outd / f"daily_{dt}.parquet")

        # ---- 口座別(見出しの閾値のみ)----------------------------------------
        wf = wal / f"dt={dt}.parquet"
        n_wal = 0
        if wf.exists():
            u = pl.read_parquet(wf).unique(subset=["oid"])
            # ★左結合は行順を保証しない。並べ直さないと wid が sz とずれる。
            m = (cur.select("oid").with_row_index("i")
                 .join(u, on="oid", how="left").sort("i"))
            wid = m["wid"].fill_null(-1).to_numpy()
            keep = wid >= 0
            n_wal = int(keep.sum())
            f_ = (ti <= TAU_HEAD)[keep]
            uq, inv = np.unique(wid[keep], return_inverse=True)
            pl.DataFrame({
                "wid": uq.astype(np.int64),
                "n_all": np.bincount(inv).astype(np.int64),
                "v_all": np.bincount(inv, weights=sz[keep]),
                "n_f": np.bincount(inv, weights=f_.astype(float)).astype(np.int64),
                "v_f": np.bincount(inv, weights=sz[keep] * f_),
            }).write_parquet(outd / f"wallet_{dt}.parquet")
        carry.write_parquet(outd / f"carry_{dt}.parquet")
        pl.DataFrame([{"dt": dt, "n_open": int(len(cx)), "n_carry": int(carry.height),
                       "n_nobbo": n_nobbo, "n_wallet_matched": n_wal,
                       "big_threshold": big}]).write_parquet(outd / f"meta_{dt}.parquet")
        print(f"  {dt} 指値 {len(cx):,} 束の間(2s) {int(fna[TAU_HEAD]):,} "
              f"繰越 {carry.height:,} 気配不明 {n_nobbo:,} {time.time()-t:.1f}s",
              file=sys.stderr, flush=True)

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    cat("cells").write_parquet(ROOT / "data" / f"fleeting_cells_{tag}.parquet")
    cat("daily").write_parquet(ROOT / "data" / f"fleeting_daily_{tag}.parquet")
    W = (cat("wallet").group_by("wid")
         .agg(n_all=pl.col("n_all").sum(), v_all=pl.col("v_all").sum(),
              n_f=pl.col("n_f").sum(), v_f=pl.col("v_f").sum()).sort("v_all",
                                                                    descending=True))
    W.write_parquet(ROOT / "data" / f"fleeting_wallet_{tag}.parquet")
    M = cat("meta")
    M.write_csv(ROOT / "data" / f"fleeting_meta_{tag}.csv")
    print(f"\n[集計] 日 {M.height} / 指値 {M['n_open'].sum():,} / 口座 {W.height:,}",
          file=sys.stderr)
    print(f"-> data/fleeting_cells_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
