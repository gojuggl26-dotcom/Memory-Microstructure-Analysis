"""見せかけの板(spoofing / layering)を疑う 18 指標を、注文 1 本ごとに測る。

【最初に — これは疑いの指標であって、証拠ではない】
この市場では[指値の 99.08% が取り消され、寿命の中央値は 565 ミリ秒](
../reports/MU/mu_hazard_report.md)である。**「すぐ消す」「大量に取り消す」は
全員がやっている既定の行動**であり、素の閾値判定は正当なマーケットメイクを
そのまま拾う。したがって全指標について母集団のベースラインを併記し、
「その口座が母集団と比べてどれだけ極端か」でしか語らない。
口座は本リポジトリ内部の連番(`wid`)だけを持ち、アドレスは出さない。

意図(spoofing の構成要件)は観測できない。ここで測れるのは行動の形だけである。

【★この指標群は将来の情報を使う。予測には使えない】
「近づいたら逃げたか」を判定するには、取り消した**あとに**値段がそこへ来たかを
見る必要がある。したがって cancel-before-touch 系は設計上ルックアヘッドを含む。
**記述と監視のための量であって、取引シグナルにはならない。**
(予測に使える板の量は `build_impact_signal.py` のほうで別に測ってある。)

【18 指標】
注文 1 本ごとの属性(A)と、口座 × 窓ごとの集計(B)に分かれる。

 A. 注文 1 本ごと
  1 large-order appearance     数量が前日の p99 以上
  2 short-lived large order    1 と 寿命 < 1 秒
  3 large order near touch     1 と 発注時に最良から 1 ティック以内
 10 fake depth persistence     最良に到達した注文のうち、約定せず取り消された割合
                               (数量加重も出す)
 11 cancel-before-touch        約定せず取り消し、かつ**取消後 1 秒以内**に
                               その値段で約定が起きた(= 残っていれば約定した)
 18 fleeting-liquidity ratio   寿命 < 1 秒の注文が占める「数量 × 表示時間」の割合
                               = 任意の瞬間に見えている板のうち 1 秒以内に消える分

【★★ spoofing の本質を直接見る — 出したあと反対側で売り抜けたか】
見せかけの買い板は「値段を上げてから**自分は売る**」ために出す。板の形をいくら
数えてもここは判らないので、**大口を最良の近くに出した口座が、その後 10 秒に
反対側で成行を出したか**を約定記録から直接引く。これが無いと、
「大口の直後に値段が動いた」は情報を持った注文(正当)と区別できない。
該当する注文は 98 日で 307,593 件。episode 表として書き出す。

【★経済的な検証 — 形だけでなく「効いたか」を測る】
見せかけの板が意味を持つのは、それを見た他者が動くときだけである。そこで
**最良の近くに出した大口の直後 10 秒の mid の動き**を、その注文の側へ符号を
合わせて記録する(買い板なら上昇が正)。同じ口座の**全注文**についても同じ量を
取り、ベースラインとする。板を出せば値段はその側へ動きやすい(前報告の
マイクロプライスの効果)ので、**0 と比べるのではなく自分自身のベースラインと
比べる**必要がある。

 B. 口座 × 窓ごと(順序のある行動)
  4 repeated large-order placement  1 分間に大口を 3 本以上
  5 repeated cancellation           取消率と 1 秒あたり取消数の最大
  6 placement/cancel cycling        同一価格に 1 分で 5 本以上出し直す
  7 same-wallet layering            同じ側の 3 価格以上へ 1 秒以内に出す
  8 multiple-level simultaneous     同じ側の 3 価格以上へ**同一ブロック**で出す
  9 asymmetric layering             使った価格数の左右差 (Lb−La)/(Lb+La)
 12 move-away-before-touch          取消 → 1 秒以内に**より遠い**価格へ出し直す
 13 order chasing price             取消 → 1 秒以内に**より近い**価格へ出し直す
 14 large-order retreat             12 を大口に限ったもの
 15 repeated re-entry               取消 → 1 秒以内に**同じ**価格へ出し直す
 16 spoof score                     2,3,11,14 と片側性を標準化して足したもの
 17 layering score                  6,7,8,9 を標準化して足したもの

【x が確定する時刻 / y の期間】
指標は監視のための記述量であり、予測の x ではない。11・12・14 は取消後 1 秒の
情報を使う(上記のとおり設計上そうしている)。16・17 の標準化は全標本の分布で
行う(標本外の主張はしない)。

【板と約定の突合】
最良気配は `bbo`(l2 由来)、約定は `fills`。注文の生存中に最良がその値段へ
到達したか、取消後に約定がその値段へ来たかを**区間の最小 / 最大**で判定する。
区間クエリはスパーステーブル(1 件あたり O(1))。

日を跨ぐ注文(全体の 0.06%)は、その日の板・約定だけで閉じないので落とす。

    uv run python scripts/build_manip.py --coin xyz:MU --days 3
    uv run python scripts/build_manip.py --coin xyz:MU
出力: data/manip_wallet_<coin>.parquet   口座 × 日 の集計
      data/manip_episodes_<coin>.parquet 最良の近くに出した大口 1 本ごと
      data/manip_daily_<coin>.csv        市場全体の日次(ベースライン)
      data/manip_meta_<coin>.csv         日ごとの検算
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

SHORT_NS = 1_000_000_000        # 「短命」の境目(1 秒)
SEQ_NS = 1_000_000_000          # 取消 → 出し直しを同一の動きとみなす間隔(1 秒)
CBT_NS = 1_000_000_000          # 取消後に約定が来たかを見る窓(1 秒)
NEAR_TICK = 1                   # 「最良の近く」(ティック)
PUSH_NS = 10_000_000_000        # 大口を出した直後に mid を見る窓(10 秒)
LAYER_MIN_LV = 3                # layering とみなす価格数
CYCLE_MIN = 5                   # 同一価格へ 1 分で何本出したら cycling か
BIG_MIN = 3                     # 1 分に大口何本で「繰り返し」か
LARGE_Q = 99.0                  # 大口の分位(前日の値を使う)


# ---- 区間の最小 / 最大(スパーステーブル)-------------------------------
def rmq_build(a: np.ndarray, is_min: bool):
    st, j = [a], 1
    while 2 * j <= len(a):
        p = st[-1]
        st.append(np.minimum(p[:-j], p[j:]) if is_min else np.maximum(p[:-j], p[j:]))
        j *= 2
    return st


def rmq_query(st, lo, hi, is_min: bool):
    """閉区間 [lo, hi] の最小(最大)。lo <= hi を前提。"""
    out = np.empty(len(lo), dtype=st[0].dtype)
    k = np.floor(np.log2(np.maximum(hi - lo + 1, 1))).astype(np.int64)
    k = np.minimum(k, len(st) - 1)
    for kk in np.unique(k):
        m = k == kk
        j = 1 << int(kk)
        lv = st[int(kk)]
        a, b = lv[lo[m]], lv[hi[m] - j + 1]
        out[m] = np.minimum(a, b) if is_min else np.maximum(a, b)
    return out


def own_taker(ep: pl.DataFrame, tk: pl.DataFrame) -> pl.DataFrame:
    """episode ごとに、その口座自身が直後 PUSH_NS に出した成行の数量を引く。

    opp = 反対側(買い板を出して**売った**= 売り抜けの形)
    same = 同じ側(値段を追って買い増した形)
    (wid, 買い成行か) ごとの累積数量を作り、t0 と t0+10 秒で差を取る。
    """
    out = ep
    for nm, opp in (("opp", True), ("same", False)):
        # 買い板(is_bid)の反対側の成行 = 売り成行(is_buy=False)
        want = (~ep["is_bid"].to_numpy()) if opp else ep["is_bid"].to_numpy()
        # ★join_asof の by は dtype が一致していないと落ちる
        e = (ep.with_columns(is_buy=pl.Series(want),
                             wid=pl.col("wid").cast(pl.Int32))
             .with_row_index("r"))
        a = (e.select("r", "wid", "is_buy", t=pl.col("t0")).sort("t")
             .join_asof(tk, on="t", by=["wid", "is_buy"], strategy="backward")
             .select("r", c0=pl.col("cum").fill_null(0)))
        b = (e.select("r", "wid", "is_buy", t=pl.col("t0") + PUSH_NS).sort("t")
             .join_asof(tk, on="t", by=["wid", "is_buy"], strategy="backward")
             .select("r", c1=pl.col("cum").fill_null(0)))
        v = (a.join(b, on="r").with_columns((pl.col("c1") - pl.col("c0"))
                                            .clip(0).alias(f"tk_{nm}"))
             .sort("r").select(f"tk_{nm}"))
        out = out.with_columns(v[f"tk_{nm}"])
    return out


def day_orders(fp: Path, bb: pl.DataFrame, fl: pl.DataFrame,
               um: pl.DataFrame, large_lots: float):
    """1 日分の注文表(その日に出て、その日に消えたもの)を作る。"""
    d = pl.read_parquet(fp)
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & (~pl.col("reduce_only"))
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pt=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  sz=(pl.col("orig_sz") / SZ_LOT).round().cast(pl.Int32),
                  rs=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int32),
                  op=(pl.col("status") == "open"),
                  fill=(pl.col("status") == "filled")))
    op = ev.filter(pl.col("op")).select("oid", "is_bid", "pt", "sz",
                                        t0=pl.col("ts"))
    tm = (ev.filter(~pl.col("op"))
          .select("oid", t1=pl.col("ts"), fill=pl.col("fill"), rs="rs")
          .unique(subset="oid", keep="first"))
    n_open, n_term = op.height, tm.height
    o = op.join(tm, on="oid", how="inner")
    n_same_day = o.height
    o = o.join(um, on="oid", how="left")          # wid
    del d, ev, op, tm

    # ---- 板・約定の配列 ---------------------------------------------------
    # ★価格は int32 で足りる(最大 125,400)。スパーステーブルが 20 段あるので
    #   ここを int64 にすると 1 日 20 万行の板でも数百 MB になり、大きい日で
    #   メモリが尽きる(実際に 2,340 万注文の日で落ちた)。
    bts = bb["ts"].cast(pl.Int64).to_numpy()
    bbid = np.rint(bb["best_bid"].to_numpy() / PX_UNIT).astype(np.int32)
    bask = np.rint(bb["best_ask"].to_numpy() / PX_UNIT).astype(np.int32)
    st_bid = rmq_build(bbid, True)                # 生存中の最良買いの最小
    st_ask = rmq_build(bask, False)               # 生存中の最良売りの最大
    sell = fl.filter(pl.col("side") == "A")       # 売りテイカー
    buy = fl.filter(pl.col("side") == "B")
    sts = sell["ts"].cast(pl.Int64).to_numpy()
    spx = np.rint(sell["px"].to_numpy() / PX_UNIT).astype(np.int32)
    bts2 = buy["ts"].cast(pl.Int64).to_numpy()
    bpx = np.rint(buy["px"].to_numpy() / PX_UNIT).astype(np.int32)
    st_sell = rmq_build(spx, True) if len(spx) else None
    st_buy = rmq_build(bpx, False) if len(bpx) else None

    o = o.drop("oid")                             # ここから先は使わない
    t0 = o["t0"].to_numpy()
    t1 = o["t1"].to_numpy()
    pt = o["pt"].to_numpy().astype(np.int32)
    isb = o["is_bid"].to_numpy()
    nb = len(bts)

    # 発注時の最良(★ts < t0 の最後の行。自分自身は入らない)
    jl = np.searchsorted(bts, t0, side="left") - 1     # 使い回す(2 度引かない)
    have0 = jl >= 0
    j0 = np.clip(jl, 0, nb - 1)
    ref0 = np.where(isb, bbid[j0], bask[j0])
    d0 = (np.where(isb, ref0 - pt, pt - ref0)).astype(np.int32)
    del ref0

    # 生存中に最良が自分の値段へ届いたか
    i0 = np.clip(np.searchsorted(bts, t0, side="right") - 1, 0, nb - 1)
    i1 = np.maximum(np.clip(np.searchsorted(bts, t1, side="left") - 1, 0, nb - 1),
                    i0)
    mb = rmq_query(st_bid, i0, i1, True)
    ma = rmq_query(st_ask, i0, i1, False)
    del i0, i1, st_bid, st_ask
    approach = np.where(isb, mb - pt, pt - ma)    # 0 = 最良に到達
    reached = have0 & (approach <= 0)
    n_neg_app = int((have0 & (approach < 0)).sum())
    del mb, ma, approach

    # 取消後 1 秒以内に、その値段で約定が起きたか(= 残っていれば約定した)
    would = np.zeros(len(t1), bool)
    for m_side, ts_f, st_f, is_min in ((isb, sts, st_sell, True),
                                       (~isb, bts2, st_buy, False)):
        if st_f is None or not m_side.any():
            continue
        a = np.searchsorted(ts_f, t1[m_side], side="right")
        b = np.searchsorted(ts_f, t1[m_side] + CBT_NS, side="right") - 1
        ok = a <= b
        v = np.full(m_side.sum(), np.nan)
        if ok.any():
            q = rmq_query(st_f, a[ok], b[ok], is_min).astype(np.float64)
            v[ok] = q
        w = np.zeros(m_side.sum(), bool)
        p_ = pt[m_side]
        w[ok] = (v[ok] <= p_[ok]) if is_min else (v[ok] >= p_[ok])
        would[m_side] = w

    # 出した直後 10 秒の mid の動き(その注文の側へ符号を合わせる)
    mid_t = ((bbid.astype(np.float64) + bask) / 2.0)
    k0 = jl
    k1 = np.searchsorted(bts, t0 + PUSH_NS, side="left") - 1
    km = np.searchsorted(bts, t0 - PUSH_NS, side="left") - 1
    okp = (k0 >= 0) & (k1 >= 0) & (t0 + PUSH_NS <= bts[-1])
    okm = (km >= 0) & (k0 >= 0)
    m0 = mid_t[np.clip(k0, 0, nb - 1)]
    m1 = mid_t[np.clip(k1, 0, nb - 1)]
    mm = mid_t[np.clip(km, 0, nb - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (m1 - m0) / m0 * 1e4
        rp = (m0 - mm) / mm * 1e4
    push = np.where(okp & (m0 > 0), np.where(isb, r, -r), np.nan)
    del m1, k1
    # ★対照: 出す**前**の 10 秒。ここが同じくらい大きければ「押した」のではなく
    #   「追いかけた」だけである(前向き / 後ろ向きの比較と同じ考え方)
    pushp = np.where(okm & (mm > 0), np.where(isb, rp, -rp), np.nan)
    del mm, km, m0, r, rp, okp, okm, jl, k0, mid_t

    o = o.with_columns(
        push=pl.Series(push),
        push_pre=pl.Series(pushp),
        d0=pl.Series(d0.astype(np.int32)),
        reached=pl.Series(reached),
        would=pl.Series(would),
        life=pl.Series((t1 - t0).astype(np.int64)),
        large=pl.col("sz") >= large_lots,
        canc=~pl.col("fill"),
        # 約定した数量(終端の残量から)。取消でも部分約定していることがある
        fsz=(pl.col("sz") - pl.col("rs")).clip(0),
    )
    # polars の平均は NaN を飛ばさない。欠測は null にしておく
    o = o.with_columns(push=pl.col("push").fill_nan(None),
                       push_pre=pl.col("push_pre").fill_nan(None))
    o = o.with_columns(
        short=pl.col("life") < SHORT_NS,
        near=pl.col("d0") <= NEAR_TICK,
        nofill=pl.col("fsz") == 0,
    )
    meta = {"n_open": n_open, "n_term": n_term, "n_same_day": n_same_day,
            "n_no_wid": int(o["wid"].null_count()),
            "n_no_bbo": int((~have0).sum()),
            "n_neg_approach": n_neg_app}
    return o, meta


def seq_metrics(o: pl.DataFrame) -> pl.DataFrame:
    """取消 → 出し直しの向きを、口座 × 側の時系列で拾う。

    ★ここは「取り消した後に何をしたか」を見るので、時間の前向き結合を使う。
      監視のための記述であって、予測の特徴量ではない。
    """
    c = (o.filter(pl.col("canc")).select("wid", "is_bid", "pt", "large",
                                         t=pl.col("t1"))
         .drop_nulls("wid").sort("t"))
    p = (o.select("wid", "is_bid", pt_n=pl.col("pt"), t_n=pl.col("t0"))
         .with_columns(t=pl.col("t_n")).drop_nulls("wid").sort("t"))
    if not c.height or not p.height:
        return o.head(0).select("wid").with_columns(
            reentry=pl.lit(0), retreat=pl.lit(0), chase=pl.lit(0),
            retreat_large=pl.lit(0))
    j = c.join_asof(p, on="t", by=["wid", "is_bid"], strategy="forward")
    j = j.filter(pl.col("t_n").is_not_null()
                 & ((pl.col("t_n") - pl.col("t")) <= SEQ_NS))
    # 買い側は価格が下がるほど mid から遠い / 売り側は上がるほど遠い
    away = pl.when(pl.col("is_bid")).then(pl.col("pt_n") < pl.col("pt")) \
             .otherwise(pl.col("pt_n") > pl.col("pt"))
    near_ = pl.when(pl.col("is_bid")).then(pl.col("pt_n") > pl.col("pt")) \
              .otherwise(pl.col("pt_n") < pl.col("pt"))
    j = j.with_columns(reentry=pl.col("pt_n") == pl.col("pt"),
                       retreat=away, chase=near_)
    return j.group_by("wid").agg(
        reentry=pl.col("reentry").sum(),
        retreat=pl.col("retreat").sum(),
        chase=pl.col("chase").sum(),
        retreat_large=(pl.col("retreat") & pl.col("large")).sum())


def window_metrics(o: pl.DataFrame) -> pl.DataFrame:
    """layering / cycling / 大口の連発 — 窓ごとに数えて口座へまとめる。"""
    w = o.drop_nulls("wid").with_columns(
        sec=(pl.col("t0") // 1_000_000_000),
        minute=(pl.col("t0") // 60_000_000_000))
    # 7 同じ側の 3 価格以上へ 1 秒以内
    lay = (w.group_by("wid", "is_bid", "sec")
           .agg(lv=pl.col("pt").n_unique(), n=pl.len())
           .filter(pl.col("lv") >= LAYER_MIN_LV)
           .group_by("wid").agg(layer_ep=pl.len(), layer_max_lv=pl.col("lv").max()))
    # 8 同一ブロック(同じ ts)で 3 価格以上
    sim = (w.group_by("wid", "is_bid", "t0")
           .agg(lv=pl.col("pt").n_unique())
           .filter(pl.col("lv") >= LAYER_MIN_LV)
           .group_by("wid").agg(simul_ep=pl.len(), simul_max_lv=pl.col("lv").max()))
    # 6 同一価格へ 1 分で CYCLE_MIN 本以上
    cyc = (w.group_by("wid", "is_bid", "pt", "minute").agg(n=pl.len())
           .filter(pl.col("n") >= CYCLE_MIN)
           .group_by("wid").agg(cycle_ep=pl.len(), cycle_max=pl.col("n").max()))
    # 4 1 分に大口 BIG_MIN 本以上
    big = (w.filter(pl.col("large")).group_by("wid", "minute").agg(n=pl.len())
           .filter(pl.col("n") >= BIG_MIN)
           .group_by("wid").agg(bigrep_ep=pl.len(), bigrep_max=pl.col("n").max()))
    # 9 使った価格数の左右差
    asym = (w.group_by("wid", "minute", "is_bid").agg(lv=pl.col("pt").n_unique())
            .group_by("wid", "minute")
            # ★n_unique() は UInt32。そのまま引くと lb < la のとき 2^32 へ
            #   回り込む(実際に asym の最大が 4,294,967,295 になっていた)
            .agg(lb=pl.col("lv").cast(pl.Int64).filter(pl.col("is_bid")).sum(),
                 la=pl.col("lv").cast(pl.Int64).filter(~pl.col("is_bid")).sum())
            .with_columns(a=(pl.col("lb") - pl.col("la"))
                          / (pl.col("lb") + pl.col("la")))
            .group_by("wid").agg(asym_mean=pl.col("a").mean(),
                                 asym_absmean=pl.col("a").abs().mean()))
    # 5 1 秒あたり取消数の最大
    cps = (w.filter(pl.col("canc"))
           .with_columns(s1=(pl.col("t1") // 1_000_000_000))
           .group_by("wid", "s1").agg(n=pl.len())
           .group_by("wid").agg(canc_per_s_max=pl.col("n").max()))
    out = lay
    for t in (sim, cyc, big, asym, cps):
        out = out.join(t, on="wid", how="full", coalesce=True)
    return out


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
    # 数量は使わない。3,500 万行 × 2 列を持ち続けると 0.6GB 無駄になる
    bb_all = bb_all.select("ts", "best_bid", "best_ask", "dt")
    have = set(bb_all["dt"].unique().to_list())
    files = [f for f in files if f.stem.split("=")[1] in have]
    fl_all = pl.read_parquet(
        ROOT / "data" / f"fills_{tag}.parquet",
        columns=["ts", "px", "sz", "side", "user", "crossed", "dt"]
    ).filter(pl.col("crossed"))
    wmap = pl.read_parquet(ROOT / "data" / f"l1user_{tag}" / "_wallets.parquet")
    n_all = fl_all.height
    # user は文字列で重い。ここで一度だけ wid に置き換える
    fl_all = (fl_all.join(wmap, on="user", how="inner")
              .select("ts", "px", "sz", "side", "dt",
                      wid=pl.col("wid").cast(pl.Int32)))
    print(f"[load] {len(files)} 日 / bbo 除外 {n_drop:,} / 成行 "
          f"{n_all:,} 行のうち口座表と突合 {fl_all.height:,}"
          f"({100 * fl_all.height / n_all:.1f}%)", file=sys.stderr)

    wparts, dparts, metas, eparts = [], [], [], []
    large_lots = None
    for fp in files:
        dt = fp.stem.split("=")[1]
        bb = bb_all.filter(pl.col("dt") == dt).sort("ts")
        fl = fl_all.filter(pl.col("dt") == dt).sort("ts")
        up = ROOT / "data" / f"l1user_{tag}" / f"dt={dt}.parquet"
        um = pl.read_parquet(up) if up.exists() else pl.DataFrame(
            schema={"oid": pl.Int64, "wid": pl.Int32})
        # 大口の閾値は前日の p99(初日だけ自分自身)
        if large_lots is None:
            tmp = pl.read_parquet(fp, columns=["status", "tif", "is_trigger",
                                               "reduce_only", "orig_sz"])
            s = tmp.filter((pl.col("status") == "open") & (~pl.col("is_trigger"))
                           & (~pl.col("reduce_only"))
                           & pl.col("tif").is_in(RESTING))["orig_sz"].to_numpy()
            large_lots = float(np.percentile(s, LARGE_Q)) / SZ_LOT
            del tmp, s
        o, meta = day_orders(fp, bb, fl, um, large_lots)
        n = o.height
        # ---- 最良の近くに出した大口 1 本ごと(自分の成行つき)---------------
        ep = o.filter(pl.col("large") & pl.col("near")).drop_nulls("wid").select(
            "wid", "is_bid", "t0", "t1", "pt", "sz", "life", "fill", "would",
            "push", "push_pre")
        tk = (fl.select(wid=pl.col("wid").cast(pl.Int32),
                        is_buy=(pl.col("side") == "B"),
                      t=pl.col("ts").cast(pl.Int64), sz=pl.col("sz"))
              .sort("wid", "is_buy", "t")
              .with_columns(cum=pl.col("sz").cum_sum().over("wid", "is_buy"))
              .select("wid", "is_buy", "t", "cum").sort("t"))
        ep = (own_taker(ep, tk) if ep.height and tk.height
              else ep.with_columns(tk_opp=pl.lit(0.0), tk_same=pl.lit(0.0)))
        eparts.append(ep.with_columns(pl.lit(dt).alias("dt")))
        szt = (o["sz"].cast(pl.Float64) * o["life"].cast(pl.Float64)).to_numpy()
        fleet = szt[o["short"].to_numpy()].sum()
        rch = o.filter(pl.col("reached"))
        dparts.append({
            "dt": dt, "n_ord": n, "large_lots": large_lots,
            "share_large": float(o["large"].mean()),
            "share_short": float(o["short"].mean()),
            "share_near": float(o["near"].mean()),
            "share_cancel": float(o["canc"].mean()),
            "fleeting_ratio": float(fleet / szt.sum()) if szt.sum() else np.nan,
            "n_reached": rch.height,
            "fake_share": float((rch["canc"] & rch["nofill"]).mean())
            if rch.height else np.nan,
            # ★sz は Int32(ロット)。日次で合計すると 2^31 に届くので必ず
            #   Int64 へ上げてから足す(上げずに出した版は 0 と 100 の間で暴れた)
            "fake_share_sz": float(
                rch.filter(pl.col("canc") & pl.col("nofill"))["sz"]
                .cast(pl.Int64).sum() / rch["sz"].cast(pl.Int64).sum())
            if rch.height else np.nan,
            "cbt_share": float((o["canc"] & o["nofill"] & o["would"]).mean()),
            "cbt_share_large": float(
                (o["canc"] & o["nofill"] & o["would"] & o["large"]).sum()
                / max(int(o["large"].sum()), 1)),
            "push_all": float(o["push"].mean()),
            "push_pre_all": float(o["push_pre"].mean()),
            "push_pre_large_near": float(
                o.filter(pl.col("large") & pl.col("near"))["push_pre"].mean())
            if int((o["large"] & o["near"]).sum()) else np.nan,
            "push_large_near": float(
                o.filter(pl.col("large") & pl.col("near"))["push"].mean())
            if int((o["large"] & o["near"]).sum()) else np.nan,
        })
        base = (o.drop_nulls("wid").group_by("wid").agg(
            n_ord=pl.len(), n_large=pl.col("large").sum(),
            n_cancel=pl.col("canc").sum(), n_fill=pl.col("fill").sum(),
            n_short_large=(pl.col("large") & pl.col("short")).sum(),
            n_large_near=(pl.col("large") & pl.col("near")).sum(),
            n_reached=pl.col("reached").sum(),
            n_fake=(pl.col("reached") & pl.col("canc") & pl.col("nofill")).sum(),
            n_cbt=(pl.col("canc") & pl.col("nofill") & pl.col("would")).sum(),
            n_cbt_large=(pl.col("canc") & pl.col("nofill") & pl.col("would")
                         & pl.col("large")).sum(),
            sz_sum=pl.col("sz").cast(pl.Int64).sum(),
            szt=(pl.col("sz").cast(pl.Float64)
                 * pl.col("life").cast(pl.Float64)).sum(),
            szt_fleet=(pl.col("sz").cast(pl.Float64)
                       * pl.col("life").cast(pl.Float64))
            .filter(pl.col("short")).sum(),
            n_bid=pl.col("is_bid").sum(),
            # 経済的な検証: 大口を最良の近くに出した直後 10 秒の mid の動き
            n_push=(pl.col("large") & pl.col("near")
                    & pl.col("push").is_not_null()).sum(),
            push_sum=pl.col("push")
            .filter(pl.col("large") & pl.col("near")).sum(),
            push_pre_sum=pl.col("push_pre")
            .filter(pl.col("large") & pl.col("near")).sum(),
            n_base=pl.col("push").is_not_null().sum(),
            base_sum=pl.col("push").sum(),
            base_pre_sum=pl.col("push_pre").sum()))
        w = base.join(seq_metrics(o), on="wid", how="left") \
                .join(window_metrics(o), on="wid", how="left") \
                .with_columns(pl.lit(dt).alias("dt"))
        wparts.append(w)
        meta["dt"] = dt
        metas.append(meta)
        # 翌日のための閾値
        large_lots = float(np.percentile(o["sz"].to_numpy(), LARGE_Q))
        print(f"  {dt} 注文 {n:>9,} 口座 {w.height:>6,} 大口 {int(o['large'].sum()):>7,}"
              f" 到達 {rch.height:>8,} 逃げ {int((o['canc']&o['nofill']&o['would']).sum()):>7,}",
              file=sys.stderr)
        del o, rch, base, w, ep, tk
        gc.collect()

    E = pl.concat(eparts, how="diagonal_relaxed")
    E.write_parquet(ROOT / "data" / f"manip_episodes_{tag}.parquet",
                    compression="zstd")
    print(f"[out] 大口 episode {E.height:,} 件 "
          f"-> data/manip_episodes_{tag}.parquet", file=sys.stderr)
    W = pl.concat(wparts, how="diagonal_relaxed")
    W.write_parquet(ROOT / "data" / f"manip_wallet_{tag}.parquet",
                    compression="zstd")
    pl.DataFrame(dparts).write_csv(ROOT / "data" / f"manip_daily_{tag}.csv")
    pl.DataFrame(metas).write_csv(ROOT / "data" / f"manip_meta_{tag}.csv")
    print(f"\n[out] 口座×日 {W.height:,} 行 -> data/manip_wallet_{tag}.parquet",
          file=sys.stderr)


if __name__ == "__main__":
    main()
