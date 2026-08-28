"""ティック水準ごとに OBI を作り、将来 log リターンへ OLS 回帰する。

【何を作るか】
最良気配から l−1 ティック離れた価格に積まれている数量だけを使った OBI を、
l = 1 … 10 の各水準について作る。

    Q^bid_l(t) = 価格 (最良買 − (l−1)·tick) にある数量
    Q^ask_l(t) = 価格 (最良売 + (l−1)·tick) にある数量
    OBI_l(t)   = (Q^bid_l − Q^ask_l) / (Q^bid_l + Q^ask_l)          ∈ [−1, +1]

**水準ごと(marginal)**の定義である。あわせて **累積(cumulative)**

    OBI^cum_l = (ΣQ^bid_{1..l} − ΣQ^ask_{1..l}) / (ΣQ^bid_{1..l} + ΣQ^ask_{1..l})

も同時に作る。水準ごとは奥へ行くほど空になりやすく指標として痩せるので、
「奥が効かない」のが空だからなのか本当に効かないのかを分けるには両方が要る。

【ティックは価格で変わる】
Hyperliquid の刻みは有効数字で決まる。実測(bbo 3,518 万行)で

    価格 < 1000  … 0.01 刻み(0.1 の倍数は 12.0% しかない)
    価格 ≥ 1000  … 0.1 刻み(0.1 の倍数が 100.000%)

だったので、内部の価格軸は 0.01 単位の整数で持ち、水準の間隔を
その側の最良気配が 1000 未満なら 1、以上なら 10 とする。標本 86 日のうち
35 日が 1000 を跨ぐので、片方に決め打ちすることはできない。

【板の再構成】
l1(注文イベント)を、注文ごとの **「板に置かれている数量」の階段関数**に直す。

    open                     … remaining_sz(= orig_sz)を置く
    filled で remaining > 0  … 部分約定。remaining_sz へ減らす(終端ではない)
    filled で remaining = 0  … 全約定。0 にする
    各種 canceled            … 0 にする

各注文の差分を取って板の差分列にする。**部分約定の途中経過が remaining_sz に
そのまま出ている**ので、「約定してから取消までの間だけ深さを過大に表示する」
という node_order_statuses 単独の既知の弱点をここでは回避できている。

★数量は 0.001 が最小単位(実測: 全行が 0.001 の倍数)なので、**整数のロット数**
で足し引きする。浮動小数のままだと空になった価格水準に 1e-13 の残りかすが残り、
それが「深さあり」と読まれて再構成した板が 100% クロスして見えた(実際に踏んだ)。

★同一 ns 内の行順は論理順と逆になっているので、ts だけでは並べられない。
ゲート2 D4 の論理順 open(0) → filled(2) → 終端(3) を明示して並べる。

日を跨いで残っている注文は carry として次の日へ持ち越す。**途中の日を飛ばすと
板が壊れる**ので、このスクリプトは常に先頭の日から全日を通す
(累積和の書き出しだけを飛ばす)。

★板に入れてはいけないもの(過去に実際に踏んだ):
  - **トリガー注文**は発火するまで板に載らない。入れると板が 99.95% クロスした
  - **テイカー**(Ioc / FrontendMarket / LiquidationMarket)は板に留まらない
  - **Rejected 系**は open を経ない終端。板に一度も載らないので除外する

【★残る偏り — 黙って約定した注文】
約定の一部は `filled` イベントを出さず、後日の取消イベントの
orig_sz − remaining_sz にしか現れない(hl-l4-pipeline の実測で
**取引数量の 29.17%**)。この間、板は約定した分だけ深さを過大に表示する。
fills に oid が無いので注文単位で引くことはできない。

規模は bbo(パイプラインが fills を統合して作った板)との突合で測る。
実測で **最良気配の数量が完全一致する格子点は 80〜85%、合計では 1.25 倍**。
そこで対照として次の 2 つを同じ回帰にかける。

    「bbo 最良」        … bbo の最良数量から作った L1 の OBI(fills 統合済み)
    「水準ごと(L1一致)」… 最良の数量が bbo と厳密に一致した格子点だけに絞ったもの

どちらも t で観測できる量による絞り込みなので先読みにはならない。
本命(「水準ごと」「累積」)とこの 2 つがずれなければ、深い水準の結果も信用できる。
負の深さが出た数も数える(整数で足し引きしているので出たら欠陥)。

【x が確定する時刻 / y の期間】
    x = OBI_l(t)                  … 時刻 t **より前**のイベントだけから作る
    y = ln(mid_{t+h}/mid_t)×10^4  … 期間は (t, t+h]

100ms 格子で、格子点 g の板は **バケット g−1 までの差分の累積**とした。
バケット g は [t_g, t_g+Δ) の事象を含むので、これを入れると最大 99.9ms の
未来が混ざる。mid も同じく厳密に t 未満の最後の bbo 行を使う。
shift(-k) に当たる操作は y にしか無い。

【対照】
- 帰無対照: x を日内で巡回シフト(日長の 1/3 = 8 時間)して同じ回帰をかける。
  x の分布と自己相関は保ったまま y との対応だけを壊す
- 不確かさ: 窓が最大 500 格子分重なるので古典的な標準誤差は使えない。
  日を単位にしたブロックブートストラップで傾きの区間を出す(集計側で実施)

    uv run python scripts/build_obi_levels.py --coin xyz:MU
出力: data/obi_levels_days/<coin>/dt=*.parquet   … 日ごとの回帰の累積和
      data/obi_levels_bins/<coin>/dt=*.parquet   … 日ごとの帯別の y 合計
      data/obi_levels_meta/<coin>/dt=*.parquet   … 日ごとの検算・占有率
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

GRID_NS = 100_000_000                                   # 100ms
NLV = 10                                                # ティック水準の数
STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]            # 100ms 単位
HLAB = ["100ms", "200ms", "500ms", "1s", "2s", "5s", "10s", "20s", "50s"]
DEFS = ["水準ごと", "累積", "bbo 最良", "水準ごと(L1一致)"]

PX_UNIT = 0.01                  # 内部の価格軸の刻み
SZ_LOT = 0.001                  # 数量の最小単位(実測: 全行が 0.001 の倍数)
BIG_PX = 100_000                # = 1000.00 を PX_UNIT で表した値。これ以上は 0.1 刻み
MARGIN = 160                    # 窓の余白(0.01 単位)。10 水準 × 0.1 刻み = 90 を覆う
MAX_G = 6_000                   # 1 かたまりの格子点の上限(10 分)
CELL_CAP = 2.0e7                # 1 かたまりの (格子点 × 価格帯) の上限
NBIN = 10                       # 図示用の帯。[−1, +1] を等幅 0.2 で 10 分割
DAY_NS = 86_400_000_000_000

RESTING = ["Alo", "Gtc"]
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]
TERMINAL = ["filled"] + CANCELS


def clean_bbo(bb: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """bbo から「値段として信じられない行」を落とす。

    ★2026-06-14 に、遠くに残っていた売り 200.0 と付き合わせる形で最良気配が
    1.16 秒だけ 1006 → 198 へ落ちる行が 12 本あった。件数は 3,500 万行中 26 本
    だが、y を二乗して測る量(σ_y・決定係数)はこの数本に支配される。
    その日の σ_y は 200ms で 0.34 なのに 500ms で 24.73 になっていた。
    2026-05-14 には両側そろって 785.85 → 586.10 → 785.87 と 1 行だけ飛ぶ例もある。

    規則は 2 つとも **その行と過去だけ**から決まる(先読みにならない)。

        1. 相対スプレッド ≤ 100bp   … 中央値 1.24bp、99.999% 点 51bp なので
                                      実際の分布のはるか外側。51 行が落ちる
        2. 直前 21 行の mid の中央値から ±5% 以内 … 3 行が落ちる

    合計 53 行 / 3,517 万行 = 0.00015%。これで日中央値から 20% 以上外れる行は
    1 本も残らない。**外れ値を y の側で切っているのではない**ことに注意する
    (y で切ると目的変数による選別になる)。
    """
    n0 = bb.height
    b = (bb.filter(pl.col("best_ask") > pl.col("best_bid")).sort("ts")
         .with_columns(mid=(pl.col("best_bid") + pl.col("best_ask")) / 2))
    b = b.with_columns(
        rs=(pl.col("best_ask") - pl.col("best_bid")) / pl.col("mid") * 1e4,
        bmed=pl.col("mid").rolling_median(21, min_samples=5).shift(1))
    b = b.filter((pl.col("rs") <= 100)
                 & (pl.col("bmed").is_null()
                    | ((pl.col("mid") / pl.col("bmed")).log().abs() <= 0.05)))
    return b.drop("mid", "rs", "bmed"), n0 - b.height


def grid_mid(bb: pl.DataFrame, t0: int, ng: int):
    """bbo を 100ms 格子へ後ろ向きに載せる。厳密に t 未満の最後の行を使う。"""
    ts = bb["ts"].cast(pl.Int64).to_numpy()
    tg = t0 + np.arange(ng, dtype=np.int64) * GRID_NS
    j = np.searchsorted(ts, tg, side="left") - 1        # ★ t 未満(= 先読みしない)
    ok = j >= 0
    j = np.where(ok, j, 0)
    pb = bb["best_bid"].to_numpy()[j]
    pa = bb["best_ask"].to_numpy()[j]
    qb = bb["bid_sz"].to_numpy()[j]
    qa = bb["ask_sz"].to_numpy()[j]
    good = ok & np.isfinite(pb) & np.isfinite(pa) & (pa > pb)   # クロスは除外
    mid = np.where(good, (pb + pa) / 2.0, np.nan)
    bi = np.where(good, np.rint(pb / PX_UNIT), 0).astype(np.int64)
    ai = np.where(good, np.rint(pa / PX_UNIT), 0).astype(np.int64)
    return mid, bi, ai, good, qb, qa


def chunk_bounds(bbi, bai, good, ng):
    """価格が動く日でも (格子点 × 価格帯) が上限を超えないように区切る。"""
    out = [0]
    g = 0
    while g < ng:
        e = min(g + MAX_G, ng)
        while e - g > 100:
            gg = good[g:e]
            if not gg.any():
                break
            w = (max(bbi[g:e][gg].max(), bai[g:e][gg].max())
                 - min(bbi[g:e][gg].min(), bai[g:e][gg].min()) + 2 * MARGIN + 1)
            if w * (e - g) <= CELL_CAP:
                break
            e = g + (e - g) // 2
        out.append(e)
        g = e
    return np.array(out, dtype=np.int64)


def day_features(fp: Path, bb_day: pl.DataFrame, carry: pl.DataFrame):
    """1 日分の x(2 定義 × 10 水準)と mid を作り、次の日へ渡す carry を返す。"""
    d = pl.read_parquet(fp)
    n_raw = d.height
    t0 = (int(d["ts"].cast(pl.Int64).min()) // DAY_NS) * DAY_NS      # その日の 00:00 UTC
    ng = DAY_NS // GRID_NS

    # ---- 注文イベントを「板に置かれている数量」の階段関数に直す -------------
    #   終端イベントにも tif / is_trigger が残っている(実測: open 済み oid 由来の
    #   終端 1,203,008 行が tif だけで漏れなく拾える)ので、oid の突合をせずに
    #   行の属性だけで板に載る注文の全イベントを選べる。
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  is_bid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  # ★数量は 0.001 が最小単位(実測)。整数のロット数で持てば
                  #   足し引きが厳密になり、空の価格水準がちょうど 0 になる。
                  #   浮動小数のままだと 1e-13 の残りかすが「深さあり」と読まれ、
                  #   再構成した板が 100% クロスして見えた(実際に踏んだ)
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS)))
    n_open = int((ev["rk"] == 0).sum())
    n_rest_term = int((ev["rk"] > 0).sum())
    # 部分約定は remaining_sz にそのまま出る(open では orig_sz == remaining_sz)
    n_part = int(((ev["rk"] == 2) & (ev["sz"] > 0)).sum())
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("sz", "gone"))
    if carry.height:
        cr = carry.select("oid", "is_bid", "pidx",
                          ts=pl.lit(t0 - 1, pl.Int64), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    # ★同一 ns の行順は論理順と逆なので、ts だけでは並べられない。
    #   ゲート2 D4 の論理順 open(0) → filled(2) → 終端(3) を rk で明示する。
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(
        prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
        last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "is_bid", "pidx", "size_after"))
    # 孤児 = この日より前に置かれた注文の終端。prev が 0 のまま終端に来るので
    #        差分は 0 になり、板は壊れない(数だけ数える)
    n_orph = int((ev["prev"].eq(0) & ev["rk"].gt(0) & ev["size_after"].eq(0)).sum())
    # ★並べ替えは polars 側でやる。numpy の argsort + 4 本の fancy index は
    #   2,000 万行で 700MB 近く増え、並行して重い処理が走っている環境では
    #   ここで落ちる(実際に 2026-07-13 で落ちた)。
    ev = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
          .filter((~pl.col("synth")) & (pl.col("delta") != 0))
          .with_columns(gi=((pl.col("ts") - t0) // GRID_NS)
                        .clip(0, ng - 1).cast(pl.Int32))
          .select("gi", "pidx", "is_bid", "delta").sort("gi"))
    gi = ev["gi"].to_numpy().astype(np.int64)
    pi = ev["pidx"].to_numpy()                          # int32 のまま(83MB 節約)
    bmask = ev["is_bid"].to_numpy()
    dv = ev["delta"].to_numpy().astype(np.float64)      # 整数値なので和は厳密
    del ev, d

    cpx_all = nxt["pidx"].to_numpy().astype(np.int64) if nxt.height else pi[:1]
    p_min = int(min(pi.min(), cpx_all.min(), carry["pidx"].min() if carry.height
                    else pi.min())) - MARGIN
    p_max = int(max(pi.max(), cpx_all.max(), carry["pidx"].max() if carry.height
                    else pi.max())) + MARGIN
    dep_b = np.zeros(p_max - p_min + 1, np.float64)
    dep_a = np.zeros(p_max - p_min + 1, np.float64)
    if carry.height:                                    # 前日から残っている深さ
        cb0 = carry["is_bid"].to_numpy()
        cpx = carry["pidx"].to_numpy().astype(np.int64) - p_min
        csz = carry["size_after"].to_numpy().astype(np.float64)
        np.add.at(dep_b, cpx[cb0], csz[cb0])
        np.add.at(dep_a, cpx[~cb0], csz[~cb0])

    mid, bbi, bai, good, qb1, qa1 = grid_mid(bb_day, t0, ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = qb1 + qa1
        xb = np.where(good & (sb > 0), (qb1 - qa1) / sb, np.nan)   # bbo 由来の L1 OBI
    X = np.full((2, NLV, ng), np.nan, np.float32)
    emp_b = np.zeros(NLV, np.int64)
    emp_a = np.zeros(NLV, np.int64)
    n_val = np.zeros(NLV, np.int64)      # OBI が定義できた格子点(片側でも数量がある)
    n_pm1 = np.zeros(NLV, np.int64)      # 片側だけ空 = OBI がちょうど ±1
    sum_b = np.zeros(NLV, np.float64)
    sum_a = np.zeros(NLV, np.float64)
    n_above = n_below = n_edge = n_used = n_neg = 0
    l1_same = l1_more = l1_less = 0
    l1_mine = l1_bbo = 0.0
    l1ok = np.zeros(ng, bool)               # 最良の数量が bbo と一致した格子点

    bnd = chunk_bounds(bbi, bai, good, ng)
    idx = np.searchsorted(gi, bnd)
    for ci in range(len(bnd) - 1):
        g0, g1 = int(bnd[ci]), int(bnd[ci + 1])
        G = g1 - g0
        s0, s1 = int(idx[ci]), int(idx[ci + 1])
        cp, cg, cb, cd = pi[s0:s1], gi[s0:s1], bmask[s0:s1], dv[s0:s1]
        gg = good[g0:g1]
        if gg.any():
            lo = int(min(bbi[g0:g1][gg].min(), bai[g0:g1][gg].min())) - MARGIN
            hi = int(max(bbi[g0:g1][gg].max(), bai[g0:g1][gg].max())) + MARGIN
            W = hi - lo + 1
            base_b = dep_b[lo - p_min: hi - p_min + 1]
            base_a = dep_a[lo - p_min: hi - p_min + 1]
            inw = (cp >= lo) & (cp <= hi)
            fl = (cg[inw] - g0) * W + (cp[inw] - lo)
            ib = cb[inw]
            dw = cd[inw]
            zr = np.zeros((1, W))
            Db = base_b + np.concatenate([zr, np.cumsum(np.bincount(
                fl[ib], weights=dw[ib], minlength=G * W).reshape(G, W), axis=0)[:-1]])
            Da = base_a + np.concatenate([zr, np.cumsum(np.bincount(
                fl[~ib], weights=dw[~ib], minlength=G * W).reshape(G, W), axis=0)[:-1]])
            # 整数のロット数で足し引きしているので負の深さは出ないはず。
            # 出たら再構成の欠陥なので数えて meta に出す(黙って潰さない)。
            n_neg += int((Db < 0).sum() + (Da < 0).sum())
            np.maximum(Db, 0.0, out=Db)
            np.maximum(Da, 0.0, out=Da)

            rows = np.arange(G)
            lb = np.where(bbi[g0:g1] >= BIG_PX, 10, 1)   # その側のティック(0.01 単位)
            la = np.where(bai[g0:g1] >= BIG_PX, 10, 1)
            jb0 = bbi[g0:g1] - lo
            ja0 = bai[g0:g1] - lo
            # ★bbo の最良気配より外側に自分の板が深さを持っていないかの突合。
            #   suffix / prefix の最大で「どこか一箇所でもあるか」を見る。
            sufb = np.maximum.accumulate(Db[:, ::-1], axis=1)[:, ::-1]
            prea = np.maximum.accumulate(Da, axis=1)
            m = gg & (jb0 + 1 < W) & (ja0 - 1 >= 0)
            n_above += int((sufb[rows[m], jb0[m] + 1] > 0).sum())
            n_below += int((prea[rows[m], ja0[m] - 1] > 0).sum())
            del sufb, prea

            Qb = np.zeros((G, NLV))
            Qa = np.zeros((G, NLV))
            okc = gg.copy()
            for l in range(NLV):
                jb = jb0 - l * lb
                ja = ja0 + l * la
                o = gg & (jb >= 0) & (ja < W)
                okc &= o
                Qb[o, l] = Db[rows[o], jb[o]]
                Qa[o, l] = Da[rows[o], ja[o]]
            n_edge += int((gg & ~okc).sum())
            n_used += int(okc.sum())
            # ★最良気配の数量を bbo と突き合わせる(この再構成の質そのもの)
            mb = Qb[:, 0] * SZ_LOT
            ma = Qa[:, 0] * SZ_LOT
            bo = np.where(okc, qb1[g0:g1], np.nan)
            ao = np.where(okc, qa1[g0:g1], np.nan)
            d1 = np.concatenate([mb[okc] - bo[okc], ma[okc] - ao[okc]])
            l1_same += int((np.abs(d1) < 1e-6).sum())
            l1_more += int((d1 > 1e-6).sum())
            l1_less += int((d1 < -1e-6).sum())
            l1_mine += float(mb[okc].sum() + ma[okc].sum())
            l1_bbo += float(np.nansum(bo[okc]) + np.nansum(ao[okc]))
            l1ok[g0:g1] = okc & (np.abs(mb - bo) < 1e-6) & (np.abs(ma - ao) < 1e-6)
            zb = (Qb == 0) & okc[:, None]
            za = (Qa == 0) & okc[:, None]
            emp_b += zb.sum(axis=0)
            emp_a += za.sum(axis=0)
            n_val += (okc[:, None] & ~(zb & za)).sum(axis=0)
            n_pm1 += (zb ^ za).sum(axis=0)
            sum_b += np.where(okc[:, None], Qb, 0).sum(axis=0) * SZ_LOT
            sum_a += np.where(okc[:, None], Qa, 0).sum(axis=0) * SZ_LOT
            Qb[~okc] = np.nan
            Qa[~okc] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                s = Qb + Qa
                X[0, :, g0:g1] = np.where(s > 0, (Qb - Qa) / s, np.nan).T
                kb = np.cumsum(Qb, axis=1)
                ka = np.cumsum(Qa, axis=1)
                sc = kb + ka
                X[1, :, g0:g1] = np.where(sc > 0, (kb - ka) / sc, np.nan).T
            del Db, Da, Qb, Qa, s, kb, ka, sc
        if s1 > s0:
            np.add.at(dep_b, cp[cb] - p_min, cd[cb])
            np.add.at(dep_a, cp[~cb] - p_min, cd[~cb])

    meta = {"n_raw": n_raw, "n_open": n_open, "n_carry_in": carry.height,
            "n_carry_out": nxt.height, "n_rest_term": n_rest_term, "n_orphan": n_orph,
            "n_partial_fill": n_part, "n_neg_depth": n_neg,
            "l1_same": l1_same, "l1_more": l1_more, "l1_less": l1_less,
            "l1_mine_sz": l1_mine, "l1_bbo_sz": l1_bbo,
            "n_grid_ok": int(good.sum()), "n_used": n_used,
            "n_above_bid": n_above, "n_below_ask": n_below, "n_edge": n_edge}
    for l in range(NLV):
        meta[f"empty_bid_l{l+1}"] = int(emp_b[l])
        meta[f"empty_ask_l{l+1}"] = int(emp_a[l])
        meta[f"n_valid_l{l+1}"] = int(n_val[l])
        meta[f"n_pm1_l{l+1}"] = int(n_pm1[l])
        meta[f"sum_bid_l{l+1}"] = float(sum_b[l])
        meta[f"sum_ask_l{l+1}"] = float(sum_a[l])
    meta["l1_ok_grid"] = int(l1ok.sum())
    return X, xb, l1ok, mid, good, nxt, meta


def accumulate(X, xb, l1ok, mid, good, dt: str):
    """日ごとの回帰の累積和と帯別集計。マスクではなく 0 埋め + 内積で回す。"""
    ng = len(mid)
    lm = np.where(good, np.log(np.where(good, mid, 1.0)), np.nan)
    shift = ng // 3                                     # 帰無対照の巡回シフト(機械的)
    ys, fys = [], []
    for s in STEPS:
        y = np.full(ng, np.nan)
        y[: ng - s] = (lm[s:] - lm[: ng - s]) * 1e4
        ys.append(y)
        fys.append(np.isfinite(y))
    rows, brows = [], []
    for di in range(4):
        for l in range(NLV):
            if di == 2 and l > 0:               # bbo 由来は最良気配しか無い
                continue
            if di == 2:
                x0 = xb
            elif di == 3:
                # ★再構成の既知の弱点(黙って約定した注文の残りかす)を避けるため、
                #   最良の数量が bbo と厳密に一致した格子点だけに絞った対照。
                #   選別に使う量は t で観測できるので先読みにはならない。
                x0 = np.where(l1ok, X[0, l], np.nan).astype(np.float64)
            else:
                x0 = X[di, l].astype(np.float64)
            for lab, xx in (("実測", x0), ("帰無対照", np.roll(x0, shift))):
                fx = np.isfinite(xx)
                for hi, s in enumerate(STEPS):
                    m = fx & fys[hi]
                    n = int(m.sum())
                    if n < 2:
                        continue
                    a = np.where(m, xx, 0.0)
                    b = np.where(m, ys[hi], 0.0)
                    rows.append({"dt": dt, "kind": lab, "defn": DEFS[di], "lv": l + 1,
                                 "h": HLAB[hi], "h_ms": s * 100, "n": n,
                                 "sx": float(a.sum()), "sy": float(b.sum()),
                                 "sxx": float(a @ a), "sxy": float(a @ b),
                                 "syy": float(b @ b)})
                    if lab == "実測":
                        k = np.clip(((a + 1.0) * (NBIN / 2.0)).astype(np.int64),
                                    0, NBIN - 1)
                        k = np.where(m, k, NBIN)         # 無効点は捨て箱へ
                        cn = np.bincount(k, minlength=NBIN + 1)
                        cs = np.bincount(k, weights=b, minlength=NBIN + 1)
                        for j in range(NBIN):
                            if cn[j]:
                                brows.append({"dt": dt, "defn": DEFS[di], "lv": l + 1,
                                              "h": HLAB[hi], "h_ms": s * 100, "bin": j,
                                              "n": int(cn[j]), "sy": float(cs[j])})
    return pl.DataFrame(rows), pl.DataFrame(brows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = ROOT / "data" / f"l1_{tag}"
    files = sorted(src.glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"{src} が空。先に fetch_l1.py を実行すること")

    outs = {k: ROOT / "data" / f"obi_levels_{k}" / tag
            for k in ("days", "bins", "meta", "carry")}
    for p in outs.values():
        p.mkdir(parents=True, exist_ok=True)

    bbo, n_drop = clean_bbo(pl.read_parquet(ROOT / "data" / f"bbo_{tag}.parquet"))
    print(f"[bbo] 値段として信じられない行を {n_drop:,} 除いた -> {bbo.height:,} 行",
          file=sys.stderr)
    carry = pl.DataFrame(schema={"oid": pl.Int64, "is_bid": pl.Boolean,
                                 "pidx": pl.Int32, "size_after": pl.Int64})
    # ★板は日を跨ぐので途中から始められない。日末の carry を毎日残しておき、
    #   落ちたときはその続きから再開する(全日やり直すと 1 時間半かかる)。
    k = 0
    for i, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        if ((outs["days"] / f"dt={dt}.parquet").exists()
                and (outs["carry"] / f"dt={dt}.parquet").exists()):
            k = i + 1
        else:
            break
    if k:
        carry = pl.read_parquet(outs["carry"] / f"dt={files[k-1].stem.split('=')[1]}.parquet")
        print(f"[再開] {k} 日ぶんを飛ばし、繰越 {carry.height:,} 本から続ける",
              file=sys.stderr)
        files = files[k:]
    for fp in files:
        dt = fp.stem.split("=")[1]
        t = time.time()
        bd = bbo.filter(pl.col("dt") == dt).sort("ts")
        if bd.is_empty():
            print(f"  {dt} bbo 無し。飛ばす", file=sys.stderr)
            continue
        X, xb, l1ok, mid, good, carry, meta = day_features(fp, bd, carry)
        if not (outs["days"] / f"dt={dt}.parquet").exists():
            R, B = accumulate(X, xb, l1ok, mid, good, dt)
            R.write_parquet(outs["days"] / f"dt={dt}.parquet")
            B.write_parquet(outs["bins"] / f"dt={dt}.parquet")
        pl.DataFrame([{"dt": dt, **meta}]).write_parquet(outs["meta"] / f"dt={dt}.parquet")
        carry.write_parquet(outs["carry"] / f"dt={dt}.parquet")
        del X, xb, l1ok, mid, good
        print(f"  {dt} 使用格子 {meta['n_used']:,} 繰越 {meta['n_carry_out']:,} "
              f"孤児 {meta['n_orphan']:,}/{meta['n_rest_term']:,} 負 {meta['n_neg_depth']:,} "
              f"L1一致 {meta['l1_same']/max(meta['l1_same']+meta['l1_more']+meta['l1_less'],1):.1%} "
              f"倍率 {meta['l1_mine_sz']/max(meta['l1_bbo_sz'],1e-9):.3f} "
              f"{time.time()-t:.1f}s", file=sys.stderr)
    print(f"-> data/obi_levels_days/{tag}/", file=sys.stderr)


if __name__ == "__main__":
    main()
