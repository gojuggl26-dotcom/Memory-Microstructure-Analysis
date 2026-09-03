"""指値注文の生存時間(ハザード)を 6 つの条件で層別して数える。

【何を測るか】
板に置かれた指値注文 1 本を 1 つの「寿命」とみなし、置かれてからの経過時間 τ に
対して次の 3 つを出す。

    S(τ)      生存率      … τ の時点でまだ板に残っている割合
    h_cancel  取消ハザード … τ まで生きた注文が、次の瞬間に取り消される率(毎秒)
    h_fill    約定ハザード … τ まで生きた注文が、次の瞬間に約定する率(毎秒)

取消と約定は**競合リスク**である。どちらか一方が起きればその注文は消えるので、
「取消率」を単独で数えると約定した注文が母集団から抜けた分だけ歪む。そこで
原因別ハザード(cause-specific hazard)と累積発生確率(cumulative incidence)の
両方を出す。推定は打ち切りを扱える生命表(actuarial)方式で、区間内の打ち切りは
危険集合から半分だけ引く。

【層別に使う 6 つの条件】いずれも**発注の瞬間までに確定している量**だけを使う。

    queue   同じ価格に自分より前に並んでいた数量(契約)
    size    自分の注文数量(契約)
    dist    同じ側の最良気配から何ティック離れた場所に置いたか(負 = 気配を改善)
    wallet  その口座がそれまでに出した注文の累計本数(その注文より前だけを数える)
    vol     直前 300 秒の mid の実現ボラティリティ(bp)
    ofi     直前 1 秒の OFI を「自分の注文へ価格を押し付ける向き」で符号付けした値

【x が確定する時刻 / y の期間】
    x = 上の 6 つ … すべて発注時刻 t0 **以前**の情報だけで決まる。
        最良気配・ボラティリティ・OFI は t0 の直前までの bbo で打ち切る
        (同一 ns の行には自分自身の発注が含まれうるため ts < t0 で切る)。
        wallet の累計本数もその注文より前の分だけを数える。
    y = t0 以降の生存・取消・約定
先読みは無い。ビンの区切りは最初の 5 日で見た分布から決め、全期間へ同じものを
当てる(`--calibrate` で分位を出せる)。

【★踏んだ落とし穴】
- **ティック幅は 0.01 で固定ではない。** Hyperliquid の価格は有効数字 5 桁なので、
  1000 未満は 0.01、1000 以上は 0.1 になる。標本期間の mid は 197〜1254 でこの
  境界をまたぐ。0.01 に固定すると 1000 以上の日だけ距離が 10 倍に出る。
- **トリガー注文**(`is_trigger`)は発火まで板に載らないので母集団から外す。
  発火後の終端イベントは `is_trigger=False` で現れるため、終端側を属性で
  絞ると取りこぼす。終端の採否は「その oid の open を追跡していたか」で決める。
- **テイカー**(tif が Ioc / FrontendMarket / LiquidationMarket)は板に留まらない。
  `Alo` と `Gtc` だけを見る。
- **reduce_only** は板には載る(深さには数える)が、取引所がポジション減少に
  合わせて自動縮小するため終端の意味が曖昧。生存の母集団からは外す。
- 寿命ちょうど 0ns の注文が 1 割ある。ハザード率(毎秒)は定義できないので
  τ=0 の確率質量として別に持つ。
- 取得元に**欠測時間帯**がある(2026-05-18 など)。欠測中に消えた注文をそのまま
  数えると「長寿」に化けるので、欠測時間の先頭で生きている注文を右打ち切りに
  する。欠測の所在はパイプラインが残したサイドカー(`_stats.json` の
  `missing_hours`、`fetch_l1_extra.py` が落とす)を正とする。イベント間隔から
  推測すると、単に閑散だっただけの時間帯を誤って欠測と判定する。

【頑健性】
標本を発注日で前半・後半に割った計数も同時に出す(`hazard_cells_half_*`)。
プールした結果だけを見て「単調だ」と言うと、どこか 1 期間の異常が全体を
支配していても気づけない。

【出力】
    data/hazard_cells_<coin>.parquet    条件 × ビン × 時間ビン × 原因 の計数
    data/hazard_cells_half_<coin>.parquet  同じものを前半・後半に分けたもの
    data/hazard_strata_<coin>.csv       条件 × ビン の要約(件数・約定率・中央値)
    data/hazard_diag_<coin>.csv         日ごとの検算(孤児・欠測・繰越)

    uv run python scripts/build_hazard.py --coin xyz:MU --days 5 --calibrate
    uv run python scripts/build_hazard.py --coin xyz:MU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

OPEN = "open"
FILL = {"filled"}
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
          "liquidatedCanceled"}
RESTING_TIF = {"Alo", "Gtc"}
SC_OPEN, SC_FILL, SC_CANCEL, SC_OTHER = 0, 1, 2, -1
CAUSE_FILL, CAUSE_CANCEL, CAUSE_CENS = 0, 1, 2

PXQ = 100.0            # 価格を整数化する単位(0.01)。ティック幅とは別物
GAP_NS = 1800 * 10**9  # サイドカーが無いときの予備判定(30 分)
VOL_W = 300            # 実現ボラティリティの窓(秒)
OFI_W_NS = 10**9       # OFI の窓(1 秒)

# 時間ビン。★このデータの ts はブロック時刻で、同じ ts に平均 6.5 行が入り、
# 隣り合うブロックの間隔は中央値 74.9ms である。したがって寿命はブロック単位に
# 量子化されていて、10ms より細かいビンは構造的に空になる。
#   ビン 0 … 寿命ちょうど 0(= 同じブロックで発注と消滅。全体の約 9%)
#   ビン 1 … 1ns〜10ms(空のはず。検算用に残す)
#   以降  … 10ms〜27.8h を 0.25 dex 刻み
TAU_LOG = np.arange(7.0, 14.0001, 0.25)
TAU_EDGES = np.concatenate([[0.0, 1.0], 10.0 ** TAU_LOG])
NT = len(TAU_EDGES)
NEAR_TICK = 10.0        # 「板の上位で競っている」注文の定義(最良から 10 ティック)

# 区切りは最初の 5 日の分位(--calibrate)から丸めて決め、全期間へ同じ値を当てる。
Q_EDGES = [1e-4, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
Q_LAB = ["先頭 0", "0-1", "1-3", "3-10", "10-30", "30-100", "100-300", "300+"]

S_EDGES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
S_LAB = ["-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", "100+"]

D_EDGES = [0, 1, 3, 6, 11, 26, 51, 101, 301, 1001]
D_LAB = ["改善 <0", "0 最良", "1-2", "3-5", "6-10", "11-25", "26-50", "51-100",
         "101-300", "301-1000", "1001+"]

W_EDGES = [1, 10, 100, 1_000, 10_000, 100_000]
W_LAB = ["初回", "1-9", "10-99", "100-999", "1k-1万", "1万-10万", "10万+"]

V_EDGES = [7.5, 13.0, 20.0, 32.0, 54.0, 85.0]
V_LAB = ["-7.5bp", "7.5-13", "13-20", "20-32", "32-54", "54-85", "85bp+"]

O_EDGES = [-2.0, -1.0, -0.3, -1e-12, 1e-12, 0.3, 1.0, 2.0]
O_LAB = ["逆 2以上", "逆 1-2", "逆 0.3-1", "逆 0-0.3", "流量なし",
         "順 0-0.3", "順 0.3-1", "順 1-2", "順 2以上"]

N_LAB = [f"対照 {i}" for i in range(8)]     # 帰無対照(無作為に振った層)

# near_* は「最良から NEAR_TICK 以内に置かれた注文」に絞った同じ層別。
# 距離は他の 5 条件すべてと強く相関するので、周辺だけを見ると
# 「実は距離の効果だった」ものを取り違える。距離を揃えた比較を並べて出す。
COVS = {"all": ["全体"], "queue": Q_LAB, "size": S_LAB, "dist": D_LAB,
        "wallet": W_LAB, "vol": V_LAB, "ofi": O_LAB,
        "near_queue": Q_LAB, "near_size": S_LAB, "near_wallet": W_LAB,
        "near_vol": V_LAB, "near_ofi": O_LAB, "null": N_LAB}
COV_COL = {c: f"b_{c}" for c in COVS if c != "all"}


def tick_of(px: np.ndarray) -> np.ndarray:
    """Hyperliquid の価格は有効数字 5 桁。1000 未満 0.01 / 1000 以上 0.1。"""
    return np.where(px >= 1000.0, 0.1, 0.01)


# =========================================================================
# bbo の前処理(35M 行を毎日読み直すとメモリが足りないので日別 npz にする)
# =========================================================================
def prep_bbo(tag: str) -> Path:
    cache = ROOT / "data" / f"_hz_bbo_{tag}"
    if cache.exists() and any(cache.glob("*.npz")):
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    print("[bbo] 日別キャッシュを作る", file=sys.stderr)
    pf = pq.ParquetFile(ROOT / "data" / f"bbo_{tag}.parquet")
    buf: dict[str, list] = {}

    def flush(dt: str) -> None:
        b = pl.concat(buf.pop(dt)).sort("ts")
        np.savez(cache / f"{dt}.npz",
                 ts=b["ts"].to_numpy().astype(np.int64),
                 bid=b["best_bid"].to_numpy().astype(np.float64),
                 ask=b["best_ask"].to_numpy().astype(np.float64),
                 bsz=b["bid_sz"].to_numpy().astype(np.float32),
                 asz=b["ask_sz"].to_numpy().astype(np.float32))

    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["ts", "best_bid", "best_ask",
                                          "bid_sz", "ask_sz", "dt"]):
        b = pl.from_arrow(batch)
        b = b.filter((pl.col("best_ask") > pl.col("best_bid"))
                     & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0))
        if not b.height:
            continue
        cur = b["dt"].max()
        for key, sub in b.partition_by("dt", as_dict=True).items():
            dt = key[0] if isinstance(key, tuple) else key
            buf.setdefault(dt, []).append(sub)
        for dt in [k for k in buf if k < cur]:      # 過ぎた日は書き出して手放す
            flush(dt)
    for dt in list(buf):
        flush(dt)
    print(f"[bbo] -> {cache}", file=sys.stderr)
    return cache


def bbo_day(cache: Path, dt: str):
    p = cache / f"{dt}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in ("ts", "bid", "ask", "bsz", "asz")}


def ofi_series(b) -> np.ndarray:
    """Cont-Kukanov-Stoikov の OFI を bbo の隣り合う行から 1 本ずつ作る。"""
    pb, pa = b["bid"], b["ask"]
    vb, va = b["bsz"].astype(np.float64), b["asz"].astype(np.float64)
    pb0, pa0, vb0, va0 = pb[:-1], pa[:-1], vb[:-1], va[:-1]
    pb1, pa1, vb1, va1 = pb[1:], pa[1:], vb[1:], va[1:]
    e = ((pb1 >= pb0) * vb1 - (pb1 <= pb0) * vb0
         - ((pa1 <= pa0) * va1 - (pa1 >= pa0) * va0))
    return np.concatenate([[0.0], e])


def vol_grid(b, day_ns: int) -> np.ndarray:
    """1 秒格子の mid から直前 300 秒の実現ボラティリティ(bp)を作る。

    g[s] は「その日の s 秒目の終わりまで」で測った値。t0 に使うときは t0 の入る
    秒の 1 つ前を引くので、t0 以降の情報は入らない。
    """
    sec = np.clip((b["ts"] - day_ns) // 10**9, 0, 86399).astype(np.int64)
    mid = 0.5 * (b["bid"] + b["ask"])
    g = np.full(86400, np.nan)
    g[sec] = mid                        # ts 昇順なので各秒の最後の値が残る
    idx = np.arange(86400)
    last = np.where(np.isnan(g), -1, idx)
    np.maximum.accumulate(last, out=last)
    ok = last >= 0
    lm = np.where(ok, g[np.where(ok, last, 0)], np.nan)
    r = np.zeros(86400)
    good = np.zeros(86400, dtype=bool)
    good[1:] = ok[1:] & ok[:-1]
    r[good] = np.log(lm[1:][good[1:]] / lm[:-1][good[1:]])
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    lo = np.maximum(idx - VOL_W + 1, 0)
    out = np.sqrt(np.maximum(c[1:] - c[lo], 0.0)) * 1e4
    out[~ok] = np.nan
    return out


# =========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="層別に使う量の分位を出して終わる(区切りを決めるため)")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    srcdir = ROOT / "data" / f"l1_{tag}"
    files = sorted(srcdir.glob("dt=*.parquet"))
    if a.days:
        files = files[: a.days]
    if not files:
        sys.exit(f"{srcdir} が空。先に fetch_l1.py を実行すること")
    cache = prep_bbo(tag)
    # ★bbo が無い日は落とす。dist / vol / ofi が作れないため、その日を残すと
    #   「全体」とそれらの層とで母集団が食い違う(bbo は l2 由来で 1 日短い)。
    drop = [f.stem.split("=")[1] for f in files
            if not (cache / f"{f.stem.split('=')[1]}.npz").exists()]
    if drop:
        print(f"[load] bbo が無いので除外: {drop}", file=sys.stderr)
        files = [f for f in files if f.stem.split("=")[1] not in drop]
    userdir = ROOT / "data" / f"l1user_{tag}"
    has_user = userdir.exists() and any(userdir.glob("dt=*.parquet"))
    print(f"[load] {len(files)} 日分 / wallet 列 {'あり' if has_user else 'なし'}",
          file=sys.stderr)

    metaf = ROOT / "data" / f"l1_meta_{tag}.csv"
    miss = {}
    if metaf.exists():
        import json as _json
        for r in pl.read_csv(metaf).iter_rows(named=True):
            h = _json.loads(r["missing_hours"] or "null") or []
            miss[r["dt"]] = [int(x) for x in h if isinstance(x, int)]
        print(f"[meta] 欠測時間つき {sum(1 for v in miss.values() if v)} 日", file=sys.stderr)
    else:
        print("[meta] サイドカー未取得。イベント間隔 30 分で予備判定する",
              file=sys.stderr)

    cnt = {k: np.zeros((len(v), NT, 3), dtype=np.int64) for k, v in COVS.items()}
    # 前半・後半(発注日で割る)。プールした結果だけで単調性を語らないため。
    cnth = [{k: np.zeros((len(v), NT, 3), dtype=np.int64) for k, v in COVS.items()}
            for _ in range(2)]
    half_cut = files[len(files) // 2].stem.split("=")[1]
    calib: dict[str, list] = {k: [] for k in ("q", "s", "d", "v", "o", "w")}
    rng = np.random.default_rng(0)

    pending = None      # 前日までに open してまだ閉じていない注文(ビン済み)
    base = None         # 価格水準ごとの持ち越し数量
    ucarry = None       # 口座ごとの累計注文本数
    ofi_scale = None    # 前日の平均 |1 秒 OFI|(規格化用。前の日の値だけを使う)
    last_ts = None
    diag = []

    for fp in files:
        dt = fp.stem.split("=")[1]
        cur = cnth[0] if dt < half_cut else cnth[1]
        day_ns = int(np.datetime64(dt, "D").astype("datetime64[ns]").astype(np.int64))
        d = (pl.scan_parquet(fp)
             .select(
                 pl.col("ts").cast(pl.Int64),
                 pl.col("oid"),
                 (pl.col("side") == "B").alias("isbid"),
                 (pl.col("px") * PXQ).round().cast(pl.Int64).alias("pq"),
                 pl.col("px"),
                 pl.when(pl.col("status") == OPEN).then(SC_OPEN)
                   .when(pl.col("status").is_in(sorted(FILL))).then(SC_FILL)
                   .when(pl.col("status").is_in(sorted(CANCEL))).then(SC_CANCEL)
                   .otherwise(SC_OTHER).cast(pl.Int8).alias("sc"),
                 pl.col("orig_sz"),
                 pl.col("tif").is_in(sorted(RESTING_TIF)).alias("rest"),
                 pl.col("reduce_only"), pl.col("is_trigger"))
             .collect(engine="streaming")
             .with_row_index("i"))

        ts_all = d["ts"].to_numpy()
        if metaf.exists():
            gap_starts = [day_ns + h * 3600 * 10**9 for h in miss.get(dt, [])]
        else:
            gi = np.flatnonzero(np.diff(ts_all) > GAP_NS)
            gap_starts = [int(x) for x in ts_all[gi]]
        n_gapcens = 0
        if last_ts is not None and ts_all[0] - last_ts > GAP_NS:
            pending, k = censor_at([cnt, cnth[0], cnth[1]], pending, int(last_ts))
            n_gapcens += k                       # 日そのものが欠けている場合
        last_ts = int(ts_all[-1])

        opens = (d.filter((pl.col("sc") == SC_OPEN) & pl.col("rest")
                          & ~pl.col("is_trigger"))
                 .with_row_index("oi"))
        terms = d.filter(pl.col("sc").is_in([SC_FILL, SC_CANCEL]))

        ids = [opens.select("oid")]
        if pending is not None and pending.height:
            ids.append(pending.select("oid"))
        term_ok = terms.join(pl.concat(ids).unique(), on="oid", how="semi")
        n_orph = terms.height - term_ok.height
        # 孤児の大半はテイカー(Ioc など)の終端。板に載らないので open が無い。
        n_orph_rest = (terms.join(pl.concat(ids).unique(), on="oid", how="anti")
                       .filter(pl.col("rest")).height)

        # ---- 板の深さ ------------------------------------------------------
        # 追加 = 今日の open / 削除 = 追跡していた注文の終端。行番号順に累積すると
        # その行の直前の深さが出る。前日からの持ち越しは base で足す。
        delta = pl.concat([
            opens.select("i", "isbid", "pq", pl.col("orig_sz").alias("dz"), "oi"),
            term_ok.select("i", "isbid", "pq", (-pl.col("orig_sz")).alias("dz"),
                           pl.lit(None, dtype=pl.UInt32).alias("oi")),
        ]).sort("i").with_columns(cum=pl.col("dz").cum_sum().over(["isbid", "pq"]))
        if base is not None:
            delta = (delta.join(base, on=["isbid", "pq"], how="left")
                     .with_columns(cum=pl.col("cum") + pl.col("b0").fill_null(0.0))
                     .drop("b0"))
        qah = (delta.filter(pl.col("oi").is_not_null()).sort("oi")
               ["cum"].to_numpy() - opens["orig_sz"].to_numpy())
        n_negq = int((qah < -1e-9).sum())
        qah = np.maximum(qah, 0.0)

        touched = delta.group_by(["isbid", "pq"]).agg(
            pl.col("cum").sort_by("i").last().alias("b0"))
        base = (touched if base is None else pl.concat(
            [touched, base.join(touched.select("isbid", "pq"),
                                on=["isbid", "pq"], how="anti")]))

        # ---- bbo から 最良気配 / ボラティリティ / OFI ----------------------
        t0 = opens["ts"].to_numpy()
        n = len(t0)
        dist = np.full(n, np.nan)
        vol = np.full(n, np.nan)
        ofi = np.full(n, np.nan)
        n_xnew = 0
        b = bbo_day(cache, dt)
        if b is not None and len(b["ts"]) > 2 and n:
            j = np.searchsorted(b["ts"], t0, side="left") - 1      # ts < t0 の最後
            ok = j >= 0
            jj = np.where(ok, j, 0)
            isb = opens["isbid"].to_numpy()
            px = opens["px"].to_numpy()
            ref = np.where(isb, b["bid"][jj], b["ask"][jj])
            dist = np.where(ok, np.where(isb, ref - px, px - ref) / tick_of(ref), np.nan)
            # 検算: bbo(l2 由来)と l1 が食い違っていないか。板に載る注文が
            # 反対側の最良を跨いでいたら、どちらかが古い。多ければ dist は信用できない。
            opp = np.where(isb, b["ask"][jj], b["bid"][jj])
            n_xnew = int((ok & np.where(isb, px >= opp, px <= opp)).sum())
            g = vol_grid(b, day_ns)
            s = (t0 - day_ns) // 10**9 - 1
            vol = np.where((s >= 0) & (s < 86400), g[np.clip(s, 0, 86399)], np.nan)
            ce = np.concatenate([[0.0], np.cumsum(ofi_series(b))])
            j1 = np.searchsorted(b["ts"], t0 - OFI_W_NS, side="left")
            raw = np.where(ok, ce[jj + 1] - ce[j1], np.nan)
            m = float(np.nanmean(np.abs(raw))) if np.isfinite(raw).any() else np.nan
            sc = ofi_scale if (ofi_scale and np.isfinite(ofi_scale)) else m
            if not (sc and np.isfinite(sc) and sc > 0):
                sc = 1.0
            ofi = np.where(isb, -raw, raw) / sc
            if np.isfinite(m) and m > 0:
                ofi_scale = m

        # ---- wallet(その注文より前の、その口座の累計注文本数)-------------
        wcnt = np.full(n, -1, dtype=np.int64)
        up = userdir / f"dt={dt}.parquet"
        if has_user and up.exists() and n:
            um = pl.read_parquet(up)                       # oid, wid
            cw = (opens.select("oid").with_row_index("r")
                  .join(um, on="oid", how="left").sort("r")
                  .with_columns(pl.int_range(pl.len()).over("wid").alias("k")))
            if ucarry is not None:
                cw = (cw.join(ucarry, on="wid", how="left")
                      .with_columns((pl.col("k") + pl.col("c0").fill_null(0)).alias("k"))
                      .drop("c0").sort("r"))
            wcnt = np.where(cw["wid"].is_null().to_numpy(), -1, cw["k"].to_numpy())
            add = cw.drop_nulls("wid").group_by("wid").len().rename({"len": "c0"})
            ucarry = (add if ucarry is None else
                      pl.concat([ucarry, add]).group_by("wid").agg(pl.col("c0").sum()))

        szo = opens["orig_sz"].to_numpy()
        if a.calibrate and n:
            m = rng.random(n) < 0.02
            calib["q"].append(qah[m]); calib["s"].append(szo[m])
            calib["d"].append(dist[m]); calib["v"].append(vol[m])
            calib["o"].append(ofi[m]); calib["w"].append(wcnt[m].astype(float))

        bq = np.searchsorted(Q_EDGES, qah, side="right").astype(np.int8)
        bs = np.searchsorted(S_EDGES, szo, side="right").astype(np.int8)
        bd = bin_dist(dist)
        bw = bin_wallet(wcnt)
        bv = bin_edges(vol, V_EDGES)
        bo = bin_edges(ofi, O_EDGES)
        near = np.isfinite(dist) & (dist <= NEAR_TICK)
        nz = lambda x: np.where(near, x, -1).astype(np.int8)
        newp = opens.select("oid", pl.col("ts").alias("t0")).with_columns(
            b_queue=pl.Series(bq), b_size=pl.Series(bs), b_dist=pl.Series(bd),
            b_wallet=pl.Series(bw), b_vol=pl.Series(bv), b_ofi=pl.Series(bo),
            b_near_queue=pl.Series(nz(bq)), b_near_size=pl.Series(nz(bs)),
            b_near_wallet=pl.Series(nz(bw)), b_near_vol=pl.Series(nz(bv)),
            b_near_ofi=pl.Series(nz(bo)),
            b_null=pl.Series(rng.integers(0, 8, n).astype(np.int8)),
            keep=pl.Series(~opens["reduce_only"].to_numpy()))
        pending = newp if pending is None else pl.concat([pending, newp])

        # ---- 終端とつき合わせて計上 ---------------------------------------
        tt = (term_ok.select("oid", pl.col("ts").alias("t1"), pl.col("sc").alias("scx"))
              .unique(subset="oid", keep="first"))
        mt = pending.join(tt, on="oid", how="inner")
        pending = pending.join(tt.select("oid"), on="oid", how="anti")
        if mt.height:
            cause = np.where(mt["scx"].to_numpy() == SC_FILL,
                             CAUSE_FILL, CAUSE_CANCEL).astype(np.int8)
            accumulate(cnt, mt, mt["t1"].to_numpy(), cause)
            accumulate(cur, mt, mt["t1"].to_numpy(), cause)   # 前半 / 後半

        for g0 in gap_starts:                 # 日中の欠測をまたいだ分を打ち切る
            pending, k = censor_at([cnt, cnth[0], cnth[1]], pending, int(g0))
            n_gapcens += k

        diag.append({"dt": dt, "opens": opens.height, "terms": terms.height,
                     "orphan": n_orph, "orphan_resting": n_orph_rest,
                     "matched": mt.height,
                     "pending": pending.height, "gaps": len(gap_starts),
                     "gap_censored": n_gapcens, "neg_queue": n_negq,
                     "no_bbo": int(np.isnan(dist).sum()), "cross_new": n_xnew})
        print(f"  {dt} open {opens.height:>9,} 照合 {mt.height:>9,} "
              f"孤児 {n_orph:>7,} 繰越 {pending.height:>7,}"
              + (f"  ★欠測 {len(gap_starts)} 打切 {n_gapcens:,}" if gap_starts or n_gapcens else ""),
              file=sys.stderr)
        del d, delta, opens, terms, term_ok, mt

    if pending is not None and pending.height:      # 標本の末尾は右打ち切り
        censor_at([cnt, cnth[0], cnth[1]], pending, int(last_ts) + 1)

    if a.calibrate:
        print("\n=== 分位(ビンの区切りを決めるため)===", file=sys.stderr)
        names = {"q": "queue 前の数量", "s": "size 自分の数量", "d": "dist ティック",
                 "v": "vol bp", "o": "ofi 規格化", "w": "wallet 累計本数"}
        ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for k, lab in names.items():
            v = np.concatenate(calib[k]) if calib[k] else np.array([])
            v = v[np.isfinite(v)]
            if not len(v):
                print(f"  {lab}: 標本なし", file=sys.stderr)
                continue
            q = np.percentile(v, ps)
            print(f"  {lab:<16} n={len(v):>9,} " +
                  " ".join(f"{p}%={x:.4g}" for p, x in zip(ps, q)), file=sys.stderr)
        return

    def cells(acc, extra=None):
        rows = []
        for cov, labs in COVS.items():
            c = acc[cov]
            for bi, lab in enumerate(labs):
                for tj in range(NT):
                    f, k, z = c[bi, tj]
                    if f == k == z == 0:
                        continue
                    r = {"cov": cov, "bin": bi, "bin_lab": lab, "tbin": tj,
                         "tau_lo": float(TAU_EDGES[tj]),
                         "tau_hi": float(TAU_EDGES[tj + 1]) if tj + 1 < NT else np.inf,
                         "d_fill": int(f), "d_cancel": int(k), "d_cens": int(z)}
                    if extra:
                        r.update(extra)
                    rows.append(r)
        return rows

    pl.DataFrame(cells(cnt)).write_parquet(ROOT / "data" / f"hazard_cells_{tag}.parquet")
    pl.DataFrame(cells(cnth[0], {"half": 0}) + cells(cnth[1], {"half": 1})).write_parquet(
        ROOT / "data" / f"hazard_cells_half_{tag}.parquet")
    print(f"[頑健性] 前半 = {files[0].stem.split('=')[1]}〜 / "
          f"後半 = {half_cut}〜{files[-1].stem.split('=')[1]}", file=sys.stderr)
    pl.DataFrame(diag).write_csv(ROOT / "data" / f"hazard_diag_{tag}.csv")
    S = summarize(cnt)
    S.write_csv(ROOT / "data" / f"hazard_strata_{tag}.csv")
    print("\n=== 条件別の要約 ===", file=sys.stderr)
    with pl.Config(tbl_rows=80, tbl_cols=12, fmt_str_lengths=20):
        print(S, file=sys.stderr)
    print(f"\n-> data/hazard_cells_{tag}.parquet / hazard_strata_{tag}.csv",
          file=sys.stderr)


# =========================================================================
def bin_dist(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), -1, dtype=np.int8)
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    out[ok] = np.where(x < 0, 0, np.searchsorted(D_EDGES, x, side="right"))[ok]
    return out


def bin_wallet(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), -1, dtype=np.int8)
    ok = v >= 0
    out[ok] = np.searchsorted(W_EDGES, v[ok], side="right")
    return out


def bin_edges(v: np.ndarray, edges) -> np.ndarray:
    out = np.full(len(v), -1, dtype=np.int8)
    ok = np.isfinite(v)
    out[ok] = np.searchsorted(edges, v[ok], side="right")
    return out


def censor_at(cnts, pending, t: int):
    """時刻 t より前に置かれた注文をそこで右打ち切りにして計上する。

    打ち切りは前半・後半どちらの累計器にも入れる必要があるので、
    累計器はリストで受ける(前半に置かれた注文が後半に打ち切られうる)。
    """
    if pending is None or not pending.height:
        return pending, 0
    cut = pending.filter(pl.col("t0") < t)
    if not cut.height:
        return pending, 0
    for c in (cnts if isinstance(cnts, list) else [cnts]):
        accumulate(c, cut, np.full(cut.height, t, dtype=np.int64),
                   np.full(cut.height, CAUSE_CENS, dtype=np.int8))
    return pending.filter(pl.col("t0") >= t), cut.height


def accumulate(cnt, tbl: pl.DataFrame, t_end: np.ndarray, cause: np.ndarray) -> None:
    """1 件 = 1 注文。寿命を時間ビンに落として (条件, ビン, 時間ビン, 原因) を数える。"""
    life = np.maximum(t_end - tbl["t0"].to_numpy(), 0)
    tb = np.clip(np.searchsorted(TAU_EDGES, life, side="right") - 1, 0, NT - 1)
    keep = tbl["keep"].to_numpy()
    cause = cause.astype(np.int64)
    np.add.at(cnt["all"].reshape(-1), (tb * 3 + cause)[keep], 1)
    for cov, col in COV_COL.items():
        b = tbl[col].to_numpy().astype(np.int64)
        m = keep & (b >= 0)
        if not m.any():
            continue
        bb = np.clip(b[m], 0, cnt[cov].shape[0] - 1)
        np.add.at(cnt[cov].reshape(-1), (bb * NT + tb[m]) * 3 + cause[m], 1)


def summarize(cnt) -> pl.DataFrame:
    rows = []
    for cov, labs in COVS.items():
        for bi, lab in enumerate(labs):
            c = cnt[cov][bi]
            f, k, z = int(c[:, 0].sum()), int(c[:, 1].sum()), int(c[:, 2].sum())
            n = f + k + z
            if n == 0:
                continue
            ex = np.cumsum(c[:, 0] + c[:, 1])
            med = float("nan")
            if ex[-1] > 0:                      # ビン内は log 時間で線形に按分
                j = int(np.searchsorted(ex, ex[-1] / 2))
                j = min(j, NT - 1)
                lo, hi = TAU_EDGES[j], (TAU_EDGES[j + 1] if j + 1 < NT else TAU_EDGES[j] * 10)
                prev = ex[j - 1] if j else 0
                w = (ex[-1] / 2 - prev) / max(ex[j] - prev, 1)
                med = (float(lo) if lo <= 0 else
                       float(np.exp(np.log(lo) + w * np.log(hi / lo))))
            rows.append({"cov": cov, "bin": bi, "bin_lab": lab, "n": n,
                         "n_fill": f, "n_cancel": k, "n_cens": z,
                         "fill_share": f / n, "cancel_share": k / n,
                         "median_life_ms": med / 1e6})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    main()
