"""板に出ている数量が何者の口座に集まっているか(wallet concentration)を測る。

【何を測るか】
1 秒ごとに板を復元し、その瞬間に**板へ数量を置いている口座**ごとの残高
Q_i からシェア w_i = Q_i / Σ_j Q_j を作り、14 の指標を出す。
分母はいずれも**その瞬間に板にある数量**であって、その日に出された注文の
総量ではない(後者は[束の間の注文](mu_fleeting_report.md)の分母)。

    number of active wallets  買いか売りに数量を置いている口座の数
    bid wallet count          買い側に数量を置いている口座の数
    ask wallet count          売り側に数量を置いている口座の数
    BBO wallet count          最良買・最良売のどちらかに数量を置いている口座の数

    wallet HHI                Σ w_i^2(1 者独占で 1、n 者均等で 1/n)
    wallet entropy            −Σ w_i ln w_i(実効口座数 exp(H) も併記)
    Gini concentration        正の残高を持つ口座だけを母集団としたジニ係数
    top-1 / top-3 / top-5 / top-10 wallet share

    concentration imbalance   (HHI_bid − HHI_ask) / (HHI_bid + HHI_ask)
    touch concentration       最良気配に置かれている数量だけで測った HHI
    deep-book concentration   最良気配以外に置かれている数量だけで測った HHI

【なぜ 1 秒格子か】
板そのものの状態を測るので、注文の出入りではなく**ある瞬間の残高**を見る。
パイプラインの L2 スナップショットと同じ 1 秒を使う。指値の寿命は中央値 565ms
なので、1 秒格子に写るのは**滞留している注文だけ**である。これは意図した挙動で、
瞬間的に出し入れされる注文は板の厚みとして誰かが取れるものではない。

実測(2026-07-15)では、ある瞬間に生きている指値は 829〜2,626 本、
口座は 135〜344 者、**最良気配に居る注文は 2〜5 本(口座 2〜3 者)**しかない。

【板の再構成】
[ティック水準別 OBI](mu_obi_levels_report.md) と同じく、注文ごとの
「板に置かれている数量」の階段関数を作って差分を取る。数量は 0.001 を 1 とする
整数で持ち(浮動小数だと空の価格に残りかすが残る)、同一 ns の行順は
ゲート2 D4 の論理順で並べ直す。母集団はトリガー注文・テイカー(Ioc など)を
除いた Alo / Gtc の指値。**reduce_only は板に載るので含める**
(ハザードや束の間の分析とはここが違う。あちらは終端の意味が曖昧なので外した。
板の厚みを数えるここでは、載っている以上は数えないと実態と合わない)。

【x が確定する時刻 / y の期間】
記述統計であって予測ではない。格子点 g の板は **バケット g−1 までの差分の累積**
とし、最良気配も t_g 未満の最後の bbo 行を使う。将来の情報は入っていない。

    uv run python scripts/build_wallet_conc.py --coin xyz:MU
出力: data/wallet_conc_daily_<coin>.parquet / .csv … 日 × 指標の平均と分位
      data/wallet_conc_hourly_<coin>.parquet       … 日 × 時 × 指標の平均
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

GRID_NS = 1_000_000_000                 # 1 秒
DAY_NS = 86_400_000_000_000
CHUNK_G = 1_800                         # 30 分ぶんの格子点をまとめて処理する
PX_UNIT = 0.01
SZ_LOT = 0.001
RESTING = ["Alo", "Gtc"]
CANCELS = ["canceled", "reduceOnlyCanceled", "selfTradeCanceled",
           "siblingFilledCanceled", "marginCanceled", "scheduledCancel",
           "liquidatedCanceled", "openInterestCapCanceled",
           "outcomeSettledCanceled", "vaultWithdrawalCanceled"]
TERMINAL = ["filled"] + CANCELS
# ★終端イベントが最後まで現れない注文がある(約定の一部は filled を出さない)。
#   放っておくと板から永久に消えず、繰越が 1,333 → 16,183 本まで積み上がり、
#   活動口座が実測の 135〜344 者に対し 1,708 者まで水増しされた。
#   実測(2 日・1,715 万本)では **観測された寿命の最大が 20.5 時間**、
#   1 日を超えたものは 1 本も無い(99.99% 点 2.4 時間)。したがって
#   24 時間まったく動きの無い注文は「消えたのに記録が無いもの」とみなして落とす。
MAX_REST_NS = 86_400_000_000_000

METRICS = ["n_active", "n_bid", "n_ask", "n_bbo", "hhi", "entropy", "eff_n",
           "gini", "top1", "top3", "top5", "top10", "conc_imb", "hhi_touch",
           "hhi_deep", "touch_share", "depth"]
QS = [10, 25, 50, 75, 90]


def day_events(fp: Path, wal: Path, carry: pl.DataFrame, t0: int):
    """1 日分の注文イベントを (格子, 価格, 側, 口座, 差分) の列に直す。"""
    d = pl.read_parquet(fp, columns=["ts", "oid", "side", "px", "status", "orig_sz",
                                     "remaining_sz", "tif", "is_trigger"])
    ev = (d.filter((~pl.col("is_trigger")) & pl.col("tif").is_in(RESTING)
                   & pl.col("status").is_in(["open"] + TERMINAL))
          .select(oid=pl.col("oid").cast(pl.Int64),
                  ts=pl.col("ts").cast(pl.Int64),
                  isbid=(pl.col("side") == "B"),
                  pidx=(pl.col("px") / PX_UNIT).round().cast(pl.Int32),
                  sz=(pl.col("remaining_sz") / SZ_LOT).round().cast(pl.Int64),
                  rk=pl.when(pl.col("status") == "open").then(0)
                     .when(pl.col("status") == "filled").then(2)
                     .otherwise(3).cast(pl.Int8),
                  gone=pl.col("status").is_in(CANCELS)))
    del d
    n_ev = ev.height
    ev = (ev.with_columns(size_after=pl.when(pl.col("gone")).then(0)
                          .otherwise(pl.col("sz")), synth=pl.lit(False))
          .drop("sz", "gone"))
    # ★l1user は **open 行だけ**の (oid, wid) なので、前日から繰り越した注文の
    #   oid はこの日のファイルに無い。carry の (oid, wid) を足しておかないと、
    #   繰越注文の終端行に wid が付かず null として捨てられ、その注文は
    #   板から永久に消えなくなる(実際に踏んだ。板が単調に膨らみ、
    #   活動口座が 176 → 3,041 まで積み上がった)。
    u = pl.read_parquet(wal).select("oid", "wid")
    if carry.height:
        u = pl.concat([u, carry.select("oid", "wid")])
    u = u.unique(subset=["oid"])
    ev = ev.join(u, on="oid", how="left")
    n_nowid = int(ev["wid"].is_null().sum())
    ev = ev.filter(pl.col("wid").is_not_null())
    if carry.height:
        # ★最後に動いた時刻をそのまま使う。t0-1 で上書きすると毎日更新されて
        #   しまい、24 時間の足切りが永久に効かなくなる。
        cr = carry.select("oid", "isbid", "pidx", "wid",
                          ts=pl.col("ts_last"), rk=pl.lit(-1, pl.Int8),
                          size_after=pl.col("size_after"), synth=pl.lit(True))
        ev = pl.concat([cr.select(ev.columns), ev])
    ev = ev.sort(["oid", "ts", "rk"])
    ev = ev.with_columns(
        prev=pl.col("size_after").shift(1).over("oid").fill_null(0),
        last=pl.col("oid") != pl.col("oid").shift(-1))
    nxt = (ev.filter(pl.col("last").fill_null(True) & (pl.col("size_after") > 0))
           .select("oid", "isbid", "pidx", "wid", "size_after",
                   ts_last=pl.col("ts")))
    ng = DAY_NS // GRID_NS
    ev = (ev.with_columns(delta=pl.col("size_after") - pl.col("prev"))
          .filter((~pl.col("synth")) & (pl.col("delta") != 0))
          .with_columns(gi=((pl.col("ts") - t0) // GRID_NS)
                        .clip(0, ng - 1).cast(pl.Int32))
          .select("gi", "pidx", "isbid", "wid", "delta").sort("gi"))
    return ev, nxt, n_ev, n_nowid


def cheap(Q: np.ndarray, tot: np.ndarray | None = None):
    """並べ替えの要らない指標(口座数・HHI・エントロピー)。"""
    if tot is None:
        tot = Q.sum(1)
    ok = tot > 0
    n = (Q > 0).sum(1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = Q / np.where(tot[:, None] > 0, tot[:, None], 1.0)
        hhi = (W * W).sum(1)
        ent = -np.where(W > 0, W * np.log(np.where(W > 0, W, 1.0)), 0.0).sum(1)
    for v in (n, hhi, ent):
        v[~ok] = np.nan
    return n, hhi, ent


def full(Q: np.ndarray):
    """★並べ替えが要る指標(ジニ・上位 k)。1 チャンクにつき 1 回だけ呼ぶ。

    ジニは**正の残高を持つ口座だけ**を母集団とする。0 の口座を含めると
    「その日に一度でも板に出した口座」の数だけ係数が水増しされ、母集団の定義が
    瞬間ごとに変わってしまう。昇順に並べると 0 は前に来るので、正の値の順位は
    k − (N − n) になり、0 を含めたまま次の形で計算できる。
    """
    tot = Q.sum(1)
    n, hhi, ent = cheap(Q, tot)
    N = Q.shape[1]
    S = np.sort(Q, axis=1)
    k = np.arange(1, N + 1)[None, :]
    n0 = np.nan_to_num(n)
    ok = tot > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        gini = (((2 * k - 2 * N + n0[:, None] - 1) * S).sum(1)
                / np.maximum(n0 * tot, 1e-12))
        cum = np.cumsum(S[:, ::-1][:, :10], axis=1) / np.where(
            tot[:, None] > 0, tot[:, None], 1.0)
    gini[~ok] = np.nan
    return {"n_active": n, "hhi": hhi, "entropy": ent, "eff_n": np.exp(ent),
            "gini": gini, "top1": np.where(ok, cum[:, 0], np.nan),
            "top3": np.where(ok, cum[:, 2], np.nan),
            "top5": np.where(ok, cum[:, 4], np.nan),
            "top10": np.where(ok, cum[:, 9], np.nan)}


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
    wdir = ROOT / "data" / f"l1user_{tag}"
    outd = ROOT / "data" / f"wallet_conc_days_{tag}"
    outd.mkdir(parents=True, exist_ok=True)

    carry = pl.DataFrame(schema={"oid": pl.Int64, "isbid": pl.Boolean,
                                 "pidx": pl.Int32, "wid": pl.Int32,
                                 "size_after": pl.Int64, "ts_last": pl.Int64})
    k0 = 0
    for i, fp in enumerate(files):
        dt = fp.stem.split("=")[1]
        if all((outd / f"{w}_{dt}.parquet").exists()
               for w in ("daily", "hourly", "carry")):
            k0 = i + 1
        else:
            break
    if k0:
        carry = pl.read_parquet(outd / f"carry_{files[k0-1].stem.split('=')[1]}.parquet")
        print(f"[再開] {k0} 日ぶんを飛ばす(繰越 {carry.height:,} 本)", file=sys.stderr)

    ng = DAY_NS // GRID_NS
    gh = np.arange(ng) // 3600
    for fp in files[k0:]:
        dt = fp.stem.split("=")[1]
        t = time.time()
        bd = bbo.filter(pl.col("dt") == dt).sort("ts")
        if bd.is_empty():
            print(f"  {dt} bbo 無し。飛ばす", file=sys.stderr)
            continue
        if not (wdir / f"dt={dt}.parquet").exists():
            print(f"  {dt} 口座情報 無し。飛ばす", file=sys.stderr)
            continue
        t0 = (int(pl.scan_parquet(fp).select(pl.col("ts").cast(pl.Int64).min())
                  .collect().item()) // DAY_NS) * DAY_NS
        n_stale = 0
        if carry.height:                      # 24 時間動きの無い注文を落とす
            keep = carry.filter(t0 - pl.col("ts_last") <= MAX_REST_NS)
            n_stale = carry.height - keep.height
            carry = keep
        prev = carry
        ev, carry, n_ev, n_nowid = day_events(fp, wdir / f"dt={dt}.parquet",
                                              prev, t0)

        # ---- 最良気配を格子へ(厳密に t 未満)--------------------------------
        bts = bd["ts"].cast(pl.Int64).to_numpy()
        tg = t0 + np.arange(ng, dtype=np.int64) * GRID_NS
        j = np.searchsorted(bts, tg, side="left") - 1
        gok = j >= 0
        j = np.where(gok, j, 0)
        bbg = np.where(gok, np.rint(bd["best_bid"].to_numpy()[j] / PX_UNIT),
                       -1).astype(np.int64)
        bag = np.where(gok, np.rint(bd["best_ask"].to_numpy()[j] / PX_UNIT),
                       -1).astype(np.int64)

        gi = ev["gi"].to_numpy().astype(np.int64)
        pidx = ev["pidx"].to_numpy().astype(np.int64)
        isb = ev["isbid"].to_numpy()
        widv = ev["wid"].to_numpy().astype(np.int64)
        dv = ev["delta"].to_numpy().astype(np.float64)
        del ev
        # ★前日から残っている注文も口座と (価格, 側, 口座) の台帳に入れる。
        #   入れないと繰越ぶんの残高を置く場所が無く、板が薄く出る。
        e = np.empty(0, np.int64)
        c_w = prev["wid"].to_numpy().astype(np.int64) if prev.height else e
        c_p = prev["pidx"].to_numpy().astype(np.int64) if prev.height else e
        c_b = prev["isbid"].to_numpy() if prev.height else np.empty(0, bool)
        c_s = (prev["size_after"].to_numpy().astype(np.float64) if prev.height
               else np.empty(0, np.float64))
        uw, inv = np.unique(np.concatenate([widv, c_w]), return_inverse=True)
        W = len(uw)
        wi, cwi = inv[: len(widv)], inv[len(widv):]
        key = np.concatenate([(pidx * 2 + isb) * W + wi,
                              (c_p * 2 + c_b) * W + cwi])
        upw, pinv = np.unique(key, return_inverse=True)
        pwi, cpwi = pinv[: len(widv)], pinv[len(widv):]
        pw_key = upw // W                            # (価格 × 2 + 側)
        dep_pair = np.zeros(len(upw), np.float64)
        dep_b = np.zeros(W, np.float64)
        dep_a = np.zeros(W, np.float64)
        if prev.height:
            np.add.at(dep_b, cwi[c_b], c_s[c_b])
            np.add.at(dep_a, cwi[~c_b], c_s[~c_b])
            np.add.at(dep_pair, cpwi, c_s)

        acc = {m: [] for m in METRICS}
        hrows = []
        chk = [0, 0, 0.0, 0.0]
        bd_qb = np.where(gok, bd["bid_sz"].to_numpy()[j], np.nan)
        for c0 in range(0, ng, CHUNK_G):
            c1 = min(c0 + CHUNK_G, ng)
            G = c1 - c0
            s0, s1 = np.searchsorted(gi, [c0, c1])
            cg, cp, cb, cw, cd = (gi[s0:s1], pidx[s0:s1], isb[s0:s1],
                                  wi[s0:s1], dv[s0:s1])
            cpw = pwi[s0:s1]
            zr = np.zeros((1, W))
            Qb = dep_b + np.concatenate([zr, np.cumsum(np.bincount(
                (cg[cb] - c0) * W + cw[cb], weights=cd[cb],
                minlength=G * W).reshape(G, W), axis=0)[:-1]])
            Qa = dep_a + np.concatenate([zr, np.cumsum(np.bincount(
                (cg[~cb] - c0) * W + cw[~cb], weights=cd[~cb],
                minlength=G * W).reshape(G, W), axis=0)[:-1]])
            np.maximum(Qb, 0, out=Qb)
            np.maximum(Qa, 0, out=Qa)
            Q = Qb + Qa
            m = full(Q)
            nb, hb, _ = cheap(Qb)
            na, ha, _ = cheap(Qa)
            m["n_bid"], m["n_ask"] = nb, na
            with np.errstate(invalid="ignore"):
                m["conc_imb"] = (hb - ha) / np.where(hb + ha > 0, hb + ha, np.nan)

            # ---- 最良気配に居る口座。価格ごとに (価格,側,口座) の台帳から引く
            Tt = np.zeros((G, W))
            Tb = np.zeros(G)
            for sb, bg in ((True, bbg[c0:c1]), (False, bag[c0:c1])):
                before = Tt.sum(1).copy()
                for p in np.unique(bg[bg >= 0]):
                    kp = p * 2 + int(sb)
                    lo = int(np.searchsorted(pw_key, kp, side="left"))
                    hi = int(np.searchsorted(pw_key, kp, side="right"))
                    if hi <= lo:
                        continue
                    ws = (upw[lo:hi] % W).astype(np.int64)   # この価格に居る口座
                    rp = np.flatnonzero(bg == p)
                    base = dep_pair[lo:hi]
                    sel = (cp == p) & (cb == sb)
                    if sel.any():
                        loc = cpw[sel] - lo                  # 台帳内の局所位置
                        A = np.bincount((cg[sel] - c0) * (hi - lo) + loc,
                                        weights=cd[sel],
                                        minlength=G * (hi - lo)).reshape(G, hi - lo)
                        Dp = base + np.concatenate(
                            [np.zeros((1, hi - lo)), np.cumsum(A, axis=0)[:-1]])
                    else:
                        Dp = np.broadcast_to(base, (G, hi - lo))
                    Tt[np.ix_(rp, ws)] += np.maximum(Dp[rp], 0)
                if sb:
                    Tb = Tt.sum(1) - before
            # ★再構成した最良買の数量を bbo と突き合わせる。板が壊れたら必ずここに出る
            qb = bd_qb[c0:c1]
            mm = np.isfinite(qb) & (bbg[c0:c1] >= 0)
            chk[0] += int(mm.sum())
            chk[1] += int((np.abs(Tb[mm] * SZ_LOT - qb[mm]) < 1e-6).sum())
            chk[2] += float((Tb[mm] * SZ_LOT).sum())
            chk[3] += float(qb[mm].sum())
            nt, ht, _ = cheap(Tt)
            _, hd, _ = cheap(np.maximum(Q - Tt, 0))
            m["n_bbo"], m["hhi_touch"], m["hhi_deep"] = nt, ht, hd
            # ★最良気配に居る数量が板全体に占める割合。これが小さいので
            #   deep の HHI は全体の HHI とほぼ同じ値になる(検算にもなる)。
            tq = Q.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                m["touch_share"] = np.where(tq > 0, Tt.sum(1) / tq, np.nan)
            m["depth"] = np.where(tq > 0, tq * SZ_LOT, np.nan)

            for k2 in METRICS:
                acc[k2].append(m[k2])
            hh = gh[c0:c1]
            for h in np.unique(hh):
                s_ = hh == h
                with np.errstate(invalid="ignore"):
                    hrows.append({"dt": dt, "hour": int(h),
                                  **{k2: float(np.nanmean(m[k2][s_]))
                                     for k2 in METRICS}})
            np.add.at(dep_b, cw[cb], cd[cb])
            np.add.at(dep_a, cw[~cb], cd[~cb])
            np.add.at(dep_pair, cpw, cd)
            del Qb, Qa, Q, Tt

        rec = {"dt": dt}
        for k2 in METRICS:
            v = np.concatenate(acc[k2])
            rec[f"{k2}_mean"] = float(np.nanmean(v))
            for q in QS:
                rec[f"{k2}_p{q}"] = float(np.nanpercentile(v, q))
        rec["n_grid"] = int(np.isfinite(np.concatenate(acc["hhi"])).sum())
        rec["n_wallet_day"] = int(W)
        rec["n_event"] = int(n_ev)
        rec["n_nowid"] = int(n_nowid)
        rec["n_carry"] = int(carry.height)
        rec["n_stale"] = int(n_stale)
        rec["bbo_chk_n"] = chk[0]
        rec["bbo_chk_same"] = chk[1]
        rec["bbo_chk_ratio"] = chk[2] / max(chk[3], 1e-9)
        pl.DataFrame([rec]).write_parquet(outd / f"daily_{dt}.parquet")
        pl.DataFrame(hrows).write_parquet(outd / f"hourly_{dt}.parquet")
        carry.write_parquet(outd / f"carry_{dt}.parquet")
        print(f"  {dt} 格子 {rec['n_grid']:,} 口座 {W} 活動 {rec['n_active_mean']:.0f} "
              f"HHI {rec['hhi_mean']:.3f} 最良 {rec['n_bbo_mean']:.2f} 者 "
              f"bbo一致 {rec['bbo_chk_same']/max(rec['bbo_chk_n'],1):.1%} "
              f"倍率 {rec['bbo_chk_ratio']:.3f} 繰越 {rec['n_carry']:,} "
              f"古い注文を落とした {n_stale:,} "
              f"{time.time()-t:.1f}s", file=sys.stderr, flush=True)

    def cat(w):
        fs = sorted(outd.glob(f"{w}_*.parquet"))
        return pl.concat([pl.read_parquet(f) for f in fs]) if fs else pl.DataFrame()

    Dd = cat("daily")
    Dd.write_parquet(ROOT / "data" / f"wallet_conc_daily_{tag}.parquet")
    Dd.write_csv(ROOT / "data" / f"wallet_conc_daily_{tag}.csv")
    cat("hourly").write_parquet(ROOT / "data" / f"wallet_conc_hourly_{tag}.parquet")
    print(f"\n[集計] 日 {Dd.height}", file=sys.stderr)
    print(f"-> data/wallet_conc_daily_{tag}.parquet ほか", file=sys.stderr)


if __name__ == "__main__":
    main()
