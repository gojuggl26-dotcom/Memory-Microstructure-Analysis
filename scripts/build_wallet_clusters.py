"""ウォレットの行動プロファイルから類似度・クラスタ・埋め込みを作る。

【7 つの行動プロファイル】
ウォレットごとに、注文の作り方を**分布**として持つ。分布どうしを比べるので
「1 本の平均」では消えてしまう癖(二山、極端な裾)を残せる。

    size-profile          注文数量の対数階級ヒストグラム(13 帯)
    quote-distance        mid からの距離 [bp] のヒストグラム(10 帯)
    lifetime              生存時間の対数階級ヒストグラム(11 帯)
    timing                発注時刻(UTC の時)のヒストグラム(24 帯)
    cancellation-pattern  取消で終わった割合 + 取消注文の生存時間分布(12 次元)
    price-placement       同じ側の最良気配から何ティック離れているか(9 帯)
    BBO behavior          最良気配率・改善率・2 ティック以内率・買い比率・両建て度(5 次元)

【距離の測り方】
分布の 6 つは **Jensen–Shannon 距離**(対称・有界 [0,1]・0 を含んでも定義できる)。
BBO behavior はスカラーの束なので、各次元を全標本で標準化してから
ユークリッド距離を $`\\sqrt{2d}`$ で割って [0,1] に収める(z 得点どうしの
期待二乗距離が 2d になるため)。

    wallet similarity = 1 − (7 つの距離の平均)

**標準化のパラメータは全標本から作っている**。これは記述的なクラスタリングで
あって標本外予測ではない。予測に使うなら訓練期間だけで作り直すこと。

【クラスタと埋め込み】
    maker cluster ID              平均連結の階層クラスタリング。クラスタ数 k は
                                  **k = 2..8 のシルエット最大**で選ぶ(範囲は先に宣言)
    behavioral cluster embedding  距離行列の古典的 MDS を 2 次元で

【対象の選び方(結果を見る前に決めた)】
滞留指値を **5,000 本以上**出し、**10 日以上**活動したウォレットに限る。
少数の注文しか無いウォレットの分布は雑音でしかないため。

    uv run python scripts/build_wallet_clusters.py --coin xyz:MU
出力: data/wallet_clusters_<coin>.csv   … ウォレット × クラスタ・埋め込み座標
      data/wallet_sim_<coin>.parquet    … 対ごとの 7 領域 + 総合の類似度
      data/wallet_prof_<coin>.parquet   … 行動プロファイル(図示用)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.01
MIN_ORDERS = 5_000
MIN_DAYS = 10
K_RANGE = range(2, 9)
CANCEL = {"canceled", "reduceOnlyCanceled", "selfTradeCanceled",
          "siblingFilledCanceled", "marginCanceled", "scheduledCancel"}
RESTING = {"Alo", "Gtc"}
L1COLS = ["ts", "oid", "side", "px", "status", "orig_sz", "tif", "is_trigger"]

SIZE_EDGES = 10.0 ** np.arange(-2.0, 4.5, 0.5)          # 13 帯
DIST_EDGES = np.array([0.5, 1, 2, 5, 10, 25, 50, 100, 250.0])   # 10 帯
LIFE_EDGES = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000.0])  # 11 帯
PLACE_EDGES = np.array([0, 1, 2, 3, 5, 10, 20, 50])     # 改善(<0)+ 8 = 9 帯
NB = {"size": len(SIZE_EDGES) + 1, "dist": len(DIST_EDGES) + 1,
      "life": len(LIFE_EDGES) + 1, "hour": 24,
      "cxl": len(LIFE_EDGES) + 2, "place": len(PLACE_EDGES) + 1, "bbo": 5}
DOMAIN = ["size", "dist", "life", "hour", "cxl", "place", "bbo"]
JP = {"size": "size-profile", "dist": "quote-distance", "life": "lifetime",
      "hour": "timing", "cxl": "cancellation-pattern", "place": "price-placement",
      "bbo": "BBO behavior"}


def js_dist(P: np.ndarray) -> np.ndarray:
    """行ごとの分布に対する Jensen–Shannon 距離の行列。"""
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    n = P.shape[0]
    D = np.zeros((n, n))
    lg = np.where(P > 0, np.log2(np.maximum(P, 1e-12)), 0.0)
    H = -(P * lg).sum(1)
    for i in range(n):
        M = 0.5 * (P[i][None, :] + P)
        lm = np.where(M > 0, np.log2(np.maximum(M, 1e-12)), 0.0)
        HM = -(M * lm).sum(1)
        js = np.clip(HM - 0.5 * (H[i] + H), 0.0, 1.0)
        D[i] = np.sqrt(js)
    return 0.5 * (D + D.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--reduce", action="store_true",
                    help="集計をやり直さず、保存済みプロファイルから作り直す")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    l1dir, l1udir = ROOT / "data" / f"l1_{tag}", ROOT / "data" / f"l1u_{tag}"
    days = sorted(p.stem.split("=")[1] for p in l1udir.glob("dt=*.parquet"))
    if a.days:
        days = days[: a.days]
    bbo = pl.scan_parquet(ROOT / "data" / f"bbo_{tag}.parquet")
    print(f"[日] {len(days)}", file=sys.stderr)

    CAP = 40_000                       # ウォレット数の上限(超えたら伸ばす)
    IDX: dict[str, int] = {}
    GLOB = {k: np.zeros((CAP, NB[k])) for k in DOMAIN if k != "bbo"}
    GLOB["bbo_raw"] = np.zeros((CAP, 5))
    NORD = np.zeros(CAP)
    DAYS: list[set] = [set() for _ in range(CAP)]

    def slot(u):
        i = IDX.get(u)
        if i is None:
            i = IDX[u] = len(IDX)
            if i >= CAP:
                sys.exit(f"ウォレットが {CAP:,} を超えた。CAP を上げること")
        return i

    if a.reduce:
        PR = pl.read_parquet(ROOT / "data" / f"wallet_prof_{tag}.parquet")
        OLD = pl.read_csv(ROOT / "data" / f"wallet_clusters_{tag}.csv")
        users = sorted(OLD["user"].to_list())
        for i, uu in enumerate(users):
            IDX[uu] = i
        for r in PR.to_dicts():
            i = IDX.get(r["user"])
            if i is None:
                continue
            if r["domain"] == "bbo":
                GLOB["bbo_raw"][i, r["bin_i"]] = r["val"]
            else:
                GLOB[r["domain"]][i, r["bin_i"]] = r["val"]
        for r in OLD.to_dicts():
            i = IDX[r["user"]]
            NORD[i] = r["n_orders"]
            DAYS[i] = set(range(int(r["n_days"])))
        print(f"[reduce] 保存済み {len(users)} 者から作り直す", file=sys.stderr)
        days = []

    for n, day in enumerate(days, 1):
        try:
            d = pl.read_parquet(l1dir / f"dt={day}.parquet", columns=L1COLS)
            u = pl.read_parquet(l1udir / f"dt={day}.parquet")
        except Exception as e:
            print(f"  {day} 読めない: {type(e).__name__}", file=sys.stderr)
            continue
        if u.height != d.height:
            sys.exit(f"{day}: user の行数が合わない")
        d = d.with_columns(user=u["user"], ts=pl.col("ts").cast(pl.Int64))
        op = d.filter((pl.col("status") == "open") & ~pl.col("is_trigger")
                      & pl.col("tif").is_in(list(RESTING)))
        if op.height == 0:
            continue
        tm = (d.filter(pl.col("status").is_in(["filled"] + list(CANCEL)))
               .select("oid", t_close="ts",
                       is_cxl=pl.col("status").is_in(list(CANCEL)))
               .group_by("oid").first())
        t0 = int(pl.Series([day]).str.to_datetime("%Y-%m-%d", time_unit="ns").cast(pl.Int64)[0])
        L = (op.select("oid", "user", "side", "px", "orig_sz", t_open="ts")
               .join(tm, on="oid", how="left")
               .with_columns(t_close=pl.col("t_close").fill_null(t0 + 86_400_000_000_000),
                             is_cxl=pl.col("is_cxl").fill_null(False))
               .with_columns(dur=(pl.col("t_close") - pl.col("t_open")) / 1e9))
        B = (bbo.filter(pl.col("dt") == day).select("ts", "best_bid", "best_ask")
                .collect().filter(pl.col("best_ask") > pl.col("best_bid")).sort("ts")
                .with_columns(ts=pl.col("ts").cast(pl.Int64)))
        if B.height < 10:
            continue
        bt = B["ts"].to_numpy()
        bb, ba = B["best_bid"].to_numpy(), B["best_ask"].to_numpy()
        j = np.clip(np.searchsorted(bt, L["t_open"].to_numpy(), side="right") - 1,
                    0, len(bt) - 1)
        mid = (bb[j] + ba[j]) / 2.0
        px = L["px"].to_numpy()
        isbid = (L["side"].to_numpy() == "B")
        dist = np.abs(px - mid) / mid * 1e4
        lv = np.where(isbid, (bb[j] - px), (px - ba[j])) / TICK
        hour = ((L["t_open"].to_numpy() - t0) // 3_600_000_000_000).clip(0, 23)
        sz = L["orig_sz"].to_numpy()
        dur = L["dur"].to_numpy()
        cxl = L["is_cxl"].to_numpy()
        users = L["user"].to_numpy()

        bi_size = np.digitize(sz, SIZE_EDGES)
        bi_dist = np.digitize(dist, DIST_EDGES)
        bi_life = np.digitize(dur, LIFE_EDGES)
        bi_place = np.where(lv < 0, 0, np.digitize(np.maximum(lv, 0), PLACE_EDGES) + 0)
        bi_place = np.clip(bi_place, 0, NB["place"] - 1)

        # ★ウォレットを整数 ID に落として bincount で一括集計する。
        #   np.add.at をウォレット × 領域の二重ループで呼ぶと 100 万回規模になり、
        #   99 日で 1 時間以上かかったうえメモリも尽きた(実際に落ちた)
        uq, uid = np.unique(users, return_inverse=True)
        nu_ = len(uq)
        rows_ = np.array([slot(x) for x in uq])
        hist = {
            "size": np.bincount(uid * NB["size"] + bi_size, minlength=nu_ * NB["size"]),
            "dist": np.bincount(uid * NB["dist"] + bi_dist, minlength=nu_ * NB["dist"]),
            "life": np.bincount(uid * NB["life"] + bi_life, minlength=nu_ * NB["life"]),
            "hour": np.bincount(uid * NB["hour"] + hour.astype(int),
                                minlength=nu_ * NB["hour"]),
            "place": np.bincount(uid * NB["place"] + bi_place, minlength=nu_ * NB["place"]),
        }
        for k, v in hist.items():
            np.add.at(GLOB[k], rows_, v.reshape(nu_, NB[k]).astype(float))
        cx = np.bincount(uid, weights=cxl.astype(float), minlength=nu_)
        cxh = np.bincount(uid[cxl] * NB["cxl"] + bi_life[cxl] + 1,
                          minlength=nu_ * NB["cxl"]).reshape(nu_, NB["cxl"]).astype(float)
        cxh[:, 0] = cx
        np.add.at(GLOB["cxl"], rows_, cxh)
        bv = np.stack([
            np.bincount(uid, weights=(lv == 0).astype(float), minlength=nu_),
            np.bincount(uid, weights=(lv < 0).astype(float), minlength=nu_),
            np.bincount(uid, weights=(np.abs(lv) <= 2).astype(float), minlength=nu_),
            np.bincount(uid, weights=isbid.astype(float), minlength=nu_),
            np.minimum(np.bincount(uid, weights=isbid.astype(float), minlength=nu_),
                       np.bincount(uid, weights=(~isbid).astype(float), minlength=nu_)),
        ], axis=1)
        np.add.at(GLOB["bbo_raw"], rows_, bv)
        np.add.at(NORD, rows_, np.bincount(uid, minlength=nu_).astype(float))
        for r in rows_:
            DAYS[r].add(day)
        del d, u, op, tm, L, B, hist, bv, cxh
        if n % 10 == 0 or n == len(days):
            print(f"  {day}  ({n}/{len(days)} 日) ウォレット {len(IDX):,}", file=sys.stderr)

    # ---- 対象の絞り込み ------------------------------------------------------
    users = sorted(u for u, i in IDX.items()
                   if NORD[i] >= MIN_ORDERS and len(DAYS[i]) >= MIN_DAYS)
    print(f"[対象] {len(users)} 者 / 全 {len(IDX):,} 者"
          f"(滞留指値 {MIN_ORDERS:,} 本以上 かつ {MIN_DAYS} 日以上)", file=sys.stderr)
    if len(users) < 4:
        sys.exit("対象が少なすぎる")

    ridx = np.array([IDX[u] for u in users])
    prof = {k: GLOB[k][ridx] for k in DOMAIN if k != "bbo"}
    bbo_m = (GLOB["bbo_raw"][ridx] if a.reduce
             else GLOB["bbo_raw"][ridx] / np.maximum(NORD[ridx][:, None], 1))
    z = (bbo_m - bbo_m.mean(0)) / np.maximum(bbo_m.std(0), 1e-9)

    D = {}
    for k in DOMAIN:
        if k == "bbo":
            dd = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
            D[k] = dd / max(dd.max(), 1e-9)      # 最大で割る(飽和させない)
        else:
            D[k] = js_dist(prof[k])
    Dall = np.mean([D[k] for k in DOMAIN], axis=0)
    np.fill_diagonal(Dall, 0.0)

    # ---- クラスタ ------------------------------------------------------------
    Z = linkage(squareform(Dall, checks=False), method="average")
    best, best_s, sil_all = None, -2, {}
    for k in K_RANGE:
        lab = fcluster(Z, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        s = []
        for i in range(len(users)):
            same = (lab == lab[i]) & (np.arange(len(users)) != i)
            if same.sum() == 0:
                continue
            ai = Dall[i, same].mean()
            bi = min(Dall[i, lab == c].mean() for c in set(lab) if c != lab[i])
            s.append((bi - ai) / max(ai, bi))
        sc = float(np.mean(s)) if s else -2
        sil_all[k] = sc
        if sc > best_s:
            best, best_s = lab, sc
    k_best = len(set(best))
    print(f"[クラスタ] 事前宣言どおりのシルエット最大: k={k_best}"
          f"(シルエット {best_s:.3f}) / 候補 {[(k, round(v, 3)) for k, v in sil_all.items()]}",
          file=sys.stderr)
    sizes = [int((best == c).sum()) for c in sorted(set(best))]
    print(f"           クラスタの大きさ {sizes}", file=sys.stderr)

    # ★シルエット最大の分割は「外れ値 3 者 vs 残り全部」に潰れうる。
    #   最小クラスタが全体の 5% 以上という制約つきの分割も併記する
    #   (この制約は結果を見てから足した。事前宣言はあくまで上の無制約版)
    #   平均連結も完全連結も外れ値を切り出すだけで k=2..8 のどこでも均衡しない
    #   (実測: average は最小 1〜3 者、complete も 3 者)。Ward だけが均衡する
    #   ので均衡版は Ward を使う。★Ward はユークリッド距離を前提とする手法で、
    #   ここでの距離は 7 つの距離の平均なので厳密には前提を満たさない。
    #   記述的な分割としてのみ扱い、シルエットは元の距離で測り直している。
    floor = max(3, int(0.05 * len(users)))
    Zc = linkage(squareform(Dall, checks=False), method="ward")
    bal, bal_s, bal_k = best, -2, k_best
    for k in K_RANGE:
        lab = fcluster(Zc, k, criterion="maxclust")
        cnt = np.bincount(lab)[1:]
        if len(cnt) < 2 or cnt.min() < floor:
            continue
        s_ = []
        for i in range(len(users)):
            same = (lab == lab[i]) & (np.arange(len(users)) != i)
            if same.sum() == 0:
                continue
            ai = Dall[i, same].mean()
            bi = min(Dall[i, lab == c].mean() for c in set(lab) if c != lab[i])
            s_.append((bi - ai) / max(ai, bi))
        sc = float(np.mean(s_)) if s_ else -2
        if sc > bal_s:
            bal, bal_s, bal_k = lab, sc, k
    print(f"[クラスタ] 最小 {floor} 者の制約つき: k={bal_k}(シルエット {bal_s:.3f})"
          f" / 大きさ {[int((bal == c).sum()) for c in sorted(set(bal))]}", file=sys.stderr)

    # ---- 埋め込み(古典的 MDS)-----------------------------------------------
    J = np.eye(len(users)) - np.ones((len(users), len(users))) / len(users)
    Bm = -0.5 * J @ (Dall ** 2) @ J
    ev, V = np.linalg.eigh(Bm)
    o = np.argsort(ev)[::-1][:2]
    emb = V[:, o] * np.sqrt(np.maximum(ev[o], 0))
    var2 = float(np.maximum(ev[o], 0).sum() / np.maximum(ev, 0).sum())

    out = pl.DataFrame({
        "user": users, "n_orders": NORD[ridx].astype(int),
        "n_days": [len(DAYS[i]) for i in ridx],
        "cluster": bal.astype(int), "cluster_sil": best.astype(int), "emb_x": emb[:, 0], "emb_y": emb[:, 1],
        "sim_mean": [float(1 - Dall[i][np.arange(len(users)) != i].mean())
                     for i in range(len(users))],
    })
    out.write_csv(ROOT / "data" / f"wallet_clusters_{tag}.csv")

    iu = np.triu_indices(len(users), 1)
    S = pl.DataFrame({"a": [users[i] for i in iu[0]], "b": [users[j] for j in iu[1]],
                      **{f"sim_{k}": 1 - D[k][iu] for k in DOMAIN},
                      "sim_all": 1 - Dall[iu]})
    S.write_parquet(ROOT / "data" / f"wallet_sim_{tag}.parquet")

    rows = []
    for i, u in enumerate(users):
        for k in DOMAIN:
            v = (prof[k][i] / max(prof[k][i].sum(), 1e-12)) if k != "bbo" else bbo_m[i]
            for b, x in enumerate(v):
                rows.append({"user": u, "cluster": int(bal[i]), "domain": k,
                             "bin_i": b, "val": float(x)})
    pl.DataFrame(rows).write_parquet(ROOT / "data" / f"wallet_prof_{tag}.parquet")

    print(f"[埋め込み] 2 次元で距離の {var2:.1%} を説明", file=sys.stderr)
    print(out.group_by("cluster").agg(pl.len(), pl.col("n_orders").median(),
                                      pl.col("sim_mean").mean()).sort("cluster")
          .to_pandas().to_string(), file=sys.stderr)
    print(f"\n-> data/wallet_clusters_{tag}.csv / wallet_sim_{tag}.parquet"
          f" / wallet_prof_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
