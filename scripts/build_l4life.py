"""指値 1 本ごとの記録から、板を「持続」「新鮮」「口座」で分解した層を作る。

    uv run python scripts/build_l4life.py --coin xyz:INTC [--start 0 --n 99]

`build_l4feat.py` の板は**数量と本数**しか分けられない。この層は
`orderlife_*`(1 行 = 指値 1 本、口座つき)を使って、同じ 1 秒格子で

  14 流動性の持続   … いま板にある数量のうち、何秒以上そこに居るぶんか
  27 束の間の流動性 … 直前の窓で消えた注文のうち、寿命が閾値未満の割合
  11 待ち行列       … いま板にある注文の「前に並んでいる数量」の平均
  17 補充           … 消えた数量に対して入った数量の比
  32 口座の集中度   … 口座数・HHI・最大シェア・エントロピー(全て厳密)
   8 除去の内訳     … 消えた注文のうち約定だったものの割合

を作る。板の水準では切らない(この層の狙いは水準ではなく**時間と主体**の分解)。

区間の足し込みで厳密に出す
--------------------------
注文 1 本は区間 [ts_open, ts_close) のあいだ板に居る。格子点 T で

    「板に居る」 ⟺ ts_open < T < ts_close

とすると、1 本の寄与は**連続した格子点の区間**になる。差分配列に
+w / −w を置いて累積和を取れば、全格子点の合計が 1 回の走査で出る。
持続ぶんも同じで、

    persistent(T, n) = resting(T) − 「開いてから n 秒未満のもの」

の右辺第 2 項は区間 [ob+1, min(cb−1, ob+n)] の足し込みで出る。

時間契約
--------
  * `ts_open < T` と `ts_close > T` の**厳密な不等号**を使う。等号を入れると
    格子点 T ちょうどに起きた事象が入り、T の情報に未来が混ざる
  * 窓の集計は全て [T−w, T)。窓の終端に T を含めない
  * ★「その注文が後で約定したか」は未来である。説明変数には
    **消えた注文の内訳**(既に消えているので過去)しか使わない
  * 口座の上位集合は**前日の出来高**で決める(当日の結果で選ばない)

★ファイルの分かれ方(実測して判ったこと)
----------------------------------------
`orderlife_*/dt=D.parquet` は **D に閉じた(約定/取消)注文**であって、
D に出した注文ではない(`dt_open` 列が別にある)。したがって日 D の板に
居た注文を集めるには**後ろ**の日を読む。ほとんど(実測 99.97%)は同じ日に
閉じるので D 〜 D+2 で足り、それより長く生きるものだけ一度索引にしておく。

遠くに置きっぱなしの注文
------------------------
実測すると、板にある数量の **98.6% は最良から 20 ティックより遠い**。
全部を混ぜると全指標が「置き去りの注文」の話になるので、発注時に最良から
NEAR ティック以内だった注文に絞った `n_` 付きの版を併せて作る。
口座の集中度は `n_` の母集団だけで測る。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import SZ_LOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTROOT = Path("E:/Memory-l4feat")
SEC = 1_000_000_000
NS = 86_400                      # 1 秒格子の点数
DAY_NS = 86_400 * SEC
NEAR = 10                        # 「最良の近く」の定義(発注時のティック)
LOOKBACK = 2                     # 何日前の発注まで読むか
FRESH = [1, 10]                  # 「新鮮」の閾値(秒)
FLEET = [0.1, 1.0]               # 「束の間」の閾値(秒)
WIN = [1, 10, 60]                # 後ろ向き窓(秒)
TOPK = 5                         # 前日実績で選ぶ上位口座の数

FAM: dict[str, str] = {}


def reg(name: str, fam: str) -> str:
    FAM[name] = fam
    return name


def ivl(n: int, lo: np.ndarray, hi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """区間 [lo, hi] に w を足した長さ n の配列(差分配列 + 累積和)。"""
    d = np.zeros(n + 1, np.float64)
    m = hi >= lo
    if m.any():
        np.add.at(d, np.clip(lo[m], 0, n), w[m])
        np.add.at(d, np.clip(hi[m] + 1, 0, n), -w[m])
    return np.cumsum(d[:n])


def cum0(x):
    c = np.empty(x.size + 1, np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    return c


def back(c, idx, w):
    return c[idx] - c[np.maximum(idx - w, 0)]


def safe(a, b, fill=np.nan):
    b = np.asarray(b, np.float64)
    ok = np.abs(b) > 1e-12
    return np.where(ok, np.asarray(a, np.float64) / np.where(ok, b, 1.0), fill)


COLS = ["oid", "ts_open", "ts_close", "is_bid", "orig_lots", "filled_lots",
        "dist_tick", "ahead_lots", "user", "life_ns"]
INF = 2 ** 62                                  # 窓の終わりまで開いたままの注文


def build_long(src: Path, days: list[str], tag: str) -> Path:
    """★`orderlife_*` は**約定/取消の日**で分かれている(発注日ではない)。

    日 D の板に居る注文は「D 以降に閉じたもの」なので、前ではなく**後ろ**の
    ファイルを読む必要がある。ほとんどは同じ日に閉じるが、何日も生き残る
    注文もあるため、寿命が LOOKBACK 日を超えるものだけ一度集めて索引にする。
    """
    fp = DATA / f"orderlife_long_{tag}.parquet"
    if fp.exists():
        return fp
    lim = LOOKBACK * DAY_NS
    fr = [pl.read_parquet(src / f"dt={d}.parquet", columns=COLS)
          .filter(pl.col("life_ns") > lim) for d in days]
    cen = DATA / f"orderlife_{tag}_censored.parquet"
    if cen.exists():
        c = pl.read_parquet(cen).with_columns(
            ts_close=pl.lit(INF, pl.Int64), life_ns=pl.lit(INF, pl.Int64),
            filled_lots=pl.lit(0, pl.Int64))
        fr.append(c.select(COLS))
    out = pl.concat(fr, how="diagonal_relaxed").unique(subset="oid", keep="first")
    out.write_parquet(fp, compression="zstd")
    print(f"[long] 長寿命の注文 {out.height:,} 本 -> {fp.name}", flush=True)
    return fp


def load(src: Path, days: list[str], dt: str, t0: int, long: pl.DataFrame):
    """日 dt の板に居た注文を集める。閉じた日が dt 〜 dt+LOOKBACK のファイル
    と、長寿命の索引から取る。"""
    i = days.index(dt)
    fr = [pl.read_parquet(src / f"dt={d}.parquet", columns=COLS)
          for d in days[i: i + LOOKBACK + 1]]
    fr.append(long)
    d = pl.concat(fr, how="diagonal_relaxed").unique(subset="oid", keep="first")
    return d.filter((pl.col("ts_close") > t0) & (pl.col("ts_open") < t0 + DAY_NS))


def day(src: Path, days: list[str], dt: str, prev_top: list[str] | None,
        long: pl.DataFrame):
    t0 = int(np.datetime64(dt, "ns").astype("int64"))   # その日の 00:00 UTC
    d = load(src, days, dt, t0, long)
    n_ord = d.height
    ob = ((d["ts_open"].to_numpy() - t0) // SEC).astype(np.int64) + 1
    cb = ((d["ts_close"].to_numpy() - t0) // SEC).astype(np.int64) - 1
    ob = np.maximum(ob, 0)
    cb = np.minimum(cb, NS - 1)
    lots = d["orig_lots"].to_numpy().astype(np.float64) * SZ_LOT
    isb = d["is_bid"].to_numpy()
    life = d["life_ns"].to_numpy() / 1e9
    ahead = d["ahead_lots"].to_numpy().astype(np.float64) * SZ_LOT
    filled = d["filled_lots"].to_numpy().astype(np.float64) * SZ_LOT
    user = d["user"].to_numpy()
    F: dict[str, np.ndarray] = {}
    idx = np.arange(NS)

    # ---- 14 いま板にある数量と、その持続ぶん ----------------------------
    # ★板の 98.6% は最良から 20 ティックより遠くに置かれた注文だった(実測)。
    #   全部を混ぜると「遠くに置きっぱなしの注文」が全指標を支配するので、
    #   **発注時に最良から NEAR ティック以内だったもの**に絞った版も作る。
    #   距離は発注時点の値(orderlife の dist_tick)で、その後の最良の動きは
    #   追わない。これは近似だが、各時点で判る量だけで決まる。
    near = np.abs(d["dist_tick"].to_numpy()) <= NEAR
    tot = None
    for bt, bsel in (("", np.ones(n_ord, bool)), ("n_", near)):
        rest = {}
        for s, tag in ((True, "b"), (False, "a")):
            m = bsel & (isb == s)
            rest[tag] = ivl(NS, ob[m], cb[m], lots[m])
            F[reg(f"{bt}live_{tag}", "14 流動性の持続")] = rest[tag]
            F[reg(f"{bt}liven_{tag}", "14 流動性の持続")] = ivl(
                NS, ob[m], cb[m], np.ones(int(m.sum())))
            for n in FRESH:
                fr = ivl(NS, ob[m], np.minimum(cb[m], ob[m] + n - 1), lots[m])
                F[reg(f"{bt}fresh{n}s_{tag}", "14 流動性の持続")] = fr
                F[reg(f"{bt}pers{n}s_{tag}", "14 流動性の持続")] = rest[tag] - fr
            # 11 前に並んでいる数量(発注時点の値の、いま板に居るぶんの平均)
            F[reg(f"{bt}ahead_{tag}", "11 待ち行列")] = safe(
                ivl(NS, ob[m], cb[m], ahead[m]), F[f"{bt}liven_{tag}"])
        F[reg(f"{bt}live_imb", "14 流動性の持続")] = safe(
            rest["b"] - rest["a"], rest["b"] + rest["a"], 0.0)
        for n in FRESH:
            F[reg(f"{bt}pers{n}s_imb", "14 流動性の持続")] = safe(
                F[f"{bt}pers{n}s_b"] - F[f"{bt}pers{n}s_a"],
                F[f"{bt}pers{n}s_b"] + F[f"{bt}pers{n}s_a"], 0.0)
            F[reg(f"{bt}fresh{n}s_imb", "14 流動性の持続")] = safe(
                F[f"{bt}fresh{n}s_b"] - F[f"{bt}fresh{n}s_a"],
                F[f"{bt}fresh{n}s_b"] + F[f"{bt}fresh{n}s_a"], 0.0)
            F[reg(f"{bt}pers{n}s_share", "14 流動性の持続")] = safe(
                F[f"{bt}pers{n}s_b"] + F[f"{bt}pers{n}s_a"],
                rest["b"] + rest["a"])
        F[reg(f"{bt}ahead_imb", "11 待ち行列")] = F[f"{bt}ahead_b"] - F[f"{bt}ahead_a"]
        if bt == "n_":
            tot_near = rest["b"] + rest["a"]
        else:
            tot = rest["b"] + rest["a"]
    F[reg("near_share", "14 流動性の持続")] = safe(tot_near, tot)

    # ---- 8 / 27 消えた注文の内訳と束の間の割合 --------------------------
    #      ★消えたあとの情報なので過去。約定「する」確率ではない
    cl = ((d["ts_close"].to_numpy() - t0) // SEC).astype(np.int64)
    inday = (cl >= 0) & (cl < NS)
    was_fill = filled > 0
    onday = (ob >= 0) & (ob < NS)

    def bn(m, w=None):
        if not m.any():
            return np.zeros(NS)
        return np.bincount(cl[m], weights=(np.ones(int(m.sum())) if w is None
                                           else w[m]), minlength=NS).astype(float)

    for bt, bsel in (("", np.ones(n_ord, bool)), ("n_", near)):
        ind = inday & bsel
        g = {"rm_n": bn(ind), "rm_l": bn(ind, lots),
             "fl_n": bn(ind & was_fill), "fl_l": bn(ind, filled)}
        for tau in FLEET:
            g[f"fle_n{tau}"] = bn(ind & (life < tau))
            g[f"fle_l{tau}"] = bn(ind & (life < tau), lots)
        g["life_s"] = bn(ind, np.minimum(life, 3600.0))
        g["life_ls"] = bn(ind, np.log1p(np.minimum(life, 3600.0)))
        om = onday & bsel
        g["ad_l"] = np.bincount(np.clip(ob, 0, NS - 1)[om], weights=lots[om],
                                minlength=NS).astype(float)
        cum = {k: cum0(v) for k, v in g.items()}
        for w in WIN:
            lab = f"{w}s"
            rn, rl = back(cum["rm_n"], idx, w), back(cum["rm_l"], idx, w)
            F[reg(f"{bt}fill_share_n_{lab}", "8 除去の内訳")] = safe(
                back(cum["fl_n"], idx, w), rn)
            F[reg(f"{bt}fill_share_l_{lab}", "8 除去の内訳")] = safe(
                back(cum["fl_l"], idx, w), rl)
            for tau in FLEET:
                t2 = f"{tau:g}".replace(".", "")
                F[reg(f"{bt}flr_n{t2}_{lab}", "27 束の間の流動性")] = safe(
                    back(cum[f"fle_n{tau}"], idx, w), rn)
                F[reg(f"{bt}flr_l{t2}_{lab}", "27 束の間の流動性")] = safe(
                    back(cum[f"fle_l{tau}"], idx, w), rl)
            F[reg(f"{bt}life_mean_{lab}", "13 注文の年齢")] = safe(
                back(cum["life_s"], idx, w), rn)
            F[reg(f"{bt}life_logmean_{lab}", "13 注文の年齢")] = safe(
                back(cum["life_ls"], idx, w), rn)
            F[reg(f"{bt}refill_{lab}", "17 補充")] = safe(back(cum["ad_l"], idx, w), rl)

    # ---- 32 口座の集中度(厳密) ----------------------------------------
    # ★最良から NEAR ティック以内に置かれた注文だけで測る。遠くに置きっぱなしの
    #   注文まで入れると「板を作っている主体」ではなく「置き去りの主体」を測る
    us = user[near]
    u, uinv = np.unique(us, return_inverse=True)
    nu = u.size
    ob, cb, lots = ob[near], cb[near], lots[near]
    tot = tot_near
    ssq = np.zeros(NS)
    ent = np.zeros(NS)
    mx = np.zeros(NS)
    cnt = np.zeros(NS)
    topmask = np.zeros(NS)
    top_set = set(prev_top or [])
    B = 48
    for k0 in range(0, nu, B):
        sel = (uinv >= k0) & (uinv < min(k0 + B, nu))
        if not sel.any():
            continue
        sub = uinv[sel] - k0
        ob_s, cb_s, lo_s = ob[sel], cb[sel], lots[sel]
        o2 = np.argsort(sub, kind="stable")
        sub, ob_s, cb_s, lo_s = sub[o2], ob_s[o2], cb_s[o2], lo_s[o2]
        nb = int(sub.max()) + 1
        cut = np.searchsorted(sub, np.arange(nb + 1))
        M = np.zeros((nb, NS))
        for j in range(nb):
            s2, e2 = cut[j], cut[j + 1]
            if e2 > s2:
                M[j] = ivl(NS, ob_s[s2:e2], cb_s[s2:e2], lo_s[s2:e2])
        ssq += (M ** 2).sum(0)
        ent += -(M * np.log(np.maximum(M, 1e-12))).sum(0)
        mx = np.maximum(mx, M.max(0))
        cnt += (M > 0).sum(0)
        if top_set:
            hit = np.array([u[k0 + j] in top_set for j in range(nb)])
            if hit.any():
                topmask += M[hit].sum(0)
        del M
    F[reg("wal_n", "32 口座の集中度")] = cnt
    F[reg("wal_hhi", "32 口座の集中度")] = safe(ssq, tot ** 2)
    F[reg("wal_eff", "32 口座の集中度")] = safe(tot ** 2, ssq)
    F[reg("wal_top1", "33 支配的なメイカー")] = safe(mx, tot)
    F[reg("wal_ent", "32 口座の集中度")] = safe(ent, tot) + np.log(np.maximum(tot, 1e-12))
    F[reg(f"wal_top{TOPK}_prev", "33 支配的なメイカー")] = safe(topmask, tot)

    # 翌日の「上位口座」は当日の実績で決める(使うのは翌日なので先読みでない)
    nxt_top = (pl.DataFrame({"user": us, "lots": lots})
               .group_by("user").agg(pl.col("lots").sum())
               .sort("lots", descending=True).head(TOPK)["user"].to_list())
    meta = dict(n_ord=n_ord, n_user=int(nu))
    return F, nxt_top, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:INTC")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=999)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = DATA / f"orderlife_{tag}"
    out = OUTROOT / f"{tag}_life"
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[3:13] for p in src.glob("dt=*.parquet"))
    sel = days[a.start:a.start + a.n]
    long = pl.read_parquet(build_long(src, days, tag))
    prev_top = None
    for dt in sel:
        t = time.time()
        F, prev_top, meta = day(src, days, dt, prev_top, long)
        df = pl.DataFrame({"dt": np.full(NS, dt),
                           "sec": np.arange(NS, dtype=np.int32)}
                          | {k: np.asarray(v, np.float32) for k, v in F.items()})
        df.write_parquet(out / f"dt={dt}.parquet", compression="zstd",
                         compression_level=3)
        print(f"  {dt}  {df.width} 列  注文 {meta['n_ord']:,} 口座 {meta['n_user']}"
              f"  {time.time()-t:.0f}s", flush=True)
    if FAM:
        pl.DataFrame({"col": list(FAM), "family": list(FAM.values())}) \
          .write_csv(DATA / f"l4life_cols_{tag}.csv")


if __name__ == "__main__":
    main()
