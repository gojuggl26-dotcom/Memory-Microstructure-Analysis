"""各特徴量の符号が何イベント先まで持続するかを、符号の推移確率行列で測る。

## 何を出すのか

特徴量 x の符号を 3 状態 {負, ちょうど 0, 正} に落とし、k イベント後の符号との
同時分布を数える。k = 1〜5。

    N_k[s, s'] = #{ t : sign(x_t) = s かつ sign(x_{t+k}) = s' }
    P_k[s, s'] = N_k[s, s'] / Σ_{s''} N_k[s, s'']

主役は右下のセル P_k[正, 正] =「今が正のとき k イベント後も正である確率」。

## ★これは「予測」ではない。同じ量の自己相関である

x_t と x_{t+k} は同じ特徴量である。ここで測っているのは x 自身の符号の持続性で
あって、将来の価格を当てる力ではない。予測力の話は reports/predicting_definition.md
と OBI/OFI レポートを参照すること。ここの数字を売買の根拠に読み替えてはならない。

先読みは無い(x_t は時刻 t で確定し、x_{t+k} は t+k で確定する。両方とも
説明変数側であり、目的変数を含まない)。shift(-k) は使っていない。

## ★0.5 と比べてはいけない — 対照は「無条件の確率」

特徴量の符号は 50:50 ではない。P(正|正) = 0.62 でも、無条件の P(正) が 0.62 なら
持続性はゼロである。

    超過 = P_k[正, 正] − P_k[·, 正]        … 列周辺確率が対照

列周辺は同じペア集合の上で取る(k が変わるとペア集合も変わるため)。

## ★状態を 2 つにしない — ちょうど 0 は実在する

OBI がちょうど 0 の行は実測で 1.35% ある(両側に同数量が出ている気配)。OFI は
最良気配が動かなかった向きで打ち消し合うとちょうど 0 になる。0 を正か負の
どちらかに押し込むと、その側の持続性が水増しされる。3 状態のまま数える。

## ★日を跨がせない

(t, t+k) の組は同じ日の中だけで作る。日の最初の行は前の行が前日なので、
差分で作る特徴量(OFI・Δスプレッド・Δ数量・リターン)はそこで無効にする。

## 不確かさと帰無対照

- 不確かさ … 隣接イベントは強く相関するので二項の区間は使えない。
  日単位のブロックブートストラップ(400 反復)で超過の区間を出す
- 帰無対照 … 日の中で符号列を無作為に並べ替える。周辺分布はそのままで
  時間の対応だけが壊れるので、持続性が本物なら超過は 0 へ潰れる。
  ★巡回シフトは使えない。巡回シフトは自己相関を保存するが、ここで測って
  いるのがその自己相関そのものだからである(他のレポートで巡回シフトを
  使っているのは、x はそのままで y との対応だけを壊す目的のとき)

## 恒等式による重複の検査

    BookSlope_t = (q^b − q^a)/h_t,           h_t > 0
    micro_t − mid_t = (I_t − 1/2)·spread_t,  spread_t > 0
    OBI_t = (q^b − q^a)/(q^b + q^a),         q^b + q^a > 0

いずれも符号は sign(q^b − q^a) に一致する。OFI_z = OFI/s は s > 0 なので
sign(OFI_z) = sign(OFI)。つまりこれらは符号の推移行列としては同じものになる
はずである。仮定せず、スクリプト内で全行を突き合わせて数値で確かめる。

    uv run python scripts/build_sign_persistence.py --coin xyz:MU
出力: data/sign_persist_counts_<coin>.parquet … 日 × 特徴量 × k × (s, s') の生計数
      data/sign_persist_cells_<coin>.parquet  … 日区分 × 特徴量 × k の集計と区間
      data/sign_persist_ident_<coin>.parquet  … 恒等式の検算結果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
KS = [1, 2, 3, 4, 5]
N_BOOT = 400
SEED = 20260828
STATES = ["負", "ちょうど 0", "正"]          # 索引 0, 1, 2

FEATS = [
    ("OBI", "板の残高の偏り (q^b-q^a)/(q^b+q^a)", "水準"),
    ("OFI", "板の流量の偏り Cont-Kukanov-Stoikov", "差分"),
    ("BookSlope", "板の傾き (q^b-q^a)/h", "水準"),
    ("micro_dev", "MicroPrice - mid", "水準"),
    ("d_spread", "スプレッド幅の変化", "差分"),
    ("d_depth", "最良気配の合計数量の変化", "差分"),
    ("ret", "mid の log リターン", "差分"),
]

# イベントの時計。BBO 系(3,517 万)と成行注文(637 万)は数え方が違うので、
# k を横に比べてはいけない。
CLOCK = {"taker_sign": "成行注文"}

# 符号を持たない特徴量(このリポジトリで算出済みだが、この分析に載せられないもの)。
# スプレッド幅・最良気配の数量・約定量・注文サイズ・到着深さはいずれも定義上
# 常に正で、P(正) = 1 になり推移行列が退化する。変化の向きに意味があるので、
# 差分をとった d_spread / d_depth を代わりに載せている。
DEGENERATE = ["スプレッド幅", "最良気配の合計数量", "約定量", "注文サイズ", "到着深さ"]


def taker_states(coin: str, days: list[str]) -> dict[str, np.ndarray]:
    """テイカーの向きを日ごとの状態列にする。買い = 正、売り = 負(0 は無い)。

    ★イベントの時計が違う。BBO 系の特徴量は「最良気配が変わった 1 行」を
    1 イベントと数えるが(98 日で 3,517 万)、これは「成行注文 1 本」を
    1 イベントと数える(637 万)。したがって k = 5 の指す実時間が 5〜6 倍違う。
    同じ表に並べても k を横に比べてはならない。

    単位は build_sign_chain.py の主系列(成行注文単位)に合わせてある。
    同一 ns・同一ユーザーの約定を 1 本にまとめる。
    """
    tag = coin.replace(":", "_")
    p = ROOT / "data" / f"fills_{tag}.parquet"
    if not p.exists():
        print(f"[taker] {p} が無いので飛ばす", file=sys.stderr)
        return {}
    f = (
        pl.read_parquet(p, columns=["ts", "user", "side", "crossed"])
        .filter(pl.col("crossed"))
        .with_columns(b=(pl.col("side") == "B").cast(pl.Int8))
        .group_by("ts", "user")
        .agg(b=pl.col("b").first())
        .with_columns(d=pl.col("ts").dt.date().cast(pl.String))
        .sort("ts", "user")            # 同一 ns の並びを決定的にする
    )
    out = {}
    for (d,), g in f.group_by("d", maintain_order=True):
        out[d] = np.where(g["b"].to_numpy() == 1, 2, 0).astype(np.int8)
    n = sum(len(v) for v in out.values())
    share = sum(int((v == 2).sum()) for v in out.values()) / n
    print(f"[taker] {n:,} 成行注文 / {len(out)} 日 / 買いの割合 {share * 100:.2f}%",
          file=sys.stderr)
    return {d: out.get(d, np.zeros(0, np.int8)) for d in days}


def day_types(days: list[str]) -> dict[str, str]:
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    return {d: ("立会日" if d in sess else "閉場日") for d in days}


def load(coin: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """必要な 4 列だけを numpy へ。日の境界も返す(RAM を節約するため)。"""
    tag = coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い。先に fetch_bbo.py を実行すること")
    d = (
        pl.scan_parquet(p)
        .select("ts", "best_bid", "best_ask", "bid_sz", "ask_sz", "dt")
        .sort("ts")
        # 板が壊れている行は落とす(CLAUDE.md「is_crossed は必ず除外」)。
        # 他のレポートと同じ条件にしてある。
        .filter(
            (pl.col("best_ask") > pl.col("best_bid"))
            & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
            & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
        )
        .collect()
    )
    days = sorted(d["dt"].unique().to_list())
    code = {v: i for i, v in enumerate(days)}
    dc = d["dt"].replace_strict(code, return_dtype=pl.Int32).to_numpy()
    cols = {c: d[c].to_numpy().astype(np.float64) for c in
            ("best_bid", "best_ask", "bid_sz", "ask_sz")}
    del d
    bnd = np.r_[0, np.flatnonzero(np.diff(dc)) + 1, len(dc)]
    print(f"[load] {len(dc):,} イベント / {len(days)} 日", file=sys.stderr)
    return cols, bnd, days


def features_for_day(pb, pa, qb, qa) -> dict[str, np.ndarray]:
    """1 日分の特徴量。差分系は先頭を NaN にして日を跨がせない。"""
    n = len(pb)
    mid = (pb + pa) / 2.0
    spread_bp = (pa - pb) / mid * 1e4
    h = spread_bp / 2.0                      # 半スプレッド[bp] > 0
    q = qb + qa

    f = {}
    f["OBI"] = (qb - qa) / q
    f["BookSlope"] = (qb - qa) / h
    # ★素朴に (pb·qa + pa·qb)/q - mid と書いてはいけない。
    # mid は 1,087 前後、micro - mid は 5e-2 前後で、q^b = q^a のとき真値は
    # ちょうど 0 だが、丸め残差 ~1e-13 が符号を持ってしまう。実測で
    # 「ちょうど 0」の行の 100% が誤って ± に分類された(全体の 0.53%)。
    # 分子を因数分解した下の形なら (q^b - q^a) が 0 のとき厳密に 0 になる。
    #     micro - mid = (q^b - q^a)(P^a - P^b) / (2(q^b + q^a))
    f["micro_dev"] = (qb - qa) * (pa - pb) / (2.0 * q)
    f["_micro_naive"] = (pb * qa + pa * qb) / q - mid    # 検算用にのみ使う

    # OFI (Cont-Kukanov-Stoikov 2014)
    ofi = np.full(n, np.nan)
    if n > 1:
        pb0, pa0, qb0, qa0 = pb[:-1], pa[:-1], qb[:-1], qa[:-1]
        pb1, pa1, qb1, qa1 = pb[1:], pa[1:], qb[1:], qa[1:]
        ofi[1:] = (
            np.where(pb1 >= pb0, qb1, 0.0) - np.where(pb1 <= pb0, qb0, 0.0)
            - np.where(pa1 <= pa0, qa1, 0.0) + np.where(pa1 >= pa0, qa0, 0.0)
        )
    f["OFI"] = ofi

    for name, v in (("d_spread", spread_bp), ("d_depth", q),
                    ("ret", np.log(mid) * 1e4)):
        dv = np.full(n, np.nan)
        if n > 1:
            dv[1:] = v[1:] - v[:-1]
        f[name] = dv
    return f


def to_state(x: np.ndarray) -> np.ndarray:
    """符号を 0=負 / 1=ちょうど 0 / 2=正 に。無効値は -1。"""
    s = np.full(len(x), -1, dtype=np.int8)
    ok = np.isfinite(x)
    s[ok] = 1
    s[ok & (x > 0)] = 2
    s[ok & (x < 0)] = 0
    return s


def count_day(s: np.ndarray, k: int) -> np.ndarray:
    """その日の k ラグ同時計数 3x3。無効値を含む組は落とす。"""
    if len(s) <= k:
        return np.zeros((3, 3), dtype=np.int64)
    a, b = s[:-k], s[k:]
    m = (a >= 0) & (b >= 0)
    if not m.any():
        return np.zeros((3, 3), dtype=np.int64)
    idx = a[m].astype(np.int64) * 3 + b[m].astype(np.int64)
    return np.bincount(idx, minlength=9).reshape(3, 3)


def excess(c: np.ndarray) -> tuple[float, float, float]:
    """(P(正|正), 無条件 P(正), 超過)。ペア集合が空なら NaN。"""
    tot = c.sum()
    if tot == 0 or c[2].sum() == 0:
        return np.nan, np.nan, np.nan
    p_cond = c[2, 2] / c[2].sum()
    p_marg = c[:, 2].sum() / tot
    return p_cond, p_marg, p_cond - p_marg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    rng = np.random.default_rng(SEED)

    cols, bnd, days = load(a.coin)
    dtype = day_types(days)
    nd = len(days)
    tk = taker_states(a.coin, days)
    names = [n for n, _, _ in FEATS] + (["taker_sign"] if tk else [])

    counts = {f: {k: np.zeros((nd, 3, 3), np.int64) for k in KS} for f in names}
    perm = {f: {k: np.zeros((nd, 3, 3), np.int64) for k in KS} for f in names}
    ident = []

    for di in range(nd):
        i0, i1 = bnd[di], bnd[di + 1]
        f = features_for_day(cols["best_bid"][i0:i1], cols["best_ask"][i0:i1],
                             cols["bid_sz"][i0:i1], cols["ask_sz"][i0:i1])
        st = {n: to_state(f[n]) for n, _, _ in FEATS}
        st["_micro_naive"] = to_state(f["_micro_naive"])
        if tk:
            st["taker_sign"] = tk.get(days[di], np.zeros(0, np.int8))

        # 恒等式: OBI / BookSlope / micro_dev の符号は一致するはず。
        # _micro_naive は「素朴に書くと壊れる」ことを示すための対照。
        base = st["OBI"]
        for other in ("BookSlope", "micro_dev", "_micro_naive"):
            mm = st[other] != base
            ident.append({
                "dt": days[di], "pair": f"OBI vs {other}", "n": int(len(base)),
                "n_mismatch": int(mm.sum()),
                # 不一致が「ちょうど 0」の行に集中しているか
                "n_mismatch_at_zero": int((mm & (base == 1)).sum()),
                "n_zero": int((base == 1).sum()),
            })

        for n in names:
            s = st[n]
            sp = s.copy()
            ok = np.flatnonzero(s >= 0)
            if len(ok) > 1:
                sp[ok] = s[rng.permutation(ok)]
            for k in KS:
                counts[n][k][di] = count_day(s, k)
                perm[n][k][di] = count_day(sp, k)
        if (di + 1) % 20 == 0:
            print(f"  {di + 1}/{nd} 日", file=sys.stderr)

    rows = []
    for n in names:
        for k in KS:
            for di in range(nd):
                c = counts[n][k][di]
                if c.sum() == 0:
                    continue
                for i in range(3):
                    for j in range(3):
                        rows.append({"dt": days[di], "day_type": dtype[days[di]],
                                     "feat": n, "k": k, "s_from": STATES[i],
                                     "s_to": STATES[j], "n": int(c[i, j])})
    pl.DataFrame(rows).write_parquet(
        ROOT / "data" / f"sign_persist_counts_{tag}.parquet")

    I = pl.DataFrame(ident)
    I.write_parquet(ROOT / "data" / f"sign_persist_ident_{tag}.parquet")

    out = []
    for n in names:
        for dtl in ["全日", "立会日", "閉場日"]:
            sel = np.array([di for di in range(nd)
                            if dtl == "全日" or dtype[days[di]] == dtl])
            if len(sel) == 0:
                continue
            for k in KS:
                C = counts[n][k][sel]
                agg = C.sum(axis=0)
                p_cond, p_marg, ex = excess(agg)
                bs = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    pick = rng.integers(0, len(sel), len(sel))
                    bs[b] = excess(C[pick].sum(axis=0))[2]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                pex = excess(perm[n][k][sel].sum(axis=0))
                # ★帰無対照は 0 に潰れ切らない。日ごとに P(正) が違う状態で
                # 条件付き確率を全日合算すると、正の多い日が分子側で重く効く
                # (集計バイアス)。その大きさが excess_perm である。
                # したがって持続性の推定値は 超過 − 帰無対照 で見る。
                row = {"feat": n, "day_type": dtl, "k": k,
                       "clock": CLOCK.get(n, "BBO イベント"),
                       "n_days": int(len(sel)), "n_pairs": int(agg.sum()),
                       "p_pos_given_pos": p_cond, "p_pos_uncond": p_marg,
                       "excess": ex, "excess_lo": float(lo), "excess_hi": float(hi),
                       "excess_perm": pex[2], "p_pos_perm": pex[0],
                       "excess_adj": ex - pex[2],
                       "excess_adj_lo": float(lo) - pex[2],
                       "excess_adj_hi": float(hi) - pex[2]}
                for i in range(3):
                    rs = agg[i].sum()
                    for j in range(3):
                        row[f"p_{i}{j}"] = agg[i, j] / rs if rs else np.nan
                    row[f"n_from_{i}"] = int(rs)
                out.append(row)
    C = pl.DataFrame(out)
    C.write_parquet(ROOT / "data" / f"sign_persist_cells_{tag}.parquet")

    print("\n=== 恒等式の検算(OBI と符号が一致するか)===", file=sys.stderr)
    for (pair,), g in I.group_by("pair"):
        mm, z = g["n_mismatch"].sum(), g["n_mismatch_at_zero"].sum()
        print(f"  {pair:<26} 不一致 {mm:>9,} / {g['n'].sum():,} 行"
              f"   うち OBI がちょうど 0 の行 {z:,}"
              f" ({z / mm * 100:.1f}%)" if mm else
              f"  {pair:<26} 不一致 0 / {g['n'].sum():,} 行  — 完全一致",
              file=sys.stderr)

    print("\n=== P(k イベント後も正 | 今が正)[%] — 全日 ===", file=sys.stderr)
    print(f"{'特徴量':<12}{'時計':<10}{'無条件':>8}"
          + "".join(f"{'k=' + str(k):>9}" for k in KS), file=sys.stderr)
    for n in names:
        s = C.filter((pl.col("feat") == n) & (pl.col("day_type") == "全日")).sort("k")
        if s.height == 0:
            continue
        line = f"{n:<12}{CLOCK.get(n, 'BBO'):<10}{s['p_pos_uncond'][0] * 100:>7.2f}%"
        for k in KS:
            line += f"{s.filter(pl.col('k') == k)['p_pos_given_pos'][0] * 100:>8.2f}%"
        print(line, file=sys.stderr)

    print("\n=== 持続性 = 超過 − 帰無対照[pp]。括弧内は帰無対照 ===", file=sys.stderr)
    print(f"{'特徴量':<12}" + "".join(f"{'k=' + str(k):>18}" for k in KS),
          file=sys.stderr)
    for n in names:
        s = C.filter((pl.col("feat") == n) & (pl.col("day_type") == "全日")).sort("k")
        if s.height == 0:
            continue
        line = f"{n:<12}"
        for k in KS:
            r = s.filter(pl.col("k") == k)
            line += (f"{r['excess_adj'][0] * 100:>+11.2f}"
                     f" ({r['excess_perm'][0] * 100:>+.2f})")
        print(line, file=sys.stderr)

    print("\n符号を持たず載せられない特徴量(常に正で退化): "
          + " / ".join(DEGENERATE), file=sys.stderr)

    print(f"\n-> data/sign_persist_cells_{tag}.parquet ほか 2 本", file=sys.stderr)


if __name__ == "__main__":
    main()
