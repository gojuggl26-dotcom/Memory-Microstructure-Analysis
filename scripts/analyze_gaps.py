"""板の隙間 13 種と OBI / OFI / 板の回復力の関係を測る。

【3 つの関係を分けて測る】
    OBI との関係  … どちらも格子点 T の量。**同時点**であって予測ではない
    OFI との関係  … x は T、OFI は (T−100ms, T] の流量。これも同時点
    回復力との関係 … x は T、回復力は (T, T+k] の未来。**ここだけが前向き**

この 3 つを 1 つの表に並べると「予測できた」と読み違えるので、
出力にも図にも direction 列(同時点 / 前向き)を必ず持たせる。

【板の回復力(resilience)の定義】
スプレッドが平常より広がった後、どれだけ戻るかで測る。

    s_T   … 格子点 T のスプレッド(bp)
    s̄_T   … **T より前**の 60 秒のスプレッド中央値(1 秒系列の移動中央値を
             1 つずらして使う。先読みしない)
    e_T   = s_T − s̄_T                     … 広がりの大きさ
    R_k   = (s_T − s_{T+k}) / e_T          … e_T > 0 の格子点でだけ定義

R = 1 で完全に戻る、0 で戻らない、負なら更に広がる。定義上 R は
(T, T+k] の未来を使うので**目的変数専用**である。説明変数には使わない。

R は分母が小さいときに発散するので、**[−1, +2] に切って**使う。この範囲は
定義から決めたもの(0 = 戻らない、1 = 完全復元)であって標本から作っていない。
切らない順位相関も併記するので、切り方の影響は読者が確かめられる。

【符号のあるものと無いものを分ける】
隙間 13 種のうち符号を持つのは gap asymmetry と void asymmetry の 2 つだけ。
残り 11 は大きさである。大きさを符号つきの OBI / OFI に当てても構造的に
0 になるので、**大きさには |OBI| / |OFI| を、符号つきには OBI / OFI を**
当てる。両方出して並べる。

【Pearson と Spearman を必ず並べる】
churn のレポートで、裾の重いこのデータでは Pearson が 1 つの外れ窓に
振り回されることを実測した(順位相関 0.311 に対し Pearson 0.095)。
ここでも Pearson は全格子点、Spearman は 100 点ごとの系統抽出で出す。

    uv run python scripts/analyze_gaps.py --coin xyz:MU
出力: data/gaps_rel_<coin>.csv   … 関係の一覧
      data/gaps_desc_<coin>.csv  … 13 種の記述統計
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
NG = 864_000
SUB = 100                                  # Spearman 用の系統抽出(100 格子点ごと)
RESIL_K = {"1s": 10, "2s": 20, "5s": 50, "10s": 100, "30s": 300}
BASE_SEC = 60                              # 平常スプレッドを測る秒数

FEATS = ["bid_first_gap", "ask_first_gap", "max_gap", "mean_gap", "median_gap",
         "gap_var", "gap_asym", "dist_next_bid_bp", "dist_next_ask_bp",
         "empty_count", "void_size", "void_asym", "depth_disc"]
SIGNED = {"gap_asym", "void_asym"}          # 符号を持つのはこの 2 つだけ


def resilience(sp: np.ndarray) -> dict[str, np.ndarray]:
    """スプレッド系列から R_k を作る。s̄ は T より前の 60 秒の中央値。"""
    s1 = sp[::10]                                        # 1 秒系列
    m = (pl.Series(s1).rolling_median(BASE_SEC, min_samples=10)
         .shift(1).to_numpy())                           # ★1 つずらす = T 未満
    base = np.repeat(m, 10)[:len(sp)]
    e = sp - base
    out = {}
    for name, k in RESIL_K.items():
        fut = np.full(len(sp), np.nan)
        fut[:len(sp) - k] = sp[k:]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (sp - fut) / e
        out[name] = np.where(e > 0, np.clip(r, -1.0, 2.0), np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    files = sorted((ROOT / "data" / f"gaps_{tag}").glob("dt=*.parquet"))
    days = [f.name.split("=")[1].removesuffix(".parquet") for f in files]
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    dty = {d: ("立会日" if d in sess else "閉場日") for d in days}
    print(f"[日] {len(days)} 日 = 立会日 {sum(v=='立会日' for v in dty.values())}"
          f" / 閉場日 {sum(v=='閉場日' for v in dty.values())}", file=sys.stderr)

    # 相手の量。(名前, 向き, 符号を持つか)
    TARGETS = ([("OBI", "同時点", True), ("|OBI|", "同時点", False),
                ("OFI", "同時点", True), ("|OFI|", "同時点", False)]
               + [(f"回復力 {k}", "前向き", True) for k in RESIL_K])

    acc = {}          # (feat, target, day_type) -> 十分統計量
    sub_rows = []
    desc = {}
    for f, day in zip(files, days):
        D = pl.read_parquet(f)
        sp = D["spread_bp"].to_numpy()
        R = resilience(sp)
        obi = D["obi"].to_numpy(); ofi = D["ofi"].to_numpy()
        tgt = {"OBI": obi, "|OBI|": np.abs(obi), "OFI": ofi, "|OFI|": np.abs(ofi)}
        tgt |= {f"回復力 {k}": v for k, v in R.items()}
        dt = dty[day]
        X = {c: D[c].to_numpy() for c in FEATS}
        for c, v in X.items():
            g = v[np.isfinite(v)]
            if len(g):
                s = desc.setdefault(c, [0, 0.0, 0.0, np.inf, -np.inf])
                s[0] += len(g); s[1] += g.sum(); s[2] += (g * g).sum()
                s[3] = min(s[3], g.min()); s[4] = max(s[4], g.max())
        for c, xv in X.items():
            for tn, _, signed in TARGETS:
                if (c in SIGNED) != signed and tn in ("OBI", "|OBI|", "OFI", "|OFI|"):
                    continue                     # 大きさ↔|·|、符号つき↔符号つき
                yv = tgt[tn]
                m = np.isfinite(xv) & np.isfinite(yv)
                n = int(m.sum())
                if n < 30:
                    continue
                x, y = xv[m], yv[m]
                k = (c, tn, dt)
                v = np.array([n, x.sum(), y.sum(), (x * y).sum(),
                              (x * x).sum(), (y * y).sum()])
                acc[k] = acc.get(k, 0) + v
        s = slice(None, None, SUB)
        sub_rows.append(pl.DataFrame(
            {"day_type": np.full(len(range(0, NG, SUB)), dt), "spread_bp": sp[s]}
            | {c: X[c][s] for c in FEATS}
            | {tn: tgt[tn][s] for tn, _, _ in TARGETS}))
        print(f"  {day}", file=sys.stderr, flush=True)

    S = pl.concat(sub_rows)
    # 図の用量反応(十分位ごとの平均)にも使うので残す
    S.write_parquet(ROOT / "data" / f"gaps_sub_{tag}.parquet")
    print(f"[抽出] Spearman 用 {S.height:,} 行", file=sys.stderr)

    rows = []
    for (c, tn, dt), v in acc.items():
        n, sx, sy, sxy, sxx, syy = v
        den = n * sxx - sx * sx
        dy = n * syy - sy * sy
        if den <= 0 or dy <= 0:
            continue
        r = (n * sxy - sx * sy) / np.sqrt(den * dy)
        b = (n * sxy - sx * sy) / den
        t = r * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - r * r, 1e-18))
        # ★polars の drop_nulls は Float64 の NaN を落とさない。明示的に外す
        sub = (S.filter(pl.col("day_type") == dt).select(c, tn).drop_nulls()
               .filter(pl.col(c).is_not_nan() & pl.col(tn).is_not_nan()))
        sp_ = (st.spearmanr(sub[c].to_numpy(), sub[tn].to_numpy()).statistic
               if sub.height > 100 else np.nan)
        # ★スプレッドを抜いた偏順位相関。dist_next_*_bp は定義上 半スプレッドを
        #   含み(実測 ρ = +0.79)、回復力はスプレッドの平均回帰そのものなので、
        #   素の相関はスプレッドの代理でしかない可能性がある。
        pr = np.nan
        if tn.startswith("回復力"):
            s3 = (S.filter(pl.col("day_type") == dt).select(c, tn, "spread_bp")
                  .drop_nulls().filter(pl.col(c).is_not_nan()
                                       & pl.col(tn).is_not_nan()
                                       & pl.col("spread_bp").is_not_nan()))
            if s3.height > 100:
                xr, yr, zr = (st.rankdata(s3[k].to_numpy())
                              for k in (c, tn, "spread_bp"))
                rxy = np.corrcoef(xr, yr)[0, 1]
                rxz = np.corrcoef(xr, zr)[0, 1]
                ryz = np.corrcoef(yr, zr)[0, 1]
                den2 = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
                pr = (rxy - rxz * ryz) / den2
        rows.append({"feat": c, "target": tn,
                     "direction": "前向き" if tn.startswith("回復力") else "同時点",
                     "day_type": dt, "n": int(n), "pearson": r, "spearman": sp_,
                     "partial_sp": pr, "beta": b, "t": t, "n_sub": sub.height})
    out = pl.DataFrame(rows).sort("feat", "target", "day_type")
    out.write_csv(ROOT / "data" / f"gaps_rel_{tag}.csv")

    dd = [{"feat": c, "n": s[0], "mean": s[1] / s[0],
           "sd": np.sqrt(max(s[2] / s[0] - (s[1] / s[0]) ** 2, 0)),
           "min": s[3], "max": s[4]} for c, s in desc.items()]
    pl.DataFrame(dd).write_csv(ROOT / "data" / f"gaps_desc_{tag}.csv")
    print(f"[関係] {out.height} 行 -> data/gaps_rel_{tag}.csv / gaps_desc_{tag}.csv",
          file=sys.stderr)


if __name__ == "__main__":
    main()
