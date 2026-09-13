"""特徴量ライブラリ全部 × 9 ホライズンの十分位分析。

    uv run python scripts/build_decile.py --coin xyz:MU

入力: data/featlib_<coin>/dt=*.parquet(説明変数 229 本)
      data/fwd_<coin>.parquet(前向きリターン 9 ホライズン × mid/micro)
出力: data/decile_<coin>.parquet   (特徴量 × 目的変数 × 十分位)の表
      data/decile_sum_<coin>.csv   (特徴量 × 目的変数)の要約
      data/decile_null_<coin>.csv  プラセボ(特徴量を 1 営業日ずらす)の要約

## 何をするか

各特徴量を十分位に切り、十分位ごとの将来リターンの平均(bp)を出す。
ホライズンは **100ms / 300ms / 500ms / 1s / 3s / 5s / 10s / 30s / 60s**、
目的変数は **mid 建て**と **microprice 建て**の両方。

## ★十分位の境目は「前日の分布」から作る

全標本の分位を使うと、格子点 T の値が T より後の情報で順位づけられる
(CLAUDE.md の「標準化・分位・クリップのパラメータを全標本から作らない」)。
ここでは **前日の分位**を境目にする。初日は基準が無いので落とす。

## ★同順位(タイ)の扱い

離散・疎な特徴量(`obi_sign`、`*_shock`、`empty_lv_b` 等)は前日の分位が
潰れ、十分位が偏る。`max_dec_share`(最大の十分位が占める割合)と
`n_dec_pop`(値が入った十分位の数)を必ず併記し、偏った特徴量を識別できるようにする。

## ★標準誤差は日でクラスタする

5 秒格子の点どうしは強く相関している([burstiness のレポート](../reports/hyperliquid/MU/mu_burstiness_report.md)
で、独立とみなした標準誤差は最大 15.8 倍過小と実測済み)。
D10 − D1 の差は**日ごとに作ってから日を単位に**標準誤差を出す。

## ★帰無対照

同じ処理を、**特徴量だけ 1 営業日ずらした**系列に対しても回す。
4,122 通りの検定に対する Bonferroni 閾値と、プラセボで実際に何本超えるかを並べる。

## 時間契約

説明変数は格子点 T までの情報だけ(featlib が保証)。分位の境目は前日まで。
目的変数は (T, T+h]。`shift(-k)` は目的変数にしか使っていない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
NPD = 17_280                       # 1 日の格子点数(5 秒おき)
NDEC = 10
BATCH = 20


def deciles(x: np.ndarray, day: np.ndarray, nd: int) -> np.ndarray:
    """前日の分位で十分位に切る。無効な点は -1。"""
    dec = np.full(x.size, -1, np.int8)
    qs = np.linspace(0, 1, NDEC + 1)[1:-1]
    prev = None
    for d in range(nd):
        k = day == d
        v = x[k]
        f = np.isfinite(v)
        if prev is not None and f.any():
            g = np.digitize(v, prev)
            dec[np.flatnonzero(k)[f]] = g[f].astype(np.int8)
        prev = np.nanquantile(v[f], qs) if f.sum() > 1000 else None
    return dec


def run(X: dict, Y: np.ndarray, ynames: list, day: np.ndarray, nd: int,
        tag: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    DUMP = NDEC * nd
    rows, summ = [], []
    for fi, (fname, x) in enumerate(X.items()):
        dec = deciles(x, day, nd)
        ok = dec >= 0
        key = np.where(ok, dec.astype(np.int64) * nd + day, DUMP)
        cnt = np.bincount(key, minlength=DUMP + 1)[:DUMP].reshape(NDEC, nd)
        tot = cnt.sum(1)
        share = tot / max(tot.sum(), 1)
        for yi, yn in enumerate(ynames):
            s = np.bincount(key, weights=np.where(ok, Y[yi], 0.0),
                            minlength=DUMP + 1)[:DUMP].reshape(NDEC, nd)
            with np.errstate(invalid="ignore", divide="ignore"):
                m_dd = np.where(cnt > 0, s / cnt, np.nan)        # (十分位, 日)
            dmean = np.where(tot > 0, s.sum(1) / np.maximum(tot, 1), np.nan)
            diff = m_dd[NDEC - 1] - m_dd[0]
            g = np.isfinite(diff)
            sp = float(np.mean(diff[g])) if g.sum() > 2 else np.nan
            se = (float(np.std(diff[g], ddof=1) / np.sqrt(g.sum()))
                  if g.sum() > 2 else np.nan)
            rk = np.arange(1, NDEC + 1, dtype=float)
            fm = np.isfinite(dmean)
            rho = (float(np.corrcoef(np.argsort(np.argsort(rk[fm])),
                                     np.argsort(np.argsort(dmean[fm])))[0, 1])
                   if fm.sum() > 3 else np.nan)
            summ.append({"feature": fname, "target": yn, "n": int(tot.sum()),
                         "n_days": int(g.sum()), "d1": dmean[0], "d10": dmean[-1],
                         "spread_bp": sp, "se_bp": se,
                         "t": sp / se if se and se > 0 else np.nan,
                         "monotone_rho": rho,
                         "max_dec_share": float(share.max()),
                         "n_dec_pop": int((tot > 0).sum())})
            for d10 in range(NDEC):
                rows.append({"feature": fname, "target": yn, "decile": d10 + 1,
                             "n": int(tot[d10]), "mean_bp": dmean[d10]})
        if (fi + 1) % 25 == 0:
            print(f"  [{fi+1}/{len(X)}] {fname}", flush=True, file=sys.stderr)
    return pl.DataFrame(rows), pl.DataFrame(summ)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--src", default="featlib")
    ap.add_argument("--suffix", default="", help="入出力名に付ける")
    ap.add_argument("--no-null", action="store_true")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    sfx = a.suffix

    fw = pl.read_parquet(D / f"fwd_{tag}{sfx}.parquet").sort("dt", "sec")
    ynames = [c for c in fw.columns if c.startswith("fwd_")]
    ref = fw.select("dt", "sec")
    days = ref["dt"].unique(maintain_order=True).to_list()
    nd = len(days)
    dmap = {d: i for i, d in enumerate(days)}
    day_all = np.array([dmap[d] for d in ref["dt"].to_list()], np.int64)
    Yall = np.column_stack([fw[c].to_numpy().astype(np.float64) for c in ynames])
    fin = np.all(np.isfinite(Yall), axis=1)
    print(f"[dec] {nd} 日 × {NPD} = {ref.height:,} 点 / "
          f"全ホライズンがそろう点 {fin.sum():,}({fin.mean()*100:.2f}%)", file=sys.stderr)
    assert ref.height == nd * NPD, "日ごとの点数がそろっていない"

    # 目的変数の素性(ゼロが多いホライズンを明示する)
    print(f"{'目的変数':<18} {'標準偏差':>9} {'ちょうど 0 の割合':>16}")
    for i, yn in enumerate(ynames):
        v = Yall[fin, i]
        print(f"  {yn:<16} {v.std():>9.3f} {np.mean(v == 0)*100:>15.1f}%")

    cols = pl.read_parquet(D / f"{a.src}_{tag}" / f"dt={days[0]}.parquet").columns
    feats = [c for c in cols if c not in ("dt", "sec") and not c.startswith("fwd_")]
    print(f"[dec] 説明変数 {len(feats)} 本 × 目的変数 {len(ynames)} 本 = "
          f"{len(feats)*len(ynames):,} 通り", file=sys.stderr)

    Y = Yall[fin].T
    day = day_all[fin]
    allrows, allsum, nullsum = [], [], []
    for b in range(0, len(feats), BATCH):
        sel = feats[b:b + BATCH]
        t = (pl.scan_parquet(str(D / f"{a.src}_{tag}" / "dt=*.parquet"))
             .select(["dt", "sec"] + sel).collect().sort("dt", "sec"))
        assert t["sec"].to_numpy()[0] == ref["sec"].to_numpy()[0] and t.height == ref.height
        X = {c: t[c].to_numpy().astype(np.float64)[fin] for c in sel}
        r, s = run(X, Y, ynames, day, nd, tag)
        allrows.append(r)
        allsum.append(s)
        if not a.no_null:
            Xp = {c: np.roll(t[c].to_numpy().astype(np.float64), NPD)[fin] for c in sel}
            _, s2 = run(Xp, Y, ynames, day, nd, tag)
            nullsum.append(s2)
        print(f"[dec] 特徴量 {min(b+BATCH, len(feats))}/{len(feats)} 完了",
              file=sys.stderr)

    R = pl.concat(allrows)
    S = pl.concat(allsum)
    R.write_parquet(D / f"decile_{tag}{sfx}.parquet", compression="zstd")
    S.write_csv(D / f"decile_sum_{tag}{sfx}.csv")
    if nullsum:
        N = pl.concat(nullsum)
        N.write_csv(D / f"decile_null_{tag}{sfx}.csv")
    print(f"\n[dec] 十分位の表 {R.height:,} 行 / 要約 {S.height:,} 行", file=sys.stderr)


if __name__ == "__main__":
    main()
