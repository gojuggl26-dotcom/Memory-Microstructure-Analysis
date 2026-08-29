r"""特徴量 34 — 注文の生存時間(競合リスクつき生存分析)。

【素材】
  `band_place_{mid}.parquet`(188 市場・497.8 万注文)。1 行 1 注文で
    lifetime_s   placed から終端までの秒数(ブロック時刻の内挿)
    terminal     filled / cancelled / censored(窓終端で生存)
    dist_pp      発注時点の mid からの距離[pp]
    in_band      報酬帯の内側か
    is_top1      発注量 1 位の口座か
    side / size / market

【★競合リスクを正しく扱う】
  取消と約定は**互いに打ち消し合う終端**である。
  「約定までの時間」を見るとき取消を単に除くと、**取消されやすい注文ほど
  約定しやすく見える**(競合リスクの古典的な誤り)。よって:

    - 生存関数 S(t)  … 「まだ板にある確率」。Kaplan–Meier(打ち切りは右打ち切りのみ)
    - CIF_fill(t)    … 「t までに約定した確率」。Aalen–Johansen
    - CIF_cancel(t)  … 「t までに取消された確率」
    CIF_fill + CIF_cancel + S = 1 が常に成り立つ(検算に使う)

  「約定までの時間の中央値」は**存在しない**ことが多い(約定率が 5.5% なので
  CIF_fill は 0.5 に達しない)。中央値を出さず CIF の水準で語る。

【出力】
  data/lifetime_curves.parquet   層別の S / CIF(対数時間グリッド)
  data/lifetime_summary.parquet  層ごとの要約
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# 対数グリッド(秒)。0 秒(同一ブロック)を別扱いするため 0 を先頭に置く
GRID = np.concatenate([[0.0], np.geomspace(1.0, 30 * 86400, 160)])


def curves(t: np.ndarray, ev: np.ndarray) -> dict:
    """競合リスク下の S / CIF を対数グリッド上で返す。

    ev: 0=打ち切り, 1=約定, 2=取消
    ビンごとに (リスク集合, 各事象数) を作って離散版の推定量を計算する。
    """
    idx = np.searchsorted(GRID, t, side="right") - 1
    idx = np.clip(idx, 0, len(GRID) - 1)
    nb = len(GRID)
    d1 = np.bincount(idx[ev == 1], minlength=nb).astype(float)
    d2 = np.bincount(idx[ev == 2], minlength=nb).astype(float)
    d0 = np.bincount(idx[ev == 0], minlength=nb).astype(float)
    tot = d1 + d2 + d0
    n = len(t) - np.concatenate([[0], np.cumsum(tot)[:-1]])   # 各ビン開始時のリスク集合
    n = np.maximum(n, 0)
    haz = np.divide(d1 + d2, n, out=np.zeros(nb), where=n > 0)
    S = np.cumprod(1.0 - haz)
    Sprev = np.concatenate([[1.0], S[:-1]])
    c1 = np.cumsum(Sprev * np.divide(d1, n, out=np.zeros(nb), where=n > 0))
    c2 = np.cumsum(Sprev * np.divide(d2, n, out=np.zeros(nb), where=n > 0))
    return {"t": GRID, "S": S, "cif_fill": c1, "cif_cancel": c2,
            "n_risk": n, "haz": haz}


def strata(P: pl.DataFrame):
    """層の定義。すべて**発注時点で判る量**で切る(将来の情報を使わない)。"""
    d = P["dist_pp"].to_numpy()
    ok = np.isfinite(d)
    q = np.full(len(d), -1)
    if ok.sum() > 100:
        b = np.quantile(d[ok], [0.25, 0.5, 0.75])
        q[ok] = np.searchsorted(b, d[ok])
    yield "全体", np.ones(P.height, bool)
    for i, lab in enumerate(["距離 最も近い 25%", "距離 25-50%", "距離 50-75%", "距離 最も遠い 25%"]):
        yield lab, q == i
    ib = P["in_band"].to_numpy()
    yield "報酬帯の内側", np.array([x is True for x in ib])
    yield "報酬帯の外側", np.array([x is False for x in ib])
    t1 = P["is_top1"].to_numpy()
    yield "支配的口座", t1
    yield "その他の口座", ~t1
    sd = P["side"].to_numpy()
    yield "long 側", sd == 0
    yield "short 側", sd == 1
    sz = P["size"].to_numpy()
    m = np.isfinite(sz) & (sz > 0)
    if m.sum() > 100:
        med = np.median(sz[m])
        yield "サイズ 中央値超", m & (sz > med)
        yield "サイズ 中央値以下", m & (sz <= med)


def main() -> int:
    fs = sorted(glob.glob(str(DATA / "band_place_*.parquet")))
    P = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    print(f"注文 {P.height:,} 件 / 市場 {P['market'].n_unique()}")

    t = P["lifetime_s"].to_numpy().astype(float)
    term = P["terminal"].to_numpy()
    ev = np.where(term == "filled", 1, np.where(term == "cancelled", 2, 0))
    # 打ち切りは lifetime が nan なので、窓終端までの時間が不明。
    # 観測できた最大時間で打ち切ったとみなす(保守的)。
    t = np.where(np.isfinite(t), t, np.nanmax(t[np.isfinite(t)]))

    rows, summ = [], []
    for lab, sel in strata(P):
        if sel.sum() < 1000:
            continue
        c = curves(t[sel], ev[sel])
        chk = np.abs(c["S"] + c["cif_fill"] + c["cif_cancel"] - 1.0).max()
        for i in range(len(GRID)):
            rows.append({"stratum": lab, "t": GRID[i], "S": c["S"][i],
                         "cif_fill": c["cif_fill"][i], "cif_cancel": c["cif_cancel"][i],
                         "n_risk": c["n_risk"][i]})
        # 要約: 生存の中央値(S が 0.5 を切る時刻)と、各時点の約定確率
        i50 = np.argmax(c["S"] <= 0.5) if (c["S"] <= 0.5).any() else -1
        def at(x):
            j = np.searchsorted(GRID, x) - 1
            return float(c["cif_fill"][max(j, 0)])
        summ.append({
            "stratum": lab, "n": int(sel.sum()),
            "median_life_s": float(GRID[i50]) if i50 >= 0 else np.nan,
            "S_at_1min": float(c["S"][np.searchsorted(GRID, 60) - 1]),
            "S_at_1h": float(c["S"][np.searchsorted(GRID, 3600) - 1]),
            "S_at_1d": float(c["S"][np.searchsorted(GRID, 86400) - 1]),
            "cif_fill_1min": at(60), "cif_fill_1h": at(3600), "cif_fill_1d": at(86400),
            "cif_fill_final": float(c["cif_fill"][-1]),
            "identity_err": float(chk),
        })
    C = pl.DataFrame(rows); S = pl.DataFrame(summ)
    C.write_parquet(DATA / "lifetime_curves.parquet")
    S.write_parquet(DATA / "lifetime_summary.parquet")

    print(f"\n検算 S + CIF_fill + CIF_cancel = 1 の最大誤差: "
          f"{S['identity_err'].max():.2e}\n")
    print(f"{'層':<20}{'n':>10}{'生存中央[s]':>12}{'1分後生存':>10}{'1時間後':>9}"
          f"{'1日後':>8}{'最終約定率':>11}")
    for r in S.iter_rows(named=True):
        ml = f"{r['median_life_s']:.0f}" if np.isfinite(r["median_life_s"]) else "—"
        print(f"{r['stratum']:<20}{r['n']:>10,}{ml:>12}{r['S_at_1min']:>10.1%}"
              f"{r['S_at_1h']:>9.1%}{r['S_at_1d']:>8.1%}{r['cif_fill_final']:>11.2%}")
    print(f"\n-> {DATA/'lifetime_curves.parquet'} / {DATA/'lifetime_summary.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
