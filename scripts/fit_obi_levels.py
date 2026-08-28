"""日ごとの累積和から水準別 OBI の回帰係数を出し、区間をつける。

build_obi_levels.py が日 × 種別 × 定義 × 水準 × ホライズンごとに
(n, Σx, Σy, Σx², Σxy, Σy²) を残しているので、ここで足し合わせて

    傾き b = (nΣxy − ΣxΣy) / (nΣx² − (Σx)²)     … OBI 1 単位あたりの bp
    相関 r = 共分散 / (σx σy)

を出す。**日を単位にしたブロックブートストラップ**(日を復元抽出して累積和を
足し直す)で区間を出す。100ms 格子で 50 秒先を見ると窓は 500 倍重なるうえ
OBI 自体の自己相関も強いので、古典的な標準誤差は使えないため。

【多重比較】格子は 31 系列 × 9 ホライズン = 279 通りある。
95% 区間に加えて **Bonferroni 補正した区間**(1 − 0.05/279)も出す。

【傾きの読み方】x は [−1, +1] の無次元量、y は bp なので、傾きはそのまま
「OBI が 0 から +1 へ振れたときの期待 log リターン(bp)」である。
水準どうしを比べるために **σx を掛けた値**(x が 1σ 動いたときの bp)も出す。
σx は全標本の記述統計であって、特徴量を作るときの標準化には使っていない
(標準化パラメータを全標本から作らないという規約に触れない)。

    uv run python scripts/fit_obi_levels.py --coin xyz:MU
出力: data/obi_levels_fit_<coin>.parquet / .csv
      data/obi_levels_dose_<coin>.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 400
SEED = 20260828
NTEST = 279                      # (10 + 10 + 1 + 10) 系列 × 9 ホライズン
COLS = ["n", "sx", "sy", "sxx", "sxy", "syy"]


def day_types(days: list[str]) -> dict[str, str]:
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    return {d: ("立会日" if d in sess else "閉場日") for d in days}


def stats(A: np.ndarray) -> tuple[np.ndarray, ...]:
    """(…, 6) の累積和から傾き・相関・σx・平均 y を出す。"""
    n, sx, sy, sxx, sxy, syy = (A[..., i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cxy = sxy - sx * sy / n
        b = np.where(vx > 0, cxy / vx, np.nan)
        r = np.where((vx > 0) & (vy > 0), cxy / np.sqrt(vx * vy), np.nan)
        sd = np.where(n > 1, np.sqrt(np.maximum(vx, 0) / (n - 1)), np.nan)
    return b, r, sd, sy / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    src = ROOT / "data" / "obi_levels_days" / tag
    files = sorted(src.glob("dt=*.parquet"))
    if not files:
        sys.exit(f"{src} が空。先に build_obi_levels.py を実行すること")
    D = pl.concat([pl.read_parquet(f) for f in files])
    days = sorted(D["dt"].unique().to_list())
    dt_of = day_types(days)
    print(f"[日] {len(days)} 日 = 立会日 {sum(v=='立会日' for v in dt_of.values())} "
          f"/ 閉場日 {sum(v=='閉場日' for v in dt_of.values())}", file=sys.stderr)

    keys = ["kind", "defn", "lv", "h", "h_ms"]
    D = D.sort(keys + ["dt"])
    grp = D.group_by(keys, maintain_order=True).agg(pl.col("dt"),
                                                    *[pl.col(c) for c in COLS])
    rng = np.random.default_rng(SEED)
    scopes = {"全日": days,
              "立会日": [d for d in days if dt_of[d] == "立会日"],
              "閉場日": [d for d in days if dt_of[d] == "閉場日"]}
    boot = {s: rng.integers(0, len(v), size=(N_BOOT, len(v))) for s, v in scopes.items()}
    q_lo, q_hi = 2.5, 97.5
    bq_lo, bq_hi = 100 * 0.5 * 0.05 / NTEST, 100 * (1 - 0.5 * 0.05 / NTEST)

    rows = []
    for rec in grp.iter_rows(named=True):
        dd = rec["dt"]
        A_all = np.stack([np.asarray(rec[c], float) for c in COLS], axis=1)
        pos = {d: i for i, d in enumerate(dd)}
        for sc, sdays in scopes.items():
            sel = [pos[d] for d in sdays if d in pos]
            if len(sel) < 3:
                continue
            A = A_all[sel]
            tot = A.sum(0)
            b, r, sd, my = stats(tot)
            # 日を復元抽出して累積和を足し直す
            idx = boot[sc][:, :len(sel)] % len(sel)
            bs = A[idx].sum(1)                       # (N_BOOT, 6)
            bb = stats(bs)[0]
            bb = bb[np.isfinite(bb)]
            lo, hi = (np.percentile(bb, [q_lo, q_hi]) if bb.size else (np.nan, np.nan))
            blo, bhi = (np.percentile(bb, [bq_lo, bq_hi]) if bb.size else (np.nan, np.nan))
            rows.append({"scope": sc, **{k: rec[k] for k in keys},
                         "n_day": len(sel), "n": float(tot[0]),
                         "slope": float(b), "ci_lo": float(lo), "ci_hi": float(hi),
                         "bonf_lo": float(blo), "bonf_hi": float(bhi),
                         "r": float(r), "sd_x": float(sd),
                         "slope_per_sd": float(b * sd), "mean_y": float(my)})
    F = pl.DataFrame(rows)
    F.write_parquet(ROOT / "data" / f"obi_levels_fit_{tag}.parquet")
    F.write_csv(ROOT / "data" / f"obi_levels_fit_{tag}.csv")

    # ---- 用量反応(帯ごとの平均 y)-------------------------------------------
    bsrc = ROOT / "data" / "obi_levels_bins" / tag
    B = pl.concat([pl.read_parquet(f) for f in sorted(bsrc.glob("dt=*.parquet"))])
    B = B.with_columns(scope=pl.col("dt").replace_strict(dt_of, default="全日"))
    Bd = (pl.concat([B.with_columns(scope=pl.lit("全日")), B])
          .group_by("scope", "defn", "lv", "h", "h_ms", "bin")
          .agg(n=pl.col("n").sum(), sy=pl.col("sy").sum())
          .with_columns(mean_bp=pl.col("sy") / pl.col("n")).sort("scope", "defn", "lv", "h_ms", "bin"))
    Bd.write_parquet(ROOT / "data" / f"obi_levels_dose_{tag}.parquet")

    # ---- 画面表示 --------------------------------------------------------------
    for defn in ["水準ごと", "累積", "水準ごと(L1一致)", "bbo 最良"]:
        print(f"\n=== 傾き[bp / OBI 1 単位] — {defn}・全日・実測 ===", file=sys.stderr)
        s = F.filter((pl.col("scope") == "全日") & (pl.col("kind") == "実測")
                     & (pl.col("defn") == defn))
        hs = s.sort("h_ms")["h"].unique(maintain_order=True).to_list()
        print("  水準 " + "".join(f"{h:>9}" for h in hs), file=sys.stderr)
        for lv in sorted(s["lv"].unique().to_list()):
            t = s.filter(pl.col("lv") == lv).sort("h_ms")
            print(f"  {lv:>4} " + "".join(f"{v:>9.3f}" for v in t["slope"]), file=sys.stderr)
        p = F.filter((pl.col("scope") == "全日") & (pl.col("kind") == "帰無対照")
                     & (pl.col("defn") == defn))
        print("  帰無 " + "".join(
            f"{p.filter(pl.col('h')==h)['slope'].abs().max():>9.3f}" for h in hs)
            + "   ← 帰無対照の絶対値の最大", file=sys.stderr)
    print(f"\n-> data/obi_levels_fit_{tag}.parquet", file=sys.stderr)


if __name__ == "__main__":
    main()
