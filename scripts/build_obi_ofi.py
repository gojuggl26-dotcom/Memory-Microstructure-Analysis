"""イベントごとに OBI と OFI を出し、将来リターンが正になる確率の行列を作る。

【2 つの説明変数】

    OBI_t = (bid_sz_t − ask_sz_t) / (bid_sz_t + ask_sz_t)        ∈ [−1, +1]

板の**残高**の偏り。今この瞬間に、最良気配のどちら側が厚いか。

    OFI_t = 1{P^b_t ≥ P^b_{t−1}}·q^b_t − 1{P^b_t ≤ P^b_{t−1}}·q^b_{t−1}
          − 1{P^a_t ≤ P^a_{t−1}}·q^a_t + 1{P^a_t ≥ P^a_{t−1}}·q^a_{t−1}

板の**変化**の偏り(Cont–Kukanov–Stoikov 2014)。1 イベントで買い側に
どれだけ足され、売り側からどれだけ引かれたか。正なら買い圧。

**この 2 つは「残高」と「流量」であって別物である。** OBI は水準、OFI は差分に近い。

【★OFI は正規化しないと帯が意味を持たない】
OFI は枚数の単位を持ち、尺度が日によって 5 倍以上変わる(実測: 1 日の標準偏差が
土曜 6.8 に対し決算日 37.7)。固定の帯を当てると、活発な日は全部が外側の帯に、
静かな日は全部が内側の帯に落ちてしまう。

そこで **直前 W イベントの標準偏差**で割って無次元にする。

    OFI_z_t = OFI_t / s_{t−1},   s_{t−1} = std(OFI_{t−W}, …, OFI_{t−1})

`shift(1)` を入れて **自分自身を尺度の計算に含めない**。全標本の標準偏差を使うと
「標準化のパラメータを全標本から作らない」という規約に反するため、
必ず後ろ向きの窓だけで作る。

【帯の切り方は先に機械的に決める】
分布を見てから決めると「結果を見てからの選択」になる。次の規則で先に決めた。

    OBI    … [−1, +1] を **等幅 0.25 で 8 分割**(対称・等間隔)
    OFI_z  … 標準得点の慣用的な切れ目 ±0.5, ±1, ±2 で **8 分割**(対称)

【x が確定する時刻 / y の期間】
    x = OBI_t または OFI_z_t   … 時刻 t までの板だけで決まる。t で確定する
    y = mid_{t+k} − mid_t      … 期間は (t, t+k]。t より後にしか判らない
先読みは無い。`shift(-k)` は y にしか使っていない。
mid > 0 なので符号(正/負)はリターンの符号と厳密に一致する。

【★同値(y = 0)を必ず分けて報告する】
価格は離散なので短いホライズンでは「変化なし」が多数を占める(k=1 で 3〜6 割)。
P(上昇) だけを出すと全セルが 50% を下回って見え、読み手が誤る。
上昇・下落・変化なしの 3 つと、動いた場合に限った P(上昇 | 変化あり) を併記する。

【対照】
- 何もしない場合: そのホライズン・その日区分・その説明変数での **無条件の** P(上昇)。
  各セルはこれと比べる。0.5 と比べてはいけない(ドリフトと同値の偏りがあるため)
- 帰無対照: 帯の割り当てを日内で巡回シフトして同じ集計をする。
  x 自身の分布と自己相関は保ったまま y との対応だけを壊す。偶然なら無条件値に潰れる

【不確かさ】
窓が重なり自己相関があるので二項の信頼区間は使えない。
**日単位のブロックブートストラップ**で、無条件値からの差 (p_up − base) の区間を出す。

    uv run python scripts/build_obi_ofi.py --coin xyz:MU
出力: data/obi_ofi_daily_<coin>.parquet  … 日 × 説明変数 × 帯 × ホライズンの素の計数
      data/obi_ofi_cells_<coin>.parquet  … 日区分 × 説明変数 × 帯 × ホライズンの集計
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 5, 10, 20, 30, 50, 100]
ROLL_W = 5_000        # OFI の尺度を測る後ろ向きの窓(イベント数)
N_BOOT = 400
SEED = 20260828

OBI_EDGES = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
OBI_LABELS = ["−1.00〜−0.75", "−0.75〜−0.50", "−0.50〜−0.25", "−0.25〜0",
              "ちょうど 0", "0〜+0.25", "+0.25〜+0.50", "+0.50〜+0.75", "+0.75〜+1.00"]

OFI_EDGES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
OFI_LABELS = ["< −2σ", "−2〜−1σ", "−1〜−0.5σ", "−0.5〜0σ",
              "ちょうど 0", "0〜+0.5σ", "+0.5〜+1σ", "+1〜+2σ", "> +2σ"]

FEATS = {"OBI": (OBI_EDGES, OBI_LABELS), "OFI_z": (OFI_EDGES, OFI_LABELS)}
NB = len(OBI_LABELS)          # 9 = 8 帯 + ちょうど 0
JOINT_K = [1, 10, 100]        # OBI と OFI の同時分布を見るホライズン


def assign_bins(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """帯番号を返す。**ちょうど 0 は専用の帯**にする(無効値は −1)。

    ちょうど 0 は実測で OBI の 1.35% を占め(両建てで同数量を出す気配)、
    将来リターンの符号は中立(P(上昇|変化あり) = 49.9%)である。
    np.digitize は既定で右開なのでこれを正側の帯に入れてしまい、
    その帯の値を 0.5pp ほど押し下げる。分けて数える。
    """
    ok = np.isfinite(x)
    b = np.digitize(np.nan_to_num(x), edges)     # 0..7、0 は帯 4 に入る
    out = np.where(b < 4, b, np.where(b == 4, 5, b + 1))   # 4 を空けて詰め直す
    out = np.where(ok & (x == 0.0), 4, out)                # ちょうど 0 を帯 4 へ
    return np.where(ok, out, -1).astype(np.int64)


def load(coin: str) -> pl.DataFrame:
    tag = coin.replace(":", "_")
    p = ROOT / "data" / f"bbo_{tag}.parquet"
    if not p.exists():
        sys.exit(f"{p} が無い。先に fetch_bbo.py を実行すること")
    d = pl.read_parquet(p).sort("ts")
    n0 = d.height
    # 板が壊れている行を落とす。CLAUDE.md の「is_crossed は必ず除外」に従う。
    d = d.filter(
        (pl.col("best_ask") > pl.col("best_bid"))
        & (pl.col("bid_sz") > 0) & (pl.col("ask_sz") > 0)
        & pl.col("best_bid").is_finite() & pl.col("best_ask").is_finite()
    )
    print(f"[load] {n0:,} 行 -> {d.height:,} 行(クロス・数量 0 を除去 {n0 - d.height:,})",
          file=sys.stderr)
    return d


def build_features(d: pl.DataFrame) -> pl.DataFrame:
    """OBI と OFI を付ける。OFI は日を跨がせない(日の初回イベントは null)。"""
    d = d.with_columns(
        mid=(pl.col("best_bid") + pl.col("best_ask")) / 2,
        obi=(pl.col("bid_sz") - pl.col("ask_sz")) / (pl.col("bid_sz") + pl.col("ask_sz")),
    )
    pb, pa = pl.col("best_bid"), pl.col("best_ask")
    qb, qa = pl.col("bid_sz"), pl.col("ask_sz")
    d = d.with_columns(
        ofi=(
            pl.when(pb >= pb.shift(1)).then(qb).otherwise(0.0)
            - pl.when(pb <= pb.shift(1)).then(qb.shift(1)).otherwise(0.0)
            - pl.when(pa <= pa.shift(1)).then(qa).otherwise(0.0)
            + pl.when(pa >= pa.shift(1)).then(qa.shift(1)).otherwise(0.0)
        )
    )
    # 日の最初の行は前の行が前日なので OFI を無効にする
    d = d.with_columns(
        ofi=pl.when(pl.col("dt") == pl.col("dt").shift(1)).then(pl.col("ofi")).otherwise(None)
    )
    # ★尺度は「直前 W イベント」だけから作る。shift(1) で自分を除く
    d = d.with_columns(
        scale=pl.col("ofi").rolling_std(ROLL_W, min_samples=ROLL_W // 2).shift(1)
    ).with_columns(
        ofi_z=pl.when(pl.col("scale") > 0).then(pl.col("ofi") / pl.col("scale")).otherwise(None)
    )
    return d


def day_types(days: list[str]) -> dict[str, str]:
    cal = xc.get_calendar("XNYS")
    sess = {x.date().isoformat() for x in cal.sessions_in_range(days[0], days[-1])}
    return {d: ("立会日" if d in sess else "閉場日") for d in days}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    d = build_features(load(a.coin))
    days = sorted(d["dt"].unique().to_list())
    dtype = day_types(days)
    print(f"[日] {len(days)} 日 = 立会日 {sum(v == '立会日' for v in dtype.values())}"
          f" / 閉場日 {sum(v == '閉場日' for v in dtype.values())}", file=sys.stderr)

    obi_all = d["obi"].to_numpy()
    ofiz_all = d["ofi_z"].to_numpy()
    print(f"[OFI] 正規化できた行 {np.isfinite(ofiz_all).sum():,} / {len(ofiz_all):,}"
          f"(残りは尺度の助走 {ROLL_W:,} イベント分と日の初回)", file=sys.stderr)
    print(f"[OFI_z] 中央値 {np.nanmedian(ofiz_all):+.4f} / "
          f"|z|>2 の割合 {np.nanmean(np.abs(ofiz_all) > 2) * 100:.2f}%", file=sys.stderr)
    print(f"[OBI]   中央値 {np.nanmedian(obi_all):+.4f} / "
          f"|OBI|>0.75 の割合 {np.nanmean(np.abs(obi_all) > 0.75) * 100:.2f}%", file=sys.stderr)

    mid_all = d["mid"].to_numpy()
    dt_all = d["dt"].to_numpy()
    rng = np.random.default_rng(SEED)

    # ★費用の目安。確率の大きさは、この半スプレッドと比べて初めて意味を持つ
    spr_bp = ((d["best_ask"] - d["best_bid"]) / d["mid"] * 1e4).to_numpy()
    meta = pl.DataFrame({
        "spread_bp_median": [float(np.median(spr_bp))],
        "spread_bp_mean": [float(spr_bp.mean())],
        "half_spread_bp_median": [float(np.median(spr_bp) / 2)],
        "n_events": [int(len(spr_bp))],
    })
    meta.write_parquet(ROOT / "data" / f"obi_ofi_meta_{tag}.parquet")
    print(f"[費用] スプレッド中央値 {np.median(spr_bp):.3f} bp / "
          f"片道の半スプレッド {np.median(spr_bp) / 2:.3f} bp", file=sys.stderr)

    rows = []
    for day in days:
        m = dt_all == day
        mi = mid_all[m]
        n = len(mi)
        if n < 200:
            continue
        for feat, (edges, labels) in FEATS.items():
            x = (obi_all if feat == "OBI" else ofiz_all)[m]
            b = assign_bins(x, edges)
            # 帰無対照: 帯だけを日内で巡回シフトする
            sh = int(rng.integers(n // 5, 4 * n // 5))
            b_pl = np.roll(b, sh)
            for k in HORIZONS:
                if n <= k:
                    continue
                y = mi[k:] - mi[:-k]
                # ★確率だけでなく **値動きの大きさ**(bp)も持つ。
                #   P(上昇) が大きくても、動く幅がスプレッドより小さければ
                #   板を取りに行っては捕まえられない。費用と比べられる形で残す。
                ybp = y / mi[:-k] * 1e4
                bb, bp = b[:-k], b_pl[:-k]
                up = (y > 0).astype(np.int64)
                dn = (y < 0).astype(np.int64)
                v = bb >= 0
                cnt = np.bincount(bb[v], minlength=NB)
                cu = np.bincount(bb[v], weights=up[v], minlength=NB)
                cd = np.bincount(bb[v], weights=dn[v], minlength=NB)
                cs = np.bincount(bb[v], weights=ybp[v], minlength=NB)
                vp = bp >= 0
                cnp = np.bincount(bp[vp], minlength=NB)
                cup = np.bincount(bp[vp], weights=up[vp], minlength=NB)
                # ★プラセボも本表と同じ「変化あり」基準で比べられるように下落も数える。
                #   生の確率で比べると、帯ごとに変化なし率が違う分だけ差が残り、
                #   帰無が潰れていないように見えてしまう(「ちょうど 0」は変化なしが
                #   他の帯の半分以下)。
                cdp = np.bincount(bp[vp], weights=dn[vp], minlength=NB)
                for i in range(NB):
                    if cnt[i] == 0:
                        continue
                    rows.append({
                        "dt": day, "day_type": dtype[day], "feat": feat,
                        "bin": labels[i], "bin_i": i, "k": k,
                        "n": int(cnt[i]), "n_up": int(cu[i]), "n_dn": int(cd[i]),
                        "sum_bp": float(cs[i]),
                        "n_pl": int(cnp[i]), "n_up_pl": int(cup[i]),
                        "n_dn_pl": int(cdp[i]),
                    })
    D = pl.DataFrame(rows)
    D.write_parquet(ROOT / "data" / f"obi_ofi_daily_{tag}.parquet")
    print(f"[集計] 日×説明変数×帯×ホライズン {D.height:,} 行", file=sys.stderr)

    # ---- 日を単位にブロックブートストラップ。無条件値からの差の区間を出す --------
    out = []
    for feat, (_, labels) in FEATS.items():
        for dtl in ["立会日", "閉場日"]:
            sub = D.filter((pl.col("feat") == feat) & (pl.col("day_type") == dtl))
            if sub.height == 0:
                continue
            dl = sorted(sub["dt"].unique().to_list())
            idx = {v: i for i, v in enumerate(dl)}
            nd = len(dl)
            for k in HORIZONS:
                sk = sub.filter(pl.col("k") == k)
                if sk.height == 0:
                    continue
                # 日 × 帯 の計数行列。無条件値も同じ行列から作るので標本が揃う
                N = np.zeros((nd, NB)); U = np.zeros((nd, NB)); Dn = np.zeros((nd, NB))
                S = np.zeros((nd, NB))
                for r in sk.iter_rows(named=True):
                    i, j = idx[r["dt"]], r["bin_i"]
                    N[i, j] += r["n"]; U[i, j] += r["n_up"]; Dn[i, j] += r["n_dn"]
                    S[i, j] += r["sum_bp"]
                tot_n, tot_u, tot_d = N.sum(), U.sum(), Dn.sum()
                base = tot_u / tot_n
                base_mv = tot_u / (tot_u + tot_d) if (tot_u + tot_d) else np.nan
                pick = rng.integers(0, nd, size=(N_BOOT, nd))
                bN = N[pick].sum(1); bU = U[pick].sum(1)          # (N_BOOT, NB)
                bbase = bU.sum(1) / np.maximum(bN.sum(1), 1)
                for j in range(NB):
                    nj, uj, dj = N[:, j].sum(), U[:, j].sum(), Dn[:, j].sum()
                    if nj < 100:
                        continue
                    pj = np.divide(bU[:, j], bN[:, j],
                                   out=np.full(N_BOOT, np.nan), where=bN[:, j] > 0)
                    diff = pj - bbase
                    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                    # ブートストラップ分布の正規近似で z を出す。400 回では
                    # Bonferroni が要求する裾の分位点(0.011%)を直接は解像できないため。
                    sd = float(np.nanstd(diff, ddof=1))
                    z = float(np.nanmean(diff)) / sd if sd > 0 else np.nan
                    mv = uj + dj
                    cj = sk.filter(pl.col("bin_i") == j)
                    plj_n = cj["n_pl"].sum()
                    plj_u, plj_d = cj["n_up_pl"].sum(), cj["n_dn_pl"].sum()
                    plj_mv = plj_u + plj_d
                    out.append({
                        "feat": feat, "day_type": dtl, "bin": labels[j], "bin_i": j, "k": k,
                        "n": int(nj), "p_up": uj / nj, "p_dn": dj / nj,
                        "p_flat": (nj - mv) / nj,
                        "p_up_move": uj / mv if mv > 0 else np.nan,
                        "base_up": base, "base_up_move": base_mv,
                        "edge": uj / nj - base,
                        "mean_bp": S[:, j].sum() / nj,     # 期待される符号つき値動き
                        "ci_lo": lo, "ci_hi": hi, "boot_z": z,
                        "p_up_placebo": plj_u / plj_n if plj_n else np.nan,
                        "p_up_move_placebo": plj_u / plj_mv if plj_mv else np.nan,
                    })
    C = pl.DataFrame(out)
    C.write_parquet(ROOT / "data" / f"obi_ofi_cells_{tag}.parquet")

    # ---- OBI と OFI は独立な情報か。同時分布で確かめる ------------------------
    # 片方が他方の言い換えなら、一方で層別したときにもう一方の効きが消えるはず。
    jrows = []
    for day in days:
        m = dt_all == day
        mi = mid_all[m]
        n = len(mi)
        if n < 200:
            continue
        bo = assign_bins(obi_all[m], OBI_EDGES)
        bf = assign_bins(ofiz_all[m], OFI_EDGES)
        for k in JOINT_K:
            if n <= k:
                continue
            y = mi[k:] - mi[:-k]
            o, f_ = bo[:-k], bf[:-k]
            v = (o >= 0) & (f_ >= 0)
            key = o[v] * NB + f_[v]
            up = (y[v] > 0).astype(np.int64)
            dn = (y[v] < 0).astype(np.int64)
            cnt = np.bincount(key, minlength=NB * NB)
            cu = np.bincount(key, weights=up, minlength=NB * NB)
            cd = np.bincount(key, weights=dn, minlength=NB * NB)
            nz = np.nonzero(cnt)[0]
            for key_i in nz:
                jrows.append({
                    "dt": day, "day_type": dtype[day], "k": k,
                    "obi_i": int(key_i // NB), "ofi_i": int(key_i % NB),
                    "n": int(cnt[key_i]), "n_up": int(cu[key_i]), "n_dn": int(cd[key_i]),
                })
    J = (
        pl.DataFrame(jrows)
        .group_by("day_type", "k", "obi_i", "ofi_i")
        .agg(n=pl.col("n").sum(), n_up=pl.col("n_up").sum(), n_dn=pl.col("n_dn").sum())
        .with_columns(
            p_up_move=pl.col("n_up") / (pl.col("n_up") + pl.col("n_dn")),
            obi_bin=pl.col("obi_i").map_elements(lambda i: OBI_LABELS[i], return_dtype=pl.String),
            ofi_bin=pl.col("ofi_i").map_elements(lambda i: OFI_LABELS[i], return_dtype=pl.String),
        )
        .sort("day_type", "k", "obi_i", "ofi_i")
    )
    J.write_parquet(ROOT / "data" / f"obi_ofi_joint_{tag}.parquet")

    # ---- 画面表示 --------------------------------------------------------------
    n_cells = C.height
    bonf = 0.05 / n_cells
    for feat in FEATS:
        for dtl in ["立会日", "閉場日"]:
            s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == dtl))
            if s.height == 0:
                continue
            print(f"\n=== {feat} / {dtl} — P(mid が k イベント後に上昇 | 変化あり)[%] ===",
                  file=sys.stderr)
            print(f"{'帯':<16}" + "".join(f"{'k=' + str(k):>9}" for k in HORIZONS)
                  + f"{'n':>14}", file=sys.stderr)
            for j, label in enumerate(FEATS[feat][1]):
                r = s.filter(pl.col("bin_i") == j)
                if r.height == 0:
                    continue
                cells = []
                for k in HORIZONS:
                    v = r.filter(pl.col("k") == k)
                    cells.append(f"{v['p_up_move'][0] * 100:>9.1f}" if v.height else f"{'--':>9}")
                print(f"{label:<16}" + "".join(cells) + f"{r['n'].max():>14,}", file=sys.stderr)
            print(f"{'無条件':<16}" + "".join(
                f"{s.filter(pl.col('k') == k)['base_up_move'][0] * 100:>9.1f}"
                for k in HORIZONS), file=sys.stderr)
            print(f"{'変化なし率':<16}" + "".join(
                f"{s.filter(pl.col('k') == k)['p_flat'].mean() * 100:>9.1f}"
                for k in HORIZONS), file=sys.stderr)
    # ---- 同時分布の表示(k=10、立会日)----------------------------------------
    s = J.filter((pl.col("day_type") == "立会日") & (pl.col("k") == 10))
    print("\n=== OBI × OFI_z の同時分布 — P(上昇|変化あり)[%] / 立会日 k=10 ===",
          file=sys.stderr)
    print("  片方で層別してももう一方が効くなら、2 つは別の情報を持っている",
          file=sys.stderr)
    print(f"{'OBI \\ OFI_z':<16}" + "".join(f"{lab:>11}" for lab in OFI_LABELS), file=sys.stderr)
    for i, olab in enumerate(OBI_LABELS):
        cells = []
        for jj in range(NB):
            r = s.filter((pl.col("obi_i") == i) & (pl.col("ofi_i") == jj))
            cells.append(f"{r['p_up_move'][0] * 100:>11.1f}"
                         if r.height and r["n"][0] >= 500 else f"{'--':>11}")
        print(f"{olab:<16}" + "".join(cells), file=sys.stderr)

    # 層別してもなお効いているかを 1 つの数字にする
    for fix, var in (("OBI", "OFI_z"), ("OFI_z", "OBI")):
        fi, vi = ("obi_i", "ofi_i") if fix == "OBI" else ("ofi_i", "obi_i")
        spreads = []
        for b in range(NB):
            r = s.filter((pl.col(fi) == b) & (pl.col("n") >= 500)).sort(vi)
            if r.height >= 4:
                spreads.append(float(r["p_up_move"].max() - r["p_up_move"].min()) * 100)
        if spreads:
            print(f"  {fix} を固定したときの {var} 方向の振れ幅: "
                  f"中央 {np.median(spreads):.1f}pp / 最大 {max(spreads):.1f}pp", file=sys.stderr)

    # ---- ★確率の大きさを費用と比べる ------------------------------------------
    hs = float(np.median(spr_bp) / 2)
    print(f"\n=== 期待される符号つき値動き(bp)/ 立会日。片道の半スプレッド = {hs:.3f} bp ===",
          file=sys.stderr)
    print("  確率が 65% でも、動く幅がこれを超えなければ板を取りに行っては捕まえられない",
          file=sys.stderr)
    for feat in FEATS:
        s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == "立会日"))
        print(f"  --- {feat} ---", file=sys.stderr)
        print(f"{'帯':<16}" + "".join(f"{'k=' + str(k):>10}" for k in HORIZONS), file=sys.stderr)
        for j, label in enumerate(FEATS[feat][1]):
            r = s.filter(pl.col("bin_i") == j)
            if r.height == 0:
                continue
            cells = []
            for k in HORIZONS:
                v = r.filter(pl.col("k") == k)
                cells.append(f"{v['mean_bp'][0]:>10.3f}" if v.height else f"{'--':>10}")
            print(f"{label:<16}" + "".join(cells), file=sys.stderr)
    best = C.filter(pl.col("day_type") == "立会日")["mean_bp"].abs().max()
    print(f"  最大の |期待値動き| = {best:.3f} bp(半スプレッド {hs:.3f} bp の {best/hs:.2f} 倍)",
          file=sys.stderr)

    # ---- 帰無対照が無条件値に潰れているか(本表と同じ「変化あり」基準で)---------
    print("\n=== 帰無対照(帯を日内で巡回シフト)===", file=sys.stderr)
    for feat in FEATS:
        for dtl in ["立会日", "閉場日"]:
            s = C.filter((pl.col("feat") == feat) & (pl.col("day_type") == dtl))
            if s.height == 0:
                continue
            pdev = (s["p_up_move_placebo"] - s["base_up_move"]).abs() * 100
            rdev = (s["p_up_move"] - s["base_up_move"]).abs() * 100
            print(f"  {feat}/{dtl}: 帰無 中央 {pdev.median():.3f}pt 最大 {pdev.max():.3f}pt"
                  f"  /  実測 中央 {rdev.median():.2f}pt 最大 {rdev.max():.2f}pt"
                  f"  (最大の比 {rdev.max() / max(pdev.max(), 1e-9):.0f} 倍)", file=sys.stderr)

    nsig = int(((C["ci_lo"] > 0) | (C["ci_hi"] < 0)).sum())
    zcrit = 3.86        # 両側 0.05/224 に対応する正規分位点
    nbonf = int((C["boot_z"].abs() > zcrit).sum())
    print(f"\n[多重比較] セル数 {n_cells}。Bonferroni の閾値は 0.05/{n_cells} = {bonf:.2e}"
          f"(|z| > {zcrit})", file=sys.stderr)
    print(f"  95% 区間が 0 をまたがないセル {nsig} / {n_cells}", file=sys.stderr)
    print(f"  Bonferroni を超えるセル       {nbonf} / {n_cells}", file=sys.stderr)
    print(f"-> data/obi_ofi_cells_{tag}.parquet / obi_ofi_joint_{tag}.parquet",
          file=sys.stderr)


if __name__ == "__main__":
    main()
