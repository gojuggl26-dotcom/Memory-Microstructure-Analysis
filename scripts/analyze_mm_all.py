"""所有 7 銘柄のメイカー(在庫つき MM)の経済的有意性をまとめる。

    uv run python scripts/analyze_mm_all.py

入力: data/inv_days_xyz_<coin>_q1*.csv(`build_inventory.py` の日次出力)
      data/inv_lots_xyz_<coin>_q1*.parquet(建玉 1 本ずつ。金額換算に使う)
出力: data/mm_all_summary.csv   銘柄 × 設定の要約
      data/mm_all_daily.csv     銘柄 × 設定 × 日
      data/mm_all_size.csv      数量掃引だけ抜き出したもの

## 判定の設計(CLAUDE.md の自己精査に沿う)

* **対照は 0 ではなく「何もしない」。** メイカーは出さなければ損益 0・在庫 0 なので、
  ここでは対照 = 0 が正しい(十分位分析と違い、ベースラインが自動で黒字になる
  構造は無い)。したがって検定は **EV > 0** でよい。
* **単価と総額を分ける。** 1 組あたり bp が僅差でも、日次総額は組数で決まる。
  日次総額(bp とドルの両方)を別に検定する。
* **中央値だけで見ない。** 最悪日と、日次で黒字だった割合を必ず併記する。
* **多重比較。** 銘柄 × 設定の全数を数え、Bonferroni の閾値を併記する。
* **標準誤差は日でクラスタ**(Newey-West、ラグ 14)。同じ日の組どうしは
  強く相関している。
* **執行できる数量。** 幽霊注文(数量 0)の bp は上界にすぎない。
  `--size` を金額でそろえた掃引が本体で、
  `Economic Significance = Signal × ExecutableSize − 費用` を直接出す。

## 時間契約

シミュレータは各時点までの板と約定だけで発注を決め、損益は約定後に実現する。
ここでの集計に学習は無いので標本内外の区別は無い(入口の門を学習した設定は
別レポート)。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
NW_LAGS = 14


def nw(v: np.ndarray) -> tuple[float, float, int]:
    """Newey-West(ラグ 14)の平均と標準誤差。"""
    v = v[np.isfinite(v)]
    n = v.size
    if n < 5:
        return float("nan"), float("nan"), n
    e = v - v.mean()
    s = float(e @ e) / n
    for L in range(1, min(NW_LAGS, n - 1) + 1):
        c = float(e[L:] @ e[:-L]) / n
        s += 2 * (1 - L / (NW_LAGS + 1)) * c
    s = max(s, 1e-18)
    return float(v.mean()), float(np.sqrt(s / n)), n


def parse(sfx: str) -> dict:
    """_q1_sz0.108_lat130 のような接尾辞を読み解く。"""
    size = re.search(r"_sz([0-9.eE+-]+)", sfx)
    lat = re.search(r"_lat(\d+)", sfx)
    return {"size_unit": float(size.group(1)) if size else 0.0,
            "lat_ms": int(lat.group(1)) if lat else 0}


def main() -> None:
    rows, daily = [], []
    for c in COINS:
        tag = f"xyz_{c}"
        mid = float((pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                     .select(((pl.col("best_bid") + pl.col("best_ask")) / 2)
                             .alias("m")).collect()["m"]).median())
        # 対象は「素の設定」だけ。門つき・出し直し等の派生は別レポートが持つ
        for f in sorted(DATA.glob(f"inv_days_{tag}_q1*.csv")):
            sfx = f.stem.replace(f"inv_days_{tag}", "")
            if re.search(r"gated|_te|_d\d|_imp|_k\d|tmax|_rq|fix|queue|price", sfx):
                continue
            p = parse(sfx)
            D = pl.read_csv(f).sort("dt")
            if D.height < 10:
                continue
            pair = D["pair_pnl_mean"].to_numpy()          # 1 組あたり(日平均)
            npair = D["n_pair"].to_numpy().astype(float)
            tot = D["total_bp"].to_numpy()                # 日次総額(bp・単位名目)
            m_pair, se_pair, nd = nw(pair)
            m_tot, se_tot, _ = nw(tot)
            # 金額換算(数量を入れた設定だけ意味がある)。
            # bp は建玉時の mid で割ってあるので、日次の中央 mid を掛けるのは
            # 近似になる。建玉 1 本ずつの m_in を使って厳密に積む。
            S = p["size_unit"]
            usd = np.full(nd, np.nan)
            if S:
                lp = DATA / f"inv_lots_{tag}{sfx}.parquet"
                if lp.exists():
                    L = (pl.read_parquet(lp)
                         .with_columns(u=pl.col("pnl") / 1e4 * S * pl.col("m_in"))
                         .group_by("dt").agg(u=pl.col("u").sum()).sort("dt"))
                    mp = {d: v for d, v in zip(L["dt"], L["u"])}
                    usd = np.array([mp.get(d, 0.0) for d in D["dt"].to_list()])
                else:
                    usd = tot / 1e4 * S * mid
            m_usd, se_usd, _ = nw(usd)
            # 建玉重みの 1 組あたり(全組の単純平均)
            ev_w = float(np.nansum(pair * npair) / max(np.nansum(npair), 1))
            rows.append({
                "coin": f"xyz:{c}", "sfx": sfx, "size_unit": S,
                "size_usd": S * mid, "lat_ms": p["lat_ms"], "n_days": nd,
                "pairs_day": float(np.nanmean(npair)),
                "ev_pair_bp": ev_w,
                "ev_pair_nw": m_pair, "se_pair": se_pair,
                "t_pair": m_pair / se_pair if se_pair > 0 else np.nan,
                "day_bp": m_tot, "se_day_bp": se_tot,
                "t_day": m_tot / se_tot if se_tot > 0 else np.nan,
                "day_usd": m_usd, "se_day_usd": se_usd,
                "t_usd": m_usd / se_usd if se_usd > 0 else np.nan,
                "worst_day_bp": float(np.nanmin(tot)),
                "best_day_bp": float(np.nanmax(tot)),
                "p_day_pos": float(np.nanmean(tot > 0)),
                "p_off_60s": float(np.nanmean(D["p_off_60s"].to_numpy())),
                "t_off_med": float(np.nanmedian(D["t_off_med"].to_numpy())),
                "forced_rate": float(np.nansum(D["forced_n"].to_numpy())
                                     / max(np.nansum(D["n_fill"].to_numpy()), 1)),
            })
            for i, dt in enumerate(D["dt"].to_list()):
                daily.append({"coin": f"xyz:{c}", "sfx": sfx, "dt": dt,
                              "n_pair": npair[i], "pair_bp": pair[i],
                              "day_bp": tot[i],
                              "day_usd": float(usd[i]) if S else float("nan")})

    S = pl.DataFrame(rows)
    S.write_csv(DATA / "mm_all_summary.csv")
    pl.DataFrame(daily).write_csv(DATA / "mm_all_daily.csv")
    S.filter(pl.col("size_unit") > 0).write_csv(DATA / "mm_all_size.csv")

    nt = S.height
    thr = float(stats.norm.ppf(1 - 0.05 / nt / 2))
    print(f"設定は全部で {nt} 通り。Bonferroni の |t| 閾値 {thr:.2f}"
          f"(名目 5% を {nt} で割る)\n")

    base = S.filter((pl.col("size_unit") == 0) & (pl.col("lat_ms") == 0))
    print("===== ① 幽霊注文(数量 0)= 上界 =====")
    print(f"{'銘柄':<12}{'日数':>5}{'組/日':>9}{'1 組 bp':>9}{'NW se':>8}{'t':>8}"
          f"{'日次 bp':>11}{'t':>7}{'黒字日':>7}{'最悪日 bp':>12}{'60s 相殺':>9}")
    for r in base.sort("ev_pair_bp", descending=True).iter_rows(named=True):
        print(f"{r['coin']:<12}{r['n_days']:>5}{r['pairs_day']:>9,.0f}"
              f"{r['ev_pair_bp']:>9.3f}{r['se_pair']:>8.3f}{r['t_pair']:>8.2f}"
              f"{r['day_bp']:>11,.0f}{r['t_day']:>7.2f}"
              f"{r['p_day_pos']*100:>6.0f}%{r['worst_day_bp']:>12,.0f}"
              f"{r['p_off_60s']*100:>8.1f}%")

    print("\n===== ② 発注遅延 =====")
    print(f"{'銘柄':<12}{'0 ms':>10}{'65 ms':>10}{'130 ms':>10}{'130-0':>9}")
    for c in COINS:
        q = S.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("size_unit") == 0))
        g = {int(r["lat_ms"]): r["ev_pair_bp"] for r in q.iter_rows(named=True)}
        if 0 in g and 130 in g:
            print(f"xyz:{c:<8}{g.get(0, np.nan):>10.3f}{g.get(65, np.nan):>10.3f}"
                  f"{g.get(130, np.nan):>10.3f}{g[130]-g[0]:>9.3f}")

    print("\n===== ③ 執行できる数量(Signal × ExecutableSize)=====")
    print("1 組あたり bp / 1 日あたりドル。ドルは その数量で実際に約定した組の合計。")
    for c in COINS:
        q = (S.filter((pl.col("coin") == f"xyz:{c}") & (pl.col("lat_ms") == 0))
             .sort("size_usd"))
        if q.height < 2:
            continue
        print(f"\n  xyz:{c}")
        print(f"    {'数量':>12}{'組/日':>9}{'1 組 bp':>9}{'t':>7}"
              f"{'日次 $':>12}{'se':>10}{'t':>7}{'年 $':>14}")
        for r in q.iter_rows(named=True):
            lab = "幽霊(0)" if r["size_unit"] == 0 else f"${r['size_usd']:,.0f}"
            yr = r["day_usd"] * 252 if np.isfinite(r["day_usd"]) else np.nan
            print(f"    {lab:>12}{r['pairs_day']:>9,.0f}{r['ev_pair_bp']:>9.3f}"
                  f"{r['t_pair']:>7.2f}"
                  f"{r['day_usd']:>12,.0f}{r['se_day_usd']:>10,.0f}"
                  f"{r['t_usd']:>7.2f}{yr:>14,.0f}")

    # ---- 分解: 建てた瞬間の edge + 畳んだ瞬間の edge + 在庫のドリフト ----
    print("\n===== ④ 1 組あたり損益の分解 =====")
    print("損益は次の 4 つに厳密に分かれる(恒等式。残差ではない)。\n")
    print("  PnL = edge_in + edge_out + drift − 手数料 ×2")
    print("    edge_in  = side × (mid − 約定値) / mid  … 建てた**瞬間**に mid から取れた幅")
    print("    edge_out = side × (約定値 − mid) / mid  … 畳んだ瞬間に取れた幅")
    print("    drift    = side × (mid_out − mid_in) / mid … 持っている間の mid の動き")
    print("★ edge_in は**発注時**ではなく**約定時**の mid で測る。約定する前に")
    print("  mid が自分の指値を追い越していれば、その時点で既に負けている。\n")
    print("★★ edge_in の mid は**約定した ns の mid** である。板を消し切る約定が")
    print("   同じブロックで気配を動かすので、これは既に「約定直後」の mid になる。")
    print("   つまり edge_in は『出したときの半スプレッド』ではなく")
    print("   『半スプレッド − 約定までに mid が動いた分』である。")
    print("   比較のため、**発注時**の半スプレッド(quoted)も別に測る。\n")
    print(f"{'銘柄':<12}{'quoted':>8}{'edge_in':>9}{'約定前の劣化':>13}"
          f"{'edge_out':>10}{'取れた幅':>10}{'drift':>10}{'手数料':>8}"
          f"{'実現 EV':>10}{'検算':>8}")
    dec = []
    for c in COINS:
        lp = DATA / f"inv_lots_xyz_{c}_q1.parquet"
        if not lp.exists():
            continue
        L = pl.read_parquet(lp)
        if "p_out" not in L.columns:
            print(f"xyz:{c:<8}  p_out が無い(古い実行)。build_inventory を回し直す")
            continue
        sd = L["side"].cast(pl.Float64).to_numpy()
        mi = L["m_in"].to_numpy()
        pi = L["p_in"].to_numpy()
        po = L["p_out"].to_numpy()
        mo = L["m_out"].to_numpy()
        e_in = sd * (mi - pi) / mi * 1e4
        e_out = -sd * (mo - po) / mi * 1e4
        drift = sd * (mo - mi) / mi * 1e4
        # 強制決済はテイカー手数料が乗るので手数料を別に積む
        fee = np.where(L["forced"].to_numpy() > 0, 0.088 + 0.846, 2 * 0.088)
        ev = float(L["pnl"].to_numpy().mean())
        chk = float(np.mean(e_in + e_out + drift - fee)) - ev
        # 発注時の半スプレッド(quoted)。約定した「建てる側」の発注だけを見る。
        quoted = np.nan
        pp = DATA / f"inv_posts_xyz_{c}_q1.parquet"
        if pp.exists():
            hs = []
            P = (pl.read_parquet(pp)
                 .filter((pl.col("filled") > 0) & pl.col("rt_pnl").is_finite()))
            for dt, g in P.partition_by("dt", as_dict=True).items():
                dt = dt[0] if isinstance(dt, tuple) else dt
                b = clean_bbo(pl.scan_parquet(DATA / f"bbo_xyz_{c}.parquet")
                              .filter(pl.col("dt") == dt).collect())[0].sort("ts")
                if b.height < 10:
                    continue
                bt = b["ts"].cast(pl.Int64).to_numpy()
                pb_, pa_ = b["best_bid"].to_numpy(), b["best_ask"].to_numpy()
                j2 = np.clip(np.searchsorted(bt, g["t"].to_numpy(),
                                             side="right") - 1, 0, bt.size - 1)
                md = 0.5 * (pb_[j2] + pa_[j2])
                hs.append((pa_[j2] - pb_[j2]) / 2 / md * 1e4)
            if hs:
                quoted = float(np.concatenate(hs).mean())
        dec.append({"coin": f"xyz:{c}", "quoted_half_bp": quoted,
                    "edge_in_bp": float(e_in.mean()),
                    "predecay_bp": quoted - float(e_in.mean()),
                    "edge_out_bp": float(e_out.mean()),
                    "edge_bp": float((e_in + e_out).mean()),
                    "drift_bp": float(drift.mean()),
                    "fee_bp": float(fee.mean()), "ev_bp": ev,
                    "resid_bp": chk, "n_pair": L.height})
        print(f"xyz:{c:<8}{quoted:>8.3f}{e_in.mean():>9.3f}"
              f"{quoted - e_in.mean():>13.3f}{e_out.mean():>10.3f}"
              f"{(e_in+e_out).mean():>10.3f}{drift.mean():>10.3f}"
              f"{fee.mean():>8.3f}{ev:>10.3f}{chk:>8.4f}")
    if dec:
        pl.DataFrame(dec).write_csv(DATA / "mm_all_decomp.csv")

    print("\n===== ⑤ Bonferroni を通った設定(EV>0 側)=====")
    win = S.filter((pl.col("t_pair") > thr) & (pl.col("ev_pair_bp") > 0))
    lose = S.filter((pl.col("t_pair") < -thr) & (pl.col("ev_pair_bp") < 0))
    print(f"正で有意 {win.height} 通り / 負で有意 {lose.height} 通り / "
          f"どちらでもない {nt - win.height - lose.height} 通り")
    with pl.Config(tbl_rows=40, tbl_width_chars=170, float_precision=3):
        print(win.sort("ev_pair_bp", descending=True)
              .select("coin", "size_usd", "lat_ms", "pairs_day", "ev_pair_bp",
                      "t_pair", "day_usd", "t_usd"))


if __name__ == "__main__":
    main()
