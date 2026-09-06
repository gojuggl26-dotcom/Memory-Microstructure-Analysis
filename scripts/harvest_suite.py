"""スイートの出力から、レポートに載せる見出しの数字を銘柄ごとに集める。

    uv run python scripts/harvest_suite.py --coins xyz:AMD,xyz:KIOXIA,...

出力: data/suite_headline_<tag>.json(銘柄ごと)

レポートを書くとき、数字は必ずここから引く(手で桁を写さない)。
x/y の時間契約は各 build_* スクリプトのものをそのまま引き継ぐ。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def f2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def one(coin: str) -> dict:
    tag = coin.replace(":", "_")
    out: dict = {"coin": coin}

    # ── 素性(fills) ────────────────────────────────────────────────
    p = pl.read_csv(DATA / f"profile_xyz_{tag.split('_')[-1]}.csv") \
        if (DATA / f"profile_xyz_{tag.split('_')[-1]}.csv").exists() else None
    p = pl.read_csv(DATA / f"profile_{tag}.csv") if (DATA / f"profile_{tag}.csv").exists() else p
    if p is not None:
        out["days"] = p.height
        out["trades_day_med"] = f2(p["n_trade"].median())
        out["usd_day_med"] = f2(p["notional"].median())
        out["px_med"] = f2(p["px_med"].median())
        out["sz_med"] = f2(p["sz_med"].median())
        out["takers_day_med"] = f2(p["n_taker"].median())
        out["liq_day_med"] = f2(p["n_liq"].median())

    # ── 建玉(oi_volume) ─────────────────────────────────────────────
    f = DATA / f"daily_oi_volume_{tag}.parquet"
    if f.exists():
        d = pl.read_parquet(f)
        out["oi_usd_med"] = f2(d["oi_mean_usd"].median())
        out["oi_users_day_med"] = f2(d["n_users"].median())

    # ── 買い売りの偏り(volume_side) ────────────────────────────────
    f = DATA / f"daily_volume_side_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        out["buy_share_mean"] = f2(d["buy_share"].mean())
        out["days_absz_gt2"] = int((d["z_hac"].abs() > 2).sum())

    # ── 注文サイズ ─────────────────────────────────────────────────
    f = DATA / f"order_size_stats_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f).filter(pl.col("day_type") == "全体") \
            if "day_type" in pl.read_csv(f).columns else pl.read_csv(f)
        try:
            allw = d.filter(pl.col("window") == "全体")
            r = allw.row(0, named=True) if allw.height else d.row(0, named=True)
            out["osz_med"] = f2(r.get("median"))
            out["osz_q99"] = f2(r.get("q99"))
            out["osz_share_ge_10"] = f2(r.get("share_ge_10"))
        except Exception:
            pass

    # ── microprice の端(帯 × k のうち |p_up − base| 最大セル) ─────
    f = DATA / f"microprice_cells_{tag}.parquet"
    if f.exists():
        d = pl.read_parquet(f).filter(pl.col("n") >= 300)
        if d.height:
            d = d.with_columns(edge=(pl.col("p_up_move") - pl.col("base_up_move")).abs())
            b = d.sort("edge", descending=True).row(0, named=True)
            out["micro_edge_max"] = f2(b["edge"])
            out["micro_edge_cell"] = f"{b['day_type']}/{b['bin']}/k={b['k']}"
            out["micro_edge_placebo"] = f2(abs(b.get("p_up_placebo", np.nan)
                                               - b.get("base_up_move", np.nan))
                                           if b.get("p_up_placebo") is not None else None)

    # ── OBI / OFI の端 ─────────────────────────────────────────────
    f = DATA / f"obi_ofi_cells_{tag}.parquet"
    if f.exists():
        d = pl.read_parquet(f).filter(pl.col("n") >= 300)
        for feat in ("obi", "ofi"):
            s = d.filter(pl.col("feat") == feat)
            if s.height:
                s = s.with_columns(e=(pl.col("p_up_move") - pl.col("base_up_move")).abs())
                b = s.sort("e", descending=True).row(0, named=True)
                out[f"{feat}_edge_max"] = f2(b["e"])
                out[f"{feat}_edge_cell"] = f"{b['day_type']}/{b['bin']}/k={b['k']}"

    # ── book slope(1 イベント先の HAC t)───────────────────────────
    f = DATA / f"book_slope_fits_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        s = d.filter((pl.col("day_type") == "立会日") & (pl.col("k") == d["k"].min()))
        if s.height:
            r = s.row(0, named=True)
            out["slope_beta"] = f2(r.get("beta_ols"))
            out["slope_t_hac"] = f2(r.get("t_hac"))

    # ── cancel rate(最良ホライズンの diff)─────────────────────────
    f = DATA / f"cancel_rate_cells_{tag}.parquet"
    if f.exists():
        d = pl.read_parquet(f)
        if d.height:
            b = d.with_columns(a=pl.col("diff_bp").abs()).sort("a", descending=True) \
                 .row(0, named=True)
            out["cancel_diff_bp"] = f2(b["diff_bp"])
            out["cancel_cell"] = f"{b['day_type']}/h={b['h_ms']}ms"
            out["cancel_placebo_bp"] = f2(b.get("placebo_diff_bp"))

    # ── 符号の持続(OBI, k=1)────────────────────────────────────────
    f = DATA / f"sign_persist_cells_{tag}.parquet"
    if f.exists():
        d = pl.read_parquet(f)
        s = d.filter((pl.col("feat").str.contains("obi") | pl.col("feat").str.contains("OBI"))
                     & (pl.col("k") == 1))
        if s.height:
            r = s.sort("n_pairs", descending=True).row(0, named=True)
            out["obi_persist_excess_k1"] = f2(r["excess"])

    # ── ACF 200ms(OBI と OFI の lag1)──────────────────────────────
    f = DATA / f"acf_200ms_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        for v in d["var"].unique().to_list():
            s = d.filter((pl.col("var") == v) & (pl.col("lag") == 1))
            if s.height:
                out[f"acf1_{v}"] = f2(s["rho"].mean())

    # ── var100(絶対リターンへの最良 r)──────────────────────────────
    f = DATA / f"var100_ols_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        s = d.filter(pl.col("y_kind").str.contains("abs")) if "y_kind" in d.columns else d
        if s.height:
            b = s.with_columns(a=pl.col("r").abs()).sort("a", descending=True).row(0, named=True)
            out["var100_best_r"] = f2(b["r"])
            out["var100_best"] = f"{b.get('feat')}/{b.get('hor')}"

    # ── spread_flow(スプレッド×約定量の相関)────────────────────────
    f = DATA / f"spread_flow_corr_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        s = d.filter(pl.col("var") == "fillsz")
        if s.height:
            out["corr_spread_fillsz"] = f2(s["spread_bp"][0])
            out["corr_qsize_fillsz"] = f2(s["qsize"][0])

    # ── vol noise(実効スプレッドとノイズ)───────────────────────────
    f = DATA / f"vol_noise_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f)
        out["noise_omega_bp_med"] = f2(d["omega_bp"].median())
        out["half_spread_bp_med"] = f2(d["half_spread_bp"].median())

    # ── xfeat 予測力 ────────────────────────────────────────────────
    f = DATA / f"xfeat_pred_{tag}.csv"
    if f.exists():
        d = pl.read_csv(f).with_columns(ab=pl.col("r_fwd").abs())
        for h in ("100ms", "1s", "10s", "60s"):
            s = d.filter(pl.col("h") == h)
            if s.height:
                b = s.sort("ab", descending=True).row(0, named=True)
                out[f"xf_{h}_max"] = f2(b["ab"])
                out[f"xf_{h}_col"] = b["col"]
                out[f"xf_{h}_sign"] = "+" if b["r_fwd"] > 0 else "-"
                out[f"xf_{h}_plc"] = f2(s["r_plc"].abs().max())
        s = d.filter((pl.col("h") == "1s") & (pl.col("col") == "delta_bp"))
        if s.height:
            out["delta1s"] = f2(s["r_fwd"][0])
            out["delta1s_mid"] = f2(s["r_mid"][0])
        # 前半後半
        s = d.filter(pl.col("h") == "1s")
        r1 = s["r_1st"].to_numpy(); r2 = s["r_2nd"].to_numpy()
        m = np.isfinite(r1) & np.isfinite(r2)
        if m.sum() > 10:
            out["oos_corr_1s"] = f2(np.corrcoef(r1[m], r2[m])[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", required=True)
    a = ap.parse_args()
    for coin in a.coins.split(","):
        tag = coin.replace(":", "_")
        d = one(coin)
        fp = DATA / f"suite_headline_{tag}.json"
        fp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{coin}: {len(d)} 項目 -> {fp.name}", flush=True)


if __name__ == "__main__":
    main()
