"""在庫を「消しやすさ」と「消す費用」で説明する(項目 3)。

    uv run python scripts/build_unwind.py --coin xyz:MU

`build_inventory.py` が出した建玉 1 本ずつの記録に、**約定した瞬間 τ の板の状態**を
貼り、在庫の向きに揃えた指標

    OBI^inv = sign(q)·OBI      OFI^inv = sign(q)·OFI

で、将来リターンではなく**在庫を安く消せるか**を直接予測できるかを見る。

ラベル
------
    Y_exit_h = 1[反対側のメイカーが h 以内に約定した]
    T_exit   = 相殺までの秒数
    C_unwind_h = h まで待ってから平らにするまでの総費用 (bp)
               = −(h 以内にメイカーで相殺できたときの損益)
                 h を超えたら τ+h の気配をテイカーで叩く

時間契約: 説明変数は τ 時点(= 約定した瞬間)までの情報だけ。
ラベルだけが τ 以降を使う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_quotes import DAY_NS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAKER_FEE, TAKER_FEE = 0.088, 0.846
HS = [0.25, 1.0, 5.0, 10.0, 30.0, 60.0]
OFIW = [0.1, 0.25, 0.5, 1.0]
NB = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    ap.add_argument("--sfx", default="_q1")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    L = pl.read_parquet(DATA / f"inv_lots_{tag}{a.sfx}.parquet")
    print(f"建玉 {L.height:,} 本", flush=True)
    out = []
    for dt, G in sorted(L.partition_by("dt", as_dict=True).items()):
        dt = dt[0] if isinstance(dt, tuple) else dt
        d = clean_bbo(pl.scan_parquet(DATA / f"bbo_{tag}.parquet")
                      .filter(pl.col("dt") == dt).collect())[0].sort("ts")
        ts = d["ts"].cast(pl.Int64).to_numpy()
        pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
        qb, qa = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
        mid = 0.5 * (pb + pa)
        nb = ts.size
        d0 = int(ts[0]) // DAY_NS * DAY_NS
        e = np.concatenate([[0.0], (
            (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
            - ((pa[1:] <= pa[:-1]) * qa[1:] - (pa[1:] >= pa[:-1]) * qa[:-1]))])
        cofi = np.cumsum(e)
        F = pl.scan_parquet(DATA / f"fills_{tag}.parquet").filter(
            pl.col("crossed") & (pl.col("dt") == dt)).select(
            "ts", "sz", "side").collect().sort("ts")
        ft = F["ts"].cast(pl.Int64).to_numpy()
        cnet = np.concatenate([[0.0], np.cumsum(
            F["sz"].to_numpy() * np.where(F["side"].to_numpy() == "B", 1.0, -1.0))])

        t = G["t_in"].to_numpy()
        sd = G["side"].to_numpy().astype(np.float64)
        p_in = G["p_in"].to_numpy()
        m_in = G["m_in"].to_numpy()
        toff = G["t_off"].to_numpy().astype(np.float64)
        pnl = G["pnl"].to_numpy().astype(np.float64)
        j = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, nb - 1)
        sb = qb[j] + qa[j]
        obi = np.where(sb > 0, (qb[j] - qa[j]) / np.maximum(sb, 1e-12), 0.0)
        R = {"dt": [dt] * t.size, "t_in": t, "side": sd.astype(np.int8),
             "t_off": toff, "pnl": pnl,
             "obi_inv": sd * obi,
             "spread_bp": (pa[j] - pb[j]) / mid[j] * 1e4,
             "depth_own": np.where(sd > 0, qb[j], qa[j]),
             "depth_opp": np.where(sd > 0, qa[j], qb[j]),
             "bbo_age_s": (t - ts[np.maximum.accumulate(
                 np.where(np.concatenate([[True], (pb[1:] != pb[:-1])
                                          | (pa[1:] != pa[:-1])]),
                          np.arange(nb), -1))[j]]) / 1e9}
        for w in OFIW:
            k = np.clip(np.searchsorted(ts, t - int(w * 1e9), side="right") - 1,
                        0, nb - 1)
            R[f"ofi_inv_{w:g}s"] = sd * (cofi[j] - cofi[k])
        for w in (1.0, 10.0):
            lo = np.searchsorted(ft, t - int(w * 1e9), side="right")
            hi = np.searchsorted(ft, t, side="right")
            R[f"taker_inv_{w:g}s"] = sd * (cnet[hi] - cnet[lo])
        # ラベル: h までに平らにする費用
        for h in HS:
            th = t + int(h * 1e9)
            jh = np.clip(np.searchsorted(ts, th, side="right") - 1, 0, nb - 1)
            xp = np.where(sd > 0, pb[jh], pa[jh])
            forced = (xp - p_in) * sd / m_in * 1e4 - MAKER_FEE - TAKER_FEE
            hit = toff <= h
            R[f"y_exit_{h:g}s"] = hit.astype(np.int8)
            R[f"c_unwind_{h:g}s"] = -np.where(hit, pnl, forced)
        out.append(pl.DataFrame(R))
    U = pl.concat(out)
    U.write_parquet(DATA / f"unwind_{tag}{a.sfx}.parquet")

    o = U["obi_inv"].to_numpy()
    f1 = U["ofi_inv_1s"].to_numpy()
    eo = np.quantile(o, np.linspace(0, 1, NB + 1)[1:-1])
    ef = np.quantile(f1, np.linspace(0, 1, NB + 1)[1:-1])
    io = np.searchsorted(eo, o, side="right")
    iff = np.searchsorted(ef, f1, side="right")
    cell = io * NB + iff
    rows = []
    for c in range(NB * NB):
        m = cell == c
        if m.sum() < 200:
            continue
        r = {"obi_d": c // NB + 1, "ofi_d": c % NB + 1, "n": int(m.sum())}
        for h in (1.0, 10.0):
            r[f"p_exit_{h:g}s"] = float(U[f"y_exit_{h:g}s"].to_numpy()[m].mean())
            r[f"c_unwind_{h:g}s"] = float(U[f"c_unwind_{h:g}s"].to_numpy()[m].mean())
        r["t_exit_med"] = float(np.median(U["t_off"].to_numpy()[m]))
        r["pnl"] = float(U["pnl"].to_numpy()[m].mean())
        rows.append(r)
    C = pl.DataFrame(rows)
    C.write_csv(DATA / f"unwind_cells_{tag}{a.sfx}.csv")
    print(f"\nセル {C.height}/100")
    for c in ("p_exit_1s", "c_unwind_1s", "c_unwind_10s", "pnl"):
        v = C[c].to_numpy()
        print(f"  {c:14s} 最小 {v.min():+.4f}  最大 {v.max():+.4f}  "
              f"幅 {v.max()-v.min():.4f}")
    print(f"\n書き出し {DATA}/unwind_cells_{tag}{a.sfx}.csv")


if __name__ == "__main__":
    main()
