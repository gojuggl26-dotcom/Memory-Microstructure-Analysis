"""往復(entry + exit)の損益を 3 通りの手仕舞い方で計算する。

    uv run python scripts/build_roundtrip.py --coin xyz:MU

これまでの `label_pnl*` は **mid の markout** であって往復ではない。
建玉を閉じる費用が入っていないので、島が本当に採算に乗るかは分からなかった。
ここでは実際に閉じる。

手仕舞いの 3 通り
-----------------
1. **Taker exit**  約定 tau の 1 秒後に反対側の気配を叩く。最も厳しい基準。
2. **Passive exit** tau に反対側の最良気配へメイカーを出し、T_max まで待つ。
   T_max までに約定しなければ**建玉が残る**(mid で評価し、別に数える)。
3. **Hybrid**     Passive を試し、T_max を超えたらテイカーで強制的に閉じる。

    PnL_roundtrip = (Exit - Entry) * side - Fee - Funding

費用(既報 `DRAM/hft_roadmap.md` の実測)
----------------------------------------
    メイカー 0.088 bp / テイカー 0.846 bp
    (取引所 0.543 + deployer 0.227 + builder 0.076)

Funding は Hyperliquid の金利成分 0.01%/8h = 0.125 bp/h を保有時間だけ按分する。
保有が 10〜60 秒なので 0.0003〜0.002 bp にしかならず、**この時間尺度では無視できる**
(MU の funding 実データは WORK_BUCKET に無いので、既定値としてこの率を使う)。

時間契約
--------
- 手仕舞いの発注は **tau 時点で観測できる板**(tau 以前の最後の bbo)で決める
- 手仕舞いの約定判定は entry と同じ待ち行列規則(反対側の最良気配の後ろに並ぶ)
- 強制手仕舞いの値段は tau+T_max 以前の最後の bbo

出力
----
`E:/Memory-quotes/<tag>_rt/dt=*.parquet`。本体と**同じ行順・同じ行数**で、
entry が 1 秒以内に約定した行だけ値が入り、それ以外は null。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import PX_UNIT, clean_bbo  # noqa: E402
from build_quotes import DAY_NS, NG, PRE_NS, fill_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
MAKER_FEE = 0.088
TAKER_FEE = 0.846
FUND_BP_H = 0.125          # 金利成分 0.01%/8h。保有 60 秒で 0.002 bp
TMAX = [10.0, 60.0]        # passive で待つ上限 (秒)
TAKER_H = 1.0              # taker exit までの保有 (秒)
ENTRY_MAX = 1.0            # entry の約定がこの秒数以内の行だけ計算する
SUB = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    outd = BULK / f"{tag}_rt"
    outd.mkdir(parents=True, exist_ok=True)
    subd = DATA / f"quotes_rt_sub_{tag}"
    subd.mkdir(parents=True, exist_ok=True)
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    bpath = DATA / f"bbo_{tag}.parquet"
    fpath = DATA / f"fills_{tag}.parquet"
    print(f"{len(files)} 日 -> {outd}", flush=True)

    for k, f in enumerate(files):
        dt = f.stem.split("=")[1]
        if (outd / f"dt={dt}.parquet").exists():
            continue
        T = pl.read_parquet(f, columns=["ts", "side", "quote_px",
                                        "label_fill_lat_s"])
        d = clean_bbo(pl.scan_parquet(bpath).filter(pl.col("dt") == dt)
                      .collect())[0].sort("ts")
        tb = d["ts"].cast(pl.Int64).to_numpy()
        pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
        qb, qa = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
        midb = 0.5 * (pb + pa)
        nb = tb.size
        d0 = int(tb[0]) // DAY_NS * DAY_NS
        # 秒格子の気配 (fill_times の A 段が使う門)
        tg = d0 + np.arange(NG, dtype=np.int64) * 10 ** 9
        gj = np.clip(np.searchsorted(tb, tg, side="right") - 1, 0, nb - 1)
        btg, atg = np.round(pb[gj] / PX_UNIT), np.round(pa[gj] / PX_UNIT)

        F = pl.scan_parquet(fpath).filter(
            pl.col("crossed") & (pl.col("dt") == dt)).select(
            "ts", "px", "sz", "side").collect().sort("ts")
        ft = F["ts"].cast(pl.Int64).to_numpy()
        fpx = np.round(F["px"].to_numpy() / PX_UNIT)
        fsz = F["sz"].to_numpy()
        fbuy = F["side"].to_numpy() == "B"

        ts = T["ts"].cast(pl.Int64).to_numpy()
        sd_all = T["side"].to_numpy().astype(np.int64)
        qpx = T["quote_px"].to_numpy().astype(np.float64)
        lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
        n_all = ts.size
        sel = np.isfinite(lat) & (lat <= ENTRY_MAX)
        idx = np.flatnonzero(sel)
        out = {c: np.full(n_all, np.nan, np.float64) for c in
               ["rt_taker_1s_bp"]
               + [f"rt_pass_lat_{int(x)}s" for x in TMAX]
               + [f"rt_pass_{int(x)}s_bp" for x in TMAX]
               + [f"rt_hyb_{int(x)}s_bp" for x in TMAX]
               + [f"rt_hold_{int(x)}s" for x in TMAX]}

        if idx.size:
            tau = (ts[idx] + lat[idx] * 1e9).astype(np.int64)
            sd = sd_all[idx]
            ent = qpx[idx]
            jt = np.clip(np.searchsorted(tb, tau, side="right") - 1, 0, nb - 1)
            ip = np.clip(np.searchsorted(tb, tau - PRE_NS, side="right") - 1,
                         0, nb - 1)
            ref = midb[ip]                       # bp の分母は約定直前の mid
            # --- 1. Taker exit: tau+1s に反対側を叩く ---
            jx = np.clip(np.searchsorted(tb, tau + int(TAKER_H * 1e9),
                                         side="right") - 1, 0, nb - 1)
            okx = (tau + int(TAKER_H * 1e9)) <= d0 + DAY_NS
            xpx = np.where(sd > 0, pb[jx], pa[jx])
            fund = FUND_BP_H * TAKER_H / 3600.0
            out["rt_taker_1s_bp"][idx] = np.where(
                okx, sd * (xpx - ent) / ref * 1e4 - MAKER_FEE - TAKER_FEE - fund,
                np.nan)

            # --- 2/3. Passive / Hybrid ---
            # tau 時点の反対側の最良気配へ出す。前に並ぶのはその時の数量
            xp = np.where(sd > 0, pa[jt], pb[jt])
            q0 = np.where(sd > 0, qa[jt], qb[jt])
            esd = -sd                            # 手仕舞いは反対の売買
            thr = np.round(xp / PX_UNIT)
            # 売り手仕舞い(esd=-1)を約定させるのは買いテイカー、逆も同様
            for T_ in TMAX:
                tau2 = np.full(idx.size, -1, np.int64)
                for s_ in (1, -1):
                    m = esd == s_
                    if not m.any():
                        continue
                    tau2[m] = fill_times(
                        tau[m], thr[m], q0[m], s_,
                        btg if s_ == 1 else atg, ft, fpx, fsz,
                        (~fbuy) if s_ == 1 else fbuy, d0)
                hit = (tau2 >= 0) & ((tau2 - tau) <= int(T_ * 1e9))
                hold = np.where(hit, (tau2 - tau) / 1e9, T_)
                out[f"rt_pass_lat_{int(T_)}s"][idx] = np.where(
                    hit, (tau2 - tau) / 1e9, np.nan)
                out[f"rt_hold_{int(T_)}s"][idx] = hold
                fund = FUND_BP_H * hold / 3600.0
                # passive で約定した場合の損益 (両脚ともメイカー)
                gp = sd * (xp - ent) / ref * 1e4 - 2 * MAKER_FEE - fund
                out[f"rt_pass_{int(T_)}s_bp"][idx] = np.where(hit, gp, np.nan)
                # hybrid: 約定しなければ tau+T_max に叩く
                jf = np.clip(np.searchsorted(tb, tau + int(T_ * 1e9),
                                             side="right") - 1, 0, nb - 1)
                okf = (tau + int(T_ * 1e9)) <= d0 + DAY_NS
                fpx2 = np.where(sd > 0, pb[jf], pa[jf])
                gf = sd * (fpx2 - ent) / ref * 1e4 - MAKER_FEE - TAKER_FEE - fund
                out[f"rt_hyb_{int(T_)}s_bp"][idx] = np.where(
                    hit, gp, np.where(okf, gf, np.nan))

        O = pl.DataFrame({c: v.astype(np.float32) for c, v in out.items()})
        O = O.with_columns([pl.when(pl.col(c).is_nan()).then(None)
                            .otherwise(pl.col(c)).alias(c) for c in O.columns])
        O.write_parquet(outd / f"dt={dt}.parquet", compression="zstd")
        O[::SUB].write_parquet(subd / f"dt={dt}.parquet", compression="zstd")
        if (k + 1) % 10 == 0 or k == 0:
            e = O["rt_pass_lat_10s"].drop_nulls().len()
            print(f"  [{k+1}/{len(files)}] {dt}  entry {idx.size:,} "
                  f"passive 10s 約定 {e:,} ({100*e/max(idx.size,1):.1f}%)",
                  flush=True)
        del T, d, F, O
    print(f"書き出し {outd} / 間引き版 {subd}")


if __name__ == "__main__":
    main()
