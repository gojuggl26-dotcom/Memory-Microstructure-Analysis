"""microprice 基準の markout ラベルを候補テーブルへ追加する。

    uv run python scripts/build_mu_labels.py --coin xyz:MU

現状のラベルは mid 基準:

    PnL   = Edge - Fee + s*(mid_{tau+h} - mid_pre)

これに microprice 基準を並べる:

    PnL^mu = Edge - Fee + s*(mu_{tau+h} - mu_pre)

    mu = mid + (spread/2) * (bid_sz - ask_sz)/(bid_sz + ask_sz)

mid が BBO の不均衡を遅れて反映しているだけなら、mid 基準の markout は
「これから起きることを先取りしている」ように見えてしまう。microprice で
消えるならその可能性が高い、という判定に使う。

時間契約
--------
- 約定時刻 tau は既存の `label_fill_lat_s`(t からの秒数)から復元する
- 基準時点は **tau - 2ms**(tau ちょうどの板は既に約定後なので使えない。
  `mu_ev_report.md` で符号が反転した事故の再発防止)
- 参照する板はすべて tau+h 以前の最後の bbo(後ろ向き asof)
- 日をまたぐ窓は null

出力
----
`E:/Memory-quotes/<tag>_mu/dt=*.parquet` に、本体と**同じ行順**で新しい列だけを書く。
本体 6.1GB を作り直さずに済ませるため。列は横に連結して使う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402
from build_quotes import DAY_NS, FEE_BP, MK_H, PRE_NS, SUB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BULK = Path("E:/Memory-quotes")
NEED = ["ts", "side", "mid", "quote_px", "label_fill_lat_s", "label_edge_bp",
        "label_markout_1s_bp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")
    outd = BULK / f"{tag}_mu"
    outd.mkdir(parents=True, exist_ok=True)
    subd = DATA / f"quotes_mu_sub_{tag}"
    subd.mkdir(parents=True, exist_ok=True)
    files = sorted((BULK / tag).glob("dt=*.parquet"))
    bpath = DATA / f"bbo_{tag}.parquet"
    print(f"{len(files)} 日 -> {outd}", flush=True)

    chk = []
    for k, f in enumerate(files):
        dt = f.stem.split("=")[1]
        if (outd / f"dt={dt}.parquet").exists():
            continue
        T = pl.read_parquet(f, columns=NEED)
        d = clean_bbo(pl.scan_parquet(bpath).filter(pl.col("dt") == dt)
                      .collect())[0].sort("ts")
        tb = d["ts"].cast(pl.Int64).to_numpy()
        pb, pa = d["best_bid"].to_numpy(), d["best_ask"].to_numpy()
        qb, qa = d["bid_sz"].to_numpy(), d["ask_sz"].to_numpy()
        midb = 0.5 * (pb + pa)
        s = qb + qa
        obi = np.where(s > 0, (qb - qa) / np.maximum(s, 1e-12), 0.0)
        mub = midb + 0.5 * (pa - pb) * obi            # microprice
        nb = tb.size
        d0 = int(tb[0]) // DAY_NS * DAY_NS

        ts = T["ts"].cast(pl.Int64).to_numpy()
        sd = T["side"].to_numpy().astype(np.float64)
        lat = T["label_fill_lat_s"].to_numpy().astype(np.float64)
        edge = T["label_edge_bp"].to_numpy().astype(np.float64)
        has = np.isfinite(lat)
        tau = np.where(has, ts + (lat * 1e9), ts).astype(np.int64)

        ip = np.clip(np.searchsorted(tb, tau - PRE_NS, side="right") - 1, 0, nb - 1)
        mu_pre, mid_pre = mub[ip], midb[ip]
        F = {}
        # microprice で測った発注の優位(mid 基準の label_edge_bp と対で見る)
        F["label_edgemu_bp"] = np.where(has, sd * (mu_pre - T["quote_px"].to_numpy())
                                        / mid_pre * 1e4, np.nan)
        for h in MK_H:
            te = tau + int(h * 1e9)
            ih = np.clip(np.searchsorted(tb, te, side="right") - 1, 0, nb - 1)
            ok = has & (te <= d0 + DAY_NS)
            F[f"label_mkmu_{h:g}s_bp"] = np.where(
                ok, sd * np.log(mub[ih] / mu_pre) * 1e4, np.nan)
            F[f"label_pnlmu_{h:g}s_bp"] = edge - FEE_BP + F[f"label_mkmu_{h:g}s_bp"]
            # 本体と同じ規則で mid 版も作り直し、既存列と一致するかを検算する
            if h == 1.0:
                mk_mid = np.where(ok, sd * np.log(midb[ih] / mid_pre) * 1e4, np.nan)
        old = T["label_markout_1s_bp"].to_numpy().astype(np.float64)
        m = np.isfinite(old) & np.isfinite(mk_mid)
        dmax = float(np.max(np.abs(old[m] - mk_mid[m]))) if m.any() else 0.0
        chk.append((dt, dmax, float(m.mean())))

        O = pl.DataFrame({c: v.astype(np.float32) for c, v in F.items()})
        O.write_parquet(outd / f"dt={dt}.parquet", compression="zstd")
        O[::SUB].write_parquet(subd / f"dt={dt}.parquet", compression="zstd")
        del T, d, O
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(files)}] {dt}  mid 版の再計算との最大差 "
                  f"{dmax:.3g} bp", flush=True)

    if chk:
        dm = np.array([c[1] for c in chk])
        i = int(np.argmax(dm))
        print(f"\n検算: 既存の label_markout_1s_bp を同じ規則で作り直した差の"
              f"最大は {dm.max():.4g} bp ({chk[i][0]})  中央 {np.median(dm):.4g}")
    print(f"書き出し {outd} / 間引き版 {subd}")


if __name__ == "__main__":
    main()
