"""1 秒足の 6 レジーム判定(ボラ 2 × 方向 3)。

    uv run python scripts/build_regime.py --coin xyz:MU

出力: data/regime_<coin>.parquet
      (ts, dt, sec, mid, r, rv10, q55, q70, vol_hi, er, ds, dirn, regime, stale_s)

指示された定義をそのまま実装している。

## 1. 入力

各秒 $`t`$ の mid を $`M_t=(\\mathrm{Bid}_t+\\mathrm{Ask}_t)/2`$、
$`p_t=\\log M_t`$、$`r_t=p_t-p_{t-1}`$ とする。
1 秒格子の値は**その秒までに観測された最後の気配**(後ろ向き)なので、
気配が動かなかった秒は $`r_t=0`$ になる。24 時間動く市場なので日を跨いで
連続させる(日の境目で切らない)。

## 2. ボラティリティ判定(ヒステリシスつき)

```math
RV_{10,t}=\\sqrt{\\textstyle\\sum_{i=0}^{9}r_{t-i}^{2}}
```

固定閾値ではなく**直近 30 分(1,800 秒)の $`RV_{10}`$ 分布**と比べる。

```math
Q_{55,t}=\\mathrm{pct}_{55}(RV_{10,\\,t-1800:t}),\\qquad
Q_{70,t}=\\mathrm{pct}_{70}(RV_{10,\\,t-1800:t})
```

$`RV_{10,t}>Q_{70,t}`$ で High へ、$`RV_{10,t}<Q_{55,t}`$ で Low へ移り、
その間は**前秒の状態を維持**する。1 秒ごとのチャタリングを防ぐため。

実装は「明示的な遷移信号を前方補完する」ことで逐次ループを使わずに書ける。
$`s_t=+1`$(High へ)/ $`-1`$(Low へ)/ $`0`$(維持)を作り、
0 を欠測にして forward fill すれば、定義どおり**最後に出た信号**が残る。

## 3. 方向判定

単純な 10 秒リターンだけでは、100→101→102→103 と 100→104→98→103 を
区別できない(どちらも +3)。そこで 2 つ使う。

```math
ER_t=\\frac{p_t-p_{t-10}}{\\sum_{i=0}^{9}|r_{t-i}|+\\epsilon},\\qquad
DS_t=\\frac{p_t-p_{t-10}}{\\sqrt{\\sum_{i=0}^{9}r_{t-i}^{2}}+\\epsilon}
```

$`-1\\le ER_t\\le 1`$。$`|DS_t|\\le\\sqrt{10}\\approx3.16`$(Cauchy–Schwarz)。
$`p_t-p_{t-10}=\\sum_{i=0}^{9}r_{t-i}`$ なので、3 つとも同じ 10 点窓で作る。

**Up** は $`ER_t\\ge0.35`$ **かつ** $`DS_t\\ge1.0`$。
**Down** は $`ER_t\\le-0.35`$ **かつ** $`DS_t\\le-1.0`$。それ以外は Neutral。
AND にするのは、一瞬の大きな動きで ER だけ上がった場面を弾くためである。

## 4. 6 レジーム

| | Neutral | Up | Down |
|---|---|---|---|
| Low | R1 | R2 | R3 |
| High | R6 | R4 | R5 |

## 立ち上がり

$`Q_{55}/Q_{70}`$ は 1,800 点そろうまで定義されない。
最初の 1,810 秒は `regime = 0`(判定不能)にする。

## 時間契約

$`RV_{10,t}`$・$`Q_{\\cdot,t}`$・$`ER_t`$・$`DS_t`$ はすべて
**時刻 $`t`$ までの値だけ**で決まる。先読みは無い。
将来リターンとの突き合わせは `analyze_regime.py` が行う。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_obi_levels import clean_bbo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EPS = 1e-12
WIN = 10            # 方向・RV の窓(秒)
QWIN = 1800         # 分位の窓(秒)= 30 分
Q_LO, Q_HI = 0.55, 0.70
TH_ER, TH_DS = 0.35, 1.0


def rsum(x: np.ndarray, w: int) -> np.ndarray:
    """後ろ向き w 点の移動和(w 点そろわない先頭は NaN)。"""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size) + 1
    out = c[i] - c[np.maximum(i - w, 0)]
    out[: w - 1] = np.nan
    return out


def sec_grid(coin: str, tag: str):
    """1 秒格子の mid(後ろ向き)と、最後の更新からの経過秒。"""
    if coin == "xyz:DRAM":
        t = (pl.scan_parquet(DATA / "DRAM" / "microprice_1s.parquet")
             .select("ts", "mid", "stale_ns", "dt").collect().sort("ts"))
        ts = t["ts"].cast(pl.Int64).to_numpy() // 1_000_000_000
        return ts, t["mid"].to_numpy(), t["stale_ns"].to_numpy() / 1e9
    b = clean_bbo(pl.read_parquet(DATA / f"bbo_{tag}.parquet"))[0].sort("ts")
    bt = b["ts"].cast(pl.Int64).to_numpy()
    mid = 0.5 * (b["best_bid"].to_numpy() + b["best_ask"].to_numpy())
    s0, s1 = int(bt[0]) // 10 ** 9, int(bt[-1]) // 10 ** 9
    ts = np.arange(s0, s1 + 1, dtype=np.int64)
    j = np.searchsorted(bt, (ts + 1) * 10 ** 9, side="left") - 1   # その秒の終わり
    ok = j >= 0
    j = np.maximum(j, 0)
    stale = ((ts + 1) * 10 ** 9 - bt[j]) / 1e9
    return ts, np.where(ok, mid[j], np.nan), np.where(ok, stale, np.nan)


def classify(r: np.ndarray, ok: np.ndarray | None = None) -> dict:
    """1 秒リターン列から 6 レジームを出す。帰無対照もこの関数を通す。

    `r` は 1 秒対数リターン。`ok` は mid が有効な秒(None なら全部有効)。
    """
    n = r.size
    rv10 = np.sqrt(np.maximum(rsum(r * r, WIN), 0.0))
    S = pl.Series(rv10)
    q55 = S.rolling_quantile(quantile=Q_LO, window_size=QWIN,
                             min_samples=QWIN, interpolation="linear").to_numpy()
    q70 = S.rolling_quantile(quantile=Q_HI, window_size=QWIN,
                             min_samples=QWIN, interpolation="linear").to_numpy()
    sig = np.where(rv10 > q70, 1, np.where(rv10 < q55, -1, 0)).astype(np.int8)
    sig = np.where(np.isfinite(q70) & np.isfinite(q55), sig, 0)
    # 0(= 維持)を飛ばして「最後に出た信号」を引き継ぐ。
    # 直前の非ゼロ信号の添字を累積最大で持ってくれば逐次ループが要らない。
    idx = np.maximum.accumulate(np.where(sig != 0, np.arange(n), 0))
    seen = np.maximum.accumulate((sig != 0).astype(np.int8))
    vol = np.where(seen > 0, sig[idx], -1)          # 信号が出るまでは Low
    vol_hi = (vol > 0).astype(np.int8)

    net = rsum(r, WIN)                      # = p_t − p_{t−10}
    absr = rsum(np.abs(r), WIN)
    er = net / (absr + EPS)
    ds = net / (rv10 + EPS)
    dirn = np.where((er >= TH_ER) & (ds >= TH_DS), 1,
                    np.where((er <= -TH_ER) & (ds <= -TH_DS), -1, 0)).astype(np.int8)

    reg = np.where(vol_hi == 0,
                   np.where(dirn == 0, 1, np.where(dirn > 0, 2, 3)),
                   np.where(dirn > 0, 4, np.where(dirn < 0, 5, 6))).astype(np.int8)
    warm = ~np.isfinite(q70) | ~np.isfinite(er)
    if ok is not None:
        warm = warm | ~ok
    reg = np.where(warm, 0, reg).astype(np.int8)
    return {"rv10": rv10, "q55": q55, "q70": q70, "vol_hi": vol_hi,
            "er": er, "ds": ds, "dirn": dirn, "regime": reg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="xyz:MU")
    a = ap.parse_args()
    tag = a.coin.replace(":", "_")

    ts, mid, stale = sec_grid(a.coin, tag)
    n = ts.size
    p = np.log(mid)
    r = np.diff(p, prepend=p[0])
    r = np.where(np.isfinite(r), r, 0.0)

    C = classify(r, np.isfinite(mid))
    rv10, q55, q70 = C["rv10"], C["q55"], C["q70"]
    vol_hi, er, ds, dirn, reg = (C["vol_hi"], C["er"], C["ds"],
                                 C["dirn"], C["regime"])

    dt = (np.datetime64("1970-01-01") + ts.astype("timedelta64[s]")
          ).astype("datetime64[D]").astype(str)
    out = pl.DataFrame({
        "ts": ts, "dt": dt, "sec": (ts % 86400).astype(np.int32),
        "mid": mid.astype(np.float64), "r": (r * 1e4).astype(np.float32),
        "rv10": (rv10 * 1e4).astype(np.float32),
        "q55": (q55 * 1e4).astype(np.float32),
        "q70": (q70 * 1e4).astype(np.float32),
        "vol_hi": vol_hi, "er": er.astype(np.float32),
        "ds": ds.astype(np.float32), "dirn": dirn, "regime": reg,
        "stale_s": stale.astype(np.float32)})
    out.write_parquet(DATA / f"regime_{tag}.parquet", compression="zstd")

    ok = reg > 0
    sh = np.bincount(reg[ok], minlength=7)[1:] / max(ok.sum(), 1)
    print(f"[regime] {a.coin}: {n:,} 秒 / {out['dt'].n_unique()} 日 / "
          f"判定できた秒 {ok.sum():,}({ok.mean()*100:.1f}%)")
    print(f"  気配が動いた秒 {np.mean(r != 0)*100:.1f}% / "
          f"更新からの経過 中央 {np.nanmedian(stale):.2f}s")
    nm = ["R1 低ボラ/中立", "R2 低ボラ/上", "R3 低ボラ/下",
          "R4 高ボラ/上", "R5 高ボラ/下", "R6 高ボラ/両建て"]
    for i, v in enumerate(sh):
        print(f"  {nm[i]:<18} {v*100:6.2f}%")
    print(f"  高ボラの秒 {np.mean(vol_hi[ok] > 0)*100:.1f}% / "
          f"方向つきの秒 {np.mean(dirn[ok] != 0)*100:.1f}%")


if __name__ == "__main__":
    main()
