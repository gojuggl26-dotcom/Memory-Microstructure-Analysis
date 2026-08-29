"""TWAP サンドイッチ戦略の経済的有意性 — 価格インパクトの実測。

【問い】
  Hyperliquid の TWAP 注文は `activated` の時点で
  **総数量・方向・所要分数が公開される**(node_twap_statuses)。
  「これから N 分かけて X 単位買う」が事前に判るなら、
  先回りして買い、TWAP に売りつける(サンドイッチ)戦略が成り立つか。

【時間契約 — CLAUDE.md 厳禁事項の遵守】
  x が確定する時刻 = t0(activated の時刻)。t0 で判るのは
    side / state_sz / state_minutes / randomize / reduceOnly / user
  のみ。**完遂したか(finished/terminated)・実際の執行量は未来情報なので使わない。**
  y = t0 以降のリターン。窓 [t0, t0+h) の符号つきリターンを測る。

【判定】
  符号つきインパクト(bp)が往復費用を超えるか。
  往復費用 = 2×(ハーフスプレッド + テイカー手数料) 〜 2×メイカー手数料 の幅で提示。

【帰無対照】
  プラセボ = 同じ日・同じ時刻分布からランダムに選んだ時刻で同じ測定。
  TWAP の方向はランダムに割り当てる。

【出力】
  data/twap_impact.csv  (イベント単位)
  data/twap_placebo.csv (プラセボ)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

PIPE = Path("C:/Users/ii562/hl-l4-pipeline")
D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
# t0 からの評価地平(秒)。state_minutes ベースの終端も別途出す
HORIZONS_S = [1, 10, 60, 300, 900, 1800]
RNG = np.random.default_rng(20260819)


def load_twap() -> pl.DataFrame:
    fs = sorted(glob.glob(str(PIPE / "data/twap/dt=*/coin=*/part-000.parquet")))
    d = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    return (d.filter(pl.col("status") == "activated")
             .with_columns(sz=pl.col("state_sz").cast(pl.Float64),
                           t0=pl.col("ts").cast(pl.Int64),
                           dt=pl.col("ts").dt.strftime("%Y-%m-%d"))
             .select("dt", "t0", "twap_id", "state_side", "sz", "state_minutes",
                     "state_randomize", "state_reduceOnly", "state_user"))


def mid_series(dt: str) -> tuple[np.ndarray, np.ndarray] | None:
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    if len(m) < 1000:
        return None
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:                    # datetime[ns] のことがある
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    return ts, m["mid"].to_numpy()


def px_at(ts: np.ndarray, mid: np.ndarray, t: np.ndarray) -> np.ndarray:
    """時刻 t 直前(以前)の最新 mid。backward asof(未来を見ない)。"""
    j = np.searchsorted(ts, t, side="right") - 1
    ok = j >= 0
    out = np.full(len(t), np.nan)
    out[ok] = mid[j[ok]]
    return out


def measure(t0: np.ndarray, sign: np.ndarray, dur_ns: np.ndarray,
            ts: np.ndarray, mid: np.ndarray) -> dict[str, np.ndarray]:
    p0 = px_at(ts, mid, t0)
    out: dict[str, np.ndarray] = {"px0": p0}
    for h in HORIZONS_S:
        ph = px_at(ts, mid, t0 + h * 1_000_000_000)
        # 日の終わりを超えた分は NaN(px_at が最終値を返すので明示的に落とす)
        ph = np.where(t0 + h * 1_000_000_000 <= ts[-1], ph, np.nan)
        out[f"bp_{h}s"] = sign * (ph / p0 - 1.0) * 1e4
    pe = px_at(ts, mid, t0 + dur_ns)
    pe = np.where(t0 + dur_ns <= ts[-1], pe, np.nan)
    out["bp_dur"] = sign * (pe / p0 - 1.0) * 1e4
    return out


def main() -> None:
    tw = load_twap()
    real_rows, plac_rows = [], []
    for dt, g in tw.group_by("dt"):
        dt = dt[0] if isinstance(dt, tuple) else dt
        s = mid_series(dt)
        if s is None:
            continue
        ts, mid = s
        t0 = g["t0"].to_numpy()
        # side: fills で検証済み(B=買い / A=売り)
        sign = np.where(g["state_side"].to_numpy() == "B", 1.0, -1.0)
        dur = np.minimum(g["state_minutes"].to_numpy().astype(np.int64), 1440) * 60_000_000_000
        m = measure(t0, sign, dur, ts, mid)
        base = g.select("dt", "t0", "twap_id", "state_side", "sz", "state_minutes",
                        "state_randomize", "state_reduceOnly", "state_user")
        real_rows.append(base.with_columns(**{k: pl.Series(v) for k, v in m.items()}))
        # プラセボ: 同じ日のランダム時刻 × ランダム方向、実イベント数の 5 倍
        n = len(g) * 5
        tp = RNG.integers(ts[0], ts[-1], size=n)
        sp = RNG.choice([1.0, -1.0], size=n)
        dp = RNG.choice(dur, size=n)
        mp = measure(tp, sp, dp, ts, mid)
        plac_rows.append(pl.DataFrame({"dt": [dt] * n, "t0": tp,
                                       **{k: v for k, v in mp.items()}}))
    real = pl.concat(real_rows).sort("t0")
    plac = pl.concat(plac_rows).sort("t0")
    real.write_csv(D / "twap_impact.csv")
    plac.write_csv(D / "twap_placebo.csv")
    print(f"実イベント {len(real)} / プラセボ {len(plac)}")


if __name__ == "__main__":
    main()
