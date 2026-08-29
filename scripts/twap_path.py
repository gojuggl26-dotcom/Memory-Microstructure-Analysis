"""TWAP イベントの価格経路 — 反転・最大逆行・費用控除後の経済性。

【なぜ経路が要るか】
  報告の第一段(twap_sandwich.py)は終点だけを見た。サンドイッチの可否は終点では決まらない。
    ・**反転**: TWAP 終了後に戻るなら「一時的インパクト」で、サンドイッチが成立する。
      戻らないなら情報であり、先回りは「情報のある相手の前に立つ」ことになる
    ・**最大逆行(MAE)**: 入ってから最も不利になった点。ここで耐えられなければ
      終点の平均は取れない(CLAUDE.md §D-16 実装可能性)
    ・**費用**: 往復のスプレッドと手数料を引く前の bp に意味はない(§A-1)

【時間契約】
  t0 = activated の時刻。経路は t0 以降のみ。方向は t0 で判る state_side。
  価格は backward asof(その時刻以前の最新 mid)。

【出力】 data/twap_path.csv(イベント × 経路統計)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

PIPE = Path("C:/Users/ii562/hl-l4-pipeline")
D = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
STEP_S = 30                      # スライス間隔と同じ
N_STEP = 240                     # 30s × 240 = 120 分
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


def day_mid(dt: str):
    fs = glob.glob(str(D / f"microprice/dt={dt}/*.parquet"))
    if not fs:
        return None
    m = (pl.read_parquet(fs[0], columns=["ts", "mid", "spread_bp", "is_crossed"])
           .filter(~pl.col("is_crossed")).sort("ts"))
    if len(m) < 1000:
        return None
    ts = m["ts"].to_numpy()
    if ts.dtype != np.int64:
        ts = ts.astype("datetime64[ns]").astype(np.int64)
    return ts, m["mid"].to_numpy(), float(np.median(m["spread_bp"].to_numpy()))


def path_stats(t0: np.ndarray, sign: np.ndarray, dur_s: np.ndarray,
               ts: np.ndarray, mid: np.ndarray) -> dict[str, np.ndarray]:
    grid = (t0[:, None] + (np.arange(N_STEP + 1) * STEP_S * 1_000_000_000)[None, :])
    j = np.searchsorted(ts, grid, side="right") - 1
    valid = (j >= 0) & (grid <= ts[-1])
    p = np.where(valid, mid[np.clip(j, 0, len(mid) - 1)], np.nan)
    bp = sign[:, None] * (p / p[:, [0]] - 1.0) * 1e4          # 符号つき累積 bp
    k_end = np.clip((dur_s // STEP_S).astype(int), 1, N_STEP)  # 終了ステップ
    idx = np.arange(N_STEP + 1)[None, :]
    in_win = idx <= k_end[:, None]
    with np.errstate(invalid="ignore"):
        mae = np.nanmin(np.where(in_win, bp, np.nan), axis=1)   # 最大逆行
        mfe = np.nanmax(np.where(in_win, bp, np.nan), axis=1)   # 最大順行
    rows = np.arange(len(t0))
    bp_end = bp[rows, k_end]
    k30 = np.clip(k_end + 60, 0, N_STEP)                        # 終了 +30 分
    bp_post = bp[rows, k30]
    return {"bp_end": bp_end, "bp_post30": bp_post,
            "reversal": bp_post - bp_end, "mae": mae, "mfe": mfe,
            "bp_5m": bp[:, 10], "bp_15m": bp[:, 30], "bp_30m": bp[:, 60]}


def main() -> None:
    tw = load_twap()
    out = []
    for key, g in tw.group_by("dt"):
        dt = key[0] if isinstance(key, tuple) else key
        s = day_mid(dt)
        if s is None:
            continue
        ts, mid, spr = s
        t0 = g["t0"].to_numpy()
        sign = np.where(g["state_side"].to_numpy() == "B", 1.0, -1.0)
        dur = np.minimum(g["state_minutes"].to_numpy().astype(np.int64), 120) * 60
        st = path_stats(t0, sign, dur, ts, mid)
        out.append(g.with_columns(spread_bp_day=pl.lit(spr),
                                  **{k: pl.Series(v) for k, v in st.items()}))
    pl.concat(out).sort("t0").write_csv(D / "twap_path.csv")
    print(f"イベント {sum(len(o) for o in out)}")


if __name__ == "__main__":
    main()
