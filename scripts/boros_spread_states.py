"""spread の「状態」を全市場で列挙する — 継続時間とその間の幅。

【何を作るか】
  `spread_pp` が一定である区間を 1 状態とし、状態ごとに
      duration   その幅が続いた秒数
      width_pp   その間の spread(pp)
      width_tick 同じものを **tick 単位**に直したもの(市場間で比較するため)
  を出す。板は**イベントでしか動かない**ので、イベント間に隙間があっても
  その間 spread は実際にその値のままである = duration は実時間として正しい。

【tick への換算】
  Boros の価格は tick t に対し  rate(t) = (1.00005^(t·tickStep) − 1)·100 [pp]。
  逆に  t = log(1 + rate/100) / (tickStep · log 1.00005)。
  幅は **best_short の tick − best_long の tick** で厳密に整数になる。
  pp のままだと金利水準で刻み幅が変わる(幾何級数)ので市場間比較に使えない。

【打ち切り】
  各市場の**最後の状態は右打ち切り**(データ終端でまだ続いている)。
  市場あたり 1 件なので全体の 0.01% 程度だが、フラグを立てて除外できるようにする。
  外挿区間(アーカイブの最終ブロック以降)にかかる状態も別途フラグを立てる。

【出力】 data/spread_states.parquet
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = Path("E:/Boros-history")
LOG105 = np.log(1.00005)


def to_tick(rate_pp: np.ndarray, tick_step: int) -> np.ndarray:
    """rate[pp] → tick(整数)。rate(t) の逆関数。"""
    return np.round(np.log1p(rate_pp / 100.0) / (tick_step * LOG105))


def states_of(path: Path, cfg: dict) -> pl.DataFrame | None:
    mid = int(re.search(r"event_book_(\d+)", path.name).group(1))
    m = cfg.get(mid)
    if m is None:
        return None
    d = pl.read_parquet(path, columns=["ts", "best_long", "best_short", "spread_pp",
                                       "mid_pp", "slope_long", "slope_short",
                                       "slope_diff", "depth_long", "depth_short",
                                       "n_lv_long", "n_lv_short", "ttm_years",
                                       "extrapolated"]).sort("ts")
    if d.height < 50:
        return None
    ts = d["ts"].to_numpy().astype(float)
    sp = d["spread_pp"].to_numpy().astype(float)
    tl = to_tick(d["best_long"].to_numpy().astype(float), m["tickStep"])
    tsh = to_tick(d["best_short"].to_numpy().astype(float), m["tickStep"])
    wtick = tsh - tl

    chg = np.r_[True, sp[1:] != sp[:-1]]
    start_i = np.flatnonzero(chg)                       # 各状態の最初のイベント
    end_i = np.r_[start_i[1:] - 1, len(ts) - 1]         # 各状態の最後のイベント
    dur = ts[np.r_[start_i[1:], len(ts) - 1]] - ts[start_i]
    censored = np.zeros(len(start_i), bool)
    censored[-1] = True                                  # 最後の状態は右打ち切り
    return pl.DataFrame({
        "market": np.full(len(start_i), mid, dtype=np.int32),
        "platform": [m["symbol"].split("-")[0]] * len(start_i),
        "symbol": [m["symbol"]] * len(start_i),
        "start_ts": ts[start_i],
        "duration": dur,
        "width_pp": sp[start_i],
        "width_tick": wtick[start_i],
        "n_events": (end_i - start_i + 1).astype(np.int32),
        "mid_pp": d["mid_pp"].to_numpy()[start_i],
        "slope_diff": d["slope_diff"].to_numpy()[start_i],
        "depth_long": d["depth_long"].to_numpy()[start_i],
        "depth_short": d["depth_short"].to_numpy()[start_i],
        "n_lv_long": d["n_lv_long"].to_numpy()[start_i],
        "n_lv_short": d["n_lv_short"].to_numpy()[start_i],
        "ttm_years": d["ttm_years"].to_numpy()[start_i],
        "censored": censored,
        "extrapolated": d["extrapolated"].to_numpy()[start_i],
    })


def main() -> int:
    cfg = {m["marketId"]: m for m in
           json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))}
    out = []
    for f in sorted(glob.glob(str(DATA / "event_book_*.parquet"))):
        s = states_of(Path(f), cfg)
        if s is not None:
            out.append(s)
    d = pl.concat(out, how="diagonal_relaxed")
    d.write_parquet(DATA / "spread_states.parquet")
    v = d.filter(~pl.col("censored") & ~pl.col("extrapolated"))
    print(f"spread の状態: {d.height:,} 件 / {d['market'].n_unique()} 市場")
    print(f"  右打ち切り {int(d['censored'].sum()):,} / 外挿区間 {int(d['extrapolated'].sum()):,}"
          f" → 解析対象 {v.height:,}")
    du = v["duration"].to_numpy()
    wt = v["width_tick"].to_numpy()
    print(f"\n継続時間(秒): 中央 {np.median(du):.2f}  平均 {du.mean():.0f}  "
          f"p90 {np.percentile(du, 90):.0f}  最大 {du.max():.0f}")
    print(f"  1 秒未満 {np.mean(du < 1):.1%} / 1 分未満 {np.mean(du < 60):.1%} / "
          f"1 時間超 {np.mean(du > 3600):.1%}")
    print(f"幅(tick): 中央 {np.median(wt):.0f}  p10 {np.percentile(wt, 10):.0f}  "
          f"p90 {np.percentile(wt, 90):.0f}  最大 {wt.max():.0f}")
    print(f"幅(pp)  : 中央 {np.median(v['width_pp'].to_numpy()):.4f}")
    print(f"\n-> {DATA / 'spread_states.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
