r"""メイカー報酬帯(incentiveRange)を全市場ぶん取得する — 特徴量 55 の材料。

【定義(Boros OpenAPI の原文)】
  incentiveRange:
    "Half-width of the incentive band around the mid implied APR for this side,
     expressed as a decimal APR (e.g. `0.005` = ±50bps).
     Resting size within this band qualifies."

  つまり **mid implied APR からの片側半幅**(APR 小数)で、long / short で別値。
  帯の内側に**置かれている残量**が報酬対象になる(約定ではなく resting)。
  トラックは毎時("Hourly maker resting-liquidity track")。

【★限界】
  エンドポイント `/v1/incentives/maker-incentives/campaigns/{marketId}` は
  **marketId と maker しか受け取らない**(OpenAPI で確認)。
  したがって**現在エポックの値しか取れない**。過去の窓(2026-05-04〜08-10)に
  そのときの帯を遡って当てることはできない。
  ここで取るのは「帯という制度が、いまどの市場でどの幅か」の断面である。

出力: data/incentive_range.parquet
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import polars as pl

SRC = Path("E:/Boros-history")
OUT = Path(__file__).resolve().parent.parent / "data"
BASE = "https://api-boros.pendle.finance/apis/v1"


def get(path: str, timeout: int = 25):
    """curl を使う(urllib は UA で 403 になることを偵察で確認済み)。"""
    r = subprocess.run(["curl", "-s", "--max-time", str(timeout), BASE + path],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def main() -> int:
    cfg = json.loads((SRC / "config/markets.json").read_text(encoding="utf-8"))
    rows = []
    for i, m in enumerate(cfg):
        mid = m["marketId"]
        d = get(f"/incentives/maker-incentives/campaigns/{mid}")
        if d is None:
            rows.append({"market": mid, "symbol": m["symbol"], "ok": False})
        else:
            al = d.get("addLiquidityIncentive") or {}
            lo, sh = al.get("long") or {}, al.get("short") or {}
            fv = d.get("filledVolumeIncentive") or {}
            fr = d.get("makerFeeRebate") or {}
            rows.append({
                "market": mid, "symbol": m["symbol"], "ok": True,
                "epoch_ts": d.get("epochTimestamp"),
                "range_long": lo.get("incentiveRange"),
                "range_short": sh.get("incentiveRange"),
                "budget_long": lo.get("budgetPerHour"),
                "budget_short": sh.get("budgetPerHour"),
                "inrange_long": float(lo.get("currentInRangeLiquidity") or 0) / 1e18,
                "inrange_short": float(sh.get("currentInRangeLiquidity") or 0) / 1e18,
                "epoch_reward": fv.get("totalEpochReward"),
                "total_maker_volume": fv.get("totalMakerVolume"),
                "fee_share_rate": fr.get("feeShareRate"),
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cfg)}", flush=True)
        time.sleep(0.12)          # 公開 API を叩き過ぎない
    d = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT / "incentive_range.parquet")
    ok = d.filter(pl.col("ok"))
    print(f"\n取得 {ok.height}/{d.height} 市場")
    print(f"-> {OUT/'incentive_range.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
