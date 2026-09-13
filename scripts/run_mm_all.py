"""所有 7 銘柄でメイカーの経済的有意性を測るための一括実行。

    uv run python scripts/run_mm_all.py [--stage base|size|all]

`build_inventory.py`(在庫を持つメイカーの事象駆動シミュレータ)を、
**xyz:MU で確かめた設定のまま**他の 6 銘柄へ当てる。

## 何を回すか

| 段 | 設定 | 何が判るか |
|---|---|---|
| base | `--qmax 1`、遅延 0 / 65 / 130 ms | 幽霊注文(数量 0)での 1 組あたり損益と、遅延への耐性 |
| size | `--size` を **金額で**そろえて掃引 | 執行できる数量の上限。Signal × ExecutableSize |

数量の掃引は銘柄ごとに違う価格を吸収するため、**$100 / $1k / $10k / $100k を
その銘柄の中央 mid で枚数に直して**与える。板の厚みは銘柄で 20 倍違うので、
枚数をそろえると比較にならない。

## 幽霊注文の仮定

既定(`--size 0`)は「自分の注文は板に影響せず、前の行列がはけた瞬間に
無限小だけ約定する」。**スプレッドが広い銘柄ほどこの仮定は苦しい**ので、
`--size` を入れた掃引が本体の検証である。`--size S` は
「前の行列 + S が流れて初めて S 全量が約定する」という保守側の all-or-nothing。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COINS = ["MU", "INTC", "AMD", "KIOXIA", "SKHX", "SMSN", "SNDK"]
LATS = [0.0, 0.065, 0.130]
USD = [100, 1_000, 10_000, 100_000]


def mid_median(c: str) -> float:
    b = (pl.scan_parquet(DATA / f"bbo_xyz_{c}.parquet")
         .select(((pl.col("best_bid") + pl.col("best_ask")) / 2).alias("m"))
         .collect()["m"])
    return float(b.median())


def run(args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "build_inventory.py")] + args
    print("$", " ".join(args), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=ROOT)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        raise SystemExit(f"失敗: {args}")
    for ln in r.stdout.splitlines():
        if ln.startswith(("組の損益", "★", "発注")):
            print("   ", ln, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "size", "all"], default="all")
    ap.add_argument("--coins", nargs="*", default=COINS)
    a = ap.parse_args()

    for c in a.coins:
        m = mid_median(c)
        print(f"\n===== xyz:{c}(中央 mid ${m:,.1f})=====", flush=True)
        if a.stage in ("base", "all"):
            for L in LATS:
                arg = ["--coin", f"xyz:{c}", "--qmax", "1"]
                if L:
                    arg += ["--lat", str(L)]
                else:
                    arg += ["--posts"]
                run(arg)
        if a.stage in ("size", "all"):
            for u in USD:
                s = round(u / m, 6)
                print(f"  -- 数量 ${u:,} = {s:g} 枚", flush=True)
                run(["--coin", f"xyz:{c}", "--qmax", "1", "--size", str(s)])
                run(["--coin", f"xyz:{c}", "--qmax", "1", "--size", str(s),
                     "--lat", "0.130"])


if __name__ == "__main__":
    main()
