"""板データの保存先を SSD と HDD で実測比較する。

このマシンは C: = NVMe SSD(Micron 2450)、D: = 7200rpm HDD(Seagate BarraCuda 2TB)。
同じファイルを両方に置き、この案件で実際に使っている読み方で時間を測る。

測る 4 パターン(いずれも板データ解析の典型):
  1. 生の逐次読み          … ファイルをそのまま端から読む(理論上の帯域)
  2. parquet 全列読み       … 1 日分を丸ごと DataFrame にする
  3. parquet 列射影         … 27 列のうち 3 列だけ読む(解析で最も多い形)
  4. 多数の小ファイル       … 594 個の窓別パーティションを開いて集計する

★キャッシュ対策: Windows は読んだファイルを RAM(15.3GB)に丸ごと載せてしまうため、
各測定の前に大きな配列を確保してページキャッシュを追い出す。
これをしないと 2 回目以降は「メモリの速度」を測ることになる。
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path("C:/Users/ii562/Downloads/Memory/data/DRAM")
SSD = Path("C:/Users/ii562/AppData/Local/Temp/claude/disk_bench")
HDD = Path("D:/disk_bench")
FLUSH_GB = 9.0          # RAM 15.3GB のうち 9GB を確保してキャッシュを追い出す


def flush_cache() -> None:
    n = int(FLUSH_GB * (1 << 30) // 8)
    a = np.empty(n, dtype=np.float64)
    a[::512] = 1.0                     # 4KB ごとに触ってページを実体化させる
    del a
    gc.collect()
    time.sleep(1.0)


def stage(dst: Path) -> dict:
    """テスト用データを配置する(SSD/HDD 双方に同じもの)。"""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "obi").mkdir(parents=True)
    (dst / "grid").mkdir(parents=True)
    days = sorted(p.name for p in (SRC / "obi").iterdir() if p.is_dir())[-30:]
    tot = 0
    for d in days:
        s = SRC / "obi" / d / "part-000.parquet"
        t = dst / "obi" / f"{d}.parquet"
        shutil.copy2(s, t)
        tot += t.stat().st_size
    small = 0
    for step in ("1ms", "10ms", "100ms", "1s", "10s", "60s"):
        sd = SRC / "cancel_grid" / f"step={step}"
        if not sd.exists():
            continue
        (dst / "grid" / step).mkdir(parents=True, exist_ok=True)
        for p in sorted(sd.iterdir()):
            f = p / "part-000.parquet"
            if f.exists():
                t = dst / "grid" / step / f"{p.name}.parquet"
                shutil.copy2(f, t)
                small += t.stat().st_size
    return {"obi_bytes": tot, "grid_bytes": small,
            "obi_files": len(days),
            "grid_files": sum(1 for _ in (dst / "grid").rglob("*.parquet"))}


def bench(root: Path, name: str) -> dict:
    files = sorted((root / "obi").glob("*.parquet"))
    grid = sorted((root / "grid").rglob("*.parquet"))
    res = {"drive": name}

    # 1. 生の逐次読み
    flush_cache()
    t0 = time.perf_counter()
    nb = 0
    for f in files:
        with open(f, "rb", buffering=0) as fh:
            while chunk := fh.read(8 << 20):
                nb += len(chunk)
    dt = time.perf_counter() - t0
    res["seq_read"] = {"sec": dt, "bytes": nb, "MB_per_s": nb / dt / 1e6}

    # 2. parquet 全列
    flush_cache()
    t0 = time.perf_counter()
    rows = 0
    for f in files:
        rows += pl.read_parquet(f).height
    dt = time.perf_counter() - t0
    res["parquet_all_cols"] = {"sec": dt, "rows": rows, "Mrows_per_s": rows / dt / 1e6}

    # 3. parquet 列射影(3 列)
    flush_cache()
    t0 = time.perf_counter()
    rows = 0
    for f in files:
        rows += pl.read_parquet(f, columns=["ts", "obi_1", "obi_10"]).height
    dt = time.perf_counter() - t0
    res["parquet_3cols"] = {"sec": dt, "rows": rows, "Mrows_per_s": rows / dt / 1e6}

    # 4. 多数の小ファイル
    flush_cache()
    t0 = time.perf_counter()
    n = 0
    for f in grid:
        n += pl.read_parquet(f, columns=["ts", "n_cancel_bid"]).height
    dt = time.perf_counter() - t0
    res["many_small_files"] = {"sec": dt, "files": len(grid), "rows": n,
                               "files_per_s": len(grid) / dt}
    return res


def main() -> None:
    out = {}
    print("配置中(SSD)...", flush=True)
    out["staged_ssd"] = stage(SSD)
    print("配置中(HDD)...", flush=True)
    t0 = time.perf_counter()
    out["staged_hdd"] = stage(HDD)
    out["hdd_write_sec"] = time.perf_counter() - t0
    print(json.dumps(out["staged_ssd"], ensure_ascii=False), flush=True)

    for root, nm in ((SSD, "SSD (NVMe)"), (HDD, "HDD (7200rpm)")):
        print(f"\n=== {nm} ===", flush=True)
        r = bench(root, nm)
        out[nm] = r
        for k in ("seq_read", "parquet_all_cols", "parquet_3cols", "many_small_files"):
            v = r[k]
            extra = (f"{v['MB_per_s']:.0f} MB/s" if "MB_per_s" in v else
                     f"{v['Mrows_per_s']:.2f} M行/s" if "Mrows_per_s" in v else
                     f"{v['files_per_s']:.1f} ファイル/s")
            print(f"  {k:<20} {v['sec']:8.2f} 秒   {extra}", flush=True)

    a, b = out["SSD (NVMe)"], out["HDD (7200rpm)"]
    out["ratio_hdd_over_ssd"] = {k: b[k]["sec"] / a[k]["sec"]
                                 for k in ("seq_read", "parquet_all_cols",
                                           "parquet_3cols", "many_small_files")}
    Path("C:/Users/ii562/Downloads/Memory/data/DRAM/disk_bench.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== HDD は SSD の何倍の時間がかかるか ===")
    for k, v in out["ratio_hdd_over_ssd"].items():
        print(f"  {k:<20} {v:6.1f} 倍")


if __name__ == "__main__":
    main()
