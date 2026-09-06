r"""OOM で落ちた解析 4 銘柄を直列に再実行(切り離し用の小オーケストレータ)。"""
import subprocess
import time
from pathlib import Path

PY = r"C:\Users\ii562\Downloads\Memory\.venv\Scripts\python.exe"
SC = Path(r"C:\Users\ii562\Downloads\Memory\scripts")
OUT = Path("E:/Memory-lighter")
LOG = OUT / "_run2.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {m}\n")


for s in ["MU", "DRAM", "SNDK", "SKHYNIXUSD"]:
    if (OUT / "ana" / f"rho_{s}.parquet").exists():
        log(f"SKIP {s}")
        continue
    log(f"START ana:{s}")
    with open(OUT / f"_o_a2_{s}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call([PY, str(SC / "lighter_analyze.py"),
                              "--symbols", s], stdout=f,
                             stderr=subprocess.STDOUT, cwd=str(SC.parent))
    log(f"DONE({rc}) ana:{s}")
(OUT / "_STAGE_ANA2_DONE").write_text("ok")
log("再実行 完了")
