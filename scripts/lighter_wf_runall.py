r"""ウォークフォワードを 1 銘柄ずつ直列に走らせる(セッションから切り離す用)。

★1 銘柄で特徴量行列に 5〜7GB 使う。RAM 15.3GB のマシンなので並列にしない
  (以前 4 並列で ArrayMemoryError を出した)。
"""
import subprocess
import time
from pathlib import Path

PY = r"C:\Users\ii562\Downloads\Memory\.venv\Scripts\python.exe"
SC = Path(r"C:\Users\ii562\Downloads\Memory\scripts")
OUT = Path("E:/Memory-lighter/wf")
LOG = OUT.parent / "_wfrun.log"
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "DRAM",
        "XAU", "XAG", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {m}\n")


log("==== walk-forward 開始(直列)====")
for s in SYMS:
    if (OUT / f"ledger_{s}.parquet").exists():
        log(f"SKIP {s}")
        continue
    log(f"START {s}")
    with open(OUT.parent / f"_wfr_{s}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call([PY, str(SC / "lighter_wf.py"), "--symbols", s],
                             stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(SC.parent))
    log(f"DONE({rc}) {s}")
(OUT / "_WF_DONE").write_text("ok")
log("==== walk-forward 完了 ====")
