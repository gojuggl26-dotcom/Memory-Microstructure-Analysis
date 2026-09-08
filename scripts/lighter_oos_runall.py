r"""凍結構成の未見期間評価を、銘柄ごとに切り離して直列実行する。

★セッション連動のバックグラウンドは 2 度落ちた実績があるので Start-Process で切り離す。
★1 銘柄ずつ別プロセスにするのは RAM(1 銘柄 5〜7GB)のため。
  途中で落ちても、済んだ銘柄の shard は残る。
"""
import subprocess
import time
from pathlib import Path

PY = r"C:\Users\ii562\Downloads\Memory\.venv\Scripts\python.exe"
SC = Path(r"C:\Users\ii562\Downloads\Memory\scripts")
OUT = Path("E:/Memory-lighter/wf")
LOG = OUT.parent / "_oosrun.log"
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "DRAM",
        "XAU", "XAG", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {m}\n")


log("==== OOS 開始(直列)====")
for s in SYMS:
    if (OUT / f"oos_{s}.parquet").exists():
        log(f"SKIP {s}")
        continue
    log(f"START {s}")
    with open(OUT.parent / f"_oos_{s}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call(
            [PY, str(SC / "lighter_oos.py"), "--symbols", s, "--shard", s],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(SC.parent))
    log(f"DONE({rc}) {s}")
(OUT / "_OOS_DONE").write_text("ok")
log("==== OOS 完了 ====")
