r"""ウォークフォワード用の拡張パース(2026-08-18〜09-07)を切り離して走らせる。

既存の E:/Memory-lighter/grid_*.parquet(窓 08-18〜09-05)は**壊さない**。
公開済みレポートの数字がそのまま再現できるようにするため、別ディレクトリへ出す。

出力: E:/Memory-lighter/wf/grid_{SYM}.parquet, bbo_{SYM}.parquet, meta_{SYM}.json

★H100 は板の被覆 14.7% で市場として成立していない(データ報告 §2)。
  パースはするが、ウォークフォワードの対象からは**結果を見る前に**外す。
"""
import subprocess
import time
from pathlib import Path

PY = r"C:\Users\ii562\Downloads\Memory\.venv\Scripts\python.exe"
SC = Path(r"C:\Users\ii562\Downloads\Memory\scripts")
OUT = Path("E:/Memory-lighter/wf")
LOG = OUT.parent / "_wfparse.log"
MAXW = 5
D0, D1 = "2026-08-18", "2026-09-07"
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "DRAM",
        "XAU", "XAG", "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "H100"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {m}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"==== wfparse {D0}..{D1} 開始 ====")
    run = []
    todo = [s for s in SYMS if not (OUT / f"meta_{s}.json").exists()]
    log(f"対象 {todo}")
    for s in todo:
        while len(run) >= MAXW:
            run = [(p, n) for p, n in run if p.poll() is None]
            time.sleep(5)
        f = open(OUT.parent / f"_wfp_{s}.log", "a", encoding="utf-8")
        p = subprocess.Popen(
            [PY, str(SC / "lighter_parse.py"), "--symbols", s,
             "--start", D0, "--end", D1, "--out", str(OUT)],
            stdout=f, stderr=subprocess.STDOUT, cwd=str(SC.parent))
        run.append((p, s))
        log(f"START {s}")
    while run:
        nxt = []
        for p, n in run:
            rc = p.poll()
            if rc is None:
                nxt.append((p, n))
            else:
                log(f"DONE({rc}) {n}")
        run = nxt
        time.sleep(5)
    (OUT / "_PARSE_DONE").write_text("ok")
    log("==== wfparse 完了 ====")


if __name__ == "__main__":
    main()
