r"""Lighter 一括パイプラインのオーケストレータ(セッションから独立に走る)。

Claude セッションのバックグラウンドはセッション終了で死ぬため、
これを Start-Process で切り離して走らせ、進捗は E:/Memory-lighter/_run.log と
ステージごとの DONE ファイルで確認する。

段取り:
  1. パース(欠けている銘柄のみ、--force が無ければ既存 grid+meta はスキップ)
  2. Binance BBO 抽出(bnb_{SYM}.parquet が無い銘柄のみ)… 1 と並行
  3. 解析(ana/rho_{SYM}.parquet が無い銘柄のみ)… grid が揃った銘柄から順に
  4. 回帰(reg_perf_{SYM 先頭}.parquet)… 解析済み銘柄から
  5. リードラグ … 2 と該当銘柄のパースが済んだら

同時実行は最大 MAXW。すべて subprocess で、この親が死んでも子は走り切る。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\ii562\Downloads\Memory\.venv\Scripts\python.exe"
SC = Path(r"C:\Users\ii562\Downloads\Memory\scripts")
OUT = Path("E:/Memory-lighter")
ANA = OUT / "ana"
LL = OUT / "ll"
LOG = OUT / "_run.log"
MAXW = 5
SYMS = ["MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "XAG", "H100",
        "AAPL", "AMZN", "DRAM", "MSFT", "NVDA", "TSLA", "XAU"]
BNB = ["DRAM", "MU", "SNDK", "SKHYNIXUSD", "SAMSUNGUSD", "AAPL", "AMZN",
       "MSFT", "NVDA", "TSLA", "XAG", "XAU"]


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


class Pool:
    def __init__(self, maxw: int):
        self.maxw = maxw
        self.run: list[tuple[subprocess.Popen, str]] = []

    def wait_slot(self):
        while len(self.run) >= self.maxw:
            self.poll()
            time.sleep(5)

    def poll(self):
        keep = []
        for p, name in self.run:
            rc = p.poll()
            if rc is None:
                keep.append((p, name))
            else:
                log(f"DONE({rc}) {name}")
        self.run = keep

    def spawn(self, args: list[str], name: str, logfile: str):
        self.wait_slot()
        f = open(OUT / logfile, "a", encoding="utf-8")
        p = subprocess.Popen([PY] + args, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(SC.parent))
        self.run.append((p, name))
        log(f"START {name}")

    def drain(self):
        while self.run:
            self.poll()
            time.sleep(5)


def grid_ok(s: str) -> bool:
    m = OUT / f"meta_{s}.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
        return d.get("d1") == "2026-09-05" and (OUT / f"grid_{s}.parquet").exists()
    except Exception:
        return False


def main() -> int:
    log("==== orchestrator 開始 ====")
    pool = Pool(MAXW)

    # 2. Binance 抽出(先に 1 枠で流す)
    need_bnb = [s for s in BNB if not (OUT / f"bnb_{s}.parquet").exists()]
    if need_bnb:
        pool.spawn([str(SC / "lighter_binance_bbo.py"),
                    "--symbols", ",".join(need_bnb)], "bnb", "_o_bnb.log")

    # 1. パース(欠け銘柄)→ 銘柄ごとに 1 プロセス
    need_parse = [s for s in SYMS if not grid_ok(s)]
    log(f"パース対象: {need_parse}")
    for s in need_parse:
        pool.spawn([str(SC / "lighter_parse.py"), "--symbols", s],
                   f"parse:{s}", f"_o_p_{s}.log")

    # 3. 解析: grid が揃った銘柄から順に(既済みはスキップ)
    pending = [s for s in SYMS if not (ANA / f"rho_{s}.parquet").exists()]
    log(f"解析対象: {pending}")
    while pending:
        launched = []
        for s in pending:
            if grid_ok(s):
                pool.spawn([str(SC / "lighter_analyze.py"), "--symbols", s],
                           f"ana:{s}", f"_o_a_{s}.log")
                launched.append(s)
        for s in launched:
            pending.remove(s)
        if pending:
            pool.poll()
            time.sleep(20)
    pool.drain()
    (OUT / "_STAGE_ANA_DONE").write_text("ok")
    log("解析 全銘柄完了")

    # 4. 回帰(2 分割)
    half = ["MU,SNDK,DRAM,SKHYNIXUSD,SAMSUNGUSD,XAU,H100",
            "AAPL,AMZN,MSFT,NVDA,TSLA,XAG"]
    for h in half:
        pool.spawn([str(SC / "lighter_reg.py"), "--symbols", h],
                   f"reg:{h[:12]}", "_o_reg.log")
    # 5. リードラグ(bnb と全パース完了後 — この時点で両方済んでいる)
    pool.drain()
    pool.spawn([str(SC / "lighter_leadlag.py")], "leadlag", "_o_ll.log")
    pool.drain()
    (OUT / "_STAGE_ALL_DONE").write_text("ok")
    log("==== 全段完了 ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
