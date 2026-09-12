r"""満期決済データの完全性を、約定データから独立に検証する。

=============================================================================
考え方
=============================================================================
オプションの建玉は **反対売買**か**満期決済**でしか終わらない。したがって:

  ある (subaccount, 銘柄) について、約定を全部足した残高が 0 でなく、
  かつその銘柄が既に満期を迎えているなら、**決済記録が必ず存在するはず**。

無ければ、その決済は API から取りこぼしている。
API のページ境界に重複が出ることを実測したので、取りこぼしも起こりうる。
この検査は API の申告件数に頼らず、**自前のデータだけで**欠損を炙り出す。

★併せて、決済の `amount` が約定から再生した残高と一致するかも照合する。
  一致しなければ、私の残高再生か API のどちらかが間違っている。

出力: 標準出力(レポート用)
"""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import zstandard as zstd

SRC = Path("E:/Derive-options/data")


def read_zst(path):
    with open(path, "rb") as fh:
        r = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    import glob

    # 1) 約定から (subaccount, 銘柄) の残高を再生
    pos = defaultdict(float)
    nrows = 0
    for p in sorted(glob.glob(str(SRC / "trades_rest" / "dt=*" / "*.jsonl.zst"))):
        for x in read_zst(p):
            nrows += 1
            q = float(x["trade_amount"])
            if x["direction"] != "buy":
                q = -q
            pos[(int(x["subaccount_id"]), x["instrument_name"])] += q
    print(f"約定 {nrows:,} 行から {len(pos):,} 組の (subaccount, 銘柄) を再生")

    # 2) 満期を迎えた銘柄(latest.json に無い = 上場終了)
    live = set()
    lp = SRC / "instruments" / "latest.json"
    if lp.exists():
        live = {x["name"] for x in json.loads(lp.read_text(encoding="utf-8"))}
    print(f"現在 active な銘柄 {len(live):,}")

    # 3) 決済データ
    S = {}
    for x in read_zst(str(SRC / "settlements" / "settlements.jsonl.zst")):
        S[(int(x["subaccount_id"]), x["instrument_name"])] = x
    print(f"決済 {len(S):,} 件\n")

    # 4) 「満期済みで残高が残っているのに決済が無い」を探す
    need = [(k, v) for k, v in pos.items()
            if abs(v) > a.tol and k[1] not in live]
    missing = [(k, v) for k, v in need if k not in S]
    print(f"満期済みで残高が残る組: {len(need):,}")
    print(f"  そのうち決済記録あり: {len(need)-len(missing):,} "
          f"({(len(need)-len(missing))/max(len(need),1)*100:.2f}%)")
    print(f"  ★決済記録が無い    : {len(missing):,}")
    if missing:
        import collections
        c = collections.Counter(k[1].split("-")[0] for k, _ in missing)
        print(f"    通貨別: {dict(c)}")
        print("    例:")
        for k, v in missing[:5]:
            print(f"      sub={k[0]} {k[1]} 残高 {v:+.4f}")

    # 5) 決済の amount が再生残高と一致するか
    ok = bad = 0
    diffs = []
    for k, x in S.items():
        if k not in pos:
            continue
        want = pos[k]
        got = float(x["amount"])
        if abs(want - got) <= max(a.tol, abs(want) * 1e-6):
            ok += 1
        else:
            bad += 1
            diffs.append((k, want, got))
    print(f"\n決済の amount と再生残高の照合: 一致 {ok:,} / 不一致 {bad:,}")
    if diffs:
        diffs.sort(key=lambda z: -abs(z[1] - z[2]))
        print("  ずれの大きい順:")
        for k, w, g in diffs[:5]:
            print(f"    sub={k[0]} {k[1]} 再生 {w:+.4f} / API {g:+.4f} "
                  f"(差 {w-g:+.4f})")
    # 6) 逆方向: 決済はあるが約定が無い(= 期間外の取引で作った建玉)
    orphan = [k for k in S if k not in pos]
    print(f"\n決済はあるが約定データに無い組: {len(orphan):,} "
          f"({len(orphan)/max(len(S),1)*100:.1f}%)")
    print("  ← 約定履歴の起点(2024-01)より前、または別経路で作った建玉")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
