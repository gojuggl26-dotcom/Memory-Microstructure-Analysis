"""WORK_BUCKET の l2/bbo(DEEP_ARCHIVE)を復元要求する。

    uv run python scripts/restore_bbo_s3.py --coins GOLD GOOGL AAPL [--go]

`--go` を付けるまで**何も要求せず**、対象の件数と容量と概算費用だけ出す。

## なぜ要るか

`xyz:GOLD` / `xyz:GOOGL` / `xyz:AAPL` は L2 が WORK_BUCKET にあるが、
ライフサイクル規則で **DEEP_ARCHIVE** に落ちているので直接は読めない。
復元要求を出してから 12 時間(Standard)待つと、一定期間だけ読めるようになる。

## 費用(Standard 取り出し・2026 年時点の us-east-1)

    取り出し   $0.02 / GB
    要求       $0.10 / 1,000 件
    転送       $0.09 / GB(インターネットへの egress)

bbo は 3 銘柄で 0.76 GB・689 件なので、**合計 $0.15 程度**である。
`--days` で日数を絞れば比例して減る。

## 注意

* `.inprogress` と `_SUCCESS` は除く(本体は `part-*.parquet` だけ)。
* 復元は**コピーを一時的に作るだけ**で、元のオブジェクトの保管クラスは変わらない。
  `--keep-days` を過ぎると一時コピーは自動で消える。
* 既に復元済み・復元中のオブジェクトは skip する。
* 読み取りだけで、削除も上書きもしない。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

RETRIEVE_USD_GB = 0.02      # Standard
REQ_USD_1K = 0.10
EGRESS_USD_GB = 0.09


def env(name: str) -> str:
    v = os.environ.get(name)
    if v:
        return v
    f = Path.home() / ".hlpipe.env"
    for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("export "):
            ln = ln[7:]
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} が見つからない")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["GOLD", "GOOGL", "AAPL"])
    ap.add_argument("--table", default="bbo")
    ap.add_argument("--tier", default="Standard",
                    choices=["Standard", "Bulk", "Expedited"])
    ap.add_argument("--keep-days", type=int, default=7)
    ap.add_argument("--go", action="store_true", help="実際に復元要求を出す")
    a = ap.parse_args()

    bucket = env("WORK_BUCKET")
    s3 = boto3.Session(profile_name="hl-artemis-ro",
                       region_name="us-east-1").client("s3")
    todo, nbytes, ready, pending = [], 0, 0, 0
    for c in a.coins:
        pref = f"l2/coin=xyz%3A{c}/{a.table}/"
        tok, n = None, 0
        while True:
            kw = {"Bucket": bucket, "Prefix": pref, "MaxKeys": 1000}
            if tok:
                kw["ContinuationToken"] = tok
            r = s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                k = o["Key"]
                if not k.endswith(".parquet") or ".inprogress" in k:
                    continue
                n += 1
                if o.get("StorageClass") not in ("DEEP_ARCHIVE", "GLACIER"):
                    ready += 1
                    continue
                todo.append(k)
                nbytes += o["Size"]
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
        print(f"xyz:{c:<6} {a.table}: {n} 件")
    gb = nbytes / 1e9
    cost = gb * (RETRIEVE_USD_GB + EGRESS_USD_GB) + len(todo) / 1000 * REQ_USD_1K
    print(f"\n復元が要るもの {len(todo)} 件 / {gb:.3f} GB / 既に読めるもの {ready} 件")
    print(f"概算費用 取り出し ${gb*RETRIEVE_USD_GB:.3f} + 要求 "
          f"${len(todo)/1000*REQ_USD_1K:.3f} + 転送 ${gb*EGRESS_USD_GB:.3f} "
          f"= **${cost:.2f}**({a.tier})")
    if not a.go:
        print("\n--go を付けると実際に復元要求を出す(いまは何もしていない)")
        return

    err = 0
    for i, k in enumerate(todo, 1):
        try:
            s3.restore_object(
                Bucket=bucket, Key=k,
                RestoreRequest={"Days": a.keep_days,
                                "GlacierJobParameters": {"Tier": a.tier}})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "RestoreAlreadyInProgress":
                pending += 1
            else:
                err += 1
                if err < 5:
                    print(f"  失敗 {k}: {code}")
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print(f"\n要求 {len(todo)} 件(うち進行中だったもの {pending} / 失敗 {err})")
    print(f"{a.tier} なので 12 時間前後で読めるようになる(保持 {a.keep_days} 日)")


if __name__ == "__main__":
    main()
