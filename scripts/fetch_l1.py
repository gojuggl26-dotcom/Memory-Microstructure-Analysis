"""WORK_BUCKET の l1(注文イベント)から必要列だけを落とす。

ティック水準別・キュー位置別の約定率には**注文 1 本ごと**の記録が要る。
bbo は最良気配の集計値しか持たないので不足で、本来は l2/lifecycle を使うが
それは DEEP_ARCHIVE に移行済みで読めない。l1 は GLACIER_IR なので直接読める。

★費用: GLACIER_IR は取り出しに $0.03/GB、加えて egress $0.09/GB がかかる。
列プルーニングすると全列の約 33%(hash と cloid が全体の 65% を占めるため)。
99 日で約 13.8 GB = 取り出し $0.41 + egress $1.24 ≈ $1.65 の見込み。

x が確定する時刻 / y の期間: 該当なし(データ取得)。

    uv run python scripts/fetch_l1.py --coin xyz:MU --days 3     # 先頭 3 日で試す
    uv run python scripts/fetch_l1.py --coin xyz:MU              # 全日
出力: data/l1_<coin>.parquet(--days 指定時は data/l1_<coin>_head<N>.parquet)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# 板の再構成に要る最小限。hash / cloid / user は全体の 67% を占めるので取らない。
# reduce_only は必須。取引所がポジション減少で注文を自動縮小するため、
# 残量が減っても約定とは限らない(hl-l4-pipeline で filled_sz を 22.8% 過大計上した罠)。
# is_trigger も必須。トリガー注文(ストップ / TP)は発火するまで板に載らないので、
# これを板へ入れると買い $991 と売り $96 が同時に生存してクロスする(実際に踏んだ)。
COLS = ["ts", "oid", "side", "px", "status", "orig_sz", "remaining_sz", "tif",
        "reduce_only", "is_trigger"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True)
    ap.add_argument("--days", type=int, default=0, help="先頭 N 日だけ(0 = 全日)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    bucket = os.environ.get("WORK_BUCKET") or sys.exit("WORK_BUCKET 未設定")
    tag = a.coin.replace(":", "_")

    inv = pl.read_parquet(ROOT / "data" / "s3_inventory.parquet")
    sel = (pl.col("coin") == a.coin) & (pl.col("layer") == "l1")
    keys = inv.filter(sel & (pl.col("leaf") == "part-000.parquet")).sort("dt").select("dt", "key").rows()
    done = set(inv.filter(sel & (pl.col("leaf") == "_SUCCESS"))["dt"].to_list())
    keys = [(d, k) for d, k in keys if d in done]
    if a.days:
        keys = keys[: a.days]
    print(f"[fetch] {len(keys)} 日分", file=sys.stderr)

    sess = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "hl-artemis-ro"))
    cr = sess.get_credentials().get_frozen_credentials()
    fs = pafs.S3FileSystem(
        access_key=cr.access_key, secret_key=cr.secret_key, session_token=cr.token, region="us-east-1"
    )

    # ★1 本にまとめると 99 日で 20GB 超になりメモリが尽きる(実際に 10 日目で落ちた)。
    #   日ごとに書き、既にある日は飛ばす(再開可能)。
    outdir = ROOT / "data" / f"l1_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(d_, k) for d_, k in keys if not (outdir / f"dt={d_}.parquet").exists()]
    print(f"[fetch] 未取得 {len(todo)} 日", file=sys.stderr)

    def one(item):
        dt, key = item
        t = pq.ParquetFile(fs.open_input_file(f"{bucket}/{key}")).read(columns=COLS)
        # polars へ変換せず arrow のまま書く。変換すると全列がもう 1 部複製され、
        # 並行して重い処理が走っている環境ではメモリを使い切る(実際に 2 度落ちた)。
        # 元ファイルは ts 昇順なので並べ替えもしない(読む側で並べ替える)。
        pq.write_table(t, outdir / f"dt={dt}.parquet", compression="zstd")
        nb = t.nbytes
        del t
        return dt, nb

    nbytes = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, (dt, nb) in enumerate(ex.map(one, todo), 1):
            nbytes += nb
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  展開後 {nbytes/1e9:.2f} GB", file=sys.stderr)
    n = len(list(outdir.glob("dt=*.parquet")))
    print(f"[fetch] {n} 日分 -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
