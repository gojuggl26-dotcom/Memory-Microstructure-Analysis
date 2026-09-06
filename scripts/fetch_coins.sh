#!/usr/bin/env bash
# WORK_BUCKET から 5 銘柄の fills を E: へ取得する。
#   bash scripts/fetch_coins.sh
#
# ★ bbo(l2/.../bbo/)は DEEP_ARCHIVE に移行済みで直接は取得できない。
#   先に scripts/restore_bbo.sh で復元を仕掛け、完了後に fetch_bbo.sh を回す。
#   fills と l1 は GLACIER_IR なので即時に取得できる。
# ★ バケット名は出力に出さない。C: の空きが少ないので E: に置く。
set -uo pipefail
set -a; . ~/.hlpipe.env; set +a
DEST=/e/Memory-mkt/raw
for c in DRAM KIOXIA SKHX SMSN SNDK; do
  mkdir -p "$DEST/fills/$c"
  aws s3 sync "s3://$WORK_BUCKET/fills/" "$DEST/fills/$c/" \
      --exclude "*" --include "*coin=xyz%3A$c/part-*.parquet" --only-show-errors \
      2>&1 | sed "s|$WORK_BUCKET|<b>|g"
  echo "fills $c 済 $(du -sh "$DEST/fills/$c" | cut -f1)"
done
echo "合計 $(du -sh "$DEST" | cut -f1)"
