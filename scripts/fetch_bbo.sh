#!/usr/bin/env bash
# 復元済みの bbo を E: へ取得する(restore_bbo.sh の完了後に実行)。
set -uo pipefail
set -a; . ~/.hlpipe.env; set +a
DEST=/e/Memory-mkt/raw/bbo
for c in DRAM KIOXIA SKHX SMSN SNDK; do
  if [ "$c" = "DRAM" ]; then src="l2/bbo/"; else src="l2/coin=xyz%3A$c/bbo/"; fi
  mkdir -p "$DEST/$c"
  aws s3 sync "s3://$WORK_BUCKET/$src" "$DEST/$c/" \
      --exclude "*" --include "*part-*.parquet" --force-glacier-transfer \
      --only-show-errors 2>&1 | sed "s|$WORK_BUCKET|<b>|g"
  echo "bbo $c 済 $(du -sh "$DEST/$c" | cut -f1)"
done
echo "bbo 合計 $(du -sh "$DEST" | cut -f1)"
