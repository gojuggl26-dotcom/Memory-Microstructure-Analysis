#!/usr/bin/env bash
# DEEP_ARCHIVE にある bbo の復元を仕掛ける(標準取り出し・約 12 時間)。
#   bash scripts/restore_bbo.sh
# 対象 1,131 オブジェクト / 2.87 GB。
# 費用: 取り出し $0.02/GB ≈ $0.06 + リクエスト $0.10/1000 ≈ $0.11 = 約 $0.17。
# 復元後 7 日間ダウンロードできる(一時保管料は $0.02 未満)。
set -uo pipefail
set -a; . ~/.hlpipe.env; set +a
REQ='{"Days":7,"GlacierJobParameters":{"Tier":"Standard"}}'
n=0; ok=0; skip=0
while IFS= read -r key; do
  n=$((n+1))
  if aws s3api restore-object --bucket "$WORK_BUCKET" --key "$key" \
       --restore-request "$REQ" >/dev/null 2>&1; then ok=$((ok+1));
  else skip=$((skip+1)); fi
  if [ $((n % 100)) -eq 0 ]; then echo "  $n 件 (要求 $ok / 既存・失敗 $skip)"; fi
done < data/_bbo_keys.txt
echo "復元要求 $n 件 (受理 $ok / それ以外 $skip)"
