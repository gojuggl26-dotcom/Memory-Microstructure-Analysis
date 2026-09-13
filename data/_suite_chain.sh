#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
# KIOXIA のスイートが終わるのを待ってから、残り銘柄を順に回す(メモリ節約のため直列)
while ! grep -q "FAILED\[xyz:KIOXIA\]" data/_suite_KIOXIA.log 2>/dev/null; do sleep 30; done
for C in AMD SKHX SMSN SNDK INTC; do
  sh scripts/run_mu_suite.sh xyz:$C > data/_suite_$C.log 2>&1
done
echo "=== SUITES ALL DONE $(date +%H:%M:%S) ==="
