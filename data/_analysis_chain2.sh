#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
while ! grep -q 'analysis done' data/_analysis_chain.log 2>/dev/null; do sleep 30; done
.venv/Scripts/python.exe scripts/analyze_wallet_orders.py --coin xyz:INTC
echo "=== wallet done $(date) ==="
