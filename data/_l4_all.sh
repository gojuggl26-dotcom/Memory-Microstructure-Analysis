#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
echo "=== post $(date +%H:%M) ==="
$PY scripts/build_l4post.py --coin xyz:INTC || exit 1
echo "=== fit pool $(date +%H:%M) ==="
$PY scripts/fit_l4feat.py --coin xyz:INTC --stage pool || exit 1
echo "=== fit daily $(date +%H:%M) ==="
$PY scripts/fit_l4feat.py --coin xyz:INTC --stage daily || exit 1
echo "=== fit corr $(date +%H:%M) ==="
$PY scripts/fit_l4feat.py --coin xyz:INTC --stage corr || exit 1
echo "=== plots $(date +%H:%M) ==="
$PY scripts/plot_l4feat.py --coin xyz:INTC || exit 1
echo "=== ALL DONE $(date +%H:%M) ==="
