#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for C in AMD KIOXIA SKHX SMSN SNDK MU INTC; do
  echo "=== build xyz:$C $(date +%H:%M:%S) ==="
  $PY scripts/build_xfeat.py --coin xyz:$C 2>&1 | tail -1
done
echo "=== BUILD DONE $(date +%H:%M:%S) ==="
