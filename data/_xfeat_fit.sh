#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for C in AMD KIOXIA SKHX SMSN SNDK MU INTC; do
  echo "=== fit xyz:$C $(date +%H:%M:%S) ==="
  $PY scripts/fit_xfeat.py --coin xyz:$C --stage pool  2>&1 | grep -E "日 /|標本|^  [0-9]+/" | tail -4
  $PY scripts/fit_xfeat.py --coin xyz:$C --stage daily 2>&1 | tail -1
  $PY scripts/plot_xfeat.py --coin xyz:$C 2>&1 | tail -1
done
echo "=== FIT DONE $(date +%H:%M:%S) ==="
