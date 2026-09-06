#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
. ~/.hlpipe.env
while ! grep -q '日分 ->' data/_fetch_l1_intc.log 2>/dev/null; do sleep 20; done
echo "=== main fetch done, start user fetch $(date) ==="
.venv/Scripts/python.exe scripts/fetch_l1_user.py --coin xyz:INTC
echo "=== user fetch done $(date) ==="
