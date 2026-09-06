#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
# 板の出どころ比較(旧コード同士)が終わってから、MU を新コードで作り直す
while [ ! -f data/featlib_bbo_effect_xyz_MU.csv ]; do sleep 20; done
sleep 15
echo "=== rebuild MU featlib with fixed code $(date) ==="
.venv/Scripts/python.exe scripts/build_featlib.py --coin xyz:MU --days 21
echo "=== done $(date) ==="
