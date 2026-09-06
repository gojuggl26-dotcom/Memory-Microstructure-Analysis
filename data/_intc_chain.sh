#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
# 1) fetch が終わるまで待つ
while ! grep -q '日分 ->' data/_fetch_l1_intc.log 2>/dev/null; do sleep 20; done
echo "=== fetch done $(date) ==="
# 2) l1 から板を組み直す
$PY scripts/build_bbo_l1.py --coin xyz:INTC || exit 1
echo "=== bbo done $(date) ==="
# 3) 特徴量ライブラリ
$PY scripts/build_featlib.py --coin xyz:INTC || exit 1
echo "=== featlib done $(date) ==="
# 4) 注文 1 本ごとの記録
$PY scripts/build_orderlife.py --coin xyz:INTC || exit 1
echo "=== orderlife done $(date) ==="
