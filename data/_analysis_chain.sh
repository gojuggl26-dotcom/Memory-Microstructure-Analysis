#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
while ! grep -q 'orderlife done' data/_intc_chain.log 2>/dev/null; do sleep 30; done
while ! grep -q '=== done' data/_mu_rebuild.log 2>/dev/null; do sleep 30; done
echo "=== inputs ready $(date) ==="
$PY scripts/analyze_featlib.py  --coin xyz:INTC && echo "--- analyze INTC ok"
$PY scripts/plot_featlib.py     --coin xyz:INTC && echo "--- plot INTC ok"
$PY scripts/featlib_delta.py    --coin xyz:INTC && echo "--- delta INTC ok"
$PY scripts/analyze_orderlife.py --coin xyz:INTC && echo "--- orderlife INTC ok"
$PY scripts/analyze_featlib.py  --coin xyz:MU && echo "--- analyze MU ok"
$PY scripts/featlib_compare.py  --a xyz:INTC --b xyz:MU && echo "--- compare ok"
echo "=== analysis done $(date) ==="
