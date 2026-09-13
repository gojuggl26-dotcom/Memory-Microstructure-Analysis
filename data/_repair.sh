#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
r() { n="$1"; shift; printf "[%s] %s ... " "$(date +%H:%M:%S)" "$n"; if "$@" > "data/_repair_$n.log" 2>&1; then echo ok; else echo NG; fi }

# SKHX: OOM だった oi_volume(メモリが空いた今なら通るはず)→ 依存の variance / plot
r skhx_oi       $PY scripts/build_oi_volume.py --coin xyz:SKHX
r skhx_var      $PY scripts/build_variance.py  --coin xyz:SKHX
r skhx_oi_plot  $PY scripts/plot_oi_volume.py  --coin xyz:SKHX
r skhx_var_ch   $PY scripts/variance_chart.py  --coin xyz:SKHX

# KIOXIA: 99 日決め打ちの assert を直したので variance をやり直し
r kiox_var      $PY scripts/build_variance.py  --coin xyz:KIOXIA
r kiox_var_ch   $PY scripts/variance_chart.py  --coin xyz:KIOXIA

# INTC: 末尾空群の IndexError を直したので markout をやり直し
r intc_markout  $PY scripts/build_markout.py   --coin xyz:INTC
r intc_mkplot   $PY scripts/plot_markout.py    --coin xyz:INTC

# 全銘柄: 3 つの図の前段(vol_signal / resil_daily / acf lag1)と図
for C in AMD KIOXIA SKHX SMSN SNDK INTC; do
  r ${C}_volsig  $PY scripts/build_vol_signal.py   --coin xyz:$C
  r ${C}_volplot $PY scripts/plot_vol.py           --coin xyz:$C
  r ${C}_resan   $PY scripts/analyze_resilience.py --coin xyz:$C
  r ${C}_resplot $PY scripts/plot_resilience.py    --coin xyz:$C
  r ${C}_acf     $PY scripts/build_acf_200ms.py    --coin xyz:$C
  r ${C}_acfplot $PY scripts/plot_acf_200ms.py     --coin xyz:$C
done
echo "REPAIR DONE $(date +%H:%M:%S)"
