#!/bin/sh
cd /c/Users/ii562/Downloads/Memory
export PYTHONIOENCODING=utf-8
while [ "$(ls data/featlib_xyz_MU_l1 2>/dev/null | wc -l)" -lt 21 ]; do sleep 20; done
sleep 10
.venv/Scripts/python.exe scripts/featlib_bbo_effect.py --coin xyz:MU
