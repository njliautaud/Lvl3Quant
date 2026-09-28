#!/bin/bash
cd /home/nick/Lvl3Quant
export PYTHONUNBUFFERED=1
python3 scripts/side_exit_sweep.py > output/side_exit_sweep/sweep.log 2>&1
