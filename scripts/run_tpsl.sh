#!/bin/bash
cd /home/jupiter/Lvl3Quant
python3 scripts/tp_sl_sortino_sweep.py 2>&1 | tee data/processed/tp_sl_sortino_sweep/sweep.log
