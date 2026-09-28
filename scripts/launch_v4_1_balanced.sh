#!/bin/bash
# V4.1 Multihead with BALANCED loss weights
#
# Problem: v4 directional loss (~11) dominates pressure losses (~0.002-0.02)
# by 500-10000x. Pressure heads are effectively untrained.
#
# Fix: Scale pressure weights to equalize gradient contribution.
# dir_loss ~11, eofi_loss ~0.02 → weight eofi by 11/0.02 ≈ 550
# Rounding to 500 for all pressure heads as a starting point.
#
# DO NOT LAUNCH UNTIL v4 FINISHES (fold 216)
#
# Usage: ssh nick@neptune "bash /home/nick/Lvl3Quant/scripts/launch_v4_1_balanced.sh"

cd /home/nick/Lvl3Quant

python3 scripts/train_v4_multihead.py \
    --data-dir data/processed/mbo_events_smart_v3 \
    --pressure-dir data/processed/smooth_pressure_targets \
    --output-dir output/v4_multihead_balanced_v1 \
    --device cuda \
    --epochs 2 \
    --batch-size 512 \
    --seq-len 100 \
    --stride 500 \
    --lr 1e-4 \
    --head-weights "1.0,500.0,500.0,500.0,500.0" \
    --horizons "1,5,10,30" \
    --d-model 96 \
    --n-layers 4 \
    --d-state 16 \
    --train-days 20 \
    --use-amp \
    2>&1 | tee output/v4_multihead_balanced_v1.log
