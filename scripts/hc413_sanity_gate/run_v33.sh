#!/usr/bin/env bash
# HC #413 sanity gate — v3.3 fold_00
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/jupiter/Lvl3Quant"
NPZ="${V33_NPZ:-$ROOT/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz}"
CKPT="${V33_CKPT:-$ROOT/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt}"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUT="${V33_OUT:-$ROOT/output/hc413_sanity_v3_3_${TS}}"

echo "[run_v33] NPZ:  $NPZ"
echo "[run_v33] CKPT: $CKPT"
echo "[run_v33] OUT:  $OUT"

python3 "$HERE/sanity_gate.py" \
    --npz "$NPZ" \
    --ckpt "$CKPT" \
    --model-family v3_3 \
    --output-dir "$OUT"
