#!/usr/bin/env bash
# HC #413 sanity gate — v3.4.2 60d ep-1 OOT
# Env overrides:
#   V342_NPZ  — path to predictions NPZ (default: Neptune landing path)
#   V342_CKPT — path to .pt checkpoint
#   V342_OUT  — output dir
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/jupiter/Lvl3Quant"

# Neptune landing path — Jupiter accesses via mounted/synced share.
# Glob for the most recent fold_00_ep1_oot_*.npz under v3_4_2_fixedmtl.
DEFAULT_DIR="$ROOT/output/cnn_mamba_v3_4_2_fixedmtl"
DEFAULT_NPZ="$(ls -t "$DEFAULT_DIR"/fold_00_ep1_oot_*.npz 2>/dev/null | head -n 1 || true)"
DEFAULT_CKPT="$(ls -t "$DEFAULT_DIR"/fold_00*ep1*.pt 2>/dev/null | head -n 1 || true)"

NPZ="${V342_NPZ:-$DEFAULT_NPZ}"
CKPT="${V342_CKPT:-$DEFAULT_CKPT}"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUT="${V342_OUT:-$ROOT/output/hc413_sanity_v3_4_2_${TS}}"

if [[ -z "$NPZ" || ! -f "$NPZ" ]]; then
    echo "[run_v342] ERROR: no NPZ found at $DEFAULT_DIR/fold_00_ep1_oot_*.npz" >&2
    echo "[run_v342] Set V342_NPZ to override." >&2
    exit 2
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[run_v342] WARNING: no CKPT found — book_gate check will FAIL." >&2
fi

echo "[run_v342] NPZ:  $NPZ"
echo "[run_v342] CKPT: $CKPT"
echo "[run_v342] OUT:  $OUT"

python3 "$HERE/sanity_gate.py" \
    --npz "$NPZ" \
    ${CKPT:+--ckpt "$CKPT"} \
    --model-family v3_4_2 \
    --output-dir "$OUT"
