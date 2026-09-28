#!/bin/bash
# Launch RL Execution Agent v3 on Neptune (RTX 3090)
# ==================================================
# Trains PPO-based RL execution agent for ES futures.
# Replaces: warm-start walk-forward CNN-Mamba training (run AFTER fold 0 completes).
#
# Prerequisites:
#   - Neptune: conda env py311-train with PyTorch + MLflow
#   - Data: CNN-Mamba v2 OOT predictions must exist on Neptune
#   - MLflow server running on Neptune (port 5000)
#
# Usage:
#   ./launch_rl_agent_neptune.sh               # full walk-forward (default)
#   ./launch_rl_agent_neptune.sh --no-bc       # skip behavioral cloning warmstart
#   ./launch_rl_agent_neptune.sh --eval        # evaluate existing checkpoint
#   ./launch_rl_agent_neptune.sh --dry-run     # check without launching

set -e

# ── Config ────────────────────────────────────────────────────────────────────
NEPTUNE_HOST="nick@neptune"
NEPTUNE_LVL3="/home/nick/Lvl3Quant"
NEPTUNE_CONDA_ENV="py311-train"
NEPTUNE_PYTHON="/home/nick/miniconda3/envs/${NEPTUNE_CONDA_ENV}/bin/python"

JUPITER_EXEC_DIR="/home/jupiter/Lvl3Quant/alpha_discovery/execution"
NEPTUNE_EXEC_DIR="${NEPTUNE_LVL3}/alpha_discovery/execution"

# Data paths (on Neptune)
CNN_PRED_DIR="${NEPTUNE_LVL3}/output/cnn_mamba_v2_smart_v3_mar"
PTST_PRED_DIR="${NEPTUNE_LVL3}/output/patchtst_smart_v3_mar"
OUTPUT_DIR="${NEPTUNE_LVL3}/output/rl_execution_agent"
LOG_FILE="${NEPTUNE_LVL3}/output/rl_execution_agent.log"

# Training config
N_ITERATIONS=200          # PPO iterations per walk-forward window
ROLLOUT_STEPS=4096        # Steps per rollout
DEVICE="cuda"
EXPERIMENT_NAME="rl_execution_agent"

# Parse CLI args
MODE="train"
BC_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --no-bc)     BC_FLAG="--no-bc-warmstart" ;;
        --eval)      MODE="eval" ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "=== RL EXECUTION AGENT v3 — NEPTUNE LAUNCH SCRIPT ==="
echo "Date: $(date)"
echo "Mode: ${MODE}"
echo "Dry run: ${DRY_RUN}"
echo ""

# ── Step 1: Verify Neptune is reachable ────────────────────────────────────────
echo "=== Step 1: Checking Neptune connectivity ==="
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${NEPTUNE_HOST}" "echo 'Neptune OK'" 2>/dev/null; then
    echo "ERROR: Cannot reach Neptune at ${NEPTUNE_HOST}"
    echo "Check VPN/Tailscale connection."
    exit 1
fi
echo "Neptune reachable."
echo ""

# ── Step 2: Check data exists on Neptune ──────────────────────────────────────
echo "=== Step 2: Checking CNN-Mamba predictions on Neptune ==="
CNN_COUNT=$(ssh "${NEPTUNE_HOST}" "ls ${CNN_PRED_DIR}/fold_*_oot_predictions.npz 2>/dev/null | wc -l")
echo "CNN-Mamba v2 fold files on Neptune: ${CNN_COUNT}"

if [ "${CNN_COUNT}" -lt 5 ]; then
    echo "WARNING: Only ${CNN_COUNT} CNN-Mamba fold files found on Neptune."
    echo "         Need at least 65 for full walk-forward (60d train + 5d eval)."
    echo "         Script will train on available folds."
fi

PTST_COUNT=$(ssh "${NEPTUNE_HOST}" "ls ${PTST_PRED_DIR}/fold_*_oot_predictions.npz 2>/dev/null | wc -l")
echo "PatchTST fold files on Neptune: ${PTST_COUNT}"
if [ "${PTST_COUNT}" -lt 1 ]; then
    echo "NOTE: No PatchTST predictions found. Will use zeros for PatchTST features."
fi
echo ""

# ── Step 3: Kill any existing RL agent training on Neptune ─────────────────────
echo "=== Step 3: Stopping any existing RL agent processes on Neptune ==="
ssh "${NEPTUNE_HOST}" "pkill -f 'rl_execution_agent' 2>/dev/null; pkill -f 'rl_exec_agent' 2>/dev/null; echo 'Cleanup done (may show no processes to kill — that is OK)'"
sleep 2
echo ""

# ── Step 4: Sync RL agent script to Neptune ───────────────────────────────────
echo "=== Step 4: Syncing RL execution agent script to Neptune ==="
ssh "${NEPTUNE_HOST}" "mkdir -p ${NEPTUNE_EXEC_DIR}"
scp -o ConnectTimeout=15 \
    "${JUPITER_EXEC_DIR}/rl_execution_agent.py" \
    "${NEPTUNE_HOST}:${NEPTUNE_EXEC_DIR}/"
echo "Script synced."
echo ""

# ── Step 5: Create output directory on Neptune ────────────────────────────────
echo "=== Step 5: Creating output directory on Neptune ==="
ssh "${NEPTUNE_HOST}" "mkdir -p ${OUTPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# ── Step 6: Verify PyTorch + stable-baselines3 availability ──────────────────
echo "=== Step 6: Checking Python environment on Neptune ==="
ssh "${NEPTUNE_HOST}" "
    source /home/nick/miniconda3/etc/profile.d/conda.sh
    conda activate ${NEPTUNE_CONDA_ENV}
    echo 'Python: '\"$\(which python\)\"
    python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'
    python -c 'import mlflow; print(\"MLflow:\", mlflow.__version__)' 2>/dev/null || echo 'MLflow: NOT found (will skip MLflow logging)'
    python -c 'import numpy; print(\"NumPy:\", numpy.__version__)'
    python -c 'import scipy; print(\"SciPy:\", scipy.__version__)' 2>/dev/null || echo 'SciPy: NOT found (PatchTST interpolation disabled)'
    nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo "=== DRY RUN COMPLETE — Not launching ==="
    echo "To launch: run this script without --dry-run"
    exit 0
fi

# ── Step 7: Launch ─────────────────────────────────────────────────────────────
echo "=== Step 7: Launching RL Execution Agent on Neptune ==="
echo "  Mode:        ${MODE}"
echo "  Iterations:  ${N_ITERATIONS} per WF window"
echo "  Rollout:     ${ROLLOUT_STEPS} steps"
echo "  Device:      ${DEVICE}"
echo "  Experiment:  ${EXPERIMENT_NAME}"
echo "  Log:         ${LOG_FILE}"
echo ""

LAUNCH_CMD="${NEPTUNE_PYTHON} -u ${NEPTUNE_EXEC_DIR}/rl_execution_agent.py \
    --mode ${MODE} \
    --cnn-pred-dir ${CNN_PRED_DIR} \
    --ptst-pred-dir ${PTST_PRED_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --n-iterations ${N_ITERATIONS} \
    --rollout-steps ${ROLLOUT_STEPS} \
    --device ${DEVICE} \
    --experiment-name ${EXPERIMENT_NAME} \
    ${BC_FLAG}"

ssh "${NEPTUNE_HOST}" "
    source /home/nick/miniconda3/etc/profile.d/conda.sh
    conda activate ${NEPTUNE_CONDA_ENV}
    nohup ${LAUNCH_CMD} \
        > ${LOG_FILE} 2>&1 &
    echo \"LAUNCHED PID=\$!\"
    echo \"Log: ${LOG_FILE}\"
    disown
"

echo ""
echo "=== LAUNCH COMPLETE ==="
echo ""
echo "Monitor training:"
echo "  ssh ${NEPTUNE_HOST} 'tail -f ${LOG_FILE}'"
echo ""
echo "Check GPU utilization:"
echo "  ssh ${NEPTUNE_HOST} 'nvidia-smi'"
echo ""
echo "Check process:"
echo "  ssh ${NEPTUNE_HOST} 'pgrep -f rl_execution_agent && echo running || echo stopped'"
echo ""
echo "MLflow results (if MLflow running on Neptune):"
echo "  http://neptune:5000  (experiment: ${EXPERIMENT_NAME})"
echo ""
echo "Output files:"
echo "  Best checkpoint: ${OUTPUT_DIR}/best_fold_overall.pt"
echo "  WF results:      ${OUTPUT_DIR}/walkforward_results.json"
echo ""
echo "IMPORTANT: This script was launched in background with nohup."
echo "  Neptune can be disconnected and training will continue."
echo ""
echo "To kill training:"
echo "  ssh ${NEPTUNE_HOST} 'pkill -f rl_execution_agent'"
