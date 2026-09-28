#!/usr/bin/env bash
# deploy_to_razer.sh — Deploy MFE/MAE predictor to Razer for GPU training
# =========================================================================
# IMPORTANT: Run this from Jupiter AFTER verifying the script runs on CPU
# with --max-folds 2 --epochs 5.
#
# Razer: claude@razer | RTX 3070 8GB | Windows 11
# Target: C:\Users\claude\Lvl3Quant\alpha_discovery\execution\
#
# Usage:
#   ./deploy_to_razer.sh              # deploy and launch with default args
#   ./deploy_to_razer.sh --dry-run    # show commands without executing
#   ./deploy_to_razer.sh --epochs 80  # pass custom epochs to training
#
# DO NOT launch on Razer until Jupiter smoke test passes.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAZER_USER="claude"
RAZER_HOST="razer"
RAZER_PASSWORD="${CLUSTER_SSH_PASSWORD:-}"

RAZER_TARGET_DIR="C:/Users/claude/Lvl3Quant/alpha_discovery/execution"
RAZER_OUTPUT_DIR="C:/Users/claude/Lvl3Quant/output/mfe_mae_predictor"
RAZER_LOG_DIR="C:/Users/claude/Lvl3Quant/logs"

SCRIPT_NAME="mfe_mae_predictor.py"
LOCAL_SCRIPT="$(dirname "$(realpath "$0")")/${SCRIPT_NAME}"

# Training args for Razer (GPU, full training)
TRAIN_ARGS="--gpu --epochs 80 --batch-size 2048 --hidden-dim 256 --n-layers 4 --dropout 0.3"

DRY_RUN=0
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --epochs)
            TRAIN_ARGS="--gpu --epochs $2 --batch-size 2048 --hidden-dim 256 --n-layers 4"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

ssh_cmd() {
    # sshpass for non-interactive SSH
    sshpass -p "$RAZER_PASSWORD" ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=30 \
        "${RAZER_USER}@${RAZER_HOST}" "$@"
}

scp_to_razer() {
    local src="$1"
    local dst="$2"
    sshpass -p "$RAZER_PASSWORD" scp -o StrictHostKeyChecking=no \
        "$src" "${RAZER_USER}@${RAZER_HOST}:$dst"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo "MFE/MAE Predictor — Deploy to Razer"
echo "============================================================"
echo "  Script: ${LOCAL_SCRIPT}"
echo "  Target: ${RAZER_USER}@${RAZER_HOST}:${RAZER_TARGET_DIR}"
echo "  Train args: ${TRAIN_ARGS}"
echo ""

if [[ ! -f "$LOCAL_SCRIPT" ]]; then
    echo "ERROR: Script not found at ${LOCAL_SCRIPT}"
    exit 1
fi

if ! command -v sshpass &> /dev/null; then
    echo "ERROR: sshpass not installed. Install with: sudo apt-get install sshpass"
    exit 1
fi

# Check Razer is reachable
echo "Checking Razer connectivity..."
if ! sshpass -p "$RAZER_PASSWORD" ssh -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "${RAZER_USER}@${RAZER_HOST}" "echo OK" 2>/dev/null; then
    echo "ERROR: Cannot reach Razer at ${RAZER_HOST}. Check VPN/Tailscale."
    exit 1
fi
echo "  Razer reachable."

# ---------------------------------------------------------------------------
# Step 1: Create target directories on Razer
# ---------------------------------------------------------------------------
echo ""
echo "Step 1: Creating target directories on Razer..."
run_cmd ssh_cmd "mkdir -p '${RAZER_TARGET_DIR}' '${RAZER_OUTPUT_DIR}' '${RAZER_LOG_DIR}'"

# ---------------------------------------------------------------------------
# Step 2: Copy the training script
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Copying ${SCRIPT_NAME} to Razer..."
run_cmd scp_to_razer "$LOCAL_SCRIPT" "${RAZER_TARGET_DIR}/${SCRIPT_NAME}"
echo "  Copied ${SCRIPT_NAME}"

# ---------------------------------------------------------------------------
# Step 3: Verify required data exists on Razer
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Verifying data availability on Razer..."
run_cmd ssh_cmd "
    if [ -d 'C:/Users/claude/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar' ]; then
        echo '  CNN-Mamba predictions: OK'
    else
        echo '  WARNING: CNN-Mamba predictions not found — run sync-data first'
    fi
    if [ -d 'C:/Users/claude/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/mfe_mae_analysis' ]; then
        echo '  MFE/MAE labels: OK'
    else
        echo '  ERROR: MFE/MAE labels not found — these must be synced from Jupiter'
    fi
"

# ---------------------------------------------------------------------------
# Step 4: Check Python and dependencies on Razer
# ---------------------------------------------------------------------------
echo ""
echo "Step 4: Checking Python dependencies on Razer..."
run_cmd ssh_cmd "
    python --version 2>&1 || python3 --version 2>&1 || echo 'Python not found'
    python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")' 2>&1
    python -c 'import mlflow; print(f\"MLflow {mlflow.__version__}\")' 2>&1 || echo 'MLflow not installed (non-fatal)'
    python -c 'import scipy; print(f\"scipy OK\")' 2>&1 || echo 'scipy not installed — pip install scipy'
"

# ---------------------------------------------------------------------------
# Step 5: Quick smoke test on Razer (CPU, 2 folds, 3 epochs)
# ---------------------------------------------------------------------------
echo ""
echo "Step 5: Running smoke test on Razer (CPU, 2 folds, 5 epochs)..."
SMOKE_LOG="${RAZER_LOG_DIR}/mfe_mae_smoketest_$(date +%Y%m%d_%H%M%S).log"
run_cmd ssh_cmd "
    cd '${RAZER_TARGET_DIR}' &&
    python ${SCRIPT_NAME} --max-folds 2 --epochs 5 --batch-size 512 \
        > '${SMOKE_LOG}' 2>&1 &&
    echo 'Smoke test PASSED' ||
    (echo 'Smoke test FAILED — check log:'; tail -30 '${SMOKE_LOG}'; exit 1)
"

# ---------------------------------------------------------------------------
# Step 6: Launch full GPU training in background
# ---------------------------------------------------------------------------
echo ""
echo "Step 6: Launching full GPU training on Razer..."
TRAINING_LOG="${RAZER_LOG_DIR}/mfe_mae_training_$(date +%Y%m%d_%H%M%S).log"

# Windows: use start /B for background; we use nohup via MSYS bash
LAUNCH_CMD="python '${RAZER_TARGET_DIR}/${SCRIPT_NAME}' ${TRAIN_ARGS} ${EXTRA_ARGS} \
    > '${TRAINING_LOG}' 2>&1 &"

if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Launching training..."
    echo "  Log: ${TRAINING_LOG}"

    # Use schtasks on Windows for persistent background job
    # This survives SSH session disconnect
    TASK_NAME="MfeMaeTraining_$(date +%H%M%S)"
    run_cmd ssh_cmd "
        schtasks /Create /TN '${TASK_NAME}' /TR \
            'python \"${RAZER_TARGET_DIR}/${SCRIPT_NAME}\" ${TRAIN_ARGS} >> \"${TRAINING_LOG}\" 2>&1' \
            /SC ONCE /ST 00:00 /F &&
        schtasks /Run /TN '${TASK_NAME}' &&
        echo 'Training scheduled and launched as: ${TASK_NAME}'
    " 2>/dev/null || {
        # Fallback: nohup via bash (if MSYS/Git Bash is available)
        echo "  schtasks failed, trying nohup fallback..."
        run_cmd ssh_cmd "
            nohup python '${RAZER_TARGET_DIR}/${SCRIPT_NAME}' ${TRAIN_ARGS} \
                > '${TRAINING_LOG}' 2>&1 </dev/null &
            echo \"Training launched, PID: \$!\"
        "
    }
else
    echo "[DRY-RUN] Would launch: python ${SCRIPT_NAME} ${TRAIN_ARGS}"
    echo "[DRY-RUN] Log: ${TRAINING_LOG}"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Script deployed to: ${RAZER_TARGET_DIR}/${SCRIPT_NAME}"
echo "  Training log:       ${TRAINING_LOG}"
echo "  Output dir:         ${RAZER_OUTPUT_DIR}"
echo ""
echo "  Monitor training:"
echo "    ssh ${RAZER_USER}@${RAZER_HOST} 'tail -f ${TRAINING_LOG}'"
echo ""
echo "  Check GPU utilization:"
echo "    ssh ${RAZER_USER}@${RAZER_HOST} 'nvidia-smi'"
echo ""
echo "  Retrieve results when done:"
echo "    mkdir -p /home/jupiter/Lvl3Quant/output/mfe_mae_predictor_razer"
echo "    sshpass -p \"\$CLUSTER_SSH_PASSWORD\" scp -r \\"
echo "      ${RAZER_USER}@${RAZER_HOST}:${RAZER_OUTPUT_DIR}/ \\"
echo "      /home/jupiter/Lvl3Quant/output/mfe_mae_predictor_razer/"
echo ""
echo "  MLflow: http://localhost:5000 → experiment 'mfe_mae_predictor_razer'"
echo "============================================================"
