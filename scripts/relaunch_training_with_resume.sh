#!/bin/bash
# relaunch_training_with_resume.sh — HC #487 R1
#
# Kill-and-resume protocol for CNN-Mamba training. Picks up the most recent
# fold_NN_intra_ckpt.pt and relaunches with --resume-from-intra-ckpt so we
# never lose hours of training to a fresh restart.
#
# Usage:
#   ./relaunch_training_with_resume.sh <node> <output_dir> <fold_idx> <extra_args>
#
# Example:
#   ./relaunch_training_with_resume.sh neptune \
#       /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2 \
#       0 \
#       --warmstart-ckpt /home/nick/Lvl3Quant/output/cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt
#
# Protocol:
#   1. SSH to <node>, locate fold_NN_intra_ckpt.pt in <output_dir>
#   2. If exists and not corrupt → relaunch trainer with --resume-from-intra-ckpt
#   3. If absent → fall back to fresh start (warn loud, log to SESSION_STATE)
#   4. VERIFY first 60s of log shows "RESUMED from intra_ckpt"; if not → kill + fresh
#
# This script is the operational implementation of HC #487 R1. It is meant
# to be called AFTER a kill (divergence, OOM, etc) and NEVER manually
# replaced with a fresh `python train_cnn_mamba_v3_2.py` invocation.

set -euo pipefail

NODE="${1:?usage: relaunch_training_with_resume.sh <node> <output_dir> <fold_idx> [extra_args...]}"
OUTPUT_DIR="${2:?missing output_dir}"
FOLD_IDX="${3:?missing fold_idx}"
shift 3
EXTRA_ARGS="$*"

FOLD_PAD=$(printf "%02d" "$FOLD_IDX")
INTRA_CKPT="$OUTPUT_DIR/fold_${FOLD_PAD}_intra_ckpt.pt"

NODE_HOST=""
NODE_USER=""
TRAINING_VENV=""
TRAINER_PATH=""
case "$NODE" in
  neptune)
    NODE_HOST="neptune"
    NODE_USER="nick"
    TRAINING_VENV="/home/nick/training-env/bin/python"
    TRAINER_PATH="alpha_discovery/deep_models/train_cnn_mamba_v3_2.py"
    REPO_ROOT="/home/nick/Lvl3Quant"
    ;;
  razer)
    NODE_HOST="razer"
    NODE_USER="claude"
    TRAINING_VENV="C:/Python311/python.exe"
    TRAINER_PATH="alpha_discovery/deep_models/train_cnn_mamba_v3_2.py"
    REPO_ROOT="C:/Users/claude/Lvl3Quant"
    ;;
  jupiter)
    NODE_HOST="localhost"
    NODE_USER="jupiter"
    TRAINING_VENV="/usr/bin/python3"
    TRAINER_PATH="alpha_discovery/deep_models/train_cnn_mamba_v3_2.py"
    REPO_ROOT="/home/jupiter/Lvl3Quant"
    ;;
  *)
    echo "[relaunch] ERROR unknown node: $NODE (expected neptune|razer|jupiter)"
    exit 2
    ;;
esac

echo "[relaunch] node=$NODE host=$NODE_USER@$NODE_HOST"
echo "[relaunch] output_dir=$OUTPUT_DIR fold=$FOLD_PAD"
echo "[relaunch] expected intra_ckpt=$INTRA_CKPT"

# Step 1: probe for existing intra_ckpt on remote
PROBE_CMD="if [ -f '$INTRA_CKPT' ]; then stat -c '%s %y' '$INTRA_CKPT'; else echo MISSING; fi"
if [ "$NODE" = "jupiter" ]; then
    PROBE_OUT=$(bash -c "$PROBE_CMD" 2>&1 || true)
else
    PROBE_OUT=$(ssh -o ConnectTimeout=15 "$NODE_USER@$NODE_HOST" "$PROBE_CMD" 2>&1 || echo "SSH_FAIL")
fi
echo "[relaunch] probe: $PROBE_OUT"

RESUME_FLAG=""
if [[ "$PROBE_OUT" =~ ^MISSING|^SSH_FAIL ]]; then
    echo "[relaunch] WARN no intra_ckpt found — falling back to fresh start"
    echo "[relaunch] WARN $(date -u +%FT%TZ) FRESH_FALLBACK node=$NODE dir=$OUTPUT_DIR" \
        >> /home/jupiter/Lvl3Quant/logs/relaunch_resume.log
else
    RESUME_FLAG="--resume-from-intra-ckpt $INTRA_CKPT"
    echo "[relaunch] OK will resume from intra_ckpt"
fi

# Step 2: build trainer invocation
LAUNCH_CMD="$TRAINING_VENV -u $TRAINER_PATH $EXTRA_ARGS $RESUME_FLAG"
LOG_PATH="$OUTPUT_DIR/relaunch_$(date +%Y%m%d_%H%M%S).log"
REMOTE_BASH="cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT:\${PYTHONPATH:-} && nohup $LAUNCH_CMD > $LOG_PATH 2>&1 < /dev/null & echo PID=\$!"

echo "[relaunch] command:"
echo "  $LAUNCH_CMD"
echo "[relaunch] log: $LOG_PATH"

# Step 3: dispatch
if [ "$NODE" = "jupiter" ]; then
    eval "$REMOTE_BASH"
else
    ssh -o ConnectTimeout=15 "$NODE_USER@$NODE_HOST" "bash -lc \"$REMOTE_BASH\""
fi

# Step 4: verify resume actually happened (only if we requested it)
if [ -n "$RESUME_FLAG" ]; then
    echo "[relaunch] waiting 60s then checking log for RESUMED marker..."
    sleep 60
    if [ "$NODE" = "jupiter" ]; then
        VERIFY=$(grep -c "RESUMED from intra_ckpt" "$LOG_PATH" 2>/dev/null || echo 0)
    else
        VERIFY=$(ssh -o ConnectTimeout=15 "$NODE_USER@$NODE_HOST" "grep -c 'RESUMED from intra_ckpt' '$LOG_PATH' 2>/dev/null || echo 0")
    fi
    if [ "$VERIFY" -ge 1 ]; then
        echo "[relaunch] ✓ verified: log shows 'RESUMED from intra_ckpt'"
    else
        echo "[relaunch] ✗ NO resume marker in log after 60s — investigate"
        echo "[relaunch] FAIL $(date -u +%FT%TZ) node=$NODE dir=$OUTPUT_DIR no_resume_marker" \
            >> /home/jupiter/Lvl3Quant/logs/relaunch_resume.log
        exit 3
    fi
fi

echo "[relaunch] done"
