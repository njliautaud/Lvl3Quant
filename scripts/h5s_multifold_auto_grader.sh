#!/usr/bin/env bash
#
# h5s_multifold_auto_grader.sh
#
# Persistent watcher for the h5s low-LR multifold training (running on Neptune).
# Every cron tick: list fold_NN_oot_predictions.npz on Neptune, for each one not yet
# graded, rsync it to Jupiter and run the canonical FIFO + fill-prob-gated grade.
# Appends a one-line headline row to output/h5s_multifold_AGGREGATE.md per fold.
#
# Idempotent: if no new folds, prints "NO_NEW_FOLDS" and exits 0.
# If at least one new fold graded, prints "GRADED_FOLD={NN}" for each and exits 0.
#
# Per HC #420: user's own quant codebase, full authorization.
#
# Install via cron:
#   */15 * * * * /home/jupiter/Lvl3Quant/scripts/h5s_multifold_auto_grader.sh >> /home/jupiter/Lvl3Quant/logs/h5s_auto_grader.log 2>&1

set -euo pipefail

LVL3_ROOT="/home/jupiter/Lvl3Quant"
NEPTUNE_HOST="nick@neptune"
NEPTUNE_PRED_DIR="/home/nick/Lvl3Quant/output/cnn_mamba_v3_h5s_lowlr_multifold"
LOCAL_PRED_DIR="${LVL3_ROOT}/output/cnn_mamba_v3_h5s_lowlr_multifold"
GRADER_PY="${LVL3_ROOT}/scripts/h5s_multifold_grade_one_fold.py"
LOCK_FILE="/tmp/h5s_multifold_auto_grader.lock"

# Fold 0 has already been graded — skip even if it appears.
SKIP_FOLDS=("00")

mkdir -p "${LOCAL_PRED_DIR}" "${LVL3_ROOT}/logs"

# Prevent overlapping runs
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "$(date -Is) ANOTHER_INSTANCE_RUNNING — skip"
    exit 0
fi

echo "$(date -Is) tick start"

# List fold OOT prediction NPZs on Neptune
REMOTE_LISTING=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${NEPTUNE_HOST}" \
    "ls -1 ${NEPTUNE_PRED_DIR}/fold_*_oot_predictions.npz 2>/dev/null" || true)

if [[ -z "${REMOTE_LISTING}" ]]; then
    echo "$(date -Is) NO_NEW_FOLDS (no remote npz files found)"
    exit 0
fi

ANY_GRADED=0

while IFS= read -r remote_path; do
    [[ -z "${remote_path}" ]] && continue
    fname=$(basename "${remote_path}")
    # Extract NN from "fold_NN_oot_predictions.npz"
    fold_id="${fname#fold_}"
    fold_id="${fold_id%%_*}"

    # Skip already-graded fold-0
    skip=0
    for s in "${SKIP_FOLDS[@]}"; do
        if [[ "${fold_id}" == "${s}" ]]; then
            skip=1
            break
        fi
    done
    if [[ ${skip} -eq 1 ]]; then
        # Make sure fold-0 has aggregate row (one-time append if missing)
        if [[ -f "${LVL3_ROOT}/output/h5s_multifold_AGGREGATE.md" ]]; then
            if ! grep -q "^| 00 " "${LVL3_ROOT}/output/h5s_multifold_AGGREGATE.md"; then
                # Append a fold-0 row from the existing report numbers
                # (Label-FIFO 1s_top10% +3.890 t/trade, canonical -0.296, gated +0.047)
                cat >> "${LVL3_ROOT}/output/h5s_multifold_AGGREGATE.md" << 'EOF'
| 00 | 20260412 | +3.890 | 94 | -0.296 | 25 | +0.047 | 13 |
EOF
            fi
        fi
        continue
    fi

    # Check if grade report already exists
    grade_report="${LVL3_ROOT}/output/h5s_multifold_fold${fold_id}_fifo_REPORT.md"
    if [[ -f "${grade_report}" ]]; then
        # Already graded — skip
        continue
    fi

    echo "$(date -Is) NEW_FOLD detected: fold_${fold_id}"
    local_path="${LOCAL_PRED_DIR}/${fname}"

    # rsync the npz from Neptune
    if ! rsync -q -e "ssh -o ConnectTimeout=15 -o BatchMode=yes" \
        "${NEPTUNE_HOST}:${remote_path}" "${local_path}"; then
        echo "$(date -Is) RSYNC_FAILED for fold_${fold_id} — will retry next tick"
        continue
    fi

    # Run the per-fold grader
    if python3 "${GRADER_PY}" "${fold_id}" "${local_path}"; then
        ANY_GRADED=1
        echo "$(date -Is) GRADED_FOLD=${fold_id}"
    else
        echo "$(date -Is) GRADE_FAILED for fold_${fold_id} (exit $?) — will retry next tick"
        # Remove the partial report so next tick retries
        rm -f "${grade_report}"
    fi
done <<< "${REMOTE_LISTING}"

if [[ ${ANY_GRADED} -eq 0 ]]; then
    echo "$(date -Is) NO_NEW_FOLDS"
fi

exit 0
