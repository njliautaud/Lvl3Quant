#!/usr/bin/env bash
# transfer_data.sh — MBO data persistence and cross-machine sync
# ================================================================
# Copies MBO data to permanent storage on Jupiter and SCPs to Saturn.
# NEVER deletes source files.
#
# Usage:
#   ./transfer_data.sh                     # Copy IS MBO to permanent storage
#   ./transfer_data.sh --oot               # Copy OOT MBO to permanent storage
#   ./transfer_data.sh --to-saturn         # Sync to Saturn
#   ./transfer_data.sh --oot --to-saturn   # Copy OOT + sync to Saturn
#   ./transfer_data.sh --verify            # Verify counts match across machines
#   ./transfer_data.sh --all               # Full sync: IS + OOT + Saturn + verify

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
LVL3_ROOT="${HOME}/Lvl3Quant"

# Source directories (where MBO data currently lives)
MBO_IS_SRC="${LVL3_ROOT}/mbo"
MBO_OOT_SRC="${LVL3_ROOT}/mbo_oot"

# Permanent storage directories on Jupiter
MBO_RAW_DIR="${LVL3_ROOT}/data/raw/mbo"
MBO_IS_DST="${MBO_RAW_DIR}/is"      # In-sample (Aug-Nov 2025)
MBO_OOT_DST="${MBO_RAW_DIR}/oot"    # Out-of-time (Dec 2025 - Mar 2026)

# Saturn config
SATURN_USER="saturn"
SATURN_HOST="saturn"
SATURN_MBO_DIR="/home/saturn/Lvl3Quant/data/raw/mbo"

# Prediction directories to sync
CNN_PRED_DIR="${LVL3_ROOT}/data/processed/cnn_sim_predictions"
GNN_PRED_DIR="${LVL3_ROOT}/data/processed/gnn_sim_predictions"
CNN_OOT_PRED_DIR="${LVL3_ROOT}/data/processed/cnn_oot_predictions"
GNN_OOT_PRED_DIR="${LVL3_ROOT}/data/processed/gnn_oot_predictions"

# ─── Helper Functions ─────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_files() {
    local dir="$1"
    local pattern="${2:-*.dbn*}"
    if [ -d "$dir" ]; then
        find "$dir" -name "$pattern" -type f | wc -l
    else
        echo "0"
    fi
}

copy_mbo() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [ ! -d "$src" ]; then
        log "SKIP: Source directory does not exist: $src"
        return 1
    fi

    local src_count
    src_count=$(count_files "$src")
    if [ "$src_count" -eq 0 ]; then
        log "SKIP: No MBO files found in $src"
        return 1
    fi

    mkdir -p "$dst"
    log "Copying ${label} MBO: $src -> $dst ($src_count files)"

    # Use rsync for efficient incremental copy (skip existing, preserve timestamps)
    if command -v rsync &>/dev/null; then
        rsync -av --progress --ignore-existing "$src/" "$dst/"
    else
        # Fallback: cp with no-clobber
        cp -nv "$src"/*.dbn* "$dst/" 2>/dev/null || true
    fi

    local dst_count
    dst_count=$(count_files "$dst")
    log "  Source: $src_count files, Destination: $dst_count files"

    if [ "$dst_count" -lt "$src_count" ]; then
        log "  WARNING: Destination has fewer files than source!"
        return 1
    fi

    log "  OK: ${label} MBO copy complete"
    return 0
}

sync_to_saturn() {
    local src="$1"
    local remote_dst="$2"
    local label="$3"

    if [ ! -d "$src" ]; then
        log "SKIP: Source directory does not exist: $src"
        return 1
    fi

    local src_count
    src_count=$(count_files "$src")
    if [ "$src_count" -eq 0 ]; then
        log "SKIP: No MBO files in $src"
        return 1
    fi

    log "Syncing ${label} to Saturn: $src -> ${SATURN_USER}@${SATURN_HOST}:${remote_dst}"

    # Create remote directory
    ssh "${SATURN_USER}@${SATURN_HOST}" "mkdir -p ${remote_dst}" 2>/dev/null || {
        log "ERROR: Cannot SSH to Saturn (${SATURN_HOST}). Is it reachable?"
        return 1
    }

    # Sync with rsync over SSH
    if command -v rsync &>/dev/null; then
        rsync -avz --progress --ignore-existing \
            "$src/" "${SATURN_USER}@${SATURN_HOST}:${remote_dst}/"
    else
        scp -r "$src"/*.dbn* "${SATURN_USER}@${SATURN_HOST}:${remote_dst}/" 2>/dev/null || true
    fi

    # Verify count on Saturn
    local saturn_count
    saturn_count=$(ssh "${SATURN_USER}@${SATURN_HOST}" \
        "find ${remote_dst} -name '*.dbn*' -type f | wc -l" 2>/dev/null || echo "?")

    log "  Local: $src_count files, Saturn: $saturn_count files"

    if [ "$saturn_count" != "?" ] && [ "$saturn_count" -lt "$src_count" ]; then
        log "  WARNING: Saturn has fewer files than local!"
        return 1
    fi

    log "  OK: ${label} sync to Saturn complete"
    return 0
}

verify_counts() {
    log ""
    log "=== FILE COUNT VERIFICATION ==="
    log ""

    local all_ok=true

    # Local directories
    for dir_label in \
        "MBO IS src:${MBO_IS_SRC}" \
        "MBO IS dst:${MBO_IS_DST}" \
        "MBO OOT src:${MBO_OOT_SRC}" \
        "MBO OOT dst:${MBO_OOT_DST}" \
        "CNN IS preds:${CNN_PRED_DIR}" \
        "CNN OOT preds:${CNN_OOT_PRED_DIR}" \
        "GNN IS preds:${GNN_PRED_DIR}" \
        "GNN OOT preds:${GNN_OOT_PRED_DIR}"
    do
        label="${dir_label%%:*}"
        dir="${dir_label#*:}"
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f | wc -l)
            log "  ${label}: ${count} files  ($dir)"
        else
            log "  ${label}: MISSING  ($dir)"
        fi
    done

    # Saturn directories
    log ""
    log "  --- Saturn (${SATURN_HOST}) ---"
    if ssh "${SATURN_USER}@${SATURN_HOST}" "echo ok" &>/dev/null; then
        for remote_dir in \
            "${SATURN_MBO_DIR}/is" \
            "${SATURN_MBO_DIR}/oot"
        do
            saturn_count=$(ssh "${SATURN_USER}@${SATURN_HOST}" \
                "[ -d ${remote_dir} ] && find ${remote_dir} -type f | wc -l || echo MISSING" 2>/dev/null)
            log "  Saturn ${remote_dir##*/}: ${saturn_count} files  ($remote_dir)"
        done
    else
        log "  Saturn: UNREACHABLE"
    fi

    log ""
    log "=== Verification complete ==="
}

# ─── Main ─────────────────────────────────────────────────────────────────

DO_IS=false
DO_OOT=false
DO_SATURN=false
DO_VERIFY=false
DO_PREDS=false

for arg in "$@"; do
    case "$arg" in
        --is)       DO_IS=true ;;
        --oot)      DO_OOT=true ;;
        --to-saturn) DO_SATURN=true ;;
        --verify)   DO_VERIFY=true ;;
        --preds)    DO_PREDS=true ;;
        --all)      DO_IS=true; DO_OOT=true; DO_SATURN=true; DO_VERIFY=true; DO_PREDS=true ;;
        --help|-h)
            echo "Usage: $0 [--is] [--oot] [--to-saturn] [--verify] [--preds] [--all]"
            echo ""
            echo "  --is          Copy IS MBO data to permanent storage"
            echo "  --oot         Copy OOT MBO data to permanent storage"
            echo "  --to-saturn   Sync permanent storage to Saturn"
            echo "  --verify      Verify file counts across machines"
            echo "  --preds       Also sync prediction files"
            echo "  --all         Do everything"
            echo ""
            echo "Default (no args): copy IS MBO to permanent storage"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --help)"
            exit 1
            ;;
    esac
done

# Default: just copy IS
if ! $DO_IS && ! $DO_OOT && ! $DO_SATURN && ! $DO_VERIFY && ! $DO_PREDS; then
    DO_IS=true
fi

log "=== MBO Data Transfer ==="
log "  LVL3_ROOT: ${LVL3_ROOT}"
log ""

if $DO_IS; then
    copy_mbo "$MBO_IS_SRC" "$MBO_IS_DST" "IS"
fi

if $DO_OOT; then
    copy_mbo "$MBO_OOT_SRC" "$MBO_OOT_DST" "OOT"
fi

if $DO_PREDS; then
    log ""
    log "=== Syncing Prediction Files ==="
    for pred_dir_label in \
        "CNN IS:${CNN_PRED_DIR}" \
        "CNN OOT:${CNN_OOT_PRED_DIR}" \
        "GNN IS:${GNN_PRED_DIR}" \
        "GNN OOT:${GNN_OOT_PRED_DIR}"
    do
        label="${pred_dir_label%%:*}"
        dir="${pred_dir_label#*:}"
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f | wc -l)
            log "  ${label}: ${count} prediction files"
        else
            log "  ${label}: directory not found ($dir)"
        fi
    done
fi

if $DO_SATURN; then
    log ""
    log "=== Syncing to Saturn ==="
    if [ -d "$MBO_IS_DST" ]; then
        sync_to_saturn "$MBO_IS_DST" "${SATURN_MBO_DIR}/is" "IS MBO"
    fi
    if [ -d "$MBO_OOT_DST" ]; then
        sync_to_saturn "$MBO_OOT_DST" "${SATURN_MBO_DIR}/oot" "OOT MBO"
    fi
fi

if $DO_VERIFY; then
    verify_counts
fi

log ""
log "Done."
