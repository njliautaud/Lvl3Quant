#!/bin/bash
# Bulk sync MBO NPZ files from Razer → Jupiter
# Designed to fill the May-June 2026 data gap
# Run manually when Razer comes online (Monday morning)

RAZER_HOST="claude@razer"
RAZER_DIR="C:/Users/claude/Lvl3Quant/data/processed/mbo_events/"
JUPITER_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_events/"
LOG="/home/jupiter/Lvl3Quant/logs/razer_bulk_sync.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') === BULK SYNC START ===" >> "$LOG"

# Check Razer is reachable
if ! sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    "${RAZER_HOST}" "echo ok" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Razer unreachable" >> "$LOG"
    echo "Razer unreachable — try again later"
    exit 1
fi

# List remote files
echo "$(date '+%Y-%m-%d %H:%M:%S') Listing remote files..." >> "$LOG"
REMOTE_FILES=$(sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    "${RAZER_HOST}" "dir /b \"${RAZER_DIR}*.npz\" 2>nul" 2>/dev/null)

if [ -z "$REMOTE_FILES" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') No remote NPZ files found (or Razer offline)" >> "$LOG"
    echo "No remote NPZ files found"
    exit 1
fi

# Count what we need
TOTAL=0
SYNCED=0
SKIPPED=0

for FILE in $REMOTE_FILES; do
    FILE=$(echo "$FILE" | tr -d '\r')  # Remove Windows line endings
    TOTAL=$((TOTAL + 1))

    # Skip if we already have it locally
    if [ -f "${JUPITER_DIR}${FILE}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Download
    echo "$(date '+%Y-%m-%d %H:%M:%S') Syncing ${FILE}..." >> "$LOG"
    sshpass -p "${CLUSTER_SSH_PASSWORD}" scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
        "${RAZER_HOST}:${RAZER_DIR}${FILE}" "${JUPITER_DIR}${FILE}.tmp" 2>/dev/null

    if [ $? -eq 0 ] && [ -f "${JUPITER_DIR}${FILE}.tmp" ]; then
        SIZE=$(stat -c%s "${JUPITER_DIR}${FILE}.tmp")
        if [ "$SIZE" -gt 1000 ]; then
            mv "${JUPITER_DIR}${FILE}.tmp" "${JUPITER_DIR}${FILE}"
            echo "$(date '+%Y-%m-%d %H:%M:%S')   OK: ${FILE} (${SIZE} bytes)" >> "$LOG"
            SYNCED=$((SYNCED + 1))
        else
            rm -f "${JUPITER_DIR}${FILE}.tmp"
            echo "$(date '+%Y-%m-%d %H:%M:%S')   SKIP: ${FILE} too small (${SIZE} bytes)" >> "$LOG"
        fi
    else
        rm -f "${JUPITER_DIR}${FILE}.tmp"
        echo "$(date '+%Y-%m-%d %H:%M:%S')   FAIL: ${FILE}" >> "$LOG"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') === BULK SYNC DONE ===" >> "$LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') Remote: ${TOTAL}, Synced: ${SYNCED}, Already had: ${SKIPPED}" >> "$LOG"
echo "Remote: ${TOTAL} files, Synced: ${SYNCED} new, Already had: ${SKIPPED}"
