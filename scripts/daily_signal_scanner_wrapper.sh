#!/bin/bash
# Daily Signal Scanner Wrapper
# Runs the high-confidence signal scanner and writes alerts for autonomy_inject pickup.
#
# Cron example (run at 3:30 PM ET on weekdays):
#   30 15 * * 1-5 /home/jupiter/Lvl3Quant/scripts/daily_signal_scanner_wrapper.sh
#
# Or via PM2:
#   pm2 start /home/jupiter/Lvl3Quant/scripts/daily_signal_scanner_wrapper.sh --name daily-scanner --cron "30 19 * * 1-5" --no-autorestart

set -e

BASE="/home/jupiter/Lvl3Quant"
SCANNER="${BASE}/scripts/growth_research/daily_signal_scanner.py"
STATE_DIR="${BASE}/state"
LOG_DIR="${BASE}/output/growth_research/scanner_logs"
ALERT_FILE="${STATE_DIR}/high_confidence_alert.txt"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

DATE=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/scan_${DATE}.log"

echo "[$(date)] Starting daily signal scan..." | tee -a "${LOG_FILE}"

# Run the scanner
cd "${BASE}"
python3 "${SCANNER}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] Scanner FAILED with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
    echo "SCANNER_FAILURE: Daily signal scanner failed at $(date). Check ${LOG_FILE}" > "${ALERT_FILE}"
    exit ${EXIT_CODE}
fi

# Check for high-confidence setups
if [ -f "${ALERT_FILE}" ]; then
    echo "[$(date)] HIGH CONFIDENCE SETUP FOUND - alert written to ${ALERT_FILE}" | tee -a "${LOG_FILE}"
    cat "${ALERT_FILE}" | tee -a "${LOG_FILE}"
else
    echo "[$(date)] No high-confidence setups today." | tee -a "${LOG_FILE}"
fi

echo "[$(date)] Daily scan complete." | tee -a "${LOG_FILE}"
