#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# VIX Daily Allocator — Cron Wrapper
# ─────────────────────────────────────────────────────────────────────
#
# Runs the VIX allocator in dry-run mode, logging output.
#
# CRITICAL TIMING: Must run during market hours to avoid 1-day VIX lag.
# Recommended: 3:30-3:45 PM ET on weekdays.
#
# Crontab entry (3:30 PM ET = 19:30 UTC, or 20:30 UTC during EST):
#   30 19 * * 1-5 /home/jupiter/Lvl3Quant/scripts/growth_research/vix_allocator_cron_wrapper.sh
#
# During EST (Nov-Mar), use 20:30 UTC instead:
#   30 20 * * 1-5 /home/jupiter/Lvl3Quant/scripts/growth_research/vix_allocator_cron_wrapper.sh
#
# Or use the TZ trick to handle DST automatically:
#   CRON_TZ=America/New_York
#   30 15 * * 1-5 /home/jupiter/Lvl3Quant/scripts/growth_research/vix_allocator_cron_wrapper.sh
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOCATOR="${SCRIPT_DIR}/vix_daily_allocator.py"
LOG_DIR="/home/jupiter/Lvl3Quant/output/growth_research"
LOG_FILE="${LOG_DIR}/vix_allocator_cron_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

# Activate Python environment (adjust if using conda/venv)
# Jupiter uses system Python with yfinance installed
export PATH="/usr/local/bin:/usr/bin:${PATH}"

# Check if market is likely open (basic weekday check — doesn't handle holidays)
DOW=$(date +%u)  # 1=Mon, 7=Sun
if [ "$DOW" -gt 5 ]; then
    echo "$(date): Weekend — skipping VIX allocator run" | tee -a "${LOG_FILE}"
    exit 0
fi

echo "$(date): Starting VIX allocator (dry-run)" | tee "${LOG_FILE}"
echo "─────────────────────────────────────────" | tee -a "${LOG_FILE}"

# Run allocator in dry-run mode, capture output to both log and stdout
python3 "${ALLOCATOR}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "─────────────────────────────────────────" | tee -a "${LOG_FILE}"
echo "$(date): Allocator finished (exit code: ${EXIT_CODE})" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
