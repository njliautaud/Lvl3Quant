#!/usr/bin/env bash
# start_mbo_recorder.sh — Launch the MBO recorder pinned to dedicated CPU cores.
#
# Per DIRECTIVES.md item 5a: cores 0-1 are reserved on Jupiter for the live
# Rithmic MBO recorder so it never gets starved when training/backtest jobs
# spike CPU usage. Do NOT change the cpuset without coordinating with the user.
#
# This script is the canonical PM2 entrypoint for the `mbo-recorder` process.
# It wraps the unmodified mbo_recorder.py with `taskset` and the protobuf
# Python-impl env var (workaround for descriptor incompatibility on Jupiter).

set -euo pipefail

CORES="0-1"
SCRIPT_DIR="/home/jupiter/Lvl3Quant/live_trading_linux"
RECORDER="${SCRIPT_DIR}/mbo_recorder.py"

cd "${SCRIPT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

exec /usr/bin/taskset -c "${CORES}" /usr/bin/python3 "${RECORDER}" \
    --symbol ESM6 \
    --exchange CME \
    --flush-minutes 5
