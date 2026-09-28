#!/bin/bash
# Launch memory governor as detached daemon. Idempotent.
if pgrep -f memory_governor.py >/dev/null 2>&1; then
    echo "already running pid=$(pgrep -f memory_governor.py)"
    exit 0
fi
cd /home/nick/Lvl3Quant
setsid python3 /home/nick/Lvl3Quant/memory_governor.py </dev/null >>/tmp/memory_governor.out 2>&1 &
disown
sleep 2
if pgrep -f memory_governor.py >/dev/null 2>&1; then
    echo "started pid=$(pgrep -f memory_governor.py)"
else
    echo "FAILED to start"
    exit 1
fi
