#!/bin/bash
# Quick AVO engine progress checker
echo "=== AVO Engine Status $(date) ==="

for log in /tmp/avo_*_run.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log | sed 's/avo_//;s/_run//')
    last_step=$(grep -o 'step [0-9]*/[0-9]*' "$log" | tail -1)
    last_verdict=$(grep -o 'ACCEPTED\|REJECTED\|verdict.*' "$log" | tail -1)
    echo "  $name: $last_step | $last_verdict"
done

# Check if PIDs are still running
for pid in $(pgrep -f "avo run"); do
    echo "  avo PID $pid still running"
done
