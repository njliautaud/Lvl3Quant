#!/bin/bash
# Local log cleanup for Neptune - runs via crontab
# Truncates any log over 10MB, cleans /tmp
for dir in /home/nick/Lvl3Quant/logs /home/nick/Lvl3Quant/output/*/logs /home/nick/Lvl3Quant/output; do
    find "$dir" -maxdepth 2 -name "*.log" -size +10M -exec sh -c 'tail -500 "$1" > "$1.tmp" && mv "$1.tmp" "$1"' _ {} \; 2>/dev/null
done
find /tmp -maxdepth 2 -type f -size +100M -delete 2>/dev/null
find /home/nick/Lvl3Quant -maxdepth 4 -name "*.log" -size +50M -exec truncate -s 0 {} \; 2>/dev/null
