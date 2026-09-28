#!/bin/bash
# watch_and_process.sh — Watch for download completion, then auto-process
LOG="/home/nick/Lvl3Quant/output/watch_download.log"
exec > >(tee -a "") 2>&1

echo "Thu Apr 30 12:06:57 AM EDT 2026: Watching for GLBX download completion..."
while true; do
    PART=
    ZIP=
    
    if [ -z "" ] && [ -n "" ] && [ -s "" ]; then
        SIZE=
        if [ "" -gt 1000000 ]; then
            echo "Thu Apr 30 12:06:57 AM EDT 2026: Download COMPLETE! ZIP= SIZE="
            echo "Thu Apr 30 12:06:57 AM EDT 2026: Launching processing pipeline..."
            /home/nick/Lvl3Quant/process_new_mbo_data.sh
            echo "Thu Apr 30 12:06:57 AM EDT 2026: Pipeline finished. Exiting watcher."
            exit 0
        fi
    elif [ -n "" ]; then
        SIZE=0
        SIZE_MB=0
        echo "Thu Apr 30 12:06:57 AM EDT 2026: Still downloading... MB"
    fi
    sleep 60
done
