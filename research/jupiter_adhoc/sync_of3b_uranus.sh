#!/bin/bash
SRC=/home/jupiter/Lvl3Quant/data/processed/of3_large_order
LOG=/home/jupiter/sync_of3b_uranus.log
SENT=0; FAILED=0; SKIPPED=0
echo "$(date) START" > $LOG
for f in $SRC/*.npz; do
  fname=$(basename $f)
  if grep -q "OK.*$fname" /home/jupiter/sync_of3_uranus.log 2>/dev/null; then
    SKIPPED=$((SKIPPED+1))
    continue
  fi
  SSHPASS="${CLUSTER_SSH_PASSWORD}" sshpass -e scp -o StrictHostKeyChecking=no $f nick@winnode:"C:/Users/nick/Lvl3Quant/data/processed/of3_large_order/$fname" 2>>$LOG
  if [ $? -eq 0 ]; then SENT=$((SENT+1)); echo "$(date) OK[$SENT] $fname" >> $LOG
  else FAILED=$((FAILED+1)); echo "$(date) FAIL[$FAILED] $fname" >> $LOG; fi
done
echo "$(date) DONE sent=$SENT skipped=$SKIPPED failed=$FAILED" >> $LOG
