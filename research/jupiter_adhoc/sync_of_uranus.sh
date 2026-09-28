#!/bin/bash
SRC=/home/jupiter/Lvl3Quant/data/processed/orderflow_features
LOG=/home/jupiter/sync_of_uranus.log
SENT=0; FAILED=0
echo "$(date) START" > $LOG
for f in $SRC/*.npz $SRC/*.json; do
  fname=$(basename $f)
  SSHPASS="${CLUSTER_SSH_PASSWORD}" sshpass -e scp -o StrictHostKeyChecking=no $f nick@winnode:"C:/Users/nick/Lvl3Quant/data/processed/orderflow_features/$fname" 2>>$LOG
  if [ $? -eq 0 ]; then SENT=$((SENT+1)); echo "$(date) OK[$SENT] $fname" >> $LOG
  else FAILED=$((FAILED+1)); echo "$(date) FAIL[$FAILED] $fname" >> $LOG; fi
done
echo "$(date) SYNC_COMPLETE sent=$SENT failed=$FAILED" >> $LOG
