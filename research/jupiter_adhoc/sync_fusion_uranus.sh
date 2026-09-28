#!/bin/bash
SRC=/home/jupiter/Lvl3Quant/data/processed/fusion_context
LOG=/home/jupiter/sync_fusion_uranus.log
SENT=0; FAILED=0
echo "START" > $LOG
for f in $SRC/*.npz; do
  fname=$(basename $f)
  SSHPASS="${CLUSTER_SSH_PASSWORD}" sshpass -e scp -o StrictHostKeyChecking=no $f nick@winnode:"C:/Users/nick/Lvl3Quant/data/processed/fusion_context/$fname" 2>>$LOG
  if [ $? -eq 0 ]; then SENT=$((SENT+1)); echo "OK[$SENT] $fname" >> $LOG
  else FAILED=$((FAILED+1)); echo "FAIL $fname" >> $LOG; fi
done
echo "DONE sent=$SENT failed=$FAILED" >> $LOG
