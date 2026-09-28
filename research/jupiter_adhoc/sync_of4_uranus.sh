#!/bin/bash
SRC=/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth
LOG=/home/jupiter/sync_of4_uranus.log
SENT=0; FAILED=0
echo "$(date) START" > $LOG
for f in $SRC/*.npz; do
  fname=$(basename $f)
  SSHPASS="${CLUSTER_SSH_PASSWORD}" sshpass -e scp -o StrictHostKeyChecking=no $f nick@winnode:"C:/Users/nick/Lvl3Quant/data/processed/of4_dom_depth/$fname" 2>>$LOG
  if [ $? -eq 0 ]; then SENT=$((SENT+1)); echo "$(date) OK[$SENT] $fname" >> $LOG
  else FAILED=$((FAILED+1)); echo "$(date) FAIL[$FAILED] $fname" >> $LOG; fi
done
echo "$(date) DONE sent=$SENT failed=$FAILED" >> $LOG
