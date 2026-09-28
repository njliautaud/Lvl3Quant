#!/bin/bash
# HC #518 R6 raw-MBO migration Neptune -> Razer (resumed 2026-06-10)
R="claude@razer"
LOG=/home/nick/Lvl3Quant/output/mbo_migrate_razer.log
ssh $R "mkdir C:\\Users\\claude\\Lvl3Quant\\data\\raw\\spy_mbo 2>nul" 
for d in mbo spy_mbo; do
  for f in /home/nick/Lvl3Quant/data/raw/$d/*; do
    base=$(basename "$f")
    lsize=$(stat -Lc%s "$f")
    rsize=$(ssh $R "for %A in (C:\\Users\\claude\\Lvl3Quant\\data\\raw\\$d\\$base) do @echo %~zA" 2>/dev/null | tr -d "\r")
    if [ "$rsize" = "$lsize" ]; then echo "SKIP $d/$base" >> $LOG; continue; fi
    echo "COPY $d/$base ($lsize bytes)" >> $LOG
    scp -q "$f" "$R:C:/Users/claude/Lvl3Quant/data/raw/$d/" >> $LOG 2>&1
  done
done
echo "MIGRATION COMPLETE $(date)" >> $LOG
