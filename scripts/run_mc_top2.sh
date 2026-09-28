#!/bin/bash
LVL3=/home/jupiter/Lvl3Quant
OOS_DIR=$LVL3/data/processed/oos_wf_fill_sim
OUT_DIR=$LVL3/data/processed/mc_results
mkdir -p $OUT_DIR
while pgrep -f qf_passive_lowvol_jupiter.py > /dev/null 2>&1; do sleep 30; done
echo MC_START
for CONFIG in tp13_prime_chase tp15_h2h; do
  TMPDIR=/tmp/mc_input_${CONFIG}
  rm -rf $TMPDIR && mkdir -p $TMPDIR
  for f in ${OOS_DIR}/*_oos_${CONFIG}.json; do ln -s $f $TMPDIR/$(basename $f); done
  python3 $LVL3/scripts/monte_carlo_validation.py --input $TMPDIR --sims 10000 > ${OUT_DIR}/mc_${CONFIG}.txt 2>&1 &
  echo launched $CONFIG PID=$!
done
wait
echo MC_DONE
