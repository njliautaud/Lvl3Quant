#!/bin/bash
source /home/nick/miniconda3/bin/activate py311-train
cd /home/nick/Lvl3Quant
python process_missing_mbo.py --force 20260421 20260422 20260423 20260424 20260426 20260427 20260428 20260429 --workers 1 >> output/reprocess_apr_labels.log 2>&1
echo "DONE at Thu Apr 30 10:06:02 AM EDT 2026" >> output/reprocess_apr_labels.log
