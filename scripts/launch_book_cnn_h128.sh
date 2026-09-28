#!/bin/bash
cd /home/nick/Lvl3Quant
export MLFLOW_TRACKING_URI='http://jupiter:5000'
mkdir -p logs scripts output
exec /home/nick/training-env/bin/python alpha_discovery/deep_models/train_v2_branch_book_cnn.py   --depth 10 --folds 5,6,7,8,9 --epochs 12 --batch-size 128 --hidden 128   --lr 3e-4 --max-train-days 20 --train-stride 1000 --workers 4   --output-dir /home/nick/Lvl3Quant/output/v2_book_cnn_d10_h128_e12
