"""2-fold CNN test"""
import os
import sys

# Small memory settings
os.environ["EVENT_WINDOW_SIZE"] = "500"
os.environ["EVENT_STRIDE"] = "250"
os.environ["EVENT_BATCH_SIZE"] = "32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

sys.argv = [
    "train_event_cnn_1d.py",
    "--n-folds", "2",
    "--device", "cuda",
    "--skip-transfer",
    "--output-dir", "/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/cnn_2fold_test_apr18",
]

from train_event_cnn_1d import main
main()
