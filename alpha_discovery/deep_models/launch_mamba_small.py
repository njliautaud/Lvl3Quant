"""Launcher for Mamba with reduced memory settings to avoid OOM."""
import os
import sys

# Set env vars BEFORE importing the training script
os.environ["EVENT_WINDOW_SIZE"] = "500"
os.environ["EVENT_STRIDE"] = "250"
os.environ["EVENT_BATCH_SIZE"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Override sys.argv
sys.argv = [
    "train_event_mamba.py",
    "--n-folds", "5",
    "--device", "cuda",
    "--skip-transfer",
    "--output-dir", os.path.join(os.path.dirname(__file__), "results", "event_mamba_w500_apr01"),
]

# Import and run
from train_event_mamba import main
main()
