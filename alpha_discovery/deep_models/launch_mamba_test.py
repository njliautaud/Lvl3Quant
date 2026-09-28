"""Quick Mamba test - 2 folds"""
import os
import sys

print("=== LAUNCHER STARTING ===", flush=True)

# Set env vars for small memory footprint
os.environ["EVENT_WINDOW_SIZE"] = "500"
os.environ["EVENT_STRIDE"] = "250"
os.environ["EVENT_BATCH_SIZE"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

print("=== ENV VARS SET ===", flush=True)

# Override sys.argv
sys.argv = [
    "train_event_mamba.py",
    "--n-folds", "2",
    "--device", "cuda",
    "--skip-transfer",
    "--output-dir", "/tmp/mamba_quick_test",
]

print("=== IMPORTING train_event_mamba ===", flush=True)

# Import and run
from train_event_mamba import main

print("=== CALLING main() ===", flush=True)
main()
print("=== COMPLETED ===", flush=True)
