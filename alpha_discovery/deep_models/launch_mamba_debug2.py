"""Debug launcher with monkey-patched main"""
import os
import sys

print("=== LAUNCHER STARTING ===", flush=True)

# Set env vars
os.environ["EVENT_WINDOW_SIZE"] = "500"
os.environ["EVENT_STRIDE"] = "250"
os.environ["EVENT_BATCH_SIZE"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

sys.argv = [
    "train_event_mamba.py",
    "--n-folds", "2",
    "--device", "cuda",
    "--skip-transfer",
    "--output-dir", "/tmp/mamba_quick_test",
]

print("=== IMPORTING ===", flush=True)
from train_event_mamba import *

print("=== PATCHING main() ===", flush=True)

# Monkey-patch to add debug output
original_main = main

def debug_main():
    print(">>> INSIDE main() - line 1", flush=True)
    args = parse_args()
    print(f">>> args parsed: n_folds={args.n_folds}", flush=True)
    print(f">>> About to call logger.info", flush=True)
    try:
        logger.info("=" * 60)
        print(f">>> logger.info worked!", flush=True)
    except Exception as e:
        print(f">>> logger.info FAILED: {e}", flush=True)
    
    # Call original
    return original_main()

print("=== CALLING debug_main() ===", flush=True)
debug_main()
print("=== COMPLETED ===", flush=True)
