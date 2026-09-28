"""Quick launcher to test train_event_hawkes.py with visible errors."""
import sys, os, traceback

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=== Hawkes Test Launch ===", flush=True)

os.environ["MLFLOW_TRACKING_URI"] = "http://neptune-win:5000"
os.environ["HAWKES_HIDDEN_DIM"] = "128"
os.environ["HAWKES_N_LAYERS"] = "2"
os.environ["EVENT_WINDOW_SIZE"] = "500"
os.environ["EVENT_BATCH_SIZE"] = "128"
os.environ["EVENT_N_FOLDS"] = "5"

sys.argv = [
    "train_event_hawkes.py",
    "--skip-transfer",
    "--output-dir", "results/event_hawkes",
    "--n-folds", "5",
    "--device", "cuda",
]

try:
    print("Running train_event_hawkes.py ...", flush=True)
    exec(open("train_event_hawkes.py").read())
except Exception:
    traceback.print_exc()
    sys.exit(1)
