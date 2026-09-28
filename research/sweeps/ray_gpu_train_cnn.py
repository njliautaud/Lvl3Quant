"""Launch EventCNN1D training on GPU via Ray"""
import ray
import os
import sys

ray.init(address="jupiter:6379")

@ray.remote(num_gpus=1)
def train_event_cnn():
    """Train EventCNN1D on GPU worker (Razer)"""
    os.chdir("/home/jupiter/Lvl3Quant")
    sys.path.insert(0, "/home/jupiter/Lvl3Quant")

    import subprocess
    result = subprocess.run(
        ["python3", "alpha_discovery/deep_models/train_event_cnn_1d.py",
         "--n-folds", "5"],
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

print("Launching EventCNN1D on GPU...")
future = train_event_cnn.remote()
print(f"Ray task submitted: {future}")
print("Training running on Razer GPU. Check MLflow for progress.")
