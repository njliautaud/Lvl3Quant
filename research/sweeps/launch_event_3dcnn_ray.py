"""Launch Event 3D CNN training via Ray on Razer GPU"""
import ray
import subprocess
import os

# Connect to Ray cluster
ray.init(address="jupiter:6379")

@ray.remote(num_gpus=1)
def train_event_3dcnn():
    """Train Event 3D CNN - runs on GPU worker"""
    import subprocess
    import os

    # Set environment for small GPU (Razer RTX 3070 8GB)
    os.environ["EVENT3D_CHANNELS"] = "32"  # Reduce channels for 8GB GPU
    os.environ["EVENT3D_LAYERS"] = "3"     # Fewer layers
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        "python3",
        "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_event_3d_cnn.py",
        "--n_files", "30",  # Limit data for 8GB VRAM
        "--batch_size", "16",
        "--epochs_per_fold", "3",
        "--n_folds", "3"
    ]

    result = subprocess.run(
        cmd,
        cwd="/home/jupiter/Lvl3Quant",
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

# Submit job
print("Submitting Event 3D CNN training to Ray cluster...")
future = train_event_3dcnn.remote()
print(f"Job submitted. Ray object ref: {future}")
print("Training running on GPU worker (Razer)...")
print("Check Ray dashboard at http://jupiter:8265")
