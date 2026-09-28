import ray
import os
import sys

# Initialize Ray client
ray.init(address="auto")

@ray.remote(num_gpus=1)
def run_mamba_training():
    import subprocess
    import os
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'EVENT_N_FOLDS': '2',
        'EVENT_EPOCHS': '3',
        'MAMBA_D_MODEL': '128',
        'MAMBA_N_LAYERS': '4',
        'MAMBA_D_STATE': '64',
        'EVENT_BATCH_SIZE': '64',
        'EVENT_WINDOW_SIZE': '1000',
        'EVENT_STRIDE': '500',
    })
    
    # Run training
    result = subprocess.run([
        '/home/nick/training-env/bin/python', '-u',
        '/home/nick/Lvl3Quant/alpha_discovery/deep_models/train_event_mamba.py',
        '--data-dir', '/tmp/mbo_clean_neptune',
        '--output-dir', f'/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/mamba_clean_baseline'
    ], env=env, capture_output=True, text=True)
    
    return result.returncode, result.stdout, result.stderr

# Submit task
task = run_mamba_training.remote()
print(f"Task submitted: {task}")
print("Training started on GPU worker...")
