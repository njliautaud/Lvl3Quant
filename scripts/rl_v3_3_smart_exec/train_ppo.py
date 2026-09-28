"""
PPO training entry for RL_v3_3_smart_exec (HC #396 weekend mandate).

Usage (Razer):
    python train_ppo.py \
        --npz C:\\Users\\claude\\Lvl3Quant\\data\\v3_3\\fold_00_predictions.npz \
        --output-dir C:\\Users\\claude\\Lvl3Quant\\output\\rl_v3_3_smart_exec \
        --total-timesteps 1000000

Logs to MLflow experiment "RL_v3_3_smart_exec" (Jupiter:5000 via Tailscale).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure SB3 is importable
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError as e:
    print(f"FATAL: stable-baselines3 not importable: {e}", file=sys.stderr)
    sys.exit(2)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARN: mlflow not importable; continuing without remote tracking", file=sys.stderr)

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from env import V33SmartExecEnv, OBS_DIM, N_ACTIONS  # noqa: E402


class MLflowCallback(BaseCallback):
    """Periodically log SB3 metrics to MLflow."""

    def __init__(self, log_every: int = 2048):
        super().__init__()
        self.log_every = log_every
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.log_every:
            return True
        self._last = self.num_timesteps
        if not MLFLOW_AVAILABLE:
            return True
        try:
            logger_dict = self.logger.name_to_value
            for k, v in logger_dict.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    safe_k = k.replace("/", "_")
                    mlflow.log_metric(safe_k, float(v), step=self.num_timesteps)
        except Exception as e:
            print(f"[mlflow] log error: {e}", file=sys.stderr)
        return True


def make_env_factory(npz_path: str, seed: int):
    def _f():
        return V33SmartExecEnv(npz_path=npz_path, seed=seed)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to v3.3 fold_00_predictions.npz")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000",
                    help="Jupiter MLflow over Tailscale")
    ap.add_argument("--mlflow-experiment", default="RL_v3_3_smart_exec")
    ap.add_argument("--no-mlflow", action="store_true",
                    help="Disable MLflow tracking (useful if server unreachable)")
    ap.add_argument("--checkpoint-every", type=int, default=100_000)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output_dir={output_dir}")
    print(f"[init] npz={args.npz}")
    print(f"[init] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[init] device={torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build vec env
    envs = [make_env_factory(args.npz, args.seed + i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    # MLflow
    run_id = None
    mlflow_enabled = MLFLOW_AVAILABLE and not args.no_mlflow
    if mlflow_enabled:
        # Test connectivity first with short timeout
        try:
            import urllib.request
            import socket
            socket.setdefaulttimeout(3)
            # Just test base URL connectivity; 200/404 both prove reachability
            try:
                urllib.request.urlopen(args.mlflow_uri + "/", timeout=3)
            except urllib.error.HTTPError:
                pass  # server responded (any code) = reachable
            print(f"[mlflow] connectivity OK to {args.mlflow_uri}")
        except Exception as e:
            print(f"[mlflow] WARN: {args.mlflow_uri} unreachable ({e}); falling back to local file store")
            args.mlflow_uri = "file:" + str(Path(args.output_dir) / "mlruns")
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=f"ppo_v3_3_{int(time.time())}")
            run_id = mlflow_run.info.run_id
            print(f"[mlflow] tracking_uri={args.mlflow_uri} experiment={args.mlflow_experiment}")
            print(f"[mlflow] run_id={run_id}")
            mlflow.log_params({
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
                "policy": "MlpPolicy",
                "net_arch": "[256, 256]",
            })
        except Exception as e:
            print(f"[mlflow] start_run failed: {e}", file=sys.stderr)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        device=device,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tb"),
    )

    print(f"[init] policy params: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"[train] starting PPO for {args.total_timesteps:,} timesteps...")

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // max(1, args.n_envs)),
        save_path=str(output_dir / "ckpt"),
        name_prefix="ppo_v3_3",
    )
    callbacks = [ckpt_cb]
    if mlflow_enabled and run_id:
        callbacks.append(MLflowCallback(log_every=args.n_steps))

    try:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("[train] interrupted by signal, saving final checkpoint...")
    finally:
        final_path = output_dir / "ppo_v3_3_final.zip"
        model.save(str(final_path))
        print(f"[train] saved {final_path}")
        if mlflow_enabled and run_id:
            try:
                mlflow.log_artifact(str(final_path))
                mlflow.end_run()
            except Exception as e:
                print(f"[mlflow] end_run error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
