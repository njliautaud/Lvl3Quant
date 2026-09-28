"""
PPO v3 training script — HC #399 canonical-replay reward.

Uses CanonicalReplayEnv (env_v3_canonical.py) — reward signal is sourced
from the same canonical primitives used by `full_market_replay()` and
`ppo_v2_1_canonical_replay_eval.py:canonical_reprice()`. NO env-internal
reward proxy. This is what HC #399 #1 mandates.

Saves: output_dir/ppo_v3_canonical_final.zip
       output_dir/ckpt_v3/ppo_v3_canonical_<NSTEPS>_steps.zip (every 50k)
MLflow experiment: RL_v3_3_smart_exec_v3_canonical_reward
Run name: ppo_v3_canonical_seed<N>

CHECKPOINTING (HC #398 compliance):
  CheckpointCallback writes weights every 50_000 timesteps to ckpt_v3/.
  PPO can resume from the latest ckpt via `model = PPO.load(latest)` then
  `model.set_env(vec_env); model.learn(...)`. The --resume-from CLI flag
  handles this.

NOT MALWARE. Pure ML training driver.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    print("WARN: mlflow not importable", file=sys.stderr)

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parents[1]))
from env_v3_canonical import CanonicalReplayEnv, OBS_DIM, N_ACTIONS  # noqa: E402


# ============================================================================
# Default paths (Razer/Jupiter compatible — override via --npz / --labels-dir)
# ============================================================================
DEFAULT_NPZ_JUPITER = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
DEFAULT_LABELS_JUPITER = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels"
DEFAULT_OUTPUT_JUPITER = "/home/jupiter/Lvl3Quant/output/rl_v3_3_smart_exec_v3"

DEFAULT_NPZ_RAZER = "C:\\Users\\claude\\Lvl3Quant\\output\\cnn_mamba_v3_3_uncertainty_weighted\\fold_00_predictions.npz"
DEFAULT_LABELS_RAZER = "C:\\Users\\claude\\Lvl3Quant\\data\\processed\\mbo_events_smart_v3_fifo_labels"
DEFAULT_OUTPUT_RAZER = "C:\\Users\\claude\\Lvl3Quant\\output\\rl_v3_3_smart_exec_v3"


# ============================================================================
# MLflow logging callback (mirrors train_ppo_v2.py pattern)
# ============================================================================
class MLflowCallback(BaseCallback):
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


# ============================================================================
# Env factory
# ============================================================================
def make_env_factory(npz_path: str, labels_dir: str, seed: int):
    def _f():
        return CanonicalReplayEnv(
            npz_path=npz_path,
            labels_dir=labels_dir,
            seed=seed,
            episode_day_idx=None,  # random day each episode
            deterministic_queue=True,
        )
    return _f


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ_JUPITER,
                    help="Path to v3.3 predictions NPZ")
    ap.add_argument("--labels-dir", default=DEFAULT_LABELS_JUPITER,
                    help="Path to FIFO labels directory")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_JUPITER,
                    help="Output directory for weights and checkpoints")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=50_000,
                    help="Per-env step interval for ckpt (HC #398)")
    ap.add_argument("--resume-from", default=None,
                    help="Path to a .zip checkpoint to resume from")
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--mlflow-experiment",
                    default="RL_v3_3_smart_exec_v3_canonical_reward")
    ap.add_argument("--run-name", default=None,
                    help="MLflow run name (default: ppo_v3_canonical_seed<N>)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = f"ppo_v3_canonical_seed{args.seed}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] output_dir={output_dir}")
    print(f"[init] npz={args.npz}")
    print(f"[init] labels_dir={args.labels_dir}")
    print(f"[init] total_timesteps={args.total_timesteps:,}")
    print(f"[init] n_envs={args.n_envs} n_steps={args.n_steps} batch={args.batch_size}")
    print(f"[init] lr={args.learning_rate} gamma={args.gamma} ent_coef={args.ent_coef}")
    print(f"[init] seed={args.seed}")
    print(f"[init] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[init] device={torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build vec env
    envs = [make_env_factory(args.npz, args.labels_dir, args.seed + i)
            for i in range(args.n_envs)]
    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)
    print(f"[init] obs_dim={OBS_DIM} n_actions={N_ACTIONS}")

    # MLflow setup
    run_id = None
    mlflow_enabled = MLFLOW_AVAILABLE and not args.no_mlflow
    if mlflow_enabled:
        try:
            import urllib.request, socket
            socket.setdefaulttimeout(3)
            try:
                urllib.request.urlopen(args.mlflow_uri + "/", timeout=3)
            except urllib.error.HTTPError:
                pass
            print(f"[mlflow] connectivity OK to {args.mlflow_uri}")
        except Exception as e:
            print(f"[mlflow] WARN: {args.mlflow_uri} unreachable ({e}); local fallback")
            args.mlflow_uri = "file:" + str(output_dir / "mlruns_v3")
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=args.run_name)
            run_id = mlflow_run.info.run_id
            print(f"[mlflow] tracking_uri={args.mlflow_uri} "
                  f"experiment={args.mlflow_experiment}")
            print(f"[mlflow] run_id={run_id}")
            mlflow.log_params({
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ent_coef": args.ent_coef,
                "clip_range": args.clip_range,
                "n_epochs": args.n_epochs,
                "seed": args.seed,
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
                "policy": "MlpPolicy",
                "net_arch": "[256, 256]",
                "variant": "v3_canonical_reward",
                "reward_source": "canonical_replay_primitives",
                "hc_compliance": "HC#392+HC#397+HC#397B+HC#398+HC#399",
                "resume_from": args.resume_from or "scratch",
            })
        except Exception as e:
            print(f"[mlflow] start_run failed: {e}", file=sys.stderr)

    # Build or load model
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            print(f"FATAL: --resume-from path missing: {resume_path}", file=sys.stderr)
            sys.exit(3)
        print(f"[init] RESUMING from {resume_path}")
        model = PPO.load(str(resume_path), env=vec_env, device=device)
        # Need to override hyperparameters that may have changed
        model.learning_rate = args.learning_rate
        model.ent_coef = args.ent_coef
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            device=device,
            seed=args.seed,
            tensorboard_log=str(output_dir / "tb_v3"),
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[init] policy params: {n_params:,}")
    print(f"[train] starting PPO v3 (CANONICAL reward) "
          f"for {args.total_timesteps:,} timesteps...")

    # Checkpoint callback — HC #398 compliance (every 50k global steps)
    ckpt_save_freq_per_env = max(1, args.checkpoint_every // max(1, args.n_envs))
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_save_freq_per_env,
        save_path=str(ckpt_dir),
        name_prefix="ppo_v3_canonical",
    )
    callbacks = [ckpt_cb]
    if mlflow_enabled and run_id:
        callbacks.append(MLflowCallback(log_every=args.n_steps))

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=(args.resume_from is None),
        )
    except KeyboardInterrupt:
        print("[train] interrupted, saving final...")
    finally:
        elapsed = time.time() - t0
        final_path = output_dir / "ppo_v3_canonical_final.zip"
        model.save(str(final_path))
        print(f"[train] saved final: {final_path}")
        print(f"[train] elapsed: {elapsed:.1f}s = {elapsed/60:.1f}min")

        if mlflow_enabled and run_id:
            try:
                mlflow.log_metric("train_wallclock_sec", elapsed)
                mlflow.log_metric("train_wallclock_min", elapsed / 60.0)
                mlflow.log_artifact(str(final_path))
                mlflow.end_run()
                print(f"[mlflow] run {run_id} closed")
            except Exception as e:
                print(f"[mlflow] end_run err: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
