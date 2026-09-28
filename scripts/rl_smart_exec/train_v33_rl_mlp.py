"""
Train hand-rolled PPO MLP policy on the v3.3 smart-execution env (HC #396).

Architecture:
    shared trunk: Linear(state_dim → 256) ReLU Linear(256 → 256) ReLU
    policy head:  Linear(256 → n_actions) → categorical
    value head:   Linear(256 → 1)

Hyperparameters (per task brief):
    lr=3e-4, clip=0.2, n_envs=8, n_steps=128, n_epochs=4,
    gamma=0.99, gae_lambda=0.95, value_coef=0.5, ent_coef=0.01,
    max_grad_norm=0.5, total_timesteps=1_000_000.

Logging:
    MLflow experiment 'RL_v3_3_smart_exec' (env: MLFLOW_TRACKING_URI).
    Metrics every rollout (n_envs * n_steps = 1024 transitions):
        rollout/ep_reward_mean, rollout/ep_round_trips_mean,
        rollout/ep_len_mean, rollout/win_rate, rollout/mean_ticks_per_trade,
        rollout/action_dist_<i>, train/value_loss, train/policy_loss,
        train/entropy, train/approx_kl.
    Checkpoints every 100k steps and final.

Held-out eval:
    On the LAST OOT day (e.g. 20260227), run deterministic-policy episodes
    starting at every sample (or stride). Compute Sharpe / WR / ticks/trade /
    day-concentration vs HC #344 ≤ 0.20 gate.

Output directory: <out_dir>/ (defaults to project /output/rl_v3_3_smart_exec_<ts>/)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python train_v33_rl_mlp.py` from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v33_rl_dataset import build_dataset, save_feature_stats, STATE_DIM
from v33_rl_env import V33SmartExecEnv, VecEnv, N_ACTIONS

# ----------------------------------------------------------------------
# Policy / value network
# ----------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        if deterministic:
            action = logits.argmax(dim=-1)
            logp = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action, logp, value


# ----------------------------------------------------------------------
# PPO loop
# ----------------------------------------------------------------------
@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 128
    n_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_every: int = 100_000
    seed: int = 2026


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Standard GAE-λ. Shapes: (T, N) for rewards/values/dones, (N,) last_value."""
    T, N = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].astype(np.float32)
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train(
    npz_path: str,
    out_dir: str,
    total_timesteps: int,
    n_envs: int,
    smoke: bool = False,
    device_str: str = "auto",
    experiment_name: str = "RL_v3_3_smart_exec",
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig(total_timesteps=total_timesteps, n_envs=n_envs)
    if smoke:
        cfg.total_timesteps = 1024
        cfg.n_envs = 2
        cfg.n_steps = 64
        cfg.minibatch_size = 32
        cfg.checkpoint_every = 1024

    # ---- Device ----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] device={device}, smoke={smoke}, total_timesteps={cfg.total_timesteps}")

    # ---- Data ----
    print(f"[train] loading dataset from {npz_path}")
    ds = build_dataset(npz_path)
    print(f"[train] dataset n={ds.n_samples}, days={ds.oot_dates}")
    save_feature_stats(ds, out_dir / "feature_stats.npz")

    # Eligible-start indices: exclude last OOT day so it's held out for eval.
    # (HC #0 sliding WF: do not train across the held-out boundary.)
    train_day_max = max(0, int(ds.day_idx.max()) - 1)
    train_mask = ds.day_idx <= train_day_max
    eligible = np.where(train_mask)[0]
    # Drop the last cfg-max-steps of each training day
    max_steps_env = 200
    for di in range(train_day_max + 1):
        day_mask = (ds.day_idx == di)
        last_idxs = np.where(day_mask)[0][-max_steps_env:]
        eligible = np.setdiff1d(eligible, last_idxs, assume_unique=False)
    print(f"[train] train days=0..{train_day_max}, held-out day={int(ds.day_idx.max())} ({ds.oot_dates[-1] if ds.oot_dates else 'N/A'})")
    print(f"[train] eligible training starts={len(eligible)}")

    venv = VecEnv(cfg.n_envs, ds, eligible_indices=eligible, max_steps=max_steps_env)
    state_dim = venv.observation_dim
    n_actions = venv.n_actions
    print(f"[train] state_dim={state_dim}, n_actions={n_actions}")

    # ---- Model ----
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = ActorCritic(state_dim, n_actions, hidden=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---- MLflow (best-effort) ----
    mlflow_run = None
    try:
        import mlflow
        tracking = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"rl_v33_smart_exec_{int(time.time())}")
        mlflow.log_params({
            "state_dim": state_dim,
            "n_actions": n_actions,
            "n_envs": cfg.n_envs,
            "n_steps": cfg.n_steps,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "clip_range": cfg.clip_range,
            "total_timesteps": cfg.total_timesteps,
            "value_coef": cfg.value_coef,
            "ent_coef": cfg.ent_coef,
            "smoke": int(smoke),
            "device": str(device),
            "held_out_oot_date": ds.oot_dates[-1] if ds.oot_dates else "",
            "n_train_days": train_day_max + 1,
            "head_count": 32,
        })
        print(f"[train] MLflow tracking @ {tracking}, run_id={mlflow_run.info.run_id}")
    except Exception as e:
        print(f"[train] WARN: MLflow disabled ({e})")

    # ---- Rollout buffers ----
    T, N = cfg.n_steps, cfg.n_envs
    obs_buf = np.zeros((T, N, state_dim), dtype=np.float32)
    act_buf = np.zeros((T, N), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=bool)
    val_buf = np.zeros((T, N), dtype=np.float32)

    obs = venv.reset()
    global_step = 0
    last_ckpt_step = 0
    ep_rewards: list[float] = []
    ep_round_trips: list[int] = []
    ep_pnl_per_trade: list[float] = []
    iter_idx = 0
    t0 = time.time()

    while global_step < cfg.total_timesteps:
        # ---- Rollout ----
        for t in range(T):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, logp, value = model.act(obs_t, deterministic=False)
            actions_np = action.detach().cpu().numpy()
            next_obs, rew, done, infos = venv.step(actions_np)

            obs_buf[t] = obs
            act_buf[t] = actions_np
            logp_buf[t] = logp.detach().cpu().numpy()
            rew_buf[t] = rew
            done_buf[t] = done
            val_buf[t] = value.detach().cpu().numpy()

            for i, info in enumerate(infos):
                if done[i] and "episode_round_trips" in info:
                    rt_pnl = info.get("episode_round_trip_pnl", [])
                    ep_rewards.append(float(np.sum(rt_pnl) if rt_pnl else 0.0))
                    ep_round_trips.append(int(info["episode_round_trips"]))
                    ep_pnl_per_trade.extend(float(x) for x in rt_pnl)

            obs = next_obs
            global_step += N

        # ---- Bootstrap value for last state ----
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, _, last_value = model.act(obs_t)
            last_value_np = last_value.detach().cpu().numpy()

        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_value_np, cfg.gamma, cfg.gae_lambda)
        # Normalize advantages
        adv_flat = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten
        obs_f = obs_buf.reshape(T * N, state_dim)
        act_f = act_buf.reshape(T * N)
        old_logp_f = logp_buf.reshape(T * N)
        adv_f = adv_flat.reshape(T * N)
        ret_f = ret.reshape(T * N)

        # ---- PPO update ----
        idxs = np.arange(T * N)
        policy_losses, value_losses, entropies, kls = [], [], [], []
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T * N, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                mb_obs = torch.as_tensor(obs_f[mb], dtype=torch.float32, device=device)
                mb_act = torch.as_tensor(act_f[mb], dtype=torch.int64, device=device)
                mb_old_logp = torch.as_tensor(old_logp_f[mb], dtype=torch.float32, device=device)
                mb_adv = torch.as_tensor(adv_f[mb], dtype=torch.float32, device=device)
                mb_ret = torch.as_tensor(ret_f[mb], dtype=torch.float32, device=device)

                logits, values = model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_ret - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl)

        # ---- Per-iteration metrics ----
        iter_idx += 1
        action_dist = np.bincount(act_f.astype(int), minlength=N_ACTIONS).astype(np.float32)
        action_dist /= max(1.0, action_dist.sum())
        ep_reward_mean = float(np.mean(ep_rewards[-200:])) if ep_rewards else 0.0
        ep_rt_mean = float(np.mean(ep_round_trips[-200:])) if ep_round_trips else 0.0
        if ep_pnl_per_trade:
            recent = ep_pnl_per_trade[-2000:]
            mean_ticks_per_trade = float(np.mean(recent))
            win_rate = float(np.mean(np.array(recent) > 0.0))
        else:
            mean_ticks_per_trade = 0.0
            win_rate = 0.0
        elapsed = time.time() - t0
        sps = global_step / max(1e-6, elapsed)

        msg = (f"[iter {iter_idx:4d}] step={global_step:>8d}/{cfg.total_timesteps} "
               f"ep_rew={ep_reward_mean:+.3f} rt={ep_rt_mean:.2f} "
               f"ticks/trd={mean_ticks_per_trade:+.3f} wr={win_rate*100:.1f}% "
               f"act={['%.2f' % x for x in action_dist]} "
               f"pl={np.mean(policy_losses):+.4f} vl={np.mean(value_losses):.4f} "
               f"H={np.mean(entropies):.3f} kl={np.mean(kls):.4f} "
               f"sps={sps:.0f}")
        print(msg, flush=True)

        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metrics({
                    "rollout/ep_reward_mean": ep_reward_mean,
                    "rollout/ep_round_trips_mean": ep_rt_mean,
                    "rollout/mean_ticks_per_trade": mean_ticks_per_trade,
                    "rollout/win_rate": win_rate,
                    "train/policy_loss": float(np.mean(policy_losses)),
                    "train/value_loss": float(np.mean(value_losses)),
                    "train/entropy": float(np.mean(entropies)),
                    "train/approx_kl": float(np.mean(kls)),
                    "perf/steps_per_sec": sps,
                    **{f"rollout/action_dist_{i}": float(action_dist[i]) for i in range(N_ACTIONS)},
                }, step=global_step)
            except Exception as e:
                print(f"[train] MLflow log failed: {e}")

        # ---- Checkpoint ----
        if global_step - last_ckpt_step >= cfg.checkpoint_every or global_step >= cfg.total_timesteps:
            ckpt_path = out_dir / f"policy_step_{global_step}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "state_dim": state_dim,
                "n_actions": n_actions,
            }, ckpt_path)
            last_ckpt_step = global_step
            print(f"[train] checkpoint saved → {ckpt_path}")
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_artifact(str(ckpt_path))
                except Exception:
                    pass

    # ---- Held-out evaluation on last OOT day ----
    print("[train] running held-out eval on last OOT day ...")
    eval_metrics = run_holdout_eval(model, ds, device, max_steps_env=max_steps_env)
    print(f"[train] EVAL: {eval_metrics}")
    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metrics({f"eval/{k}": float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)})
            mlflow.log_artifact(str(out_dir / "eval_metrics.json"))
            mlflow.end_run()
        except Exception:
            pass

    return out_dir


# ----------------------------------------------------------------------
# Held-out evaluation
# ----------------------------------------------------------------------
def run_holdout_eval(model: ActorCritic, ds, device, max_steps_env: int = 200, stride: int = 50):
    """Roll deterministic policy across the last OOT day. Returns metrics dict."""
    last_day = int(ds.day_idx.max())
    day_mask = (ds.day_idx == last_day)
    day_indices = np.where(day_mask)[0]
    if len(day_indices) < max_steps_env + 1:
        return {"n_episodes": 0, "note": "insufficient samples"}
    # Stride start points
    starts = day_indices[: len(day_indices) - max_steps_env : stride]

    all_pnl: list[float] = []
    all_days: list[int] = []  # for HC #344 day-conc
    n_trades = 0
    for s in starts:
        env = V33SmartExecEnv(ds, eligible_indices=np.array([int(s)]), max_steps=max_steps_env,
                              rng=np.random.default_rng(int(s) % (2**31)))
        # Force the env to deterministically start at s
        env.eligible_indices = np.array([int(s)], dtype=np.int64)
        obs = env.reset()
        ep_pnl = []
        done = False
        steps = 0
        while not done and steps < max_steps_env:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = model.act(ot, deterministic=True)
            obs, r, done, info = env.step(int(action.item()))
            steps += 1
            if done and "episode_round_trip_pnl" in info:
                ep_pnl = info["episode_round_trip_pnl"]
        for p in ep_pnl:
            all_pnl.append(float(p))
            all_days.append(last_day)
            n_trades += 1

    if not all_pnl:
        return {"n_episodes": int(len(starts)), "n_trades": 0, "note": "policy never traded"}
    pnl_arr = np.array(all_pnl, dtype=np.float64)
    # Sharpe (per trade — annualisation depends on trade frequency, keep raw):
    sharpe = float(pnl_arr.mean() / (pnl_arr.std() + 1e-8))
    win_rate = float((pnl_arr > 0).mean())
    mean_ticks = float(pnl_arr.mean())
    pf = float(pnl_arr[pnl_arr > 0].sum() / max(1e-8, -pnl_arr[pnl_arr < 0].sum()))
    # Day concentration (HC #344): single-day P&L over total absolute P&L.
    # With one held-out day only, this is trivially 1.0 — flag it.
    total_abs = float(np.abs(pnl_arr).sum())
    day_conc = 1.0 if total_abs > 0 else 0.0
    return {
        "n_episodes": int(len(starts)),
        "n_trades": int(n_trades),
        "mean_ticks_per_trade": mean_ticks,
        "win_rate": win_rate,
        "sharpe_per_trade": sharpe,
        "profit_factor": pf,
        "day_concentration_HC344": day_conc,
        "total_pnl_ticks": float(pnl_arr.sum()),
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")
    ap.add_argument("--out-dir", default=f"/home/jupiter/Lvl3Quant/output/rl_v3_3_smart_exec_{int(time.time())}")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Smoke test (1024 timesteps, CPU OK)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--experiment-name", default="RL_v3_3_smart_exec")
    args = ap.parse_args()

    train(
        npz_path=args.npz,
        out_dir=args.out_dir,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        smoke=args.smoke,
        device_str=args.device,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
