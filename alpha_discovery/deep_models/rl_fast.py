#!/usr/bin/env python3
"""
RL Execution Agent v2 — GPU-vectorized PPO with 30-feature state
================================================================
Expanded features (24 static + 6 dynamic), improved architecture,
checkpointing/resume, multiple reward functions, training history.

Usage:
    python rl_fast.py --epochs 200 --reward drawdown
    python rl_fast.py --epochs 200 --reward drawdown --resume
    python rl_fast.py --eval --reward drawdown
"""
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json, os, sys, argparse, logging, time, glob
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
MODEL_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'rl_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICK, TICK_VAL, COMM = 0.25, 12.50, 0.376
SPREAD_COST = 0.5  # ticks paid on market order exit
SCAN_STEP = 100  # Decision every 10 seconds (100 bars at 100ms)
MAX_HOLD_STEPS = 36000 // SCAN_STEP  # 60 min max hold = 36000 bars / SCAN_STEP

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('rl_fast')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(MODEL_DIR.parent / f'rl_fast_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


# ── Feature dimensions ──
STATIC_DIM = 24
DYNAMIC_DIM = 6  # in_pos, direction, unreal_pnl, hold_time, max_fav, max_adv
STATE_DIM = STATIC_DIM + DYNAMIC_DIM  # 30
N_ACTIONS = 3  # SKIP/HOLD, ENTER, EXIT


# ── Precompute all features for all days into tensors ──

def precompute_day(preds, mids, book=None):
    """Precompute 24 static features for one day, subsampled at SCAN_STEP intervals."""
    n = len(preds)
    w = 3000

    # Subsample indices: skip first 30min (18000 bars) and last 15min (9000 bars)
    indices = np.arange(18000, n - 9000, SCAN_STEP)
    n_steps = len(indices)
    if n_steps < 10:
        return None

    # --- Base signals ---
    returns = np.diff(mids) / np.maximum(mids[:-1], 1e-10)
    returns = np.insert(returns, 0, 0)
    roll_vol = pd.Series(returns).rolling(w, min_periods=100).std().values
    roll_mean = pd.Series(preds).rolling(w, min_periods=50).mean().values
    roll_std = np.maximum(pd.Series(preds).rolling(w, min_periods=50).std().values, 1e-10)
    z_scores = (preds - roll_mean) / roll_std
    vol_pctile = pd.Series(roll_vol).rolling(w, min_periods=100).rank(pct=True).values * 100
    pred_std = pd.Series(preds).rolling(w, min_periods=100).std().values

    # --- Rolling z-score means ---
    z_s = pd.Series(z_scores)
    rz10 = z_s.rolling(10, min_periods=1).mean().values
    rz50 = z_s.rolling(50, min_periods=1).mean().values
    rz100 = z_s.rolling(100, min_periods=1).mean().values
    rz500 = z_s.rolling(500, min_periods=1).mean().values
    rz1000 = z_s.rolling(1000, min_periods=1).mean().values

    # --- EMA z-score span 1000 ---
    ema_z_1000 = z_s.ewm(span=1000, min_periods=1).mean().values

    # --- Crossover signals ---
    fast_cross_medium = rz10 - rz100
    medium_cross_slow = rz100 - rz1000

    # --- Vol trend ---
    vol_s = pd.Series(roll_vol)
    vol_trend = (vol_s - vol_s.rolling(100, min_periods=10).mean()).values

    # --- Price momentum ---
    mid_s = pd.Series(mids)
    mom_50 = mid_s.pct_change(50).values
    mom_200 = mid_s.pct_change(200).values

    # --- Z-score acceleration (change over 10 bars) ---
    z_accel = z_s.diff(10).values

    # --- Book features ---
    if book is not None and len(book) == n:
        bid_d1 = book[:, 0, 1]
        ask_d1 = book[:, 10, 1]
        bt = book[:, :5, 1].sum(axis=1)
        at = book[:, 10:15, 1].sum(axis=1)
        imb = (bt - at) / (bt + at + 1e-10)
        spread = np.abs(book[:, 10, 0] - book[:, 0, 0])
        depth_ratio = np.clip(bt / (at + 1e-10), 0, 4) - 2  # center around 0, clip -2 to 2
        # Imbalance trend: 100-bar change in imbalance
        imb_trend = pd.Series(imb).diff(100).values
    else:
        bid_d1 = ask_d1 = imb = np.zeros(n)
        spread = np.ones(n)
        depth_ratio = np.zeros(n)
        imb_trend = np.zeros(n)

    # --- Subsample and stack 24 static features ---
    feat = np.column_stack([
        z_scores[indices] / 5.0,                                    # 0: z_score
        preds[indices],                                              # 1: raw_prediction
        np.nan_to_num(pred_std[indices], nan=0) * 10,               # 2: pred_std
        np.nan_to_num(rz10[indices], nan=0) / 5.0,                  # 3: rolling_z_10
        np.nan_to_num(rz50[indices], nan=0) / 5.0,                  # 4: rolling_z_50
        np.nan_to_num(rz100[indices], nan=0) / 5.0,                 # 5: rolling_z_100
        np.nan_to_num(rz500[indices], nan=0) / 5.0,                 # 6: rolling_z_500
        np.nan_to_num(rz1000[indices], nan=0) / 5.0,                # 7: rolling_z_1000
        np.nan_to_num(ema_z_1000[indices], nan=0) / 5.0,            # 8: ema_z_1000
        np.nan_to_num(fast_cross_medium[indices], nan=0),            # 9: fast_cross_medium
        np.nan_to_num(medium_cross_slow[indices], nan=0),            # 10: medium_cross_slow
        np.log1p(np.abs(bid_d1[indices])) / 5.0,                    # 11: bid_depth_l1
        np.log1p(np.abs(ask_d1[indices])) / 5.0,                    # 12: ask_depth_l1
        imb[indices],                                                # 13: book_imbalance
        spread[indices] / 4.0,                                       # 14: spread
        np.clip(depth_ratio[indices], -2, 2),                        # 15: depth_ratio
        np.nan_to_num(imb_trend[indices], nan=0),                    # 16: imbalance_trend
        np.nan_to_num(roll_vol[indices], nan=0) * 1000,              # 17: realized_vol
        np.nan_to_num(vol_pctile[indices], nan=50) / 100,            # 18: vol_percentile
        np.nan_to_num(vol_trend[indices], nan=0) * 100,              # 19: vol_trend
        np.nan_to_num(mom_50[indices], nan=0) * 1000,                # 20: price_momentum_50
        np.nan_to_num(mom_200[indices], nan=0) * 1000,               # 21: price_momentum_200
        (indices - 18000) / max(n - 27000, 1),                       # 22: time_of_day
        np.nan_to_num(z_accel[indices], nan=0),                      # 23: z_score_acceleration
    ]).astype(np.float32)

    feat = np.nan_to_num(feat, nan=0, posinf=1, neginf=-1)

    return {
        'features': feat,           # (n_steps, 24) static features
        'mids': mids[indices],      # (n_steps,) mid prices at each step
        'z_scores': z_scores[indices],
        'n_steps': n_steps,
    }


# ── Policy Network ──

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.actor_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = torch.relu(self.ln1(self.fc1(x)))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.actor_head(h), self.critic_head(h)


# ── Reward functions ──

def compute_reward(pnl_t, max_adv, reward_type):
    if reward_type == 'raw':
        return pnl_t
    elif reward_type == 'drawdown':
        return pnl_t - 0.05 * max(0, -max_adv)
    elif reward_type == 'sharpe_like':
        return pnl_t / (1.0 + abs(pnl_t) * 0.1)
    elif reward_type == 'calmar':
        return pnl_t / (1.0 + abs(max_adv) * 0.2)
    else:
        return pnl_t


# ── Episode runner ──

def run_episode_gpu(policy, day_data, reward_type='drawdown'):
    """Run one episode on CPU for collection, return CPU transitions."""
    feats = torch.tensor(day_data['features'], dtype=torch.float32)  # (T, 24) on CPU
    mids = day_data['mids']
    z = day_data['z_scores']
    T = day_data['n_steps']

    states, actions, rewards, log_probs, values = [], [], [], [], []

    # Move policy to CPU for episode collection
    policy.cpu()

    # Position tracking
    in_pos = False
    pos_dir = 0
    entry_price = 0.0
    entry_step = 0
    max_fav = 0.0
    max_adv = 0.0
    total_pnl = 0.0
    trades = []
    peak = 0.0
    max_dd = 0.0

    for t in range(T):
        # Build dynamic state
        if in_pos:
            unreal = (mids[t] - entry_price) / TICK * pos_dir
            hold = (t - entry_step) / T
        else:
            unreal = 0.0
            hold = 0.0

        dyn = torch.tensor([
            float(in_pos), float(pos_dir),
            np.clip(unreal / 50, -1, 1), np.clip(hold, 0, 2),
            np.clip(max_fav / 30, 0, 2), np.clip(max_adv / 30, -2, 0),
        ], dtype=torch.float32)  # CPU

        state = torch.cat([feats[t], dyn])  # CPU
        states.append(state)

        with torch.no_grad():
            logits, value = policy(state.unsqueeze(0))
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        lp = dist.log_prob(action)

        actions.append(action.item())
        log_probs.append(lp.detach())
        values.append(value.squeeze().detach())

        reward = 0.0
        a = action.item()

        if a == 1 and not in_pos:  # ENTER
            direction = 1 if z[t] > 0 else -1
            in_pos = True
            pos_dir = direction
            entry_price = mids[t]
            entry_step = t
            max_fav = 0.0
            max_adv = 0.0

        elif a == 2 and in_pos:  # EXIT
            pnl_t = (mids[t] - entry_price) / TICK * pos_dir - COMM - SPREAD_COST
            reward = compute_reward(pnl_t, max_adv, reward_type)
            total_pnl += pnl_t * TICK_VAL
            trades.append({'pnl_t': pnl_t, 'hold': t - entry_step})
            in_pos = False
            pos_dir = 0

        # Update position tracking
        if in_pos:
            cur_pnl = (mids[t] - entry_price) / TICK * pos_dir
            max_fav = max(max_fav, cur_pnl)
            max_adv = min(max_adv, cur_pnl)

            # Force exit at max hold time (60 min)
            force_exit = (t - entry_step) >= MAX_HOLD_STEPS or t == T - 1
            if force_exit:
                pnl_t = cur_pnl - COMM - SPREAD_COST
                reward = compute_reward(pnl_t, max_adv, reward_type)
                total_pnl += pnl_t * TICK_VAL
                trades.append({'pnl_t': pnl_t, 'hold': t - entry_step, 'forced': True})
                in_pos = False
                pos_dir = 0

        peak = max(peak, total_pnl)
        max_dd = max(max_dd, peak - total_pnl)
        rewards.append(reward)

    # Move policy back to GPU for PPO update
    policy.to(device)

    return states, actions, rewards, log_probs, values, total_pnl, trades, max_dd


# ── Checkpointing ──

def save_checkpoint(policy, optimizer, epoch, best_pnl, history, reward_type):
    ckpt_path = MODEL_DIR / f'rl_v2_checkpoint_epoch{epoch}.pt'
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_pnl': best_pnl,
        'history': history,
    }, str(ckpt_path))
    # Keep only last 3 checkpoints
    ckpts = sorted(glob.glob(str(MODEL_DIR / 'rl_v2_checkpoint_epoch*.pt')))
    while len(ckpts) > 3:
        os.remove(ckpts.pop(0))
    log.info(f"  Checkpoint saved: epoch {epoch}")


def load_latest_checkpoint(policy, optimizer):
    ckpts = sorted(glob.glob(str(MODEL_DIR / 'rl_v2_checkpoint_epoch*.pt')))
    if not ckpts:
        log.info("No checkpoint found, starting from scratch.")
        return 0, -np.inf, []
    ckpt_path = ckpts[-1]
    log.info(f"Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_pnl'], ckpt.get('history', [])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_history(history):
    hist_path = MODEL_DIR / 'rl_v2_history.json'
    with open(str(hist_path), 'w') as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)


# ── Training ──

def load_data():
    log.info("Loading data...")
    pred_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in pred_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)}")

    day_datas = []
    for date in dates:
        preds = pred_data[f'{date}_preds']
        mids = pred_data[f'{date}_mid']
        book_file = BOOK_DIR / f'{date}_book_tensors.npz'
        book = np.load(str(book_file))['book_tensors'] if book_file.exists() else None
        dd = precompute_day(preds, mids, book)
        if dd:
            dd['date'] = date
            day_datas.append(dd)
            log.info(f"  {date}: {dd['n_steps']} steps")

    n_train = max(1, len(day_datas) - 7)
    train_days = day_datas[:n_train]
    eval_days = day_datas[n_train:]
    log.info(f"Train: {n_train} days, Eval: {len(eval_days)} days")
    return train_days, eval_days


def count_trades_fast(policy, days, reward_type):
    """Quick trade count on a few days without storing transitions."""
    total_trades = 0
    n = min(3, len(days))
    with torch.no_grad():
        for dd in days[:n]:
            states, _, _, _, _, _, trades, _ = run_episode_gpu(policy, dd, reward_type)
            total_trades += len(trades)
            del states
    gc.collect()
    return total_trades / max(n, 1)


def train(epochs=200, reward_type='drawdown', resume=False):
    train_days, eval_days = load_data()

    policy = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    gamma = 0.99
    lam = 0.95
    eps_clip = 0.2

    start_epoch = 0
    best_pnl = -np.inf
    history = []

    if resume:
        start_epoch, best_pnl, history = load_latest_checkpoint(policy, optimizer)

    log.info(f"Training epochs {start_epoch} to {start_epoch + epochs - 1}, reward={reward_type}")
    total_params = sum(p.numel() for p in policy.parameters())
    log.info(f"Model params: {total_params:,}")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        epoch_pnl = []
        epoch_trades = 0
        all_states, all_actions, all_rewards, all_lps, all_vals = [], [], [], [], []

        for dd in train_days:
            states, actions, rewards, lps, vals, pnl, trades, mdd = run_episode_gpu(
                policy, dd, reward_type)
            epoch_pnl.append(pnl)
            epoch_trades += len(trades)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_lps.extend(lps)
            all_vals.extend(vals)

        # Compute GAE returns
        returns = []
        gae, nv = 0, 0
        for i in reversed(range(len(all_rewards))):
            delta = all_rewards[i] + gamma * nv - all_vals[i].item()
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + all_vals[i].item())
            nv = all_vals[i].item()

        # PPO update
        states_t = torch.stack(all_states).to(device)
        actions_t = torch.tensor(all_actions, device=device)
        old_lps_t = torch.stack(all_lps).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(all_vals).to(device)
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch PPO updates
        n = len(states_t)
        bs = min(2048, n)
        total_loss = 0.0
        for _ in range(4):
            idx = torch.randperm(n, device=device)[:bs]
            logits, new_vals = policy(states_t[idx])
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - old_lps_t[idx])
            s1 = ratio * advantages[idx]
            s2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[idx]
            loss = -torch.min(s1, s2).mean() + 0.5 * nn.MSELoss()(new_vals.squeeze(), returns_t[idx]) - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        # Free GPU memory from PPO update
        del states_t, actions_t, old_lps_t, returns_t, values_t, advantages
        del all_states, all_actions, all_rewards, all_lps, all_vals, returns
        torch.cuda.empty_cache()
        gc.collect()

        avg_pnl = np.mean(epoch_pnl)
        trades_per_day = epoch_trades / max(len(train_days), 1)
        dt = time.time() - t0

        # Log every epoch
        log.info(f"  Epoch {epoch:>3d}: Avg P&L ${avg_pnl:>8,.2f} | Total ${sum(epoch_pnl):>10,.2f} | "
                f"Trades {trades_per_day:.1f}/day | Best ${best_pnl:>8,.2f} | {dt:.1f}s")

        # Track history
        history.append({
            'epoch': epoch,
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(sum(epoch_pnl), 2),
            'trades_per_day': round(trades_per_day, 1),
            'loss': round(total_loss / 4, 4),
        })

        # Save best model
        if avg_pnl > best_pnl:
            best_pnl = avg_pnl
            torch.save(policy.state_dict(), str(MODEL_DIR / f'rl_v2_best_{reward_type}.pt'))
            log.info(f"  ** New best: ${best_pnl:,.2f} **")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(policy, optimizer, epoch, best_pnl, history, reward_type)

        # Save history every epoch
        save_history(history)

    # Final checkpoint
    save_checkpoint(policy, optimizer, epoch, best_pnl, history, reward_type)

    # Evaluate
    evaluate(policy, eval_days, reward_type)

    return best_pnl


def evaluate(policy, eval_days, reward_type):
    log.info(f"\n--- Evaluation ({len(eval_days)} held-out days) ---")
    policy.eval()
    eval_total = 0
    eval_trades = 0
    with torch.no_grad():
        for dd in eval_days:
            states, _, _, _, _, pnl, trades, mdd = run_episode_gpu(policy, dd, reward_type)
            eval_total += pnl
            eval_trades += len(trades)
            log.info(f"  {dd['date']}: P&L ${pnl:>8,.2f} | {len(trades)} trades | Max DD ${mdd:,.2f}")
            del states
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"  EVAL TOTAL: ${eval_total:,.2f} | {eval_trades} trades over {len(eval_days)} days")
    log.info(f"  EVAL AVG: ${eval_total / max(len(eval_days), 1):,.2f}/day | "
            f"{eval_trades / max(len(eval_days), 1):.1f} trades/day")


def eval_only(reward_type):
    """Load best model and evaluate."""
    _, eval_days = load_data()
    policy = PolicyNetwork().to(device)
    model_path = MODEL_DIR / f'rl_v2_best_{reward_type}.pt'
    if not model_path.exists():
        log.error(f"No model found at {model_path}")
        return
    policy.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    evaluate(policy, eval_days, reward_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Execution Agent v2')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reward', default='drawdown', choices=['raw', 'drawdown', 'sharpe_like', 'calmar'])
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluate best model only')
    args = parser.parse_args()

    if args.eval:
        eval_only(args.reward)
    else:
        log.info(f"\n{'='*60}\nRL Execution Agent v2 — reward: {args.reward}\n{'='*60}")
        best = train(args.epochs, args.reward, args.resume)
        log.info(f"DONE: {args.reward} | Best avg P&L ${best:,.2f}")
