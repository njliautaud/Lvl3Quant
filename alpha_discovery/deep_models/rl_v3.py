#!/usr/bin/env python3
"""
RL Execution Agent v3 — LSTM Policy + 3-Action Chase Execution
===============================================================
Option A: 3 actions with LSTM memory.
  - SKIP = 0: Don't enter (flat) or keep holding (in position)
  - ENTER = 1: Enter in CNN's predicted direction (passive+chase, probabilistic fill)
  - EXIT = 2: Exit position (market order, pays spread)

Execution is always chase 1t/3r, simulated as passive limit with ~8-10% fill probability.
Direction comes from CNN z-score sign.

Usage:
    python rl_v3.py --epochs 500 --reward drawdown
    python rl_v3.py --resume --epochs 500
    python rl_v3.py --eval
"""
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json, os, sys, argparse, logging, time, glob as globmod
from pathlib import Path
from datetime import datetime

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
MODEL_DIR = RESULTS_DIR / 'rl_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376       # Per side
SPREAD_TICKS = 1.0             # Full spread cost for market exit
BASE_FILL_PROB = 0.08          # 8% base fill probability per decision step
FILL_PROB_SLOPE = 0.02         # +2% per unit of abs(z) above 1.5
MAX_FILL_PROB = 0.20           # Cap at 20%
MAX_HOLD_STEPS = 120           # 60 min max hold (120 steps * 300 bars/step * 100ms = 60min)
SCAN_STEP = 300                # Decision every 300 bars = 30 seconds (3x faster epochs)
LSTM_LOOKBACK = 15             # 15 decision steps * 30s = 7.5 min of context

# ── Actions (3 only) ──
SKIP = 0     # Don't enter (flat) or keep holding (in position)
ENTER = 1    # Enter trade in CNN's predicted direction (passive+chase, probabilistic fill)
EXIT = 2     # Exit position (market order, pays spread)
N_ACTIONS = 3

# ── Feature dims ──
STATIC_DIM = 24
DYNAMIC_DIM = 6   # in_position, direction, unrealized_pnl, hold_time, max_favorable, max_adverse
STATE_DIM = STATIC_DIM + DYNAMIC_DIM  # 30

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('rl_v3')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'rl_v3_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


# ============================================================
# PRECOMPUTE STATIC FEATURES (same as v2)
# ============================================================

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

    # --- Z-score acceleration ---
    z_accel = z_s.diff(10).values

    # --- Book features ---
    if book is not None and len(book) == n:
        bid_d1 = book[:, 0, 1]
        ask_d1 = book[:, 10, 1]
        bt = book[:, :5, 1].sum(axis=1)
        at = book[:, 10:15, 1].sum(axis=1)
        imb = (bt - at) / (bt + at + 1e-10)
        spread = np.abs(book[:, 10, 0] - book[:, 0, 0])
        depth_ratio = np.clip(bt / (at + 1e-10), 0, 4) - 2
        imb_trend = pd.Series(imb).diff(100).values
    else:
        bid_d1 = ask_d1 = imb = np.zeros(n)
        spread = np.ones(n)
        depth_ratio = np.zeros(n)
        imb_trend = np.zeros(n)

    # --- Stack 24 static features ---
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
        'features': feat,           # (n_steps, 24)
        'mids': mids[indices],      # (n_steps,)
        'z_scores': z_scores[indices],
        'n_steps': n_steps,
    }


# ============================================================
# LSTM POLICY NETWORK
# ============================================================

class LSTMPolicy(nn.Module):
    """Actor-Critic with LSTM temporal context."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128, lstm_hidden=64):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # Encode current state
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # LSTM for temporal context
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, encoded_seq, hx=None):
        """
        Args:
            encoded_seq: (batch, seq_len, hidden) — already encoded states
            hx: optional (h0, c0) tuple for LSTM
        Returns:
            logits: (batch, n_actions)
            value: (batch, 1)
            hx: new LSTM hidden state
        """
        lstm_out, hx = self.lstm(encoded_seq, hx)
        # Use last timestep output
        last = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        logits = self.actor(last)
        value = self.critic(last)
        return logits, value, hx

    def encode(self, state):
        """Encode a single state vector through the encoder."""
        return self.encoder(state)

    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden state on CPU (detached)."""
        h0 = torch.zeros(1, batch_size, self.lstm_hidden, device='cpu')
        c0 = torch.zeros(1, batch_size, self.lstm_hidden, device='cpu')
        return (h0.detach(), c0.detach())


# ============================================================
# REWARD FUNCTIONS
# ============================================================

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


# ============================================================
# EPISODE RUNNER
# ============================================================

def run_episode(policy, day_data, reward_type='drawdown', deterministic=False):
    """
    Run one episode (one trading day) on CPU for collection.
    Returns transitions + stats.

    3-action space:
      SKIP=0: stay flat or hold position
      ENTER=1: enter in CNN z-score direction (passive+chase, probabilistic fill)
      EXIT=2: exit position (market order, pays spread)
    """
    feats = torch.tensor(day_data['features'], dtype=torch.float32)  # (T, 24) CPU
    mids = day_data['mids']
    z = day_data['z_scores']
    T = day_data['n_steps']

    # Move policy to CPU for episode collection
    policy.cpu()
    policy.eval()

    # Transition storage
    all_encoded = []    # encoded states for LSTM input
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []

    # ---- Position State ----
    in_position = False
    pos_dir = 0        # +1 long, -1 short
    entry_price = 0.0
    entry_step = 0
    max_fav = 0.0
    max_adv = 0.0
    total_pnl = 0.0
    trades = []
    peak = 0.0
    max_dd = 0.0

    # Fill tracking
    fill_attempts = 0
    fill_successes = 0

    # LSTM state — use stateful (carry hidden forward) for speed
    hx = policy.init_hidden(batch_size=1)  # CPU
    # Detach initial hidden state from any prior computation graph
    hx = (hx[0].detach(), hx[1].detach())

    # Pre-allocate dynamic feature buffer
    dyn_buf = torch.zeros(6, dtype=torch.float32)

    with torch.no_grad():
        for t in range(T):
            # ---- Build 6 dynamic features (in-place) ----
            if in_position:
                unreal = (mids[t] - entry_price) / TICK * pos_dir
                hold_norm = (t - entry_step) / max(T, 1)
                dyn_buf[0] = 1.0
                dyn_buf[1] = float(pos_dir)
                dyn_buf[2] = max(-1.0, min(1.0, unreal / 50))
                dyn_buf[3] = min(hold_norm, 2.0)
                dyn_buf[4] = min(max_fav / 30, 2.0)
                dyn_buf[5] = max(max_adv / 30, -2.0)
            else:
                dyn_buf[0] = 0.0
                dyn_buf[1] = 0.0
                dyn_buf[2] = 0.0
                dyn_buf[3] = 0.0
                dyn_buf[4] = 0.0
                dyn_buf[5] = 0.0

            state = torch.cat([feats[t], dyn_buf])  # (30,) CPU

            # Encode state + single-step LSTM with stateful hidden
            encoded = policy.encode(state.unsqueeze(0))  # (1, hidden)
            encoded_seq = encoded.unsqueeze(1)  # (1, 1, hidden)
            logits, value, hx = policy(encoded_seq, hx)
            # Detach LSTM hidden states every step to prevent graph accumulation
            hx = (hx[0].detach(), hx[1].detach())
            logits = logits.squeeze(0)  # (n_actions,)
            value = value.squeeze()     # scalar

            # Action masking: prevent invalid actions
            mask = torch.full((N_ACTIONS,), -1e8)

            if not in_position:
                # Flat: can SKIP or ENTER
                mask[SKIP] = 0
                mask[ENTER] = 0
            else:
                # In position: can SKIP (hold) or EXIT
                mask[SKIP] = 0
                mask[EXIT] = 0

            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            lp = torch.log(probs[action] + 1e-8)

            a = action.item()
            actions_list.append(a)
            log_probs_list.append(lp.detach())
            values_list.append(value.detach())
            all_encoded.append(state.detach())

            # ---- Execute action ----
            reward = 0.0

            if not in_position:
                if a == ENTER:
                    # Direction from CNN z-score sign
                    direction = 1 if z[t] > 0 else -1

                    # Fill probability: 8% + (abs_z - 1.5) * 2%, capped at 20%
                    abs_z = abs(z[t])
                    fill_prob = BASE_FILL_PROB + (abs_z - 1.5) * FILL_PROB_SLOPE
                    fill_prob = min(max(fill_prob, 0.0), MAX_FILL_PROB)

                    fill_attempts += 1

                    # Deterministic pseudo-random fill decision
                    # Uses bar index for reproducibility
                    bar_index = 18000 + t * SCAN_STEP  # Approximate original bar index
                    pseudo_rand = ((bar_index * 2654435761) % 1000) / 1000.0

                    if pseudo_rand < fill_prob:
                        # Filled! Passive limit entry — no spread cost
                        in_position = True
                        pos_dir = direction
                        entry_price = mids[t]
                        entry_step = t
                        max_fav = 0.0
                        max_adv = 0.0
                        fill_successes += 1
                    # else: no fill, stay flat, reward = 0

                # else: SKIP when flat, reward = 0

            elif in_position:
                if a == EXIT:
                    # Market exit: pays spread + commission both sides
                    exit_pnl = (mids[t] - entry_price) / TICK * pos_dir
                    net_pnl = exit_pnl - SPREAD_TICKS - 2 * COMMISSION_TICKS
                    # Entry was passive (0 spread), exit is market (1 tick spread + commission both sides)

                    reward = compute_reward(net_pnl, max_adv, reward_type)
                    total_pnl += net_pnl * TICK_VAL
                    trades.append({
                        'pnl_t': net_pnl,
                        'hold': t - entry_step,
                        'max_fav': max_fav,
                        'max_adv': max_adv,
                        'direction': pos_dir,
                    })
                    in_position = False
                    pos_dir = 0

                elif a == SKIP:
                    # Hold — dense reward from unrealized P&L change
                    if t > 0:
                        prev_unreal = (mids[max(0, t-1)] - entry_price) / TICK * pos_dir
                        cur_unreal = (mids[t] - entry_price) / TICK * pos_dir
                        reward = (cur_unreal - prev_unreal) * 0.01  # Small shaping signal

            # ---- Update position tracking ----
            if in_position:
                cur_pnl = (mids[t] - entry_price) / TICK * pos_dir
                max_fav = max(max_fav, cur_pnl)
                max_adv = min(max_adv, cur_pnl)

                # Force exit at max hold time or end of day
                force_exit = (t - entry_step) >= MAX_HOLD_STEPS or t == T - 1
                if force_exit:
                    exit_pnl = cur_pnl
                    net_pnl = exit_pnl - SPREAD_TICKS - 2 * COMMISSION_TICKS
                    reward = compute_reward(net_pnl, max_adv, reward_type)
                    total_pnl += net_pnl * TICK_VAL
                    trades.append({
                        'pnl_t': net_pnl,
                        'hold': t - entry_step,
                        'forced': True,
                        'max_fav': max_fav,
                        'max_adv': max_adv,
                        'direction': pos_dir,
                    })
                    in_position = False
                    pos_dir = 0

            # Track drawdown
            peak = max(peak, total_pnl)
            max_dd = max(max_dd, peak - total_pnl)
            rewards_list.append(reward)

    policy.to(device)
    policy.train()

    # Compute stats
    fill_rate = fill_successes / max(fill_attempts, 1)

    stats = {
        'total_pnl': total_pnl,
        'trades': trades,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'fill_rate': fill_rate,
        'fill_attempts': fill_attempts,
        'fill_successes': fill_successes,
    }

    return all_encoded, actions_list, rewards_list, log_probs_list, values_list, stats


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(policy, optimizer, epoch, best_pnl, history, reward_type):
    ckpt_path = MODEL_DIR / f'rl_v3_checkpoint_epoch{epoch}.pt'
    try:
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_pnl': best_pnl,
            'history': history,
            'reward_type': reward_type,
        }, str(ckpt_path))
    except Exception as e:
        log.warning(f"Failed to save checkpoint: {e}")
        return

    # Keep only last 3 checkpoints
    ckpts = sorted(globmod.glob(str(MODEL_DIR / 'rl_v3_checkpoint_epoch*.pt')))
    while len(ckpts) > 3:
        try:
            os.remove(ckpts.pop(0))
        except OSError:
            pass
    log.info(f"  Checkpoint saved: epoch {epoch}")


def load_latest_checkpoint(policy, optimizer):
    ckpts = sorted(globmod.glob(str(MODEL_DIR / 'rl_v3_checkpoint_epoch*.pt')))
    if not ckpts:
        log.info("No v3 checkpoint found, starting from scratch.")
        return 0, -np.inf, []
    ckpt_path = ckpts[-1]
    log.info(f"Resuming from {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch'] + 1, ckpt['best_pnl'], ckpt.get('history', [])
    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        return 0, -np.inf, []


def save_history(history):
    hist_path = MODEL_DIR / 'rl_v3_history.json'
    try:
        with open(str(hist_path), 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save history: {e}")


# ============================================================
# PPO TRAINING
# ============================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    returns = []
    gae = 0.0
    nv = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * nv - values[i].item()
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[i].item())
        nv = values[i].item()
    return returns


def ppo_update(policy, optimizer, states, actions, old_log_probs, returns, values,
               eps_clip=0.2, mini_batch_size=1024, ppo_epochs=4):
    """
    PPO policy gradient update.
    Uses the encoded states directly (no LSTM during update — we use the
    pre-computed LSTM outputs stored as values/log_probs during collection).
    """
    states_t = torch.stack(states).to(device)
    actions_t = torch.tensor(actions, device=device, dtype=torch.long)
    old_lps_t = torch.stack(old_log_probs).to(device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    values_t = torch.stack(values).to(device)
    advantages = returns_t - values_t.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states_t)
    bs = min(mini_batch_size, n)
    total_loss = 0.0

    # For the PPO update, we pass states through encoder + a single-step LSTM
    # This approximates the sequential policy while being batch-efficient
    policy.to(device)

    for _ in range(ppo_epochs):
        idx = torch.randperm(n, device=device)[:bs]

        # Encode batch of states
        batch_states = states_t[idx]  # (bs, 30)
        encoded = policy.encode(batch_states)  # (bs, hidden)
        # Single-step LSTM (no temporal context during PPO update — context
        # was captured during collection via log_probs)
        encoded_seq = encoded.unsqueeze(1)  # (bs, 1, hidden)
        logits, new_vals, _ = policy(encoded_seq, None)
        logits = logits.squeeze(1) if logits.dim() > 2 else logits

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_lps = dist.log_prob(actions_t[idx])
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_lps - old_lps_t[idx])
        s1 = ratio * advantages[idx]
        s2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[idx]

        actor_loss = -torch.min(s1, s2).mean()
        critic_loss = nn.MSELoss()(new_vals.squeeze(), returns_t[idx])
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    # Free GPU memory
    del states_t, actions_t, old_lps_t, returns_t, values_t, advantages
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / ppo_epochs


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    log.info("Loading data...")
    if not PRED_FILE.exists():
        log.error(f"Prediction file not found: {PRED_FILE}")
        sys.exit(1)

    pred_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in pred_data.files if k.endswith('_preds')))
    log.info(f"Found {len(dates)} dates in predictions")

    day_datas = []
    for date in dates:
        try:
            preds = pred_data[f'{date}_preds']
            mids = pred_data[f'{date}_mid']
        except KeyError:
            log.warning(f"  Skipping {date}: missing preds or mid")
            continue

        if len(preds) < 5000:
            log.warning(f"  Skipping {date}: only {len(preds)} bars")
            continue

        book_file = BOOK_DIR / f'{date}_book_tensors.npz'
        book = None
        if book_file.exists():
            try:
                book = np.load(str(book_file))['book_tensors']
            except Exception as e:
                log.warning(f"  {date}: failed to load book tensors: {e}")

        dd = precompute_day(preds, mids, book)
        if dd is not None:
            dd['date'] = date
            day_datas.append(dd)
            log.info(f"  {date}: {dd['n_steps']} steps, book={'yes' if book is not None else 'no'}")

    n_train = max(1, len(day_datas) - 7)
    train_days = day_datas[:n_train]
    eval_days = day_datas[n_train:]
    log.info(f"Train: {n_train} days, Eval: {len(eval_days)} days")
    return train_days, eval_days


# ============================================================
# TRAINING LOOP
# ============================================================

def train(epochs=500, reward_type='drawdown', resume=False):
    train_days, eval_days = load_data()

    policy = LSTMPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    start_epoch = 0
    best_pnl = -np.inf
    history = []

    if resume:
        start_epoch, best_pnl, history = load_latest_checkpoint(policy, optimizer)

    total_params = sum(p.numel() for p in policy.parameters())
    log.info(f"LSTMPolicy params: {total_params:,}")
    log.info(f"Training epochs {start_epoch} to {start_epoch + epochs - 1}, reward={reward_type}")
    log.info(f"Actions: SKIP=0, ENTER=1, EXIT=2 (3-action chase execution)")
    log.info(f"Entry: CNN z-score direction, fill prob {BASE_FILL_PROB:.0%}-{MAX_FILL_PROB:.0%}, "
             f"passive limit (0 spread)")
    log.info(f"Exit: market order, {SPREAD_TICKS} tick spread + {COMMISSION_TICKS*2:.3f} commission")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        epoch_pnl = []
        epoch_trades = 0
        epoch_fill_attempts = 0
        epoch_fill_successes = 0

        all_states, all_actions, all_rewards, all_lps, all_vals = [], [], [], [], []

        for dd in train_days:
            try:
                states, actions, rewards, lps, vals, stats = run_episode(
                    policy, dd, reward_type)
            except Exception as e:
                log.warning(f"  Episode failed for {dd.get('date', '?')}: {e}")
                continue

            epoch_pnl.append(stats['total_pnl'])
            epoch_trades += stats['n_trades']
            epoch_fill_attempts += stats['fill_attempts']
            epoch_fill_successes += stats['fill_successes']

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_lps.extend(lps)
            all_vals.extend(vals)

            # Free per-episode data immediately
            del states, actions, rewards, lps, vals, stats

        if len(all_states) < 100:
            log.warning(f"  Epoch {epoch}: too few transitions ({len(all_states)}), skipping update")
            continue

        # Compute GAE returns
        returns = compute_gae(all_rewards, all_vals)

        # PPO update
        try:
            avg_loss = ppo_update(policy, optimizer, all_states, all_actions,
                                  all_lps, returns, all_vals,
                                  mini_batch_size=1024, ppo_epochs=4)
        except Exception as e:
            log.error(f"  PPO update failed at epoch {epoch}: {e}")
            # Try to recover
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            continue

        # Free memory aggressively after PPO update
        del all_states, all_actions, all_rewards, all_lps, all_vals, returns
        gc.collect()
        torch.cuda.empty_cache()

        # Stats
        avg_pnl = np.mean(epoch_pnl) if epoch_pnl else 0
        total_epoch_pnl = sum(epoch_pnl) if epoch_pnl else 0
        trades_per_day = epoch_trades / max(len(train_days), 1)
        fill_rate = epoch_fill_successes / max(epoch_fill_attempts, 1)
        dt = time.time() - t0

        log.info(f"  Epoch {epoch:>3d}: Avg P&L ${avg_pnl:>8,.2f} | Total ${total_epoch_pnl:>10,.2f} | "
                 f"Trades {trades_per_day:.1f}/day | FillRate {fill_rate:.1%} | "
                 f"Fills {epoch_fill_successes}/{epoch_fill_attempts} | "
                 f"Best ${best_pnl:>8,.2f} | {dt:.1f}s")

        # History
        history.append({
            'epoch': epoch,
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(total_epoch_pnl, 2),
            'trades_per_day': round(trades_per_day, 1),
            'fill_rate': round(fill_rate, 4),
            'fill_attempts': epoch_fill_attempts,
            'fill_successes': epoch_fill_successes,
            'loss': round(avg_loss, 4),
            'time_sec': round(dt, 1),
        })

        # Save best model
        if avg_pnl > best_pnl:
            best_pnl = avg_pnl
            try:
                torch.save(policy.state_dict(), str(MODEL_DIR / f'rl_v3_best_{reward_type}.pt'))
                log.info(f"  ** New best: ${best_pnl:,.2f} **")
            except Exception as e:
                log.warning(f"  Failed to save best model: {e}")

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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(policy, eval_days, reward_type):
    log.info(f"\n--- Evaluation ({len(eval_days)} held-out days) ---")
    eval_total = 0
    eval_trades = 0
    eval_fill_attempts = 0
    eval_fill_successes = 0

    for dd in eval_days:
        try:
            states, _, _, _, _, stats = run_episode(policy, dd, reward_type, deterministic=True)
            del states
        except Exception as e:
            log.warning(f"  Eval failed for {dd.get('date', '?')}: {e}")
            continue

        eval_total += stats['total_pnl']
        eval_trades += stats['n_trades']
        eval_fill_attempts += stats['fill_attempts']
        eval_fill_successes += stats['fill_successes']

        log.info(f"  {dd['date']}: P&L ${stats['total_pnl']:>8,.2f} | "
                 f"{stats['n_trades']} trades | "
                 f"FillRate {stats['fill_rate']:.1%} | "
                 f"Max DD ${stats['max_dd']:,.2f}")

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    n_eval = max(len(eval_days), 1)
    overall_fill_rate = eval_fill_successes / max(eval_fill_attempts, 1)
    log.info(f"  EVAL TOTAL: ${eval_total:,.2f} | {eval_trades} trades over {len(eval_days)} days")
    log.info(f"  EVAL AVG: ${eval_total / n_eval:,.2f}/day | "
             f"{eval_trades / n_eval:.1f} trades/day | "
             f"FillRate {overall_fill_rate:.1%}")


def eval_only(reward_type):
    """Load best model and evaluate on all data."""
    _, eval_days = load_data()
    policy = LSTMPolicy().to(device)
    model_path = MODEL_DIR / f'rl_v3_best_{reward_type}.pt'
    if not model_path.exists():
        log.error(f"No model found at {model_path}")
        return
    try:
        policy.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return
    evaluate(policy, eval_days, reward_type)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Execution Agent v3 — 3-Action Chase Execution')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--reward', default='drawdown',
                        choices=['raw', 'drawdown', 'sharpe_like', 'calmar'])
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluate best model only')
    args = parser.parse_args()

    if args.eval:
        eval_only(args.reward)
    else:
        log.info(f"\n{'='*70}")
        log.info(f"RL Execution Agent v3 — 3-Action Chase (SKIP/ENTER/EXIT)")
        log.info(f"Reward: {args.reward} | Epochs: {args.epochs} | Device: {device}")
        log.info(f"{'='*70}")
        best = train(args.epochs, args.reward, args.resume)
        log.info(f"DONE: {args.reward} | Best avg P&L ${best:,.2f}")
