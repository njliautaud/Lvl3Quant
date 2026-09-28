#!/usr/bin/env python3
"""
RL Execution Agent v3.3 — EPISODIC REWARD (End-of-Day Total P&L)
=================================================================
Changes from v3.2:
  1. EPISODIC REWARD: All step rewards = 0. At END of episode, reward = total_day_pnl - DAILY_COST.
     No per-trade reward shaping. The agent must learn to maximize total daily P&L.
  2. DAILY_COST = $6.76 ($142/month / 21 trading days) deducted from end-of-day reward.
  3. Per-trade P&L still tracked for logging but NOT used as step reward.

Architecture unchanged from v3.2:
  - LSTM policy: 30→256→LN→128→LSTM(64)→actor/critic
  - 3 actions: SKIP=0, ENTER=1, EXIT=2
  - SCAN_STEP=300 (30s decisions), LR=3e-4
  - GPU forward passes during episode collection, CPU storage
  - All OOM fixes (gc.collect, empty_cache, detach LSTM states)

Usage:
    python rl_v3_3.py --epochs 500
    python rl_v3_3.py --resume --epochs 500
    python rl_v3_3.py --eval
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
COMMISSION_TICKS = 0.376          # Per side commission in ticks

# Cost model (HC #231(A)): commission only — NO spread crossing cost
# Spread terms below are kept at 0.0 so legacy arithmetic still resolves to 0.376
CHASE_ENTRY_SPREAD = 0.0          # HC #231(A): no spread cost
MARKET_EXIT_SPREAD = 0.0          # HC #231(A): no spread cost
TOTAL_ENTRY_COST_CHASE = CHASE_ENTRY_SPREAD + COMMISSION_TICKS   # 0.376 ticks
TOTAL_EXIT_COST = MARKET_EXIT_SPREAD + COMMISSION_TICKS           # 0.376 ticks
TOTAL_RT_CHASE = TOTAL_ENTRY_COST_CHASE + TOTAL_EXIT_COST         # 0.752 ticks

# For tracking alternative cost scenarios (HC #231(A): all equal to commission):
MARKET_ENTRY_SPREAD = 0.0
PASSIVE_ENTRY_SPREAD = 0.0       # Passive fill = no spread on entry
TOTAL_ENTRY_COST_MARKET = MARKET_ENTRY_SPREAD + COMMISSION_TICKS  # 0.376 ticks
TOTAL_ENTRY_COST_PASSIVE = PASSIVE_ENTRY_SPREAD + COMMISSION_TICKS  # 0.376 ticks

# Daily capital cost: AMP R|API $100 + API+ $25 + CME MBO $17 = $142/mo / 21 trading days
DAILY_COST = 6.76
DAILY_COST_TICKS = DAILY_COST / TICK_VAL  # Convert to ticks for reward signal

SCAN_STEP = 10                    # Decision every 10 bars (~5000 events = ~30s at stride 500)
MAX_HOLD_STEPS = 1200 // SCAN_STEP  # ~60 min max hold = 120 steps
LSTM_LOOKBACK = 15                # 15 decision steps context

# ── Actions ──
SKIP = 0     # Don't enter (flat) or keep holding (in position)
ENTER = 1    # Enter in CNN's predicted direction
EXIT = 2     # Exit position
N_ACTIONS = 3

# ── Feature dims ──
STATIC_DIM = 24
DYNAMIC_DIM = 6   # in_position, direction, unrealized_pnl, hold_time, max_favorable, max_adverse
STATE_DIM = STATIC_DIM + DYNAMIC_DIM  # 30

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('rl_v33')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'rl_v33_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


# ============================================================
# PRECOMPUTE STATIC FEATURES (same as v3.1)
# ============================================================

def precompute_day(preds, mids, book=None):
    """Precompute 24 static features for one day, subsampled at SCAN_STEP intervals."""
    n = len(preds)
    # Adapt window based on data size — prediction-level data is ~24K/day
    # vs raw event-level data which is ~12M/day
    if n < 100000:  # Prediction-level data (subsampled)
        w = 500
        warmup = min(w, n // 5)
        cooldown = min(250, n // 10)
    else:  # Raw event-level data
        w = 3000
        warmup = 18000
        cooldown = 9000

    # Subsample indices
    indices = np.arange(warmup, n - cooldown, SCAN_STEP)
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
# LSTM POLICY NETWORK (unchanged from v3.1)
# ============================================================

class LSTMPolicy(nn.Module):
    """Actor-Critic with LSTM temporal context.
    Architecture: 30 -> 256 -> LayerNorm -> ReLU -> 128 -> LSTM(64) -> actor/critic
    """

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128, lstm_hidden=64):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # Encode current state: 30 -> 256 -> LN -> 128
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden),
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
        lstm_out, hx = self.lstm(encoded_seq, hx)
        last = lstm_out[:, -1, :]
        logits = self.actor(last)
        value = self.critic(last)
        return logits, value, hx

    def encode(self, state):
        return self.encoder(state)

    def init_hidden(self, batch_size=1, dev=None):
        dev = dev or device
        h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=dev)
        return (h0.detach(), c0.detach())


# ============================================================
# EPISODE RUNNER — EPISODIC REWARD (end-of-day P&L)
# ============================================================

def run_episode(policy, day_data, deterministic=False):
    """
    Run one episode (one trading day). Policy stays on GPU for forward passes,
    results stored on CPU. No computation graph accumulates.

    v3.3 EPISODIC REWARD:
      - ALL step rewards = 0 (SKIP, ENTER, HOLD, EXIT — all zero)
      - Track cumulative P&L throughout the episode
      - At the LAST step: reward = total_day_chase_pnl_ticks - DAILY_COST_TICKS
      - Per-trade P&L still tracked for logging but NOT used as step reward
    """
    feats = torch.tensor(day_data['features'], dtype=torch.float32)  # (T, 24) CPU
    mids = day_data['mids']
    z = day_data['z_scores']
    T = day_data['n_steps']

    # Policy stays on GPU — forward passes on GPU, results stored on CPU
    policy.to(device)
    policy.eval()

    # Transition storage (all CPU)
    all_states = []
    actions_list = []
    rewards_list = []
    log_probs_list = []
    values_list = []

    # ---- Position State ----
    in_position = False
    pos_dir = 0
    entry_price = 0.0
    entry_step = 0
    max_fav = 0.0
    max_adv = 0.0
    total_chase_pnl_ticks = 0.0   # Cumulative chase P&L in ticks (for episodic reward)
    total_chase_pnl = 0.0         # In USD for logging
    total_market_pnl = 0.0
    total_passive_pnl = 0.0
    trades = []
    peak = 0.0
    max_dd = 0.0

    # LSTM state — on GPU for forward passes
    hx = policy.init_hidden(batch_size=1, dev=device)
    hx = (hx[0].detach(), hx[1].detach())

    # Pre-allocate dynamic feature buffer (CPU)
    dyn_buf = torch.zeros(6, dtype=torch.float32)

    # Action mask tensors (pre-allocate on GPU)
    mask_flat = torch.tensor([ 0.0,  0.0, -1e8], device=device)   # not in position: SKIP+ENTER ok
    mask_pos  = torch.tensor([ 0.0, -1e8,  0.0], device=device)   # in position: SKIP+EXIT ok

    with torch.no_grad():
        for t in range(T):
            # ---- Build 6 dynamic features on CPU ----
            if in_position:
                unreal = float((mids[t] - entry_price) / TICK * pos_dir)
                hold_norm = float((t - entry_step) / max(T, 1))
                dyn_buf[0] = 1.0
                dyn_buf[1] = float(pos_dir)
                dyn_buf[2] = float(max(-1.0, min(1.0, unreal / 50)))
                dyn_buf[3] = float(min(hold_norm, 2.0))
                dyn_buf[4] = float(min(max_fav / 30, 2.0))
                dyn_buf[5] = float(max(max_adv / 30, -2.0))
            else:
                dyn_buf.zero_()

            state = torch.cat([feats[t], dyn_buf])  # (30,) CPU

            # GPU forward pass: send state to GPU, run encoder + LSTM, get results back to CPU
            state_gpu = state.unsqueeze(0).to(device)   # (1, 30) GPU
            encoded = policy.encode(state_gpu)           # (1, 128) GPU
            encoded_seq = encoded.unsqueeze(1)           # (1, 1, 128) GPU
            logits, value, hx = policy(encoded_seq, hx)  # GPU
            hx = (hx[0].detach(), hx[1].detach())       # detach to prevent graph accumulation

            # Action masking on GPU
            logits_sq = logits.squeeze(0)                # (3,) GPU
            mask = mask_flat if not in_position else mask_pos
            masked_logits = logits_sq + mask
            probs = torch.softmax(masked_logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            lp = torch.log(probs[action] + 1e-8)

            # Move results to CPU immediately
            a = action.item()
            actions_list.append(a)
            log_probs_list.append(lp.cpu())
            values_list.append(value.squeeze().cpu())
            all_states.append(state.detach())  # already CPU

            # ---- Execute action ----
            # EPISODIC: all step rewards = 0, reward assigned at end
            reward = 0.0

            if not in_position:
                if a == ENTER:
                    direction = 1 if z[t] > 0 else -1
                    in_position = True
                    pos_dir = direction
                    entry_price = float(mids[t])
                    entry_step = t
                    max_fav = 0.0
                    max_adv = 0.0

            elif in_position:
                if a == EXIT:
                    raw_pnl_ticks = float((mids[t] - entry_price) / TICK * pos_dir)

                    # Chase P&L: 0.5t entry + 1.0t exit + 0.752t commission = 2.252t RT
                    chase_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_CHASE - TOTAL_EXIT_COST

                    # Market P&L: 1.0t entry + 1.0t exit + 0.752t commission = 2.752t RT
                    market_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_MARKET - TOTAL_EXIT_COST

                    # Passive P&L: 0t entry + 1.0t exit + 0.752t commission = 1.752t RT
                    passive_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_PASSIVE - TOTAL_EXIT_COST

                    # Track cumulative P&L (for episodic reward at end)
                    total_chase_pnl_ticks += chase_pnl
                    total_chase_pnl += chase_pnl * TICK_VAL
                    total_market_pnl += market_pnl * TICK_VAL
                    total_passive_pnl += passive_pnl * TICK_VAL

                    trades.append({
                        'chase_pnl_ticks': round(chase_pnl, 3),
                        'market_pnl_ticks': round(market_pnl, 3),
                        'passive_pnl_ticks': round(passive_pnl, 3),
                        'chase_pnl_usd': round(chase_pnl * TICK_VAL, 2),
                        'market_pnl_usd': round(market_pnl * TICK_VAL, 2),
                        'passive_pnl_usd': round(passive_pnl * TICK_VAL, 2),
                        'hold_steps': t - entry_step,
                        'direction': pos_dir,
                        'max_fav': round(max_fav, 2),
                        'max_adv': round(max_adv, 2),
                    })
                    in_position = False
                    pos_dir = 0

                # SKIP while in position: reward = 0 (episodic)

            # ---- Update position tracking ----
            if in_position:
                cur_pnl = float((mids[t] - entry_price) / TICK * pos_dir)
                max_fav = float(max(max_fav, cur_pnl))
                max_adv = float(min(max_adv, cur_pnl))

                # Force exit at max hold time or end of day
                force_exit = (t - entry_step) >= MAX_HOLD_STEPS or t == T - 1
                if force_exit:
                    raw_pnl_ticks = cur_pnl

                    chase_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_CHASE - TOTAL_EXIT_COST
                    market_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_MARKET - TOTAL_EXIT_COST
                    passive_pnl = raw_pnl_ticks - TOTAL_ENTRY_COST_PASSIVE - TOTAL_EXIT_COST

                    # Track cumulative P&L (for episodic reward at end)
                    total_chase_pnl_ticks += chase_pnl
                    total_chase_pnl += chase_pnl * TICK_VAL
                    total_market_pnl += market_pnl * TICK_VAL
                    total_passive_pnl += passive_pnl * TICK_VAL

                    trades.append({
                        'chase_pnl_ticks': round(chase_pnl, 3),
                        'market_pnl_ticks': round(market_pnl, 3),
                        'passive_pnl_ticks': round(passive_pnl, 3),
                        'chase_pnl_usd': round(chase_pnl * TICK_VAL, 2),
                        'market_pnl_usd': round(market_pnl * TICK_VAL, 2),
                        'passive_pnl_usd': round(passive_pnl * TICK_VAL, 2),
                        'hold_steps': t - entry_step,
                        'direction': pos_dir,
                        'max_fav': round(max_fav, 2),
                        'max_adv': round(max_adv, 2),
                        'forced': True,
                    })
                    in_position = False
                    pos_dir = 0

            # Track drawdown (on chase P&L)
            peak = max(peak, total_chase_pnl)
            max_dd = max(max_dd, peak - total_chase_pnl)

            # EPISODIC REWARD: assign total daily P&L minus cost at LAST step only
            if t == T - 1:
                reward = total_chase_pnl_ticks - DAILY_COST_TICKS

            rewards_list.append(reward)

    # Policy already on GPU — just switch back to train mode
    policy.train()

    stats = {
        'total_chase_pnl': total_chase_pnl,
        'total_market_pnl': total_market_pnl,
        'total_passive_pnl': total_passive_pnl,
        'total_chase_pnl_ticks': total_chase_pnl_ticks,
        'daily_cost': DAILY_COST,
        'net_daily_pnl': total_chase_pnl - DAILY_COST,
        'trades': trades,
        'max_dd': max_dd,
        'n_trades': len(trades),
    }

    return all_states, actions_list, rewards_list, log_probs_list, values_list, stats


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(policy, optimizer, epoch, best_pnl, history):
    ckpt_path = MODEL_DIR / f'rl_v33_checkpoint_epoch{epoch}.pt'
    try:
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_pnl': best_pnl,
            'history': history,
            'reward_type': 'episodic',
        }, str(ckpt_path))
    except Exception as e:
        log.warning(f"Failed to save checkpoint: {e}")
        return

    # Keep only last 3 checkpoints
    ckpts = sorted(globmod.glob(str(MODEL_DIR / 'rl_v33_checkpoint_epoch*.pt')))
    while len(ckpts) > 3:
        try:
            os.remove(ckpts.pop(0))
        except OSError:
            pass
    log.info(f"  Checkpoint saved: epoch {epoch}")


def load_latest_checkpoint(policy, optimizer):
    ckpts = sorted(globmod.glob(str(MODEL_DIR / 'rl_v33_checkpoint_epoch*.pt')))
    if not ckpts:
        log.info("No v3.3 checkpoint found, starting from scratch.")
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
    hist_path = MODEL_DIR / 'rl_v33_history.json'
    try:
        with open(str(hist_path), 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save history: {e}")


# ============================================================
# PPO TRAINING (unchanged from v3.1)
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
               eps_clip=0.2, mini_batch_size=2048, ppo_epochs=4):
    """PPO policy gradient update."""
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

    # Policy already on GPU (stays there always)

    for _ in range(ppo_epochs):
        idx = torch.randperm(n, device=device)[:bs]

        batch_states = states_t[idx]
        encoded = policy.encode(batch_states)
        encoded_seq = encoded.unsqueeze(1)
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
# DATA LOADING (unchanged from v3.1)
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

def train(epochs=500, resume=False):
    train_days, eval_days = load_data()

    policy = LSTMPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    start_epoch = 0
    best_pnl = -np.inf
    history = []

    if resume:
        start_epoch, best_pnl, history = load_latest_checkpoint(policy, optimizer)

    total_params = sum(p.numel() for p in policy.parameters())
    log.info(f"LSTMPolicy params: {total_params:,}")
    log.info(f"Training epochs {start_epoch} to {start_epoch + epochs - 1}")
    log.info("v3.3: EPISODIC REWARD — total daily P&L minus $6.76 capital cost")
    log.info("No per-trade reward shaping. End-of-day settlement only.")
    log.info(f"Daily cost: ${6.76} (AMP R|API $100 + API+ $25 + CME MBO $17 = $142/mo / 21 days)")
    log.info(f"Actions: SKIP=0, ENTER=1, EXIT=2 (3-action, instant fill)")
    log.info(f"Chase cost model: entry {CHASE_ENTRY_SPREAD}t + exit {MARKET_EXIT_SPREAD}t + comm {COMMISSION_TICKS*2:.3f}t = {TOTAL_RT_CHASE:.3f}t RT")
    log.info(f"  vs Market: {TOTAL_ENTRY_COST_MARKET + TOTAL_EXIT_COST:.3f}t RT | Passive: {TOTAL_ENTRY_COST_PASSIVE + TOTAL_EXIT_COST:.3f}t RT")
    log.info(f"LR: 3e-4, SCAN_STEP: {SCAN_STEP} (30s), Reward: episodic (end-of-day P&L - ${DAILY_COST})")

    best_daily_pnl = -np.inf

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        epoch_chase_pnl = []
        epoch_market_pnl = []
        epoch_passive_pnl = []
        epoch_net_pnl = []
        epoch_trades = 0

        all_states, all_actions, all_rewards, all_lps, all_vals = [], [], [], [], []

        for dd in train_days:
            try:
                states, actions, rewards, lps, vals, stats = run_episode(policy, dd)
            except Exception as e:
                log.warning(f"  Episode failed for {dd.get('date', '?')}: {e}")
                continue

            epoch_chase_pnl.append(stats['total_chase_pnl'])
            epoch_market_pnl.append(stats['total_market_pnl'])
            epoch_passive_pnl.append(stats['total_passive_pnl'])
            epoch_net_pnl.append(stats['net_daily_pnl'])
            epoch_trades += stats['n_trades']

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_lps.extend(lps)
            all_vals.extend(vals)

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
                                  mini_batch_size=2048, ppo_epochs=4)
        except Exception as e:
            log.error(f"  PPO update failed at epoch {epoch}: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            continue

        # Free memory
        del all_states, all_actions, all_rewards, all_lps, all_vals, returns
        gc.collect()
        torch.cuda.empty_cache()

        # Stats
        avg_chase = np.mean(epoch_chase_pnl) if epoch_chase_pnl else 0
        avg_market = np.mean(epoch_market_pnl) if epoch_market_pnl else 0
        avg_passive = np.mean(epoch_passive_pnl) if epoch_passive_pnl else 0
        avg_net = np.mean(epoch_net_pnl) if epoch_net_pnl else 0
        total_chase = sum(epoch_chase_pnl) if epoch_chase_pnl else 0
        trades_per_day = epoch_trades / max(len(train_days), 1)
        dt = time.time() - t0

        # Track best daily P&L
        if avg_net > best_daily_pnl:
            best_daily_pnl = avg_net

        log.info(f"  Epoch {epoch:>3d}: NetPnL ${avg_net:>8,.2f}/day (chase ${avg_chase:>8,.2f} - cost ${DAILY_COST}) | "
                 f"Trades {trades_per_day:.1f}/day | Chase/Mkt/Pas ${avg_chase:>7,.2f}/${avg_market:>7,.2f}/${avg_passive:>7,.2f} | "
                 f"Loss {avg_loss:.4f} | Best ${best_daily_pnl:>8,.2f} | {dt:.1f}s")

        # History
        history.append({
            'epoch': epoch,
            'avg_chase_pnl': round(avg_chase, 2),
            'avg_market_pnl': round(avg_market, 2),
            'avg_passive_pnl': round(avg_passive, 2),
            'avg_net_pnl': round(avg_net, 2),
            'total_chase_pnl': round(total_chase, 2),
            'trades_per_day': round(trades_per_day, 1),
            'loss': round(avg_loss, 4),
            'time_sec': round(dt, 1),
            'best_daily_pnl': round(best_daily_pnl, 2),
        })

        # Save best model (based on net daily P&L — chase P&L minus daily cost)
        if avg_net > best_pnl:
            best_pnl = avg_net
            try:
                torch.save(policy.state_dict(), str(MODEL_DIR / 'rl_v33_best.pt'))
                log.info(f"  ** New best: ${best_pnl:,.2f}/day net **")
            except Exception as e:
                log.warning(f"  Failed to save best model: {e}")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(policy, optimizer, epoch, best_pnl, history)

        # Save history every epoch
        save_history(history)

    # Final checkpoint
    save_checkpoint(policy, optimizer, epoch, best_pnl, history)

    # Evaluate
    evaluate(policy, eval_days)

    return best_pnl


# ============================================================
# EVALUATION
# ============================================================

def evaluate(policy, eval_days):
    log.info(f"\n--- Evaluation ({len(eval_days)} held-out days) ---")
    eval_chase_total = 0
    eval_market_total = 0
    eval_passive_total = 0
    eval_net_total = 0
    eval_trades = 0

    for dd in eval_days:
        try:
            states, _, _, _, _, stats = run_episode(policy, dd, deterministic=True)
            del states
        except Exception as e:
            log.warning(f"  Eval failed for {dd.get('date', '?')}: {e}")
            continue

        eval_chase_total += stats['total_chase_pnl']
        eval_market_total += stats['total_market_pnl']
        eval_passive_total += stats['total_passive_pnl']
        eval_net_total += stats['net_daily_pnl']
        eval_trades += stats['n_trades']

        log.info(f"  {dd['date']}: NetPnL ${stats['net_daily_pnl']:>8,.2f} (chase ${stats['total_chase_pnl']:>8,.2f} - cost ${DAILY_COST}) | "
                 f"Market ${stats['total_market_pnl']:>8,.2f} | Passive ${stats['total_passive_pnl']:>8,.2f} | "
                 f"{stats['n_trades']} trades | Max DD ${stats['max_dd']:,.2f}")

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    n_eval = max(len(eval_days), 1)
    log.info(f"  EVAL TOTAL:  Net ${eval_net_total:,.2f} | Chase ${eval_chase_total:,.2f} | Market ${eval_market_total:,.2f} | "
             f"Passive ${eval_passive_total:,.2f} | {eval_trades} trades over {len(eval_days)} days")
    log.info(f"  EVAL AVG:    Net ${eval_net_total / n_eval:,.2f}/day | Chase ${eval_chase_total / n_eval:,.2f}/day | "
             f"Market ${eval_market_total / n_eval:,.2f}/day | "
             f"Passive ${eval_passive_total / n_eval:,.2f}/day | "
             f"{eval_trades / n_eval:.1f} trades/day")
    log.info(f"  COST GAPS:  Chase-Market ${(eval_chase_total - eval_market_total):,.2f} | "
             f"Passive-Chase ${(eval_passive_total - eval_chase_total):,.2f}")


def eval_only():
    """Load best model and evaluate on held-out data."""
    _, eval_days = load_data()
    policy = LSTMPolicy().to(device)
    model_path = MODEL_DIR / 'rl_v33_best.pt'
    if not model_path.exists():
        log.error(f"No model found at {model_path}")
        return
    try:
        policy.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return
    evaluate(policy, eval_days)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL v3.3 — Episodic Reward (End-of-Day P&L)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluate best model only')
    args = parser.parse_args()

    if args.eval:
        eval_only()
    else:
        log.info(f"\n{'='*70}")
        log.info(f"RL v3.3: EPISODIC REWARD — End-of-Day Total P&L")
        log.info(f"  Changes from v3.2:")
        log.info(f"  1. EPISODIC reward — all step rewards=0, end-of-day P&L minus daily cost")
        log.info(f"  2. Daily cost: ${DAILY_COST} (AMP $142/mo / 21 trading days)")
        log.info(f"  3. Chase costs ({TOTAL_RT_CHASE:.3f}t RT) unchanged from v3.2")
        log.info(f"Epochs: {args.epochs} | Device: {device}")
        log.info(f"{'='*70}")
        best = train(args.epochs, args.resume)
        log.info(f"DONE: Best avg Net Daily P&L ${best:,.2f}")
