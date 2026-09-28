#!/usr/bin/env python3
"""
RL Growth Allocator — PPO-based Dynamic Asset Allocation
=========================================================
GPU-accelerated reinforcement learning agent that learns optimal allocation
between growth assets (TQQQ, BTC-USD, SVXY, QQQ, CASH) based on market conditions.

Hypothesis: Simple rules (200MA, vol-targeting) work individually. Can an RL agent
learn to DYNAMICALLY ALLOCATE between them better than static weights?

Design:
  - State: ~30 market condition features (VIX, RSI, credit spread, vol, momentum, etc.)
  - Action: 20 discrete preset allocations across 5 assets
  - Reward: Sharpe-adjusted daily return with drawdown penalty
  - Algorithm: PPO (Proximal Policy Optimization) — stable for discrete actions
  - Walk-forward: sliding 252d train, 63d test

HC compliance: sliding window only, risk-adjusted metrics (Sharpe/Sortino/PF/WR).
"""

import json
import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

import platform
if platform.system() == 'Windows':
    _ROOT = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _ROOT = Path('/home/jupiter/Lvl3Quant')
OUTPUT_DIR = _ROOT / 'output' / 'growth_research' / 'rl_allocator'
CACHE_DIR = _ROOT / 'output' / 'growth_research' / 'cache'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward params
TRAIN_DAYS = 252        # 1 year sliding train
TEST_DAYS = 63          # 1 quarter OOT

# PPO hyperparameters
PPO_CLIP = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
GAMMA = 0.99            # discount factor
GAE_LAMBDA = 0.95       # GAE lambda
LR = 3e-4
EPOCHS_PER_UPDATE = 4   # PPO epochs per rollout
MINIBATCH_SIZE = 64
N_EPISODES = 100        # episodes per fold
MAX_GRAD_NORM = 0.5
HIDDEN_DIM = 128
DRAWDOWN_PENALTY_THRESHOLD = 0.10  # penalize drawdowns > 10%
DRAWDOWN_PENALTY_COEFF = 2.0

# Tradeable assets
ASSETS = ['TQQQ', 'BTC-USD', 'SVXY', 'QQQ']
CASH_LABEL = 'CASH'
ALL_ASSETS = ASSETS + [CASH_LABEL]

# Preset allocation portfolios (action space)
# Each tuple: (TQQQ%, BTC%, SVXY%, QQQ%, CASH%)
ALLOCATIONS = [
    # Pure single-asset
    (1.0, 0.0, 0.0, 0.0, 0.0),   # 0:  100% TQQQ
    (0.0, 1.0, 0.0, 0.0, 0.0),   # 1:  100% BTC
    (0.0, 0.0, 1.0, 0.0, 0.0),   # 2:  100% SVXY
    (0.0, 0.0, 0.0, 1.0, 0.0),   # 3:  100% QQQ
    (0.0, 0.0, 0.0, 0.0, 1.0),   # 4:  100% CASH

    # 50/50 pairs
    (0.5, 0.5, 0.0, 0.0, 0.0),   # 5:  50 TQQQ / 50 BTC
    (0.5, 0.0, 0.5, 0.0, 0.0),   # 6:  50 TQQQ / 50 SVXY
    (0.5, 0.0, 0.0, 0.5, 0.0),   # 7:  50 TQQQ / 50 QQQ
    (0.5, 0.0, 0.0, 0.0, 0.5),   # 8:  50 TQQQ / 50 CASH
    (0.0, 0.5, 0.5, 0.0, 0.0),   # 9:  50 BTC / 50 SVXY
    (0.0, 0.5, 0.0, 0.5, 0.0),   # 10: 50 BTC / 50 QQQ
    (0.0, 0.5, 0.0, 0.0, 0.5),   # 11: 50 BTC / 50 CASH
    (0.0, 0.0, 0.5, 0.5, 0.0),   # 12: 50 SVXY / 50 QQQ
    (0.0, 0.0, 0.0, 0.5, 0.5),   # 13: 50 QQQ / 50 CASH

    # Diversified 3-way
    (0.34, 0.33, 0.33, 0.0, 0.0),  # 14: equal TQQQ/BTC/SVXY
    (0.34, 0.33, 0.0, 0.33, 0.0),  # 15: equal TQQQ/BTC/QQQ
    (0.25, 0.25, 0.25, 0.25, 0.0), # 16: equal all growth
    (0.20, 0.20, 0.20, 0.20, 0.20),# 17: equal weight all 5

    # Conservative / risk-off tilts
    (0.30, 0.0, 0.0, 0.30, 0.40),  # 18: 30 TQQQ / 30 QQQ / 40 CASH
    (0.0, 0.25, 0.0, 0.25, 0.50),  # 19: 25 BTC / 25 QQQ / 50 CASH
]

N_ACTIONS = len(ALLOCATIONS)
ALLOC_ARRAY = np.array(ALLOCATIONS, dtype=np.float32)  # (N_ACTIONS, 5)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_all_data(start='2015-01-01', end=None):
    """Download all tickers needed for features + targets."""
    if end is None:
        end = dt.date.today().isoformat()

    cache_file = CACHE_DIR / f'rl_allocator_data_{end}.parquet'
    if cache_file.exists():
        print(f"[DATA] Loading cached data from {cache_file.name}")
        return pd.read_parquet(cache_file)

    tickers = {
        # Feature tickers
        'SPY': 'SPY',
        'QQQ': 'QQQ',
        'VIX': '^VIX',
        'TLT': 'TLT',
        'SHY': 'SHY',
        'IEF': 'IEF',
        'HYG': 'HYG',
        'GLD': 'GLD',
        'IWM': 'IWM',
        # Target asset tickers
        'TQQQ': 'TQQQ',
        'BTC-USD': 'BTC-USD',
        'SVXY': 'SVXY',
    }

    prices = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                prices[name] = df['Close']
            else:
                print(f"    WARNING: {name} has only {len(df)} rows, skipping")
        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")

    price_df = pd.DataFrame(prices)
    price_df = price_df.ffill(limit=5)

    price_df.to_parquet(cache_file)
    print(f"[DATA] Saved {len(price_df)} rows, {len(price_df.columns)} columns to cache")
    return price_df


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build ~30 market condition features for the RL state space."""
    feat = pd.DataFrame(index=raw.index)

    def safe_sma(s, n):
        return s.rolling(n, min_periods=max(1, n // 2)).mean()

    def rsi(s, n=14):
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # --- VIX features ---
    if 'VIX' in raw.columns:
        feat['vix_level'] = raw['VIX']
        feat['vix_percentile_63d'] = raw['VIX'].rolling(63).rank(pct=True)
        feat['vix_percentile_252d'] = raw['VIX'].rolling(252).rank(pct=True)

    # --- SPY features ---
    if 'SPY' in raw.columns:
        spy = raw['SPY']
        spy_ret = spy.pct_change()

        feat['spy_rsi_14'] = rsi(spy, 14)

        # Distance from 200MA (key signal)
        sma200 = safe_sma(spy, 200)
        feat['spy_dist_200ma'] = (spy / sma200) - 1

        # Realized vol at multiple windows
        feat['spy_rvol_5d'] = spy_ret.rolling(5).std() * np.sqrt(252)
        feat['spy_rvol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
        feat['spy_rvol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252)

        # Vol ratio (short vs long) — rising = vol expansion
        feat['vol_ratio_5_63'] = feat['spy_rvol_5d'] / (feat['spy_rvol_63d'] + 1e-10)

        # Drawdown from rolling max
        rolling_max = spy.rolling(252, min_periods=1).max()
        feat['spy_drawdown'] = (spy / rolling_max) - 1

        # Momentum
        feat['spy_mom_21d'] = spy.pct_change(21)
        feat['spy_mom_63d'] = spy.pct_change(63)

    # --- Credit spread proxy (HYG/IEF) ---
    if 'HYG' in raw.columns and 'IEF' in raw.columns:
        credit_ratio = raw['HYG'] / raw['IEF']
        feat['credit_spread_ratio'] = credit_ratio
        feat['credit_spread_21d_chg'] = credit_ratio.pct_change(21)

    # --- Yield curve proxy (TLT/SHY) ---
    if 'TLT' in raw.columns and 'SHY' in raw.columns:
        yield_curve = raw['TLT'] / raw['SHY']
        feat['yield_curve_proxy'] = yield_curve
        feat['yield_curve_slope'] = yield_curve.pct_change(21)

    # --- BTC momentum ---
    if 'BTC-USD' in raw.columns:
        btc = raw['BTC-USD']
        feat['btc_mom_21d'] = btc.pct_change(21)
        feat['btc_mom_63d'] = btc.pct_change(63)
        feat['btc_rvol_20d'] = btc.pct_change().rolling(20).std() * np.sqrt(365)

    # --- TQQQ momentum ---
    if 'TQQQ' in raw.columns:
        feat['tqqq_mom_21d'] = raw['TQQQ'].pct_change(21)
        feat['tqqq_mom_63d'] = raw['TQQQ'].pct_change(63)

    # --- Term structure proxy (VIX vs realized vol) ---
    if 'VIX' in raw.columns and 'SPY' in raw.columns:
        rv21 = raw['SPY'].pct_change().rolling(21).std() * np.sqrt(252)
        feat['vix_vs_rvol'] = raw['VIX'] / (rv21 * 100 + 1e-10)

    # --- Breadth proxy (IWM/SPY) ---
    if 'IWM' in raw.columns and 'SPY' in raw.columns:
        iwm_spy = raw['IWM'] / raw['SPY']
        feat['breadth_ratio'] = iwm_spy
        feat['breadth_21d_chg'] = iwm_spy.pct_change(21)

    # --- Gold as risk-off indicator ---
    if 'GLD' in raw.columns:
        feat['gold_mom_21d'] = raw['GLD'].pct_change(21)

    # --- Calendar features (cyclical encoding) ---
    feat['month_sin'] = np.sin(2 * np.pi * feat.index.month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * feat.index.month / 12)

    print(f"[FEATURES] Built {len(feat.columns)} features")
    return feat


# ══════════════════════════════════════════════════════════════
# 3. DAILY ASSET RETURNS
# ══════════════════════════════════════════════════════════════

def compute_asset_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for each tradeable asset. CASH = 0."""
    returns = pd.DataFrame(index=raw.index)
    for asset in ASSETS:
        if asset in raw.columns:
            returns[asset] = raw[asset].pct_change()
        else:
            returns[asset] = 0.0
    returns[CASH_LABEL] = 0.0  # risk-free approximation
    returns = returns.fillna(0.0)
    return returns


# ══════════════════════════════════════════════════════════════
# 4. RL ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class GrowthAllocEnv:
    """
    RL environment for growth asset allocation.

    State: market condition features (normalized)
    Action: index into ALLOCATIONS table
    Reward: Sharpe-adjusted daily return with drawdown penalty
    """

    def __init__(self, features: np.ndarray, asset_returns: np.ndarray,
                 alloc_array: np.ndarray):
        """
        Args:
            features: (T, n_features) normalized state features
            asset_returns: (T, 5) daily returns for [TQQQ, BTC, SVXY, QQQ, CASH]
            alloc_array: (N_ACTIONS, 5) allocation weights per action
        """
        self.features = features
        self.asset_returns = asset_returns
        self.alloc_array = alloc_array
        self.n_steps = len(features)
        self.n_actions = len(alloc_array)

        # Running stats for Sharpe reward
        self.rolling_returns = []
        self.rolling_window = 20  # 20-day rolling for vol estimate

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cumulative_return = 1.0
        self.peak = 1.0
        self.rolling_returns = []
        return self.features[0]

    def step(self, action: int):
        """Execute one step. Returns (next_state, reward, done, info)."""
        if self.step_idx >= self.n_steps - 1:
            return self.features[-1], 0.0, True, {}

        # Portfolio return = weighted sum of asset returns
        weights = self.alloc_array[action]  # (5,)
        daily_ret = np.dot(weights, self.asset_returns[self.step_idx])

        # Clip extreme returns (data errors)
        daily_ret = np.clip(daily_ret, -0.30, 0.30)

        # Update cumulative return and peak
        self.cumulative_return *= (1 + daily_ret)
        self.peak = max(self.peak, self.cumulative_return)

        # Rolling returns for vol estimate
        self.rolling_returns.append(daily_ret)
        if len(self.rolling_returns) > self.rolling_window:
            self.rolling_returns.pop(0)

        # --- Reward: Sharpe-adjusted return ---
        # Use rolling vol to normalize return (encourages risk-adjusted performance)
        if len(self.rolling_returns) >= 5:
            rolling_vol = np.std(self.rolling_returns) + 1e-8
        else:
            rolling_vol = 0.01  # default vol before enough data

        reward = daily_ret / rolling_vol

        # Drawdown penalty
        current_dd = (self.cumulative_return / self.peak) - 1
        if current_dd < -DRAWDOWN_PENALTY_THRESHOLD:
            excess_dd = abs(current_dd) - DRAWDOWN_PENALTY_THRESHOLD
            reward -= DRAWDOWN_PENALTY_COEFF * excess_dd

        self.step_idx += 1
        done = (self.step_idx >= self.n_steps - 1)
        next_state = self.features[min(self.step_idx, self.n_steps - 1)]

        info = {
            'daily_return': daily_ret,
            'cumulative_return': self.cumulative_return,
            'drawdown': current_dd,
            'action': action,
        }

        return next_state, reward, done, info


# ══════════════════════════════════════════════════════════════
# 5. PPO ACTOR-CRITIC NETWORK
# ══════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """PPO Actor-Critic with shared backbone."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value.squeeze(-1)

    def get_action_and_value(self, state, action=None):
        """
        Given state, return action, log_prob, entropy, and value.
        If action is provided, evaluate that action instead of sampling.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════
# 6. PPO TRAINING
# ══════════════════════════════════════════════════════════════

class PPOTrainer:
    """Self-contained PPO trainer for the growth allocation environment."""

    def __init__(self, n_features: int, n_actions: int, device=DEVICE):
        self.device = device
        self.model = ActorCritic(n_features, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.n_actions = n_actions

    def collect_rollout(self, env):
        """Collect a full episode rollout."""
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        state = env.reset()
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(state_t)

            action_np = action.item()
            next_state, reward, done, info = env.step(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)

            state = next_state

        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'rewards': np.array(rewards, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO update step."""
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)

        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(states)
        total_loss_info = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'n_updates': 0}

        for _ in range(EPOCHS_PER_UPDATE):
            # Shuffle and create minibatches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    mb_states, mb_actions
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + VALUE_COEFF * value_loss + ENTROPY_COEFF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss_info['policy_loss'] += policy_loss.item()
                total_loss_info['value_loss'] += value_loss.item()
                total_loss_info['entropy'] += (-entropy_loss).item()
                total_loss_info['n_updates'] += 1

        # Average losses
        n = max(total_loss_info['n_updates'], 1)
        return {
            'policy_loss': total_loss_info['policy_loss'] / n,
            'value_loss': total_loss_info['value_loss'] / n,
            'entropy': total_loss_info['entropy'] / n,
        }

    def train_on_env(self, env, n_episodes=N_EPISODES, verbose=True):
        """Train the agent for n_episodes on the environment."""
        best_reward = -np.inf
        best_state_dict = None
        episode_rewards = []

        for ep in range(n_episodes):
            rollout = self.collect_rollout(env)
            loss_info = self.update(rollout)

            ep_reward = rollout['rewards'].sum()
            ep_cum_return = np.prod(1 + np.array([
                np.dot(ALLOC_ARRAY[a], env.asset_returns[i])
                for i, a in enumerate(rollout['actions'])
            ])) - 1

            episode_rewards.append(ep_reward)

            if ep_reward > best_reward:
                best_reward = ep_reward
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (ep + 1) % 20 == 0:
                recent_avg = np.mean(episode_rewards[-20:])
                print(f"    Episode {ep+1}/{n_episodes}: "
                      f"reward={ep_reward:.2f}, avg20={recent_avg:.2f}, "
                      f"cum_ret={ep_cum_return:.2%}, "
                      f"entropy={loss_info['entropy']:.3f}")

        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model = self.model.to(self.device)

        return episode_rewards

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Get greedy actions for a batch of states."""
        self.model.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            logits, _ = self.model(states_t)
            actions = logits.argmax(dim=-1).cpu().numpy()
        return actions

    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        """Get action probabilities for a batch of states."""
        self.model.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            logits, _ = self.model(states_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


# ══════════════════════════════════════════════════════════════
# 7. BASELINES
# ══════════════════════════════════════════════════════════════

def compute_baseline_returns(asset_returns: pd.DataFrame, raw: pd.DataFrame,
                             start_idx: int, end_idx: int):
    """
    Compute returns for baseline strategies over a test window.

    Returns dict of strategy_name -> daily return array.
    """
    test_rets = asset_returns.iloc[start_idx:end_idx]
    dates = test_rets.index
    n = len(test_rets)
    baselines = {}

    # --- 1. Equal weight all 5 assets ---
    eq_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    eq_rets = test_rets.values @ eq_weights
    baselines['Equal Weight (5)'] = eq_rets

    # --- 2. TQQQ + 200MA ---
    sma200 = raw['SPY'].rolling(200, min_periods=200).mean()
    tqqq_ma_rets = np.zeros(n)
    for i, date in enumerate(dates):
        if date in sma200.index and not pd.isna(sma200.loc[date]):
            if raw['SPY'].loc[date] > sma200.loc[date]:
                tqqq_ma_rets[i] = test_rets.iloc[i].get('TQQQ', 0.0)
    baselines['TQQQ + 200MA'] = tqqq_ma_rets

    # --- 3. Best static allocation (optimized on train) ---
    # This is computed per-fold in walk_forward, so we use a placeholder here.
    # The actual best-static is computed inline in walk_forward.

    # --- 4. Buy and hold QQQ ---
    qqq_rets = test_rets['QQQ'].values if 'QQQ' in test_rets.columns else np.zeros(n)
    baselines['Buy & Hold QQQ'] = qqq_rets

    return baselines


def find_best_static_allocation(train_returns: np.ndarray):
    """
    Find the best static allocation from ALLOCATIONS based on train period Sharpe.
    Returns (best_action_idx, best_sharpe).
    """
    best_sharpe = -np.inf
    best_action = 0

    for i, weights in enumerate(ALLOCATIONS):
        port_rets = train_returns @ np.array(weights)
        if np.std(port_rets) < 1e-10:
            sharpe = 0
        else:
            sharpe = (np.mean(port_rets) / np.std(port_rets)) * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_action = i

    return best_action, best_sharpe


# ══════════════════════════════════════════════════════════════
# 8. WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════

def run_walk_forward(features: pd.DataFrame, asset_returns: pd.DataFrame,
                     raw: pd.DataFrame):
    """
    Sliding walk-forward: 252d train, 63d test.
    Train PPO agent on each fold, evaluate OOT.
    """
    # Align features and returns
    common_idx = features.index.intersection(asset_returns.index)
    features = features.loc[common_idx].copy()
    asset_returns = asset_returns.loc[common_idx].copy()

    feat_cols = features.columns.tolist()
    n_features = len(feat_cols)
    feat_values = features.values.astype(np.float32)
    feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Asset returns array: columns = [TQQQ, BTC-USD, SVXY, QQQ, CASH]
    ret_cols = ASSETS + [CASH_LABEL]
    ret_values = asset_returns[ret_cols].values.astype(np.float32)
    ret_values = np.nan_to_num(ret_values, nan=0.0, posinf=0.0, neginf=0.0)

    n_total = len(feat_values)
    fold_starts = list(range(TRAIN_DAYS, n_total - TEST_DAYS, TEST_DAYS))

    print(f"[WF] {n_total} aligned samples, {n_features} features, {N_ACTIONS} actions")
    print(f"[WF] {len(fold_starts)} folds, "
          f"dates {common_idx[TRAIN_DAYS]} to {common_idx[-1]}")

    # Storage
    rl_daily_returns = []
    rl_actions = []
    rl_dates = []
    baseline_daily = defaultdict(list)
    baseline_dates = []
    best_static_daily = []
    fold_metrics = []

    for fold_i, test_start in enumerate(fold_starts):
        test_end = min(test_start + TEST_DAYS, n_total)
        train_start = test_start - TRAIN_DAYS

        train_dates = common_idx[train_start:test_start]
        test_dates = common_idx[test_start:test_end]

        # --- Fit scaler on train features ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(feat_values[train_start:test_start])
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = scaler.transform(feat_values[test_start:test_end])
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        R_train = ret_values[train_start:test_start]
        R_test = ret_values[test_start:test_end]

        if fold_i % 5 == 0:
            print(f"\n  Fold {fold_i+1}/{len(fold_starts)}: "
                  f"train {train_dates[0].strftime('%Y-%m-%d')} -> "
                  f"test {test_dates[0].strftime('%Y-%m-%d')} to "
                  f"{test_dates[-1].strftime('%Y-%m-%d')}")

        # --- Train PPO agent ---
        env = GrowthAllocEnv(X_train, R_train, ALLOC_ARRAY)
        trainer = PPOTrainer(n_features, N_ACTIONS, device=DEVICE)
        verbose_fold = (fold_i % 5 == 0)
        trainer.train_on_env(env, n_episodes=N_EPISODES, verbose=verbose_fold)

        # --- Evaluate on test ---
        test_actions = trainer.predict(X_test)
        test_port_returns = np.array([
            np.dot(ALLOC_ARRAY[a], R_test[i]) for i, a in enumerate(test_actions)
        ])

        rl_daily_returns.extend(test_port_returns.tolist())
        rl_actions.extend(test_actions.tolist())
        rl_dates.extend(test_dates.tolist())

        # --- Baselines on test ---
        bl = compute_baseline_returns(asset_returns, raw, test_start, test_end)
        for name, rets in bl.items():
            baseline_daily[name].extend(rets.tolist())
        baseline_dates.extend(test_dates.tolist())

        # --- Best static allocation (trained on in-sample) ---
        best_static_idx, best_static_sharpe = find_best_static_allocation(R_train)
        static_weights = ALLOC_ARRAY[best_static_idx]
        static_test_rets = R_test @ static_weights
        best_static_daily.extend(static_test_rets.tolist())

        # Per-fold summary
        if len(test_port_returns) > 0:
            fold_cum = np.prod(1 + test_port_returns) - 1
            fold_std = np.std(test_port_returns) + 1e-10
            fold_sharpe = (np.mean(test_port_returns) / fold_std) * np.sqrt(252)
            fold_metrics.append({
                'fold': fold_i + 1,
                'test_start': str(test_dates[0].date()),
                'test_end': str(test_dates[-1].date()),
                'cum_return': round(fold_cum * 100, 2),
                'sharpe': round(fold_sharpe, 3),
                'best_static_alloc': ALLOCATIONS[best_static_idx],
                'best_static_sharpe_train': round(best_static_sharpe, 3),
            })

        # Cleanup
        del trainer, env
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'rl_returns': rl_daily_returns,
        'rl_actions': rl_actions,
        'rl_dates': rl_dates,
        'baselines': dict(baseline_daily),
        'best_static_returns': best_static_daily,
        'baseline_dates': baseline_dates,
        'fold_metrics': fold_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 9. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════

def permutation_test(rl_returns, rl_actions, asset_returns_aligned,
                     n_permutations=1000, seed=42):
    """
    Shuffle state-action mapping to verify RL agent beats random allocation.
    Returns p-value (fraction of random permutations with higher Sharpe).
    """
    rng = np.random.RandomState(seed)

    # RL agent Sharpe
    rl_rets = np.array(rl_returns)
    rl_sharpe = (np.mean(rl_rets) / (np.std(rl_rets) + 1e-10)) * np.sqrt(252)

    # Permutation test: shuffle actions, compute portfolio returns
    actions = np.array(rl_actions)
    n = len(actions)
    better_count = 0

    for _ in range(n_permutations):
        perm_actions = rng.permutation(actions)
        perm_rets = np.array([
            np.dot(ALLOC_ARRAY[a], asset_returns_aligned[i])
            for i, a in enumerate(perm_actions)
        ])
        perm_sharpe = (np.mean(perm_rets) / (np.std(perm_rets) + 1e-10)) * np.sqrt(252)
        if perm_sharpe >= rl_sharpe:
            better_count += 1

    p_value = better_count / n_permutations

    print(f"\n[PERMUTATION TEST] RL Sharpe: {rl_sharpe:.3f}")
    print(f"  {n_permutations} permutations, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  SIGNIFICANT at 5% level — RL agent is NOT random")
    else:
        print(f"  NOT significant — RL agent may be equivalent to random allocation")

    return {
        'rl_sharpe': round(rl_sharpe, 4),
        'p_value': round(p_value, 4),
        'n_permutations': n_permutations,
        'significant_5pct': p_value < 0.05,
    }


# ══════════════════════════════════════════════════════════════
# 10. METRICS + REPORTING
# ══════════════════════════════════════════════════════════════

def compute_strategy_metrics(daily_returns, name='Strategy'):
    """Compute CAGR, Sharpe, Sortino, MaxDD, WR, PF from daily returns."""
    rets = np.array(daily_returns)
    rets = rets[~np.isnan(rets)]

    if len(rets) < 21:
        return {'name': name, 'error': 'insufficient data'}

    mean_daily = np.mean(rets)
    std_daily = np.std(rets, ddof=1) + 1e-10
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10

    sharpe = (mean_daily / std_daily) * np.sqrt(252)
    sortino = (mean_daily / (downside_std + 1e-10)) * np.sqrt(252)

    cum = np.cumprod(1 + rets)
    total_return = cum[-1] - 1
    n_years = len(rets) / 252
    cagr = (cum[-1] ** (1 / max(n_years, 0.01))) - 1 if cum[-1] > 0 else -1.0

    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = np.min(drawdowns)

    wr = np.mean(rets > 0) if np.any(rets != 0) else 0.0
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = -np.sum(rets[rets < 0])
    pf = gross_profit / (gross_loss + 1e-10)

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'n_days': len(rets),
        'n_years': round(n_years, 2),
    }


def evaluate_and_report(wf_results, asset_returns: pd.DataFrame):
    """Generate comprehensive evaluation report and save results."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS -- RL Growth Allocator (PPO)")
    print("=" * 80)

    all_metrics = {}

    # RL Agent
    rl_m = compute_strategy_metrics(wf_results['rl_returns'], 'RL Agent (PPO)')
    all_metrics['rl_agent'] = rl_m

    # Baselines
    for bl_name, bl_rets in wf_results['baselines'].items():
        m = compute_strategy_metrics(bl_rets, bl_name)
        all_metrics[bl_name] = m

    # Best static (walk-forward oracle)
    bst_m = compute_strategy_metrics(wf_results['best_static_returns'], 'Best Static (WF)')
    all_metrics['best_static'] = bst_m

    # Print comparison table
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>8} {'WR':>6} {'PF':>6}")
    print("-" * 75)
    for key, m in all_metrics.items():
        if 'error' not in m:
            print(f"{m['name']:<25} {m['cagr']:>7}% {m['sharpe']:>8} {m['sortino']:>9} "
                  f"{m['max_dd']:>7}% {m['win_rate']:>5}% {m['profit_factor']:>6}")

    # RL action distribution
    action_counts = pd.Series(wf_results['rl_actions']).value_counts()
    print(f"\n{'=' * 60}")
    print("RL AGENT ACTION DISTRIBUTION")
    print(f"{'=' * 60}")
    for action_idx in action_counts.index[:10]:
        pct = action_counts[action_idx] / len(wf_results['rl_actions']) * 100
        alloc_str = ", ".join([
            f"{ALL_ASSETS[j]}={ALLOCATIONS[action_idx][j]:.0%}"
            for j in range(5) if ALLOCATIONS[action_idx][j] > 0
        ])
        print(f"  Action {action_idx:>2} ({pct:>5.1f}%): {alloc_str}")

    # Beat baselines?
    print(f"\n{'=' * 60}")
    rl_sharpe = rl_m.get('sharpe', 0)
    print(f"RL Agent Sharpe: {rl_sharpe}")
    for bl_name in list(wf_results['baselines'].keys()) + ['best_static']:
        bl_m = all_metrics.get(bl_name, {})
        bl_sharpe = bl_m.get('sharpe', 0)
        if rl_sharpe > bl_sharpe:
            print(f"  BEATS {bl_m.get('name', bl_name)}: {rl_sharpe:.3f} vs {bl_sharpe:.3f}")
        else:
            print(f"  LOSES to {bl_m.get('name', bl_name)}: {rl_sharpe:.3f} vs {bl_sharpe:.3f}")

    # Permutation test
    ret_cols = ASSETS + [CASH_LABEL]
    common_idx = [d for d in wf_results['rl_dates'] if d in asset_returns.index]
    if len(common_idx) == len(wf_results['rl_dates']):
        aligned_rets = asset_returns.loc[common_idx, ret_cols].values
    else:
        # Fallback: reconstruct from returns
        aligned_rets = np.zeros((len(wf_results['rl_actions']), 5))
        for i, d in enumerate(wf_results['rl_dates']):
            if d in asset_returns.index:
                aligned_rets[i] = asset_returns.loc[d, ret_cols].values

    perm_result = permutation_test(
        wf_results['rl_returns'],
        wf_results['rl_actions'],
        aligned_rets,
        n_permutations=1000,
    )
    all_metrics['permutation_test'] = perm_result

    return all_metrics


# ══════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 80)
    print("RL GROWTH ALLOCATOR -- PPO Dynamic Asset Allocation")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Start time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {ALL_ASSETS}")
    print(f"Actions: {N_ACTIONS} preset allocations")
    print(f"PPO: clip={PPO_CLIP}, entropy={ENTROPY_COEFF}, episodes={N_EPISODES}")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test (sliding)")
    print("=" * 80)

    # Step 1: Download data
    print("\n[1/6] Downloading data...")
    raw = download_all_data(start='2015-01-01')
    print(f"  Raw data: {len(raw)} rows, {len(raw.columns)} columns")
    print(f"  Date range: {raw.index[0]} to {raw.index[-1]}")

    available_assets = [a for a in ASSETS if a in raw.columns]
    print(f"  Available target assets: {available_assets}")
    if len(available_assets) < 3:
        print("ERROR: Need at least 3 target assets. Exiting.")
        sys.exit(1)

    # Step 2: Feature engineering
    print("\n[2/6] Computing features...")
    features = compute_features(raw)
    features = features.dropna()
    print(f"  Features: {len(features)} rows after dropna, {len(features.columns)} columns")
    print(f"  Feature list: {features.columns.tolist()}")

    # Step 3: Asset returns
    print("\n[3/6] Computing asset returns...")
    asset_returns = compute_asset_returns(raw)
    print(f"  Asset returns: {len(asset_returns)} rows")

    # Step 4: Walk-forward training
    print("\n[4/6] Running walk-forward PPO training...")
    wf_results = run_walk_forward(features, asset_returns, raw)

    # Step 5: Evaluate
    print("\n[5/6] Evaluating results...")
    all_metrics = evaluate_and_report(wf_results, asset_returns)

    # Step 6: Save results
    print("\n[6/6] Saving results...")

    # Metrics JSON
    results_file = OUTPUT_DIR / 'results.json'
    serializable = {}
    for k, v in all_metrics.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Daily returns CSV
    rl_ret_df = pd.DataFrame({
        'date': wf_results['rl_dates'],
        'daily_return': wf_results['rl_returns'],
        'action': wf_results['rl_actions'],
    })
    rl_ret_df.to_csv(OUTPUT_DIR / 'rl_agent_daily_returns.csv', index=False)

    # Baseline returns
    for bl_name, bl_rets in wf_results['baselines'].items():
        safe_name = bl_name.lower().replace(' ', '_').replace('&', 'and').replace('+', '_')
        bl_df = pd.DataFrame({
            'date': wf_results['baseline_dates'][:len(bl_rets)],
            'daily_return': bl_rets,
        })
        bl_df.to_csv(OUTPUT_DIR / f'baseline_{safe_name}.csv', index=False)

    # Best static returns
    bst_df = pd.DataFrame({
        'date': wf_results['baseline_dates'][:len(wf_results['best_static_returns'])],
        'daily_return': wf_results['best_static_returns'],
    })
    bst_df.to_csv(OUTPUT_DIR / 'baseline_best_static_wf.csv', index=False)

    # Fold metrics
    fold_df = pd.DataFrame(wf_results['fold_metrics'])
    fold_df.to_csv(OUTPUT_DIR / 'fold_metrics.csv', index=False)

    # Action distribution
    action_dist = pd.Series(wf_results['rl_actions']).value_counts(normalize=True)
    action_detail = []
    for idx in action_dist.index:
        alloc_str = ", ".join([
            f"{ALL_ASSETS[j]}={ALLOCATIONS[idx][j]:.0%}"
            for j in range(5) if ALLOCATIONS[idx][j] > 0
        ])
        action_detail.append({
            'action_idx': idx,
            'frequency': round(action_dist[idx] * 100, 2),
            'allocation': alloc_str,
        })
    action_df = pd.DataFrame(action_detail)
    action_df.to_csv(OUTPUT_DIR / 'action_distribution.csv', index=False)

    elapsed = time.time() - t0
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
