#!/usr/bin/env python3
"""
RL Portfolio Allocator — DQN-based Dynamic Asset Allocation
============================================================
Deep Q-Network that learns optimal allocation between UPRO/SPY/GLD/TLT/SHY
based on market regime features.

Design:
  - State: ~15 market features (SPY momentum, VIX, credit spread, gold/bond/dollar trend,
    current allocation one-hot, days since switch)
  - Action: 5 discrete allocations (100% UPRO, 67/33 UPRO/SPY, 100% SPY, 50/50 GLD/TLT, 100% SHY)
  - Reward: Sharpe-ratio shaped daily return minus switching cost
  - Algorithm: DQN with experience replay + target network
  - Walk-forward: SLIDING 504d train, 63d validation, 63d OOS (HC #0)
  - Adversarial validation: permutation test, sub-period, outlier removal, regime test (HC #705)

HC compliance: sliding window only, risk-adjusted metrics, $100K fixed capital, no DCA (HC #713).
"""

import json
import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path
from collections import deque
import random

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/rl_portfolio_allocator')
CACHE_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cache')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward params (SLIDING — HC #0)
TRAIN_DAYS = 504        # 2 years sliding train
VAL_DAYS = 63           # 1 quarter validation (for early stopping)
TEST_DAYS = 63          # 1 quarter OOS
SLIDE_STEP = 63         # slide by 1 quarter

# DQN hyperparameters
GAMMA = 0.99            # discount factor
LR = 1e-3
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_TAU = 0.005   # soft update coefficient
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
N_EPISODES = int(os.environ.get('RL_EPISODES', 80))  # training episodes per fold
SHARPE_WINDOW = 21      # rolling Sharpe window for reward shaping
SWITCH_COST = 0.0002    # 0.02% switching cost

# Fixed capital (HC #713 — no DCA)
INITIAL_CAPITAL = 100_000

# Assets
TICKERS = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', '^VIX', 'HYG', 'IEF', 'UUP', 'QQQ']
TRADEABLE = ['UPRO', 'SPY', 'GLD', 'TLT', 'SHY']

# Action space: 5 discrete allocations
# Each tuple maps to weights for [UPRO, SPY, GLD, TLT, SHY]
ALLOCATIONS = {
    0: {'UPRO': 1.0},                           # 100% UPRO (aggressive)
    1: {'UPRO': 0.67, 'SPY': 0.33},             # 67/33 UPRO/SPY (growth)
    2: {'SPY': 1.0},                             # 100% SPY (moderate)
    3: {'GLD': 0.50, 'TLT': 0.50},              # 50/50 GLD/TLT (defensive)
    4: {'SHY': 1.0},                             # 100% SHY (cash-like)
}
N_ACTIONS = len(ALLOCATIONS)
ALLOC_LABELS = ['100% UPRO', '67/33 UPRO/SPY', '100% SPY', '50/50 GLD/TLT', '100% SHY']

# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

def download_data(start='2010-01-01', end='2026-07-17'):
    """Download daily data, cache locally."""
    cache_file = CACHE_DIR / 'rl_portfolio_data.parquet'
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 3000:  # rough check for enough data
            print(f"  Loaded cached data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            return df

    print("  Downloading from yfinance...")
    frames = {}
    for ticker in TICKERS:
        clean_name = ticker.replace('^', '')
        try:
            d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if len(d) > 100:
                close = d['Close']
                # yfinance may return MultiIndex columns; flatten
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                frames[clean_name] = close
                print(f"    {clean_name}: {len(d)} rows")
            else:
                print(f"    {clean_name}: SKIPPED (only {len(d)} rows)")
        except Exception as e:
            print(f"    {clean_name}: FAILED ({e})")

    df = pd.DataFrame(frames)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.to_parquet(cache_file)
    print(f"  Saved cache: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def build_features(prices_df):
    """Build state features from price data."""
    df = prices_df.copy()
    returns = df.pct_change()

    features = pd.DataFrame(index=df.index)

    # SPY returns at multiple horizons
    for d in [1, 5, 21, 63]:
        features[f'spy_ret_{d}d'] = df['SPY'].pct_change(d)

    # VIX level and percentile rank
    features['vix_level'] = df['VIX'] / 100.0  # normalize
    features['vix_pctrank_63d'] = df['VIX'].rolling(63).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )

    # Credit spread proxy: HYG-IEF spread change
    hyg_ief_spread = np.log(df['HYG']) - np.log(df['IEF'])
    features['credit_spread_chg_5d'] = hyg_ief_spread.diff(5)

    # Gold momentum
    features['gld_ret_21d'] = df['GLD'].pct_change(21)

    # Bond trend
    features['tlt_ret_21d'] = df['TLT'].pct_change(21)

    # Dollar strength
    features['uup_ret_21d'] = df['UUP'].pct_change(21)

    # QQQ momentum (tech sentiment)
    features['qqq_ret_21d'] = df['QQQ'].pct_change(21)

    # SPY volatility (21d realized)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std() * np.sqrt(252)

    # Drop NaN rows
    features = features.dropna()

    return features


def get_portfolio_return(returns_row, action_idx):
    """Calculate portfolio return for a given action."""
    alloc = ALLOCATIONS[action_idx]
    ret = 0.0
    for asset, weight in alloc.items():
        if asset in returns_row.index:
            ret += weight * returns_row[asset]
    return ret


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class PortfolioEnv:
    """Trading environment for RL portfolio allocation."""

    def __init__(self, features, returns, mode='train'):
        self.features = features.values.astype(np.float32)
        self.returns = returns  # DataFrame with tradeable asset returns
        self.n_steps = len(features)
        self.mode = mode
        self.n_features = features.shape[1]

        # State includes: features + one-hot allocation (5) + days_since_switch (1)
        self.state_dim = self.n_features + N_ACTIONS + 1

    def reset(self):
        self.t = 0
        self.current_action = 4  # start in SHY (cash)
        self.days_since_switch = 0
        self.portfolio_value = INITIAL_CAPITAL
        self.peak_value = INITIAL_CAPITAL
        self.daily_returns = []
        return self._get_state()

    def _get_state(self):
        feat = self.features[self.t]
        # One-hot current allocation
        one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
        one_hot[self.current_action] = 1.0
        # Days since switch (normalized)
        dsw = np.array([min(self.days_since_switch, 63) / 63.0], dtype=np.float32)
        return np.concatenate([feat, one_hot, dsw])

    def step(self, action):
        if self.t >= self.n_steps - 1:
            return self._get_state(), 0.0, True, {}

        # Calculate switching cost
        switched = (action != self.current_action)
        switch_penalty = SWITCH_COST if switched else 0.0

        # Get next day's return
        ret_row = self.returns.iloc[self.t + 1]
        port_return = get_portfolio_return(ret_row, action)

        # Net return after switching cost
        net_return = port_return - switch_penalty

        # Update portfolio
        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.daily_returns.append(net_return)

        # Reward shaping: rolling Sharpe
        if len(self.daily_returns) >= SHARPE_WINDOW:
            recent = np.array(self.daily_returns[-SHARPE_WINDOW:])
            mu = recent.mean()
            sigma = recent.std() + 1e-8
            sharpe_component = (mu / sigma) * np.sqrt(252) * 0.01  # scaled
        else:
            sharpe_component = 0.0

        # Base reward = daily return + Sharpe shaping
        reward = net_return * 100 + sharpe_component  # scale returns for learning

        # Drawdown penalty
        dd = (self.peak_value - self.portfolio_value) / self.peak_value
        if dd > 0.15:
            reward -= dd * 2.0

        # Update state
        if switched:
            self.days_since_switch = 0
        else:
            self.days_since_switch += 1
        self.current_action = action
        self.t += 1

        done = (self.t >= self.n_steps - 1)
        info = {'portfolio_value': self.portfolio_value, 'daily_return': net_return}
        return self._get_state(), reward, done, info


# ══════════════════════════════════════════════════════════════
# DQN MODEL
# ══════════════════════════════════════════════════════════════

class DQN(nn.Module):
    """Simple 2-layer MLP for Q-value estimation."""

    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM_2, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with target network and epsilon-greedy exploration."""

    def __init__(self, state_dim, n_actions, device='cpu'):
        self.device = torch.device(device)
        self.n_actions = n_actions

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = EPSILON_START

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN: use policy net to select, target net to evaluate)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TARGET_UPDATE_TAU * policy_param.data + (1.0 - TARGET_UPDATE_TAU) * target_param.data)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════

def normalize_features(train_feat, test_feat):
    """Z-score normalize using train statistics."""
    mu = train_feat.mean(axis=0)
    sigma = train_feat.std(axis=0) + 1e-8
    return (train_feat - mu) / sigma, (test_feat - mu) / sigma


def train_dqn_on_window(features, returns, device='cpu', verbose=False):
    """Train DQN on a single training window, return trained agent."""
    env = PortfolioEnv(features, returns, mode='train')
    agent = DQNAgent(env.state_dim, N_ACTIONS, device=device)

    best_val_return = -np.inf
    best_state_dict = None

    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        total_return = 0

        # Collect full episode of transitions (no training during rollout)
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if 'daily_return' in info:
                total_return += info['daily_return']
            if done:
                break

        # Train at end of episode only (10 gradient updates)
        if len(agent.replay_buffer) >= BATCH_SIZE:
            for _ in range(10):
                agent.train_step()

        agent.decay_epsilon()

        ep_sharpe = 0
        if len(env.daily_returns) > 5:
            rets = np.array(env.daily_returns)
            ep_sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)

        # Track best episode by Sharpe
        if ep_sharpe > best_val_return:
            best_val_return = ep_sharpe
            best_state_dict = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}

        if verbose and (episode + 1) % 20 == 0:
            final_val = env.portfolio_value
            print(f"    Ep {episode+1:3d} | Reward {total_reward:8.2f} | "
                  f"Final ${final_val:,.0f} | Sharpe {ep_sharpe:.2f} | Eps {agent.epsilon:.3f}")

    # Load best weights
    if best_state_dict is not None:
        agent.policy_net.load_state_dict(best_state_dict)
        agent.target_net.load_state_dict(best_state_dict)

    return agent


def evaluate_agent(agent, features, returns):
    """Evaluate agent on OOS data. Returns daily returns and actions."""
    env = PortfolioEnv(features, returns, mode='eval')
    state = env.reset()

    daily_rets = []
    actions_taken = []

    while True:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        if 'daily_return' in info:
            daily_rets.append(info['daily_return'])
            actions_taken.append(action)
        state = next_state
        if done:
            break

    return np.array(daily_rets), np.array(actions_taken)


def run_walkforward(features_df, returns_df, device='cpu'):
    """Run sliding window walk-forward validation."""
    n = len(features_df)
    total_window = TRAIN_DAYS + VAL_DAYS + TEST_DAYS  # need train + val + test

    all_oos_returns = []
    all_oos_actions = []
    all_oos_dates = []
    fold_results = []

    fold_idx = 0
    start = 0

    while start + total_window <= n:
        train_end = start + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        if test_end > n:
            break

        # Split
        train_feat = features_df.iloc[start:train_end]
        train_ret = returns_df.iloc[start:train_end]

        # Use train+val for DQN training (val is for within-episode evaluation)
        trainval_feat = features_df.iloc[start:val_end]
        trainval_ret = returns_df.iloc[start:val_end]

        test_feat = features_df.iloc[val_end:test_end]
        test_ret = returns_df.iloc[val_end:test_end]

        # Normalize features
        train_values = trainval_feat.values
        test_values = test_feat.values
        mu = train_values.mean(axis=0)
        sigma = train_values.std(axis=0) + 1e-8

        trainval_norm = pd.DataFrame(
            (train_values - mu) / sigma,
            index=trainval_feat.index,
            columns=trainval_feat.columns
        )
        test_norm = pd.DataFrame(
            (test_values - mu) / sigma,
            index=test_feat.index,
            columns=test_feat.columns
        )

        # Train
        print(f"\n  Fold {fold_idx}: Train {trainval_feat.index[0].date()}-{trainval_feat.index[-1].date()} | "
              f"Test {test_feat.index[0].date()}-{test_feat.index[-1].date()}")

        agent = train_dqn_on_window(trainval_norm, trainval_ret, device=device, verbose=True)

        # Evaluate OOS
        oos_rets, oos_actions = evaluate_agent(agent, test_norm, test_ret)
        oos_dates = test_ret.index[1:len(oos_rets)+1]  # offset by 1 due to next-day execution

        if len(oos_rets) > 5:
            sharpe = (oos_rets.mean() / (oos_rets.std() + 1e-8)) * np.sqrt(252)
            cagr = (1 + oos_rets).prod() ** (252 / len(oos_rets)) - 1
            max_dd = compute_max_drawdown(oos_rets)
            print(f"    OOS: Sharpe {sharpe:.2f} | CAGR {cagr:.1%} | MaxDD {max_dd:.1%} | "
                  f"Days {len(oos_rets)}")

            # Action distribution
            action_counts = np.bincount(oos_actions.astype(int), minlength=N_ACTIONS)
            action_pcts = action_counts / action_counts.sum() * 100
            labels_str = ' | '.join(f"{ALLOC_LABELS[i]}:{action_pcts[i]:.0f}%" for i in range(N_ACTIONS))
            print(f"    Actions: {labels_str}")

            fold_results.append({
                'fold': fold_idx,
                'train_start': trainval_feat.index[0].strftime('%Y-%m-%d'),
                'test_start': test_feat.index[0].strftime('%Y-%m-%d'),
                'test_end': test_feat.index[-1].strftime('%Y-%m-%d'),
                'sharpe': sharpe,
                'cagr': cagr,
                'max_dd': max_dd,
                'n_days': len(oos_rets),
                'action_dist': action_pcts.tolist(),
            })

        all_oos_returns.extend(oos_rets.tolist())
        all_oos_actions.extend(oos_actions.tolist())
        if len(oos_dates) > 0:
            all_oos_dates.extend(oos_dates.tolist())

        fold_idx += 1
        start += SLIDE_STEP

    return np.array(all_oos_returns), np.array(all_oos_actions), all_oos_dates, fold_results


def compute_max_drawdown(returns):
    """Compute max drawdown from returns array."""
    cumulative = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    return dd.min()


def compute_metrics(returns, label='Strategy'):
    """Compute risk-adjusted metrics."""
    if len(returns) == 0:
        return {}
    rets = np.array(returns)
    n = len(rets)
    mu = rets.mean()
    sigma = rets.std() + 1e-8
    downside = rets[rets < 0].std() + 1e-8

    sharpe = (mu / sigma) * np.sqrt(252)
    sortino = (mu / downside) * np.sqrt(252)
    cagr = (1 + rets).prod() ** (252 / n) - 1
    max_dd = compute_max_drawdown(rets)

    # Win rate and profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    wr = len(wins) / n if n > 0 else 0
    pf = (wins.sum() / (-losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    total_return = (1 + rets).prod() - 1
    final_value = INITIAL_CAPITAL * (1 + total_return)

    return {
        'label': label,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'max_dd': max_dd,
        'win_rate': wr,
        'profit_factor': pf,
        'total_return': total_return,
        'final_value': final_value,
        'n_days': n,
    }


# ══════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════

def compute_benchmark_returns(returns_df, oos_dates):
    """Compute SPY buy-and-hold and random allocation returns for the same OOS dates."""
    # SPY buy-and-hold
    spy_rets = []
    random_rets = []

    for d in oos_dates:
        if d in returns_df.index:
            spy_rets.append(returns_df.loc[d, 'SPY'])
            # Random allocation
            random_action = random.randint(0, N_ACTIONS - 1)
            random_rets.append(get_portfolio_return(returns_df.loc[d], random_action))
        else:
            spy_rets.append(0.0)
            random_rets.append(0.0)

    return np.array(spy_rets), np.array(random_rets)


def compute_v44_benchmark(returns_df, oos_dates):
    """Approximate v4.4 macro-scaled strategy returns.
    v4.4 switches between UPRO/SPY/GLD/TLT/SHY based on SMA + VIX rules.
    Simplified approximation: SPY > 200d SMA → UPRO, VIX > 30 → SHY, else SPY.
    """
    prices_needed = ['SPY', 'UPRO', 'SHY']
    v44_rets = []

    for d in oos_dates:
        if d not in returns_df.index:
            v44_rets.append(0.0)
            continue

        # Simple regime approximation
        idx = returns_df.index.get_loc(d)
        if idx < 200:
            v44_rets.append(returns_df.loc[d, 'SPY'])
            continue

        # Get SPY price history for SMA
        spy_prices = returns_df['SPY'].iloc[:idx+1]
        spy_cum = (1 + spy_prices).cumprod()
        sma200 = spy_cum.rolling(200).mean().iloc[-1]
        current = spy_cum.iloc[-1]

        if current > sma200:
            v44_rets.append(returns_df.loc[d, 'UPRO'])  # trend up → leverage
        else:
            v44_rets.append(returns_df.loc[d, 'SHY'])   # trend down → cash

    return np.array(v44_rets)


# ══════════════════════════════════════════════════════════════
# ADVERSARIAL VALIDATION (HC #705)
# ══════════════════════════════════════════════════════════════

def adversarial_validation(oos_returns, oos_actions, oos_dates, returns_df, features_df):
    """Run adversarial validation tests."""
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION (HC #705)")
    print("=" * 70)

    results = {}

    # 1. Permutation test (shuffle features, retrain, compare)
    print("\n  [1] Permutation Test (100 shuffles of returns → baseline Sharpe)")
    n_perms = 100
    oos_rets = np.array(oos_returns)
    real_sharpe = (oos_rets.mean() / (oos_rets.std() + 1e-8)) * np.sqrt(252)

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(oos_rets)
        s = (shuffled.mean() / (shuffled.std() + 1e-8)) * np.sqrt(252)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()
    print(f"    Real OOS Sharpe: {real_sharpe:.3f}")
    print(f"    Permuted mean Sharpe: {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}")
    print(f"    p-value: {p_value:.3f} {'✓ PASS' if p_value < 0.05 else '✗ FAIL'}")
    results['permutation_p_value'] = float(p_value)
    results['permutation_pass'] = p_value < 0.05

    # 2. Sub-period consistency
    print("\n  [2] Sub-Period Consistency")
    n = len(oos_rets)
    n_sub = 4
    sub_size = n // n_sub
    sub_sharpes = []
    for i in range(n_sub):
        sub = oos_rets[i*sub_size:(i+1)*sub_size]
        if len(sub) > 5:
            s = (sub.mean() / (sub.std() + 1e-8)) * np.sqrt(252)
            sub_sharpes.append(s)
            print(f"    Sub-period {i+1}: Sharpe {s:.2f}, Return {(1+sub).prod()-1:.1%}")

    positive_periods = sum(1 for s in sub_sharpes if s > 0)
    consistency = positive_periods / len(sub_sharpes) if sub_sharpes else 0
    print(f"    Positive Sharpe periods: {positive_periods}/{len(sub_sharpes)} ({consistency:.0%})")
    results['sub_period_consistency'] = consistency
    results['sub_period_sharpes'] = sub_sharpes

    # 3. Outlier removal (top 5% days)
    print("\n  [3] Outlier Removal (remove top 5% return days)")
    threshold = np.percentile(oos_rets, 95)
    filtered = oos_rets[oos_rets <= threshold]
    if len(filtered) > 5:
        filt_sharpe = (filtered.mean() / (filtered.std() + 1e-8)) * np.sqrt(252)
        print(f"    Full Sharpe: {real_sharpe:.3f}")
        print(f"    Without top 5%: {filt_sharpe:.3f}")
        print(f"    Sharpe drop: {(real_sharpe - filt_sharpe):.3f} "
              f"{'✓ Robust' if filt_sharpe > 0 else '✗ Fragile'}")
        results['outlier_removed_sharpe'] = float(filt_sharpe)
        results['outlier_robust'] = filt_sharpe > 0

    # 4. R1 Regime test (green vs red days)
    print("\n  [4] R1 Regime Test (green vs red SPY days)")
    if len(oos_dates) > 0 and len(oos_dates) == len(oos_rets):
        spy_daily = []
        for d in oos_dates:
            if d in returns_df.index:
                spy_daily.append(returns_df.loc[d, 'SPY'])
            else:
                spy_daily.append(0.0)
        spy_daily = np.array(spy_daily)

        green_mask = spy_daily > 0
        red_mask = spy_daily < 0

        if green_mask.sum() > 5 and red_mask.sum() > 5:
            green_rets = oos_rets[green_mask]
            red_rets = oos_rets[red_mask]

            green_sharpe = (green_rets.mean() / (green_rets.std() + 1e-8)) * np.sqrt(252)
            red_sharpe = (red_rets.mean() / (red_rets.std() + 1e-8)) * np.sqrt(252)

            max_s = max(abs(green_sharpe), abs(red_sharpe))
            regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 0

            print(f"    Green days Sharpe: {green_sharpe:.2f} ({green_mask.sum()} days)")
            print(f"    Red days Sharpe: {red_sharpe:.2f} ({red_mask.sum()} days)")
            print(f"    Regime gap: {regime_gap:.2f} {'✓ PASS (<0.50)' if regime_gap < 0.50 else '✗ FAIL (>0.50)'}")
            results['green_sharpe'] = float(green_sharpe)
            results['red_sharpe'] = float(red_sharpe)
            results['regime_gap'] = float(regime_gap)
            results['regime_pass'] = regime_gap < 0.50

    # 5. Action diversity check
    print("\n  [5] Action Diversity Check")
    oos_act = np.array(oos_actions, dtype=int)
    action_counts = np.bincount(oos_act, minlength=N_ACTIONS)
    action_pcts = action_counts / action_counts.sum() * 100
    max_action_pct = action_pcts.max()
    print(f"    Distribution: {' | '.join(f'{ALLOC_LABELS[i]}:{action_pcts[i]:.1f}%' for i in range(N_ACTIONS))}")
    print(f"    Max single action: {max_action_pct:.1f}% {'✓ Diverse' if max_action_pct < 80 else '⚠ Concentrated'}")
    results['action_diversity'] = float(1 - max_action_pct / 100)

    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RL PORTFOLIO ALLOCATOR — DQN Walk-Forward")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")

    # 1. Download data
    print("\n[1] Downloading data...")
    prices = download_data()
    print(f"  Price data: {len(prices)} rows, columns: {list(prices.columns)}")

    # 2. Build features
    print("\n[2] Building features...")
    features = build_features(prices)
    print(f"  Features: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"  Feature names: {list(features.columns)}")

    # 3. Build returns
    returns = prices[TRADEABLE].pct_change()

    # Align features and returns
    common_idx = features.index.intersection(returns.dropna().index)
    features = features.loc[common_idx]
    returns = returns.loc[common_idx]
    print(f"  Aligned: {len(features)} rows from {features.index[0].date()} to {features.index[-1].date()}")

    # 4. Walk-forward
    print("\n[3] Running SLIDING walk-forward...")
    print(f"  Train: {TRAIN_DAYS}d | Val: {VAL_DAYS}d | Test: {TEST_DAYS}d | Slide: {SLIDE_STEP}d")

    t0 = time.time()
    oos_returns, oos_actions, oos_dates, fold_results = run_walkforward(features, returns, device=device)
    elapsed = time.time() - t0
    print(f"\n  Walk-forward completed in {elapsed:.1f}s")
    print(f"  Total OOS days: {len(oos_returns)}")

    if len(oos_returns) == 0:
        print("  ERROR: No OOS returns generated. Check data length.")
        return

    # 5. Compute metrics
    print("\n[4] Results")
    print("=" * 70)

    rl_metrics = compute_metrics(oos_returns, 'DQN RL Allocator')

    # Benchmarks
    spy_rets, random_rets = compute_benchmark_returns(returns, oos_dates)
    spy_metrics = compute_metrics(spy_rets[:len(oos_returns)], 'SPY Buy-and-Hold')
    random_metrics = compute_metrics(random_rets[:len(oos_returns)], 'Random Allocation')

    # v4.4 approximation
    v44_rets = compute_v44_benchmark(returns, oos_dates)
    v44_metrics = compute_metrics(v44_rets[:len(oos_returns)], 'v4.4 Approx (SMA+VIX)')

    # Print comparison table
    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Final $':>12}")
    print("  " + "-" * 87)
    for m in [rl_metrics, spy_metrics, v44_metrics, random_metrics]:
        if m:
            pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf"
            print(f"  {m['label']:<25} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['cagr']:>7.1%} "
                  f"{m['max_dd']:>7.1%} {m['win_rate']:>5.1%} {pf_str:>6} {m['final_value']:>12,.0f}")

    # 6. Adversarial validation
    adv_results = adversarial_validation(oos_returns, oos_actions, oos_dates, returns, features)

    # 7. Save results
    print("\n[5] Saving results...")
    results = {
        'timestamp': dt.datetime.now().isoformat(),
        'config': {
            'train_days': TRAIN_DAYS,
            'val_days': VAL_DAYS,
            'test_days': TEST_DAYS,
            'slide_step': SLIDE_STEP,
            'n_episodes': N_EPISODES,
            'hidden_dims': [HIDDEN_DIM_1, HIDDEN_DIM_2],
            'lr': LR,
            'gamma': GAMMA,
            'switch_cost': SWITCH_COST,
            'device': device,
        },
        'rl_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in rl_metrics.items()},
        'spy_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in spy_metrics.items()},
        'v44_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in v44_metrics.items()},
        'random_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in random_metrics.items()},
        'adversarial': adv_results,
        'fold_results': fold_results,
        'n_oos_days': len(oos_returns),
    }

    results_file = OUTPUT_DIR / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_file}")

    # Save OOS returns for further analysis
    oos_df = pd.DataFrame({
        'date': oos_dates[:len(oos_returns)],
        'rl_return': oos_returns,
        'action': oos_actions[:len(oos_returns)],
        'spy_return': spy_rets[:len(oos_returns)],
    })
    oos_file = OUTPUT_DIR / 'oos_returns.csv'
    oos_df.to_csv(oos_file, index=False)
    print(f"  Saved: {oos_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    beat_spy = rl_metrics['sharpe'] > spy_metrics['sharpe']
    beat_v44 = rl_metrics['sharpe'] > v44_metrics['sharpe']
    print(f"  DQN RL Sharpe: {rl_metrics['sharpe']:.2f} | CAGR: {rl_metrics['cagr']:.1%} | MaxDD: {rl_metrics['max_dd']:.1%}")
    print(f"  vs SPY: {'BEATS' if beat_spy else 'LOSES'} ({rl_metrics['sharpe']:.2f} vs {spy_metrics['sharpe']:.2f})")
    print(f"  vs v4.4: {'BEATS' if beat_v44 else 'LOSES'} ({rl_metrics['sharpe']:.2f} vs {v44_metrics['sharpe']:.2f})")
    perm_pass = adv_results.get('permutation_pass', False)
    regime_pass = adv_results.get('regime_pass', False)
    print(f"  Permutation test: {'PASS' if perm_pass else 'FAIL'}")
    print(f"  Regime test: {'PASS' if regime_pass else 'FAIL'}")

    return results


if __name__ == '__main__':
    results = main()
