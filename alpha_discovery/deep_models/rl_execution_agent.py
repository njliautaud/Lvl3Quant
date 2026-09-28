#!/usr/bin/env python3
"""
RL Execution Agent — PPO-based trade management
=================================================
Takes CNN prediction + book state + position state as input.
Learns optimal entry/exit/hold decisions to maximize risk-adjusted P&L.

State space: CNN z-score, raw pred, book imbalance, depth, spread,
             vol, position P&L, hold time, time of day, pred_std,
             recent price path, rolling conviction
Action space: HOLD, EXIT, SKIP (discrete)
Reward: Risk-adjusted P&L (multiple formulations tested)

Trains on WF OOT prediction data + book tensors.
Uses Neptune RTX 3090 GPU.

Usage:
    python alpha_discovery/deep_models/rl_execution_agent.py --train
    python alpha_discovery/deep_models/rl_execution_agent.py --eval
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
MODEL_DIR = RESULTS_DIR / 'rl_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('rl_agent')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'rl_agent_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)


# ============================================================
# ENVIRONMENT
# ============================================================

@dataclass
class TradeState:
    """Full state visible to the RL agent at each decision point."""
    # CNN signal
    z_score: float          # Normalized conviction (-5 to +5)
    raw_prediction: float   # Un-normalized CNN output
    pred_std: float         # Model uncertainty (rolling std of predictions)
    rolling_z_10: float     # 10-bar rolling mean z-score
    rolling_z_50: float     # 50-bar rolling mean z-score
    rolling_z_100: float    # 100-bar rolling mean z-score

    # Book state
    bid_depth_l1: float     # L1 bid depth (lots)
    ask_depth_l1: float     # L1 ask depth (lots)
    book_imbalance: float   # (bid-ask)/(bid+ask) at L1-L5
    spread: float           # Current spread in ticks
    depth_ratio: float      # Total bid depth / total ask depth (L1-L5)

    # Volatility
    realized_vol: float     # Trailing realized vol
    vol_percentile: float   # Vol percentile (0-100)
    vol_trend: float        # Vol change over last 100 bars

    # Position state
    in_position: bool
    position_direction: int # +1 long, -1 short, 0 flat
    unrealized_pnl: float   # Current P&L in ticks
    hold_time_bars: int     # How long we've held
    entry_price: float
    max_favorable: float    # Best P&L seen during this trade (ticks)
    max_adverse: float      # Worst P&L seen during this trade (ticks)

    # Context
    time_of_day: float      # 0-1 (fraction of RTH session)
    bars_since_last_trade: int

    def to_tensor(self) -> torch.Tensor:
        """Convert to normalized feature tensor."""
        features = [
            self.z_score / 5.0,
            self.raw_prediction,
            self.pred_std * 10,
            self.rolling_z_10 / 5.0,
            self.rolling_z_50 / 5.0,
            self.rolling_z_100 / 5.0,
            np.log1p(self.bid_depth_l1) / 5.0,
            np.log1p(self.ask_depth_l1) / 5.0,
            self.book_imbalance,
            self.spread / 4.0,
            np.clip(self.depth_ratio - 1.0, -2, 2),
            self.realized_vol * 1000,
            self.vol_percentile / 100.0,
            self.vol_trend * 100,
            float(self.in_position),
            self.position_direction,
            np.clip(self.unrealized_pnl / 50.0, -1, 1),
            np.clip(self.hold_time_bars / 18000, 0, 2),  # Normalize to 30-min
            np.clip(self.max_favorable / 30.0, 0, 2),
            np.clip(self.max_adverse / 30.0, -2, 0),
            self.time_of_day,
            np.clip(self.bars_since_last_trade / 18000, 0, 2),
        ]
        return torch.tensor(features, dtype=torch.float32)


STATE_DIM = 22  # Must match to_tensor output size

# Actions
ACTION_SKIP = 0     # Don't enter (when flat) or keep holding (when in position)
ACTION_ENTER = 1    # Enter trade in CNN's predicted direction (only when flat)
ACTION_EXIT = 2     # Exit current position (only when in position)
N_ACTIONS = 3


class TradingEnvironment:
    """
    Simulates trading one day using CNN predictions + book data.
    The RL agent makes decisions at each scan point (every 10 bars = 1 second).
    """

    def __init__(self, preds, mids, book_tensors=None, commission=COMMISSION_TICKS):
        self.preds = preds
        self.mids = mids
        self.book = book_tensors  # (n_bars, 20, 4) or None
        self.n_bars = len(preds)
        self.commission = commission
        self.scan_step = 10  # Decision every 1 second

        # Precompute signals
        self._precompute()

        # State
        self.reset()

    def _precompute(self):
        """Precompute z-scores, vol, rolling stats."""
        n = self.n_bars
        window = 3000

        # Returns for vol
        returns = np.diff(self.mids) / np.maximum(self.mids[:-1], 1e-10)
        returns = np.insert(returns, 0, 0)

        # Rolling vol
        self.roll_vol = pd.Series(returns).rolling(window, min_periods=100).std().values

        # Z-score
        roll_mean = pd.Series(self.preds).rolling(window, min_periods=50).mean().values
        roll_std = np.maximum(pd.Series(self.preds).rolling(window, min_periods=50).std().values, 1e-10)
        self.z_scores = (self.preds - roll_mean) / roll_std

        # Vol percentile
        self.vol_pctile = pd.Series(self.roll_vol).rolling(window, min_periods=100).rank(pct=True).values * 100

        # Pred std (model uncertainty)
        self.pred_std = pd.Series(self.preds).rolling(window, min_periods=100).std().values

        # Rolling mean z-scores at different windows
        z_series = pd.Series(self.z_scores)
        self.roll_z_10 = z_series.rolling(10, min_periods=1).mean().values
        self.roll_z_50 = z_series.rolling(50, min_periods=1).mean().values
        self.roll_z_100 = z_series.rolling(100, min_periods=1).mean().values

        # Vol trend
        vol_series = pd.Series(self.roll_vol)
        self.vol_trend = (vol_series - vol_series.rolling(100, min_periods=10).mean()).values

        # Book features (if available)
        if self.book is not None:
            self.bid_depth_l1 = self.book[:, 0, 1]   # L1 bid depth
            self.ask_depth_l1 = self.book[:, 10, 1]   # L1 ask depth
            bid_total = self.book[:, :5, 1].sum(axis=1)
            ask_total = self.book[:, 10:15, 1].sum(axis=1)
            self.book_imbalance = (bid_total - ask_total) / (bid_total + ask_total + 1e-10)
            self.depth_ratio = bid_total / (ask_total + 1e-10)
            self.spread = np.abs(self.book[:, 10, 0] - self.book[:, 0, 0])  # Ask L1 - Bid L1 price_rel
        else:
            self.bid_depth_l1 = np.zeros(n)
            self.ask_depth_l1 = np.zeros(n)
            self.book_imbalance = np.zeros(n)
            self.depth_ratio = np.ones(n)
            self.spread = np.ones(n)

    def reset(self):
        """Reset environment to start of day."""
        self.bar_idx = 18000  # Skip first 30 min
        self.in_position = False
        self.position_dir = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.max_fav = 0.0
        self.max_adv = 0.0
        self.bars_since_trade = 0
        self.trades = []
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.max_dd = 0.0
        return self._get_state()

    def _get_state(self) -> TradeState:
        i = self.bar_idx
        if i >= self.n_bars or np.isnan(self.z_scores[i]):
            i = max(0, min(i, self.n_bars - 1))

        unreal_pnl = 0.0
        if self.in_position:
            unreal_pnl = (self.mids[i] - self.entry_price) / TICK * self.position_dir

        return TradeState(
            z_score=float(np.nan_to_num(self.z_scores[i])),
            raw_prediction=float(np.nan_to_num(self.preds[i])),
            pred_std=float(np.nan_to_num(self.pred_std[i])),
            rolling_z_10=float(np.nan_to_num(self.roll_z_10[i])),
            rolling_z_50=float(np.nan_to_num(self.roll_z_50[i])),
            rolling_z_100=float(np.nan_to_num(self.roll_z_100[i])),
            bid_depth_l1=float(self.bid_depth_l1[i]),
            ask_depth_l1=float(self.ask_depth_l1[i]),
            book_imbalance=float(self.book_imbalance[i]),
            spread=float(self.spread[i]),
            depth_ratio=float(np.clip(self.depth_ratio[i], 0, 10)),
            realized_vol=float(np.nan_to_num(self.roll_vol[i])),
            vol_percentile=float(np.nan_to_num(self.vol_pctile[i])),
            vol_trend=float(np.nan_to_num(self.vol_trend[i])),
            in_position=self.in_position,
            position_direction=self.position_dir,
            unrealized_pnl=unreal_pnl,
            hold_time_bars=self.bar_idx - self.entry_bar if self.in_position else 0,
            entry_price=self.entry_price,
            max_favorable=self.max_fav,
            max_adverse=self.max_adv,
            time_of_day=(self.bar_idx - 18000) / (234000 - 27000),
            bars_since_last_trade=self.bars_since_trade,
        )

    def step(self, action: int) -> tuple:
        """
        Execute one step. Returns (next_state, reward, done, info).
        """
        i = self.bar_idx
        reward = 0.0
        info = {}

        # End of day check
        if i >= self.n_bars - 9000:  # Last 15 min = no new trades
            if self.in_position:
                # Force exit at market
                exit_pnl = (self.mids[i] - self.entry_price) / TICK * self.position_dir
                net_pnl = exit_pnl - self.commission - SPREAD_TICKS * 0.5  # Market exit cost
                reward = net_pnl
                self.total_pnl += net_pnl * TICK_VAL
                self.trades.append({'pnl_ticks': net_pnl, 'type': 'forced_exit'})
                self.in_position = False
            return self._get_state(), reward, True, info

        if action == ACTION_ENTER and not self.in_position:
            # Enter trade in CNN's direction
            direction = 1 if self.z_scores[i] > 0 else -1
            self.in_position = True
            self.position_dir = direction
            self.entry_price = self.mids[i]
            self.entry_bar = i
            self.max_fav = 0.0
            self.max_adv = 0.0
            self.bars_since_trade = 0
            info['action'] = 'enter'

        elif action == ACTION_EXIT and self.in_position:
            # Exit position (market order — pays spread)
            exit_pnl = (self.mids[i] - self.entry_price) / TICK * self.position_dir
            net_pnl = exit_pnl - self.commission - SPREAD_TICKS * 0.5
            reward = net_pnl
            self.total_pnl += net_pnl * TICK_VAL
            self.trades.append({
                'pnl_ticks': net_pnl,
                'hold_bars': i - self.entry_bar,
                'max_fav': self.max_fav,
                'max_adv': self.max_adv,
                'type': 'rl_exit'
            })
            self.in_position = False
            self.position_dir = 0
            info['action'] = 'exit'
            info['pnl'] = net_pnl

        else:
            # HOLD or SKIP — no action
            info['action'] = 'hold' if self.in_position else 'skip'

        # Update position tracking
        if self.in_position:
            current_pnl = (self.mids[i] - self.entry_price) / TICK * self.position_dir
            self.max_fav = max(self.max_fav, current_pnl)
            self.max_adv = min(self.max_adv, current_pnl)

            # Force exit at 60 min max hold
            if i - self.entry_bar >= 36000:
                exit_pnl = (self.mids[i] - self.entry_price) / TICK * self.position_dir
                net_pnl = exit_pnl - self.commission - SPREAD_TICKS * 0.5
                reward = net_pnl
                self.total_pnl += net_pnl * TICK_VAL
                self.trades.append({'pnl_ticks': net_pnl, 'type': 'max_hold_exit'})
                self.in_position = False

        # Track drawdown
        self.peak_equity = max(self.peak_equity, self.total_pnl)
        dd = self.peak_equity - self.total_pnl
        self.max_dd = max(self.max_dd, dd)

        self.bars_since_trade += self.scan_step
        self.bar_idx += self.scan_step

        done = self.bar_idx >= self.n_bars - 9000
        return self._get_state(), reward, done, info


# ============================================================
# PPO AGENT
# ============================================================

class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        shared = self.shared(state)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def get_action(self, state, deterministic=False):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.log(probs[action] + 1e-8)
        return action.item(), log_prob, value


class RewardShaper:
    """Multiple reward formulations to test."""

    @staticmethod
    def raw_pnl(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        """Pure P&L in ticks."""
        return pnl_ticks

    @staticmethod
    def sharpe_like(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        """P&L penalized by magnitude (encourages consistency)."""
        return pnl_ticks / (1 + abs(pnl_ticks) * 0.1)

    @staticmethod
    def drawdown_penalized(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        """P&L with penalty for drawdown during trade."""
        dd_penalty = max(0, -max_dd_ticks) * 0.05  # 5% of max adverse
        return pnl_ticks - dd_penalty

    @staticmethod
    def time_efficient(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        """Reward quick profitable trades, penalize long holds."""
        time_factor = max(0.5, 1.0 - hold_bars / 36000)  # Decay over 60 min
        return pnl_ticks * time_factor

    @staticmethod
    def calmar(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        """P&L / max drawdown ratio."""
        if max_dd_ticks == 0:
            return pnl_ticks
        return pnl_ticks / (1 + abs(max_dd_ticks) * 0.2)


class PPOTrainer:
    """PPO training loop."""

    def __init__(self, device='cuda', lr=3e-4, gamma=0.99, eps_clip=0.2,
                 reward_fn='drawdown_penalized'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

        reward_fns = {
            'raw_pnl': RewardShaper.raw_pnl,
            'sharpe_like': RewardShaper.sharpe_like,
            'drawdown_penalized': RewardShaper.drawdown_penalized,
            'time_efficient': RewardShaper.time_efficient,
            'calmar': RewardShaper.calmar,
        }
        self.reward_fn = reward_fns.get(reward_fn, RewardShaper.drawdown_penalized)
        self.reward_fn_name = reward_fn
        log.info(f"PPO Agent | Device: {self.device} | Reward: {reward_fn}")

    def collect_episode(self, env: TradingEnvironment):
        """Run one episode (one trading day), collect transitions."""
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        state = env.reset()
        done = False

        while not done:
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor.squeeze(0))

            next_state, reward, done, info = env.step(action)

            # Shape reward
            if info.get('action') == 'exit':
                trade = env.trades[-1] if env.trades else {}
                reward = self.reward_fn(
                    reward,
                    max_dd_ticks=trade.get('max_adv', 0),
                    hold_bars=trade.get('hold_bars', 0),
                )

            states.append(state_tensor.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            dones.append(done)

            state = next_state

        return states, actions, rewards, log_probs, values, dones

    def compute_returns(self, rewards, values, dones):
        """Compute GAE advantages and returns."""
        returns = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
                gae = 0
            delta = rewards[i] + self.gamma * next_value - values[i].item()
            gae = delta + self.gamma * 0.95 * gae
            returns.insert(0, gae + values[i].item())
            next_value = values[i].item()

        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self, states, actions, old_log_probs, returns, values):
        """PPO policy update."""
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        advantages = returns - torch.stack(values).detach().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            logits, new_values = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()

    def train(self, envs: list, n_epochs=100):
        """Train across multiple day environments."""
        log.info(f"Training on {len(envs)} days for {n_epochs} epochs")

        best_avg_pnl = -np.inf
        for epoch in range(n_epochs):
            epoch_pnl = []
            epoch_trades = []

            for env in envs:
                states, actions, rewards, log_probs, values, dones = self.collect_episode(env)

                if len(states) < 10:
                    continue

                returns = self.compute_returns(rewards, values, dones)
                a_loss, c_loss, ent = self.update(states, actions, log_probs, returns, values)

                epoch_pnl.append(env.total_pnl)
                epoch_trades.append(len(env.trades))

            avg_pnl = np.mean(epoch_pnl) if epoch_pnl else 0
            avg_trades = np.mean(epoch_trades) if epoch_trades else 0

            if epoch % 5 == 0:
                log.info(f"  Epoch {epoch:>3d}: Avg P&L ${avg_pnl:>8,.2f} | "
                        f"Avg trades {avg_trades:.1f} | "
                        f"Total ${sum(epoch_pnl):>10,.2f}")

            if avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                torch.save(self.policy.state_dict(),
                          str(MODEL_DIR / f'rl_best_{self.reward_fn_name}.pt'))

        log.info(f"Best avg P&L: ${best_avg_pnl:,.2f}")
        return best_avg_pnl

    def evaluate(self, envs: list):
        """Evaluate policy without updating."""
        self.policy.eval()
        results = []

        with torch.no_grad():
            for env in envs:
                state = env.reset()
                done = False
                while not done:
                    state_tensor = state.to_tensor().to(self.device)
                    action, _, _ = self.policy.get_action(state_tensor, deterministic=True)
                    state, _, done, _ = env.step(action)

                results.append({
                    'pnl': env.total_pnl,
                    'trades': len(env.trades),
                    'max_dd': env.max_dd,
                })

        self.policy.train()
        return results


# ============================================================
# MAIN
# ============================================================

def load_environments():
    """Load all WF OOT days as trading environments."""
    log.info("Loading WF predictions...")
    pred_data = np.load(str(PRED_FILE), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in pred_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)}")

    envs = []
    for date in dates:
        preds = pred_data[f'{date}_preds']
        mids = pred_data[f'{date}_mid']
        if len(preds) < 5000:
            continue

        # Try to load book tensors
        book_file = BOOK_DIR / f'{date}_book_tensors.npz'
        book = None
        if book_file.exists():
            book_data = np.load(str(book_file))
            book = book_data['book_tensors']

        envs.append(TradingEnvironment(preds, mids, book))
        log.info(f"  {date}: {len(preds)} bars, book={'yes' if book is not None else 'no'}")

    return envs, dates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--reward', default='drawdown_penalized',
                       choices=['raw_pnl', 'sharpe_like', 'drawdown_penalized', 'time_efficient', 'calmar'])
    parser.add_argument('--all-rewards', action='store_true', help='Train with all reward functions')
    args = parser.parse_args()

    envs, dates = load_environments()

    if args.train or args.all_rewards:
        rewards_to_test = ['raw_pnl', 'sharpe_like', 'drawdown_penalized', 'time_efficient', 'calmar'] \
            if args.all_rewards else [args.reward]

        for reward_fn in rewards_to_test:
            log.info(f"\n{'='*60}")
            log.info(f"Training with reward: {reward_fn}")
            log.info(f"{'='*60}")

            # Split: 24 days train, 7 days eval
            train_envs = envs[:24]
            eval_envs = envs[24:]

            trainer = PPOTrainer(reward_fn=reward_fn)
            trainer.train(train_envs, n_epochs=args.epochs)

            # Evaluate
            log.info("\nEvaluation on held-out days:")
            eval_results = trainer.evaluate(eval_envs)
            for i, r in enumerate(eval_results):
                log.info(f"  Day {dates[24+i]}: P&L ${r['pnl']:>8,.2f} | "
                        f"Trades {r['trades']} | Max DD ${r['max_dd']:>6,.2f}")
            total_eval = sum(r['pnl'] for r in eval_results)
            log.info(f"  TOTAL eval P&L: ${total_eval:,.2f}")

    if args.eval:
        trainer = PPOTrainer(reward_fn=args.reward)
        model_path = MODEL_DIR / f'rl_best_{args.reward}.pt'
        if model_path.exists():
            trainer.policy.load_state_dict(torch.load(str(model_path)))
            results = trainer.evaluate(envs)
            total = sum(r['pnl'] for r in results)
            log.info(f"Full evaluation: ${total:,.2f} across {len(results)} days")
        else:
            log.error(f"No model found at {model_path}")
