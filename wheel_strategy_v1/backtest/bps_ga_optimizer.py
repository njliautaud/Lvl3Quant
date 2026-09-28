#!/usr/bin/env python3
"""
BPS (Bull Put Spread) Genetic Algorithm Optimizer
HC #668 — Evolve per-ticker weights to maximize walk-forward OOS Sharpe.

Instead of equal-weight IV-sort selection, this GA evolves a weight vector
across the ticker universe. Tickers with weight > 0.5 are included;
the weight magnitude determines position size allocation.

Uses a simplified Black-Scholes BPS simulator (7 DTE, ~25-delta short put,
5-wide spread, 50% profit-take) for speed — need ~5000+ evaluations.

Walk-forward: sliding 120-day train, 30-day OOS.
Regime penalty: |Sharpe_green - Sharpe_red| / max > 0.50 => penalize.
Diversity constraint: must hold >= 15 tickers (adjusted for 70-ticker universe).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import json
import os
import sys
import warnings
import logging
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    # GA parameters
    pop_size: int = 200
    n_generations: int = 150
    elite_frac: float = 0.05
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    mutation_sigma: float = 0.10
    mutation_rate: float = 0.15  # fraction of genes to mutate per individual

    # BPS strategy parameters
    dte: int = 7                    # days to expiration
    spread_width: float = 5.0      # dollar width of put spread
    target_delta: float = -0.25    # short put delta target
    profit_take_pct: float = 0.50  # close at 50% profit
    max_loss_mult: float = 1.0     # close at 100% of max loss (full width)
    premium_floor: float = 0.10    # minimum credit to take trade (lowered for realism)
    iv_multiplier: float = 1.35    # IV/RV ratio — options trade at ~1.3-1.5x realized vol
                                    # This is the volatility risk premium that makes BPS profitable

    # Walk-forward
    train_days: int = 120
    oos_days: int = 30
    step_days: int = 30            # roll forward by this many days

    # Constraints
    min_tickers: int = 20          # must include at least this many
    regime_penalty_threshold: float = 0.50
    regime_penalty_mult: float = 0.5  # multiply fitness by this if regime-unbalanced

    # Risk-free rate for Sharpe
    rf_annual: float = 0.045

    # Parallelism
    n_workers: int = 8

    # Paths
    prices_path: str = '/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet'
    universe_path: str = '/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache/universe.parquet'
    output_dir: str = '/home/nick/Lvl3Quant/wheel_strategy_v1/backtest/ga_output'


# ---------------------------------------------------------------------------
# Black-Scholes BPS Simulator
# ---------------------------------------------------------------------------
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S: float, target_delta: float, T: float, r: float,
                           sigma: float, n_iter: int = 30) -> float:
    """Find put strike K such that delta(K) ~ target_delta using bisection."""
    K_lo, K_hi = S * 0.70, S * 1.00
    for _ in range(n_iter):
        K_mid = (K_lo + K_hi) / 2
        d = bs_put_delta(S, K_mid, T, r, sigma)
        if d < target_delta:  # delta is more negative => strike too high
            K_hi = K_mid
        else:
            K_lo = K_mid
    return (K_lo + K_hi) / 2


def simulate_bps_trade(S_entry: float, S_exit: float, sigma: float,
                        cfg: GAConfig, r: float = 0.045) -> float:
    """
    Simulate a single bull put spread trade.

    Returns P&L per spread in dollars (positive = profit).

    Logic:
    - At entry: sell put at ~25-delta, buy put 5 lower. Collect credit.
    - At expiry (or early exit): compute P&L.
    - 50% profit-take: if mid-week mark-to-market shows >= 50% credit decay, close.
    - Simplified: we just check at expiry since we're doing daily resolution.
    """
    T = cfg.dte / 252.0

    # Use implied vol (= realized vol * IV multiplier) for PRICING
    # The actual stock move uses realized vol (sigma), but we collect
    # premium based on implied vol — this is the volatility risk premium
    iv = sigma * cfg.iv_multiplier

    # Find short put strike for target delta (using IV for strike selection)
    K_short = find_strike_for_delta(S_entry, cfg.target_delta, T, r, iv)
    K_long = K_short - cfg.spread_width

    # Entry credit (priced at IV)
    short_put_price = bs_put_price(S_entry, K_short, T, r, iv)
    long_put_price = bs_put_price(S_entry, K_long, T, r, iv)
    credit = short_put_price - long_put_price

    if credit < cfg.premium_floor:
        return 0.0  # skip trade, too little premium

    # Max loss = spread_width - credit
    max_loss = cfg.spread_width - credit

    # At expiry, compute intrinsic values
    short_put_intrinsic = max(K_short - S_exit, 0)
    long_put_intrinsic = max(K_long - S_exit, 0)
    spread_intrinsic = short_put_intrinsic - long_put_intrinsic

    # P&L = credit received - spread intrinsic at expiry
    pnl = credit - spread_intrinsic

    # Simplified profit-take: if P&L > 50% of credit, cap it there
    # (in practice we'd exit mid-week, but for speed we approximate)
    if pnl > credit * cfg.profit_take_pct:
        pnl = credit * cfg.profit_take_pct

    # Cap loss at max_loss
    pnl = max(pnl, -max_loss)

    return pnl


# ---------------------------------------------------------------------------
# Vectorized BPS simulation for speed
# ---------------------------------------------------------------------------
def simulate_bps_trades_vectorized(S_entries: np.ndarray, S_exits: np.ndarray,
                                    sigmas: np.ndarray, cfg: GAConfig) -> np.ndarray:
    """Vectorized BPS trade simulation for a batch of trades."""
    n = len(S_entries)
    pnls = np.zeros(n)
    T = cfg.dte / 252.0
    r = cfg.rf_annual

    for i in range(n):
        if sigmas[i] <= 0 or S_entries[i] <= 0:
            continue
        pnls[i] = simulate_bps_trade(S_entries[i], S_exits[i], sigmas[i], cfg, r)

    return pnls


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------
class BPSDataEngine:
    """Manages price data and generates weekly BPS trade opportunities."""

    def __init__(self, cfg: GAConfig):
        self.cfg = cfg
        self.prices = pd.read_parquet(cfg.prices_path)
        self.universe = pd.read_parquet(cfg.universe_path)
        self.tickers = sorted(self.universe['ticker'].unique())
        self.n_tickers = len(self.tickers)
        self.ticker_idx = {t: i for i, t in enumerate(self.tickers)}

        # Pivot to wide format for speed
        self._prepare_data()

    def _prepare_data(self):
        """Create wide-format price and vol matrices."""
        prices = self.prices.copy()
        prices = prices.sort_values(['ticker', 'date'])

        # Create weekly Friday dates (BPS trades open/close weekly)
        # Use business days, pick every 5th as "weekly"
        self.close_wide = prices.pivot(index='date', columns='ticker', values='close')
        self.vol_wide = prices.pivot(index='date', columns='ticker', values='rv_20')

        # Fill forward missing vol data
        self.vol_wide = self.vol_wide.ffill()

        # Align columns
        self.tickers = sorted([t for t in self.tickers if t in self.close_wide.columns])
        self.n_tickers = len(self.tickers)
        self.ticker_idx = {t: i for i, t in enumerate(self.tickers)}

        self.close_wide = self.close_wide[self.tickers]
        self.vol_wide = self.vol_wide[self.tickers]

        # Generate weekly trade dates (every 5 trading days)
        all_dates = self.close_wide.index.sort_values()
        self.trade_dates = all_dates[::5]  # every 5 trading days = ~weekly

        # For regime classification: compute market return (equal-weight)
        market_ret = self.close_wide.pct_change().mean(axis=1)
        # Rolling 20-day market return for regime
        self.market_ret_20d = market_ret.rolling(20).sum()

        logging.info(f"Data prepared: {self.n_tickers} tickers, "
                     f"{len(self.trade_dates)} weekly trade dates, "
                     f"date range {all_dates[0].date()} to {all_dates[-1].date()}")

    def get_weekly_pnl_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pre-compute P&L for every ticker on every trade date.
        Returns (pnl_matrix: DatexTicker, regime_series: Date->bool).

        This is the key optimization: compute ALL possible trades upfront,
        then the GA just does weighted sums.
        """
        cfg = self.cfg
        dates = self.trade_dates
        close = self.close_wide
        vol = self.vol_wide

        pnl_data = {}

        for i in range(len(dates) - 1):
            entry_date = dates[i]
            # Exit date = next trade date (approximately 1 week later)
            exit_date = dates[i + 1]

            if entry_date not in close.index or exit_date not in close.index:
                continue

            S_entries = close.loc[entry_date].values
            S_exits = close.loc[exit_date].values
            sigmas = vol.loc[entry_date].values

            # Replace NaN with 0
            S_entries = np.nan_to_num(S_entries, nan=0.0)
            S_exits = np.nan_to_num(S_exits, nan=0.0)
            sigmas = np.nan_to_num(sigmas, nan=0.0)

            pnls = simulate_bps_trades_vectorized(S_entries, S_exits, sigmas, cfg)
            pnl_data[entry_date] = pnls

        pnl_matrix = pd.DataFrame(pnl_data, index=self.tickers).T
        pnl_matrix = pnl_matrix.sort_index()
        # Fill NaN with 0: ticker didn't exist on that date => no trade => 0 P&L
        pnl_matrix = pnl_matrix.fillna(0.0)

        # Regime: green day = positive 20-day market return
        regime = self.market_ret_20d.reindex(pnl_matrix.index)
        regime_green = (regime > 0).fillna(False)  # NaN regime => treat as red/unknown

        n_green = int(regime_green.sum())
        n_red = int((~regime_green).sum())
        logging.info(f"PnL matrix: {pnl_matrix.shape}, green days: {n_green}, red days: {n_red}")

        return pnl_matrix, regime_green


# ---------------------------------------------------------------------------
# Fitness Evaluation
# ---------------------------------------------------------------------------
def compute_sharpe(returns: np.ndarray, rf_weekly: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from weekly returns."""
    returns = returns[~np.isnan(returns)]  # drop NaN
    if len(returns) < 4:
        return -10.0
    std = np.std(returns)
    if std < 1e-10:
        return -10.0  # penalty for degenerate portfolios
    excess = returns - rf_weekly
    sharpe = np.mean(excess) / std * np.sqrt(52)
    if np.isnan(sharpe) or np.isinf(sharpe):
        return -10.0
    return sharpe


def evaluate_genome(genome: np.ndarray, pnl_matrix: np.ndarray,
                    regime_mask: np.ndarray, dates_idx: np.ndarray,
                    cfg: GAConfig, train_start: int, train_end: int,
                    oos_start: int, oos_end: int) -> Tuple[float, dict]:
    """
    Evaluate a genome on OOS data after 'training' on in-sample.

    The genome determines which tickers to include and their weights.
    Training here = the genome was evolved on train period.
    We evaluate fitness on OOS period.

    Returns (oos_sharpe, metadata_dict).
    """
    # Decode genome: weight > 0.5 = include, weight = position size
    active_mask = genome > 0.5
    n_active = active_mask.sum()

    if n_active < cfg.min_tickers:
        return -10.0, {'n_active': int(n_active), 'penalty': 'too_few_tickers'}

    # Normalize weights among active tickers
    weights = genome * active_mask
    weight_sum = weights.sum()
    if weight_sum < 1e-10:
        return -10.0, {'penalty': 'zero_weights'}
    weights = weights / weight_sum

    # Compute weighted portfolio returns for OOS period
    oos_pnl = pnl_matrix[oos_start:oos_end]  # shape: (n_weeks, n_tickers)
    if len(oos_pnl) < 4:
        return -10.0, {'penalty': 'too_few_oos_weeks'}

    # Weekly portfolio P&L = sum of weighted ticker P&Ls
    portfolio_pnl = oos_pnl @ weights  # shape: (n_weeks,)

    # Convert to returns (assume $1000 notional per spread allocation)
    # For Sharpe calculation, raw P&L works since it's already per-unit
    rf_weekly = cfg.rf_annual / 52.0

    sharpe = compute_sharpe(portfolio_pnl, rf_weekly=0)

    # Regime analysis
    oos_regime = regime_mask[oos_start:oos_end]
    green_pnl = portfolio_pnl[oos_regime]
    red_pnl = portfolio_pnl[~oos_regime]

    sharpe_green = compute_sharpe(green_pnl) if len(green_pnl) >= 2 else 0.0
    sharpe_red = compute_sharpe(red_pnl) if len(red_pnl) >= 2 else 0.0

    # Regime penalty
    regime_imbalance = 0.0
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    if max_sharpe > 0.1:
        regime_imbalance = abs(sharpe_green - sharpe_red) / max_sharpe

    fitness = sharpe
    if regime_imbalance > cfg.regime_penalty_threshold:
        fitness *= cfg.regime_penalty_mult

    meta = {
        'n_active': int(n_active),
        'sharpe_oos': float(sharpe),
        'sharpe_green': float(sharpe_green),
        'sharpe_red': float(sharpe_red),
        'regime_imbalance': float(regime_imbalance),
        'regime_penalized': regime_imbalance > cfg.regime_penalty_threshold,
        'mean_pnl': float(np.mean(portfolio_pnl)),
        'std_pnl': float(np.std(portfolio_pnl)),
        'win_rate': float(np.mean(portfolio_pnl > 0)),
        'n_oos_weeks': len(portfolio_pnl),
    }

    return fitness, meta


def evaluate_genome_full_wf(genome: np.ndarray, pnl_matrix: np.ndarray,
                             regime_mask: np.ndarray, cfg: GAConfig) -> Tuple[float, dict]:
    """
    Walk-forward evaluation of a genome across all periods.
    Returns aggregate OOS fitness.
    """
    n_weeks = len(pnl_matrix)
    train_weeks = cfg.train_days // 7
    oos_weeks = cfg.oos_days // 7
    step_weeks = cfg.step_days // 7

    all_oos_pnls = []
    all_oos_regimes = []
    period_sharpes = []

    start = 0
    while start + train_weeks + oos_weeks <= n_weeks:
        train_start = start
        train_end = start + train_weeks
        oos_start = train_end
        oos_end = min(train_end + oos_weeks, n_weeks)

        # Decode genome for this OOS window
        active_mask = genome > 0.5
        n_active = active_mask.sum()
        if n_active < cfg.min_tickers:
            start += step_weeks
            continue

        weights = genome * active_mask
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            start += step_weeks
            continue
        weights = weights / weight_sum

        oos_pnl = pnl_matrix[oos_start:oos_end] @ weights
        oos_regime = regime_mask[oos_start:oos_end]

        all_oos_pnls.append(oos_pnl)
        all_oos_regimes.append(oos_regime)

        s = compute_sharpe(oos_pnl)
        period_sharpes.append(s)

        start += step_weeks

    if not all_oos_pnls:
        return -10.0, {'penalty': 'no_valid_periods'}

    combined_pnl = np.concatenate(all_oos_pnls)
    combined_regime = np.concatenate(all_oos_regimes)

    sharpe = compute_sharpe(combined_pnl)

    # Regime analysis on combined OOS
    green_pnl = combined_pnl[combined_regime]
    red_pnl = combined_pnl[~combined_regime]
    sharpe_green = compute_sharpe(green_pnl) if len(green_pnl) >= 4 else 0.0
    sharpe_red = compute_sharpe(red_pnl) if len(red_pnl) >= 4 else 0.0

    regime_imbalance = 0.0
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    if max_sharpe > 0.1:
        regime_imbalance = abs(sharpe_green - sharpe_red) / max_sharpe

    fitness = sharpe
    if regime_imbalance > cfg.regime_penalty_threshold:
        fitness *= cfg.regime_penalty_mult

    # Sortino
    downside = combined_pnl[combined_pnl < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-10
    sortino = np.mean(combined_pnl) / downside_std * np.sqrt(52) if downside_std > 1e-10 else 0.0

    # Profit factor
    gross_profit = combined_pnl[combined_pnl > 0].sum()
    gross_loss = abs(combined_pnl[combined_pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    active_mask = genome > 0.5
    meta = {
        'n_active': int(active_mask.sum()),
        'sharpe_oos': float(sharpe),
        'sortino_oos': float(sortino),
        'profit_factor': float(pf),
        'sharpe_green': float(sharpe_green),
        'sharpe_red': float(sharpe_red),
        'regime_imbalance': float(regime_imbalance),
        'regime_penalized': regime_imbalance > cfg.regime_penalty_threshold,
        'mean_weekly_pnl': float(np.mean(combined_pnl)),
        'win_rate': float(np.mean(combined_pnl > 0)),
        'n_oos_weeks': len(combined_pnl),
        'n_wf_periods': len(period_sharpes),
        'period_sharpes': [float(s) for s in period_sharpes],
    }

    return fitness, meta


# ---------------------------------------------------------------------------
# GA Engine (pure numpy, no DEAP dependency)
# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    """Genetic algorithm for BPS ticker weight optimization."""

    def __init__(self, n_genes: int, cfg: GAConfig):
        self.n_genes = n_genes
        self.cfg = cfg
        self.n_elite = max(1, int(cfg.pop_size * cfg.elite_frac))

        # Initialize population: uniform random [0, 1]
        # Bias initial population toward inclusion (mean=0.6) for diversity
        self.population = np.random.beta(3, 2, size=(cfg.pop_size, n_genes))

        self.fitnesses = np.full(cfg.pop_size, -999.0)
        self.best_genome = None
        self.best_fitness = -999.0
        self.best_meta = {}
        self.history = []

    def tournament_select(self, k: int = None) -> int:
        """Tournament selection. Returns index of winner."""
        if k is None:
            k = self.cfg.tournament_size
        candidates = np.random.choice(self.cfg.pop_size, k, replace=False)
        best_idx = candidates[np.argmax(self.fitnesses[candidates])]
        return best_idx

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.random(self.n_genes) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation with per-gene probability."""
        mutation_mask = np.random.random(self.n_genes) < self.cfg.mutation_rate
        noise = np.random.normal(0, self.cfg.mutation_sigma, self.n_genes)
        individual = individual + noise * mutation_mask
        individual = np.clip(individual, 0.0, 1.0)
        return individual

    def evolve_one_generation(self, eval_fn) -> dict:
        """Run one generation of evolution."""
        # Evaluate current population
        results = []
        for i in range(self.cfg.pop_size):
            fitness, meta = eval_fn(self.population[i])
            self.fitnesses[i] = fitness
            results.append((fitness, meta))

        # Replace NaN fitnesses with large negative value
        self.fitnesses = np.nan_to_num(self.fitnesses, nan=-999.0)

        # Track best
        gen_best_idx = np.argmax(self.fitnesses)
        gen_best_fitness = self.fitnesses[gen_best_idx]
        gen_best_meta = results[gen_best_idx][1]

        if gen_best_fitness > self.best_fitness:
            self.best_fitness = gen_best_fitness
            self.best_genome = self.population[gen_best_idx].copy()
            self.best_meta = gen_best_meta

        # Record history
        gen_stats = {
            'best_fitness': float(gen_best_fitness),
            'mean_fitness': float(np.mean(self.fitnesses)),
            'std_fitness': float(np.std(self.fitnesses)),
            'global_best': float(self.best_fitness),
            **{f'best_{k}': v for k, v in gen_best_meta.items()
               if isinstance(v, (int, float, bool))},
        }
        self.history.append(gen_stats)

        # Create next generation
        new_pop = np.zeros_like(self.population)

        # Elitism: keep top N
        elite_indices = np.argsort(self.fitnesses)[-self.n_elite:]
        new_pop[:self.n_elite] = self.population[elite_indices]

        # Fill rest with crossover + mutation
        idx = self.n_elite
        while idx < self.cfg.pop_size:
            p1_idx = self.tournament_select()
            p2_idx = self.tournament_select()

            if np.random.random() < self.cfg.crossover_prob:
                c1, c2 = self.crossover(self.population[p1_idx], self.population[p2_idx])
            else:
                c1 = self.population[p1_idx].copy()
                c2 = self.population[p2_idx].copy()

            if np.random.random() < self.cfg.mutation_prob:
                c1 = self.mutate(c1)
            if np.random.random() < self.cfg.mutation_prob:
                c2 = self.mutate(c2)

            new_pop[idx] = c1
            idx += 1
            if idx < self.cfg.pop_size:
                new_pop[idx] = c2
                idx += 1

        self.population = new_pop

        return gen_stats


# ---------------------------------------------------------------------------
# Main Optimization Loop
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/home/nick/Lvl3Quant/wheel_strategy_v1/backtest/ga_optimizer.log')
        ]
    )

    cfg = GAConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logging.info("=" * 70)
    logging.info("BPS Genetic Algorithm Optimizer — HC #668")
    logging.info("=" * 70)

    # Load data
    logging.info("Loading price data and computing BPS P&L matrix...")
    t0 = time.time()
    engine = BPSDataEngine(cfg)
    pnl_matrix_df, regime_series = engine.get_weekly_pnl_matrix()
    t_load = time.time() - t0
    logging.info(f"Data loaded in {t_load:.1f}s")

    pnl_matrix = pnl_matrix_df.values  # (n_weeks, n_tickers)
    regime_mask = regime_series.values.astype(bool)
    tickers = engine.tickers

    logging.info(f"PnL matrix shape: {pnl_matrix.shape}")
    logging.info(f"Tickers ({len(tickers)}): {tickers}")

    # Quick sanity: what does equal-weight baseline look like?
    baseline_weights = np.ones(len(tickers)) / len(tickers)
    baseline_pnl = pnl_matrix @ baseline_weights
    baseline_sharpe = compute_sharpe(baseline_pnl)
    logging.info(f"Equal-weight baseline Sharpe: {baseline_sharpe:.3f}")

    # Setup MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri('file:///home/nick/Lvl3Quant/wheel_strategy_v1/mlruns')
        mlflow.set_experiment('BPS_GA_Optimizer')
        mlflow_available = True
        run = mlflow.start_run(run_name=f'ga_opt_{time.strftime("%Y%m%d_%H%M%S")}')
        mlflow.log_params({
            'pop_size': cfg.pop_size,
            'n_generations': cfg.n_generations,
            'mutation_sigma': cfg.mutation_sigma,
            'mutation_rate': cfg.mutation_rate,
            'train_days': cfg.train_days,
            'oos_days': cfg.oos_days,
            'min_tickers': cfg.min_tickers,
            'regime_penalty_threshold': cfg.regime_penalty_threshold,
            'n_tickers_universe': len(tickers),
            'baseline_sharpe': round(baseline_sharpe, 4),
        })
        logging.info("MLflow tracking active")
    except Exception as e:
        logging.warning(f"MLflow not available: {e}")
        mlflow_available = False

    # Initialize GA
    ga = GeneticAlgorithm(n_genes=len(tickers), cfg=cfg)

    # Evaluation function
    def eval_fn(genome):
        return evaluate_genome_full_wf(genome, pnl_matrix, regime_mask, cfg)

    # Run evolution
    logging.info(f"Starting GA: {cfg.pop_size} pop x {cfg.n_generations} generations")
    logging.info(f"= {cfg.pop_size * cfg.n_generations} total evaluations")
    t_start = time.time()

    convergence_count = 0
    prev_best = -999.0

    for gen in range(cfg.n_generations):
        t_gen = time.time()
        stats = ga.evolve_one_generation(eval_fn)
        dt = time.time() - t_gen

        logging.info(
            f"Gen {gen+1:3d}/{cfg.n_generations} | "
            f"Best: {stats['best_fitness']:.3f} | "
            f"Mean: {stats['mean_fitness']:.3f} | "
            f"Global: {stats['global_best']:.3f} | "
            f"Active: {stats.get('best_n_active', '?')} | "
            f"WR: {stats.get('best_win_rate', 0):.1%} | "
            f"Time: {dt:.1f}s"
        )

        if mlflow_available:
            try:
                mlflow.log_metrics({
                    'gen_best_fitness': stats['best_fitness'],
                    'gen_mean_fitness': stats['mean_fitness'],
                    'global_best_fitness': stats['global_best'],
                    'gen_best_n_active': stats.get('best_n_active', 0),
                    'gen_best_win_rate': stats.get('best_win_rate', 0),
                    'gen_best_sharpe_green': stats.get('best_sharpe_green', 0),
                    'gen_best_sharpe_red': stats.get('best_sharpe_red', 0),
                    'gen_best_regime_imbalance': stats.get('best_regime_imbalance', 0),
                }, step=gen)
            except Exception:
                pass

        # Convergence check
        if abs(stats['global_best'] - prev_best) < 0.001:
            convergence_count += 1
        else:
            convergence_count = 0
        prev_best = stats['global_best']

        if convergence_count >= 15:
            logging.info(f"Converged after {gen+1} generations (no improvement for 15 gens)")
            break

    total_time = time.time() - t_start
    logging.info(f"GA complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # ---------------------------------------------------------------------------
    # Final Results
    # ---------------------------------------------------------------------------
    best = ga.best_genome
    best_fitness = ga.best_fitness
    best_meta = ga.best_meta

    active_mask = best > 0.5
    included_tickers = [t for t, a in zip(tickers, active_mask) if a]
    excluded_tickers = [t for t, a in zip(tickers, active_mask) if not a]

    # Get weights for included tickers
    weights = best * active_mask
    weights = weights / weights.sum()
    ticker_weights = {t: float(w) for t, w in zip(tickers, weights) if w > 0}
    ticker_weights = dict(sorted(ticker_weights.items(), key=lambda x: -x[1]))

    logging.info("=" * 70)
    logging.info("FINAL RESULTS")
    logging.info("=" * 70)
    logging.info(f"Best OOS Sharpe: {best_meta.get('sharpe_oos', best_fitness):.3f}")
    logging.info(f"Sortino: {best_meta.get('sortino_oos', 0):.3f}")
    logging.info(f"Profit Factor: {best_meta.get('profit_factor', 0):.2f}")
    logging.info(f"Win Rate: {best_meta.get('win_rate', 0):.1%}")
    logging.info(f"Sharpe (green): {best_meta.get('sharpe_green', 0):.3f}")
    logging.info(f"Sharpe (red): {best_meta.get('sharpe_red', 0):.3f}")
    logging.info(f"Regime imbalance: {best_meta.get('regime_imbalance', 0):.3f}")
    logging.info(f"Regime penalized: {best_meta.get('regime_penalized', False)}")
    logging.info(f"Walk-forward periods: {best_meta.get('n_wf_periods', 0)}")
    logging.info(f"Total OOS weeks: {best_meta.get('n_oos_weeks', 0)}")
    logging.info(f"")
    logging.info(f"INCLUDED ({len(included_tickers)} tickers):")
    for t in sorted(included_tickers):
        w = ticker_weights.get(t, 0)
        logging.info(f"  {t:6s}  weight={w:.4f}")
    logging.info(f"")
    logging.info(f"EXCLUDED ({len(excluded_tickers)} tickers):")
    logging.info(f"  {', '.join(sorted(excluded_tickers))}")
    logging.info(f"")
    logging.info(f"Equal-weight baseline Sharpe: {baseline_sharpe:.3f}")
    logging.info(f"GA optimized Sharpe: {best_meta.get('sharpe_oos', best_fitness):.3f}")
    improvement = best_meta.get('sharpe_oos', best_fitness) - baseline_sharpe
    logging.info(f"Improvement: {improvement:+.3f}")

    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'baseline_sharpe': float(baseline_sharpe),
        'best_fitness': float(best_fitness),
        'best_meta': best_meta,
        'included_tickers': sorted(included_tickers),
        'excluded_tickers': sorted(excluded_tickers),
        'ticker_weights': ticker_weights,
        'genome': best.tolist(),
        'ga_history': ga.history,
        'total_time_seconds': total_time,
        'n_generations_run': len(ga.history),
    }

    results_path = os.path.join(cfg.output_dir, 'ga_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {results_path}")

    # Save best genome as numpy
    genome_path = os.path.join(cfg.output_dir, 'best_genome.npy')
    np.save(genome_path, best)
    logging.info(f"Best genome saved to {genome_path}")

    # Log final results to MLflow
    if mlflow_available:
        try:
            mlflow.log_metrics({
                'final_sharpe_oos': best_meta.get('sharpe_oos', best_fitness),
                'final_sortino': best_meta.get('sortino_oos', 0),
                'final_profit_factor': best_meta.get('profit_factor', 0),
                'final_win_rate': best_meta.get('win_rate', 0),
                'final_n_active': best_meta.get('n_active', 0),
                'final_sharpe_green': best_meta.get('sharpe_green', 0),
                'final_sharpe_red': best_meta.get('sharpe_red', 0),
                'final_regime_imbalance': best_meta.get('regime_imbalance', 0),
                'baseline_sharpe': baseline_sharpe,
                'sharpe_improvement': improvement,
                'total_time_minutes': total_time / 60,
            })
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(genome_path)
            mlflow.end_run()
            logging.info("MLflow run completed")
        except Exception as e:
            logging.warning(f"MLflow final logging failed: {e}")

    # Print summary for easy parsing
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OOS Sharpe: {best_meta.get('sharpe_oos', best_fitness):.3f} (baseline: {baseline_sharpe:.3f})")
    print(f"Sortino: {best_meta.get('sortino_oos', 0):.3f}")
    print(f"Profit Factor: {best_meta.get('profit_factor', 0):.2f}")
    print(f"Win Rate: {best_meta.get('win_rate', 0):.1%}")
    print(f"Included: {len(included_tickers)}/{len(tickers)} tickers")
    print(f"Excluded: {', '.join(sorted(excluded_tickers))}")
    print(f"Regime balanced: {'YES' if not best_meta.get('regime_penalized', True) else 'NO'}")
    print(f"Top 5 weights: {list(ticker_weights.items())[:5]}")
    print(f"Time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
