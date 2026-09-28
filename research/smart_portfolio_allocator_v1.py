#!/usr/bin/env python3
"""
Smart Portfolio Allocator v1 — Regime-Based Multi-Strategy Allocation
=====================================================================
Meta-strategy that decides DAILY which allocation across 5 sub-strategies
to use, based on market regime signals (all T-1, executed T+1).

Key insight: We have multiple decent strategies but treat them independently.
This combines them intelligently based on regime detection.

Sub-strategies (simple ETF/index proxies):
1. UPRO momentum (SPY > 200SMA → UPRO, else SHY)
2. Stat arb proxy (mean reversion on correlated ETF pairs)
3. Premium selling proxy (short vol: collect implied premium, pay drawdowns)
4. Tail risk buying (long VIX calls proxy when VIX cheap, sell when expensive)
5. Quality dividend (SCHD/VIG buy-and-hold)

Regime detection (all T-1 data only):
- Trending, Mean-reverting, High vol, Low vol, Crisis

Validation:
- Walk-forward: 252d lookback, daily rebalance (max 1/week)
- Data: 2010-2026 from yfinance
- Costs: 10 bps per rebalance
- Regime-agnostic test (HC #428)
- Permutation test: 50 perms, shuffle regime-to-allocation mapping (HC #718)
- Lag sensitivity test (HC #724)
- Compare to: SPY B&H, 60/40, equal-weight-all-strategies

Author: Claude (Head of Quant)
Date: 2026-07-21
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import sys
import time
import json
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

RESULTS_DIR = '/home/jupiter/Lvl3Quant/research/reports'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def download_with_retry(tickers, start, end, retries=3):
    """Download data with retry logic."""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)
            if data is not None and len(data) > 50:
                return data
            print(f"  Attempt {attempt+1}: got {len(data) if data is not None else 0} rows, retrying...")
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {tickers} after {retries} attempts")


def calc_metrics(returns, rf_annual=0.04, periods_per_year=252):
    """Calculate risk-adjusted metrics from a return series."""
    if len(returns) < 10 or returns.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
                'cagr': 0, 'max_dd': 0, 'total_ret': 0, 'n_periods': 0,
                'annual_vol': 0, 'calmar': 0}

    rf_per = rf_annual / periods_per_year
    excess = returns - rf_per

    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = excess.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

    downside = returns[returns < 0].std()
    sortino = excess.mean() / downside * np.sqrt(periods_per_year) if (downside is not None and downside > 0) else 0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = (returns > 0).mean()

    cum = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and cum.iloc[-1] > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    total_ret = cum.iloc[-1] - 1

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'total_ret': round(total_ret, 4),
        'annual_vol': round(ann_vol, 4),
        'calmar': round(calmar, 3),
        'n_periods': len(returns)
    }


def regime_split(returns, spy_returns):
    """Split returns by SPY regime: green (>0), red (<0), flat (~0)."""
    green = returns[spy_returns > 0.002]
    red = returns[spy_returns < -0.002]
    flat = returns[(spy_returns >= -0.002) & (spy_returns <= 0.002)]
    return {
        'green': calc_metrics(green) if len(green) > 10 else None,
        'red': calc_metrics(red) if len(red) > 10 else None,
        'flat': calc_metrics(flat) if len(flat) > 10 else None,
    }


def regime_agnostic_check(regime_results):
    """HC #428 R1: |Sharpe_green - Sharpe_red| / max(...) <= 0.50"""
    g = regime_results.get('green')
    r = regime_results.get('red')
    if g is None or r is None:
        return True, 0.0
    sg, sr = g['sharpe'], r['sharpe']
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return True, 0.0
    ratio = abs(sg - sr) / denom
    return ratio <= 0.50, round(ratio, 3)


def permutation_test_allocation(returns, regime_series, allocation_fn, n_perms=50):
    """
    Permutation test: shuffle the regime-to-allocation MAPPING.
    This tests whether the specific regime→allocation rules matter,
    or if any random allocation schedule would work equally well.
    """
    actual_sharpe = calc_metrics(returns)['sharpe']

    # Get unique regimes
    unique_regimes = regime_series.dropna().unique().tolist()
    n_regimes = len(unique_regimes)
    perm_sharpes = []

    for i in range(n_perms):
        # Shuffle the regime labels (break the regime→allocation link)
        shuffled_regimes = regime_series.copy()
        rng = np.random.RandomState(i + 1000)
        regime_map = dict(zip(unique_regimes, rng.permutation(unique_regimes)))
        shuffled_regimes = shuffled_regimes.map(regime_map)

        # Recompute returns with shuffled regime→allocation
        perm_returns = allocation_fn(shuffled_regimes)
        if perm_returns is not None and len(perm_returns) > 10:
            perm_sharpes.append(calc_metrics(perm_returns)['sharpe'])

    if len(perm_sharpes) == 0:
        return 1.0, 0, actual_sharpe

    perm_sharpes = np.array(perm_sharpes)
    p_value = (np.sum(perm_sharpes >= actual_sharpe) + 1) / (len(perm_sharpes) + 1)
    return round(p_value, 4), len(perm_sharpes), actual_sharpe


def lag_sensitivity_test(signal_series, returns, lags=[0, 1, 2, 3, 5]):
    """
    HC #724: Test sensitivity to signal lag.
    If strategy works at lag=0 but dies at lag=1, it's likely lookahead.
    We REQUIRE lag>=1 for production (T-1 signals, T+1 execution).
    """
    results = {}
    for lag in lags:
        lagged_signal = signal_series.shift(lag)
        aligned = pd.concat([lagged_signal, returns], axis=1).dropna()
        if len(aligned) < 50:
            results[f'lag_{lag}'] = {'sharpe': 0, 'n': 0}
            continue
        # The "returns" here are already computed; we just need to check
        # if the signal at different lags still correlates with returns
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        results[f'lag_{lag}'] = {
            'correlation': round(corr, 4) if not np.isnan(corr) else 0,
            'n': len(aligned)
        }
    return results


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_all_data():
    """Download all required data."""
    print("=" * 70)
    print("SMART PORTFOLIO ALLOCATOR v1")
    print("Regime-Based Multi-Strategy Allocation")
    print("=" * 70)
    print()

    start_date = '2009-01-01'  # Extra year for warmup
    end_date = '2026-07-21'

    # Core tickers
    core_tickers = ['SPY', 'SHY', 'TLT', 'VIX', 'UPRO', 'SCHD', 'VIG',
                    'XLK', 'XLF', 'XLE', 'XLV', 'XLU', 'GLD', 'IEF']

    print("Downloading core data...")
    core_data = download_with_retry(core_tickers, start_date, end_date)

    # Handle multi-level columns from yfinance
    if isinstance(core_data.columns, pd.MultiIndex):
        close = core_data['Close']
    else:
        close = core_data

    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")

    # Download VIX separately (^VIX)
    print("Downloading VIX index...")
    vix_data = download_with_retry('^VIX', start_date, end_date)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data['Close'].squeeze()
    else:
        vix = vix_data['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.name = 'VIX'

    # Stat arb pairs for mean reversion
    pair_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLU', 'GLD', 'IEF', 'TLT']
    # These are already in core_data

    print(f"  VIX data: {len(vix)} days")

    return close, vix


# ============================================================
# SUB-STRATEGY SIMULATORS
# ============================================================

def strategy_upro_momentum(spy_close, upro_close, shy_close):
    """
    Strategy 1: UPRO momentum
    SPY > 200-day SMA → UPRO (3x leveraged SPY), else → SHY (short treasury)
    All signals use T-1 data, execution at T+1 close.
    """
    sma200 = spy_close.rolling(200).mean()

    # Signal: SPY above 200 SMA (T-1)
    signal = (spy_close > sma200).astype(float).shift(1)  # T-1 signal

    upro_ret = upro_close.pct_change()
    shy_ret = shy_close.pct_change()

    # T+1 execution: signal from T-1, applied to T+1 returns
    returns = signal.shift(1) * upro_ret + (1 - signal.shift(1)) * shy_ret
    returns = returns.dropna()

    return returns


def strategy_stat_arb(close_df):
    """
    Strategy 2: Stat arb proxy — mean reversion on sector ETF pairs.
    Uses z-score of price ratio, mean-reverts when |z| > 2.
    Pairs: XLK/XLF, XLE/XLV, GLD/IEF
    """
    pairs = [('XLK', 'XLF'), ('XLE', 'XLV'), ('GLD', 'IEF')]
    all_returns = []

    for t1, t2 in pairs:
        if t1 not in close_df.columns or t2 not in close_df.columns:
            continue

        ratio = close_df[t1] / close_df[t2]
        ratio_ma = ratio.rolling(60).mean()
        ratio_std = ratio.rolling(60).std()
        z = (ratio - ratio_ma) / ratio_std

        # T-1 signal
        z_signal = z.shift(1)

        ret1 = close_df[t1].pct_change()
        ret2 = close_df[t2].pct_change()

        # Mean reversion: short t1/long t2 when z > 2, long t1/short t2 when z < -2
        # T+1 execution
        pair_ret = pd.Series(0.0, index=close_df.index)

        # Go short spread when z > 2 (expect mean reversion down)
        short_spread = z_signal > 2.0
        pair_ret[short_spread] = (-ret1[short_spread] + ret2[short_spread]) / 2

        # Go long spread when z < -2 (expect mean reversion up)
        long_spread = z_signal < -2.0
        pair_ret[long_spread] = (ret1[long_spread] - ret2[long_spread]) / 2

        all_returns.append(pair_ret)

    if not all_returns:
        return pd.Series(0.0, index=close_df.index)

    # Equal weight across pairs
    combined = pd.concat(all_returns, axis=1).mean(axis=1)
    return combined


def strategy_premium_selling(spy_close, vix_series):
    """
    Strategy 3: Premium selling proxy.
    Collect weekly implied premium (VIX/100 * SPY * sqrt(5/252)) as income,
    but pay out when realized vol exceeds implied (drawdown events).
    Uses T-1 VIX for decision, T+1 execution.
    """
    spy_ret = spy_close.pct_change()

    # Align VIX to SPY index
    vix_aligned = vix_series.reindex(spy_close.index).ffill()

    # Realized vol (20-day trailing)
    realized_vol = spy_ret.rolling(20).std() * np.sqrt(252)

    # Weekly premium collected (simplified: daily accrual = annual premium / 252)
    # Premium = VIX level / 100 (annualized), we collect daily
    daily_premium = (vix_aligned / 100) / 252  # T-1 VIX

    # Cost: when SPY drops > 2% in a day, we "pay out" proportionally
    # This simulates being short puts that get exercised
    payout = pd.Series(0.0, index=spy_close.index)

    # Large down moves trigger losses (short put assignment proxy)
    big_down = spy_ret < -0.02
    payout[big_down] = spy_ret[big_down] * 2  # 2x leverage on big downs (short put exposure)

    # T-1 signal: only sell premium when VIX > realized (positive VRP)
    vrp_positive = (vix_aligned.shift(1) > realized_vol.shift(1))

    # T+1 returns: premium income + payout losses
    returns = pd.Series(0.0, index=spy_close.index)
    returns[vrp_positive] = daily_premium.shift(1)[vrp_positive] + payout[vrp_positive]

    # When VRP negative (realized > implied), stay flat
    returns = returns.dropna()
    return returns


def strategy_tail_risk(spy_close, vix_series):
    """
    Strategy 4: Tail risk buying proxy.
    Buy VIX calls (proxy) when VIX < 15, sell when VIX > 25.
    Sized small (2% of portfolio per position).
    Uses T-1 VIX for decision, T+1 execution.
    """
    # Align VIX to SPY index
    vix_aligned = vix_series.reindex(spy_close.index).ffill()
    vix_ret = vix_aligned.pct_change()

    # Position: long tail risk when VIX is cheap (T-1 signal)
    vix_prev = vix_aligned.shift(1)

    # Position sizing: small allocation, scaled by how cheap VIX is
    position = pd.Series(0.0, index=spy_close.index)

    # Buy tail risk when VIX < 15 (cheap protection)
    cheap = vix_prev < 15
    position[cheap] = 0.02  # 2% allocation to tail protection

    # Medium allocation when VIX 15-20
    medium = (vix_prev >= 15) & (vix_prev < 20)
    position[medium] = 0.01

    # Sell/take profit when VIX > 25 (tail event happening)
    # Model as: we bought at cheap levels, VIX spike gives us a return
    # Proxy: long VIX futures return * position size

    # Returns come from VIX moves while positioned
    returns = position.shift(1) * vix_ret  # T+1 execution

    # Cap gains at 50% daily (VIX can spike but options have convexity limits)
    returns = returns.clip(-0.05, 0.50)

    return returns.dropna()


def strategy_quality_dividend(schd_close, vig_close):
    """
    Strategy 5: Quality dividend — SCHD/VIG buy-and-hold.
    50/50 split. Simple and steady.
    """
    schd_ret = schd_close.pct_change()
    vig_ret = vig_close.pct_change()

    # Equal weight
    returns = (schd_ret + vig_ret) / 2
    return returns.dropna()


# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regime(spy_close, vix_series, idx):
    """
    Detect market regime using T-1 data only.
    Returns one of: 'trending', 'mean_reverting', 'high_vol', 'low_vol', 'crisis'

    Priority order (crisis > high_vol > trending > mean_reverting > low_vol)
    """
    lookback = 252  # 1 year lookback for SMAs

    if idx < lookback + 1:
        return 'unknown'

    # All data up to T-1 (strict anti-lookahead)
    spy_t1 = spy_close.iloc[:idx]
    vix_t1 = vix_series.iloc[:idx] if idx <= len(vix_series) else vix_series

    current_spy = spy_t1.iloc[-1]
    current_vix = vix_t1.iloc[-1] if len(vix_t1) > 0 else 20

    # SMAs (T-1)
    sma50 = spy_t1.iloc[-50:].mean() if len(spy_t1) >= 50 else current_spy
    sma200 = spy_t1.iloc[-200:].mean() if len(spy_t1) >= 200 else current_spy

    # Realized vol (20-day, annualized)
    recent_rets = spy_t1.pct_change().iloc[-20:]
    realized_vol = recent_rets.std() * np.sqrt(252) * 100  # as percentage

    # ADX proxy: absolute directional movement over 20 days
    price_20d_ago = spy_t1.iloc[-20] if len(spy_t1) >= 20 else current_spy
    directional_move = abs(current_spy / price_20d_ago - 1) * 100

    # RSI (14-day)
    deltas = spy_t1.diff().iloc[-14:]
    gain = deltas[deltas > 0].sum() / 14
    loss = abs(deltas[deltas < 0].sum()) / 14
    rs = gain / loss if loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # Crisis: VIX > 30 AND SPY below 200 SMA
    if current_vix > 30 and current_spy < sma200:
        return 'crisis'

    # High vol: VIX > 25 or realized vol > 20%
    if current_vix > 25 or realized_vol > 20:
        return 'high_vol'

    # Trending: SPY above both 50 and 200 SMA, strong directional movement
    if current_spy > sma50 and current_spy > sma200 and directional_move > 3:
        return 'trending'

    # Low vol: VIX < 15, realized vol < 12%
    if current_vix < 15 and realized_vol < 12:
        return 'low_vol'

    # Default: mean-reverting (range-bound)
    return 'mean_reverting'


# ============================================================
# ALLOCATION RULES
# ============================================================

ALLOCATION_RULES = {
    'trending':       {'upro': 0.40, 'quality_div': 0.20, 'premium': 0.20, 'cash': 0.20, 'stat_arb': 0.00, 'tail_risk': 0.00},
    'mean_reverting': {'upro': 0.00, 'quality_div': 0.30, 'premium': 0.30, 'cash': 0.00, 'stat_arb': 0.40, 'tail_risk': 0.00},
    'high_vol':       {'upro': 0.00, 'quality_div': 0.20, 'premium': 0.00, 'cash': 0.30, 'stat_arb': 0.10, 'tail_risk': 0.40},
    'crisis':         {'upro': 0.00, 'quality_div': 0.20, 'premium': 0.00, 'cash': 0.30, 'stat_arb': 0.00, 'tail_risk': 0.50},
    'low_vol':        {'upro': 0.30, 'quality_div': 0.20, 'premium': 0.40, 'cash': 0.00, 'stat_arb': 0.10, 'tail_risk': 0.00},
    'unknown':        {'upro': 0.00, 'quality_div': 0.50, 'premium': 0.00, 'cash': 0.50, 'stat_arb': 0.00, 'tail_risk': 0.00},
}

REBALANCE_COST_BPS = 10  # 10 bps per rebalance
MIN_REBALANCE_DAYS = 5   # Max 1 rebalance per week


# ============================================================
# MAIN PORTFOLIO ALLOCATOR
# ============================================================

def run_allocator(close_df, vix_series, regime_override=None):
    """
    Run the smart portfolio allocator.
    regime_override: if provided, a Series of regime labels (for permutation test)
    Returns: daily portfolio returns, regime series
    """
    spy_close = close_df['SPY']

    # Compute all sub-strategy returns
    print("  Computing sub-strategy returns...")

    # Handle UPRO (started 2009, may have gaps)
    if 'UPRO' in close_df.columns:
        upro_close = close_df['UPRO']
    else:
        # Proxy: 3x SPY daily return
        upro_close = spy_close.copy()

    shy_close = close_df['SHY'] if 'SHY' in close_df.columns else spy_close * 0 + 1

    strat_returns = {}
    strat_returns['upro'] = strategy_upro_momentum(spy_close, upro_close, shy_close)
    strat_returns['stat_arb'] = strategy_stat_arb(close_df)
    strat_returns['premium'] = strategy_premium_selling(spy_close, vix_series)
    strat_returns['tail_risk'] = strategy_tail_risk(spy_close, vix_series)

    schd = close_df['SCHD'] if 'SCHD' in close_df.columns else close_df['VIG'] if 'VIG' in close_df.columns else spy_close
    vig = close_df['VIG'] if 'VIG' in close_df.columns else spy_close
    strat_returns['quality_div'] = strategy_quality_dividend(schd, vig)

    # Cash return (proxy: SHY return or risk-free)
    strat_returns['cash'] = shy_close.pct_change().fillna(0)

    # Align all returns to common index
    common_idx = spy_close.index
    for k in strat_returns:
        strat_returns[k] = strat_returns[k].reindex(common_idx).fillna(0)

    # Detect regimes for each day
    print("  Detecting regimes...")
    if regime_override is not None:
        regime_series = regime_override.reindex(common_idx).fillna('unknown')
    else:
        regimes = []
        for i in range(len(common_idx)):
            # Use T-1 data only
            vix_aligned = vix_series.reindex(common_idx)
            r = detect_regime(spy_close, vix_aligned, i)
            regimes.append(r)
        regime_series = pd.Series(regimes, index=common_idx)

    # Walk-forward allocation
    print("  Running walk-forward allocation...")
    warmup = 252  # 1 year warmup
    portfolio_returns = pd.Series(0.0, index=common_idx)
    current_alloc = ALLOCATION_RULES['unknown']
    last_rebalance_idx = -999

    for i in range(warmup, len(common_idx)):
        regime = regime_series.iloc[i]
        target_alloc = ALLOCATION_RULES.get(regime, ALLOCATION_RULES['unknown'])

        # Check if we should rebalance (max 1 per week)
        days_since_rebal = i - last_rebalance_idx
        alloc_changed = any(target_alloc[k] != current_alloc.get(k, 0) for k in target_alloc)

        if alloc_changed and days_since_rebal >= MIN_REBALANCE_DAYS:
            # Apply rebalance cost
            turnover = sum(abs(target_alloc[k] - current_alloc.get(k, 0)) for k in target_alloc) / 2
            rebal_cost = turnover * REBALANCE_COST_BPS / 10000
            current_alloc = target_alloc
            last_rebalance_idx = i
        else:
            rebal_cost = 0.0

        # Compute weighted return
        day_return = 0.0
        for strat_name, weight in current_alloc.items():
            if weight > 0 and strat_name in strat_returns:
                day_return += weight * strat_returns[strat_name].iloc[i]

        portfolio_returns.iloc[i] = day_return - rebal_cost

    # Trim to OOS period (after warmup)
    oos_returns = portfolio_returns.iloc[warmup:]
    oos_regimes = regime_series.iloc[warmup:]

    return oos_returns, oos_regimes, strat_returns, warmup


def make_permuted_allocator(close_df, vix_series, strat_returns, warmup, common_idx):
    """Create a function that takes shuffled regimes and returns portfolio returns."""
    def allocate_with_regimes(regime_series):
        portfolio_returns = pd.Series(0.0, index=common_idx)
        current_alloc = ALLOCATION_RULES['unknown']
        last_rebalance_idx = -999

        for i in range(warmup, len(common_idx)):
            regime = regime_series.iloc[i] if i < len(regime_series) else 'unknown'
            target_alloc = ALLOCATION_RULES.get(regime, ALLOCATION_RULES['unknown'])

            days_since_rebal = i - last_rebalance_idx
            alloc_changed = any(target_alloc[k] != current_alloc.get(k, 0) for k in target_alloc)

            if alloc_changed and days_since_rebal >= MIN_REBALANCE_DAYS:
                turnover = sum(abs(target_alloc[k] - current_alloc.get(k, 0)) for k in target_alloc) / 2
                rebal_cost = turnover * REBALANCE_COST_BPS / 10000
                current_alloc = target_alloc
                last_rebalance_idx = i
            else:
                rebal_cost = 0.0

            day_return = 0.0
            for strat_name, weight in current_alloc.items():
                if weight > 0 and strat_name in strat_returns:
                    day_return += weight * strat_returns[strat_name].iloc[i]

            portfolio_returns.iloc[i] = day_return - rebal_cost

        return portfolio_returns.iloc[warmup:]

    return allocate_with_regimes


# ============================================================
# BENCHMARK STRATEGIES
# ============================================================

def benchmark_spy_bh(spy_close, warmup):
    """SPY buy and hold."""
    returns = spy_close.pct_change().iloc[warmup:]
    return returns.dropna()


def benchmark_60_40(spy_close, tlt_close, warmup):
    """60/40 stocks/bonds."""
    spy_ret = spy_close.pct_change()
    tlt_ret = tlt_close.pct_change()
    returns = (0.6 * spy_ret + 0.4 * tlt_ret).iloc[warmup:]
    return returns.dropna()


def benchmark_equal_weight(strat_returns, warmup):
    """Equal weight all strategies (no regime detection)."""
    strat_names = ['upro', 'stat_arb', 'premium', 'tail_risk', 'quality_div']
    n = len(strat_names)
    combined = pd.Series(0.0, index=strat_returns['upro'].index)
    for name in strat_names:
        if name in strat_returns:
            combined += strat_returns[name] / n
    return combined.iloc[warmup:]


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    start_time = time.time()

    # Download data
    close_df, vix_series = download_all_data()

    # Check data availability
    print("\nData availability:")
    for col in close_df.columns:
        valid = close_df[col].dropna()
        if len(valid) > 0:
            print(f"  {col}: {valid.index[0].date()} to {valid.index[-1].date()} ({len(valid)} days)")

    vix_valid = vix_series.dropna()
    print(f"  VIX: {vix_valid.index[0].date()} to {vix_valid.index[-1].date()} ({len(vix_valid)} days)")

    # Fill SCHD/VIG/UPRO gaps (these ETFs started later)
    for col in ['SCHD', 'VIG', 'UPRO']:
        if col in close_df.columns:
            close_df[col] = close_df[col].ffill().bfill()

    # Run main allocator
    print("\n" + "=" * 70)
    print("RUNNING SMART PORTFOLIO ALLOCATOR")
    print("=" * 70)

    oos_returns, oos_regimes, strat_returns, warmup = run_allocator(close_df, vix_series)

    # Trim to 2010+ (after warmup)
    oos_start = oos_returns.index[0]
    print(f"\nOOS period: {oos_start.date()} to {oos_returns.index[-1].date()}")
    print(f"OOS periods: {len(oos_returns)} days")

    # Regime distribution
    print("\n--- REGIME DISTRIBUTION ---")
    regime_counts = oos_regimes.value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(oos_regimes) * 100
        print(f"  {regime:20s}: {count:5d} days ({pct:5.1f}%)")

    # ============================================================
    # RESULTS: Smart Allocator
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS: SMART PORTFOLIO ALLOCATOR")
    print("=" * 70)

    allocator_metrics = calc_metrics(oos_returns)
    print("\n--- Overall Performance ---")
    for k, v in allocator_metrics.items():
        print(f"  {k:15s}: {v}")

    # Regime-agnostic test
    spy_returns_oos = close_df['SPY'].pct_change().reindex(oos_returns.index).fillna(0)
    regime_results = regime_split(oos_returns, spy_returns_oos)
    passed, ratio = regime_agnostic_check(regime_results)

    print(f"\n--- Regime-Agnostic Test (HC #428) ---")
    print(f"  Regime disparity ratio: {ratio}")
    print(f"  PASSED: {passed}")
    if regime_results['green']:
        print(f"  Green day Sharpe: {regime_results['green']['sharpe']}")
    if regime_results['red']:
        print(f"  Red day Sharpe:   {regime_results['red']['sharpe']}")
    if regime_results['flat']:
        print(f"  Flat day Sharpe:  {regime_results['flat']['sharpe']}")

    # ============================================================
    # INDIVIDUAL SUB-STRATEGIES
    # ============================================================
    print("\n" + "=" * 70)
    print("INDIVIDUAL SUB-STRATEGY PERFORMANCE (standalone)")
    print("=" * 70)

    sub_strat_metrics = {}
    for name in ['upro', 'stat_arb', 'premium', 'tail_risk', 'quality_div']:
        if name in strat_returns:
            sr = strat_returns[name].iloc[warmup:]
            m = calc_metrics(sr)
            sub_strat_metrics[name] = m
            print(f"\n  {name:15s}: Sharpe={m['sharpe']:+.3f}  Sortino={m['sortino']:+.3f}  "
                  f"PF={m['pf']:.2f}  WR={m['wr']:.3f}  CAGR={m['cagr']:.3%}  MaxDD={m['max_dd']:.2%}")

    # ============================================================
    # BENCHMARKS
    # ============================================================
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    spy_bh = benchmark_spy_bh(close_df['SPY'], warmup)
    spy_metrics = calc_metrics(spy_bh)

    tlt = close_df['TLT'] if 'TLT' in close_df.columns else close_df['SHY']
    bm_60_40 = benchmark_60_40(close_df['SPY'], tlt, warmup)
    bm_60_40_metrics = calc_metrics(bm_60_40)

    eq_weight = benchmark_equal_weight(strat_returns, warmup)
    eq_weight_metrics = calc_metrics(eq_weight)

    print(f"\n  {'Strategy':30s} {'Sharpe':>8s} {'Sortino':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s}")
    print(f"  {'-'*72}")

    results_table = {
        'Smart Allocator': allocator_metrics,
        'SPY Buy & Hold': spy_metrics,
        '60/40': bm_60_40_metrics,
        'Equal Weight All': eq_weight_metrics,
    }

    for name, m in results_table.items():
        print(f"  {name:30s} {m['sharpe']:+8.3f} {m['sortino']:+8.3f} "
              f"{m['cagr']:7.2%} {m['max_dd']:7.2%} {m['calmar']:+8.3f}")

    # ============================================================
    # PERMUTATION TEST (HC #718)
    # ============================================================
    print("\n" + "=" * 70)
    print("PERMUTATION TEST (HC #718)")
    print("Shuffling regime→allocation mapping, 50 permutations")
    print("=" * 70)

    common_idx = close_df['SPY'].index
    alloc_fn = make_permuted_allocator(close_df, vix_series, strat_returns, warmup, common_idx)

    # Full regime series (not just OOS)
    full_regimes = pd.Series('unknown', index=common_idx)
    vix_aligned = vix_series.reindex(common_idx)
    for i in range(len(common_idx)):
        full_regimes.iloc[i] = detect_regime(close_df['SPY'], vix_aligned, i)

    p_value, n_perms, actual_sharpe = permutation_test_allocation(
        oos_returns, full_regimes, alloc_fn, n_perms=50
    )

    print(f"\n  Actual Sharpe:    {actual_sharpe:.3f}")
    print(f"  Permutations run: {n_perms}")
    print(f"  P-value:          {p_value:.4f}")
    print(f"  SIGNIFICANT (p<0.10): {p_value < 0.10}")

    # ============================================================
    # LAG SENSITIVITY TEST (HC #724)
    # ============================================================
    print("\n" + "=" * 70)
    print("LAG SENSITIVITY TEST (HC #724)")
    print("Testing if strategy degrades with additional signal lag")
    print("=" * 70)

    # Encode regime as numeric for correlation test
    regime_numeric = oos_regimes.map({
        'trending': 1, 'mean_reverting': 0, 'high_vol': -1,
        'low_vol': 0.5, 'crisis': -2, 'unknown': 0
    }).astype(float)

    lag_results = lag_sensitivity_test(regime_numeric, oos_returns, lags=[0, 1, 2, 3, 5, 10])
    for lag_name, lr in lag_results.items():
        print(f"  {lag_name}: {lr}")

    # Also test: re-run allocator with +1 day extra lag on regime detection
    print("\n  Testing allocator with +1 extra day lag on regime signals...")
    lagged_regimes = full_regimes.shift(1).fillna('unknown')
    oos_ret_lagged, _, _, _ = run_allocator(close_df, vix_series, regime_override=lagged_regimes)
    lagged_metrics = calc_metrics(oos_ret_lagged)
    print(f"  Original Sharpe: {allocator_metrics['sharpe']:.3f}")
    print(f"  +1 lag Sharpe:   {lagged_metrics['sharpe']:.3f}")
    sharpe_decay = abs(allocator_metrics['sharpe'] - lagged_metrics['sharpe'])
    print(f"  Decay:           {sharpe_decay:.3f}")
    print(f"  STABLE (decay < 0.20): {sharpe_decay < 0.20}")

    # ============================================================
    # YEARLY BREAKDOWN
    # ============================================================
    print("\n" + "=" * 70)
    print("YEARLY BREAKDOWN")
    print("=" * 70)

    oos_df = pd.DataFrame({
        'return': oos_returns,
        'regime': oos_regimes,
        'year': oos_returns.index.year
    })

    print(f"\n  {'Year':>6s} {'Return':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Dominant Regime':>20s}")
    print(f"  {'-'*54}")

    yearly_data = []
    for year, grp in oos_df.groupby('year'):
        yr_ret = grp['return']
        yr_metrics = calc_metrics(yr_ret)
        dom_regime = grp['regime'].mode().iloc[0] if len(grp) > 0 else 'unknown'
        yearly_data.append({
            'year': year,
            'return': yr_metrics['total_ret'],
            'sharpe': yr_metrics['sharpe'],
            'max_dd': yr_metrics['max_dd'],
            'dominant_regime': dom_regime
        })
        print(f"  {year:>6d} {yr_metrics['total_ret']:+7.2%} {yr_metrics['sharpe']:+8.3f} "
              f"{yr_metrics['max_dd']:+7.2%} {dom_regime:>20s}")

    # ============================================================
    # REGIME-STRATIFIED PERFORMANCE
    # ============================================================
    print("\n" + "=" * 70)
    print("REGIME-STRATIFIED ALLOCATOR PERFORMANCE")
    print("=" * 70)

    for regime in ['trending', 'mean_reverting', 'high_vol', 'low_vol', 'crisis']:
        mask = oos_regimes == regime
        if mask.sum() > 10:
            regime_ret = oos_returns[mask]
            rm = calc_metrics(regime_ret)
            print(f"\n  {regime:20s} ({mask.sum():4d} days):")
            print(f"    Sharpe={rm['sharpe']:+.3f}  Sortino={rm['sortino']:+.3f}  "
                  f"PF={rm['pf']:.2f}  CAGR={rm['cagr']:.3%}  MaxDD={rm['max_dd']:.2%}")

    # ============================================================
    # DRAWDOWN ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("DRAWDOWN ANALYSIS")
    print("=" * 70)

    cum = (1 + oos_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Find top 5 drawdowns
    in_dd = False
    dd_periods = []
    dd_start = None

    for i in range(len(dd)):
        if dd.iloc[i] < -0.02 and not in_dd:
            dd_start = dd.index[i]
            in_dd = True
        elif dd.iloc[i] >= -0.005 and in_dd:
            dd_end = dd.index[i]
            dd_depth = dd.loc[dd_start:dd_end].min()
            dd_periods.append((dd_start, dd_end, dd_depth))
            in_dd = False

    if in_dd and dd_start is not None:
        dd_periods.append((dd_start, dd.index[-1], dd.loc[dd_start:].min()))

    dd_periods.sort(key=lambda x: x[2])
    print(f"\n  Top 5 drawdowns:")
    for j, (ds, de, depth) in enumerate(dd_periods[:5]):
        duration = (de - ds).days
        dominant_regime = oos_regimes.loc[ds:de].mode().iloc[0] if len(oos_regimes.loc[ds:de]) > 0 else '?'
        print(f"    {j+1}. {ds.date()} to {de.date()} | Depth: {depth:.2%} | "
              f"Duration: {duration}d | Regime: {dominant_regime}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Smart Portfolio Allocator v1")
    print(f"  OOS Period: {oos_start.date()} to {oos_returns.index[-1].date()} ({len(oos_returns)} days)")
    print(f"  Sharpe: {allocator_metrics['sharpe']}")
    print(f"  Sortino: {allocator_metrics['sortino']}")
    print(f"  CAGR: {allocator_metrics['cagr']:.2%}")
    print(f"  Max DD: {allocator_metrics['max_dd']:.2%}")
    print(f"  Calmar: {allocator_metrics['calmar']:.3f}")
    print(f"  Profit Factor: {allocator_metrics['pf']}")
    print(f"  Win Rate: {allocator_metrics['wr']:.2%}")
    print(f"  Regime-Agnostic: {'PASS' if passed else 'FAIL'} (ratio={ratio})")
    print(f"  Permutation p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.10 else 'NOT significant'})")
    print(f"  Lag Stability: {'STABLE' if sharpe_decay < 0.20 else 'UNSTABLE'} (decay={sharpe_decay:.3f})")
    print(f"  Runtime: {elapsed:.1f}s")

    # Compare to v1 strategies
    print(f"\n  --- vs v1 Creative Income Results ---")
    print(f"  v1 VIX CSP:        Sharpe -0.27 (bad)")
    print(f"  v1 Div Capture:    Sharpe 1.52 but -52% MaxDD (bad risk)")
    print(f"  v1 Carry Rotation: Sharpe 0.23 (meh)")
    print(f"  THIS ALLOCATOR:    Sharpe {allocator_metrics['sharpe']:+.3f}, MaxDD {allocator_metrics['max_dd']:.2%}")

    print(f"\n  --- vs Existing Validated Strategies ---")
    print(f"  Stat Arb:          Sharpe 0.81")
    print(f"  UPRO+200SMA:       Sharpe 0.82")
    print(f"  Wheel:             Sharpe 0.365")
    print(f"  THIS ALLOCATOR:    Sharpe {allocator_metrics['sharpe']:+.3f}")

    # Save results
    results = {
        'experiment': 'smart_portfolio_allocator_v1',
        'date': datetime.now().isoformat(),
        'oos_start': str(oos_start.date()),
        'oos_end': str(oos_returns.index[-1].date()),
        'n_oos_days': len(oos_returns),
        'allocator_metrics': allocator_metrics,
        'benchmark_spy': spy_metrics,
        'benchmark_60_40': bm_60_40_metrics,
        'benchmark_equal_weight': eq_weight_metrics,
        'sub_strategy_metrics': sub_strat_metrics,
        'regime_distribution': {k: int(v) for k, v in regime_counts.items()},
        'regime_agnostic_pass': passed,
        'regime_agnostic_ratio': ratio,
        'permutation_p_value': p_value,
        'lag_sensitivity_decay': sharpe_decay,
        'lag_stable': sharpe_decay < 0.20,
        'yearly_data': yearly_data,
        'runtime_seconds': round(elapsed, 1),
    }

    results_path = os.path.join(RESULTS_DIR, 'smart_portfolio_allocator_v1_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = main()
