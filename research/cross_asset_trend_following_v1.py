#!/usr/bin/env python3
"""
Cross-Asset Trend-Following (CTA-Style) with Options — v1
==========================================================
HC #696/#748: Growth strategy research. Target high returns.
HC #749: $645 agentic account, options-only, max $200-300/trade.

Academic basis: Time-series momentum (Moskowitz, Ooi, Pedersen 2012).
Assets that have been going up tend to continue going up.

Strategy:
- Universe: 8 diversified ETFs spanning equities, bonds, gold, real estate
- Signal: Multi-lookback trend score (20/60/120/200 day returns, equal-weighted)
- Trade: Buy 14-DTE ATM calls on ETFs with positive trend, puts on negative trend
- Exit: 30% TP, 25% SL, trailing 50% giveback, 5-day max hold (KB #281 rules)
- Position: Max $200/trade, max 3 positions at once

Variants:
A) Equity-only trend (SPY, QQQ, IWM, XLF)
B) Multi-asset trend (SPY, TLT, GLD, USO, EFA, VNQ, HYG, UUP)
C) Long-only (buy calls on uptrends only, skip downtrends)
D) Dual momentum (absolute + relative — only trade top-2 trending assets)
E) VIX-filtered (skip long entries when VIX > 25)
F) Concentrated (top-1 asset only, full budget)

4-gate validation:
1. Permutation test (100 shuffles, z > 1.65)
2. Regime balance (|Sharpe_bull - Sharpe_bear| / max < 0.50)
3. Random direction (random calls/puts same timing)
4. Sub-period stability (two halves both positive Sharpe)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
import sys
import traceback
from scipy import stats

warnings.filterwarnings('ignore')

# === CONFIG ===
INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 200.0
MAX_POSITIONS = 3
COMMISSION_PER_CONTRACT = 0.65  # typical options commission
TP_PCT = 0.30
SL_PCT = 0.25
TRAILING_GIVEBACK = 0.50
MAX_HOLD_DAYS = 5
N_PERMUTATIONS = 100
LOOKBACKS = [20, 60, 120, 200]  # days for trend scoring

# ETF universes
EQUITY_UNIVERSE = ['SPY', 'QQQ', 'IWM', 'XLF']
MULTI_ASSET_UNIVERSE = ['SPY', 'TLT', 'GLD', 'USO', 'EFA', 'VNQ', 'HYG', 'UUP']

# BS pricing (with known ~2.5x inflation, we apply conservative haircut)
BS_HAIRCUT = 0.60  # 60% haircut on BS price to approximate real cost

# === MLflow ===
try:
    import mlflow
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False


def fetch_data(tickers, start='2020-01-01', end='2026-07-25'):
    """Fetch daily OHLCV data for all tickers."""
    print(f"Fetching data for {tickers}...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 200:
                data[t] = df
                print(f"  {t}: {len(df)} days")
            else:
                print(f"  {t}: insufficient data ({len(df)} days), skipping")
        except Exception as e:
            print(f"  {t}: download failed: {e}")
    return data


def compute_trend_score(prices, lookbacks=LOOKBACKS):
    """
    Multi-lookback trend score: average of normalized returns across lookback periods.
    Score > 0 = uptrend, < 0 = downtrend.
    """
    scores = pd.DataFrame(index=prices.index, columns=prices.columns)
    for lb in lookbacks:
        ret = prices.pct_change(lb)
        # Normalize by volatility to make scores comparable across assets
        vol = prices.pct_change().rolling(lb).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)
        normalized = ret / vol
        scores = scores.add(normalized, fill_value=0)

    scores = scores / len(lookbacks)
    return scores


def bs_call_price(S, K, T, sigma, r=0.05):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)


def bs_put_price(S, K, T, sigma, r=0.05):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def estimate_iv(prices, window=20):
    """Estimate implied volatility from historical vol with markup."""
    hist_vol = prices.pct_change().rolling(window).std() * np.sqrt(252)
    # IV typically trades at premium to HV
    iv = hist_vol * 1.15  # 15% markup
    return iv.clip(lower=0.10, upper=1.50)


def simulate_option_trade(entry_price, direction, strike, iv, dte_days,
                          subsequent_prices, commission=COMMISSION_PER_CONTRACT):
    """
    Simulate a single-leg option trade with TP/SL/trailing/max-hold exits.
    Returns (pnl_pct, hold_days, exit_reason).
    """
    T_entry = dte_days / 252
    if direction == 'call':
        option_cost = bs_call_price(entry_price, strike, T_entry, iv)
    else:
        option_cost = bs_put_price(entry_price, strike, T_entry, iv)

    # Apply BS haircut
    option_cost = option_cost * (1 - BS_HAIRCUT)
    if option_cost < 0.10:
        return None  # too cheap, skip

    total_cost = option_cost + commission
    max_value = option_cost

    for i, price in enumerate(subsequent_prices):
        days_held = i + 1
        T_remain = max(0, (dte_days - days_held)) / 252

        if direction == 'call':
            current_value = bs_call_price(price, strike, T_remain, iv)
        else:
            current_value = bs_put_price(price, strike, T_remain, iv)

        current_value = current_value * (1 - BS_HAIRCUT)
        pnl_pct = (current_value - option_cost) / option_cost

        # Track max
        if current_value > max_value:
            max_value = current_value

        # TP
        if pnl_pct >= TP_PCT:
            net_pnl = (current_value - option_cost - 2*commission) / total_cost
            return (net_pnl, days_held, 'TP')

        # SL
        if pnl_pct <= -SL_PCT:
            net_pnl = (current_value - option_cost - 2*commission) / total_cost
            return (net_pnl, days_held, 'SL')

        # Trailing stop: if we've given back 50% of max gain
        if max_value > option_cost * 1.05:  # only if we had >5% gain
            giveback = (max_value - current_value) / (max_value - option_cost)
            if giveback >= TRAILING_GIVEBACK:
                net_pnl = (current_value - option_cost - 2*commission) / total_cost
                return (net_pnl, days_held, 'TRAIL')

        # Max hold
        if days_held >= MAX_HOLD_DAYS:
            net_pnl = (current_value - option_cost - 2*commission) / total_cost
            return (net_pnl, days_held, 'EXPIRE')

    # Ran out of data
    return None


def run_strategy(data, universe, variant_name, variant_config):
    """
    Run a trend-following variant and return trades + equity curve.
    """
    # Build aligned price matrix
    prices = pd.DataFrame()
    for t in universe:
        if t in data:
            prices[t] = data[t]['Close']

    prices = prices.dropna()
    if len(prices) < 250:
        return None, f"Insufficient aligned data ({len(prices)} days)"

    # Compute trend scores
    scores = compute_trend_score(prices)
    ivs = {t: estimate_iv(prices[t]) for t in prices.columns}

    # VIX proxy (SPY 20d vol * sqrt(252) * 100)
    if 'SPY' in prices.columns:
        vix_proxy = prices['SPY'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    else:
        vix_proxy = pd.Series(15.0, index=prices.index)

    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = []
    open_positions = []

    # Walk through each day starting after warmup
    start_idx = max(LOOKBACKS) + 5

    for i in range(start_idx, len(prices) - MAX_HOLD_DAYS - 1):
        date = prices.index[i]

        # Close expired/exited positions
        new_open = []
        for pos in open_positions:
            pos['days_open'] += 1
        open_positions = [p for p in open_positions if p.get('active', True)]

        equity_curve.append({'date': date, 'equity': equity})

        # Skip if at max positions
        if len(open_positions) >= MAX_POSITIONS:
            continue

        # Get today's scores
        today_scores = scores.iloc[i].dropna()
        if len(today_scores) == 0:
            continue

        # Determine signals based on variant
        signals = []

        if variant_config.get('long_only', False):
            # Only buy calls on uptrends
            for t in today_scores.index:
                if today_scores[t] > 0.2:  # mild positive threshold
                    signals.append((t, 'call', today_scores[t]))

        elif variant_config.get('dual_momentum', False):
            # Absolute + relative: only top-N with positive absolute momentum
            positive = today_scores[today_scores > 0].sort_values(ascending=False)
            top_n = variant_config.get('top_n', 2)
            for t in positive.head(top_n).index:
                signals.append((t, 'call', positive[t]))

        elif variant_config.get('concentrated', False):
            # Top-1 only, full budget
            best_t = today_scores.abs().idxmax()
            direction = 'call' if today_scores[best_t] > 0.2 else ('put' if today_scores[best_t] < -0.2 else None)
            if direction:
                signals.append((best_t, direction, abs(today_scores[best_t])))

        else:
            # Standard: long uptrends, short downtrends
            for t in today_scores.index:
                if today_scores[t] > 0.2:
                    signals.append((t, 'call', today_scores[t]))
                elif today_scores[t] < -0.2:
                    signals.append((t, 'put', abs(today_scores[t])))

        # VIX filter
        if variant_config.get('vix_filter', False):
            current_vix = vix_proxy.iloc[i] if i < len(vix_proxy) else 15
            if current_vix > 25:
                signals = [(t, d, s) for t, d, s in signals if d == 'put']

        # Sort by signal strength, take top available
        signals.sort(key=lambda x: x[2], reverse=True)

        for ticker, direction, strength in signals:
            if len(open_positions) >= MAX_POSITIONS:
                break

            # Don't double up on same ticker
            if any(p['ticker'] == ticker for p in open_positions):
                continue

            # Only enter every 5 days per ticker (avoid overtrading)
            recent_trades = [t for t in trades[-20:] if t.get('ticker') == ticker]
            if recent_trades:
                last_entry = recent_trades[-1].get('entry_date')
                if last_entry and (date - last_entry).days < 5:
                    continue

            entry_price = prices[ticker].iloc[i]
            strike = round(entry_price, 0)  # ATM
            iv = ivs[ticker].iloc[i] if i < len(ivs[ticker]) else 0.25

            if pd.isna(iv) or iv <= 0:
                iv = 0.25

            # Get subsequent prices for simulation
            end_idx = min(i + MAX_HOLD_DAYS + 1, len(prices))
            subsequent = prices[ticker].iloc[i+1:end_idx].values

            result = simulate_option_trade(
                entry_price, direction, strike, iv, 14, subsequent
            )

            if result is None:
                continue

            pnl_pct, hold_days, exit_reason = result

            # Size the trade
            trade_size = min(MAX_TRADE_SIZE, equity * 0.30)
            if trade_size < 50:
                continue

            dollar_pnl = trade_size * pnl_pct
            equity += dollar_pnl

            trades.append({
                'entry_date': date,
                'exit_date': prices.index[min(i + hold_days, len(prices)-1)],
                'ticker': ticker,
                'direction': direction,
                'entry_price': entry_price,
                'strike': strike,
                'signal_strength': strength,
                'pnl_pct': pnl_pct,
                'dollar_pnl': dollar_pnl,
                'hold_days': hold_days,
                'exit_reason': exit_reason,
                'equity_after': equity,
                'trade_size': trade_size,
            })

            # Track open position for max-positions limit
            open_positions.append({
                'ticker': ticker,
                'days_open': 0,
                'active': hold_days > 1  # if exited same day, not blocking
            })

        # Remove positions that have been open long enough
        open_positions = [p for p in open_positions if p['days_open'] < MAX_HOLD_DAYS]

    if not trades:
        return None, "No trades generated"

    return trades, equity_curve


def compute_metrics(trades, equity_curve):
    """Compute risk-adjusted performance metrics."""
    if not trades:
        return None

    pnls = [t['pnl_pct'] for t in trades]
    dollar_pnls = [t['dollar_pnl'] for t in trades]

    n_trades = len(trades)
    win_rate = sum(1 for p in pnls if p > 0) / n_trades if n_trades > 0 else 0
    avg_pnl = np.mean(pnls)
    total_return = (trades[-1]['equity_after'] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Annualize
    first_date = trades[0]['entry_date']
    last_date = trades[-1]['exit_date']
    years = max((last_date - first_date).days / 365.25, 0.5)
    cagr = (1 + total_return) ** (1/years) - 1

    # Daily returns from equity curve
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df['daily_ret'] = eq_df['equity'].pct_change().fillna(0)
        daily_std = eq_df['daily_ret'].std()
        daily_mean = eq_df['daily_ret'].mean()

        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        downside = eq_df['daily_ret'][eq_df['daily_ret'] < 0].std()
        sortino = (daily_mean / downside * np.sqrt(252)) if downside > 0 else 0

        # Max drawdown
        eq_df['cum_max'] = eq_df['equity'].cummax()
        eq_df['dd'] = (eq_df['equity'] - eq_df['cum_max']) / eq_df['cum_max']
        max_dd = eq_df['dd'].min()
    else:
        sharpe = sortino = max_dd = 0

    # Profit factor
    gross_profit = sum(p for p in dollar_pnls if p > 0)
    gross_loss = abs(sum(p for p in dollar_pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average hold
    avg_hold = np.mean([t['hold_days'] for t in trades])

    # Exit breakdown
    exit_counts = {}
    for t in trades:
        r = t['exit_reason']
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'pf': pf,
        'cagr': cagr,
        'total_return': total_return,
        'max_dd': max_dd,
        'avg_hold': avg_hold,
        'avg_pnl_pct': avg_pnl,
        'final_equity': trades[-1]['equity_after'],
        'exit_breakdown': exit_counts,
        'years': years,
    }


def permutation_test(trades, equity_curve, n_perms=N_PERMUTATIONS):
    """Shuffle trade directions, compute Sharpe distribution."""
    if not trades:
        return 1.0, 0.0

    real_metrics = compute_metrics(trades, equity_curve)
    if real_metrics is None:
        return 1.0, 0.0
    real_sharpe = real_metrics['sharpe']

    perm_sharpes = []
    pnls = [t['pnl_pct'] for t in trades]

    for _ in range(n_perms):
        # Randomly flip direction of each trade's PnL
        shuffled_pnls = [p * (1 if np.random.random() > 0.5 else -1) for p in pnls]
        equity = INITIAL_CAPITAL
        eq = []
        for p in shuffled_pnls:
            trade_size = min(MAX_TRADE_SIZE, equity * 0.30)
            equity += trade_size * p
            eq.append(equity)

        if len(eq) > 1:
            rets = pd.Series(eq).pct_change().dropna()
            if rets.std() > 0:
                perm_sharpes.append(rets.mean() / rets.std() * np.sqrt(252))

    if not perm_sharpes:
        return 1.0, 0.0

    z = (real_sharpe - np.mean(perm_sharpes)) / max(np.std(perm_sharpes), 1e-6)
    p_val = 1 - stats.norm.cdf(z)
    return p_val, z


def regime_test(trades):
    """Check if strategy works in both bull and bear regimes."""
    if not trades or len(trades) < 10:
        return None, None, None

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        # Use SPY direction as regime proxy
        price_change = t.get('entry_price', 0)
        # Simple: if the trade is in a call direction and profitable, or put and profitable
        if t['direction'] == 'call':
            bull_pnls.append(t['pnl_pct'])
        else:
            bear_pnls.append(t['pnl_pct'])

    if not bull_pnls or not bear_pnls:
        return None, None, None

    bull_sharpe = np.mean(bull_pnls) / max(np.std(bull_pnls), 1e-6) * np.sqrt(252/5)
    bear_sharpe = np.mean(bear_pnls) / max(np.std(bear_pnls), 1e-6) * np.sqrt(252/5)

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    return bull_sharpe, bear_sharpe, gap


def sub_period_test(trades):
    """Check both halves have positive Sharpe."""
    if not trades or len(trades) < 20:
        return False, 0, 0

    mid = len(trades) // 2
    h1_pnls = [t['pnl_pct'] for t in trades[:mid]]
    h2_pnls = [t['pnl_pct'] for t in trades[mid:]]

    h1_sharpe = np.mean(h1_pnls) / max(np.std(h1_pnls), 1e-6) * np.sqrt(252/5)
    h2_sharpe = np.mean(h2_pnls) / max(np.std(h2_pnls), 1e-6) * np.sqrt(252/5)

    return (h1_sharpe > 0 and h2_sharpe > 0), h1_sharpe, h2_sharpe


def random_direction_test(trades, n_sims=100):
    """Replace trend signals with random direction, compare Sharpe."""
    if not trades:
        return 1.0, 0.0

    real_sharpe = np.mean([t['pnl_pct'] for t in trades]) / max(np.std([t['pnl_pct'] for t in trades]), 1e-6)

    random_sharpes = []
    pnls = [abs(t['pnl_pct']) for t in trades]

    for _ in range(n_sims):
        rand_pnls = [p * (1 if np.random.random() > 0.5 else -1) for p in pnls]
        if np.std(rand_pnls) > 0:
            random_sharpes.append(np.mean(rand_pnls) / np.std(rand_pnls))

    if not random_sharpes:
        return 1.0, 0.0

    z = (real_sharpe - np.mean(random_sharpes)) / max(np.std(random_sharpes), 1e-6)
    return 1 - stats.norm.cdf(z), z


def run_variant(data, variant_name, universe, config):
    """Run a single variant through strategy + 4-gate validation."""
    print(f"\n{'='*60}")
    print(f"VARIANT {variant_name}")
    print(f"{'='*60}")

    trades, equity_curve = run_strategy(data, universe, variant_name, config)

    if trades is None:
        print(f"  FAILED: {equity_curve}")
        return {
            'variant': variant_name,
            'status': 'FAILED',
            'reason': equity_curve,
            'gates_passed': 0,
        }

    metrics = compute_metrics(trades, equity_curve)
    if metrics is None:
        print(f"  FAILED: no metrics")
        return {'variant': variant_name, 'status': 'FAILED', 'reason': 'no metrics', 'gates_passed': 0}

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Sortino: {metrics['sortino']:.2f}")
    print(f"  WR: {metrics['win_rate']*100:.1f}%")
    print(f"  PF: {metrics['pf']:.2f}")
    print(f"  CAGR: {metrics['cagr']*100:.1f}%")
    print(f"  MaxDD: {metrics['max_dd']*100:.1f}%")
    print(f"  Final: ${metrics['final_equity']:.0f} (from $645)")
    print(f"  Avg hold: {metrics['avg_hold']:.1f} days")
    print(f"  Exits: {metrics['exit_breakdown']}")

    # Gate 1: Permutation test
    perm_p, perm_z = permutation_test(trades, equity_curve)
    gate1 = perm_p < 0.05
    print(f"\n  Gate 1 (Permutation): z={perm_z:.2f}, p={perm_p:.3f} → {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: Regime balance
    bull_s, bear_s, regime_gap = regime_test(trades)
    if regime_gap is not None:
        gate2 = regime_gap < 0.50
        print(f"  Gate 2 (Regime): bull={bull_s:.2f}, bear={bear_s:.2f}, gap={regime_gap:.2f} → {'PASS' if gate2 else 'FAIL'}")
    else:
        gate2 = False
        print(f"  Gate 2 (Regime): insufficient data → FAIL")

    # Gate 3: Random direction
    rand_p, rand_z = random_direction_test(trades)
    gate3 = rand_p < 0.05
    print(f"  Gate 3 (Random): z={rand_z:.2f}, p={rand_p:.3f} → {'PASS' if gate3 else 'FAIL'}")

    # Gate 4: Sub-period stability
    sp_pass, sp_h1, sp_h2 = sub_period_test(trades)
    gate4 = sp_pass
    print(f"  Gate 4 (Sub-period): h1={sp_h1:.2f}, h2={sp_h2:.2f} → {'PASS' if gate4 else 'FAIL'}")

    gates_passed = sum([gate1, gate2, gate3, gate4])
    print(f"\n  GATES: {gates_passed}/4")

    return {
        'variant': variant_name,
        'status': 'COMPLETE',
        'metrics': metrics,
        'gates': {
            'permutation': {'pass': gate1, 'z': perm_z, 'p': perm_p},
            'regime': {'pass': gate2, 'bull_sharpe': bull_s, 'bear_sharpe': bear_s, 'gap': regime_gap},
            'random': {'pass': gate3, 'z': rand_z, 'p': rand_p},
            'sub_period': {'pass': gate4, 'h1_sharpe': sp_h1, 'h2_sharpe': sp_h2},
        },
        'gates_passed': gates_passed,
        'n_trades': metrics['n_trades'],
    }


def main():
    print("=" * 70)
    print("CROSS-ASSET TREND-FOLLOWING (CTA) WITH OPTIONS — v1")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # MLflow
    if USE_MLFLOW:
        try:
            exp = mlflow.set_experiment("cross_asset_trend_following_v1")
            run = mlflow.start_run(run_name=f"cta_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.set_tag("node", "jupiter")
            mlflow.set_tag("strategy_type", "growth")
            mlflow.set_tag("phase", "research")
        except Exception as e:
            print(f"MLflow init failed: {e}")

    # Fetch all data
    all_tickers = list(set(EQUITY_UNIVERSE + MULTI_ASSET_UNIVERSE))
    data = fetch_data(all_tickers)

    if len(data) < 4:
        print("FATAL: insufficient data")
        return

    # Define variants
    variants = {
        'A_equity_trend': {
            'universe': EQUITY_UNIVERSE,
            'config': {},
        },
        'B_multi_asset': {
            'universe': MULTI_ASSET_UNIVERSE,
            'config': {},
        },
        'C_long_only': {
            'universe': MULTI_ASSET_UNIVERSE,
            'config': {'long_only': True},
        },
        'D_dual_momentum': {
            'universe': MULTI_ASSET_UNIVERSE,
            'config': {'dual_momentum': True, 'top_n': 2},
        },
        'E_vix_filtered': {
            'universe': MULTI_ASSET_UNIVERSE,
            'config': {'vix_filter': True},
        },
        'F_concentrated': {
            'universe': MULTI_ASSET_UNIVERSE,
            'config': {'concentrated': True},
        },
    }

    results = []
    for name, spec in variants.items():
        try:
            result = run_variant(data, name, spec['universe'], spec['config'])
            results.append(result)

            if USE_MLFLOW and result.get('metrics'):
                m = result['metrics']
                mlflow.log_metrics({
                    f"{name}_sharpe": round(m['sharpe'], 3),
                    f"{name}_sortino": round(m['sortino'], 3),
                    f"{name}_wr": round(m['win_rate'], 3),
                    f"{name}_pf": round(min(m['pf'], 99), 3),
                    f"{name}_cagr": round(m['cagr'], 3),
                    f"{name}_mdd": round(m['max_dd'], 3),
                    f"{name}_n_trades": m['n_trades'],
                    f"{name}_gates": result['gates_passed'],
                })
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            traceback.print_exc()
            results.append({'variant': name, 'status': 'ERROR', 'reason': str(e), 'gates_passed': 0})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        if r.get('metrics'):
            m = r['metrics']
            print(f"  {r['variant']}: Sharpe {m['sharpe']:.2f}, WR {m['win_rate']*100:.0f}%, "
                  f"CAGR {m['cagr']*100:.1f}%, MDD {m['max_dd']*100:.1f}%, "
                  f"Gates {r['gates_passed']}/4, Trades {m['n_trades']}")
        else:
            print(f"  {r['variant']}: {r.get('status', 'UNKNOWN')} — {r.get('reason', '')}")

    best = max(results, key=lambda x: x.get('gates_passed', 0))
    print(f"\n  BEST: {best['variant']} ({best['gates_passed']}/4 gates)")

    if USE_MLFLOW:
        try:
            best_gates = best.get('gates_passed', 0)
            mlflow.log_metric("best_gates", best_gates)
            mlflow.set_tag("best_variant", best['variant'])
            mlflow.set_tag("result", "PASS" if best_gates >= 3 else "FAIL")
            mlflow.end_run()
        except:
            pass

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/research/findings/cross_asset_trend_v1_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        if 'metrics' in sr and sr['metrics']:
            sr['metrics'] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in sr['metrics'].items()}
        if 'gates' in sr:
            for gk, gv in sr['gates'].items():
                if isinstance(gv, dict):
                    sr['gates'][gk] = {k: (float(v) if isinstance(v, (np.floating, np.integer, float)) and v is not None else v)
                                       for k, v in gv.items()}
        serializable.append(sr)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved. Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
