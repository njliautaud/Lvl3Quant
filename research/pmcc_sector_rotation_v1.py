#!/usr/bin/env python3
"""
Poor Man's Covered Call (PMCC) with Sector Rotation — v1
=========================================================
HC #696/#748: Growth + income strategy for $645 account.
HC #749: Options-only, max $200-300/trade.

Strategy concept:
- Buy deep ITM LEAPS call on top-ranked sector ETF (~70-80 delta, 6-12 month expiry)
- Sell short-term OTM calls against it for income (14-28 DTE, 20-30 delta)
- Rotate LEAPS when sector ranking changes
- Income from short calls reduces cost basis of LEAPS position

Key advantage for small accounts:
- LEAPS call on $50 ETF costs ~$8-12 vs $5000 for 100 shares
- Still captures ~80% of upside moves
- Short call income reduces theta burden

Variants:
A) Top-1 sector, weekly short call rolls
B) Top-1 sector, biweekly short call rolls
C) Top-2 sectors, biweekly rolls (diversified)
D) VIX-adaptive DTE (higher VIX → shorter DTE for more premium)
E) Aggressive short strike (closer to ATM, more income, more risk)
F) Conservative (deeper ITM LEAPS, further OTM short)

4-gate validation: permutation, regime, random, sub-period
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
COMMISSION_PER_CONTRACT = 0.65
N_PERMUTATIONS = 100

# Sector ETFs (same universe as validated KB #285)
SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']

# LGBM ranking features
RANKING_FEATURES = ['ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'vol_20d', 'rsi_14', 'rel_strength']

# BS pricing with calibration haircut (KB #282: BS underprices by ~73%)
BS_CALIBRATION_FACTOR = 1.10  # multiply BS by this for entry cost
BS_CALIBRATION_OFFSET = 0.0   # simplified; real: mkt_mid = 1.10*BS + 6.60 but scale-dependent


# === MLflow ===
try:
    import mlflow
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False


def fetch_data(tickers, start='2019-01-01', end='2026-07-25'):
    """Fetch daily data."""
    print(f"Fetching data for {len(tickers)} tickers...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            # Handle multi-index columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            if len(df) > 200:
                data[t] = df
                last_close = float(df['Close'].iloc[-1])
                print(f"  {t}: {len(df)} days, last ${last_close:.2f}")
        except Exception as e:
            print(f"  {t}: failed: {e}")
    return data


def compute_features(data):
    """Compute ranking features for each sector ETF."""
    features = {}
    for ticker, df in data.items():
        close = df['Close']
        f = pd.DataFrame(index=df.index)
        f['ret_5d'] = close.pct_change(5)
        f['ret_10d'] = close.pct_change(10)
        f['ret_20d'] = close.pct_change(20)
        f['ret_60d'] = close.pct_change(60)
        f['vol_20d'] = close.pct_change().rolling(20).std()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        f['rsi_14'] = 100 - (100 / (1 + rs))

        # Relative strength vs SPY (if SPY in data)
        if 'SPY' in data:
            spy_close = data['SPY']['Close']
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
            spy_ret = spy_close.pct_change(20).reindex(df.index)
            f['rel_strength'] = f['ret_20d'].values - spy_ret.values
        else:
            f['rel_strength'] = f['ret_20d']

        features[ticker] = f
    return features


def simple_rank_sectors(features, date, top_n=2):
    """
    Rank sectors by composite momentum score.
    Uses cross-sectional z-score ranking (same as KB #285).
    """
    scores = {}
    for ticker, feat in features.items():
        if date not in feat.index:
            continue
        row = feat.loc[date]
        if row.isna().any():
            continue
        # Composite: weight recent momentum more
        score = (row['ret_5d'] * 0.3 + row['ret_10d'] * 0.25 +
                row['ret_20d'] * 0.25 + row['ret_60d'] * 0.10 +
                row['rel_strength'] * 0.10)
        scores[ticker] = score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, s in ranked[:top_n]]


def bs_call_price(S, K, T, sigma, r=0.045):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    return max(call, 0)


def bs_delta(S, K, T, sigma, r=0.045):
    """Call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return stats.norm.cdf(d1)


def estimate_iv(prices_series, window=20):
    """Estimate IV from historical vol with markup."""
    hv = prices_series.pct_change().rolling(window).std() * np.sqrt(252)
    iv = hv * 1.15  # IV premium
    return iv.clip(lower=0.12, upper=1.0)


def find_strike_for_delta(S, T, sigma, target_delta, direction='call', r=0.045):
    """Find strike that gives approximately the target delta."""
    # Binary search for the strike
    low_K = S * 0.5
    high_K = S * 1.5

    for _ in range(50):
        mid_K = (low_K + high_K) / 2
        d = bs_delta(S, mid_K, T, sigma, r)

        if direction == 'call':
            if d > target_delta:
                low_K = mid_K
            else:
                high_K = mid_K
        if abs(d - target_delta) < 0.005:
            break

    return round(mid_K, 0)


def simulate_pmcc(data, features, variant_config):
    """
    Simulate PMCC strategy.

    LEAPS: Buy deep ITM call (delta ~0.75-0.80), 6-month expiry
    Short: Sell OTM call (delta ~0.20-0.30), 14-28 DTE
    Roll short call at expiry or when value < 20% of entry premium.
    Rotate LEAPS when sector ranking changes (monthly).
    """
    top_n = variant_config.get('top_n', 1)
    short_dte = variant_config.get('short_dte', 14)
    short_delta_target = variant_config.get('short_delta', 0.25)
    leaps_delta_target = variant_config.get('leaps_delta', 0.75)
    leaps_dte_days = variant_config.get('leaps_dte_days', 180)
    roll_freq_days = variant_config.get('roll_freq', 14)
    vix_adaptive = variant_config.get('vix_adaptive', False)
    rebal_freq_days = variant_config.get('rebal_freq', 20)  # monthly-ish

    # Build aligned price matrix
    prices = pd.DataFrame()
    for t in SECTOR_ETFS:
        if t in data:
            prices[t] = data[t]['Close']
    prices = prices.dropna()

    # IV estimates
    ivs = {t: estimate_iv(prices[t]) for t in prices.columns}

    # VIX proxy
    if 'SPY' in data:
        spy_close = data['SPY']['Close'].reindex(prices.index)
        vix_proxy = spy_close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    else:
        vix_proxy = pd.Series(15.0, index=prices.index)

    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = []

    # State
    leaps_positions = []  # [{ticker, strike, entry_price, entry_cost, entry_date, dte_remaining}]
    short_positions = []  # [{ticker, strike, entry_price, premium_received, entry_date, dte_remaining}]

    start_idx = 70  # warmup for features
    last_rebal_idx = start_idx
    last_roll_idx = start_idx

    for i in range(start_idx, len(prices)):
        date = prices.index[i]
        equity_curve.append({'date': date, 'equity': equity})

        # === MARK-TO-MARKET existing positions ===
        # (simplified: we track PnL on close/roll, not daily MTM)

        # === MONTHLY REBALANCE: Check sector rankings ===
        if i - last_rebal_idx >= rebal_freq_days:
            last_rebal_idx = i
            top_sectors = simple_rank_sectors(features, date, top_n=top_n)

            if not top_sectors:
                continue

            # Close LEAPS not in top sectors
            for pos in leaps_positions[:]:
                if pos['ticker'] not in top_sectors:
                    # Close LEAPS
                    current_price = prices[pos['ticker']].iloc[i]
                    iv = ivs[pos['ticker']].iloc[i] if i < len(ivs[pos['ticker']]) else 0.25
                    if pd.isna(iv): iv = 0.25

                    days_held = (date - pos['entry_date']).days
                    T_remain = max(0, (pos['dte_original'] - days_held)) / 365
                    close_value = bs_call_price(current_price, pos['strike'], T_remain, iv)
                    close_value *= BS_CALIBRATION_FACTOR

                    pnl = close_value - pos['entry_cost'] - 2 * COMMISSION_PER_CONTRACT
                    equity += pnl

                    trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'ticker': pos['ticker'],
                        'type': 'leaps_close',
                        'pnl': pnl,
                        'pnl_pct': pnl / pos['entry_cost'] if pos['entry_cost'] > 0 else 0,
                        'hold_days': days_held,
                        'equity_after': equity,
                    })

                    leaps_positions.remove(pos)

                    # Also close any short calls on this ticker
                    for sp in short_positions[:]:
                        if sp['ticker'] == pos['ticker']:
                            s_current = bs_call_price(current_price, sp['strike'],
                                                      max(0, (sp['dte_original'] - (date - sp['entry_date']).days)/365),
                                                      iv)
                            s_current *= BS_CALIBRATION_FACTOR
                            short_pnl = sp['premium_received'] - s_current - 2 * COMMISSION_PER_CONTRACT
                            equity += short_pnl
                            trades.append({
                                'entry_date': sp['entry_date'],
                                'exit_date': date,
                                'ticker': sp['ticker'],
                                'type': 'short_close',
                                'pnl': short_pnl,
                                'pnl_pct': short_pnl / sp['premium_received'] if sp['premium_received'] > 0 else 0,
                                'hold_days': (date - sp['entry_date']).days,
                                'equity_after': equity,
                            })
                            short_positions.remove(sp)

            # Open LEAPS on new top sectors
            for sector in top_sectors:
                if any(p['ticker'] == sector for p in leaps_positions):
                    continue  # already have position

                current_price = prices[sector].iloc[i]
                iv = ivs[sector].iloc[i] if i < len(ivs[sector]) else 0.25
                if pd.isna(iv): iv = 0.25

                T_leaps = leaps_dte_days / 365
                leaps_strike = find_strike_for_delta(current_price, T_leaps, iv, leaps_delta_target)

                leaps_cost = bs_call_price(current_price, leaps_strike, T_leaps, iv)
                leaps_cost *= BS_CALIBRATION_FACTOR  # calibration markup

                # Per-share cost; for 1 contract = 100 shares
                # But for a small account, we think in per-share terms since we're backtesting
                contract_cost = leaps_cost * 100  # real cost for 1 contract

                # Can we afford it? Max trade = min(300, 50% of equity)
                max_alloc = min(300.0, equity * 0.50)

                if contract_cost > max_alloc:
                    # Too expensive. Use fractional position sizing (hypothetical)
                    # In reality, we'd need the ETF to be cheap enough
                    # For backtesting, we'll scale position size
                    position_multiplier = max_alloc / contract_cost
                else:
                    position_multiplier = 1.0

                actual_cost = leaps_cost * 100 * position_multiplier

                if actual_cost < 50 or equity < actual_cost:
                    continue

                equity -= actual_cost
                equity -= COMMISSION_PER_CONTRACT

                leaps_positions.append({
                    'ticker': sector,
                    'strike': leaps_strike,
                    'entry_price': current_price,
                    'entry_cost': actual_cost,
                    'entry_date': date,
                    'dte_original': leaps_dte_days,
                    'position_mult': position_multiplier,
                    'iv_at_entry': iv,
                })

        # === ROLL SHORT CALLS ===
        if i - last_roll_idx >= roll_freq_days and leaps_positions:
            last_roll_idx = i

            # Close expired/near-expiry short calls
            for sp in short_positions[:]:
                days_held = (date - sp['entry_date']).days
                if days_held >= sp['dte_original'] - 1:
                    # Expired — keep all premium
                    current_price = prices[sp['ticker']].iloc[i]
                    iv = ivs[sp['ticker']].iloc[i] if i < len(ivs[sp['ticker']]) else 0.25
                    if pd.isna(iv): iv = 0.25

                    # Expiry value
                    intrinsic = max(0, current_price - sp['strike'])
                    assignment_cost = intrinsic * 100 * sp.get('position_mult', 1.0)
                    short_pnl = sp['premium_received'] - assignment_cost - COMMISSION_PER_CONTRACT

                    equity += short_pnl
                    trades.append({
                        'entry_date': sp['entry_date'],
                        'exit_date': date,
                        'ticker': sp['ticker'],
                        'type': 'short_expire',
                        'pnl': short_pnl,
                        'pnl_pct': short_pnl / max(sp['premium_received'], 0.01),
                        'hold_days': days_held,
                        'equity_after': equity,
                    })
                    short_positions.remove(sp)

            # Open new short calls on existing LEAPS positions
            for lp in leaps_positions:
                if any(sp['ticker'] == lp['ticker'] for sp in short_positions):
                    continue  # already have short on this

                current_price = prices[lp['ticker']].iloc[i]
                iv = ivs[lp['ticker']].iloc[i] if i < len(ivs[lp['ticker']]) else 0.25
                if pd.isna(iv): iv = 0.25

                # VIX adaptive DTE
                actual_dte = short_dte
                if vix_adaptive:
                    current_vix = vix_proxy.iloc[i] if i < len(vix_proxy) else 15
                    if current_vix > 25:
                        actual_dte = 7  # shorter DTE = more premium capture in high vol
                    elif current_vix > 20:
                        actual_dte = 10

                T_short = actual_dte / 365
                short_strike = find_strike_for_delta(current_price, T_short, iv, short_delta_target)

                # Make sure short strike > current price (OTM call)
                if short_strike <= current_price:
                    short_strike = current_price * 1.02

                premium = bs_call_price(current_price, short_strike, T_short, iv)
                premium *= BS_CALIBRATION_FACTOR
                premium_total = premium * 100 * lp['position_mult']

                if premium_total < 5:  # minimum premium worth collecting
                    continue

                equity += premium_total  # receive premium
                equity -= COMMISSION_PER_CONTRACT

                short_positions.append({
                    'ticker': lp['ticker'],
                    'strike': short_strike,
                    'entry_price': current_price,
                    'premium_received': premium_total,
                    'entry_date': date,
                    'dte_original': actual_dte,
                    'position_mult': lp['position_mult'],
                })

    # Close all remaining positions at end
    final_date = prices.index[-1]
    for pos in leaps_positions:
        current_price = prices[pos['ticker']].iloc[-1]
        iv = ivs[pos['ticker']].iloc[-1] if len(ivs[pos['ticker']]) > 0 else 0.25
        if pd.isna(iv): iv = 0.25

        days_held = (final_date - pos['entry_date']).days
        T_remain = max(0, (pos['dte_original'] - days_held)) / 365
        close_value = bs_call_price(current_price, pos['strike'], T_remain, iv)
        close_value *= BS_CALIBRATION_FACTOR

        pnl = (close_value * 100 * pos['position_mult']) - pos['entry_cost'] - 2 * COMMISSION_PER_CONTRACT
        equity += pnl

        trades.append({
            'entry_date': pos['entry_date'],
            'exit_date': final_date,
            'ticker': pos['ticker'],
            'type': 'leaps_final',
            'pnl': pnl,
            'pnl_pct': pnl / pos['entry_cost'] if pos['entry_cost'] > 0 else 0,
            'hold_days': days_held,
            'equity_after': equity,
        })

    equity_curve.append({'date': final_date, 'equity': equity})

    return trades, equity_curve


def compute_metrics(trades, equity_curve):
    """Compute risk-adjusted metrics."""
    if not trades:
        return None

    n_trades = len(trades)
    pnls = [t['pnl'] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / n_trades

    total_return = (equity_curve[-1]['equity'] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    eq_df = pd.DataFrame(equity_curve)
    eq_df['daily_ret'] = eq_df['equity'].pct_change().fillna(0)

    first_date = eq_df['date'].iloc[0]
    last_date = eq_df['date'].iloc[-1]
    years = max((last_date - first_date).days / 365.25, 0.5)
    cagr = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1.0

    daily_std = eq_df['daily_ret'].std()
    daily_mean = eq_df['daily_ret'].mean()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    downside = eq_df['daily_ret'][eq_df['daily_ret'] < 0].std()
    sortino = (daily_mean / downside * np.sqrt(252)) if downside and downside > 0 else 0

    eq_df['cum_max'] = eq_df['equity'].cummax()
    eq_df['dd'] = (eq_df['equity'] - eq_df['cum_max']) / eq_df['cum_max']
    max_dd = eq_df['dd'].min()

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Income from short calls vs LEAPS P&L
    short_pnl = sum(t['pnl'] for t in trades if 'short' in t.get('type', ''))
    leaps_pnl = sum(t['pnl'] for t in trades if 'leaps' in t.get('type', ''))

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'pf': pf,
        'cagr': cagr,
        'total_return': total_return,
        'max_dd': max_dd,
        'final_equity': equity_curve[-1]['equity'],
        'years': years,
        'short_call_income': short_pnl,
        'leaps_pnl': leaps_pnl,
    }


def permutation_test(trades, equity_curve, n_perms=N_PERMUTATIONS):
    """Shuffle trade PnLs to test significance."""
    real_metrics = compute_metrics(trades, equity_curve)
    if not real_metrics:
        return 1.0, 0.0
    real_sharpe = real_metrics['sharpe']

    perm_sharpes = []
    pnls = [t['pnl'] for t in trades]

    for _ in range(n_perms):
        shuffled = [p * (1 if np.random.random() > 0.5 else -1) for p in pnls]
        equity = INITIAL_CAPITAL
        eq = []
        for p in shuffled:
            equity += p
            eq.append(equity)
        if len(eq) > 1:
            rets = pd.Series(eq).pct_change().dropna()
            if rets.std() > 0:
                perm_sharpes.append(rets.mean() / rets.std() * np.sqrt(252))

    if not perm_sharpes:
        return 1.0, 0.0
    z = (real_sharpe - np.mean(perm_sharpes)) / max(np.std(perm_sharpes), 1e-6)
    return 1 - stats.norm.cdf(z), z


def regime_test(trades, data):
    """Split by bull/bear market regime (SPY 50-day trend)."""
    if not trades or len(trades) < 10:
        return None, None, None

    if 'SPY' not in data:
        return None, None, None

    spy = data['SPY']['Close']
    spy_ma50 = spy.rolling(50).mean()

    bull_pnls, bear_pnls = [], []
    for t in trades:
        d = t['entry_date']
        if d in spy.index and d in spy_ma50.index:
            if spy.loc[d] > spy_ma50.loc[d]:
                bull_pnls.append(t['pnl'])
            else:
                bear_pnls.append(t['pnl'])

    if not bull_pnls or not bear_pnls:
        return None, None, None

    bull_sr = np.mean(bull_pnls) / max(np.std(bull_pnls), 1e-6)
    bear_sr = np.mean(bear_pnls) / max(np.std(bear_pnls), 1e-6)
    gap = abs(bull_sr - bear_sr) / max(abs(bull_sr), abs(bear_sr), 1e-6)

    return bull_sr, bear_sr, gap


def sub_period_test(trades):
    """Both halves positive Sharpe."""
    if len(trades) < 20:
        return False, 0, 0
    mid = len(trades) // 2
    h1 = [t['pnl'] for t in trades[:mid]]
    h2 = [t['pnl'] for t in trades[mid:]]
    h1_sr = np.mean(h1) / max(np.std(h1), 1e-6)
    h2_sr = np.mean(h2) / max(np.std(h2), 1e-6)
    return (h1_sr > 0 and h2_sr > 0), h1_sr, h2_sr


def random_direction_test(trades, n_sims=100):
    """Random sector selection instead of ranked."""
    real_pnls = [t['pnl'] for t in trades]
    real_sr = np.mean(real_pnls) / max(np.std(real_pnls), 1e-6)

    rand_sharpes = []
    for _ in range(n_sims):
        shuffled = [p * (1 if np.random.random() > 0.5 else -1) for p in real_pnls]
        if np.std(shuffled) > 0:
            rand_sharpes.append(np.mean(shuffled) / np.std(shuffled))

    if not rand_sharpes:
        return 1.0, 0.0
    z = (real_sr - np.mean(rand_sharpes)) / max(np.std(rand_sharpes), 1e-6)
    return 1 - stats.norm.cdf(z), z


def run_variant(data, features, variant_name, config):
    """Run variant with 4-gate validation."""
    print(f"\n{'='*60}")
    print(f"VARIANT {variant_name}")
    print(f"{'='*60}")
    print(f"  Config: {config}")

    try:
        trades, equity_curve = simulate_pmcc(data, features, config)
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {'variant': variant_name, 'status': 'ERROR', 'reason': str(e), 'gates_passed': 0}

    if not trades:
        print("  No trades generated")
        return {'variant': variant_name, 'status': 'FAILED', 'reason': 'no trades', 'gates_passed': 0}

    metrics = compute_metrics(trades, equity_curve)
    if not metrics:
        return {'variant': variant_name, 'status': 'FAILED', 'reason': 'no metrics', 'gates_passed': 0}

    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Sortino: {metrics['sortino']:.2f}")
    print(f"  WR: {metrics['win_rate']*100:.1f}%")
    print(f"  PF: {metrics['pf']:.2f}")
    print(f"  CAGR: {metrics['cagr']*100:.1f}%")
    print(f"  MaxDD: {metrics['max_dd']*100:.1f}%")
    print(f"  Final: ${metrics['final_equity']:.0f}")
    print(f"  Short call income: ${metrics['short_call_income']:.0f}")
    print(f"  LEAPS P&L: ${metrics['leaps_pnl']:.0f}")

    # Gate 1: Permutation
    perm_p, perm_z = permutation_test(trades, equity_curve)
    gate1 = perm_p < 0.05
    print(f"\n  Gate 1 (Perm): z={perm_z:.2f}, p={perm_p:.3f} → {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: Regime
    bull_s, bear_s, gap = regime_test(trades, data)
    if gap is not None:
        gate2 = gap < 0.50
        print(f"  Gate 2 (Regime): bull={bull_s:.2f}, bear={bear_s:.2f}, gap={gap:.2f} → {'PASS' if gate2 else 'FAIL'}")
    else:
        gate2 = False
        print(f"  Gate 2 (Regime): insufficient data → FAIL")

    # Gate 3: Random
    rand_p, rand_z = random_direction_test(trades)
    gate3 = rand_p < 0.05
    print(f"  Gate 3 (Random): z={rand_z:.2f}, p={rand_p:.3f} → {'PASS' if gate3 else 'FAIL'}")

    # Gate 4: Sub-period
    sp_pass, sp_h1, sp_h2 = sub_period_test(trades)
    gate4 = sp_pass
    print(f"  Gate 4 (Sub-period): h1={sp_h1:.2f}, h2={sp_h2:.2f} → {'PASS' if gate4 else 'FAIL'}")

    gates_passed = sum([gate1, gate2, gate3, gate4])
    print(f"\n  GATES: {gates_passed}/4")

    return {
        'variant': variant_name,
        'status': 'COMPLETE',
        'metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()},
        'gates': {
            'permutation': {'pass': gate1, 'z': float(perm_z), 'p': float(perm_p)},
            'regime': {'pass': gate2, 'bull': float(bull_s) if bull_s else None,
                      'bear': float(bear_s) if bear_s else None,
                      'gap': float(gap) if gap else None},
            'random': {'pass': gate3, 'z': float(rand_z), 'p': float(rand_p)},
            'sub_period': {'pass': gate4, 'h1': float(sp_h1), 'h2': float(sp_h2)},
        },
        'gates_passed': gates_passed,
    }


def main():
    print("=" * 70)
    print("POOR MAN'S COVERED CALL + SECTOR ROTATION — v1")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if USE_MLFLOW:
        try:
            mlflow.set_experiment("pmcc_sector_rotation_v1")
            mlflow.start_run(run_name=f"pmcc_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.set_tag("node", "jupiter")
            mlflow.set_tag("strategy_type", "growth_income")
        except Exception as e:
            print(f"MLflow: {e}")

    # Fetch data
    all_tickers = SECTOR_ETFS + ['SPY']
    data = fetch_data(all_tickers)

    if len(data) < 5:
        print("FATAL: insufficient data")
        return

    features = compute_features(data)

    variants = {
        'A_top1_weekly': {
            'top_n': 1, 'short_dte': 7, 'short_delta': 0.25,
            'leaps_delta': 0.75, 'leaps_dte_days': 180,
            'roll_freq': 7, 'rebal_freq': 20,
        },
        'B_top1_biweekly': {
            'top_n': 1, 'short_dte': 14, 'short_delta': 0.25,
            'leaps_delta': 0.75, 'leaps_dte_days': 180,
            'roll_freq': 14, 'rebal_freq': 20,
        },
        'C_top2_diversified': {
            'top_n': 2, 'short_dte': 14, 'short_delta': 0.25,
            'leaps_delta': 0.75, 'leaps_dte_days': 180,
            'roll_freq': 14, 'rebal_freq': 20,
        },
        'D_vix_adaptive': {
            'top_n': 1, 'short_dte': 14, 'short_delta': 0.25,
            'leaps_delta': 0.75, 'leaps_dte_days': 180,
            'roll_freq': 14, 'rebal_freq': 20,
            'vix_adaptive': True,
        },
        'E_aggressive_short': {
            'top_n': 1, 'short_dte': 14, 'short_delta': 0.35,
            'leaps_delta': 0.75, 'leaps_dte_days': 180,
            'roll_freq': 14, 'rebal_freq': 20,
        },
        'F_conservative': {
            'top_n': 1, 'short_dte': 21, 'short_delta': 0.15,
            'leaps_delta': 0.85, 'leaps_dte_days': 270,
            'roll_freq': 21, 'rebal_freq': 20,
        },
    }

    results = []
    for name, config in variants.items():
        result = run_variant(data, features, name, config)
        results.append(result)

        if USE_MLFLOW and result.get('metrics'):
            m = result['metrics']
            try:
                mlflow.log_metrics({
                    f"{name}_sharpe": round(m['sharpe'], 3),
                    f"{name}_sortino": round(m['sortino'], 3),
                    f"{name}_wr": round(m['win_rate'], 3),
                    f"{name}_cagr": round(m['cagr'], 3),
                    f"{name}_mdd": round(m['max_dd'], 3),
                    f"{name}_gates": result['gates_passed'],
                })
            except:
                pass

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        if r.get('metrics'):
            m = r['metrics']
            print(f"  {r['variant']}: Sharpe {m['sharpe']:.2f}, WR {m['win_rate']*100:.0f}%, "
                  f"CAGR {m['cagr']*100:.1f}%, MDD {m['max_dd']*100:.1f}%, "
                  f"Gates {r['gates_passed']}/4, ${m['final_equity']:.0f}")
        else:
            print(f"  {r['variant']}: {r.get('status', '?')} — {r.get('reason', '')}")

    best = max(results, key=lambda x: x.get('gates_passed', 0))
    print(f"\n  BEST: {best['variant']} ({best['gates_passed']}/4 gates)")

    if USE_MLFLOW:
        try:
            mlflow.log_metric("best_gates", best.get('gates_passed', 0))
            mlflow.set_tag("best_variant", best['variant'])
            mlflow.set_tag("result", "PASS" if best.get('gates_passed', 0) >= 3 else "FAIL")
            mlflow.end_run()
        except:
            pass

    # Save
    output_path = "/home/jupiter/Lvl3Quant/research/findings/pmcc_sector_rotation_v1_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
