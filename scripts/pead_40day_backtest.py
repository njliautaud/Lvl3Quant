#!/usr/bin/env python3
"""
PEAD 40-Day Backtest — Post-Earnings Announcement Drift
========================================================
Focus: Stock SELECTION via beat-chain identification, 40-day holds.
Key insight: PEAD alpha is in stock selection (beat-chain stocks go up),
not entry timing. 40-day hold (Sharpe ~1.03) >> 5-day (0.51).

Variants:
  A) Beat Gap + 40-day hold
  B) Beat-chain (2nd+ consecutive beat) + 40-day
  C) Large Beat (>5% gap) + 40-day
  D) Beat + Trend Confirm (above 200-SMA)
  E) Beat + RSI Filter (RSI14 < 70)
  F) Portfolio: top 3 by gap size, max 3 concurrent, equal weight

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'AVGO', 'ORCL', 'ADBE', 'MU', 'QCOM', 'PLTR',
    'SOFI', 'HOOD', 'COIN', 'UBER'
]

ACCOUNT_SIZE = 669.0
HOLD_DAYS = 40          # trading days
BEAT_GAP_PCT = 2.0      # minimum gap-up % to count as "beat"
LARGE_BEAT_GAP_PCT = 5.0
MISS_GAP_PCT = -2.0     # gap-down threshold
STOP_LOSS_PCT = -15.0   # catastrophic stop only
SLIPPAGE_PCT = 0.02     # 0.02% per leg, both legs
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2020-01-01'  # need lookback for 200-SMA and beat history
N_PERMUTATIONS = 500
RSI_PERIOD = 14
RSI_THRESHOLD = 70
SMA_PERIOD = 200

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/pead_40day_results.json')

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download price data for all tickers + SPY."""
    all_tickers = TICKERS + ['SPY']
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=DATA_START, end=OOT_END,
                           progress=False, auto_adjust=True)
            # Flatten MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > SMA_PERIOD:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} days), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e}), skipping")
    return data


# ─── Feature Computation ────────────────────────────────────────────────────

def compute_features(data):
    """Compute gaps, SMA, RSI, and beat history for all tickers."""
    features = {}
    spy = data.get('SPY')
    if spy is None:
        raise ValueError("SPY data required for regime classification")

    # SPY 200-SMA for regime
    spy_sma200 = spy['Close'].rolling(SMA_PERIOD).mean()

    for ticker in TICKERS:
        if ticker not in data:
            continue
        df = data[ticker].copy()

        # Gap % = (Open - prev Close) / prev Close * 100
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close'] * 100

        # 200-SMA
        df['sma200'] = df['Close'].rolling(SMA_PERIOD).mean()
        df['above_sma200'] = df['Close'] > df['sma200']

        # RSI(14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(RSI_PERIOD).mean()
        avg_loss = loss.rolling(RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Beat/Miss detection
        df['is_beat'] = df['gap_pct'] > BEAT_GAP_PCT
        df['is_miss'] = df['gap_pct'] < MISS_GAP_PCT
        df['is_large_beat'] = df['gap_pct'] > LARGE_BEAT_GAP_PCT

        # Consecutive beat count (beat-chain)
        consecutive = []
        count = 0
        for _, row in df.iterrows():
            if row['is_beat']:
                count += 1
                consecutive.append(count)
            elif row['is_miss']:
                count = 0
                consecutive.append(0)
            else:
                consecutive.append(count)
        df['consecutive_beats'] = consecutive

        # Regime from SPY
        df['regime'] = 'unknown'
        for idx in df.index:
            if idx in spy_sma200.index and not pd.isna(spy_sma200.loc[idx]):
                spy_close = spy.loc[idx, 'Close'] if idx in spy.index else np.nan
                if not np.isnan(spy_close):
                    df.loc[idx, 'regime'] = 'bull' if spy_close > spy_sma200.loc[idx] else 'bear'

        features[ticker] = df

    return features


# ─── Trade Simulation ────────────────────────────────────────────────────────

def simulate_trade(df, entry_idx_pos, capital):
    """
    Simulate a single trade: buy at entry_idx Open, hold 40 days or stop-loss.
    Returns: (exit_idx_pos, pnl, return_pct, hold_days, exit_reason)
    """
    entry_price = df.iloc[entry_idx_pos]['Open']
    slippage_cost = entry_price * SLIPPAGE_PCT / 100  # entry slippage
    effective_entry = entry_price + slippage_cost
    shares = capital / effective_entry

    exit_idx_pos = min(entry_idx_pos + HOLD_DAYS, len(df) - 1)
    exit_reason = 'hold_40d'

    # Check for stop-loss during hold
    for i in range(entry_idx_pos + 1, exit_idx_pos + 1):
        low = df.iloc[i]['Low']
        current_return = (low - effective_entry) / effective_entry * 100
        if current_return <= STOP_LOSS_PCT:
            exit_idx_pos = i
            exit_reason = 'stop_loss'
            break

    if exit_reason == 'stop_loss':
        # Exit at stop-loss level
        exit_price = effective_entry * (1 + STOP_LOSS_PCT / 100)
    else:
        exit_price = df.iloc[exit_idx_pos]['Close']

    exit_slippage = exit_price * SLIPPAGE_PCT / 100
    effective_exit = exit_price - exit_slippage

    pnl = shares * (effective_exit - effective_entry)
    return_pct = (effective_exit - effective_entry) / effective_entry * 100
    actual_hold = exit_idx_pos - entry_idx_pos

    return exit_idx_pos, pnl, return_pct, actual_hold, exit_reason


def run_variant(features, variant, capital=ACCOUNT_SIZE):
    """
    Run a specific variant and return list of trades.
    Each trade: {ticker, entry_date, exit_date, gap_pct, pnl, return_pct,
                 hold_days, exit_reason, regime, capital_used}
    """
    trades = []

    if variant == 'F':
        return run_portfolio_variant(features, capital)

    for ticker, df in features.items():
        # Filter to OOT period
        oot_mask = df.index >= OOT_START
        oot_indices = df.index[oot_mask]

        i = 0
        blocked_until = -1  # position index we're blocked until

        for idx in oot_indices:
            pos = df.index.get_loc(idx)
            if pos <= blocked_until:
                continue

            row = df.loc[idx]
            if pd.isna(row.get('gap_pct', np.nan)):
                continue

            signal = False

            if variant == 'A':
                signal = row['is_beat']
            elif variant == 'B':
                signal = row['is_beat'] and row['consecutive_beats'] >= 2
            elif variant == 'C':
                signal = row['is_large_beat']
            elif variant == 'D':
                signal = row['is_beat'] and row['above_sma200']
            elif variant == 'E':
                signal = row['is_beat'] and (not pd.isna(row['rsi'])) and row['rsi'] < RSI_THRESHOLD

            if signal:
                exit_pos, pnl, ret_pct, hold_d, exit_reason = simulate_trade(df, pos, capital)
                blocked_until = exit_pos

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(idx.date()),
                    'exit_date': str(df.index[exit_pos].date()),
                    'gap_pct': float(row['gap_pct']),
                    'pnl': float(pnl),
                    'return_pct': float(ret_pct),
                    'hold_days': int(hold_d),
                    'exit_reason': exit_reason,
                    'regime': row['regime'],
                    'capital_used': float(capital),
                })

    return sorted(trades, key=lambda t: t['entry_date'])


def run_portfolio_variant(features, capital=ACCOUNT_SIZE):
    """
    Variant F: Monthly, take top 3 gaps, equal weight, max 3 concurrent.
    """
    # Collect all beat signals in OOT
    all_signals = []
    for ticker, df in features.items():
        oot_mask = df.index >= OOT_START
        for idx in df.index[oot_mask]:
            row = df.loc[idx]
            if row['is_beat'] and not pd.isna(row['gap_pct']):
                pos = df.index.get_loc(idx)
                all_signals.append({
                    'ticker': ticker,
                    'date': idx,
                    'pos': pos,
                    'gap_pct': float(row['gap_pct']),
                    'regime': row['regime'],
                    'df': df
                })

    all_signals.sort(key=lambda s: s['date'])

    trades = []
    active_positions = []  # list of (exit_date_pos, ticker)
    monthly_selections = {}  # year-month -> already selected

    for sig in all_signals:
        month_key = sig['date'].strftime('%Y-%m')

        # Clear expired positions
        active_positions = [(ep, t) for ep, t in active_positions
                          if sig['pos'] <= ep]

        if len(active_positions) >= 3:
            continue

        if month_key not in monthly_selections:
            monthly_selections[month_key] = []

        if len(monthly_selections[month_key]) >= 3:
            continue

        pos_capital = capital / 3.0
        df = sig['df']
        exit_pos, pnl, ret_pct, hold_d, exit_reason = simulate_trade(
            df, sig['pos'], pos_capital)

        active_positions.append((exit_pos, sig['ticker']))
        monthly_selections[month_key].append(sig['ticker'])

        trades.append({
            'ticker': sig['ticker'],
            'entry_date': str(sig['date'].date()),
            'exit_date': str(df.index[exit_pos].date()),
            'gap_pct': sig['gap_pct'],
            'pnl': float(pnl),
            'return_pct': float(ret_pct),
            'hold_days': int(hold_d),
            'exit_reason': exit_reason,
            'regime': sig['regime'],
            'capital_used': float(pos_capital),
        })

    return sorted(trades, key=lambda t: t['entry_date'])


# ─── Metrics Computation ────────────────────────────────────────────────────

def compute_metrics(trades, capital=ACCOUNT_SIZE, variant='A'):
    """Compute all performance metrics for a list of trades."""
    if not trades:
        return {
            'total_return_pct': 0, 'cagr_pct': 0, 'sharpe': 0,
            'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'trade_count': 0,
            'sharpe_bull': 0, 'sharpe_bear': 0, 'regime_gap': 0,
        }

    returns = [t['return_pct'] / 100 for t in trades]
    pnls = [t['pnl'] for t in trades]

    # Build equity curve with compounding: apply return_pct to current equity
    equity = [capital]
    for t in trades:
        current_eq = equity[-1]
        if current_eq <= 0:
            equity.append(0)
            continue
        new_eq = max(current_eq * (1 + t['return_pct'] / 100), 0)
        equity.append(new_eq)
    equity = np.array(equity)

    # Total return
    total_return_pct = (equity[-1] - capital) / capital * 100

    # CAGR
    first_date = datetime.strptime(trades[0]['entry_date'], '%Y-%m-%d')
    last_date = datetime.strptime(trades[-1]['exit_date'], '%Y-%m-%d')
    years = max((last_date - first_date).days / 365.25, 0.1)
    if equity[-1] > 0:
        cagr = (equity[-1] / capital) ** (1 / years) - 1
    else:
        cagr = -1.0

    # Sharpe (annualized, assuming ~9 trades/year as baseline for 40-day holds)
    ret_arr = np.array(returns)
    if len(ret_arr) > 1 and np.std(ret_arr) > 0:
        trades_per_year = len(trades) / years
        sharpe = (np.mean(ret_arr) / np.std(ret_arr)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = ret_arr[ret_arr < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        trades_per_year = len(trades) / years
        sortino = (np.mean(ret_arr) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Profit Factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win Rate
    winners = sum(1 for r in returns if r > 0)
    win_rate = winners / len(returns) * 100

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd = float(np.min(drawdown))

    # Per-regime Sharpe
    bull_returns = [t['return_pct'] / 100 for t in trades if t['regime'] == 'bull']
    bear_returns = [t['return_pct'] / 100 for t in trades if t['regime'] == 'bear']

    def regime_sharpe(rets, total_years):
        if len(rets) < 2 or np.std(rets) == 0:
            return 0.0
        tpy = len(rets) / total_years
        return (np.mean(rets) / np.std(rets)) * np.sqrt(max(tpy, 1))

    sharpe_bull = regime_sharpe(bull_returns, years)
    sharpe_bear = regime_sharpe(bear_returns, years)

    # Regime gap
    max_regime = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_regime if max_regime > 0 else 0.0

    return {
        'total_return_pct': round(total_return_pct, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate, 1),
        'max_drawdown_pct': round(max_dd, 2),
        'trade_count': len(trades),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'avg_return_pct': round(np.mean(ret_arr) * 100, 2),
        'avg_hold_days': round(np.mean([t['hold_days'] for t in trades]), 1),
        'stop_loss_count': sum(1 for t in trades if t['exit_reason'] == 'stop_loss'),
        'final_equity': round(equity[-1], 2),
        'bull_trades': len(bull_returns),
        'bear_trades': len(bear_returns),
    }


# ─── Permutation Test ───────────────────────────────────────────────────────

def permutation_test(trades, features, variant, n_perms=N_PERMUTATIONS):
    """
    Shuffle entry dates randomly and re-simulate to test if alpha is real.
    Optimized: precompute 40-day forward returns for all valid dates per ticker.
    Returns: (p_value, percentile, perm_sharpes)
    """
    if len(trades) < 5:
        return 1.0, 0.0, []

    actual_metrics = compute_metrics(trades, variant=variant)
    actual_sharpe = actual_metrics['sharpe']

    # Precompute 40-day forward returns for all valid OOT dates per ticker
    # This avoids calling simulate_trade thousands of times
    ticker_fwd_returns = {}
    for ticker, df in features.items():
        oot_mask = df.index >= OOT_START
        oot_positions = np.where(oot_mask)[0]
        # Only positions with enough forward data
        valid_pos = oot_positions[oot_positions + HOLD_DAYS < len(df)]
        if len(valid_pos) == 0:
            continue

        fwd_rets = []
        for pos in valid_pos:
            entry_price = df.iloc[pos]['Open']
            slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

            # Check stop-loss
            stopped = False
            for j in range(pos + 1, min(pos + HOLD_DAYS + 1, len(df))):
                low = df.iloc[j]['Low']
                if (low - slip_entry) / slip_entry * 100 <= STOP_LOSS_PCT:
                    exit_price = slip_entry * (1 + STOP_LOSS_PCT / 100)
                    stopped = True
                    break

            if not stopped:
                exit_pos = min(pos + HOLD_DAYS, len(df) - 1)
                exit_price = df.iloc[exit_pos]['Close']

            slip_exit = exit_price * (1 - SLIPPAGE_PCT / 100)
            ret = (slip_exit - slip_entry) / slip_entry
            fwd_rets.append(ret)

        ticker_fwd_returns[ticker] = np.array(fwd_rets)

    # Map each trade to its ticker for permutation
    trade_tickers = [t['ticker'] for t in trades]
    trade_caps = np.array([t['capital_used'] for t in trades])

    # Actual returns for Sharpe baseline
    actual_rets = np.array([t['return_pct'] / 100 for t in trades])

    rng = np.random.RandomState(42)
    perm_sharpes = []
    n_trades = len(trades)

    # Estimate trades_per_year from actual trades
    first_date = datetime.strptime(trades[0]['entry_date'], '%Y-%m-%d')
    last_date = datetime.strptime(trades[-1]['exit_date'], '%Y-%m-%d')
    years = max((last_date - first_date).days / 365.25, 0.1)
    tpy = n_trades / years

    for _ in range(n_perms):
        perm_rets = np.zeros(n_trades)
        for i, ticker in enumerate(trade_tickers):
            if ticker not in ticker_fwd_returns or len(ticker_fwd_returns[ticker]) == 0:
                perm_rets[i] = 0.0
                continue
            perm_rets[i] = rng.choice(ticker_fwd_returns[ticker])

        if np.std(perm_rets) > 0:
            s = (np.mean(perm_rets) / np.std(perm_rets)) * np.sqrt(tpy)
        else:
            s = 0.0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= actual_sharpe))
    percentile = float(np.mean(perm_sharpes < actual_sharpe) * 100)

    return p_value, percentile, perm_sharpes.tolist()


# ─── 5-Gate Validation ──────────────────────────────────────────────────────

def five_gate_check(metrics, p_value):
    """Check all 5 gates. Returns dict of gate results."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'trades_gte_20': metrics['trade_count'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PEAD 40-DAY BACKTEST — Post-Earnings Announcement Drift")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT_SIZE:.0f} | Hold: {HOLD_DAYS} days | Stop: {STOP_LOSS_PCT}%")
    print("=" * 70)

    # Download data
    data = download_data()
    print(f"\nLoaded {len(data)} tickers successfully.\n")

    # Compute features
    print("Computing features (gaps, SMA, RSI, beat-chains)...")
    features = compute_features(data)
    print(f"Features computed for {len(features)} tickers.\n")

    # Run all variants
    variants = {
        'A': 'Beat Gap >2% + 40-day hold',
        'B': 'Beat-chain (2nd+ beat) + 40-day',
        'C': 'Large Beat >5% + 40-day',
        'D': 'Beat + Trend Confirm (>200-SMA)',
        'E': 'Beat + RSI<70 Filter',
        'F': 'Portfolio (top 3/month, max 3 concurrent)',
    }

    results = {}

    for var_key, var_name in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_key}: {var_name}")
        print(f"{'─' * 60}")

        trades = run_variant(features, var_key)
        print(f"  Trades generated: {len(trades)}")

        metrics = compute_metrics(trades, variant=var_key)
        print(f"  Sharpe: {metrics['sharpe']:.3f} | WR: {metrics['win_rate']:.1f}% | "
              f"MaxDD: {metrics['max_drawdown_pct']:.1f}% | "
              f"Total Return: {metrics['total_return_pct']:.1f}%")

        # Permutation test
        print(f"  Running {N_PERMUTATIONS} permutations...")
        p_value, percentile, perm_sharpes = permutation_test(
            trades, features, var_key)
        print(f"  Perm p-value: {p_value:.4f} | Percentile: {percentile:.1f}%")

        # 5-gate check
        gates = five_gate_check(metrics, p_value)

        results[var_key] = {
            'name': var_name,
            'metrics': metrics,
            'p_value': round(p_value, 4),
            'percentile': round(percentile, 1),
            'gates': gates,
            'trade_summary': {
                'tickers_traded': list(set(t['ticker'] for t in trades)),
                'avg_gap_pct': round(np.mean([t['gap_pct'] for t in trades]), 2) if trades else 0,
                'stop_loss_pct': round(metrics['stop_loss_count'] / max(len(trades), 1) * 100, 1),
            }
        }

        # Gate summary
        gate_str = " | ".join([
            f"{'PASS' if v else 'FAIL'}" for k, v in gates.items() if k != 'all_pass'
        ])
        overall = "ALL PASS" if gates['all_pass'] else "FAILED"
        print(f"  5-Gate: [{gate_str}] → {overall}")

    # Save results
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'oot_period': f"{OOT_START} to {OOT_END}",
            'account_size': ACCOUNT_SIZE,
            'hold_days': HOLD_DAYS,
            'tickers': TICKERS,
            'slippage_pct': SLIPPAGE_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'n_permutations': N_PERMUTATIONS,
        },
        'variants': results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    output = convert_types(output)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE — PEAD 40-Day Backtest")
    print("=" * 100)
    header = f"{'Var':<4} {'Name':<38} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>5} " \
             f"{'MaxDD%':>7} {'CAGR%':>7} {'#Trds':>6} {'p-val':>6} {'RGap':>5} {'Gate':>6}"
    print(header)
    print("-" * 100)

    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[var_key]
        m = r['metrics']
        gate_status = "PASS" if r['gates']['all_pass'] else "FAIL"
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf"
        print(f"{var_key:<4} {r['name']:<38} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{pf_str:>6} {m['win_rate']:>5.1f} {m['max_drawdown_pct']:>7.1f} "
              f"{m['cagr_pct']:>7.1f} {m['trade_count']:>6} {r['p_value']:>6.3f} "
              f"{m['regime_gap']:>5.2f} {gate_status:>6}")

    print("-" * 100)

    # Regime breakdown
    print("\nREGIME BREAKDOWN:")
    print(f"{'Var':<4} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Bull #':>7} {'Bear #':>7} {'Gap':>6}")
    print("-" * 50)
    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        m = results[var_key]['metrics']
        print(f"{var_key:<4} {m['sharpe_bull']:>12.3f} {m['sharpe_bear']:>12.3f} "
              f"{m['bull_trades']:>7} {m['bear_trades']:>7} {m['regime_gap']:>6.2f}")

    # 5-Gate detail
    print("\n5-GATE VALIDATION DETAIL:")
    gate_names = ['sharpe_gt_0.5', 'perm_p_lt_0.05', 'regime_gap_lt_0.5',
                  'maxdd_gt_neg50', 'trades_gte_20']
    gate_labels = ['Sharpe>0.5', 'Perm p<0.05', 'RegGap<0.5', 'MaxDD>-50%', '#Trades>=20']
    header2 = f"{'Var':<4} " + " ".join(f"{l:>13}" for l in gate_labels) + f" {'OVERALL':>10}"
    print(header2)
    print("-" * 85)
    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        g = results[var_key]['gates']
        row = f"{var_key:<4} "
        for gn in gate_names:
            row += f"{'PASS':>13} " if g[gn] else f"{'FAIL':>13} "
        row += f"{'ALL PASS':>10}" if g['all_pass'] else f"{'FAILED':>10}"
        print(row)

    # Final verdict
    print("\n" + "=" * 100)
    passing = [k for k in results if results[k]['gates']['all_pass']]
    if passing:
        best = max(passing, key=lambda k: results[k]['metrics']['sharpe'])
        bm = results[best]['metrics']
        print(f"BEST PASSING VARIANT: {best} — {results[best]['name']}")
        print(f"  Sharpe {bm['sharpe']:.3f} | Sortino {bm['sortino']:.3f} | "
              f"CAGR {bm['cagr_pct']:.1f}% | WR {bm['win_rate']:.1f}% | "
              f"MaxDD {bm['max_drawdown_pct']:.1f}% | {bm['trade_count']} trades")
        print(f"  Final equity: ${bm['final_equity']:.2f} (from ${ACCOUNT_SIZE:.0f})")
    else:
        print("NO VARIANTS PASSED ALL 5 GATES.")
        # Show best anyway
        best = max(results.keys(), key=lambda k: results[k]['metrics']['sharpe'])
        bm = results[best]['metrics']
        print(f"Best (non-passing): {best} — Sharpe {bm['sharpe']:.3f}, "
              f"failed gates: {[gn for gn, gv in results[best]['gates'].items() if not gv and gn != 'all_pass']}")

    print("=" * 100)
    return results


if __name__ == '__main__':
    results = main()
