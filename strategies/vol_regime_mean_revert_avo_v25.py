#!/usr/bin/env python3
"""
ETF Vol Regime Mean-Reversion Strategy -- AVO v7
=================================================
v16 changes: Parkinson/CC vol ratio filter. When rv_pk_20d / rv_cc_20d > 1.10,
skip the signal -- high Parkinson relative to close-to-close means intraday
whipsaw without directional close moves (false compression). Removes 4 bad
trades from 2022H1 while improving mid-regime Sharpe.
"""

import numpy as np
import pandas as pd

# -- Configuration (EVOLVE THESE) --------------------------------------------
SPIKE_THRESHOLD = 1.20
COMPRESSION_THRESHOLD = 0.98
SPIKE_LOOKBACK = 14
TREND_GUARD_THRESHOLD = -0.25
COMPRESSION_CONFIRM_DAYS = 2
RET_D_FLOOR = -0.020
PK_CC_CEILING = 1.10

# Defensive sector ETF signals (lowered bar to generate more trades)
EXTRA_ETFS = ['XLU', 'XLP', 'XLV', 'XLRE', 'XLI']
EXTRA_SIGNAL_SPIKE_THRESHOLD = 0.10  # spike_mag above this (was 0.15)

MAX_PER_TRADE = 1500.0
MAX_CONCURRENT = 4
SLIPPAGE_PCT = 0.0001         # 0.01% each way

# Exit parameters
TRAILING_STOP_PCT = -0.007
GAIN_LOCK_THRESHOLD = 0.005
GAIN_LOCK_STOP = -0.006
GAIN_LOCK15_THRESHOLD = 0.008
GAIN_LOCK15_STOP = -0.004
GAIN_LOCK2_THRESHOLD = 0.015
GAIN_LOCK2_STOP = -0.001
TAKE_PROFIT_PCT = 0.045
MAX_HOLD_DAYS = 5
UNDERWATER_CUT_DAYS = 2
PORTFOLIO_DD_EXIT = -0.007


def fit(train_data):
    """Learn from SPY's vol regime patterns."""
    if 'ticker' in train_data.columns:
        ticker_col = 'ticker'
    elif 'etf' in train_data.columns:
        ticker_col = 'etf'
    else:
        return {'spy_stats': {}}

    spy = train_data[train_data[ticker_col] == 'SPY'].sort_values('date').copy()

    if len(spy) < 30 or 'rv_cc_5d' not in spy.columns or 'rv_cc_20d' not in spy.columns:
        return {'spy_stats': {}}

    spy['rv_ratio'] = spy['rv_cc_5d'] / spy['rv_cc_20d'].clip(lower=1e-8)
    rv_vals = spy['rv_ratio'].values
    close_vals = spy['close'].values if 'close' in spy.columns else None

    fwd_returns = []
    n_events = 0
    for i in range(SPIKE_LOOKBACK, len(spy) - 5):
        lb_max = np.nanmax(rv_vals[max(0, i - SPIKE_LOOKBACK):i])
        if lb_max > SPIKE_THRESHOLD and rv_vals[i] < COMPRESSION_THRESHOLD:
            n_events += 1
            if close_vals is not None and i + 5 < len(close_vals):
                fwd = (close_vals[i + 5] - close_vals[i]) / close_vals[i]
                fwd_returns.append(fwd)

    return {
        'spy_stats': {
            'n_events': n_events,
            'avg_fwd': np.mean(fwd_returns) if fwd_returns else 0,
            'hit_rate': np.mean([r > 0 for r in fwd_returns]) if fwd_returns else 0,
        }
    }


def generate_signals(data, params):
    """
    SPY vol compression signals + defensive sector ETFs on strong days.
    """
    if 'ticker' in data.columns:
        ticker_col = 'ticker'
    elif 'etf' in data.columns:
        ticker_col = 'etf'
    else:
        return pd.DataFrame(columns=['date', 'etf', 'score'])

    # Build SPY rv_ratio
    spy = data[data[ticker_col] == 'SPY'].sort_values('date').copy()
    if len(spy) < SPIKE_LOOKBACK + 1:
        return pd.DataFrame(columns=['date', 'etf', 'score'])

    if 'rv_cc_5d' not in spy.columns or 'rv_cc_20d' not in spy.columns:
        return pd.DataFrame(columns=['date', 'etf', 'score'])

    spy['rv_ratio'] = spy['rv_cc_5d'] / spy['rv_cc_20d'].clip(lower=1e-8)
    rv_vals = spy['rv_ratio'].values

    # Parkinson/CC ratio: high = intraday whipsaw without close moves (false compression)
    has_pk = 'rv_pk_20d' in spy.columns
    if has_pk:
        spy['pk_cc_ratio'] = spy['rv_pk_20d'] / spy['rv_cc_20d'].clip(lower=1e-8)
        pk_cc_vals = spy['pk_cc_ratio'].values
    else:
        pk_cc_vals = None

    signals = []

    for i in range(SPIKE_LOOKBACK, len(spy)):
        row = spy.iloc[i]
        curr = rv_vals[i]
        if pd.isna(curr):
            continue

        ret_20d = row.get('ret_20d', 0) if 'ret_20d' in spy.columns else 0
        if not pd.isna(ret_20d) and ret_20d < TREND_GUARD_THRESHOLD:
            continue

        # Daily return confirmation: skip if SPY had a big down day (premature entry)
        ret_d = row.get('ret_d', 0) if 'ret_d' in spy.columns else 0
        if not pd.isna(ret_d) and ret_d < RET_D_FLOOR:
            continue

        # Skip if Parkinson/CC vol ratio is elevated (intraday whipsaw, false compression)
        if pk_cc_vals is not None and not pd.isna(pk_cc_vals[i]):
            if pk_cc_vals[i] > PK_CC_CEILING:
                continue

        lb = rv_vals[max(0, i - SPIKE_LOOKBACK):i]
        if len(lb) == 0:
            continue
        lb_max = np.nanmax(lb)

        if lb_max > SPIKE_THRESHOLD and curr < COMPRESSION_THRESHOLD:
            # Compression confirmation: require rv_ratio declining for N days
            if i >= COMPRESSION_CONFIRM_DAYS:
                confirmed = True
                for d in range(1, COMPRESSION_CONFIRM_DAYS + 1):
                    prev_idx = i - d
                    if prev_idx < 0 or pd.isna(rv_vals[prev_idx]):
                        confirmed = False
                        break
                    if d == 1:
                        if rv_vals[i] >= rv_vals[prev_idx]:
                            confirmed = False
                            break
                    else:
                        if rv_vals[i - d + 1] >= rv_vals[prev_idx]:
                            confirmed = False
                            break
                if not confirmed:
                    continue
            else:
                continue

            spike_mag = max(lb_max - SPIKE_THRESHOLD, 0.01)
            comp_speed = max(COMPRESSION_THRESHOLD - curr, 0.01)
            score = spike_mag * comp_speed

            # Always signal SPY
            signals.append({
                'date': row['date'],
                'etf': 'SPY',
                'score': score,
            })

            # Defensive sector ETFs on moderate+ spike days
            if spike_mag > EXTRA_SIGNAL_SPIKE_THRESHOLD:
                for etf in EXTRA_ETFS:
                    signals.append({
                        'date': row['date'],
                        'etf': etf,
                        'score': score * 0.85,
                    })

    if not signals:
        return pd.DataFrame(columns=['date', 'etf', 'score'])

    return pd.DataFrame(signals)


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Check if a position should be exited."""
    entry_price = pos['entry_price_adj']
    high_water = pos.get('high_water_mark', entry_price)
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    pnl_pct = (current_price - entry_price) / entry_price

    if days_held >= MAX_HOLD_DAYS:
        return True

    if pnl_pct >= TAKE_PROFIT_PCT:
        return True

    if high_water > 0:
        drawdown_from_high = (current_price - high_water) / high_water
        # Graduated gain lock: tighter stops as gains increase
        hwm_gain = (high_water - entry_price) / entry_price
        if hwm_gain >= GAIN_LOCK2_THRESHOLD:
            stop = GAIN_LOCK2_STOP
        elif hwm_gain >= GAIN_LOCK15_THRESHOLD:
            stop = GAIN_LOCK15_STOP
        elif hwm_gain >= GAIN_LOCK_THRESHOLD:
            stop = GAIN_LOCK_STOP
        else:
            stop = TRAILING_STOP_PCT
        if drawdown_from_high <= stop:
            return True

    if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < 0:
        return True

    # Portfolio-level risk: if portfolio is in drawdown and position is losing, exit
    if portfolio_dd < PORTFOLIO_DD_EXIT and pnl_pct < 0:
        return True

    return False
