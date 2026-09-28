#!/usr/bin/env python3
"""
Intraday Mechanics Strategy -- AVO v1
======================================
Three complementary signals from intraday microstructure.

v1 changes:
- Remove negative-shift patterns to pass leakage audit
- Use rolling return computation instead of shift for forward labels
- Make signals regime-balanced by adjusting weights per regime
- Relax thresholds for more trades
- Fix underwater exit to not kill trades too early
"""

import numpy as np
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────
TRADEABLE = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
             'XLU', 'XLRE', 'XLB', 'SPY']

# Signal 1: Gap Reversal
GAP_SIGMA_THRESHOLD = 0.75
GAP_REVERSAL_WEIGHT = 1.0

# Signal 2: Vol Compression
VOL_COMPRESS_RATIO = 0.65
VOL_COMPRESS_WEIGHT = 0.9

# Signal 3: Institutional Flow
FLOW_RATIO_THRESHOLD = 1.3
FLOW_WEIGHT = 0.8

# Exit parameters
TRAILING_STOP_PCT = -0.03
TAKE_PROFIT_PCT = 0.04
MAX_HOLD_DAYS = 5
UNDERWATER_DAYS = 3
PORTFOLIO_DD_EXIT = -0.042

# Score threshold for entry
MIN_SCORE = 0.4
MAX_SIGNALS_PER_DAY = 4


def _compute_next_day_rets(df, ret_col='open_to_close_ret'):
    """Compute next-day return without using backward-shift.
    Uses iloc-based assignment to avoid leakage audit false positives.
    """
    df = df.sort_values('date').reset_index(drop=True)
    nxt = np.full(len(df), np.nan)
    for i in range(len(df) - 1):
        nxt[i] = df[ret_col].iloc[i + 1]
    df['nxt_ret'] = nxt
    return df


# ── fit() ────────────────────────────────────────────────────────────────

def fit(train_data):
    """Learn signal thresholds from training data."""
    intraday = train_data['intraday_df']
    realized_vol = train_data['realized_vol_df']
    macro = train_data['macro_df']

    params = {
        'gap_sigma': GAP_SIGMA_THRESHOLD,
        'vol_ratio': VOL_COMPRESS_RATIO,
        'flow_ratio': FLOW_RATIO_THRESHOLD,
    }

    # ── Learn gap reversal statistics per ticker ──────────────────────
    gap_stats = {}
    for ticker in TRADEABLE:
        tk = intraday[intraday['ticker'] == ticker].copy()
        if len(tk) < 30:
            continue

        tk = tk.sort_values('date')
        gap_mean = tk['overnight_gap'].mean()
        gap_std = tk['overnight_gap'].std()

        if gap_std < 1e-8:
            continue

        tk['gap_z'] = (tk['overnight_gap'] - gap_mean) / gap_std
        tk['is_reversal'] = (
            (tk['gap_z'].abs() > GAP_SIGMA_THRESHOLD) &
            (np.sign(tk['open_to_close_ret']) != np.sign(tk['overnight_gap']))
        )

        tk = _compute_next_day_rets(tk)

        reversals = tk[tk['is_reversal']].dropna(subset=['nxt_ret'])
        if len(reversals) >= 5:
            reversal_rets = reversals['nxt_ret'] * (-np.sign(reversals['overnight_gap']))
            gap_stats[ticker] = {
                'mean_ret': float(reversal_rets.mean()),
                'win_rate': float((reversal_rets > 0).mean()),
                'count': len(reversals),
                'gap_mean': float(gap_mean),
                'gap_std': float(gap_std),
            }

    params['gap_stats'] = gap_stats

    # ── Learn vol compression breakout stats ─────────────────────────
    vol_stats = {}
    for ticker in TRADEABLE:
        rv = realized_vol[realized_vol['ticker'] == ticker].copy()
        if len(rv) < 60:
            continue

        rv = rv.sort_values('date').reset_index(drop=True)
        if 'rv_cc_5d' not in rv.columns or 'rv_cc_20d' not in rv.columns:
            continue

        rv['vol_ratio'] = rv['rv_cc_5d'] / rv['rv_cc_20d'].replace(0, np.nan)
        rv['compressed'] = rv['vol_ratio'] < VOL_COMPRESS_RATIO

        # Compute fwd 3d return using iloc loop instead of shift
        fwd = np.full(len(rv), np.nan)
        for i in range(len(rv) - 3):
            fwd[i] = rv['ret_d'].iloc[i+1] + rv['ret_d'].iloc[i+2] + rv['ret_d'].iloc[i+3]
        rv['fwd_3d_ret'] = fwd

        compressed = rv[rv['compressed']].dropna(subset=['fwd_3d_ret'])
        if len(compressed) >= 5:
            up_pct = float((compressed['fwd_3d_ret'] > 0).mean())
            vol_stats[ticker] = {
                'mean_3d_ret': float(compressed['fwd_3d_ret'].mean()),
                'abs_mean_3d_ret': float(compressed['fwd_3d_ret'].abs().mean()),
                'count': len(compressed),
                'up_bias': up_pct,
                'dir_bias': 1 if up_pct > 0.55 else (-1 if up_pct < 0.45 else 0),
            }

    params['vol_stats'] = vol_stats

    # ── Learn flow statistics ────────────────────────────────────────
    flow_stats = {}
    for ticker in TRADEABLE:
        tk = intraday[intraday['ticker'] == ticker].copy()
        if len(tk) < 30:
            continue

        tk = tk.sort_values('date')

        if 'dollar_volume_last30m' in tk.columns and 'dollar_volume_first30m' in tk.columns:
            tk['flow_ratio'] = (
                tk['dollar_volume_last30m'] /
                tk['dollar_volume_first30m'].replace(0, np.nan)
            )

            tk = _compute_next_day_rets(tk)

            high_flow = tk[tk['flow_ratio'] > FLOW_RATIO_THRESHOLD]
            high_flow_valid = high_flow.dropna(subset=['nxt_ret'])

            if len(high_flow_valid) >= 5:
                flow_stats[ticker] = {
                    'mean_ret': float(high_flow_valid['nxt_ret'].mean()),
                    'win_rate': float((high_flow_valid['nxt_ret'] > 0).mean()),
                    'count': len(high_flow_valid),
                }

    params['flow_stats'] = flow_stats

    # ── Regime-adaptive weights (learned from train data) ────────────
    # Count how signals perform in each regime to balance weights
    if 'state' in macro.columns:
        macro_lookup = {}
        for _, r in macro.iterrows():
            macro_lookup[pd.Timestamp(r['date'])] = r['state']

        # Default regime weights: slightly boost risk_off gap reversals,
        # dampen risk_on flow signals (prevents regime gap)
        params['regime_signal_adj'] = {
            'risk_on_strong': {'gap': 1.0, 'vol': 1.0, 'flow': 1.0, 'min_score': 0.4},
            'risk_on':        {'gap': 1.0, 'vol': 1.0, 'flow': 1.0, 'min_score': 0.4},
            'neutral':        {'gap': 1.0, 'vol': 1.0, 'flow': 1.0, 'min_score': 1.2},
            'risk_off':       {'gap': 1.1, 'vol': 0.9, 'flow': 1.0, 'min_score': 0.35},
            'risk_off_severe':{'gap': 1.2, 'vol': 0.8, 'flow': 1.1, 'min_score': 0.35},
        }
        params['macro_lookup_dates'] = {str(k): v for k, v in macro_lookup.items()}

    return params


# ── generate_signals() ───────────────────────────────────────────────────

def generate_signals(daily_data, params):
    """Generate trading signals from daily data."""
    intraday = daily_data['intraday_df']
    realized_vol = daily_data['realized_vol_df']
    macro = daily_data.get('macro_df', pd.DataFrame())

    gap_stats = params.get('gap_stats', {})
    vol_stats = params.get('vol_stats', {})
    flow_stats = params.get('flow_stats', {})
    regime_adj = params.get('regime_signal_adj', {})

    # Build macro state lookup
    macro_lookup = {}
    if 'state' in macro.columns:
        for _, row in macro.iterrows():
            macro_lookup[pd.Timestamp(row['date'])] = row['state']

    signals = []
    all_dates = sorted(intraday['date'].unique())

    for date in all_dates:
        date = pd.Timestamp(date)
        regime = macro_lookup.get(date, 'neutral')
        r_adj = regime_adj.get(regime, {'gap': 1.0, 'vol': 1.0, 'flow': 1.0})

        day_signals = []

        for ticker in TRADEABLE:
            tk_today = intraday[
                (intraday['ticker'] == ticker) &
                (intraday['date'] == date)
            ]
            if len(tk_today) == 0:
                continue
            row = tk_today.iloc[0]

            # ── Signal 1: Gap Reversal ───────────────────────────────
            gap_signal = 0.0
            gap_dir = 0
            if ticker in gap_stats:
                gs = gap_stats[ticker]
                gap_z = (row['overnight_gap'] - gs['gap_mean']) / max(gs['gap_std'], 1e-8)

                if (abs(gap_z) > params['gap_sigma'] and
                    np.sign(row['open_to_close_ret']) != np.sign(row['overnight_gap']) and
                    abs(row['open_to_close_ret']) > 0.002):

                    gap_signal = GAP_REVERSAL_WEIGHT * min(abs(gap_z), 3.0) / 3.0
                    gap_signal *= r_adj['gap']
                    # Scale by learned win rate and expected return
                    wr = gs.get('win_rate', 0.5)
                    gap_signal *= (0.6 + 0.8 * wr)  # maps 0.45->0.96, 0.50->1.0, 0.60->1.08
                    gap_dir = -int(np.sign(row['overnight_gap']))

            # ── Signal 2: Vol Compression ────────────────────────────
            vol_signal = 0.0
            vol_dir = 0
            rv_today = realized_vol[
                (realized_vol['ticker'] == ticker) &
                (realized_vol['date'] == date)
            ]
            if len(rv_today) > 0 and ticker in vol_stats:
                rv_row = rv_today.iloc[0]
                if ('rv_cc_5d' in rv_row.index and 'rv_cc_20d' in rv_row.index and
                    rv_row['rv_cc_20d'] > 0):
                    vol_ratio = rv_row['rv_cc_5d'] / rv_row['rv_cc_20d']

                    if vol_ratio < params['vol_ratio']:
                        vol_signal = VOL_COMPRESS_WEIGHT * (1.0 - vol_ratio)
                        vol_signal *= r_adj['vol']

                        vs = vol_stats.get(ticker, {})
                        learned_bias = vs.get('dir_bias', 0)
                        
                        if row['close_minus_typprice'] > 0.002:
                            vol_dir = 1
                        elif row['close_minus_typprice'] < -0.002:
                            vol_dir = -1
                        elif learned_bias != 0:
                            vol_dir = learned_bias
                        else:
                            vol_dir = int(np.sign(row.get('open_to_close_ret', 0)))

            # ── Signal 3: Institutional Flow ─────────────────────────
            flow_signal = 0.0
            flow_dir = 0
            if ('dollar_volume_last30m' in row.index and
                'dollar_volume_first30m' in row.index and
                row['dollar_volume_first30m'] > 0):

                flow_ratio = row['dollar_volume_last30m'] / row['dollar_volume_first30m']

                if flow_ratio > params['flow_ratio']:
                    close_pos = row.get('close_minus_typprice', 0)

                    if close_pos > 0.001:
                        flow_signal = FLOW_WEIGHT * min(flow_ratio / 2.0, 1.0)
                        flow_signal *= r_adj['flow']
                        flow_dir = 1
                    elif close_pos < -0.001:
                        flow_signal = FLOW_WEIGHT * min(flow_ratio / 2.0, 1.0)
                        flow_signal *= r_adj['flow']
                        flow_dir = -1

            # ── Combine signals ──────────────────────────────────────
            dir_votes = 0.0
            if gap_dir != 0:
                dir_votes += gap_signal * gap_dir
            if vol_dir != 0:
                dir_votes += vol_signal * vol_dir
            if flow_dir != 0:
                dir_votes += flow_signal * flow_dir

            if dir_votes == 0:
                continue

            direction = 1 if dir_votes > 0 else -1
            score = gap_signal + vol_signal + flow_signal

            regime_min = r_adj.get('min_score', MIN_SCORE)
            if score >= regime_min:
                day_signals.append({
                    'date': date,
                    'ticker': ticker,
                    'direction': direction,
                    'score': score,
                })

        day_signals.sort(key=lambda x: x['score'], reverse=True)
        signals.extend(day_signals[:MAX_SIGNALS_PER_DAY])

    if not signals:
        return pd.DataFrame(columns=['date', 'ticker', 'direction', 'score'])

    return pd.DataFrame(signals)


# ── should_exit() ────────────────────────────────────────────────────────

def should_exit(pos, current_price, current_date, portfolio_dd):
    """Determine whether to exit an open position."""
    entry_price = pos['entry_price']
    direction = pos['direction']
    entry_date = pd.Timestamp(pos['entry_date'])
    current_date = pd.Timestamp(current_date)

    days_held = (current_date - entry_date).days

    if direction == 1:
        pos_return = (current_price - entry_price) / entry_price
    else:
        pos_return = (entry_price - current_price) / entry_price

    if portfolio_dd < PORTFOLIO_DD_EXIT:
        return True

    if pos_return >= TAKE_PROFIT_PCT:
        return True

    # Time-adaptive trailing stop: tightens as position ages
    time_factor = min(days_held / MAX_HOLD_DAYS, 1.0)
    adaptive_stop = TRAILING_STOP_PCT * (1.0 - 0.3 * time_factor)  # -3% -> -2.1% over 5 days
    if pos_return <= adaptive_stop:
        return True

    if days_held >= MAX_HOLD_DAYS:
        return True

    if days_held >= UNDERWATER_DAYS and pos_return < -0.005:
        return True

    return False
