#!/usr/bin/env python3
"""
ETF Flow Reversal Strategy -- v1 (Tight Exits + Signal Quality)
================================================================
Thesis: volume spikes during price dips = forced institutional selling → reversion.

v1 changes from seed:
  - Tighter exits: underwater cut at 1 day (proven from insider_momentum),
    max hold 3 days, trailing -2%, TP 3.5%
  - Dip deceleration filter: skip if today's dip is accelerating (ret_1d
    is much worse than threshold — catching a falling knife)
  - Trend guard: skip if 20d return is deeply negative
  - Score refinement: use ret_20d as trend bonus

This is the file AVO evolves.
"""

import numpy as np
import pandas as pd

# -- Configuration (EVOLVE THESE) --------------------------------------------
VOLUME_LOOKBACK = 20          # rolling window for volume mean/std
VOLUME_Z_THRESHOLD = 2.0     # z-score above which volume is "spiking"
PRICE_DIP_THRESHOLD = -0.01  # ret_1d must be below this
REL_STRENGTH_BONUS = True    # boost score if ETF had positive rel_strength before dip

# Quality filters
TREND_GUARD_RET20D = -0.12   # skip ETF if its 20d return below this (tighter downtrend filter)
DIP_DECEL_THRESHOLD = -0.035 # skip if ret_1d is too extreme (tighter falling knife filter)


MAX_PER_TRADE = 2000.0       # dollars per position
MAX_CONCURRENT = 2           # max simultaneous positions
SLIPPAGE_PCT = 0.0001        # 0.01% each way

# Exit parameters — PROVEN TIGHT EXITS from insider_momentum research
TRAILING_STOP_PCT = -0.02    # -2% trailing stop
TAKE_PROFIT_PCT = 0.035      # +3.5% take profit
MAX_HOLD_DAYS = 3            # max hold (3 days, not 5)
UNDERWATER_CUT_DAYS = 1      # cut losers after 1 day (proven winning continuation)
UNDERWATER_TOLERANCE = -0.004  # allow up to -0.4% before cutting


def fit(train_data):
    """
    Analyze volume patterns in the training window to learn which ETFs show
    the strongest mean-reversion after forced selling events.
    """
    etfs = train_data['etf'].unique()
    etf_stats = {}

    for etf in etfs:
        edf = train_data[train_data['etf'] == etf].sort_values('date').copy()
        if len(edf) < VOLUME_LOOKBACK + 10:
            continue

        edf['vol_mean'] = edf['volume'].rolling(VOLUME_LOOKBACK, min_periods=10).mean()
        edf['vol_std'] = edf['volume'].rolling(VOLUME_LOOKBACK, min_periods=10).std()
        edf['vol_z'] = (edf['volume'] - edf['vol_mean']) / edf['vol_std'].clip(lower=1)

        spike_mask = (edf['vol_z'] > VOLUME_Z_THRESHOLD) & (edf['ret_1d'] < PRICE_DIP_THRESHOLD)
        spike_days = edf[spike_mask].index

        if len(spike_days) < 2:
            continue

        fwd_returns = []
        close_vals = edf['close'].values
        dates_arr = edf.index.values

        for idx in spike_days:
            pos = np.searchsorted(dates_arr, idx)
            if pos + 5 < len(close_vals):
                fwd_5d = (close_vals[pos + 5] - close_vals[pos]) / close_vals[pos]
                fwd_returns.append(fwd_5d)

        if fwd_returns:
            avg_fwd_return = np.mean(fwd_returns)
            hit_rate = np.mean([r > 0 for r in fwd_returns])
            n_events = len(fwd_returns)
        else:
            avg_fwd_return = 0.0
            hit_rate = 0.0
            n_events = 0

        etf_stats[etf] = {
            'avg_fwd_5d_return': avg_fwd_return,
            'hit_rate': hit_rate,
            'n_events': n_events,
        }

    reversal_scores = {}
    for etf, stats in etf_stats.items():
        if stats['n_events'] >= 2:
            reversal_scores[etf] = stats['avg_fwd_5d_return'] * stats['hit_rate']
        else:
            reversal_scores[etf] = 0.0

    return {
        'etf_stats': etf_stats,
        'reversal_scores': reversal_scores,
        'tradeable_etfs': [e for e, s in reversal_scores.items() if s > 0],
    }


def generate_signals(data, params):
    """
    Generate entry signals: volume spike + price dip with quality filters.
    """
    tradeable = params.get('tradeable_etfs', [])
    reversal_scores = params.get('reversal_scores', {})

    if not tradeable:
        tradeable = data['etf'].unique().tolist()

    signals = []
    etfs = data['etf'].unique()

    for etf in etfs:
        if etf not in tradeable:
            continue

        edf = data[data['etf'] == etf].sort_values('date').copy()
        if len(edf) < VOLUME_LOOKBACK:
            continue

        edf['vol_mean'] = edf['volume'].rolling(VOLUME_LOOKBACK, min_periods=10).mean()
        edf['vol_std'] = edf['volume'].rolling(VOLUME_LOOKBACK, min_periods=10).std()
        edf['vol_z'] = (edf['volume'] - edf['vol_mean']) / edf['vol_std'].clip(lower=1)

        # Multi-day volume: avg vol_z over last 2 days (more reliable than single day)
        edf['vol_z_2d'] = edf['vol_z'].rolling(2, min_periods=1).mean()

        last_signal_date = None
        for idx, row in edf.iterrows():
            vol_z = row.get('vol_z', 0)
            vol_z_2d = row.get('vol_z_2d', 0)
            ret_1d = row.get('ret_1d', 0)
            ret_20d = row.get('ret_20d', 0)
            ret_60d = row.get('ret_60d', 0)
            rel_strength = row.get('rel_strength_spy', 0)

            if pd.isna(vol_z) or pd.isna(ret_1d):
                continue

            # 2-day cooldown per ETF: skip if same ETF fired yesterday
            sig_date = pd.Timestamp(row['date'])
            if last_signal_date is not None:
                gap = np.busday_count(
                    np.datetime64(last_signal_date, 'D'),
                    np.datetime64(sig_date, 'D')
                )
                if gap < 2:
                    continue

            # Trend guard: skip if ETF is in a real downtrend
            # Wider in bear markets (ret_20d is naturally more negative)
            trend_guard = -0.16 if (not pd.isna(ret_60d) and ret_60d < -0.03) else TREND_GUARD_RET20D
            if not pd.isna(ret_20d) and ret_20d < trend_guard:
                continue

            # Dip deceleration: skip extreme single-day drops (catching falling knife)
            if ret_1d < DIP_DECEL_THRESHOLD:
                continue

            # Relative strength gate: skip if ETF is deeply underperforming SPY
            # Wider gate in bear markets (many ETFs naturally underperform SPY)
            rs_gate = -0.12 if (not pd.isna(ret_60d) and ret_60d < -0.03) else -0.08
            if not pd.isna(rel_strength) and rel_strength < rs_gate:
                continue

            # Core signal: volume spike + price dip
            # Use 2-day average vol_z for more stable detection
            effective_vol_z = max(vol_z, vol_z_2d) if not pd.isna(vol_z_2d) else vol_z

            # Adaptive volume threshold: in strong bull markets, require a bigger
            # volume spike.  Normal dips in uptrends are profit-taking / rotation,
            # not forced selling.  Only the most extreme spikes revert reliably.
            vol_threshold = VOLUME_Z_THRESHOLD
            if not pd.isna(ret_60d) and ret_60d > 0.08:
                vol_threshold = 2.3  # strong bull: require bigger spike
            elif not pd.isna(ret_60d) and ret_60d > 0.03:
                vol_threshold = 2.1  # mild bull: slightly higher bar for profit-taking noise
            elif not pd.isna(ret_60d) and ret_60d < -0.03:
                vol_threshold = 1.8  # bear: lower bar (forced selling more common)

            if effective_vol_z > vol_threshold and ret_1d < PRICE_DIP_THRESHOLD:
                # Score = volume intensity * dip magnitude
                score = effective_vol_z * abs(ret_1d)

                # 60-day momentum bonus: ETF with strong longer-term trend
                # dipping on volume is more likely forced selling (not trend breakdown)
                if not pd.isna(ret_60d) and ret_60d > 0:
                    long_trend_bonus = min(ret_60d * 7.0, 1.0)  # cap at 1.0
                    score *= (1.0 + long_trend_bonus)
                elif not pd.isna(ret_20d) and ret_20d > 0:
                    # Fallback to 20d if 60d not available
                    trend_bonus = max(0.0, ret_20d + 0.05) * 10.0
                    score *= (1.0 + trend_bonus)

                if REL_STRENGTH_BONUS and not pd.isna(rel_strength):
                    if rel_strength > 0:
                        score *= (1.0 + min(rel_strength, 0.5))
                    else:
                        # Penalize ETFs underperforming SPY (dip may be fundamental)
                        score *= max(0.2, 1.0 + rel_strength * 4.0)

                rev_score = reversal_scores.get(etf, 0.5)
                score *= (1.0 + max(rev_score, 0))

                signals.append({
                    'date': row['date'],
                    'etf': etf,
                    'score': score,
                })
                last_signal_date = sig_date

    if not signals:
        return pd.DataFrame(columns=['date', 'etf', 'score'])

    return pd.DataFrame(signals)


def should_exit(pos, current_price, current_date, portfolio_dd):
    """
    Tight exit rules — proven from insider_momentum research.
    Key: cut losers after 1 day (winning continuation filter).
    """
    entry_price = pos['entry_price_adj']
    high_water = pos.get('high_water_mark', entry_price)
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    pnl_pct = (current_price - entry_price) / entry_price

    # 0. Portfolio drawdown defense: if portfolio is losing, cut faster
    if portfolio_dd is not None and portfolio_dd < -0.01:
        # In portfolio drawdown: exit any position that's not solidly green
        if pnl_pct < 0.003:
            return True

    # 1. Winning continuation: cut losers after 1 day (with small tolerance)
    if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < UNDERWATER_TOLERANCE:
        return True

    # 2. Max hold days
    if days_held >= MAX_HOLD_DAYS:
        return True

    # 3. Take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True

    # 4. Trailing stop from high water mark
    if high_water > 0:
        drawdown_from_high = (current_price - high_water) / high_water
        if drawdown_from_high <= TRAILING_STOP_PCT:
            return True

    return False
