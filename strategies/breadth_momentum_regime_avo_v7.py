#!/usr/bin/env python3
"""
Breadth Momentum Regime — v1 (Fix frequency + high-vol)
========================================================
Core fixes from seed:
1. Removed direction-flip requirement (was blocking consecutive same-direction trades)
2. During VIX>25, ONLY trade defensives regardless of breadth (offensives get destroyed in high-vol)
3. Reduced cooldown from 3 to 1 day
4. Widened breadth thresholds (deterioration <50%, thrust >55%) for more signals
5. Added 5d dip filter for defensives (require recent pullback for better entry)
6. Increased MAX_CONCURRENT to 3
"""

import numpy as np
import pandas as pd

# -- Configuration -----------------------------------------------------------
OFFENSIVE_SECTORS = ['XLK', 'XLF', 'XLI', 'XLE']  # buy on breadth thrust (low-vol only)
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']          # buy on breadth deterioration
HIGH_VOL_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE']  # expanded defensives for high-vol (XLRE = real estate)

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']

# Breadth parameters
RSP_SPY_LOOKBACK = 10
BREADTH_MA_SHORT = 20
BREADTH_MA_LONG = 50
BREADTH_THRUST_THRESHOLD = 0.60       # require clearer thrust signal
BREADTH_DETERIORATION_THRESHOLD = 0.50  # raised from 0.40 for more signals
RSP_SPY_THRESHOLD = 0.0

# VIX regime
VIX_HIGH_THRESHOLD = 25.0

# Trade management
MAX_HOLD_DAYS = 5
MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0005  # 5 bps realistic slippage

# Exit parameters
TRAILING_STOP_LOW_VOL = -0.012
TRAILING_STOP_HIGH_VOL = -0.025
TAKE_PROFIT_PCT = 0.03
UNDERWATER_EXIT_DAYS = 1  # cut losers after 1 day

# Dip filter for defensives
DEFENSIVE_DIP_LOOKBACK = 5
DEFENSIVE_DIP_THRESHOLD = -0.005  # require -0.5% dip in 5 days

# Cooldown
COOLDOWN_DAYS = 2  # 2-day cooldown balances frequency vs overtrading


def compute_breadth(prices):
    """Compute % of sector ETFs above their 20d and 50d SMAs."""
    breadth = pd.DataFrame(index=prices.index)
    above_20d = pd.DataFrame(index=prices.index)
    above_50d = pd.DataFrame(index=prices.index)

    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        p = prices[etf]
        sma20 = p.rolling(BREADTH_MA_SHORT).mean()
        sma50 = p.rolling(BREADTH_MA_LONG).mean()
        above_20d[etf] = (p > sma20).astype(float)
        above_50d[etf] = (p > sma50).astype(float)

    n_etfs_20 = above_20d.sum(axis=1)
    n_etfs_50 = above_50d.sum(axis=1)
    total = above_20d.count(axis=1).replace(0, 1)

    breadth['pct_above_20d'] = n_etfs_20 / total
    breadth['pct_above_50d'] = n_etfs_50 / total

    return breadth


def compute_rsp_spy_signal(rsp, spy):
    """Compute RSP/SPY ratio change as breadth proxy."""
    ratio = rsp / spy
    ratio_change = ratio.pct_change(RSP_SPY_LOOKBACK)
    return ratio_change


def generate_signals(prices, spy, vix, rsp=None, breadth=None):
    """Generate breadth momentum regime signals."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    if breadth is None or (hasattr(breadth, 'empty') and breadth.empty):
        breadth = compute_breadth(prices)

    if rsp is not None and not (hasattr(rsp, 'empty') and rsp.empty):
        rsp_spy_change = compute_rsp_spy_signal(rsp, spy)
    else:
        available = [e for e in SECTOR_ETFS if e in prices.columns]
        if available:
            rsp_proxy = prices[available].mean(axis=1)
            rsp_spy_change = compute_rsp_spy_signal(rsp_proxy, spy)
        else:
            return signals

    # Pre-compute dip returns for defensives
    dip_ret = {}
    for etf in DEFENSIVE_SECTORS:
        if etf in prices.columns:
            dip_ret[etf] = prices[etf].pct_change(DEFENSIVE_DIP_LOOKBACK)

    # Breadth momentum (rate of change in breadth)
    breadth_mom_50d = breadth['pct_above_50d'].diff(3)  # 3-day change in breadth

    # Breadth acceleration (2nd derivative): is momentum itself accelerating or decelerating?
    breadth_accel = breadth_mom_50d.diff(3)  # change in 3d momentum over 3 more days

    last_trade_date = None

    for date in prices.index:
        if date not in breadth.index:
            continue

        pct_50d = breadth.loc[date, 'pct_above_50d']
        pct_20d = breadth.loc[date, 'pct_above_20d']
        rsp_change = rsp_spy_change.get(date, np.nan)

        if pd.isna(pct_50d) or pd.isna(rsp_change):
            continue

        # Breadth momentum (is breadth improving or deteriorating?)
        b_mom = breadth_mom_50d.get(date, 0.0) if date in breadth_mom_50d.index else 0.0
        if pd.isna(b_mom):
            b_mom = 0.0

        # Cooldown
        if last_trade_date is not None:
            days_since = (date - last_trade_date).days
            if days_since < COOLDOWN_DAYS:
                continue

        # Get current VIX
        vix_val = vix.get(date, 20.0) if date in vix.index else 20.0
        if pd.isna(vix_val):
            vix_val = 20.0
        high_vol = vix_val > VIX_HIGH_THRESHOLD

        # During HIGH VOL: only trade defensives (offensives get crushed)
        # Stricter entry: require BOTH breadth weakness AND RSP underperformance
        if high_vol:
            if pct_50d < 0.45 and rsp_change < -0.003:
                candidates = HIGH_VOL_SECTORS
                direction = 'defensive'
            else:
                continue
        else:
            # Low vol: use breadth signals for direction
            # Require stronger RSP signal (±0.005) to filter noise
            if pct_50d < BREADTH_DETERIORATION_THRESHOLD and rsp_change < -0.005:
                candidates = DEFENSIVE_SECTORS
                direction = 'defensive'
            elif pct_50d > BREADTH_THRUST_THRESHOLD and rsp_change > 0.005:
                # Quality gate: breadth should still be improving, not flat/declining
                if b_mom < -0.05:
                    continue  # breadth declining despite level being high — skip

                # Acceleration gate: skip offensive when breadth momentum is decelerating
                b_accel = breadth_accel.get(date, 0.0) if date in breadth_accel.index else 0.0
                if pd.isna(b_accel):
                    b_accel = 0.0
                if b_accel < -0.08:
                    continue  # momentum decelerating — thrust is fading

                candidates = OFFENSIVE_SECTORS
                direction = 'offensive'
            else:
                continue

        # Pick best sector by 5-day relative strength within candidates
        best_etf = None
        best_rs = -np.inf
        idx = prices.index.get_loc(date)
        if idx < 5:
            continue

        for etf in candidates:
            if etf not in prices.columns:
                continue

            # For defensives in low-vol, require a small dip (better entry)
            # In high-vol, skip dip filter — breadth signal alone is sufficient
            if direction == 'defensive' and not high_vol and etf in dip_ret:
                dv = dip_ret[etf].get(date, 0.0)
                if pd.isna(dv):
                    dv = 0.0
                if dv > DEFENSIVE_DIP_THRESHOLD:
                    continue

            etf_ret = prices[etf].iloc[idx] / prices[etf].iloc[idx - 5] - 1
            spy_ret = spy.iloc[idx] / spy.iloc[idx - 5] - 1
            rs = etf_ret - spy_ret
            if rs > best_rs:
                best_rs = rs
                best_etf = etf

        if best_etf is not None:
            signals.loc[date, best_etf] = True
            last_trade_date = date

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Direction-aware exit logic."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))

    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    # Determine if this is an offensive or defensive position
    ticker = pos.get('ticker', '')
    is_offensive = ticker in OFFENSIVE_SECTORS

    # Update high water mark
    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price
    high_water = pos.get(hwm_key, entry_price)
    drawdown_from_high = (current_price - high_water) / high_water

    # Detect high-vol from portfolio DD proxy
    is_stressed = abs(portfolio_dd) > 0.03

    # Max hold: offensive gets slightly longer to capture momentum
    if is_stressed:
        max_hold = MAX_HOLD_DAYS + 2
    elif is_offensive:
        max_hold = MAX_HOLD_DAYS + 1  # extra day for momentum trades
    else:
        max_hold = MAX_HOLD_DAYS
    if days_held >= max_hold:
        return True

    # Take profit
    if is_stressed:
        tp = 0.045
    elif is_offensive:
        tp = 0.025  # tighter TP for offensives — take quick profits
    else:
        tp = TAKE_PROFIT_PCT
    if pnl_pct >= tp:
        return True

    # Cut losers: offensive gets cut faster (momentum should work immediately)
    if is_offensive:
        # Offensive: if not green by day 1, signal was wrong
        if days_held >= 1 and pnl_pct < -0.006:
            return True
    else:
        # Defensive: more patient
        loss_cut = -0.008 if is_stressed else -0.005
        if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < loss_cut:
            return True

    # Trailing stop
    trailing_stop = TRAILING_STOP_LOW_VOL
    if is_stressed:
        trailing_stop = TRAILING_STOP_HIGH_VOL

    if drawdown_from_high <= trailing_stop:
        return True

    return False
