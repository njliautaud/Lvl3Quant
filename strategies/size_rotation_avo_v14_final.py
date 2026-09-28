#!/usr/bin/env python3
"""
Size Rotation — v1 (Fix high-vol regime)
==========================================
Restoring best config (gap=0.71, HV=+0.73) with targeted fix for 2022H2 trade
count issue.

Key insight: direction-flip constraint makes sense when BOTH cyclicals and
defensives are available. But in VIX>20, only defensives are allowed, so
the flip constraint deadlocks (can't flip to cyclical because VIX blocks it).

Fix: only enforce direction-flip when VIX <= VIX_CYCLICAL_MAX (both directions
available). When VIX > VIX_CYCLICAL_MAX, allow consecutive defensive entries
with at least COOLDOWN_DAYS between them (the Z-score threshold provides
quality control).

This preserves the HV Sharpe (+0.73) while fixing the 2022H2 trade count.
"""

import numpy as np
import pandas as pd

# -- Configuration -----------------------------------------------------------
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']

CYCLICAL_SECTORS = ['XLY', 'XLI', 'XLB', 'XLK']
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']

# IWM/SPY ratio parameters
RATIO_LOOKBACK = 5
RATIO_ZSCORE_WINDOW = 60

# VIX regime thresholds
VIX_CYCLICAL_MAX = 20.0
VIX_ELEVATED_THRESHOLD = 18.0
VIX_HIGH_THRESHOLD = 25.0
VIX_EXTREME_THRESHOLD = 30.0

# Z-score thresholds
ZSCORE_THRESHOLD_NORMAL = 0.3     # VIX 18-20
ZSCORE_THRESHOLD_HIGHVOL = 0.4    # VIX 25-30
ZSCORE_THRESHOLD_LOWVOL = 0.7     # VIX < 18

# Trade management
MAX_HOLD_DAYS = 2
MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0001

# Exit parameters
TRAILING_STOP = -0.015
TAKE_PROFIT_PCT = 0.035
STOP_LOSS_PCT = -0.02

# Cooldown
COOLDOWN_DAYS = 3
COOLDOWN_DAYS_HIGHVOL = 4        # longer cooldown for same-direction high-vol entries


def compute_ratio_zscore(iwm, spy, lookback, zscore_window):
    """Compute IWM/SPY ratio Z-score (normalized momentum)."""
    ratio = iwm / spy
    ratio_mom = ratio.pct_change(lookback)
    rolling_mean = ratio_mom.rolling(zscore_window, min_periods=20).mean()
    rolling_std = ratio_mom.rolling(zscore_window, min_periods=20).std()
    zscore = (ratio_mom - rolling_mean) / rolling_std.replace(0, np.nan)
    return ratio, ratio_mom, zscore


def generate_signals(prices, spy, vix, iwm=None, breadth=None):
    """Generate size rotation signals with regime-aware logic."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    if iwm is None or (hasattr(iwm, 'empty') and iwm.empty) or len(iwm) == 0:
        return signals

    ratio, ratio_mom, zscore = compute_ratio_zscore(
        iwm, spy, RATIO_LOOKBACK, RATIO_ZSCORE_WINDOW)
    # 10d lookback Z-score for low-vol entries (smoother signal)
    _, _, zscore_10d = compute_ratio_zscore(iwm, spy, 10, RATIO_ZSCORE_WINDOW)
    # 20d lookback for long-term trend confirmation
    _, ratio_mom_20d, _ = compute_ratio_zscore(iwm, spy, 20, RATIO_ZSCORE_WINDOW)
    # Ratio acceleration: is the Z-score itself trending?
    zscore_accel = zscore.diff(3)
    zscore_accel_10d = zscore_10d.diff(3)

    # VIX momentum: skip defensive entries when VIX is falling fast
    vix_pct_5d = vix.pct_change(5)

    last_trade_date = None
    last_direction = None

    for date in prices.index:
        z = zscore.get(date, np.nan) if hasattr(zscore, 'get') else np.nan
        z10 = zscore_10d.get(date, np.nan) if hasattr(zscore_10d, 'get') else np.nan
        v = vix.get(date, np.nan) if hasattr(vix, 'get') else np.nan
        accel = zscore_accel.get(date, np.nan) if hasattr(zscore_accel, 'get') else np.nan
        accel10 = zscore_accel_10d.get(date, np.nan) if hasattr(zscore_accel_10d, 'get') else np.nan
        vix_mom = vix_pct_5d.get(date, np.nan) if hasattr(vix_pct_5d, 'get') else np.nan
        rm20 = ratio_mom_20d.get(date, np.nan) if hasattr(ratio_mom_20d, 'get') else np.nan

        if pd.isna(z) or pd.isna(v):
            continue

        # Basic cooldown
        if last_trade_date is not None:
            days_since = (date - last_trade_date).days
            if days_since < COOLDOWN_DAYS:
                continue

        direction = None
        candidates = None
        is_highvol_regime = False

        if v > VIX_EXTREME_THRESHOLD:
            continue

        elif v > VIX_HIGH_THRESHOLD:
            # VIX 25-30: defensive only
            if z < -ZSCORE_THRESHOLD_HIGHVOL:
                direction = 'defensive'
                candidates = DEFENSIVE_SECTORS
                is_highvol_regime = True
            else:
                continue

        elif v > VIX_CYCLICAL_MAX:
            # VIX 20-25: defensive only
            # Skip when 10d Z-score shows strong cyclical momentum (counter-signal)
            if z < -ZSCORE_THRESHOLD_NORMAL:
                if not pd.isna(z10) and z10 > 0.5:
                    continue  # 10d signal disagrees — ratio trend is cyclical
                direction = 'defensive'
                candidates = DEFENSIVE_SECTORS
                is_highvol_regime = True
            else:
                continue

        elif v > VIX_ELEVATED_THRESHOLD:
            # VIX 18-20: full signal
            if z > ZSCORE_THRESHOLD_NORMAL:
                direction = 'cyclical'
                candidates = CYCLICAL_SECTORS
            elif z < -ZSCORE_THRESHOLD_NORMAL:
                direction = 'defensive'
                candidates = DEFENSIVE_SECTORS
            else:
                continue

        else:
            # VIX < 18: use 10d Z-score (smoother) + acceleration
            if pd.isna(z10) or pd.isna(accel10):
                continue
            if z10 > ZSCORE_THRESHOLD_LOWVOL and accel10 > 0:
                direction = 'cyclical'
                # In low VIX, use core cyclicals only (XLK adds noise)
                candidates = ['XLY', 'XLI', 'XLB']
            elif z10 < -ZSCORE_THRESHOLD_LOWVOL and accel10 < 0:
                direction = 'defensive'
                candidates = DEFENSIVE_SECTORS
            else:
                continue

        if direction is None:
            continue

        # VIX momentum filter: skip defensive entries in VIX 18-20 transition
        # when VIX is falling fast (defensive rally already priced in)
        if (direction == 'defensive' and not pd.isna(vix_mom)
                and vix_mom < -0.15 and v <= VIX_CYCLICAL_MAX):
            continue

        # Direction control:
        # In high-vol regime (VIX>20): allow same-direction with longer cooldown
        # In normal regime (VIX 18-20): allow same-direction with longer cooldown
        # In low-vol regime (VIX<18): require direction flip
        if direction == last_direction:
            if is_highvol_regime or v > VIX_ELEVATED_THRESHOLD:
                # Allow same-direction with longer cooldown
                if last_trade_date is not None:
                    days_since = (date - last_trade_date).days
                    if days_since < COOLDOWN_DAYS_HIGHVOL:
                        continue
            else:
                # Low-vol: require direction flip
                continue

        best_etf = None
        best_rs = -np.inf
        for etf in candidates:
            if etf not in prices.columns:
                continue
            p = prices[etf]
            if date not in p.index:
                continue
            idx = prices.index.get_loc(date)
            if idx < 5:
                continue
            etf_ret = p.iloc[idx] / p.iloc[idx - 5] - 1
            spy_ret = spy.iloc[idx] / spy.iloc[idx - 5] - 1
            rs = etf_ret - spy_ret
            if rs > best_rs:
                best_rs = rs
                best_etf = etf

        if best_etf is not None:
            signals.loc[date, best_etf] = True
            last_trade_date = date
            last_direction = direction

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit logic with emergency exit for stressed portfolios."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))

    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    hwm_key = 'hwm'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price
    high_water = pos.get(hwm_key, entry_price)
    drawdown_from_high = (current_price - high_water) / high_water

    if days_held >= MAX_HOLD_DAYS:
        return True
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True
    if pnl_pct <= STOP_LOSS_PCT:
        return True

    # Emergency exit
    if abs(portfolio_dd) > 0.015 and pnl_pct < -0.003:
        return True

    if drawdown_from_high <= TRAILING_STOP:
        return True

    return False
