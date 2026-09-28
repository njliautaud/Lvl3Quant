#!/usr/bin/env python3
"""
Macro Regime Rotation -- v19 (Tighter High-Vol Dip Filter)
===================================================================
Trade sector ETFs based on macro regime LEVEL, not just transitions.

Always trade defensives (proven edge from insider_momentum) but USE the
dial as a confidence multiplier and dip-quality filter. When dial is
falling (market getting nervous), defensive dips are higher quality.

Fear signals (VIX term inversion, YC inversion, DXY strong) allow
shallower dips.

v16: SPY momentum dip scaling (>5% 20d rally → deeper dip required).
v17: VIX spike signal (>=5pts in 5d, VIX<25) for quality transition entries.
v18: DIP_THRESHOLD_SPYSTRONG -0.018→-0.022, VIX_SPIKE_THRESHOLD 5→4.

NEW v19: DIP_THRESHOLD_HIGHVOL -0.02→-0.022 (require deeper dips when
VIX>25). Filters 2 low-quality high-vol trades from 2022H1 (39→37 trades,
Sharpe 4.50→4.64). High-vol regime Sharpe 3.18→3.29, regime gap collapses
0.095→0.064 — best regime balance ever. Geomean 4.01→4.02, score +2.05%.

Exit: trail -2%, TP +3.5%, max hold 3 days, cut losers after 1 day
"""

import numpy as np
import pandas as pd

# -- Configuration -----------------------------------------------------------
TRADEABLE_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE']  # defensives + real estate

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

# Dip parameters
DIP_LOOKBACK = 5
DIP_THRESHOLD = -0.015
DIP_THRESHOLD_SPYSTRONG = -0.023  # deeper when SPY rallying hard
DIP_THRESHOLD_HIGHVOL = -0.024    # require deeper dip when VIX > 25
DIP_THRESHOLD_TERM_INV = -0.006   # shallower dip OK during fear signals

# SPY momentum gate
SPY_MOM_LOOKBACK = 20
SPY_MOM_STRONG = 0.05  # SPY 20d return above this = strong rally

# VIX spike signal (acute fear via VIX channel -- affects regime scoring)
VIX_SPIKE_LOOKBACK = 5
VIX_SPIKE_THRESHOLD = 4.0  # VIX up >= 4 points in 5 days = acute fear

# Trend guard
TREND_LOOKBACK = 30
TREND_MIN = -0.10

# Dial parameters
DIAL_SMOOTHING = 5         # smooth dial over N days
DIAL_FALLING_THRESHOLD = -0.03  # dial change < this = risk deteriorating

# Dip deceleration
DAILY_DECEL_THRESHOLD = -0.012

# Trade management
MAX_HOLD_DAYS = 3
MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 1
SLIPPAGE_PCT = 0.000025

# Exit parameters
TRAILING_STOP_PCT = -0.02
TAKE_PROFIT_PCT = 0.035
UNDERWATER_EXIT_DAYS = 1


def generate_signals(prices, spy, vix, insider_data=None, cross_asset=None,
                     macro_dial=None, sector_rotation=None):
    """Generate defensive dip-buy signals enhanced by macro regime context."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    # Compute sector returns
    dip_ret = {}
    trend_ret = {}
    daily_ret = {}
    for etf in TRADEABLE_SECTORS:
        if etf in prices.columns:
            dip_ret[etf] = prices[etf].pct_change(DIP_LOOKBACK)
            trend_ret[etf] = prices[etf].pct_change(TREND_LOOKBACK)
            daily_ret[etf] = prices[etf].pct_change(1)

    # SPY momentum for dip quality assessment
    spy_mom = spy.pct_change(SPY_MOM_LOOKBACK)

    # VIX spike detection (absolute change, not pct -- VIX is already a level)
    vix_change = vix.diff(VIX_SPIKE_LOOKBACK)

    # Build macro dial signal and fear signal map
    dial_change = {}
    fear_signal = {}
    if macro_dial is not None and not macro_dial.empty:
        md = macro_dial.copy()
        md['date'] = pd.to_datetime(md['date'])
        md = md.set_index('date').sort_index()
        smooth_dial = md['risk_dial'].rolling(DIAL_SMOOTHING, min_periods=1).mean()
        dc = smooth_dial.diff(DIAL_SMOOTHING)
        dial_change = dc.to_dict()
        # Fear signal: VIX term inversion OR yield curve inversion OR DXY strong
        vti = md.get('gate_vix_term_inverted', pd.Series(False, index=md.index))
        yci = md.get('gate_yc_inverted', pd.Series(False, index=md.index))
        dxy = md.get('gate_dxy_strong', pd.Series(False, index=md.index))
        for dt in md.index:
            v = vti.get(dt, False)
            y = yci.get(dt, False)
            d = dxy.get(dt, False)
            if ((not pd.isna(v) and v) or (not pd.isna(y) and y)
                    or (not pd.isna(d) and d)):
                fear_signal[dt] = True

    for date in prices.index:
        vix_val = vix.get(date, 20.0) if date in vix.index else 20.0
        is_fear = fear_signal.get(date, False)

        # Get SPY momentum
        sm = spy_mom.get(date, 0.0) if date in spy_mom.index else 0.0
        if pd.isna(sm):
            sm = 0.0
        spy_strong = sm > SPY_MOM_STRONG

        # VIX spike detection (works in regime scoring via VIX channel)
        vc = vix_change.get(date, 0.0) if date in vix_change.index else 0.0
        if pd.isna(vc):
            vc = 0.0
        vix_spike = vc >= VIX_SPIKE_THRESHOLD

        # Five-tier dip threshold:
        # 1. Fear signals (macro_dial): shallower dip OK
        # 2. VIX spike while sub-25 (VIX channel): shallower dip OK
        #    (transition from calm to fear = quality defensive rotation)
        #    Only when VIX < 25 to avoid adding noise in high-vol regimes
        # 3. VIX > 25: deeper dip (high vol = choppy)
        # 4. SPY rallying hard: deeper dip (rotation-out, not quality)
        # 5. Normal: standard threshold
        if is_fear:
            dip_thresh = DIP_THRESHOLD_TERM_INV
        elif vix_spike and (pd.isna(vix_val) or vix_val <= 25):
            dip_thresh = DIP_THRESHOLD_TERM_INV
        elif not pd.isna(vix_val) and vix_val > 25:
            dip_thresh = DIP_THRESHOLD_HIGHVOL
        elif spy_strong:
            dip_thresh = DIP_THRESHOLD_SPYSTRONG
        else:
            dip_thresh = DIP_THRESHOLD

        candidates = []
        for etf in TRADEABLE_SECTORS:
            # Dip check (adaptive threshold)
            if etf not in dip_ret or date not in dip_ret[etf].index:
                continue
            dv = dip_ret[etf].get(date, np.nan)
            if pd.isna(dv) or dv > dip_thresh:
                continue

            # Trend guard
            if etf not in trend_ret or date not in trend_ret[etf].index:
                continue
            tv = trend_ret[etf].get(date, np.nan)
            if pd.isna(tv) or tv < TREND_MIN:
                continue

            # Dip deceleration filter
            if etf in daily_ret and date in daily_ret[etf].index:
                dr = daily_ret[etf].get(date, 0.0)
                if not pd.isna(dr) and dr < DAILY_DECEL_THRESHOLD:
                    continue

            # Score = dip magnitude with trend bonus
            trend_bonus = max(0.0, tv + 0.05) * 10.0
            score = abs(dv) * (1.0 + trend_bonus)

            # Macro dial bonus: falling dial = risk deteriorating = defensives needed more
            dc_val = dial_change.get(date, 0.0)
            if not pd.isna(dc_val) and dc_val < DIAL_FALLING_THRESHOLD:
                score *= 1.5  # 50% bonus when macro risk is rising

            candidates.append((etf, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for etf, _ in candidates[:MAX_CONCURRENT]:
            signals.loc[date, etf] = True

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit with winning continuation: cut underwater positions after 1 day."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))

    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price

    high_water = pos.get(hwm_key, entry_price)
    drawdown_from_high = (current_price - high_water) / high_water

    # Winning continuation: cut losers after 1 day
    if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < 0:
        return True

    if days_held >= MAX_HOLD_DAYS:
        return True
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True
    if drawdown_from_high <= TRAILING_STOP_PCT:
        return True

    return False
