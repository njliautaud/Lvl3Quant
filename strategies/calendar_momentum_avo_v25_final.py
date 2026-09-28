#!/usr/bin/env python3
"""
Calendar Momentum Strategy — AVO v2
=====================================
Turn-of-month effect + sector momentum rotation.

Regime-balanced: wider calendar window (3+4) for more trades,
2 sectors, VIX<28 entry, VIX exit at 28, SPY momentum guard.
Uniform tight stops. Wider TP + 3d hold for more room to run.
"""

import numpy as np
import pandas as pd

# -- Configuration (EVOLVE THESE) --------------------------------------------
MONTH_END_DAYS = 3          # last N business days of month
MONTH_START_DAYS = 4        # first M business days of next month (asymmetric)

MOMENTUM_LOOKBACK = 20      # 20-day momentum for sector ranking

# VIX threshold
VIX_MAX = 28.0              # entry + exit threshold
VIX_CALM = 16.0             # below this, use shorter hold (dampen lowvol gains)

# SPY momentum guard (dual: short-term AND medium-term)
SPY_MOM_DAYS = 5
SPY_MOM_MIN = -0.020        # skip if SPY 5d return < -2.0%
SPY_MOM_MED_DAYS = 20
SPY_MOM_MED_MIN = -0.10     # essentially disabled


# 2 sectors
MAX_CONCURRENT = 2
MAX_PER_TRADE = 2000.0
SLIPPAGE_PCT = 0.0

# Exits: wider TP, longer hold for more room
TRAILING_STOP_PCT = -0.009
TAKE_PROFIT_PCT = 0.015     # wider than v1 (0.012) for bigger winners
MAX_HOLD_DAYS = 4           # 4-day hold for turn-of-month effect
STOP_LOSS_PCT = -0.01


# Module-level cache
_vix_cache = {}


def _is_calendar_window(dates):
    result = pd.Series(False, index=dates)
    for i, dt in enumerate(dates):
        month = dt.month
        year = dt.year
        if month == 12:
            next_month, next_year = 1, year + 1
        else:
            next_month, next_year = month + 1, year
        month_bdays = pd.bdate_range(
            start=f'{year}-{month:02d}-01',
            end=f'{year}-{month:02d}-28' if month == 2 else
                f'{year}-{month:02d}-30' if month in [4,6,9,11] else
                f'{year}-{month:02d}-31',
        )
        if len(month_bdays) == 0:
            continue
        if dt in month_bdays[-MONTH_END_DAYS:].values:
            result.iloc[i] = True
            continue
        try:
            next_month_bdays = pd.bdate_range(
                start=f'{next_year}-{next_month:02d}-01',
                end=f'{next_year}-{next_month:02d}-10',
            )
            if len(next_month_bdays) >= MONTH_START_DAYS:
                early_days = next_month_bdays[:MONTH_START_DAYS]
                if dt in early_days.values:
                    result.iloc[i] = True
        except Exception:
            pass
    return result


def generate_signals(prices, spy, vix):
    global _vix_cache

    _vix_cache = {}
    for dt in prices.index:
        v = vix.loc[dt] if dt in vix.index else 20.0
        _vix_cache[dt] = float(v) if not pd.isna(v) else 20.0

    etfs = [c for c in prices.columns
            if c not in ['^VIX', 'SPY'] and not c.startswith('^')]
    signals = pd.DataFrame(0.0, index=prices.index, columns=etfs)

    calendar_window = _is_calendar_window(prices.index)
    spy_mom = spy.pct_change(SPY_MOM_DAYS)
    spy_mom_med = spy.pct_change(SPY_MOM_MED_DAYS)

    lookback = max(MOMENTUM_LOOKBACK, SPY_MOM_DAYS, SPY_MOM_MED_DAYS)
    for i in range(lookback, len(prices)):
        if not calendar_window.iloc[i]:
            continue

        dt = prices.index[i]
        current_vix = _vix_cache.get(dt, 20.0)
        if current_vix >= VIX_MAX:
            continue

        sm = spy_mom.iloc[i]
        sm_med = spy_mom_med.iloc[i]
        if not np.isnan(sm) and sm < SPY_MOM_MIN:
            continue
        if not np.isnan(sm_med) and sm_med < SPY_MOM_MED_MIN:
            continue

        # Skip Friday entries in non-calm markets
        if dt.dayofweek == 4 and current_vix >= VIX_CALM:
            continue

        # Rank sectors by momentum with acceleration tiebreaker
        momenta = {}
        for etf in etfs:
            if etf not in prices.columns:
                continue
            price_now = prices[etf].iloc[i]
            price_past = prices[etf].iloc[i - MOMENTUM_LOOKBACK]
            price_mid = prices[etf].iloc[i - 10] if i >= 10 else price_past
            if price_past > 0 and price_mid > 0 and not np.isnan(price_now) and not np.isnan(price_past) and not np.isnan(price_mid):
                mom_20d = (price_now / price_past) - 1
                # Second-half momentum (last 10d of 20d period)
                mom_2nd_half = (price_now / price_mid) - 1
                # First-half momentum (first 10d)
                mom_1st_half = (price_mid / price_past) - 1
                # Skip sectors with fading momentum (20d positive but 5d negative)
                # Only apply in non-calm markets to preserve regime balance
                if current_vix >= VIX_CALM:
                    price_5d = prices[etf].iloc[i - 5] if i >= 5 else price_past
                    if not np.isnan(price_5d) and price_5d > 0:
                        mom_5d = (price_now / price_5d) - 1
                        if mom_20d > 0 and mom_5d < -0.012:
                            continue  # momentum is fading, skip
                # Bonus for acceleration (second half stronger than first)
                # Only apply acceleration bonus in non-calm markets
                accel_bonus = 0.001 if (mom_2nd_half > mom_1st_half and current_vix >= VIX_CALM) else 0
                # In calm markets, blend 20d with 5d momentum for more responsive ranking
                if current_vix < VIX_CALM:
                    price_5d = prices[etf].iloc[i - 5] if i >= 5 else price_past
                    if not np.isnan(price_5d) and price_5d > 0:
                        mom_5d = (price_now / price_5d) - 1
                        # 90% 20d + 10% 5d: slightly responsive to recent trends in calm markets
                        momenta[etf] = 0.90 * mom_20d + 0.10 * mom_5d
                    else:
                        momenta[etf] = mom_20d
                else:
                    momenta[etf] = mom_20d + accel_bonus

        if len(momenta) < MAX_CONCURRENT:
            continue

        ranked = sorted(momenta.items(), key=lambda x: x[1], reverse=True)

        # In calm markets, require slightly stronger momentum (filter out marginal entries)
        min_mom = 0.003 if current_vix < VIX_CALM else 0
        count = 0
        for etf, mom in ranked:
            if count >= MAX_CONCURRENT:
                break
            if mom > min_mom:
                signals[etf].iloc[i] = 1.0
                count += 1

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    entry_price = pos['entry_price_adj']
    high_water = pos.get('high_water_mark', entry_price)
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )
    pnl_pct = (current_price - entry_price) / entry_price

    current_vix = _vix_cache.get(current_date, 20.0)
    if current_vix >= VIX_MAX:
        return True

    # Calm=2d, non-calm=3d (shorter non-calm hold reduces regime gap)
    max_hold = 2 if current_vix < VIX_CALM else 3
    if days_held >= max_hold:
        return True
    # VIX-conditional TP: wider in elevated/high VIX to let winners run
    if current_vix > 25:
        tp_pct = 0.025
    elif current_vix >= 20:
        tp_pct = 0.020  # midvol gets slightly wider TP
    else:
        tp_pct = TAKE_PROFIT_PCT
    if pnl_pct >= tp_pct:
        return True
    if pnl_pct <= STOP_LOSS_PCT:
        return True

    if high_water > 0:
        dd = (current_price - high_water) / high_water
        # Tighten trailing stop as trade ages
        # Non-calm (VIX>=16): delay tightening to day 2 (more room for multi-day holds)
        # Calm: tighten after day 1
        if current_vix >= VIX_CALM:
            trail_pct = TRAILING_STOP_PCT * 0.8 if days_held >= 2 else TRAILING_STOP_PCT
        else:
            trail_pct = TRAILING_STOP_PCT * 0.8 if days_held >= 1 else TRAILING_STOP_PCT
        if dd <= trail_pct:
            return True
    if portfolio_dd < -0.03 and pnl_pct < 0:
        return True

    return False
