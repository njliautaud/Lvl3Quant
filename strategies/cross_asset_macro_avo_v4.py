#!/usr/bin/env python3
"""
Cross-Asset Macro Sector Rotation -- v3
========================================
Enhanced defensive dip-buy with multi-layer macro filters.

Improvements from v2:
- Relative dip scoring: prefer sectors that dipped more than the all-sector average
  (sector-specific weakness vs broad market weakness = better bounce)
- Tiny underwater tolerance (-0.05%): allow nearly-flat trades to survive past day 1
  instead of cutting at exactly 0
"""

import numpy as np
import pandas as pd

# -- Configuration -----------------------------------------------------------
TRADEABLE_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE']

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

# Dip parameters
DIP_LOOKBACK = 5
DIP_THRESHOLD = -0.015

# Trend guard
TREND_LOOKBACK = 30
TREND_MIN = -0.10

# Bond filters
TLT_CRASH_LOOKBACK = 10
TLT_CRASH_THRESHOLD = -0.04
TLT_TREND_LOOKBACK = 30
TLT_DOWNTREND_THRESHOLD = -0.02

# VIX filters
VIX_MIN_ENTRY = 14.0
VIX_MA_LOOKBACK = 20
VIX_ELEVATED_RATIO = 1.10

# Dollar strength
UUP_LOOKBACK = 15
UUP_STRONG_THRESHOLD = 0.02

# Gold risk-off confirmation
GLD_RISKOFF_LOOKBACK = 11
GLD_RISKOFF_THRESHOLD = 0.01

# Dip deceleration
DAILY_DECEL_THRESHOLD = -0.012

# Trade management
MAX_HOLD_DAYS = 3
MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 1
SLIPPAGE_PCT = 0.0001

# Exit parameters
TRAILING_STOP_PCT = -0.02
TAKE_PROFIT_PCT = 0.035
UNDERWATER_EXIT_DAYS = 1
UNDERWATER_TOLERANCE = -0.0005  # allow nearly-flat trades to continue


def generate_signals(prices, spy, vix, macro_data):
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    # Compute dip returns for ALL sectors (for relative dip calculation)
    all_dip_ret = {}
    for etf in SECTOR_ETFS:
        if etf in prices.columns:
            all_dip_ret[etf] = prices[etf].pct_change(DIP_LOOKBACK)

    # Sector returns for tradeable sectors
    dip_ret = {}
    trend_ret = {}
    daily_ret = {}
    for etf in TRADEABLE_SECTORS:
        if etf in prices.columns:
            dip_ret[etf] = all_dip_ret[etf]
            trend_ret[etf] = prices[etf].pct_change(TREND_LOOKBACK)
            daily_ret[etf] = prices[etf].pct_change(1)

    # Cross-asset signals
    tlt = macro_data['TLT'] if 'TLT' in macro_data.columns else None
    tlt_crash_series = tlt.pct_change(TLT_CRASH_LOOKBACK) if tlt is not None else None
    tlt_trend_series = tlt.pct_change(TLT_TREND_LOOKBACK) if tlt is not None else None

    uup = macro_data['UUP'] if 'UUP' in macro_data.columns else None
    uup_ret_series = uup.pct_change(UUP_LOOKBACK) if uup is not None else None

    gld = macro_data['GLD'] if 'GLD' in macro_data.columns else None
    gld_ret_series = gld.pct_change(GLD_RISKOFF_LOOKBACK) if gld is not None else None

    vix_ma = vix.rolling(VIX_MA_LOOKBACK).mean() if vix is not None else None

    for date in prices.index:
        # VIX minimum gate: skip when too complacent
        if vix is not None and date in vix.index:
            vix_val = vix.get(date, np.nan)
            if not pd.isna(vix_val) and vix_val < VIX_MIN_ENTRY:
                continue

        # TLT crash filter: skip during rate shocks
        if tlt_crash_series is not None and date in tlt_crash_series.index:
            tlt_r = tlt_crash_series.get(date, 0.0)
            if not pd.isna(tlt_r) and tlt_r < TLT_CRASH_THRESHOLD:
                continue

        # Count dipping sectors and find deepest dip
        dipping_count = 0
        deepest_dip = 0.0
        for etf in TRADEABLE_SECTORS:
            if etf in dip_ret and date in dip_ret[etf].index:
                dv_check = dip_ret[etf].get(date, np.nan)
                if not pd.isna(dv_check) and dv_check <= DIP_THRESHOLD:
                    dipping_count += 1
                    deepest_dip = min(deepest_dip, dv_check)

        # VIX term structure check
        vix_elevated = False
        if vix_ma is not None and date in vix_ma.index and vix is not None:
            vix_val_ts = vix.get(date, np.nan)
            vix_ma_val = vix_ma.get(date, np.nan)
            if not pd.isna(vix_val_ts) and not pd.isna(vix_ma_val) and vix_ma_val > 0:
                vix_elevated = (vix_val_ts / vix_ma_val) > VIX_ELEVATED_RATIO

        # Dollar strength check
        uup_strong = False
        if uup_ret_series is not None and date in uup_ret_series.index:
            uup_r = uup_ret_series.get(date, 0.0)
            if not pd.isna(uup_r) and uup_r > UUP_STRONG_THRESHOLD:
                uup_strong = True

        # TLT downtrend check
        tlt_downtrend = False
        if tlt_trend_series is not None and date in tlt_trend_series.index:
            tlt_t = tlt_trend_series.get(date, 0.0)
            if not pd.isna(tlt_t) and tlt_t < TLT_DOWNTREND_THRESHOLD:
                tlt_downtrend = True

        # Gold risk-off confirmation
        gld_riskoff = False
        if gld_ret_series is not None and date in gld_ret_series.index:
            gld_r = gld_ret_series.get(date, 0.0)
            if not pd.isna(gld_r) and gld_r > GLD_RISKOFF_THRESHOLD:
                gld_riskoff = True

        # Breadth filter with adaptive thresholds
        if vix_elevated:
            if dipping_count < 3 and deepest_dip > -0.035:
                continue
        elif uup_strong or tlt_downtrend:
            if dipping_count < 3 and deepest_dip > -0.025:
                continue
        elif gld_riskoff:
            if dipping_count < 1 and deepest_dip > -0.025:
                continue
        else:
            if dipping_count < 2 and deepest_dip > -0.025:
                continue

        # Find best candidate
        candidates = []
        for etf in TRADEABLE_SECTORS:
            if etf not in dip_ret or date not in dip_ret[etf].index:
                continue
            dv = dip_ret[etf].get(date, np.nan)
            if pd.isna(dv) or dv > DIP_THRESHOLD:
                continue

            if etf not in trend_ret or date not in trend_ret[etf].index:
                continue
            tv = trend_ret[etf].get(date, np.nan)
            if pd.isna(tv) or tv < TREND_MIN:
                continue

            if etf in daily_ret and date in daily_ret[etf].index:
                dr = daily_ret[etf].get(date, 0.0)
                if not pd.isna(dr) and dr < DAILY_DECEL_THRESHOLD:
                    continue

            # Relative dip: how much this sector dipped vs average of all sectors
            avg_dip = 0.0
            n_sectors = 0
            for s in SECTOR_ETFS:
                if s in all_dip_ret and date in all_dip_ret[s].index:
                    sv = all_dip_ret[s].get(date, np.nan)
                    if not pd.isna(sv):
                        avg_dip += sv
                        n_sectors += 1
            if n_sectors > 0:
                avg_dip /= n_sectors
            rel_dip = dv - avg_dip  # negative = sector dipped more than average

            # Score: dip magnitude with trend bonus + relative dip bonus
            trend_bonus = max(0.0, tv + 0.05) * 10.0
            rel_bonus = max(0.0, abs(rel_dip) * 5.0) if rel_dip < 0 else 0.0
            score = abs(dv) * (1.0 + trend_bonus + rel_bonus)

            # Gold confirmation bonus
            if gld_riskoff:
                score *= 1.2

            candidates.append((etf, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for etf, _ in candidates[:MAX_CONCURRENT]:
            signals.loc[date, etf] = True

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
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

    # Cut underwater positions after 1 day (with tiny tolerance for nearly-flat)
    if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < UNDERWATER_TOLERANCE:
        return True
    if days_held >= MAX_HOLD_DAYS:
        return True
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True
    if drawdown_from_high <= TRAILING_STOP_PCT:
        return True

    return False
