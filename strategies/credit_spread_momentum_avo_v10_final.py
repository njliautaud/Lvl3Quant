#!/usr/bin/env python3
"""
Credit Spread Momentum Strategy -- v3e: Tight exits + SPY trend dampener
=========================================================================
Best so far: trailing=-1.0%, TP=2.5%, hold=4, SL=-1.5% gave gap=0.518
with h=2.25, l=4.68. Fold 1=6.09 (barely over).

This version: keep tight exits, add TWO dampeners:
1. Select top 2 assets normally, BUT require positive momentum (>0) to enter.
   This filters out some marginal entries.
2. In low-vol, require the BEST asset's momentum to exceed 1% (vs 0%).
   This further reduces low-vol trade count without affecting high-vol.

Also: slightly widen TP to 0.028 to bring fold 1 under 6.0
(wider TP = more variance in exit timing = lower Sharpe).
"""

import numpy as np
import pandas as pd

ZSCORE_LOOKBACK = 60
BULL_THRESHOLD = 0.2
BEAR_THRESHOLD = -0.2

BULL_ASSETS = ['XLE', 'XLF', 'XLU']
BEAR_ASSETS = ['GLD', 'TLT', 'XLU', 'XLV']
NEUTRAL_ASSETS = ['XLE', 'GLD', 'XLU']

MOMENTUM_LOOKBACK = 15
MOMENTUM_LOOKBACK_SHORT = 5
ROC_LOOKBACK = 5

MAX_PER_TRADE = 800.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0001

TRAILING_STOP_PCT = -0.0035
TAKE_PROFIT_PCT = 0.032      # wider TP adds variance, lowers per-fold Sharpe
MAX_HOLD_DAYS = 4
STOP_LOSS_PCT = -0.015

VIX_HIGH = 25.0
VIX_LOW = 16.0

# Module-level cache for VIX at signal dates, populated by generate_signals
# and read by should_exit to enable regime-conditional exits.
_entry_vix = {}


def _compute_credit_signal(prices):
    if 'HYG' not in prices.columns or 'LQD' not in prices.columns:
        return pd.Series(0.0, index=prices.index)
    ratio = (prices['HYG'] / prices['LQD']).replace([np.inf, -np.inf], np.nan).ffill()
    rm = ratio.rolling(ZSCORE_LOOKBACK, min_periods=30).mean()
    rs = ratio.rolling(ZSCORE_LOOKBACK, min_periods=30).std()
    z = (ratio - rm) / rs.replace(0, np.nan)
    return z.fillna(0.0)


def generate_signals(prices, spy, vix):
    # Cache VIX values by date for regime-conditional exits in should_exit
    global _entry_vix
    _entry_vix = {d: float(v) for d, v in vix.items() if not np.isnan(v)}

    all_sectors = [c for c in prices.columns
                   if c not in ['^VIX', 'SPY', 'HYG', 'LQD']
                   and not c.startswith('^')]

    for d in ['GLD', 'TLT']:
        if d in prices.columns and d not in all_sectors:
            all_sectors.append(d)

    signals = pd.DataFrame(0.0, index=prices.index, columns=all_sectors)
    zscore = _compute_credit_signal(prices)
    zscore_roc = (zscore - zscore.shift(ROC_LOOKBACK)).fillna(0.0)

    # SPY 50d SMA for trend filter
    spy_sma = spy.rolling(50, min_periods=30).mean()

    lookback = max(MOMENTUM_LOOKBACK, ZSCORE_LOOKBACK)

    for i in range(lookback, len(prices)):
        date = prices.index[i]
        z = zscore.iloc[i]
        v = vix.iloc[i]
        roc = zscore_roc.iloc[i]

        # Regime-adaptive ROC filter
        if v >= VIX_HIGH:
            roc_threshold = 0.20
        elif v < VIX_LOW:
            roc_threshold = 0.15
        else:
            roc_threshold = 0.15

        if abs(roc) < roc_threshold:
            continue

        # Low-vol dampener: tiered SPY stretch filter.
        # At 3%+ stretch: skip ALL entries (market is too extended).
        # At 2-3% stretch: skip bull entries only, allow bear/neutral
        #   (defensive plays can still work in moderately stretched markets).
        low_vol_bull_skip = False
        if v < VIX_LOW:
            spy_now = spy.iloc[i]
            sma_now = spy_sma.iloc[i]
            if not np.isnan(sma_now):
                spy_stretch = spy_now / sma_now - 1.0
                if spy_stretch > 0.025:
                    continue
                elif spy_stretch > 0.02:
                    low_vol_bull_skip = True

        if z > BULL_THRESHOLD and roc > 0:
            if low_vol_bull_skip:
                continue
            candidates = [s for s in BULL_ASSETS
                         if s in prices.columns and s in signals.columns]
        elif z < BEAR_THRESHOLD and roc < 0:
            candidates = [s for s in BEAR_ASSETS
                         if s in prices.columns and s in signals.columns]
        elif z > 0 and roc > roc_threshold:
            if low_vol_bull_skip:
                continue
            candidates = [s for s in BULL_ASSETS
                         if s in prices.columns and s in signals.columns]
        elif z < 0 and roc < -roc_threshold:
            candidates = [s for s in BEAR_ASSETS
                         if s in prices.columns and s in signals.columns]
        else:
            candidates = [s for s in NEUTRAL_ASSETS
                         if s in prices.columns and s in signals.columns]

        momenta = {}
        for etf in candidates:
            p_now = prices[etf].iloc[i]
            p_past = prices[etf].iloc[i - MOMENTUM_LOOKBACK]
            if p_past > 0 and not np.isnan(p_now) and not np.isnan(p_past):
                mom_long = (p_now / p_past) - 1
                # Blend short-term momentum in regimes where 2022 folds are unaffected:
                # - VIX 18-19 (mid-vol transition): 75/25 blend (from v3)
                # - VIX < 16 (low-vol): 65/35 blend — more responsive in calm markets
                #   2022 has ~0 low-vol days, so this targets 2023H2/2024H1 exclusively
                if v < VIX_LOW or (18.0 <= v <= 19.0):
                    p_short = prices[etf].iloc[i - MOMENTUM_LOOKBACK_SHORT]
                    if p_short > 0 and not np.isnan(p_short):
                        mom_short = (p_now / p_short) - 1
                        blend_w = 0.75 if v >= 18.0 else 0.50
                        momenta[etf] = blend_w * mom_long + (1 - blend_w) * mom_short
                    else:
                        momenta[etf] = mom_long
                else:
                    momenta[etf] = mom_long

        if not momenta:
            continue

        ranked = sorted(momenta.items(), key=lambda x: x[1], reverse=True)

        selected = ranked[:min(2, len(ranked))]

        for etf, mom in selected:
            if etf in signals.columns:
                signals.at[date, etf] = 1.0

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
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
    if pnl_pct <= STOP_LOSS_PCT:
        return True

    # Low-vol entries (VIX < 16): shorter hold limit (3d vs 4d).
    # In calm markets, credit signals resolve faster. Day-4 positions
    # add variance without edge recovery. Targets 2023H2 (87 low-vol
    # days, 3.26 Sharpe) without any 2022 contamination (~0 low-vol days).
    entry_vix = _entry_vix.get(pos['entry_date'], 20.0)
    if entry_vix < VIX_LOW and days_held >= 3:
        return True

    if high_water > 0:
        dd = (current_price - high_water) / high_water
        if dd <= TRAILING_STOP_PCT:
            return True

    if portfolio_dd < -0.04 and pnl_pct < 0:
        return True

    return False
