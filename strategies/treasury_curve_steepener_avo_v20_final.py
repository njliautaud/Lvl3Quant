#!/usr/bin/env python3
"""
Treasury Curve Steepener Strategy -- AVO v16
=============================================
Hybrid approach:
- VIX >= 20: pure dual momentum across all sectors (best for HV)
- VIX 15-20: curve-confirmed momentum (curve direction filters sector group,
  then momentum picks best within group) — combines curve's directional
  insight with momentum quality filter
- VIX < 15: no new entries
"""

import numpy as np
import pandas as pd

CURVE_LOOKBACK = 90
MOMENTUM_DAYS = 10

MOM_LOOKBACK = 20
MOM_SHORT = 5
TOP_N = 1
REBAL_DAYS = 4

STEEPENING_SECTORS = ['XLF', 'XLI', 'XLE', 'XLB']
FLATTENING_SECTORS = ['XLK', 'XLY', 'XLRE']
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']

MAX_HOLD_DAYS = 3
TRAILING_STOP_PCT = -0.003
STOP_LOSS_PCT = -0.012
TAKE_PROFIT_PCT = 0.06
UNDERWATER_EXIT_DAYS = 1
MAX_CONCURRENT = 1


def generate_signals(prices, spy, vix):
    if 'TLT' not in prices.columns or 'SHY' not in prices.columns:
        return pd.DataFrame(0.0, index=prices.index,
                            columns=STEEPENING_SECTORS + FLATTENING_SECTORS + DEFENSIVE_SECTORS)

    curve = prices['TLT'] / prices['SHY']
    curve_ma = curve.rolling(CURVE_LOOKBACK).mean()
    curve_std = curve.rolling(CURVE_LOOKBACK).std()
    curve_z = (curve - curve_ma) / curve_std
    curve_mom = curve_z.diff(MOMENTUM_DAYS)

    vix_aligned = vix.reindex(prices.index, method='ffill').fillna(20.0)

    # SPY trend filter for mid-vol
    spy_ma50 = spy.rolling(50).mean()
    # SPY short-term momentum
    spy_mom10 = spy.pct_change(10)

    all_sectors = STEEPENING_SECTORS + FLATTENING_SECTORS + DEFENSIVE_SECTORS
    signals = pd.DataFrame(0.0, index=prices.index, columns=all_sectors)

    day_count = 0
    warmup = max(CURVE_LOOKBACK + MOMENTUM_DAYS, MOM_LOOKBACK + 1, 51)

    for i in range(warmup, len(prices)):
        day_count += 1
        if day_count % REBAL_DAYS != 0:
            continue

        cm = curve_mom.iloc[i]
        vx = vix_aligned.iloc[i]
        if pd.isna(cm):
            continue

        # Skip very low vol
        if vx < 12:
            continue

        if vx >= 20:
            # High/elevated vol: pure momentum across all sectors
            # Require SPY short-term momentum positive (avoid catching falling knives)
            sm = spy_mom10.iloc[i] if i < len(spy_mom10) else np.nan
            if pd.isna(sm) or sm <= 0:
                continue
            candidate_sectors = all_sectors
        else:
            # Mid vol (15-20): curve-directed + SPY trend filter
            spy_price = spy.iloc[i] if i < len(spy) else np.nan
            spy_ma = spy_ma50.iloc[i] if i < len(spy_ma50) else np.nan
            if pd.isna(spy_price) or pd.isna(spy_ma) or spy_price < spy_ma:
                continue  # skip MV entries when SPY below 50d MA

            thresh = 0.5
            if cm < -thresh:
                candidate_sectors = STEEPENING_SECTORS
            elif cm > thresh:
                candidate_sectors = FLATTENING_SECTORS
            else:
                # Neutral curve: use all sectors
                candidate_sectors = all_sectors

        # Triple momentum filter with curve alignment bonus
        mom_scores = {}
        for sec in candidate_sectors:
            if sec in prices.columns:
                p_now = prices[sec].iloc[i]
                p_20 = prices[sec].iloc[i - MOM_LOOKBACK]
                p_10 = prices[sec].iloc[i - 10]

                if p_20 > 0 and p_10 > 0 and not pd.isna(p_now) and not pd.isna(p_20) and not pd.isna(p_10):
                    mom_20 = (p_now - p_20) / p_20
                    mom_10 = (p_now - p_10) / p_10

                    p_5 = prices[sec].iloc[i - MOM_SHORT]
                    if p_5 > 0 and not pd.isna(p_5):
                        mom_5 = (p_now - p_5) / p_5
                        if mom_20 > 0 and mom_5 > 0:
                            score = (mom_20 + mom_10 + mom_5) / 3
                            # Curve alignment bonus for ranking
                            if cm < 0 and sec in STEEPENING_SECTORS:
                                score *= 3.0
                            elif cm > 0 and sec in FLATTENING_SECTORS:
                                score *= 3.0
                            mom_scores[sec] = score

        if len(mom_scores) == 0:
            continue

        sorted_secs = sorted(mom_scores, key=mom_scores.get, reverse=True)[:TOP_N]

        for sec in sorted_secs:
            if sec in signals.columns:
                signals.iloc[i, signals.columns.get_loc(sec)] = 1.0

    return signals


def should_exit(pos, curr_price, date, portfolio_dd):
    entry_price = pos['entry_price_adj']
    hwm = pos.get('high_water_mark', entry_price)

    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(date, 'D')
    )

    ret_from_entry = (curr_price - entry_price) / entry_price

    if days_held >= UNDERWATER_EXIT_DAYS and ret_from_entry < 0:
        return True

    if days_held >= MAX_HOLD_DAYS:
        return True
    if ret_from_entry >= TAKE_PROFIT_PCT:
        return True
    if ret_from_entry <= STOP_LOSS_PCT:
        return True
    if hwm > 0:
        ret_from_hwm = (curr_price - hwm) / hwm
        if ret_from_hwm <= TRAILING_STOP_PCT:
            return True

    return False
