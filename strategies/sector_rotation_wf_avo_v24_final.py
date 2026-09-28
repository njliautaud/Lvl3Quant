#!/usr/bin/env python3
"""
Sector Rotation Walk-Forward Strategy -- v96_ret60d_trend_quality
============================================================
v14 base + ret_60d trend quality scoring: sectors with stronger
long-term trend (positive ret_60d) get higher scores. This
differentiates candidates on multi-candidate days by favoring
sectors in established uptrends where dip-buying is more reliable.
Replaces static reversion weight with trend-quality-aware scoring.
"""

import numpy as np
import pandas as pd

RS_RANK_CUTOFF = 7
LEAD_LAG_MIN = 0.005
MIN_COMPOSITE_SCORE = 0.28
MIN_DISPERSION = 0.007
MARKET_RET20D_MEDIAN_MIN = -0.06

W_REL_STRENGTH = 0.05
W_LEAD_LAG = 0.50
W_RS_RANK = 0.20
W_REVERSION = 0.25

REV_RET1D_RANGE = (-0.06, -0.001)
REV_RET1D_HIGH_DISP = (-0.08, -0.002)
REV_RET1D_LOW_DISP = (-0.04, -0.001)

TREND_GUARD_RET20D_MIN = -0.15
TREND_GUARD_RET20D_MAX = 0.15
REL_STRENGTH_MIN = -0.08

MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 1
SLIPPAGE_PCT = 0.0001

TRAILING_STOP_PCT = -0.008
TAKE_PROFIT_PCT = 0.050
MAX_HOLD_DAYS = 3
UNDERWATER_CUT_DAYS = 1
UNDERWATER_CUT_TOL = -0.008
PORTFOLIO_DD_CUT = -0.03

DISP_HIGH_MULT = 1.5
DISP_LOW_MULT = 0.7
TOP_RANK_BOOST = 0.05
MOMENTUM_CROSS_BOOST = 0.08  # bonus for positive momentum crossover
MOMENTUM_CROSS_PENALTY = -0.05  # penalty for bearish momentum crossover
TREND_QUALITY_BOOST = 0.04  # bonus for strong 60d trend (above median)
PULLBACK_UPTREND_BOOST = 0.05  # bonus when ret_60d positive but ret_20d negative (classic pullback)


def fit(train_data):
    etfs = train_data['etf'].unique()
    tradeable = list(etfs)
    all_rs = train_data['rel_strength_spy'].dropna()
    all_ll = train_data['lead_lag_score_5d'].dropna()
    all_r60 = train_data['ret_60d'].dropna() if 'ret_60d' in train_data.columns else pd.Series(dtype=float)

    disp_by_date = train_data.groupby('date')['ret_1d'].std()
    median_disp = float(disp_by_date.median()) if len(disp_by_date) > 0 else 0.01

    return {
        'tradeable_etfs': tradeable,
        'rs_p25': float(all_rs.quantile(0.25)) if len(all_rs) > 0 else -0.1,
        'rs_p75': float(all_rs.quantile(0.75)) if len(all_rs) > 0 else 0.1,
        'll_p25': float(all_ll.quantile(0.25)) if len(all_ll) > 0 else -0.1,
        'll_p75': float(all_ll.quantile(0.75)) if len(all_ll) > 0 else 0.1,
        'r60_p25': float(all_r60.quantile(0.25)) if len(all_r60) > 0 else -0.1,
        'r60_p75': float(all_r60.quantile(0.75)) if len(all_r60) > 0 else 0.1,
        'median_disp': median_disp,
    }


def _normalize(val, p25, p75):
    iqr = p75 - p25
    if iqr < 1e-8:
        return 0.5
    centered = (val - (p25 + p75) / 2.0) / iqr
    return min(max(centered, -2.0), 2.0) / 4.0 + 0.5


def generate_signals(data, params):
    tradeable = params.get('tradeable_etfs', [])
    if not tradeable:
        tradeable = data['etf'].unique().tolist()
    rs_p25 = params.get('rs_p25', -0.1)
    rs_p75 = params.get('rs_p75', 0.1)
    ll_p25 = params.get('ll_p25', -0.1)
    ll_p75 = params.get('ll_p75', 0.1)
    r60_p25 = params.get('r60_p25', -0.1)
    r60_p75 = params.get('r60_p75', 0.1)
    median_disp = params.get('median_disp', 0.01)

    signals = []
    dates = sorted(data['date'].unique())

    daily_dispersion = {}
    daily_median_ret20d = {}
    for date in dates:
        day_data = data[data['date'] == date]
        if 'ret_1d' in day_data.columns and len(day_data) >= 3:
            disp = day_data['ret_1d'].std()
            daily_dispersion[date] = disp if not pd.isna(disp) else 0.0
        else:
            daily_dispersion[date] = 0.0
        if 'ret_20d' in day_data.columns:
            med = day_data['ret_20d'].median()
            daily_median_ret20d[date] = med if not pd.isna(med) else 0.0
        else:
            daily_median_ret20d[date] = 0.0

    for date in dates:
        disp = daily_dispersion.get(date, 0.0)
        if disp < MIN_DISPERSION:
            continue
        med_ret20d = daily_median_ret20d.get(date, 0.0)
        if med_ret20d < MARKET_RET20D_MEDIAN_MIN:
            continue

        if median_disp > 1e-8:
            disp_ratio = disp / median_disp
        else:
            disp_ratio = 1.0
        
        if disp_ratio > DISP_HIGH_MULT:
            rev_range = REV_RET1D_HIGH_DISP
            high_disp_day = True
        elif disp_ratio < DISP_LOW_MULT:
            rev_range = REV_RET1D_LOW_DISP
            high_disp_day = False
        else:
            rev_range = REV_RET1D_RANGE
            high_disp_day = False

        day_data = data[data['date'] == date]
        for _, row in day_data.iterrows():
            etf = row['etf']
            if etf not in tradeable:
                continue

            rs = row.get('rel_strength_spy', 0)
            rs_rank = row.get('rs_rank_among_sectors', 99)
            ll_score = row.get('lead_lag_score_5d', 0)
            ret_1d = row.get('ret_1d', 0)
            ret_20d = row.get('ret_20d', 0)
            ret_60d = row.get('ret_60d', 0)
            mom_cross = row.get('momentum_cross_20_60', 0)

            if pd.isna(rs) or pd.isna(rs_rank) or pd.isna(ll_score):
                continue
            if not pd.isna(ret_20d) and ret_20d < TREND_GUARD_RET20D_MIN:
                continue
            if not pd.isna(ret_20d) and ret_20d > TREND_GUARD_RET20D_MAX:
                continue
            if rs < REL_STRENGTH_MIN:
                continue

            if pd.isna(ret_1d):
                continue
            if not (rev_range[0] <= ret_1d <= rev_range[1]):
                continue
            if ll_score < LEAD_LAG_MIN:
                continue
            if rs_rank > RS_RANK_CUTOFF:
                continue

            rs_norm = _normalize(rs, rs_p25, rs_p75)
            ll_norm = _normalize(ll_score, ll_p25, ll_p75)
            rank_norm = 1.0 - (rs_rank - 1) / 10.0
            rev_norm = min(np.sqrt(abs(ret_1d) / 0.03), 1.0)

            score = (W_REL_STRENGTH * rs_norm + W_LEAD_LAG * ll_norm + 
                     W_RS_RANK * max(rank_norm, 0.0) + W_REVERSION * rev_norm)

            if high_disp_day and rs_rank <= 3:
                score += TOP_RANK_BOOST

            # Momentum cross: boost uptrend, penalize downtrend
            if not pd.isna(mom_cross):
                if mom_cross > 0:
                    score += MOMENTUM_CROSS_BOOST
                elif mom_cross < 0:
                    score += MOMENTUM_CROSS_PENALTY

            # Trend quality: bonus for sectors with strong 60d trend
            if not pd.isna(ret_60d):
                r60_norm = _normalize(ret_60d, r60_p25, r60_p75)
                if r60_norm > 0.6:  # above median trend
                    score += TREND_QUALITY_BOOST * (r60_norm - 0.5)

            # Pullback in uptrend: ret_60d positive + ret_20d negative = classic reversion setup
            if not pd.isna(ret_60d) and not pd.isna(ret_20d):
                if ret_60d > 0.02 and ret_20d < -0.01:
                    score += PULLBACK_UPTREND_BOOST

            if score < MIN_COMPOSITE_SCORE:
                continue

            signals.append({'date': date, 'etf': etf, 'score': round(float(score), 4)})

    if not signals:
        return pd.DataFrame(columns=['date', 'etf', 'score'])
    return pd.DataFrame(signals)


def should_exit(pos, current_price, current_date, portfolio_dd):
    entry_price = pos['entry_price_adj']
    high_water = pos.get('high_water_mark', entry_price)
    days_held = np.busday_count(np.datetime64(pos['entry_date'], 'D'), np.datetime64(current_date, 'D'))
    pnl_pct = (current_price - entry_price) / entry_price
    if days_held >= MAX_HOLD_DAYS:
        return True
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True
    if high_water > 0:
        drawdown_from_high = (current_price - high_water) / high_water
        if drawdown_from_high <= TRAILING_STOP_PCT:
            return True
    if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < UNDERWATER_CUT_TOL:
        return True
    if portfolio_dd < PORTFOLIO_DD_CUT:
        return True
    return False
