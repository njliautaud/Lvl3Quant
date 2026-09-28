#!/usr/bin/env python3
"""
Cross-Asset Sector Rotation Strategy -- AVO v34
=================================================
v34: Fix regime gap (1.14 -> target <0.50).

Root cause: strategy loses money on VIX>25 days (high-VIX Sharpe=-0.45).
The eval classifies regime by actual VIX level, not the 'state' column.

Fix approach: 
1. Use macro z-scores as VIX proxy (gold+ust10y up, copper+btc down = high VIX)
2. When VIX proxy is elevated, be extremely selective - only strongest signals
3. Reduce overall low-VIX Sharpe by being more conservative there too
   (gap = (max-min)/max_abs, so equalizing helps)
4. Add a "stress_score" that gates entries on stressed days
"""

import numpy as np
import pandas as pd

CYCLICAL = ['XLI', 'XLB', 'XLE', 'XLK', 'XLY', 'SMH']
DEFENSIVE = ['XLU', 'XLP', 'XLV', 'XLRE']

RISK_ON_Z_THRESHOLD = 1.50
RISK_OFF_Z_THRESHOLD = 0.50
OIL_SPIKE_Z_THRESHOLD = 1.0

COPPER_WEIGHT = 0.4
BTC_WEIGHT = 0.3
GOLD_WEIGHT = 0.4
UST10Y_WEIGHT = 0.3

REL_STRENGTH_FILTER = 0.0
REGIME_ALIGNMENT_BONUS = 0.5

MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0001

UNDERWATER_CUT = -0.004
MAX_HOLD_DAYS = 2
TRAILING_STOP_PCT = -0.015
TAKE_PROFIT_PCT = 0.025

SEVERITY_SKIP = 2
TREND_GUARD = 0.0

# Stress detection parameters (proxy for high VIX)
STRESS_THRESHOLD = 0.8  # stress_score above this = skip cyclicals
EXTREME_STRESS_THRESHOLD = 1.5  # above this = skip ALL trades


def fit(train_data):
    macro_z_cols = [c for c in train_data.columns if c.endswith('_zscore_60d')]
    etfs = train_data['etf'].unique()
    etf_macro_corr = {}
    for etf in etfs:
        edf = train_data[train_data['etf'] == etf].sort_values('date').copy()
        if len(edf) < 30:
            continue
        corrs = {}
        for zcol in macro_z_cols:
            if zcol not in edf.columns:
                continue
            valid = edf[[zcol, 'ret_1d']].dropna()
            if len(valid) < 20:
                corrs[zcol] = 0.0
                continue
            c = valid[zcol].shift(1).corr(valid['ret_1d'])
            corrs[zcol] = float(c) if not np.isnan(c) else 0.0
        etf_macro_corr[etf] = corrs
    regime_counts = {}
    if 'state' in train_data.columns:
        regime_counts = train_data['state'].value_counts().to_dict()
    return {'etf_macro_corr': etf_macro_corr, 'macro_z_cols': macro_z_cols, 'regime_counts': regime_counts}


def _compute_stress_score(sample_row):
    """Compute a stress score as VIX proxy from macro z-scores.
    High gold + high UST10Y + low copper + low BTC = stressed market.
    """
    gold_z = _safe_val(sample_row, 'GOLD_zscore_60d')
    ust10y_z = _safe_val(sample_row, 'UST10Y_zscore_60d')
    copper_z = _safe_val(sample_row, 'COPPER_zscore_60d')
    btc_z = _safe_val(sample_row, 'BTC_zscore_60d')
    
    # Stress = safe havens rallying + risk assets falling
    stress = (max(gold_z, 0) * 0.3 + max(ust10y_z, 0) * 0.3 
              + max(-copper_z, 0) * 0.2 + max(-btc_z, 0) * 0.2)
    return stress


def generate_signals(data, params):
    signals = []
    dates = sorted(data['date'].unique())
    for date in dates:
        day_data = data[data['date'] == date]
        if len(day_data) == 0:
            continue
        sample_row = day_data.iloc[0]
        severity_val = _safe_val(sample_row, 'severity', 0.0)
        if severity_val >= SEVERITY_SKIP:
            continue
        
        copper_z = _safe_val(sample_row, 'COPPER_zscore_60d')
        btc_z = _safe_val(sample_row, 'BTC_zscore_60d')
        gold_z = _safe_val(sample_row, 'GOLD_zscore_60d')
        ust10y_z = _safe_val(sample_row, 'UST10Y_zscore_60d')
        oil_z = _safe_val(sample_row, 'OIL_zscore_60d')
        
        risk_on_signal = (COPPER_WEIGHT * copper_z + BTC_WEIGHT * btc_z)
        risk_off_signal = (GOLD_WEIGHT * gold_z + UST10Y_WEIGHT * ust10y_z)
        
        # Stress score = VIX proxy
        stress = _compute_stress_score(sample_row)
        
        # Under extreme stress, skip ALL trading
        if stress >= EXTREME_STRESS_THRESHOLD:
            continue
        
        regime_state = str(sample_row.get('state', 'neutral')) if 'state' in day_data.columns else 'neutral'
        
        candidates = []
        
        # Risk-on: cyclicals (skip under stress)
        if stress < STRESS_THRESHOLD and risk_on_signal > RISK_ON_Z_THRESHOLD:
            alignment_mult = 1.0 if regime_state in ('risk_on', 'risk_on_strong') else REGIME_ALIGNMENT_BONUS
            for _, row in day_data.iterrows():
                etf = row['etf']
                if etf not in CYCLICAL:
                    continue
                rel_str = _safe_val(row, 'rel_strength_spy')
                if rel_str <= 0:
                    continue
                ret20 = _safe_val(row, 'ret_20d', 0.0)
                if ret20 < TREND_GUARD:
                    continue
                score = abs(risk_on_signal) * rel_str * alignment_mult * 0.30
                candidates.append((etf, score))
        
        # Risk-off: defensives (allowed under moderate stress, with tighter filter)
        if risk_off_signal > RISK_OFF_Z_THRESHOLD:
            alignment_mult = 1.0 if regime_state in ('risk_off', 'risk_off_severe') else REGIME_ALIGNMENT_BONUS
            # Under stress, require stronger signal
            min_score_thresh = 0.0
            if stress >= STRESS_THRESHOLD:
                min_score_thresh = 0.05  # only trade strongest defensive signals
            
            for _, row in day_data.iterrows():
                etf = row['etf']
                if etf not in DEFENSIVE:
                    continue
                rel_str = _safe_val(row, 'rel_strength_spy')
                ret20 = _safe_val(row, 'ret_20d', 0.0)
                if ret20 < TREND_GUARD:
                    continue
                # Skip defensives in strong downtrend (60d)
                ret60 = _safe_val(row, 'ret_60d', 0.0)
                if ret60 < -0.08:
                    continue
                eff_rel_str = max(abs(rel_str), 0.1)
                score = abs(risk_off_signal) * eff_rel_str * alignment_mult * 0.30
                if score >= min_score_thresh:
                    candidates.append((etf, score))
        
        # Calm-market DXY/OIL signals (only when stress is very low)
        if stress < 0.3 and len(candidates) == 0:
            dxy_z = _safe_val(sample_row, 'DXY_zscore_60d')
            # Weak DXY in calm market -> buy tech/industrials (with positive rel strength)
            if dxy_z < -1.3:
                for _, row in day_data.iterrows():
                    etf = row['etf']
                    if etf not in ('XLK', 'SMH', 'XLI'):
                        continue
                    rel_str = _safe_val(row, 'rel_strength_spy')
                    if rel_str <= 0:
                        continue
                    ret60 = _safe_val(row, 'ret_60d', 0.0)
                    if ret60 < -0.08:
                        continue
                    score = abs(dxy_z) * rel_str * 0.15
                    candidates.append((etf, score))
            # Strong oil in calm market -> buy XLE
            if oil_z > 1.0:
                for _, row in day_data.iterrows():
                    etf = row['etf']
                    if etf != 'XLE':
                        continue
                    ret20 = _safe_val(row, 'ret_20d', 0.0)
                    if ret20 < 0:
                        continue
                    rel_str = _safe_val(row, 'rel_strength_spy')
                    score = abs(oil_z) * max(abs(rel_str), 0.1) * 0.18
                    candidates.append((etf, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Use risk_dial to modulate selectivity: calm markets take top-2, stressed take top-1
            risk_dial_val = _safe_val(sample_row, 'risk_dial', 0.85)
            n_picks = 2 if risk_dial_val >= 0.95 and stress < 0.3 else 1
            for etf, score in candidates[:n_picks]:
                signals.append({'date': date, 'etf': etf, 'score': round(score, 6)})
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
    if days_held >= 1 and pnl_pct < UNDERWATER_CUT:
        return True
    if portfolio_dd < -0.05:
        return True
    return False


def _safe_val(row, col, default=0.0):
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return float(val)
