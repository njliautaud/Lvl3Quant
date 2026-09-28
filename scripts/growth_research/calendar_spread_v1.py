#!/usr/bin/env python3
"""
Calendar Spread v1 — DIVERSIFYING Strategy (Low Correlation to Momentum Spreads)
==================================================================================

Calendar spread = sell near-dated ATM call + buy same-strike longer-dated ATM call.
Profits from:
  1. Faster theta decay in near-dated option
  2. Stock staying near strike (max profit at strike at near expiry)
  3. Vol term structure contango (near IV < far IV)

This is fundamentally DIFFERENT from directional momentum spreads:
  - Momentum: profits from price MOVEMENT in a direction
  - Calendar: profits from price STABILITY near strike + time decay

Key question: Does this offer genuinely different returns from our Sharpe 1.87
production bull call spread strategy? Measured via monthly return correlation.

Variants tested:
  A. LGBM vol-ranked top 3 sectors, VIX>20 filter
  B. LGBM vol-ranked top 3 sectors, GRU regime>0.4 filter
  C. LOWEST vol sectors (calendars prefer low vol — theta dominates)
  D. All 11 sectors equally (structural baseline)
  E. VIX term structure filter (only enter in contango, VIX_ratio < 0.95)

Walk-forward: monthly rebalance, 2017-2026. $645 starting capital.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

from research.tools.options_pricer import (
    bs_call_price, estimate_iv, compute_atr,
    COMMISSION_RT_SPREAD, DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ─── Config ───────────────────────────────────────────────────────────

STARTING_CAPITAL = 645.0
MAX_PCT_PER_POSITION = 0.40       # 40% max per position (per user spec)
NEAR_DTE = 21                     # Front month: 21 DTE
FAR_DTE = 49                      # Back month: 49 DTE (28 DTE remaining at near expiry)
COMMISSION_PER_SPREAD = 2.60      # $0.65/leg x 4 legs round-trip

BASE = Path('/home/jupiter/Lvl3Quant')
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU',
               'XLB', 'XLRE', 'XLC']

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ─── MLflow ───────────────────────────────────────────────────────────

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — logging locally only")


# ─── Data Loading ─────────────────────────────────────────────────────

def download_data():
    """Download sector ETFs + VIX + VIX3M via yfinance."""
    import yfinance as yf

    tickers = SECTOR_ETFS + ['SPY', '^VIX', '^VIX3M']
    fprint(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start='2016-01-01', end='2026-07-27', progress=False)

    close = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw[('Close',)]
    high = raw['High'] if 'High' in raw.columns.get_level_values(0) else raw[('High',)]
    low = raw['Low'] if 'Low' in raw.columns.get_level_values(0) else raw[('Low',)]

    # Handle MultiIndex columns from yfinance
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(1)
    if isinstance(high.columns, pd.MultiIndex):
        high.columns = high.columns.get_level_values(1)
    if isinstance(low.columns, pd.MultiIndex):
        low.columns = low.columns.get_level_values(1)

    # Rename VIX columns
    rename = {'^VIX': 'VIX', '^VIX3M': 'VIX3M'}
    close = close.rename(columns=rename)
    high = high.rename(columns=rename)
    low = low.rename(columns=rename)

    close = close.ffill().dropna(how='all')
    high = high.ffill().dropna(how='all')
    low = low.ffill().dropna(how='all')

    fprint(f"  Data: {len(close)} days, {close.columns.tolist()}")
    return close, high, low


# ─── GRU Regime Loader ────────────────────────────────────────────────

def load_regime_predictions():
    """Load GRU regime predictions from the regime detector model."""
    regime_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_path.exists():
        fprint("  WARNING: Regime predictions not found — using VIX proxy")
        return None

    data = np.load(regime_path, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']
    regime_series = pd.Series(scores, index=dates, name='regime_score')
    if regime_series.index.duplicated().any():
        regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    fprint(f"  Regime scores: {len(regime_series)}, range [{scores.min():.3f}, {scores.max():.3f}]")
    return regime_series


def vix_proxy_regime(vix_series):
    """Fallback: VIX-based regime proxy. Low VIX = bull (high score)."""
    scores = 1.0 - (vix_series - 12) / 23.0
    return scores.clip(0, 1).rename('regime_score')


# ─── Feature Engineering for LGBM ────────────────────────────────────

def build_vol_features(close, high, low, vix_series, vix3m_series):
    """
    Build VOL-focused features for LGBM ranking.
    Calendar spreads care about vol characteristics MORE than momentum.
    """
    features_by_date = {}
    valid_sectors = [s for s in SECTOR_ETFS if s in close.columns]
    dates = close.index

    for i in range(252, len(dates)):
        dt = dates[i]
        row_feats = {}

        for sector in valid_sectors:
            px = close[sector].iloc[max(0, i-252):i+1].values
            hi = high[sector].iloc[max(0, i-252):i+1].values if sector in high.columns else None
            lo = low[sector].iloc[max(0, i-252):i+1].values if sector in low.columns else None

            if len(px) < 63 or np.isnan(px[-1]):
                continue

            spot = px[-1]
            rets = np.diff(np.log(px))

            # ─── Vol features (primary for calendars) ───
            rv_21 = np.std(rets[-21:]) * np.sqrt(252) if len(rets) >= 21 else 0.2
            rv_63 = np.std(rets[-63:]) * np.sqrt(252) if len(rets) >= 63 else 0.2

            # Vol ratio (term structure proxy): short-term vs long-term RV
            vol_ratio = rv_21 / (rv_63 + 1e-8)

            # Vol trend: is vol rising or falling?
            if len(rets) >= 42:
                rv_prev = np.std(rets[-42:-21]) * np.sqrt(252)
                vol_trend = rv_21 / (rv_prev + 1e-8) - 1.0
            else:
                vol_trend = 0.0

            # Vol of vol (stability of volatility)
            if len(rets) >= 63:
                rolling_vol = pd.Series(rets[-63:]).rolling(5).std() * np.sqrt(252)
                vol_of_vol = rolling_vol.std() / (rolling_vol.mean() + 1e-8)
            else:
                vol_of_vol = 0.5

            # ATR-based vol
            if hi is not None and lo is not None and len(hi) >= 14:
                atr = compute_atr(hi, lo, px, period=14)
                atr_pct = atr / spot
            else:
                atr_pct = rv_21 / np.sqrt(252)
                atr = atr_pct * spot

            # ─── VIX term structure features (key for calendars) ───
            vix_val = 20.0
            if vix_series is not None and dt in vix_series.index:
                v = vix_series.loc[dt]
                if not np.isnan(v):
                    vix_val = float(v)
            vix_term_slope = 0.95  # Default mild contango
            if vix3m_series is not None and dt in vix3m_series.index:
                v3m = vix3m_series.loc[dt]
                if not np.isnan(v3m) and v3m > 0:
                    vix_term_slope = vix_val / float(v3m)

            # ─── Some momentum features (secondary) ───
            mom_21 = px[-1] / px[-22] - 1 if len(px) >= 22 else 0
            mom_63 = px[-1] / px[-64] - 1 if len(px) >= 64 else 0

            # Mean reversion signal (calendars prefer range-bound)
            if len(px) >= 63:
                sma_63 = np.mean(px[-63:])
                dist_from_mean = (px[-1] - sma_63) / sma_63
            else:
                dist_from_mean = 0.0

            row_feats[sector] = {
                'rv_21': rv_21,
                'rv_63': rv_63,
                'vol_ratio': vol_ratio,
                'vol_trend': vol_trend,
                'vol_of_vol': vol_of_vol,
                'atr_pct': atr_pct,
                'atr': atr,
                'vix_level': vix_val,
                'vix_term_slope': vix_term_slope,
                'mom_21': mom_21,
                'mom_63': mom_63,
                'dist_from_mean': dist_from_mean,
                'spot': spot,
            }

        if row_feats:
            features_by_date[dt] = row_feats

    return features_by_date


# ─── LGBM Walk-Forward Ranking ────────────────────────────────────────

def lgbm_wf_rank(features_by_date, close, train_periods=24, advance=1):
    """
    Walk-forward LightGBM ranking for calendar spread suitability.

    Label: which sectors had the BEST calendar spread outcomes?
    Proxy: sectors that stayed range-bound (low |price move|) with moderate vol
    = ideal for calendar spreads.
    """
    import lightgbm as lgb

    feature_names = ['rv_21', 'rv_63', 'vol_ratio', 'vol_trend', 'vol_of_vol',
                     'atr_pct', 'vix_level', 'vix_term_slope', 'mom_21', 'mom_63',
                     'dist_from_mean']

    # Get monthly rebalance dates from close index (fast)
    monthly_close_dates = close.resample('ME').last().index
    all_dates_arr = np.array(sorted(features_by_date.keys()))

    # Map each monthly date to nearest available feature date (vectorized)
    mapped_dates = []
    for md in monthly_close_dates:
        mask = all_dates_arr <= md
        if mask.any():
            mapped_dates.append(all_dates_arr[mask][-1])
    monthly_dates = mapped_dates

    if len(monthly_dates) < train_periods + 5:
        fprint(f"  Not enough monthly periods ({len(monthly_dates)}) for WF")
        return {}

    # Pre-build the full dataset for all dates/sectors (vectorized)
    fprint(f"  Pre-building training matrix for {len(monthly_dates)} monthly dates...")
    valid_close_sectors = [s for s in SECTOR_ETFS if s in close.columns]

    # Build {(date_idx, sector): (features, fwd_price_change)} lookup
    date_sector_data = {}
    for i, td in enumerate(monthly_dates[:-1]):
        next_td = monthly_dates[i + 1]
        if td not in features_by_date:
            continue

        for sector, feats in features_by_date[td].items():
            if sector not in valid_close_sectors:
                continue

            x = [feats.get(fn, 0.0) for fn in feature_names]

            # Forward price change for label
            try:
                p_entry = close.loc[td, sector] if td in close.index else np.nan
                p_exit = close.loc[next_td, sector] if next_td in close.index else np.nan
            except (KeyError, IndexError):
                continue

            if pd.isna(p_entry) or pd.isna(p_exit) or p_entry <= 0:
                continue

            abs_move = abs(float(p_exit) / float(p_entry) - 1)
            vol = feats.get('rv_21', 0.2)
            vol_bonus = 1.0 if 0.15 <= vol <= 0.35 else 0.5
            calendar_score = vol_bonus / (abs_move + 0.01)

            date_sector_data[(i, sector)] = (x, calendar_score)

    fprint(f"  Pre-built {len(date_sector_data)} training samples")

    rankings = {}

    for t_idx in range(train_periods, len(monthly_dates) - 1):
        test_date = monthly_dates[t_idx]
        train_start = max(0, t_idx - train_periods)

        # Gather training data from pre-built lookup
        X_train, y_train = [], []
        for i in range(train_start, t_idx):
            for sector in valid_close_sectors:
                key = (i, sector)
                if key in date_sector_data:
                    x, y = date_sector_data[key]
                    X_train.append(x)
                    y_train.append(y)

        if len(X_train) < 20:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train LGBM
        dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_names, free_raw_data=True)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'min_child_samples': 5,
            'verbose': -1,
            'seed': 42,
            'n_jobs': 1,  # Avoid excessive parallelism
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        # Predict rankings for test date
        if test_date not in features_by_date:
            continue

        sector_scores = {}
        test_feats = features_by_date[test_date]
        for sector, feats in test_feats.items():
            x = np.array([[feats.get(fn, 0.0) for fn in feature_names]])
            score = model.predict(x)[0]
            sector_scores[sector] = float(score)

        rankings[test_date] = sector_scores

    fprint(f"  LGBM rankings computed for {len(rankings)} periods")
    return rankings


# ─── Calendar Spread Pricing ─────────────────────────────────────────

def estimate_calendar_iv(atr, spot, vix):
    """
    Estimate IV for calendar spread pricing.

    Calendar spreads are particularly sensitive to IV estimation because
    the time value differential IS the trade. ATR-based IV underestimates
    actual options IV for most sector ETFs (IV includes risk premium).

    Method: use VIX as a sector IV proxy, adjusted by sector beta.
    Sector ETFs typically have IV = VIX * sector_beta * 0.9-1.1.
    For simplicity, use max(ATR-based IV, VIX/100 * 0.95) as the floor.
    """
    atr_iv = estimate_iv(atr, spot, vix)

    # VIX is SPX IV in percentage points. Sector ETFs typically have
    # IV close to VIX (with sector-specific adjustments).
    # Use VIX as a more realistic floor for option pricing.
    vix_based_iv = vix / 100.0  # VIX 20 -> 20% IV

    # Use the higher of ATR-based and VIX-based (options are rarely
    # cheaper than VIX implies for sector ETFs)
    sigma = max(atr_iv, vix_based_iv * 0.90)

    return sigma


def price_calendar_spread(spot, strike, near_dte, far_dte, atr, vix,
                          haircut=DEFAULT_HAIRCUT):
    """
    Price a calendar spread: sell near call + buy far call at same strike.

    Calendar spread is traded as a single spread order. The haircut applies
    to the NET debit, not each leg independently.

    Returns:
        (net_debit_per_share, near_call_price, far_call_price, sigma)
        net_debit = (far_call - near_call) * (1 + haircut)
    """
    T_near = near_dte / 365.0
    T_far = far_dte / 365.0

    sigma = estimate_calendar_iv(atr, spot, vix)

    near_call = bs_call_price(spot, strike, T_near, sigma=sigma)
    far_call = bs_call_price(spot, strike, T_far, sigma=sigma)

    # Net debit at fair value
    fair_debit = far_call - near_call

    # Apply haircut on the spread (single order), not per-leg
    net_debit = fair_debit * (1.0 + haircut)

    return max(net_debit, 0.01), near_call, far_call, sigma


def calendar_spread_exit_value(exit_price, strike, remaining_dte, atr, vix,
                               near_call_intrinsic, haircut=DEFAULT_HAIRCUT):
    """
    Value of the calendar spread at near-month expiry.

    At near expiry:
    - Near call: intrinsic value max(exit_price - strike, 0) — we owe this
    - Far call: still has remaining_dte left — we own this

    The spread is closed as a single order, so haircut applies to net value.

    Returns exit value per share after haircut.
    """
    # Near call at expiry: intrinsic only
    near_intrinsic = max(exit_price - strike, 0.0)

    # Far call still has time value
    T_remaining = remaining_dte / 365.0
    sigma = estimate_calendar_iv(atr, exit_price, vix)

    far_value_fair = bs_call_price(exit_price, strike, T_remaining, sigma=sigma)

    # Net spread value at fair
    net_fair = far_value_fair - near_intrinsic

    if net_fair <= 0:
        return 0.0  # Calendar worthless (stock moved too far)

    # Apply haircut on the net spread value (selling the spread)
    net_value = net_fair * (1.0 - haircut)

    return max(net_value, 0.0)


# ─── Backtest Engine ──────────────────────────────────────────────────

def run_variant(name, close, high, low, vix_series, vix3m_series,
                features_by_date, lgbm_rankings, regime_scores,
                sector_selector, filter_func=None):
    """
    Run calendar spread backtest for a given variant.

    Args:
        sector_selector: function(date, features, rankings) -> list of sector picks
        filter_func: function(date, vix, regime) -> bool (True = trade)
    """
    fprint(f"\n--- Variant {name} ---")

    valid_sectors = [s for s in SECTOR_ETFS if s in close.columns]
    rebal_dates = close.resample('ME').last().index
    # Start after enough data for features
    start_idx = max(24, next((i for i, d in enumerate(rebal_dates)
                              if d >= pd.Timestamp('2017-06-01')), 24))

    trades = []
    equity = STARTING_CAPITAL
    equity_curve = [(rebal_dates[start_idx], equity)]

    for ridx in range(start_idx, len(rebal_dates) - 1):
        rebal_date = rebal_dates[ridx]
        next_rebal = rebal_dates[ridx + 1]

        if equity <= 50:
            equity_curve.append((next_rebal, equity))
            continue

        # Get VIX on rebal date
        vix_val = 20.0
        if vix_series is not None:
            vix_at = vix_series[vix_series.index <= rebal_date]
            if len(vix_at) > 0:
                vix_val = float(vix_at.iloc[-1])

        # Get VIX3M
        vix3m_val = vix_val * 1.05  # Default slight contango
        if vix3m_series is not None:
            vix3m_at = vix3m_series[vix3m_series.index <= rebal_date]
            if len(vix3m_at) > 0:
                vix3m_val = float(vix3m_at.iloc[-1])

        vix_ratio = vix_val / (vix3m_val + 1e-8)

        # Get regime score
        regime_val = 0.5
        if regime_scores is not None:
            rs_at = regime_scores[regime_scores.index <= rebal_date]
            if len(rs_at) > 0:
                regime_val = float(rs_at.iloc[-1])

        # Apply variant filter
        if filter_func is not None:
            if not filter_func(rebal_date, vix_val, vix_ratio, regime_val):
                equity_curve.append((next_rebal, equity))
                continue

        # Get feature data for this date
        feat_date = None
        for d in sorted(features_by_date.keys(), reverse=True):
            if d <= rebal_date:
                feat_date = d
                break

        feats = features_by_date.get(feat_date, {}) if feat_date else {}

        # LGBM rankings: find nearest date <= rebal_date
        ranks = {}
        if lgbm_rankings:
            rank_dates = sorted(lgbm_rankings.keys())
            rank_candidates = [d for d in rank_dates if d <= rebal_date]
            if rank_candidates:
                ranks = lgbm_rankings[rank_candidates[-1]]

        # Select sectors
        picks = sector_selector(rebal_date, feats, ranks, valid_sectors)
        if not picks:
            equity_curve.append((next_rebal, equity))
            continue

        # Size positions
        max_per_position = equity * MAX_PCT_PER_POSITION
        per_pick = min(max_per_position, equity / max(len(picks), 1))

        period_pnl = 0.0

        for sector in picks:
            if sector not in close.columns:
                continue

            # Get entry price
            entry_prices = close[sector][close.index <= rebal_date]
            if len(entry_prices) == 0:
                continue
            spot = float(entry_prices.iloc[-1])
            if np.isnan(spot) or spot <= 0:
                continue

            # ATM strike
            strike = round(spot)

            # Get ATR for IV estimation
            if sector in high.columns and sector in low.columns:
                hi_vals = high[sector][high.index <= rebal_date].values[-60:]
                lo_vals = low[sector][low.index <= rebal_date].values[-60:]
                cl_vals = close[sector][close.index <= rebal_date].values[-60:]
                if len(hi_vals) >= 14:
                    atr = compute_atr(hi_vals, lo_vals, cl_vals, period=14)
                else:
                    atr = spot * 0.02
            else:
                atr = spot * 0.02

            # Price the calendar spread
            net_debit, near_call, far_call, sigma = price_calendar_spread(
                spot, strike, NEAR_DTE, FAR_DTE, atr, vix_val
            )

            # Cost per contract
            cost_per_contract = net_debit * 100  # 100 shares per contract

            if cost_per_contract <= 0 or np.isnan(cost_per_contract) or cost_per_contract > per_pick:
                continue

            # Cap contracts to limit single-position risk
            n_contracts = max(1, min(3, int(per_pick / cost_per_contract)))
            total_cost = cost_per_contract * n_contracts

            # Get exit price (at near expiry ~21 trading days later)
            exit_date_target = rebal_date + pd.Timedelta(days=NEAR_DTE * 1.4)  # Calendar days
            exit_prices = close[sector][(close.index > rebal_date) &
                                        (close.index <= exit_date_target)]
            if len(exit_prices) == 0:
                # Use next rebal date
                exit_prices = close[sector][(close.index > rebal_date) &
                                            (close.index <= next_rebal)]
            if len(exit_prices) == 0:
                continue

            exit_price = float(exit_prices.iloc[-1])
            actual_exit_date = exit_prices.index[-1]

            if np.isnan(exit_price) or exit_price <= 0:
                continue

            # Get exit ATR for IV estimation
            exit_hi = high[sector][high.index <= actual_exit_date].values[-60:] if sector in high.columns else None
            exit_lo = low[sector][low.index <= actual_exit_date].values[-60:] if sector in low.columns else None
            exit_cl = close[sector][close.index <= actual_exit_date].values[-60:]
            if exit_hi is not None and exit_lo is not None and len(exit_hi) >= 14:
                exit_atr = compute_atr(exit_hi, exit_lo, exit_cl, period=14)
            else:
                exit_atr = exit_price * 0.02

            # Get exit VIX
            exit_vix = vix_val
            if vix_series is not None:
                vix_exit_at = vix_series[vix_series.index <= actual_exit_date]
                if len(vix_exit_at) > 0:
                    exit_vix = float(vix_exit_at.iloc[-1])

            # Calendar spread value at near expiry
            remaining_dte = FAR_DTE - NEAR_DTE  # 28 days remaining on far call
            exit_value_per_share = calendar_spread_exit_value(
                exit_price, strike, remaining_dte, exit_atr, exit_vix,
                near_call_intrinsic=max(exit_price - strike, 0.0)
            )

            # PnL per share
            pnl_per_share = exit_value_per_share - net_debit
            pnl = pnl_per_share * 100 * n_contracts - COMMISSION_PER_SPREAD * n_contracts

            # Calendar max loss capped at debit paid
            if pnl < -total_cost:
                pnl = -total_cost

            period_pnl += pnl
            price_move = exit_price / spot - 1

            trades.append({
                'pnl': pnl,
                'entry_date': str(rebal_date.date()),
                'exit_date': str(actual_exit_date.date()),
                'ticker': sector,
                'spot': spot,
                'exit_price': exit_price,
                'strike': strike,
                'net_debit': net_debit * 100 * n_contracts,
                'exit_value': exit_value_per_share * 100 * n_contracts,
                'sigma': sigma,
                'vix': vix_val,
                'vix_ratio': vix_ratio,
                'regime': regime_val,
                'price_move': price_move,
                'n_contracts': n_contracts,
            })

        equity += period_pnl
        equity = max(0, equity)
        equity_curve.append((next_rebal, equity))

    if not trades:
        fprint(f"  No trades generated")
        return None

    # Convert equity curve
    eq_dates, eq_vals = zip(*equity_curve)
    equity_series = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates))

    # Monthly returns for correlation analysis
    monthly_eq = equity_series.resample('ME').last().dropna()
    monthly_returns = monthly_eq.pct_change().dropna()

    fprint(f"  Trades: {len(trades)}")
    fprint(f"  Final equity: ${equity:.0f}")

    return {
        'variant': name,
        'trades': trades,
        'equity_series': equity_series,
        'monthly_returns': monthly_returns,
    }


# ─── Sector Selectors ────────────────────────────────────────────────

def select_lgbm_top3(date, feats, ranks, valid_sectors):
    """Select top 3 sectors by LGBM vol-ranking score."""
    if not ranks:
        # Fallback: use vol features if available, else first 3
        if feats:
            vol_scores = [(s, feats[s].get('vol_ratio', 1.0))
                         for s in valid_sectors if s in feats]
            vol_scores.sort(key=lambda x: x[1])
            return [s for s, _ in vol_scores[:3]]
        return valid_sectors[:3]
    sorted_sectors = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in sorted_sectors[:3] if s in valid_sectors]


def select_lowest_vol(date, feats, ranks, valid_sectors):
    """Select 3 LOWEST vol sectors (calendars prefer low vol)."""
    if not feats:
        return valid_sectors[:3]  # Fallback: first 3
    vol_scores = []
    for s in valid_sectors:
        if s in feats:
            vol_scores.append((s, feats[s].get('rv_21', 1.0)))
    if not vol_scores:
        return valid_sectors[:3]
    vol_scores.sort(key=lambda x: x[1])  # Lowest vol first
    return [s for s, _ in vol_scores[:3]]


def select_all_sectors(date, feats, ranks, valid_sectors):
    """Trade top 5 sectors by lowest recent vol (broad but sized for $645)."""
    if not feats:
        return valid_sectors[:5]
    vol_scores = []
    for s in valid_sectors:
        if s in feats:
            vol_scores.append((s, feats[s].get('rv_21', 1.0)))
    vol_scores.sort(key=lambda x: x[1])
    return [s for s, _ in vol_scores[:5]]


# ─── Filter Functions ─────────────────────────────────────────────────

def filter_vix_gt20(date, vix, vix_ratio, regime):
    """Only trade when VIX > 20 (higher premium = better theta income)."""
    return vix > 20.0


def filter_regime_gt04(date, vix, vix_ratio, regime):
    """Only trade when GRU regime > 0.4 (bull regime)."""
    return regime > 0.4


def filter_contango(date, vix, vix_ratio, regime):
    """Only trade when VIX term structure is in contango (VIX/VIX3M < 0.95)."""
    return vix_ratio < 0.95


# ─── Correlation Analysis ────────────────────────────────────────────

def load_production_returns():
    """
    Load production v4 sector bull call spread monthly returns for correlation comparison.
    If not available, simulate a simple momentum spread baseline.
    """
    results_path = RESULTS_DIR / 'sector_etf_momentum_v4_results.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        # Check if monthly returns are stored
        if 'monthly_returns' in data:
            dates = pd.to_datetime(data['monthly_returns']['dates'])
            vals = data['monthly_returns']['values']
            return pd.Series(vals, index=dates, name='momentum_spread')

    # Fallback: load from momentum_debit_spread results
    alt_path = RESULTS_DIR / 'momentum_debit_spread_etf_v1_results.json'
    if alt_path.exists():
        with open(alt_path) as f:
            data = json.load(f)
        if 'monthly_returns' in data:
            dates = pd.to_datetime(data['monthly_returns']['dates'])
            vals = data['monthly_returns']['values']
            return pd.Series(vals, index=dates, name='momentum_spread')

    fprint("  No production spread returns found — will compute proxy from SPY momentum")
    return None


def compute_correlation(cal_returns, mom_returns):
    """Compute correlation between calendar and momentum spread monthly returns."""
    if mom_returns is None or len(cal_returns) < 6:
        return None

    # Align dates
    common = cal_returns.index.intersection(mom_returns.index)
    if len(common) < 6:
        return None

    c = cal_returns.loc[common]
    m = mom_returns.loc[common]

    corr = c.corr(m)
    return {
        'correlation': float(corr),
        'n_months': len(common),
        'interpretation': (
            'LOW correlation (diversifying!)' if abs(corr) < 0.3 else
            'MODERATE correlation (some diversification)' if abs(corr) < 0.6 else
            'HIGH correlation (NOT diversifying)'
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    fprint("=" * 70)
    fprint("CALENDAR SPREAD v1 — Diversification Research")
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    if MLFLOW_OK:
        mlflow.set_experiment("calendar_spread_v1")
        mlflow.start_run(run_name=f"cal_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # 1. Load data
    fprint("\n[1/6] Loading data...")
    close, high, low = download_data()

    vix_series = close['VIX'] if 'VIX' in close.columns else None
    vix3m_series = close['VIX3M'] if 'VIX3M' in close.columns else None
    spy_prices = close['SPY'] if 'SPY' in close.columns else None

    if vix_series is not None:
        fprint(f"  VIX range: {vix_series.min():.1f} - {vix_series.max():.1f}")
    if vix3m_series is not None:
        fprint(f"  VIX3M range: {vix3m_series.min():.1f} - {vix3m_series.max():.1f}")

    # 2. Load GRU regime
    fprint("\n[2/6] Loading regime predictions...")
    regime_scores = load_regime_predictions()
    if regime_scores is None and vix_series is not None:
        regime_scores = vix_proxy_regime(vix_series)
        fprint("  Using VIX proxy for regime")

    # 3. Build vol features
    fprint("\n[3/6] Building vol features for LGBM...")
    features_by_date = build_vol_features(close, high, low, vix_series, vix3m_series)
    fprint(f"  Feature dates: {len(features_by_date)}")

    # 4. LGBM walk-forward ranking
    fprint("\n[4/6] Running LGBM walk-forward ranking...")
    lgbm_rankings = lgbm_wf_rank(features_by_date, close)

    # 5. Run all 5 variants
    fprint("\n[5/6] Running backtest variants...")

    variants = {
        'A': ('LGBM Top3 + VIX>20', select_lgbm_top3, filter_vix_gt20),
        'B': ('LGBM Top3 + Regime>0.4', select_lgbm_top3, filter_regime_gt04),
        'C': ('Lowest Vol 3 (no filter)', select_lowest_vol, None),
        'D': ('All 11 Sectors (baseline)', select_all_sectors, None),
        'E': ('LGBM Top3 + Contango', select_lgbm_top3, filter_contango),
    }

    results = {}
    for key, (name, selector, filt) in variants.items():
        result = run_variant(
            f"{key}: {name}", close, high, low,
            vix_series, vix3m_series,
            features_by_date, lgbm_rankings, regime_scores,
            sector_selector=selector,
            filter_func=filt,
        )
        if result is not None:
            results[key] = result

    if not results:
        fprint("\nNO VARIANTS PRODUCED TRADES")
        if MLFLOW_OK:
            mlflow.log_param("status", "NO_TRADES")
            mlflow.end_run()
        return

    # 6. Analysis and validation
    fprint("\n[6/6] Analysis and validation...")
    fprint("=" * 70)

    mom_returns = load_production_returns()

    best_key = None
    best_sharpe = -999
    variant_summaries = {}

    for key, result in results.items():
        name = result['variant']
        trades = result['trades']
        monthly_rets = result['monthly_returns']

        # Run adversarial validation
        val_result = validate_trades(
            trades,
            initial_capital=STARTING_CAPITAL,
            spy_prices=spy_prices,
            strategy_name=name,
        )
        val_result.print_summary()

        # Correlation with momentum spreads
        corr_info = compute_correlation(monthly_rets, mom_returns)
        if corr_info:
            fprint(f"\n  Correlation with momentum spread: {corr_info['correlation']:.3f}")
            fprint(f"  ({corr_info['n_months']} overlapping months)")
            fprint(f"  => {corr_info['interpretation']}")

        # Ticker breakdown
        tdf = pd.DataFrame(trades)
        fprint(f"\n  Ticker breakdown:")
        for ticker, g in tdf.groupby('ticker'):
            if len(g) >= 3:
                wr = (g['pnl'] > 0).mean()
                fprint(f"    {ticker}: {len(g)}t, WR {wr:.0%}, "
                       f"PnL ${g['pnl'].sum():.0f}, "
                       f"avg |move| {g['price_move'].abs().mean():.1%}")

        # Win rate by VIX level
        tdf['vix_bucket'] = pd.cut(tdf['vix'], bins=[0, 15, 20, 25, 100],
                                   labels=['<15', '15-20', '20-25', '>25'])
        fprint(f"\n  WR by VIX level:")
        for bucket, g in tdf.groupby('vix_bucket', observed=True):
            if len(g) >= 3:
                wr = (g['pnl'] > 0).mean()
                fprint(f"    VIX {bucket}: {len(g)}t, WR {wr:.0%}, avg PnL ${g['pnl'].mean():.1f}")

        # Win rate by price move magnitude
        tdf['move_bucket'] = pd.cut(tdf['price_move'].abs(),
                                    bins=[0, 0.02, 0.05, 0.10, 1.0],
                                    labels=['<2%', '2-5%', '5-10%', '>10%'])
        fprint(f"\n  WR by |price move| (calendar sweet spot = small moves):")
        for bucket, g in tdf.groupby('move_bucket', observed=True):
            if len(g) >= 3:
                wr = (g['pnl'] > 0).mean()
                fprint(f"    |move| {bucket}: {len(g)}t, WR {wr:.0%}, avg PnL ${g['pnl'].mean():.1f}")

        summary = {
            'variant': name,
            'n_trades': val_result.n_trades,
            'sharpe': val_result.sharpe,
            'sortino': val_result.sortino,
            'cagr': val_result.cagr,
            'max_dd': val_result.max_dd,
            'win_rate': val_result.win_rate,
            'profit_factor': val_result.profit_factor,
            'final_equity': val_result.final_equity,
            'gates_passed': val_result.gates_passed,
            'gates_total': val_result.gates_total,
            'correlation_with_momentum': corr_info if corr_info else None,
        }
        variant_summaries[key] = summary

        if val_result.sharpe > best_sharpe:
            best_sharpe = val_result.sharpe
            best_key = key

    # Summary table
    fprint(f"\n{'='*90}")
    fprint(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'CAGR':>8} "
           f"{'MaxDD':>8} {'PF':>6} {'Gates':>6} {'Final':>8}")
    fprint("-" * 90)
    for key in sorted(variant_summaries.keys()):
        s = variant_summaries[key]
        fprint(f"{s['variant']:<35} {s['sharpe']:>7.2f} {s['sortino']:>8.2f} "
               f"{s['win_rate']:>5.1%} {s['cagr']:>7.1%} {s['max_dd']:>7.1%} "
               f"{s['profit_factor']:>6.2f} {s['gates_passed']:>2}/{s['gates_total']:<2} "
               f"${s['final_equity']:>7.0f}")

    # Best variant
    if best_key:
        best = variant_summaries[best_key]
        fprint(f"\n{'='*70}")
        fprint(f"BEST VARIANT: {best['variant']}")
        fprint(f"  Sharpe:  {best['sharpe']:.2f}")
        fprint(f"  Sortino: {best['sortino']:.2f}")
        fprint(f"  CAGR:    {best['cagr']:.1%}")
        fprint(f"  MaxDD:   {best['max_dd']:.1%}")
        fprint(f"  WR:      {best['win_rate']:.1%}")
        fprint(f"  PF:      {best['profit_factor']:.2f}")
        fprint(f"  Final:   ${best['final_equity']:.0f}")
        fprint(f"  Gates:   {best['gates_passed']}/{best['gates_total']}")

        if best.get('correlation_with_momentum'):
            ci = best['correlation_with_momentum']
            fprint(f"  Corr w/ momentum: {ci['correlation']:.3f} => {ci['interpretation']}")

    # Correlation matrix across variants
    if len(results) >= 2:
        fprint(f"\n{'='*70}")
        fprint("INTER-VARIANT CORRELATION MATRIX (monthly returns)")
        fprint("-" * 70)
        keys = sorted(results.keys())
        header = f"{'':>6}" + "".join(f"{k:>8}" for k in keys)
        fprint(header)
        for k1 in keys:
            row = f"{k1:>6}"
            for k2 in keys:
                r1 = results[k1]['monthly_returns']
                r2 = results[k2]['monthly_returns']
                common = r1.index.intersection(r2.index)
                if len(common) >= 3:
                    c = r1.loc[common].corr(r2.loc[common])
                    row += f"{c:>8.3f}"
                else:
                    row += f"{'N/A':>8}"
            fprint(row)

    # KEY DIVERSIFICATION QUESTION
    fprint(f"\n{'='*70}")
    fprint("KEY QUESTION: Are calendar spreads diversifying vs directional spreads?")
    fprint("-" * 70)
    if mom_returns is not None and best_key:
        ci = variant_summaries[best_key].get('correlation_with_momentum')
        if ci:
            fprint(f"  Correlation: {ci['correlation']:.3f}")
            fprint(f"  Verdict: {ci['interpretation']}")
            if abs(ci['correlation']) < 0.3:
                fprint("  => YES, calendar spreads offer genuine diversification")
                fprint("     Different return driver (theta/vol) vs direction")
            elif abs(ci['correlation']) < 0.6:
                fprint("  => PARTIAL diversification benefit")
            else:
                fprint("  => NO, too correlated with momentum spreads")
        else:
            fprint("  Insufficient overlapping data for correlation test")
    else:
        fprint("  No momentum spread returns available for comparison")
        fprint("  Calendar spread returns are driven by theta decay + vol structure")
        fprint("  Theoretically orthogonal to directional momentum")

    # Save results
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'starting_capital': STARTING_CAPITAL,
        'near_dte': NEAR_DTE,
        'far_dte': FAR_DTE,
        'best_variant': best_key,
        'variants': {},
    }
    for key, summary in variant_summaries.items():
        save_data['variants'][key] = summary
        # Include monthly returns for future correlation analysis
        if key in results:
            mr = results[key]['monthly_returns']
            save_data['variants'][key]['monthly_returns'] = {
                'dates': [str(d.date()) for d in mr.index],
                'values': [float(v) for v in mr.values],
            }

    save_path = RESULTS_DIR / 'calendar_spread_v1_results.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {save_path}")

    # MLflow logging
    if MLFLOW_OK:
        mlflow.log_param("near_dte", NEAR_DTE)
        mlflow.log_param("far_dte", FAR_DTE)
        mlflow.log_param("starting_capital", STARTING_CAPITAL)
        mlflow.log_param("best_variant", best_key)
        mlflow.log_param("n_variants", len(variant_summaries))

        for key, summary in variant_summaries.items():
            for metric in ['sharpe', 'sortino', 'cagr', 'max_dd', 'win_rate',
                          'profit_factor', 'final_equity']:
                mlflow.log_metric(f"{key}_{metric}", summary[metric])
            mlflow.log_metric(f"{key}_gates", summary['gates_passed'])
            if summary.get('correlation_with_momentum'):
                mlflow.log_metric(f"{key}_corr_momentum",
                                summary['correlation_with_momentum']['correlation'])

        if best_key:
            for metric in ['sharpe', 'sortino', 'cagr', 'max_dd', 'win_rate',
                          'profit_factor', 'final_equity']:
                mlflow.log_metric(f"best_{metric}", variant_summaries[best_key][metric])

        mlflow.log_artifact(str(save_path))
        mlflow.end_run()
        fprint("MLflow run logged")

    fprint(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return save_data


if __name__ == '__main__':
    main()
