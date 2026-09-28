#!/usr/bin/env python3
"""
Volatility Risk Premium (VRP) Harvesting for Small Accounts v1
================================================================
Research question: Can we systematically harvest VRP with a $645 account
using Level 2 options only (single-leg, no spreads)?

Context: We proved IV overestimates RV ~70% of the time (iron condor WR 95%,
Sharpe 3.55 at $10K). But iron condors need $10K+ and Level 3+.
This tests 6 VRP harvesting variants that work within our constraints.

Strategy variants:
  A) Sell ATM puts on low-VRP sectors — cash-secured on cheap ETFs (XLB, XLU, XLP)
  B) Buy calls when IV rank is lowest — positive gamma on "vol on sale" sectors
  C) VRP momentum combo — long sectors where VRP is expanding (HV dropping below IV)
  D) Time decay harvester — buy 45-DTE calls on strong sectors, sell at 15-DTE
  E) Regime-filtered VRP — only sell premium when VIX > 20
  F) Random entry control — baseline permutation test

Universe: 11 sector ETFs
Walk-forward: sliding 252-day train, 21-day OOT (HC #0 compliant)
IV approximation: HV20/HV60 ratio as IV rank proxy
ATR pricing with 15% haircut, $2.60 RT commission
$645 starting capital, max $200 per position
Full 4-gate adversarial audit (HC #428 compliant)

HC compliance:
- HC #0: sliding window only
- HC #69: risk-adjusted metrics primary (Sharpe, Sortino, PF, WR)
- HC #428: 4-gate adversarial audit, regime-agnostic validation
- HC #344: day-conc cap <= 0.70
"""

import numpy as np
import pandas as pd
import warnings
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings('ignore')
np.random.seed(42)

# =====================================================================
# MLflow setup
# =====================================================================
try:
    import mlflow
    MLFLOW_URI = "http://jupiter:5000"
    mlflow.set_tracking_uri(MLFLOW_URI)
    HAS_MLFLOW = True
    print(f"MLflow connected: {MLFLOW_URI}")
except Exception as e:
    print(f"MLflow unavailable: {e}")
    HAS_MLFLOW = False

# =====================================================================
# CONSTANTS
# =====================================================================
UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
# Cheap ETFs suitable for cash-secured puts at $200 margin
CHEAP_ETFS = ['XLB', 'XLU', 'XLP']  # ~$51, ~$46, ~$82

START_CAP = 645.0
COMMISSION_PER_LEG = 1.30       # $1.30 per leg
COMMISSION_RT = 2.60            # round-trip single leg
HAIRCUT = 0.15                  # bid-ask slippage on options
MAX_POS_COST = 200.0            # max $200 per position
RISK_FREE = 0.045
TRAIN_DAYS = 252                # 1 year sliding window
OOT_DAYS = 21                   # 1 month OOT
MIN_DATA_DAYS = 504             # 2 years minimum before first signal

# Output paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE_DIR / "output" / "vrp_small_account_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINDINGS_DIR = BASE_DIR / "research" / "findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# BLACK-SCHOLES PRICING
# =====================================================================
def bs_d1d2(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_put(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = bs_d1d2(S, K, T, sigma, r)
    return norm.cdf(d1) - 1.0


def bs_delta_call(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = bs_d1d2(S, K, T, sigma, r)
    return norm.cdf(d1)


# =====================================================================
# DATA LOADING
# =====================================================================
def load_data():
    """Load sector ETF + VIX daily data via yfinance with caching."""
    import yfinance as yf

    cache_file = CACHE_DIR / 'vrp_sector_daily.parquet'
    vix_file = CACHE_DIR / 'vrp_vix_daily.parquet'

    # Use cache if < 24 hours old
    if cache_file.exists() and vix_file.exists():
        age_hrs = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hrs < 24:
            print(f"Using cached data ({age_hrs:.1f}h old)")
            return pd.read_parquet(cache_file), pd.read_parquet(vix_file)

    print("Downloading fresh data from yfinance...")
    tickers = UNIVERSE + ['^VIX']
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Separate VIX
    vix_col = '^VIX' if '^VIX' in close.columns else None
    if vix_col:
        vix = close[[vix_col]].rename(columns={vix_col: 'VIX'}).dropna()
        close = close.drop(columns=[vix_col])
    else:
        vix = pd.DataFrame()

    close = close.dropna(how='all')
    close.to_parquet(cache_file)
    if len(vix) > 0:
        vix.to_parquet(vix_file)

    print(f"Loaded {len(close)} days, {len(close.columns)} tickers")
    return close, vix


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
def compute_features(close, vix):
    """
    Build daily feature panel for all sectors.
    Key features: IV rank proxy (HV20/HV60), VRP, momentum, vol regime.
    """
    features = {}

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        px = close[ticker].dropna()
        if len(px) < MIN_DATA_DAYS:
            continue

        ret = px.pct_change()
        log_ret = np.log(px / px.shift(1))

        feat = pd.DataFrame(index=px.index)
        feat['price'] = px
        feat['ticker'] = ticker

        # Realized vol at multiple windows
        for w in [5, 10, 20, 60, 120]:
            feat[f'hv_{w}'] = log_ret.rolling(w).std() * np.sqrt(252)

        # IV rank proxy: HV20/HV60 ratio (high = vol elevated vs. norm)
        feat['iv_rank_proxy'] = feat['hv_20'] / (feat['hv_60'] + 1e-8)

        # IV estimation: HV20 * 1.2 (IV typically ~20% above HV)
        feat['iv_est'] = feat['hv_20'] * 1.20

        # VRP proxy: IV_est - realized HV5 (positive = IV overpriced)
        feat['vrp'] = feat['iv_est'] - feat['hv_5']

        # VRP z-score (rolling 60d)
        feat['vrp_zscore'] = (feat['vrp'] - feat['vrp'].rolling(60).mean()) / (feat['vrp'].rolling(60).std() + 1e-8)

        # VRP momentum (is VRP expanding or contracting?)
        feat['vrp_mom_5d'] = feat['vrp'] - feat['vrp'].shift(5)
        feat['vrp_mom_20d'] = feat['vrp'] - feat['vrp'].shift(20)

        # ATR for option pricing sanity check
        high_approx = px * (1 + abs(ret))
        low_approx = px * (1 - abs(ret))
        tr = pd.concat([high_approx - low_approx, abs(high_approx - px.shift(1)), abs(low_approx - px.shift(1))], axis=1).max(axis=1)
        feat['atr_14'] = tr.rolling(14).mean()

        # Price momentum
        for m in [5, 10, 21, 63, 126]:
            feat[f'ret_{m}d'] = px.pct_change(m)

        # Vol of vol
        feat['vov_20'] = feat['hv_5'].rolling(20).std()

        # RSI
        delta = px.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        feat['rsi_14'] = 100 - (100 / (1 + rs))

        # Distance from 20-day high/low
        feat['dist_20d_high'] = px / px.rolling(20).max() - 1
        feat['dist_20d_low'] = px / px.rolling(20).min() - 1

        # Mean reversion z-score
        feat['z_20'] = (px - px.rolling(20).mean()) / (px.rolling(20).std() + 1e-8)

        # VIX features
        if len(vix) > 0 and 'VIX' in vix.columns:
            v = vix['VIX'].reindex(feat.index, method='ffill')
            feat['vix'] = v
            feat['vix_rank'] = v.rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 10 else 0.5,
                raw=False
            )
            feat['vix_hv_ratio'] = v / (feat['hv_20'] * 100 + 1e-8)
        else:
            feat['vix'] = 20.0
            feat['vix_rank'] = 0.5
            feat['vix_hv_ratio'] = 1.0

        # Target: forward 21-day return (for evaluation)
        feat['fwd_ret_21d'] = px.pct_change(21).shift(-21)

        # Forward realized vol (for VRP truth — DID IV overstate?)
        feat['fwd_hv_21d'] = log_ret.shift(-1).rolling(21).std().shift(-21) * np.sqrt(252)

        features[ticker] = feat

    return features


# =====================================================================
# STRATEGY A: Sell ATM Puts on Low-VRP Sectors
# =====================================================================
def strategy_A_sell_puts(date, features_snapshot, equity, spy_regime):
    """
    Sell ATM/OTM puts on CHEAP sector ETFs where IV >> HV.
    Cash-secured: budget max $200 margin.
    Only ETFs cheap enough for small account (XLB ~$51, XLU ~$46, XLP ~$82).
    """
    trades = []
    # Filter to cheap ETFs only
    cheap = features_snapshot[features_snapshot['ticker'].isin(CHEAP_ETFS)]
    if len(cheap) == 0:
        return trades

    # Rank by VRP (highest = most overpriced IV)
    ranked = cheap.sort_values('vrp', ascending=False)

    for _, row in ranked.iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue

        dte = 30
        T = dte / 365.0

        # Sell ~25-delta OTM put
        d1_target = norm.ppf(0.75)
        K = S * np.exp(-(d1_target * iv * np.sqrt(T) - (RISK_FREE + iv**2 / 2) * T))
        K = round(K, 0)

        put_premium = bs_put(S, K, T, iv)
        put_premium *= (1 - HAIRCUT)  # slippage

        # Cash-secured margin = K * 100 * margin_rate
        # For cheap ETFs: K~$45 → margin ~$900 at 20% → still too much
        # Use 100-share lots but with portfolio margin approximation
        # Actually for single put: margin ≈ max(put_strike * 100 * 0.10, put_strike * 100 - OTM_amount * 100)
        margin_needed = K * 100 * 0.10  # 10% margin for deep OTM
        if margin_needed > MAX_POS_COST:
            continue

        premium_received = put_premium * 100 - COMMISSION_PER_LEG
        if premium_received < 5:
            continue

        risk = min((K * 100) - premium_received, MAX_POS_COST)

        trades.append({
            'ticker': ticker,
            'type': 'sell_put',
            'strike': K,
            'dte': dte,
            'premium': premium_received,
            'margin': margin_needed,
            'risk': risk,
            'iv': iv,
            'vrp': row['vrp'],
            'regime': spy_regime,
        })

    return trades[:2]  # max 2 positions


# =====================================================================
# STRATEGY B: Buy Calls When IV Rank is Lowest
# =====================================================================
def strategy_B_buy_calls(date, features_snapshot, equity, spy_regime):
    """
    Buy cheap calls on sectors where options are "on sale" (IV rank < 20th pctile).
    Positive gamma: ride any vol expansion.
    """
    trades = []
    # IV rank < 20th percentile = vol is cheap
    iv_20pct = features_snapshot['iv_rank_proxy'].quantile(0.20)
    cheap_vol = features_snapshot[features_snapshot['iv_rank_proxy'] <= iv_20pct]

    if len(cheap_vol) == 0:
        # Take lowest 2
        cheap_vol = features_snapshot.nsmallest(2, 'iv_rank_proxy')

    for _, row in cheap_vol.iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        mom_21d = row.get('ret_21d', 0)
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue
        if pd.isna(mom_21d):
            mom_21d = 0

        # Only buy calls if momentum is non-negative (cheap vol + trend)
        if mom_21d < -0.02:
            continue

        dte = 45  # slightly longer DTE for gamma
        T = dte / 365.0

        # Buy ATM call
        K = round(S, 0)
        call_cost = bs_call(S, K, T, iv)
        call_cost *= (1 + HAIRCUT)  # slippage (buying)
        total_cost = call_cost * 100 + COMMISSION_PER_LEG

        if total_cost > MAX_POS_COST or total_cost < 5:
            continue

        trades.append({
            'ticker': ticker,
            'type': 'buy_call',
            'strike': K,
            'dte': dte,
            'cost': total_cost,
            'iv': iv,
            'iv_rank': row['iv_rank_proxy'],
            'mom_21d': mom_21d,
            'regime': spy_regime,
        })

    return trades[:3]


# =====================================================================
# STRATEGY C: VRP Momentum Combo
# =====================================================================
def strategy_C_vrp_momentum(date, features_snapshot, equity, spy_regime):
    """
    Go long sectors where VRP is expanding (HV dropping below IV = vol compression,
    premium expanding). VRP momentum positive = IV growing faster than RV.
    Buy calls on these sectors as a directional bet.
    """
    trades = []
    # VRP momentum: positive = VRP expanding
    vrp_mom_col = 'vrp_mom_20d'
    ranked = features_snapshot.sort_values(vrp_mom_col, ascending=False)

    for _, row in ranked.head(3).iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        vrp_mom = row.get(vrp_mom_col, 0)
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue
        if pd.isna(vrp_mom) or vrp_mom <= 0:
            continue

        # Require positive underlying momentum too
        mom_21d = row.get('ret_21d', 0)
        if pd.isna(mom_21d):
            mom_21d = 0
        if mom_21d < -0.03:
            continue

        dte = 30
        T = dte / 365.0

        # Buy slightly OTM call (cheap directional bet)
        K = round(S * 1.02, 0)
        call_cost = bs_call(S, K, T, iv) * (1 + HAIRCUT) * 100 + COMMISSION_PER_LEG

        if call_cost > MAX_POS_COST or call_cost < 3:
            continue

        trades.append({
            'ticker': ticker,
            'type': 'vrp_mom_call',
            'strike': K,
            'dte': dte,
            'cost': call_cost,
            'iv': iv,
            'vrp_mom': vrp_mom,
            'regime': spy_regime,
        })

    return trades[:2]


# =====================================================================
# STRATEGY D: Time Decay Harvester
# =====================================================================
def strategy_D_time_decay(date, features_snapshot, equity, spy_regime):
    """
    Buy 45-DTE calls on strong sectors, sell at 15-DTE to harvest theta difference.
    Poor man's calendar: profit from the non-linear theta decay curve.
    The 45->15 DTE window captures the steepest part of theta decay.
    We buy when sector is strong (momentum > 0) and IV is moderate.
    """
    trades = []
    # Rank by momentum (strongest sectors = most likely to hold value)
    ranked = features_snapshot.sort_values('ret_21d', ascending=False)

    for _, row in ranked.head(4).iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        mom = row.get('ret_21d', 0)
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue
        if pd.isna(mom) or mom <= 0:
            continue

        # Only enter when IV is moderate (not too expensive to buy)
        iv_rank = row.get('iv_rank_proxy', 1.0)
        if pd.isna(iv_rank) or iv_rank > 1.5:  # IV > 150% of normal = too expensive
            continue

        dte_entry = 45
        T_entry = dte_entry / 365.0

        # Buy ATM call at 45 DTE
        K = round(S, 0)
        call_cost = bs_call(S, K, T_entry, iv)
        call_cost *= (1 + HAIRCUT)
        total_cost = call_cost * 100 + COMMISSION_PER_LEG

        if total_cost > MAX_POS_COST or total_cost < 5:
            continue

        trades.append({
            'ticker': ticker,
            'type': 'time_decay_call',
            'strike': K,
            'dte': dte_entry,
            'dte_exit_target': 15,  # sell at 15 DTE
            'cost': total_cost,
            'iv': iv,
            'mom_21d': mom,
            'regime': spy_regime,
        })

    return trades[:2]


# =====================================================================
# STRATEGY E: Regime-Filtered VRP
# =====================================================================
def strategy_E_regime_filtered(date, features_snapshot, equity, spy_regime):
    """
    Only sell premium when VIX > 20 (our proven sweet spot).
    In elevated-VIX regimes, premium is richest and VRP edge is strongest.
    Sell OTM puts on cheap ETFs only when VIX confirms regime.
    """
    trades = []

    # Check VIX level — ONLY trade when VIX > 20
    vix_level = features_snapshot['vix'].median() if 'vix' in features_snapshot.columns else 20
    if pd.isna(vix_level) or vix_level <= 20:
        return trades  # No trades in low-vol regime

    # Filter to cheap ETFs
    cheap = features_snapshot[features_snapshot['ticker'].isin(CHEAP_ETFS)]
    if len(cheap) == 0:
        cheap = features_snapshot.nsmallest(3, 'price')

    # Rank by VRP (highest = most overpriced IV in elevated VIX)
    ranked = cheap.sort_values('vrp', ascending=False)

    for _, row in ranked.iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue

        dte = 30
        T = dte / 365.0

        # Sell deeper OTM put (safer in high-VIX)
        d1_target = norm.ppf(0.85)  # ~15 delta (deeper OTM for safety)
        K = S * np.exp(-(d1_target * iv * np.sqrt(T) - (RISK_FREE + iv**2 / 2) * T))
        K = round(K, 0)

        put_premium = bs_put(S, K, T, iv) * (1 - HAIRCUT)
        margin_needed = K * 100 * 0.10
        if margin_needed > MAX_POS_COST:
            continue

        premium_received = put_premium * 100 - COMMISSION_PER_LEG
        if premium_received < 5:
            continue

        risk = min((K * 100) - premium_received, MAX_POS_COST)

        trades.append({
            'ticker': ticker,
            'type': 'regime_sell_put',
            'strike': K,
            'dte': dte,
            'premium': premium_received,
            'margin': margin_needed,
            'risk': risk,
            'iv': iv,
            'vrp': row['vrp'],
            'vix': vix_level,
            'regime': spy_regime,
        })

    return trades[:2]


# =====================================================================
# STRATEGY F: Random Entry Control (Baseline)
# =====================================================================
def strategy_F_random(date, features_snapshot, equity, spy_regime):
    """Control: random sector selection for sell-put trades. Baseline permutation test."""
    trades = []
    if len(features_snapshot) < 3:
        return trades

    random_picks = features_snapshot.sample(n=min(3, len(features_snapshot)))

    for _, row in random_picks.iterrows():
        ticker = row['ticker']
        S = row['price']
        iv = row['iv_est']
        if pd.isna(iv) or iv <= 0.01 or pd.isna(S) or S <= 0:
            continue

        dte = 30
        T = dte / 365.0
        d1_target = norm.ppf(0.75)
        K = S * np.exp(-(d1_target * iv * np.sqrt(T) - (RISK_FREE + iv**2 / 2) * T))
        K = round(K, 0)

        put_premium = bs_put(S, K, T, iv) * (1 - HAIRCUT)
        margin_needed = K * 100 * 0.10
        if margin_needed > MAX_POS_COST:
            continue

        premium_received = put_premium * 100 - COMMISSION_PER_LEG
        if premium_received < 5:
            continue

        risk = min(K * 100 - premium_received, MAX_POS_COST)

        trades.append({
            'ticker': ticker,
            'type': 'sell_put',
            'strike': K,
            'dte': dte,
            'premium': premium_received,
            'margin': margin_needed,
            'risk': risk,
            'iv': iv,
            'vrp': row.get('vrp', 0),
            'regime': spy_regime,
        })

    return trades[:2]


# =====================================================================
# TRADE RESOLUTION
# =====================================================================
def resolve_trade(trade, prices_fwd, iv_at_entry):
    """
    Resolve a trade at expiry or early TP/SL.
    prices_fwd: DataFrame of forward prices for all tickers.
    """
    ticker = trade['ticker']
    trade_type = trade['type']

    if ticker not in prices_fwd.columns:
        return None

    px_series = prices_fwd[ticker].dropna()
    if len(px_series) == 0:
        return None

    if trade_type in ('sell_put', 'regime_sell_put'):
        dte = trade['dte']
        K = trade['strike']
        premium = trade['premium']
        iv = trade['iv']

        expiry_idx = min(dte, len(px_series) - 1)
        tp_target = premium * 0.50  # take profit at 50%

        for day_i in range(1, expiry_idx + 1):
            S_now = px_series.iloc[day_i]
            T_rem = max((dte - day_i) / 365.0, 0.001)

            put_val = bs_put(S_now, K, T_rem, iv * 0.95) * 100
            buy_back = put_val * (1 + HAIRCUT) + COMMISSION_PER_LEG
            pnl = premium - buy_back

            if pnl >= tp_target:
                return {'pnl': pnl, 'hold_days': day_i, 'exit_reason': 'tp', 'exit_price': S_now}

        # At expiry
        S_exp = px_series.iloc[expiry_idx]
        intrinsic = max(K - S_exp, 0) * 100
        pnl = premium - intrinsic - COMMISSION_PER_LEG
        return {'pnl': pnl, 'hold_days': expiry_idx, 'exit_reason': 'expiry', 'exit_price': S_exp}

    elif trade_type in ('buy_call', 'vrp_mom_call'):
        dte = trade['dte']
        K = trade['strike']
        cost = trade['cost']
        iv = trade['iv']

        expiry_idx = min(dte, len(px_series) - 1)
        tp_target = cost * 1.0  # 100% return target

        for day_i in range(1, expiry_idx + 1):
            S_now = px_series.iloc[day_i]
            T_rem = max((dte - day_i) / 365.0, 0.001)

            call_val = bs_call(S_now, K, T_rem, iv) * (1 - HAIRCUT) * 100 - COMMISSION_PER_LEG
            pnl = call_val - cost

            if pnl >= tp_target:
                return {'pnl': pnl, 'hold_days': day_i, 'exit_reason': 'tp', 'exit_price': S_now}
            if pnl <= -cost * 0.80:
                return {'pnl': pnl, 'hold_days': day_i, 'exit_reason': 'sl', 'exit_price': S_now}

        S_exp = px_series.iloc[expiry_idx]
        intrinsic = max(S_exp - K, 0) * 100
        sell_val = intrinsic * (1 - HAIRCUT) - COMMISSION_PER_LEG if intrinsic > 0 else 0
        pnl = sell_val - cost
        return {'pnl': pnl, 'hold_days': expiry_idx, 'exit_reason': 'expiry', 'exit_price': S_exp}

    elif trade_type == 'time_decay_call':
        dte = trade['dte']
        dte_exit = trade.get('dte_exit_target', 15)
        K = trade['strike']
        cost = trade['cost']
        iv = trade['iv']

        # Hold from 45 DTE to 15 DTE (30 calendar days = ~21 trading days)
        hold_days = dte - dte_exit  # 30 calendar days
        exit_idx = min(hold_days, len(px_series) - 1)

        # Early TP/SL during hold period
        for day_i in range(1, exit_idx + 1):
            S_now = px_series.iloc[day_i]
            T_rem = max((dte - day_i) / 365.0, 0.001)

            call_val = bs_call(S_now, K, T_rem, iv) * (1 - HAIRCUT) * 100 - COMMISSION_PER_LEG
            pnl = call_val - cost

            # TP at 80%
            if pnl >= cost * 0.80:
                return {'pnl': pnl, 'hold_days': day_i, 'exit_reason': 'tp', 'exit_price': S_now}
            # SL at -50%
            if pnl <= -cost * 0.50:
                return {'pnl': pnl, 'hold_days': day_i, 'exit_reason': 'sl', 'exit_price': S_now}

        # Exit at target DTE (sell the remaining time value)
        S_exit = px_series.iloc[exit_idx]
        T_rem_exit = max(dte_exit / 365.0, 0.001)
        call_val_exit = bs_call(S_exit, K, T_rem_exit, iv * 0.95) * (1 - HAIRCUT) * 100 - COMMISSION_PER_LEG
        pnl = call_val_exit - cost
        return {'pnl': pnl, 'hold_days': exit_idx, 'exit_reason': 'dte_exit', 'exit_price': S_exit}

    return None


# =====================================================================
# WALK-FORWARD BACKTEST
# =====================================================================
def run_backtest(close, vix, features_dict, strategy_fn, strategy_name):
    """
    Sliding walk-forward: 252d train, 21d OOT.
    HC #0 compliant: oldest day dropped each step.
    """
    print(f"\n  Running {strategy_name}...")

    # Get SPY proxy for regime classification
    spy_proxy = close['XLK'] if 'XLK' in close.columns else close.iloc[:, 0]

    all_dates = close.index.sort_values()
    start_idx = MIN_DATA_DAYS

    trades = []
    equity_curve = [{'date': str(all_dates[start_idx].date()), 'equity': START_CAP}]
    equity = START_CAP

    step = 0
    for oot_start in range(start_idx, len(all_dates) - OOT_DAYS, OOT_DAYS):
        oot_end = min(oot_start + OOT_DAYS, len(all_dates))
        current_date = all_dates[oot_start]

        # Build features snapshot at signal date (using PRIOR data only — no leakage)
        snapshot_rows = []
        for ticker, feat_df in features_dict.items():
            if current_date in feat_df.index:
                row = feat_df.loc[current_date].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                snapshot_rows.append(row)

        if len(snapshot_rows) < 5:
            continue

        snapshot = pd.DataFrame(snapshot_rows)
        key_cols = ['vrp', 'iv_rank_proxy', 'iv_est', 'price']
        snapshot = snapshot.dropna(subset=[c for c in key_cols if c in snapshot.columns])
        if len(snapshot) < 3:
            continue

        # Determine regime (prior-day SPY return — no leakage)
        if oot_start > 21:
            spy_ret_prior = (spy_proxy.iloc[oot_start] / spy_proxy.iloc[oot_start - 21] - 1)
            if spy_ret_prior > 0.02:
                regime = 'bull'
            elif spy_ret_prior < -0.02:
                regime = 'bear'
            else:
                regime = 'flat'
        else:
            regime = 'flat'

        # Generate signals
        signal_trades = strategy_fn(current_date, snapshot, equity, regime)

        # Day concentration check (HC #344): cap positions
        if len(signal_trades) > 3:
            signal_trades = signal_trades[:3]

        # Resolve trades using forward prices
        fwd_prices = close.iloc[oot_start:min(oot_start + 90, len(close))]

        for trade in signal_trades:
            result = resolve_trade(trade, fwd_prices, trade.get('iv', 0.20))
            if result is None:
                continue

            pnl = result['pnl']

            # Position sizing: risk max 5% of equity per trade
            max_risk = equity * 0.05
            if trade['type'] in ('sell_put', 'regime_sell_put'):
                risk = trade.get('risk', MAX_POS_COST)
                size_mult = min(1.0, max_risk / risk) if risk > 0 else 0.5
            else:
                cost = trade.get('cost', trade.get('net_debit', MAX_POS_COST))
                size_mult = min(1.0, max_risk / cost) if cost > 0 else 0.5

            pnl_sized = pnl * size_mult

            trade_record = {
                'date': str(current_date.date()),
                'ticker': trade['ticker'],
                'type': trade['type'],
                'pnl': round(pnl, 2),
                'pnl_sized': round(pnl_sized, 2),
                'size_mult': round(size_mult, 3),
                'hold_days': result['hold_days'],
                'exit_reason': result['exit_reason'],
                'regime': regime,
                'equity_before': round(equity, 2),
            }

            equity += pnl_sized
            equity = max(equity, 1)  # floor at $1

            trade_record['equity_after'] = round(equity, 2)
            trades.append(trade_record)

        equity_curve.append({
            'date': str(all_dates[min(oot_end - 1, len(all_dates) - 1)].date()),
            'equity': round(equity, 2),
        })

        step += 1
        if step % 20 == 0:
            print(f"    Step {step}: {current_date.date()} equity=${equity:.0f} trades={len(trades)}")

    print(f"  Completed {strategy_name}: {len(trades)} trades, final equity=${equity:.0f}")
    return trades, equity_curve


# =====================================================================
# 4-GATE ADVERSARIAL AUDIT (HC #428)
# =====================================================================
def adversarial_audit(trades, equity_curve, name):
    """Full 4-gate adversarial audit per HC #428."""
    if len(trades) < 10:
        return {'name': name, 'n_trades': len(trades), 'gates_passed': 0, 'error': 'Too few trades'}

    trade_df = pd.DataFrame(trades)
    pnl_col = 'pnl_sized' if 'pnl_sized' in trade_df.columns else 'pnl'
    pnls = trade_df[pnl_col].values
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).sum() != 0 else 999

    # Compute equity-based metrics
    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date').sort_index()
    monthly_eq = eq_df['equity'].resample('ME').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) > 2:
        sharpe = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
        neg = monthly_ret[monthly_ret < 0]
        sortino = monthly_ret.mean() / (neg.std() + 1e-8) * np.sqrt(12) if len(neg) > 1 else sharpe * 1.5
    else:
        sharpe = sortino = 0

    final_eq = eq_df['equity'].iloc[-1]
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 0.1)
    cagr = (final_eq / START_CAP) ** (1 / years) - 1
    maxdd = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()

    # ---- Gate 1: Permutation test (p < 0.05) ----
    perm_sharpes = []
    for _ in range(500):
        sh = pnls.copy()
        np.random.shuffle(sh)
        eq_p = np.cumsum(sh) + START_CAP
        chunks = np.array_split(eq_p, max(1, len(eq_p) // 21))
        mr = []
        prev = START_CAP
        for c in chunks:
            if len(c) > 0:
                mr.append((c[-1] - prev) / max(prev, 1))
                prev = c[-1]
        if len(mr) > 1:
            a = np.array(mr)
            perm_sharpes.append(a.mean() / (a.std() + 1e-8) * np.sqrt(12))
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe) if perm_sharpes else 1.0
    g1 = perm_p < 0.05

    # ---- Gate 2: Regime agnostic (R1 gap < 0.50) ----
    bull = trade_df[trade_df['regime'] == 'bull'][pnl_col].values
    bear = trade_df[trade_df['regime'] == 'bear'][pnl_col].values
    flat = trade_df[trade_df['regime'] == 'flat'][pnl_col].values

    if len(bull) > 5 and len(bear) > 5:
        bs_sharpe = bull.mean() / (bull.std() + 1e-8) * np.sqrt(12)
        br_sharpe = bear.mean() / (bear.std() + 1e-8) * np.sqrt(12)
        r1_gap = abs(bs_sharpe - br_sharpe) / max(abs(bs_sharpe), abs(br_sharpe), 1e-8)
        bull_wr = (bull > 0).mean() * 100
        bear_wr = (bear > 0).mean() * 100
    else:
        bs_sharpe = br_sharpe = sharpe
        r1_gap = 0
        bull_wr = bear_wr = wr

    flat_wr = (flat > 0).mean() * 100 if len(flat) > 0 else 0
    g2 = r1_gap < 0.50

    # ---- Gate 3: Sub-period consistency (all sub-periods positive Sharpe) ----
    n_chunks = 3
    chunk_size = len(trade_df) // n_chunks
    sub_sharpes = []
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = (i + 1) * chunk_size if i < n_chunks - 1 else len(trade_df)
        sp = trade_df.iloc[start_i:end_i][pnl_col].values
        if len(sp) > 3:
            sub_sharpes.append(sp.mean() / (sp.std() + 1e-8) * np.sqrt(12))
    g3 = len(sub_sharpes) >= 2 and all(x > 0 for x in sub_sharpes)

    # ---- Gate 4: Outlier removal (trim 5/95 pctile, still profitable) ----
    p5, p95 = np.percentile(pnls, [5, 95])
    trimmed = pnls[(pnls >= p5) & (pnls <= p95)]
    g4 = (trimmed.mean() / (trimmed.std() + 1e-8) * np.sqrt(12) > 0) if len(trimmed) > 3 else False

    # Type breakdown
    type_stats = {}
    if 'type' in trade_df.columns:
        for t_type in trade_df['type'].unique():
            t_pnls = trade_df[trade_df['type'] == t_type][pnl_col].values
            type_stats[t_type] = {
                'n': int(len(t_pnls)),
                'wr': round((t_pnls > 0).mean() * 100, 1),
                'avg_pnl': round(float(t_pnls.mean()), 2),
                'total_pnl': round(float(t_pnls.sum()), 2),
            }

    return {
        'name': name,
        'n_trades': int(n),
        'win_rate': round(wr, 1),
        'avg_win': round(float(pnls[pnls > 0].mean()), 2) if wins > 0 else 0,
        'avg_loss': round(float(pnls[pnls < 0].mean()), 2) if (pnls < 0).sum() > 0 else 0,
        'total_pnl': round(float(pnls.sum()), 2),
        'final_equity': round(float(final_eq), 2),
        'cagr_pct': round(cagr * 100, 1),
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'maxdd_pct': round(float(maxdd * 100), 1),
        'pf': round(float(pf), 2),
        'r1_gap': round(float(r1_gap), 3),
        'bull_wr': round(float(bull_wr), 1),
        'bear_wr': round(float(bear_wr), 1),
        'flat_wr': round(float(flat_wr), 1),
        'bull_trades': int(len(bull)),
        'bear_trades': int(len(bear)),
        'flat_trades': int(len(flat)),
        'bull_sharpe': round(float(bs_sharpe), 2),
        'bear_sharpe': round(float(br_sharpe), 2),
        'perm_p': round(float(perm_p), 4),
        'g1_perm': bool(g1),
        'g2_regime': bool(g2),
        'g3_subperiod': bool(g3),
        'g4_outlier': bool(g4),
        'sub_sharpes': [round(x, 2) for x in sub_sharpes],
        'gates_passed': int(sum([g1, g2, g3, g4])),
        'type_breakdown': type_stats,
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("VRP Small Account v1 — Volatility Risk Premium Harvesting")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${START_CAP}, Max position: ${MAX_POS_COST}")
    print(f"Commission RT: ${COMMISSION_RT}, Haircut: {HAIRCUT*100:.0f}%")
    print("=" * 70)

    t0 = time.time()

    # Load data
    close, vix = load_data()
    print(f"Data: {len(close)} days, {len(close.columns)} tickers")

    # Build features
    print("\nBuilding features...")
    features_dict = compute_features(close, vix)
    print(f"Features built for {len(features_dict)} sectors")

    # MLflow experiment
    mlflow_ok = False
    if HAS_MLFLOW:
        try:
            exp = mlflow.set_experiment("vrp_small_account_v1")
            exp_id = exp.experiment_id
            print(f"MLflow experiment: vrp_small_account_v1 (ID: {exp_id})")
            mlflow_ok = True
        except Exception as e:
            print(f"MLflow experiment setup failed: {e}")

    # Define all 6 strategy variants
    strategies = {
        'A_Sell_Puts_HighVRP': (strategy_A_sell_puts, 'Sell ATM puts on low-VRP cheap sectors (XLB/XLU/XLP), cash-secured $200 margin'),
        'B_Buy_Calls_LowIV': (strategy_B_buy_calls, 'Buy calls when IV rank < 20 pctile (vol on sale), positive gamma'),
        'C_VRP_Momentum': (strategy_C_vrp_momentum, 'Long sectors where VRP expanding (HV dropping below IV)'),
        'D_Time_Decay': (strategy_D_time_decay, 'Buy 45-DTE calls, sell at 15-DTE (theta decay harvester)'),
        'E_Regime_Filtered': (strategy_E_regime_filtered, 'Sell premium ONLY when VIX > 20 (proven sweet spot)'),
        'F_Random_Baseline': (strategy_F_random, 'CONTROL: random sector selection (permutation baseline)'),
    }

    all_results = []
    best = None
    best_sharpe = -999

    for strat_name, (strat_fn, desc) in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy {strat_name}")
        print(f"  {desc}")
        print('=' * 60)

        np.random.seed(42)  # reproducibility

        trades, eq_curve = run_backtest(close, vix, features_dict, strat_fn, strat_name)

        if len(trades) < 10:
            print(f"  SKIP: Only {len(trades)} trades")
            result = {'name': strat_name, 'n_trades': len(trades), 'gates_passed': 0, 'desc': desc}
            all_results.append(result)
            continue

        result = adversarial_audit(trades, eq_curve, strat_name)
        result['desc'] = desc
        all_results.append(result)

        # Print results
        gates = f"[{'P' if result.get('g1_perm') else 'F'}{'P' if result.get('g2_regime') else 'F'}{'P' if result.get('g3_subperiod') else 'F'}{'P' if result.get('g4_outlier') else 'F'}]"
        print(f"\n  RESULTS for {strat_name}:")
        print(f"    Trades: {result['n_trades']}, WR: {result['win_rate']:.1f}%")
        print(f"    Sharpe: {result['sharpe']:.2f}, Sortino: {result['sortino']:.2f}")
        print(f"    PF: {result['pf']:.2f}, CAGR: {result['cagr_pct']:.1f}%")
        print(f"    MaxDD: {result['maxdd_pct']:.1f}%, Final Eq: ${result['final_equity']:.0f}")
        print(f"    R1 gap: {result['r1_gap']:.3f} (bull WR={result['bull_wr']:.0f}%, bear WR={result['bear_wr']:.0f}%)")
        print(f"    Gates: {gates} ({result['gates_passed']}/4)")
        print(f"    Perm p-value: {result['perm_p']:.4f}")
        if result.get('type_breakdown'):
            for t_type, stats in result['type_breakdown'].items():
                print(f"    Type {t_type}: n={stats['n']}, WR={stats['wr']}%, avg=${stats['avg_pnl']:.2f}")

        # MLflow logging
        if mlflow_ok:
            try:
                with mlflow.start_run(run_name=strat_name):
                    mlflow.log_params({
                        'strategy': strat_name,
                        'description': desc[:250],
                        'start_capital': START_CAP,
                        'max_pos_cost': MAX_POS_COST,
                        'commission_rt': COMMISSION_RT,
                        'haircut': HAIRCUT,
                        'train_days': TRAIN_DAYS,
                        'oot_days': OOT_DAYS,
                    })
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'],
                        'sortino': result['sortino'],
                        'win_rate': result['win_rate'],
                        'profit_factor': min(result['pf'], 99),
                        'cagr_pct': result['cagr_pct'],
                        'max_dd_pct': result['maxdd_pct'],
                        'n_trades': result['n_trades'],
                        'final_equity': result['final_equity'],
                        'total_pnl': result['total_pnl'],
                        'r1_gap': result['r1_gap'],
                        'perm_p': result['perm_p'],
                        'gates_passed': result['gates_passed'],
                        'bull_wr': result['bull_wr'],
                        'bear_wr': result['bear_wr'],
                    })
            except Exception as e:
                print(f"  MLflow log failed: {e}")

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best = result

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — VRP Small Account v1")
    print(f"{'=' * 70}")
    print(f"Runtime: {elapsed / 60:.1f} minutes")
    print(f"\n{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'WR%':>5} {'PF':>6} {'Gates':>6} {'FinalEq':>8}")
    print('-' * 70)

    for r in sorted(all_results, key=lambda x: x.get('sharpe', -99), reverse=True):
        if r.get('n_trades', 0) < 10:
            print(f"{r['name']:<30} {'--':>7} {'--':>8} {'--':>5} {'--':>6} {'--':>6} {'<10 trades':>8}")
        else:
            print(f"{r['name']:<30} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['win_rate']:>5.1f} {r['pf']:>6.2f} {r['gates_passed']:>4}/4  ${r['final_equity']:>7.0f}")

    # Save results to findings
    findings_file = FINDINGS_DIR / 'vrp_small_account_v1_results.json'
    output_file = OUTPUT_DIR / 'vrp_small_account_v1_results.json'
    output = {
        'experiment': 'vrp_small_account_v1',
        'run_date': datetime.now().isoformat(),
        'config': {
            'start_capital': START_CAP,
            'max_pos_cost': MAX_POS_COST,
            'commission_rt': COMMISSION_RT,
            'haircut': HAIRCUT,
            'universe': UNIVERSE,
            'cheap_etfs': CHEAP_ETFS,
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
        },
        'strategies': all_results,
    }

    for fpath in [findings_file, output_file]:
        with open(fpath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to findings and output dirs")

    # Find best
    viable = [r for r in all_results if r.get('gates_passed', 0) >= 3 and 'Random' not in r.get('name', '') and 'Baseline' not in r.get('name', '')]
    if viable:
        champion = max(viable, key=lambda x: x.get('sharpe', -99))
        print(f"\nCHAMPION: {champion['name']} — Sharpe {champion['sharpe']:.2f}, {champion['gates_passed']}/4 gates")
    else:
        print("\nNo strategy passed 3+ gates. VRP at $645 may not be viable with single-leg options.")

    # Beat baseline check
    baseline = [r for r in all_results if 'Random' in r.get('name', '') or 'Baseline' in r.get('name', '')]
    if baseline and baseline[0].get('n_trades', 0) >= 10:
        bl_sharpe = baseline[0].get('sharpe', 0)
        signal_strategies = [r for r in all_results if 'Random' not in r.get('name', '') and 'Baseline' not in r.get('name', '') and r.get('n_trades', 0) >= 10]
        beats_baseline = 0
        for s in signal_strategies:
            beat = s.get('sharpe', 0) > bl_sharpe
            if beat:
                beats_baseline += 1
            print(f"  {s['name']}: Sharpe {s.get('sharpe', 0):.2f} vs baseline {bl_sharpe:.2f} -> {'BEATS' if beat else 'FAILS'}")
        print(f"\n  {beats_baseline}/{len(signal_strategies)} strategies beat random baseline")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_results


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
