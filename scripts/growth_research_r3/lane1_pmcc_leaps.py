"""
Lane 1: Poor Man's Covered Calls (PMCC) on Growth Stocks
=========================================================
Strategy:
- Buy deep ITM LEAPS (simulated as delta-leveraged long position)
- Sell monthly OTM calls against them (simulated as premium income)
- Walk-forward: train optimal short call delta on rolling window

Since we don't have historical options chain data via yfinance, we SIMULATE
the PMCC economics on a DAILY basis using:
- LEAPS provides leveraged delta exposure (delta ~0.80)
- Cost basis = LEAPS price (~60-70% of stock price for deep ITM)
- Short call premium collected monthly (amortized daily)
- Cap upside at short strike, keep premium if OTM

SURVIVORSHIP BIAS WARNING: Using current S&P 500 mega-caps. These are
survivors by definition. Results will be upward biased.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Black-Scholes helpers ---
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# --- Data Download ---
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
print("Downloading data for PMCC strategy...")
data = {}
for ticker in TICKERS:
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-07-11', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 252:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
    except Exception as e:
        print(f"  {ticker} FAILED: {e}")

if not data:
    print("No data downloaded. Exiting.")
    exit(1)

# Build aligned price DataFrame
price_df = pd.DataFrame({t: data[t]['Close'] for t in data.keys()}).dropna()
returns_df = price_df.pct_change().dropna()
print(f"  Aligned: {len(price_df)} days, {len(price_df.columns)} stocks")

# Realized vol
log_ret = np.log(price_df / price_df.shift(1)).dropna()
rv_df = log_ret.rolling(21).std() * np.sqrt(252)

RISK_FREE = 0.04
LEAPS_DELTA_TARGET = 0.80

def daily_pmcc_return(stock_ret, rv, short_delta_target):
    """
    Compute daily PMCC return for a single stock.

    Economics:
    - LEAPS cost basis ~ stock_price * intrinsic_ratio (deep ITM ~ 0.65 of stock)
    - Daily delta exposure: ~0.80 * stock_return (leveraged)
    - Daily theta income: short call premium / 21 trading days (amortized monthly)
    - Upside cap: when stock moves up a lot, short call limits gains

    Returns daily return on the LEAPS cost basis.
    """
    # Cost basis ratio: deep ITM LEAPS costs about 60-70% of stock price
    cost_ratio = 0.65

    # LEAPS delta exposure
    leaps_delta = 0.80
    delta_pnl = leaps_delta * stock_ret / cost_ratio  # Leveraged return on cost basis

    # LEAPS theta decay (amortized daily over 365-day LEAPS)
    # Theta ~ sigma * S / (2 * sqrt(T)) / 365 as fraction of cost
    leaps_theta = rv * 0.02 / 365 / cost_ratio  # Very small daily for long-dated

    # Short call premium income (amortized daily)
    # Monthly OTM call at short_delta_target delta
    # Premium ~ sigma * S * sqrt(T) * N'(d) / 21 trading days
    # Simplified: premium as fraction of stock ~ sigma * sqrt(21/252) * 0.4 * short_delta_target
    monthly_premium_pct = rv * np.sqrt(21/252) * 0.4 * (1 - short_delta_target)
    daily_premium = monthly_premium_pct / 21 / cost_ratio

    # Upside cap: if stock moves up more than ~(1/short_delta_target - 1) * sigma * sqrt(1/252)
    # the short call is ITM and caps the gain
    # Simplified: cap daily upside at short_delta_target's moneyness
    cap_daily = (1 - short_delta_target) * rv / np.sqrt(252) / cost_ratio

    # Net daily return
    capped_delta_pnl = np.minimum(delta_pnl, cap_daily)

    return capped_delta_pnl + daily_premium - leaps_theta


# --- Walk-Forward ---
TRAIN_DAYS = 252
TEST_DAYS = 21

SHORT_DELTAS = [0.20, 0.25, 0.30, 0.35, 0.40]

print("\nRunning walk-forward PMCC optimization...")

oot_returns = []
oot_dates = []
window_results = []

start = TRAIN_DAYS + 21  # Need vol lookback
while start + TEST_DAYS <= len(returns_df):
    # Train: find best short delta
    best_delta = 0.30
    best_sharpe = -999

    for sd in SHORT_DELTAS:
        train_rets = []
        for t in range(start - TRAIN_DAYS, start):
            if t < len(rv_df) and t < len(returns_df):
                daily_rets = []
                for ticker in price_df.columns:
                    sr = returns_df[ticker].iloc[t]
                    rv = rv_df[ticker].iloc[t] if t < len(rv_df) else 0.25
                    if np.isnan(rv) or rv <= 0:
                        rv = 0.25
                    dr = daily_pmcc_return(sr, rv, sd)
                    if not np.isnan(dr):
                        daily_rets.append(dr)
                if daily_rets:
                    train_rets.append(np.mean(daily_rets))  # Equal-weight portfolio

        if len(train_rets) > 50:
            tr = np.array(train_rets)
            if tr.std() > 0:
                s = tr.mean() / tr.std() * np.sqrt(252)
                if s > best_sharpe:
                    best_sharpe = s
                    best_delta = sd

    # Test OOT
    test_end = min(start + TEST_DAYS, len(returns_df))
    for t in range(start, test_end):
        daily_rets = []
        for ticker in price_df.columns:
            sr = returns_df[ticker].iloc[t]
            rv = rv_df[ticker].iloc[t] if t < len(rv_df) else 0.25
            if np.isnan(rv) or rv <= 0:
                rv = 0.25
            dr = daily_pmcc_return(sr, rv, best_delta)
            if not np.isnan(dr):
                daily_rets.append(dr)
        if daily_rets:
            oot_returns.append(np.mean(daily_rets))
            oot_dates.append(returns_df.index[t])

    window_results.append({
        'test_start': str(returns_df.index[start].date()),
        'best_delta': best_delta,
        'train_sharpe': round(best_sharpe, 3),
    })

    start += TEST_DAYS
    if len(window_results) % 50 == 0:
        print(f"  Completed {len(window_results)} windows...")

print(f"  Total windows: {len(window_results)}")

# --- Analysis ---
oot_series = pd.Series(oot_returns, index=pd.DatetimeIndex(oot_dates))
oot_series = oot_series.groupby(oot_series.index).mean()
oot_series = oot_series.clip(-0.5, 0.5)

# Benchmark
spy = yf.download('SPY', start='2015-01-01', end='2026-07-11', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_ret = spy['Close'].pct_change().dropna()
spy_aligned = spy_ret.reindex(oot_series.index).dropna()
common_idx = oot_series.index.intersection(spy_aligned.index)
oot_series = oot_series.loc[common_idx]
spy_aligned = spy_aligned.loc[common_idx]

n_days = len(oot_series)

# Regime analysis
spy_full = spy.reindex(oot_series.index)
if len(spy_full) > 0 and 'Open' in spy_full.columns:
    green_mask = spy_full['Close'] > spy_full['Open']
    red_mask = ~green_mask
    green_rets = oot_series[green_mask.reindex(oot_series.index, fill_value=False)]
    red_rets = oot_series[red_mask.reindex(oot_series.index, fill_value=False)]
    sharpe_green = float(green_rets.mean() / green_rets.std() * np.sqrt(252)) if len(green_rets) > 10 and green_rets.std() > 0 else 0
    sharpe_red = float(red_rets.mean() / red_rets.std() * np.sqrt(252)) if len(red_rets) > 10 and red_rets.std() > 0 else 0
    regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.001)
else:
    sharpe_green = sharpe_red = regime_gap = 0

if n_days > 0 and oot_series.std() > 0:
    ann_ret = float(oot_series.mean() * 252)
    ann_vol = float(oot_series.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol
    downside = oot_series[oot_series < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    cum = (1 + oot_series).cumprod()
    total_ret = float(cum.iloc[-1])
    years = n_days / 252
    cagr = (total_ret ** (1/years) - 1) if years > 0 and total_ret > 0 else 0
    max_dd = max(float(((cum - cum.cummax()) / cum.cummax()).min()), -1.0)

    spy_sharpe = float(spy_aligned.mean() / spy_aligned.std() * np.sqrt(252)) if spy_aligned.std() > 0 else 0
    spy_cum = (1 + spy_aligned).cumprod()
    spy_cagr = (float(spy_cum.iloc[-1]) ** (1/years) - 1) if years > 0 and len(spy_cum) > 0 else 0
    spy_dd = max(float(((spy_cum - spy_cum.cummax()) / spy_cum.cummax()).min()), -1.0)
else:
    ann_ret = ann_vol = sharpe = sortino = cagr = max_dd = 0
    spy_sharpe = spy_cagr = spy_dd = 0

# Win rate and profit factor
wr = float((oot_series > 0).mean() * 100) if n_days > 0 else 0
gp = oot_series[oot_series > 0].sum()
gl = abs(oot_series[oot_series < 0].sum())
pf = float(gp / gl) if gl > 0 else 0

summary = {
    'strategy': 'Lane 1: PMCC on Growth Stocks',
    'tickers': TICKERS,
    'oot_days': n_days,
    'oot_years': round(n_days/252, 1),
    'annualized_return_pct': round(ann_ret * 100, 2),
    'annualized_vol_pct': round(ann_vol * 100, 2),
    'sharpe': round(sharpe, 3),
    'sortino': round(sortino, 3),
    'cagr_pct': round(cagr * 100, 2),
    'max_drawdown_pct': round(max_dd * 100, 2),
    'win_rate_pct': round(wr, 1),
    'profit_factor': round(pf, 3),
    'regime_sharpe_green': round(sharpe_green, 3),
    'regime_sharpe_red': round(sharpe_red, 3),
    'regime_gap': round(regime_gap, 3),
    'regime_test_pass': regime_gap < 0.50,
    'benchmark_spy_sharpe': round(float(spy_sharpe), 3),
    'benchmark_spy_cagr_pct': round(float(spy_cagr) * 100, 2),
    'walk_forward_windows': len(window_results),
    'survivorship_bias_warning': 'HIGH - using current mega-cap names that are survivors by definition',
    'data_quality_notes': 'Options simulated via BS approximation. No actual options chain data. Vol smile/skew not modeled. Premium estimates are rough.',
}

print("\n" + "="*60)
print("LANE 1: PMCC ON GROWTH STOCKS - RESULTS")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUTPUT_DIR, 'lane1_pmcc_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/lane1_pmcc_results.json")
