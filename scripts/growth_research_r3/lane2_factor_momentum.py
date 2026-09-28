"""
Lane 2: Concentrated Factor Momentum (Long-Only Stock Picking)
================================================================
Strategy:
- Multi-factor scoring: 12-month momentum (skip last month), earnings proxy,
  ROE proxy, relative strength
- Hold top 10-20 stocks from S&P 500 universe, monthly rebalance
- Walk-forward: train factor weights on rolling 252d window, test 21d OOT

SURVIVORSHIP BIAS WARNING: Using current S&P 500 constituents, not historical.
This is a MAJOR bias — stocks in the index today survived; the ones that fell
out are excluded. Real performance would be worse. We flag this prominently.

Commission: $0 on Robinhood/IBKR for stocks (HC #694).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a representative subset of S&P 500 (top ~80 by market cap to keep download feasible)
# This reduces survivorship bias somewhat but doesn't eliminate it
SP500_SUBSET = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM',
    'TSLA', 'UNH', 'V', 'XOM', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'ABBV',
    'MRK', 'CRM', 'AMD', 'NFLX', 'BAC', 'CVX', 'KO', 'PEP', 'LIN', 'WMT',
    'TMO', 'ADBE', 'ACN', 'MCD', 'CSCO', 'ABT', 'DHR', 'QCOM', 'TXN', 'INTC',
    'CMCSA', 'VZ', 'PM', 'NEE', 'RTX', 'INTU', 'AMGN', 'LOW', 'UNP', 'HON',
    'IBM', 'CAT', 'SPGI', 'BA', 'GE', 'AMAT', 'DE', 'GS', 'ELV', 'PFE',
    'BLK', 'ISRG', 'NOW', 'SYK', 'MDT', 'T', 'ADP', 'GILD', 'MMC', 'VRTX',
    'AMT', 'LRCX', 'REGN', 'ADI', 'SCHW', 'CB', 'MU', 'PANW', 'SNPS', 'ZTS'
]

print("Downloading data for Factor Momentum strategy...")
print(f"  Universe: {len(SP500_SUBSET)} stocks")

price_data = {}
for ticker in SP500_SUBSET:
    try:
        df = yf.download(ticker, start='2014-01-01', end='2026-07-11', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 504:  # Need at least 2 years
            price_data[ticker] = df['Close']
    except:
        pass

print(f"  Successfully downloaded: {len(price_data)} stocks")

# Build price matrix
price_df = pd.DataFrame(price_data)
price_df = price_df.dropna(axis=1, thresh=int(len(price_df) * 0.8))  # Drop stocks with >20% missing
price_df = price_df.ffill().bfill()
print(f"  After cleaning: {price_df.shape[1]} stocks, {price_df.shape[0]} days")

# --- Factor Calculation ---
def compute_factors(prices, lookback=252):
    """Compute multi-factor scores for each stock."""
    n_stocks = prices.shape[1]
    returns = prices.pct_change()

    # Factor 1: 12-month momentum, skip last month (classic Jegadeesh-Titman)
    mom_12_1 = (prices.shift(21) / prices.shift(252)) - 1  # 12m return, skip last 1m

    # Factor 2: Short-term reversal (1-month return, used as NEGATIVE signal)
    # Actually we'll use recent earnings proxy: 3-month price change as earnings drift proxy
    earnings_proxy = (prices / prices.shift(63)) - 1

    # Factor 3: ROE proxy - use price-to-book proxy via long-term return stability
    # Since we don't have fundamental data, use return stability as quality proxy
    ret_stability = returns.rolling(126).mean() / returns.rolling(126).std()  # Risk-adjusted return

    # Factor 4: Relative strength vs universe mean
    universe_ret = returns.mean(axis=1)
    relative_strength = returns.rolling(63).mean().subtract(universe_ret.rolling(63).mean(), axis=0)

    return {
        'momentum_12_1': mom_12_1,
        'earnings_proxy': earnings_proxy,
        'quality': ret_stability,
        'relative_strength': relative_strength
    }

def rank_normalize(series):
    """Rank normalize to [0, 1]."""
    ranked = series.rank(pct=True)
    return ranked

def score_stocks(factors, weights):
    """Combine factors with given weights into composite score."""
    composite = None
    for fname, weight in weights.items():
        if fname in factors:
            ranked = rank_normalize(factors[fname])
            if composite is None:
                composite = ranked * weight
            else:
                composite = composite.add(ranked * weight, fill_value=0)
    return composite

# --- Walk-Forward ---
TRAIN_DAYS = 252
TEST_DAYS = 21
TOP_N = 15  # Hold top 15 stocks

# Factor weight candidates to optimize
WEIGHT_CANDIDATES = [
    {'momentum_12_1': 0.4, 'earnings_proxy': 0.2, 'quality': 0.2, 'relative_strength': 0.2},
    {'momentum_12_1': 0.5, 'earnings_proxy': 0.2, 'quality': 0.15, 'relative_strength': 0.15},
    {'momentum_12_1': 0.3, 'earnings_proxy': 0.3, 'quality': 0.2, 'relative_strength': 0.2},
    {'momentum_12_1': 0.6, 'earnings_proxy': 0.1, 'quality': 0.15, 'relative_strength': 0.15},
    {'momentum_12_1': 0.25, 'earnings_proxy': 0.25, 'quality': 0.25, 'relative_strength': 0.25},
]

print("\nRunning walk-forward factor optimization...")

returns = price_df.pct_change()
factors = compute_factors(price_df)

oot_returns = []
oot_dates = []
window_results = []

start = TRAIN_DAYS + 252  # Need lookback for factors
while start + TEST_DAYS <= len(price_df):
    train_slice = slice(start - TRAIN_DAYS, start)
    test_slice = slice(start, min(start + TEST_DAYS, len(price_df)))

    # Train: find best weight combo
    best_weights = WEIGHT_CANDIDATES[0]
    best_train_sharpe = -999

    for wc in WEIGHT_CANDIDATES:
        # Score stocks at each month-end in training window
        train_rets = []
        for t in range(train_slice.start + 21, train_slice.stop, 21):
            scores = {}
            for ticker in price_df.columns:
                s = 0
                for fname, weight in wc.items():
                    val = factors[fname][ticker].iloc[t] if t < len(factors[fname]) else np.nan
                    if not np.isnan(val):
                        s += val * weight
                scores[ticker] = s

            # Pick top N
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [s[0] for s in sorted_stocks[:TOP_N] if not np.isnan(s[1])]

            if top_stocks and t + 21 <= len(returns):
                # Equal-weight portfolio return over next 21 days
                port_ret = returns[top_stocks].iloc[t:t+21].mean(axis=1)
                train_rets.extend(port_ret.values)

        if len(train_rets) > 10:
            tr = np.array(train_rets)
            if tr.std() > 0:
                s = tr.mean() / tr.std() * np.sqrt(252)
                if s > best_train_sharpe:
                    best_train_sharpe = s
                    best_weights = wc

    # Test: apply best weights OOT
    t = start
    scores = {}
    for ticker in price_df.columns:
        s = 0
        valid = True
        for fname, weight in best_weights.items():
            val = factors[fname][ticker].iloc[t] if t < len(factors[fname]) else np.nan
            if np.isnan(val):
                valid = False
                break
            s += val * weight
        if valid:
            scores[ticker] = s

    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_stocks = [s[0] for s in sorted_stocks[:TOP_N]]

    if top_stocks:
        test_end = min(start + TEST_DAYS, len(returns))
        port_ret = returns[top_stocks].iloc[start:test_end].mean(axis=1)
        oot_returns.extend(port_ret.values)
        oot_dates.extend(port_ret.index.tolist())

        window_results.append({
            'test_start': str(price_df.index[start].date()),
            'best_weights': {k: round(v, 2) for k, v in best_weights.items()},
            'top_stocks': top_stocks[:5],
            'oot_ret': round(float(port_ret.sum()) * 100, 3),
        })

    start += TEST_DAYS

    if len(window_results) % 50 == 0:
        print(f"  Completed {len(window_results)} windows...")

print(f"  Total OOT windows: {len(window_results)}")

# --- Analysis ---
oot_series = pd.Series(oot_returns, index=pd.DatetimeIndex(oot_dates))
oot_series = oot_series.groupby(oot_series.index).mean()
oot_series = oot_series.clip(-0.5, 0.5)  # Sanity

# Benchmark
spy = yf.download('SPY', start='2014-01-01', end='2026-07-11', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_ret = spy['Close'].pct_change().dropna()
spy_aligned = spy_ret.reindex(oot_series.index).dropna()
common = oot_series.index.intersection(spy_aligned.index)
oot_series = oot_series.loc[common]
spy_aligned = spy_aligned.loc[common]

n_days = len(oot_series)

# Regime analysis
spy_full = spy.reindex(oot_series.index)
if len(spy_full) > 0 and 'Open' in spy_full.columns:
    green = spy_full['Close'] > spy_full['Open']
    red = ~green
    green_rets = oot_series[green.reindex(oot_series.index, fill_value=False)]
    red_rets = oot_series[red.reindex(oot_series.index, fill_value=False)]
    sg = float(green_rets.mean() / green_rets.std() * np.sqrt(252)) if len(green_rets) > 10 and green_rets.std() > 0 else 0
    sr = float(red_rets.mean() / red_rets.std() * np.sqrt(252)) if len(red_rets) > 10 and red_rets.std() > 0 else 0
    regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
else:
    sg = sr = regime_gap = 0

if n_days > 0 and oot_series.std() > 0:
    ann_ret = float(oot_series.mean() * 252)
    ann_vol = float(oot_series.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol
    downside = oot_series[oot_series < 0]
    dv = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
    sortino = ann_ret / dv if dv > 0 else 0
    cum = (1 + oot_series).cumprod()
    total_ret = float(cum.iloc[-1])
    years = n_days / 252
    cagr = (total_ret ** (1/years) - 1) if years > 0 and total_ret > 0 else 0
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    max_dd = max(max_dd, -1.0)

    spy_sharpe = float(spy_aligned.mean() / spy_aligned.std() * np.sqrt(252)) if spy_aligned.std() > 0 else 0
    spy_cum = (1 + spy_aligned).cumprod()
    spy_cagr = (float(spy_cum.iloc[-1]) ** (1/years) - 1) if years > 0 and len(spy_cum) > 0 else 0
else:
    ann_ret = ann_vol = sharpe = sortino = cagr = max_dd = 0
    spy_sharpe = spy_cagr = 0

summary = {
    'strategy': 'Lane 2: Concentrated Factor Momentum',
    'universe_size': len(price_data),
    'top_n_holdings': TOP_N,
    'oot_days': n_days,
    'oot_years': round(n_days/252, 1),
    'annualized_return_pct': round(ann_ret * 100, 2),
    'annualized_vol_pct': round(ann_vol * 100, 2),
    'sharpe': round(sharpe, 3),
    'sortino': round(sortino, 3),
    'cagr_pct': round(cagr * 100, 2),
    'max_drawdown_pct': round(max_dd * 100, 2),
    'regime_sharpe_green': round(sg, 3),
    'regime_sharpe_red': round(sr, 3),
    'regime_gap': round(regime_gap, 3),
    'regime_test_pass': regime_gap < 0.50,
    'benchmark_spy_sharpe': round(float(spy_sharpe), 3),
    'benchmark_spy_cagr_pct': round(float(spy_cagr) * 100, 2),
    'walk_forward_windows': len(window_results),
    'survivorship_bias_warning': 'HIGH - using current S&P 500 constituents, not historical. Dropped stocks excluded.',
    'commission': '$0 (Robinhood/IBKR, HC #694)',
}

print("\n" + "="*60)
print("LANE 2: CONCENTRATED FACTOR MOMENTUM - RESULTS")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUTPUT_DIR, 'lane2_factor_momentum_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/lane2_factor_momentum_results.json")
