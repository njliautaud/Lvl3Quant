"""
Lane 4: Systematic Swing Trading
==================================
Strategy:
- 3-10 day holding period trades on liquid large-caps
- Signals: RSI divergence + volume confirmation + trend alignment
- Walk-forward IC test on 5-day forward returns
- Long-only (but track long vs short signal quality)

Unlike the mean-reversion that failed on daily timeframe, this targets
the SWING timeframe with trend-aligned entries.

Commission: $0 on Robinhood/IBKR (HC #694)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Large-cap liquid universe
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'MA',
    'UNH', 'HD', 'PG', 'JNJ', 'XOM', 'BAC', 'NFLX', 'CRM', 'AMD', 'COST',
    'ABBV', 'MRK', 'KO', 'PEP', 'WMT', 'CSCO', 'ADBE', 'QCOM', 'INTC', 'T',
    'GE', 'CAT', 'BA', 'GS', 'IBM', 'LOW', 'AMGN', 'MCD', 'HON', 'NEE'
]

print("Downloading data for Swing Trading strategy...")
price_data = {}
volume_data = {}
for ticker in TICKERS:
    try:
        df = yf.download(ticker, start='2014-01-01', end='2026-07-11', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 504:
            price_data[ticker] = df[['Open', 'High', 'Low', 'Close']]
            volume_data[ticker] = df['Volume']
    except:
        pass

print(f"  Downloaded: {len(price_data)} stocks")

# --- Signal Calculation ---
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_signals(ohlcv, volume):
    """Compute swing trading signals for a single stock."""
    close = ohlcv['Close']
    high = ohlcv['High']
    low = ohlcv['Low']

    signals = pd.DataFrame(index=close.index)

    # RSI
    rsi_14 = compute_rsi(close, 14)
    rsi_5 = compute_rsi(close, 5)
    signals['rsi_14'] = rsi_14
    signals['rsi_5'] = rsi_5

    # RSI divergence: price making lower low but RSI making higher low
    # Simplified: RSI rising while price is near recent low
    price_pctile = close.rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))
    rsi_pctile = rsi_14.rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))
    signals['rsi_divergence'] = rsi_pctile - price_pctile  # Positive = bullish divergence

    # Volume confirmation
    vol_ratio = volume / volume.rolling(20).mean()
    signals['vol_ratio'] = vol_ratio

    # Trend alignment
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    signals['above_sma20'] = (close > sma_20).astype(float)
    signals['above_sma50'] = (close > sma_50).astype(float)
    signals['above_sma200'] = (close > sma_200).astype(float)
    signals['trend_score'] = signals['above_sma20'] + signals['above_sma50'] + signals['above_sma200']

    # Price position relative to Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    signals['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)

    # ATR for position sizing
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    signals['atr_pct'] = tr.rolling(14).mean() / close

    # 5-day forward return (target)
    signals['fwd_5d_ret'] = close.shift(-5) / close - 1

    # Composite signal (to be weighted via walk-forward)
    # Buy when: RSI oversold + bullish divergence + trend aligned + volume spike
    signals['raw_signal'] = (
        (50 - rsi_14) / 50 * 0.3 +           # Lower RSI = more bullish
        signals['rsi_divergence'] * 0.3 +      # Bullish divergence
        signals['trend_score'] / 3 * 0.2 +     # Trend alignment
        (vol_ratio - 1).clip(-1, 1) * 0.1 +   # Volume confirmation
        (-signals['bb_position']).clip(-1, 1) * 0.1  # Near lower band
    )

    return signals.dropna()

# Compute signals for all stocks
print("\nComputing signals...")
all_signals = {}
for ticker in price_data:
    sig = compute_signals(price_data[ticker], volume_data[ticker])
    if len(sig) > 252:
        all_signals[ticker] = sig

print(f"  Signals computed for {len(all_signals)} stocks")

# --- Walk-Forward IC Test ---
TRAIN_DAYS = 252
TEST_DAYS = 21
TOP_N = 10  # Number of stocks to hold

# Weight candidates for composite signal
WEIGHT_SETS = [
    {'rsi': 0.3, 'divergence': 0.3, 'trend': 0.2, 'volume': 0.1, 'bb': 0.1},
    {'rsi': 0.4, 'divergence': 0.2, 'trend': 0.2, 'volume': 0.1, 'bb': 0.1},
    {'rsi': 0.2, 'divergence': 0.4, 'trend': 0.2, 'volume': 0.1, 'bb': 0.1},
    {'rsi': 0.2, 'divergence': 0.2, 'trend': 0.4, 'volume': 0.1, 'bb': 0.1},
    {'rsi': 0.25, 'divergence': 0.25, 'trend': 0.25, 'volume': 0.15, 'bb': 0.1},
]

def build_composite(sig_df, weights):
    """Build composite signal with given weights."""
    return (
        (50 - sig_df['rsi_14']) / 50 * weights['rsi'] +
        sig_df['rsi_divergence'] * weights['divergence'] +
        sig_df['trend_score'] / 3 * weights['trend'] +
        (sig_df['vol_ratio'] - 1).clip(-1, 1) * weights['volume'] +
        (-sig_df['bb_position']).clip(-1, 1) * weights['bb']
    )

print("\nRunning walk-forward swing trading backtest...")

# Pre-build vectorized signal matrices for speed
print("  Building signal matrices...")
# Build per-weight-set composite DataFrames and forward return DataFrame
all_tickers = list(all_signals.keys())

# Forward returns matrix
fwd_ret_df = pd.DataFrame({t: all_signals[t]['fwd_5d_ret'] for t in all_tickers})

# Composite signal matrices (one per weight set)
composite_dfs = {}
for wi, ws in enumerate(WEIGHT_SETS):
    comp = pd.DataFrame({t: build_composite(all_signals[t], ws) for t in all_tickers})
    composite_dfs[wi] = comp

# Use SPY's trading calendar as reference dates
spy_cal = yf.download('SPY', start='2014-01-01', end='2026-07-11', progress=False)
if isinstance(spy_cal.columns, pd.MultiIndex):
    spy_cal.columns = spy_cal.columns.get_level_values(0)
ref_dates = spy_cal.index

# Common dates: where fwd_ret_df has at least 20 non-NaN stocks
valid_counts = fwd_ret_df.reindex(ref_dates).notna().sum(axis=1)
common_dates = valid_counts[valid_counts >= 20].index.sort_values()
print(f"  Trading dates with >= 20 stocks: {len(common_dates)}")

# Reindex all matrices to common_dates
fwd_ret_df = fwd_ret_df.reindex(common_dates)
for wi in composite_dfs:
    composite_dfs[wi] = composite_dfs[wi].reindex(common_dates)

oot_returns = []
oot_dates = []
ic_values = []
window_results = []

start = TRAIN_DAYS
while start + TEST_DAYS <= len(common_dates):
    train_idx = slice(start - TRAIN_DAYS, start)
    test_idx = slice(start, min(start + TEST_DAYS, len(common_dates)))

    # Train: find best weights by IC (vectorized)
    best_wi = 0
    best_ic = -999

    for wi, ws in enumerate(WEIGHT_SETS):
        comp_train = composite_dfs[wi].iloc[train_idx]
        fwd_train = fwd_ret_df.iloc[train_idx]
        # Per-stock IC
        ics = []
        for t in all_tickers:
            c = comp_train[t].dropna()
            f = fwd_train[t].dropna()
            common = c.index.intersection(f.index)
            if len(common) > 30:
                ic = c.loc[common].corr(f.loc[common])
                if not np.isnan(ic):
                    ics.append(ic)
        mean_ic = np.mean(ics) if ics else 0
        if mean_ic > best_ic:
            best_ic = mean_ic
            best_wi = wi

    best_weights = WEIGHT_SETS[best_wi]
    comp_test = composite_dfs[best_wi].iloc[test_idx]
    fwd_test = fwd_ret_df.iloc[test_idx]

    # Test OOT
    for i in range(len(comp_test)):
        test_day = comp_test.index[i]
        scores = comp_test.iloc[i].dropna()
        fwd = fwd_test.iloc[i].dropna()

        if len(scores) >= TOP_N:
            top_stocks = scores.nlargest(TOP_N).index
            top_fwd = fwd.reindex(top_stocks).dropna()

            if len(top_fwd) > 0:
                port_ret = float(top_fwd.mean()) / 5
                oot_returns.append(port_ret)
                oot_dates.append(test_day)

                # IC for this day
                common_t = scores.index.intersection(fwd.index)
                if len(common_t) > 10:
                    ic = scores.loc[common_t].corr(fwd.loc[common_t])
                    if not np.isnan(ic):
                        ic_values.append(ic)

    window_results.append({
        'test_start': str(common_dates[start].date()),
        'best_weights': {k: round(v, 2) for k, v in best_weights.items()},
        'train_ic': round(best_ic, 4),
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
spy = yf.download('SPY', start='2014-01-01', end='2026-07-11', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_ret = spy['Close'].pct_change().dropna()
spy_aligned = spy_ret.reindex(oot_series.index).dropna()
common = oot_series.index.intersection(spy_aligned.index)
oot_series = oot_series.loc[common]
spy_aligned = spy_aligned.loc[common]

n_days = len(oot_series)

# Regime
spy_full = spy.reindex(oot_series.index)
if len(spy_full) > 0 and 'Open' in spy_full.columns:
    green = spy_full['Close'] > spy_full['Open']
    red = ~green
    g_rets = oot_series[green.reindex(oot_series.index, fill_value=False)]
    r_rets = oot_series[red.reindex(oot_series.index, fill_value=False)]
    sg = float(g_rets.mean() / g_rets.std() * np.sqrt(252)) if len(g_rets) > 10 and g_rets.std() > 0 else 0
    sr_ = float(r_rets.mean() / r_rets.std() * np.sqrt(252)) if len(r_rets) > 10 and r_rets.std() > 0 else 0
    regime_gap = abs(sg - sr_) / max(abs(sg), abs(sr_), 0.001)
else:
    sg = sr_ = regime_gap = 0

if n_days > 0 and oot_series.std() > 0:
    ann_ret = float(oot_series.mean() * 252)
    ann_vol = float(oot_series.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol
    ds = oot_series[oot_series < 0]
    dv = float(ds.std() * np.sqrt(252)) if len(ds) > 0 else ann_vol
    sortino = ann_ret / dv if dv > 0 else 0
    cum = (1 + oot_series).cumprod()
    total_ret = float(cum.iloc[-1])
    years = n_days / 252
    cagr = (total_ret ** (1/years) - 1) if years > 0 and total_ret > 0 else 0
    max_dd = max(float(((cum - cum.cummax()) / cum.cummax()).min()), -1.0)

    spy_sharpe = float(spy_aligned.mean() / spy_aligned.std() * np.sqrt(252)) if spy_aligned.std() > 0 else 0
    spy_cum = (1 + spy_aligned).cumprod()
    spy_cagr = (float(spy_cum.iloc[-1]) ** (1/years) - 1) if years > 0 and len(spy_cum) > 0 else 0
else:
    ann_ret = ann_vol = sharpe = sortino = cagr = max_dd = 0
    spy_sharpe = spy_cagr = 0

mean_ic = np.mean(ic_values) if ic_values else 0
median_ic = np.median(ic_values) if ic_values else 0
ic_positive_pct = sum(1 for x in ic_values if x > 0) / len(ic_values) * 100 if ic_values else 0

# Directional accuracy
dir_accuracy = float((oot_series > 0).mean() * 100) if n_days > 0 else 0

summary = {
    'strategy': 'Lane 4: Systematic Swing Trading',
    'universe_size': len(all_signals),
    'holding_period': '5 days',
    'top_n': TOP_N,
    'oot_days': n_days,
    'oot_years': round(n_days/252, 1),
    'mean_ic': round(float(mean_ic), 4),
    'median_ic': round(float(median_ic), 4),
    'ic_positive_pct': round(ic_positive_pct, 1),
    'directional_accuracy_pct': round(dir_accuracy, 1),
    'annualized_return_pct': round(ann_ret * 100, 2),
    'annualized_vol_pct': round(ann_vol * 100, 2),
    'sharpe': round(sharpe, 3),
    'sortino': round(sortino, 3),
    'cagr_pct': round(cagr * 100, 2),
    'max_drawdown_pct': round(max_dd * 100, 2),
    'regime_sharpe_green': round(sg, 3),
    'regime_sharpe_red': round(sr_, 3),
    'regime_gap': round(regime_gap, 3),
    'regime_test_pass': regime_gap < 0.50,
    'benchmark_spy_sharpe': round(float(spy_sharpe), 3),
    'benchmark_spy_cagr_pct': round(float(spy_cagr) * 100, 2),
    'walk_forward_windows': len(window_results),
    'survivorship_bias': 'MODERATE - using current large-caps, not historical membership',
    'commission': '$0 (HC #694)',
}

print("\n" + "="*60)
print("LANE 4: SYSTEMATIC SWING TRADING - RESULTS")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUTPUT_DIR, 'lane4_swing_trading_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/lane4_swing_trading_results.json")
