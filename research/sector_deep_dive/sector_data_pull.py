"""
Sector Deep Dive: Data Pull + All Analyses
Pulls 3 years of sector ETF data and runs comprehensive analysis.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '/home/jupiter/Lvl3Quant/research/sector_deep_dive'

# ============================================================
# 1. DATA PULL
# ============================================================
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Disc',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
    'SPY': 'S&P 500',
}

BROAD_ETFS = {
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'DIA': 'Dow 30',
    'TLT': 'Long Treasury',
    'GLD': 'Gold',
    'VIX': 'VIX',
}

all_tickers = list(SECTOR_ETFS.keys()) + list(BROAD_ETFS.keys())
# Remove VIX - use ^VIX
all_tickers = [t for t in all_tickers if t != 'VIX']
all_tickers.append('^VIX')

print("Pulling 3 years of daily data...")
data = yf.download(all_tickers, period='3y', interval='1d', group_by='ticker', progress=False)
print(f"Data shape: {data.shape}, date range: {data.index[0]} to {data.index[-1]}")

# Also pull 60d of intraday hourly data for time-of-day analysis
print("Pulling 60d hourly data for intraday analysis...")
sector_tickers = list(SECTOR_ETFS.keys())
hourly = yf.download(sector_tickers, period='60d', interval='1h', group_by='ticker', progress=False)
print(f"Hourly data shape: {hourly.shape}")

# Build clean daily close/volume/OHLC frames
closes = pd.DataFrame()
opens = pd.DataFrame()
highs = pd.DataFrame()
lows = pd.DataFrame()
volumes = pd.DataFrame()

for t in all_tickers:
    label = t.replace('^', '')
    try:
        closes[label] = data[t]['Close']
        opens[label] = data[t]['Open']
        highs[label] = data[t]['High']
        lows[label] = data[t]['Low']
        volumes[label] = data[t]['Volume']
    except:
        print(f"  Skipping {t} - not in data")

closes = closes.dropna(how='all')
print(f"Clean closes: {closes.shape}")

# Save raw data
closes.to_parquet(f'{OUT}/sector_closes.parquet')
opens.to_parquet(f'{OUT}/sector_opens.parquet')

# ============================================================
# ANALYSIS 1: TIME-OF-DAY RETURN PATTERNS
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: TIME-OF-DAY RETURN PATTERNS")
print("="*60)

results_tod = {}

# Overnight gap analysis: (Open - prev Close) / prev Close
overnight_returns = (opens / closes.shift(1) - 1).dropna()
intraday_returns = (closes / opens - 1).dropna()
daily_returns = closes.pct_change().dropna()

sector_list = [t for t in SECTOR_ETFS.keys()]

print("\n--- Overnight vs Intraday Returns (annualized, bps/day) ---")
print(f"{'Sector':<20} {'Overnight bps':<15} {'Intraday bps':<15} {'O/N Sharpe':<12} {'Intra Sharpe':<12}")
print("-"*74)

for s in sector_list:
    if s not in overnight_returns.columns:
        continue
    on = overnight_returns[s].dropna()
    intra = intraday_returns[s].dropna()
    on_bps = on.mean() * 10000
    intra_bps = intra.mean() * 10000
    on_sharpe = on.mean() / on.std() * np.sqrt(252) if on.std() > 0 else 0
    intra_sharpe = intra.mean() / intra.std() * np.sqrt(252) if intra.std() > 0 else 0
    print(f"{SECTOR_ETFS.get(s, s):<20} {on_bps:<15.2f} {intra_bps:<15.2f} {on_sharpe:<12.3f} {intra_sharpe:<12.3f}")
    results_tod[s] = {
        'overnight_bps': round(on_bps, 2),
        'intraday_bps': round(intra_bps, 2),
        'overnight_sharpe': round(on_sharpe, 3),
        'intraday_sharpe': round(intra_sharpe, 3),
    }

# Overnight gap -> next day reversal analysis
print("\n--- Overnight Gap Reversal Analysis ---")
print("Does a large overnight gap predict intraday reversal?")
print(f"{'Sector':<20} {'Gap>0.5% Rev%':<15} {'Gap<-0.5% Rev%':<16} {'Rev Sharpe':<12}")
print("-"*63)

for s in sector_list:
    if s not in overnight_returns.columns:
        continue
    on = overnight_returns[s]
    intra = intraday_returns[s]
    # Align
    common = on.index.intersection(intra.index)
    on = on.loc[common]
    intra = intra.loc[common]

    # Large positive gap -> intraday reversal (negative)?
    pos_gap = on > 0.005
    neg_gap = on < -0.005

    if pos_gap.sum() > 10:
        pos_rev_pct = (intra[pos_gap] < 0).mean() * 100
        neg_rev_pct = (intra[neg_gap] > 0).mean() * 100 if neg_gap.sum() > 10 else float('nan')

        # Strategy: fade the gap
        fade_returns = pd.Series(0.0, index=common)
        fade_returns[pos_gap] = -intra[pos_gap]  # short after gap up
        fade_returns[neg_gap] = intra[neg_gap]    # long after gap down
        fade_sharpe = fade_returns.mean() / fade_returns.std() * np.sqrt(252) if fade_returns.std() > 0 else 0

        print(f"{SECTOR_ETFS.get(s, s):<20} {pos_rev_pct:<15.1f} {neg_rev_pct:<16.1f} {fade_sharpe:<12.3f}")

# Hourly return patterns
print("\n--- Hourly Return Patterns (from 60d intraday data) ---")
try:
    hourly_results = {}
    for s in sector_list:
        try:
            h_close = hourly[s]['Close'].dropna()
            h_ret = h_close.pct_change().dropna()
            h_ret.index = pd.to_datetime(h_ret.index)
            by_hour = h_ret.groupby(h_ret.index.hour)
            hourly_results[s] = {
                'mean_bps': (by_hour.mean() * 10000).to_dict(),
                'sharpe': (by_hour.mean() / by_hour.std() * np.sqrt(252)).to_dict(),
            }
        except Exception as e:
            pass

    if hourly_results:
        # Find best/worst hours per sector
        print(f"{'Sector':<20} {'Best Hour':<12} {'Best bps':<10} {'Worst Hour':<12} {'Worst bps':<10}")
        print("-"*64)
        for s, hr in hourly_results.items():
            means = hr['mean_bps']
            if means:
                best_h = max(means, key=means.get)
                worst_h = min(means, key=means.get)
                print(f"{SECTOR_ETFS.get(s, s):<20} {best_h:<12} {means[best_h]:<10.1f} {worst_h:<12} {means[worst_h]:<10.1f}")
except Exception as e:
    print(f"  Hourly analysis error: {e}")

# ============================================================
# ANALYSIS 2: SECTOR ROTATION TIMING SIGNALS
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: SECTOR ROTATION TIMING")
print("="*60)

# Relative strength (sector / SPY)
spy_close = closes['SPY']
rel_strength = pd.DataFrame()
for s in sector_list:
    if s in closes.columns and s != 'SPY':
        rel_strength[s] = closes[s] / spy_close

# Momentum signals: 1m, 3m, 6m relative strength momentum
windows = {'1m': 21, '3m': 63, '6m': 126}
momentum_signals = {}

for label, w in windows.items():
    mom = rel_strength.pct_change(w)
    momentum_signals[label] = mom

# Test: Does buying the strongest relative strength sector and shorting the weakest work?
print("\n--- Relative Strength Momentum Strategy (Top 3 vs Bottom 3) ---")
print(f"{'Lookback':<10} {'Ann Return%':<13} {'Sharpe':<10} {'Sortino':<10} {'MaxDD%':<10} {'WR%':<10}")
print("-"*63)

for label, w in windows.items():
    mom = momentum_signals[label].dropna(how='all')
    strat_returns = []

    for i in range(len(mom) - 1):
        row = mom.iloc[i].dropna()
        if len(row) < 6:
            continue
        top3 = row.nlargest(3).index
        bot3 = row.nsmallest(3).index

        next_day = daily_returns.iloc[i+1] if i+1 < len(daily_returns) else None
        if next_day is None:
            continue

        long_ret = next_day[top3].mean() if all(t in next_day.index for t in top3) else 0
        short_ret = next_day[bot3].mean() if all(t in next_day.index for t in bot3) else 0
        strat_returns.append(long_ret - short_ret)

    sr = pd.Series(strat_returns)
    if len(sr) > 50:
        ann_ret = sr.mean() * 252 * 100
        sharpe = sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else 0
        downside = sr[sr < 0].std()
        sortino = sr.mean() / downside * np.sqrt(252) if downside > 0 else 0
        cum = (1 + sr).cumprod()
        maxdd = ((cum / cum.cummax()) - 1).min() * 100
        wr = (sr > 0).mean() * 100
        print(f"{label:<10} {ann_ret:<13.2f} {sharpe:<10.3f} {sortino:<10.3f} {maxdd:<10.2f} {wr:<10.1f}")

# MA crossover signals
print("\n--- Moving Average Crossover Timing (Sector ETFs) ---")
print(f"{'Sector':<20} {'50/200 Sharpe':<14} {'20/50 Sharpe':<13} {'10/30 Sharpe':<13}")
print("-"*60)

ma_results = {}
for s in sector_list:
    if s not in closes.columns or s == 'SPY':
        continue
    c = closes[s].dropna()
    r = daily_returns[s].dropna() if s in daily_returns.columns else None
    if r is None or len(c) < 201:
        continue

    results_row = {}
    for (fast, slow), label in [((50, 200), '50/200'), ((20, 50), '20/50'), ((10, 30), '10/30')]:
        ma_fast = c.rolling(fast).mean()
        ma_slow = c.rolling(slow).mean()
        signal = (ma_fast > ma_slow).astype(int)
        # Go long when fast > slow, flat otherwise
        strat_ret = signal.shift(1) * r
        strat_ret = strat_ret.dropna()
        if len(strat_ret) > 50:
            sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
            results_row[label] = round(sharpe, 3)
        else:
            results_row[label] = float('nan')

    print(f"{SECTOR_ETFS.get(s, s):<20} {results_row.get('50/200', 'N/A'):<14} {results_row.get('20/50', 'N/A'):<13} {results_row.get('10/30', 'N/A'):<13}")
    ma_results[s] = results_row

# Breadth signal: % of sectors above their 50d MA
print("\n--- Sector Breadth Signal ---")
breadth = pd.DataFrame()
for s in sector_list:
    if s in closes.columns:
        ma50 = closes[s].rolling(50).mean()
        breadth[s] = (closes[s] > ma50).astype(int)

breadth_pct = breadth.mean(axis=1)  # % of sectors above 50d MA

# When breadth is extreme (>80% or <20%), what happens next?
spy_fwd = daily_returns['SPY'].shift(-5).rolling(5).mean() * 5  # 5-day forward return
aligned = pd.DataFrame({'breadth': breadth_pct, 'fwd_5d': spy_fwd}).dropna()

high_breadth = aligned[aligned['breadth'] > 0.8]
low_breadth = aligned[aligned['breadth'] < 0.3]
mid_breadth = aligned[(aligned['breadth'] >= 0.3) & (aligned['breadth'] <= 0.8)]

print(f"High breadth (>80%): {len(high_breadth)} days, avg 5d fwd return: {high_breadth['fwd_5d'].mean()*100:.3f}%")
print(f"Low breadth (<30%):  {len(low_breadth)} days, avg 5d fwd return: {low_breadth['fwd_5d'].mean()*100:.3f}%")
print(f"Mid breadth:         {len(mid_breadth)} days, avg 5d fwd return: {mid_breadth['fwd_5d'].mean()*100:.3f}%")

# ============================================================
# ANALYSIS 3: FUNDAMENTAL FACTORS (via yfinance)
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: FUNDAMENTAL SCREENING")
print("="*60)

fund_data = {}
for s in sector_list:
    try:
        etf = yf.Ticker(s)
        info = etf.info
        fund_data[s] = {
            'name': SECTOR_ETFS.get(s, s),
            'yield_pct': info.get('yield', info.get('trailingAnnualDividendYield', 0)) or 0,
            'pe_ratio': info.get('trailingPE', 0) or 0,
            'beta': info.get('beta3Year', info.get('beta', 0)) or 0,
            'ytd_return': info.get('ytdReturn', 0) or 0,
            'expense_ratio': info.get('annualReportExpenseRatio', 0) or 0,
        }
    except:
        pass

if fund_data:
    print(f"\n{'Sector':<20} {'Yield%':<10} {'P/E':<10} {'Beta':<8} {'YTD%':<10}")
    print("-"*58)
    for s, d in sorted(fund_data.items(), key=lambda x: x[1].get('yield_pct', 0), reverse=True):
        print(f"{d['name']:<20} {d['yield_pct']*100:<10.2f} {d['pe_ratio']:<10.1f} {d['beta']:<8.2f} {d['ytd_return']*100:<10.2f}")

# Fundamental factor -> forward returns
# Group sectors by dividend yield quintile, test forward returns
print("\n--- Dividend Yield as Sector Rotation Signal ---")
# Use trailing 12m returns as proxy when fundamentals are stale
trailing_12m = closes.pct_change(252).iloc[-1]
trailing_3m = closes.pct_change(63).iloc[-1]

print(f"{'Sector':<20} {'12m Return%':<13} {'3m Return%':<12} {'Yield%':<10}")
print("-"*55)
for s in sector_list:
    if s in trailing_12m.index:
        y = fund_data.get(s, {}).get('yield_pct', 0) * 100
        print(f"{SECTOR_ETFS.get(s,s):<20} {trailing_12m[s]*100:<13.2f} {trailing_3m.get(s, 0)*100:<12.2f} {y:<10.2f}")

# ============================================================
# ANALYSIS 4: VOLATILITY COMPRESSION AS ENTRY SIGNAL
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: VOLATILITY COMPRESSION SIGNALS")
print("="*60)

vol_results = {}
for s in sector_list:
    if s not in daily_returns.columns:
        continue
    r = daily_returns[s].dropna()

    # Realized vol (20d)
    rv20 = r.rolling(20).std() * np.sqrt(252)
    # 60d average vol
    rv60_avg = rv20.rolling(60).mean()
    # Vol ratio
    vol_ratio = rv20 / rv60_avg

    # Forward returns when vol is compressed (ratio < 0.7) vs expanded (ratio > 1.3)
    fwd5 = r.shift(-5).rolling(5).sum()
    fwd10 = r.shift(-10).rolling(10).sum()

    aligned = pd.DataFrame({
        'vol_ratio': vol_ratio,
        'fwd5': fwd5,
        'fwd10': fwd10,
        'rel_perf_20d': (closes[s].pct_change(20) - spy_close.pct_change(20)) if s != 'SPY' else r.rolling(20).sum()
    }).dropna()

    compressed = aligned[aligned['vol_ratio'] < 0.7]
    normal = aligned[(aligned['vol_ratio'] >= 0.7) & (aligned['vol_ratio'] <= 1.3)]
    expanded = aligned[aligned['vol_ratio'] > 1.3]

    # Vol compression + underperformance combo signal
    combo = aligned[(aligned['vol_ratio'] < 0.7) & (aligned['rel_perf_20d'] < 0)]

    vol_results[s] = {
        'compressed_fwd5_bps': round(compressed['fwd5'].mean() * 10000, 1) if len(compressed) > 10 else None,
        'normal_fwd5_bps': round(normal['fwd5'].mean() * 10000, 1) if len(normal) > 10 else None,
        'expanded_fwd5_bps': round(expanded['fwd5'].mean() * 10000, 1) if len(expanded) > 10 else None,
        'combo_fwd5_bps': round(combo['fwd5'].mean() * 10000, 1) if len(combo) > 10 else None,
        'compressed_n': len(compressed),
        'combo_n': len(combo),
    }

print(f"\n{'Sector':<20} {'Compressed':<12} {'Normal':<10} {'Expanded':<10} {'Combo':<10} {'Combo N':<8}")
print(f"{'':20} {'5d bps':<12} {'5d bps':<10} {'5d bps':<10} {'5d bps':<10}")
print("-"*70)
for s in sector_list:
    if s in vol_results:
        v = vol_results[s]
        c = v['compressed_fwd5_bps'] or 'N/A'
        n = v['normal_fwd5_bps'] or 'N/A'
        e = v['expanded_fwd5_bps'] or 'N/A'
        cb = v['combo_fwd5_bps'] or 'N/A'
        cn = v['combo_n']

        def fmt(x):
            return f"{x:<10}" if isinstance(x, str) else f"{x:<10.1f}"

        print(f"{SECTOR_ETFS.get(s,s):<20} {fmt(c):<12} {fmt(n):<10} {fmt(e):<10} {fmt(cb):<10} {cn:<8}")

# Vol compression direction prediction
print("\n--- Does Vol Compression Predict Direction? ---")
print(f"{'Sector':<20} {'Comp->Up%':<12} {'Comp->Dn%':<12} {'Comp Sharpe':<12}")
print("-"*56)
for s in sector_list:
    if s not in daily_returns.columns:
        continue
    r = daily_returns[s].dropna()
    rv20 = r.rolling(20).std() * np.sqrt(252)
    rv60_avg = rv20.rolling(60).mean()
    vol_ratio = rv20 / rv60_avg
    fwd10 = r.shift(-10).rolling(10).sum()

    aligned = pd.DataFrame({'vr': vol_ratio, 'fwd': fwd10}).dropna()
    comp = aligned[aligned['vr'] < 0.7]
    if len(comp) > 20:
        up_pct = (comp['fwd'] > 0).mean() * 100
        dn_pct = 100 - up_pct
        sharpe = comp['fwd'].mean() / comp['fwd'].std() * np.sqrt(25.2) if comp['fwd'].std() > 0 else 0
        print(f"{SECTOR_ETFS.get(s,s):<20} {up_pct:<12.1f} {dn_pct:<12.1f} {sharpe:<12.3f}")

# ============================================================
# ANALYSIS 5: WHEEL STRATEGY OPTIMIZATION
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 5: WHEEL STRATEGY OPTIMIZATION")
print("="*60)

# Analyze which sectors have the best risk/reward for selling puts
# Key metrics: vol (premium), drawdown recovery, mean reversion
print("\n--- Put-Selling Attractiveness Score ---")
print("Score = High implied vol proxy + fast recovery + mean reversion")

wheel_scores = {}
for s in sector_list:
    if s not in daily_returns.columns:
        continue
    r = daily_returns[s].dropna()
    c = closes[s].dropna()

    # Realized vol (annualized) - proxy for premium
    rv = r.std() * np.sqrt(252)

    # Mean reversion: autocorrelation at lag 5
    if len(r) > 10:
        ac5 = r.autocorr(lag=5)
    else:
        ac5 = 0

    # Drawdown recovery speed: median days to recover from >5% drawdown
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    in_dd = dd < -0.05
    # Count consecutive DD days
    dd_lengths = []
    count = 0
    for val in in_dd:
        if val:
            count += 1
        else:
            if count > 0:
                dd_lengths.append(count)
            count = 0
    median_recovery = np.median(dd_lengths) if dd_lengths else 999

    # Sharpe of buy-and-hold (underlying quality)
    bh_sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

    # Put selling score: want HIGH vol, FAST recovery, POSITIVE drift, NEGATIVE autocorr (mean reversion)
    score = rv * 100 - median_recovery * 0.1 + bh_sharpe * 5 - ac5 * 10

    wheel_scores[s] = {
        'rv_ann': round(rv * 100, 1),
        'median_recovery_days': round(median_recovery, 0),
        'bh_sharpe': round(bh_sharpe, 3),
        'autocorr_5d': round(ac5, 3),
        'score': round(score, 2),
    }

print(f"\n{'Sector':<20} {'AnnVol%':<10} {'RecovDays':<11} {'BH Sharpe':<11} {'AC(5)':<8} {'Score':<8}")
print("-"*68)
for s, v in sorted(wheel_scores.items(), key=lambda x: x[1]['score'], reverse=True):
    print(f"{SECTOR_ETFS.get(s,s):<20} {v['rv_ann']:<10.1f} {v['median_recovery_days']:<11.0f} {v['bh_sharpe']:<11.3f} {v['autocorr_5d']:<8.3f} {v['score']:<8.2f}")

# Delta-based strike analysis
print("\n--- Optimal Strike Selection (Delta Analysis) ---")
print("Using historical data to estimate P(touch) at various % OTM levels")
print(f"\n{'Sector':<20} {'P(5%DD)30d':<12} {'P(10%DD)30d':<13} {'P(15%DD)30d':<13} {'AvgDD%':<10}")
print("-"*68)

for s in sector_list:
    if s not in daily_returns.columns:
        continue
    r = daily_returns[s].dropna()

    # Rolling 30d min return (proxy for how often you'd get assigned)
    rolling_30d_min = r.rolling(21).apply(lambda x: (1+x).cumprod().min() - 1, raw=False)
    rolling_30d_min = rolling_30d_min.dropna()

    p_5 = (rolling_30d_min < -0.05).mean() * 100
    p_10 = (rolling_30d_min < -0.10).mean() * 100
    p_15 = (rolling_30d_min < -0.15).mean() * 100
    avg_dd = rolling_30d_min.mean() * 100

    print(f"{SECTOR_ETFS.get(s,s):<20} {p_5:<12.1f} {p_10:<13.1f} {p_15:<13.1f} {avg_dd:<10.2f}")

# ============================================================
# ANALYSIS 6: CROSS-SECTOR CORRELATION REGIMES
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 6: CROSS-SECTOR CORRELATION REGIMES")
print("="*60)

# Rolling 60d correlation matrix, track average pairwise correlation
sector_returns = daily_returns[[s for s in sector_list if s in daily_returns.columns and s != 'SPY']]

def avg_pairwise_corr(window_returns):
    """Average off-diagonal correlation."""
    corr = window_returns.corr()
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    return corr.values[mask].mean()

print("Computing rolling 60d average pairwise correlation...")
rolling_corr = []
dates = []
for i in range(60, len(sector_returns)):
    window = sector_returns.iloc[i-60:i]
    if window.shape[1] > 2:
        avg_c = avg_pairwise_corr(window)
        rolling_corr.append(avg_c)
        dates.append(sector_returns.index[i])

corr_series = pd.Series(rolling_corr, index=dates)

# Current regime
current_corr = corr_series.iloc[-1]
corr_percentile = (corr_series < current_corr).mean() * 100

print(f"\nCurrent avg pairwise correlation: {current_corr:.3f}")
print(f"Percentile (vs 3y history): {corr_percentile:.0f}th")

# Regime classification
high_corr = corr_series > corr_series.quantile(0.75)  # Macro-driven
low_corr = corr_series < corr_series.quantile(0.25)   # Alpha opportunity

# Forward SPY returns by regime
spy_r = daily_returns['SPY']
fwd_20d = spy_r.shift(-20).rolling(20).sum()

regime_aligned = pd.DataFrame({
    'corr': corr_series,
    'fwd_20d': fwd_20d,
    'spy_r': spy_r,
}).dropna()

high_regime = regime_aligned[regime_aligned['corr'] > corr_series.quantile(0.75)]
low_regime = regime_aligned[regime_aligned['corr'] < corr_series.quantile(0.25)]

print(f"\n--- Regime Performance ---")
print(f"High correlation regime (macro-driven):")
print(f"  Days: {len(high_regime)}, Avg 20d fwd return: {high_regime['fwd_20d'].mean()*100:.2f}%")
print(f"  Daily vol: {high_regime['spy_r'].std()*np.sqrt(252)*100:.1f}%")

print(f"Low correlation regime (alpha opportunity):")
print(f"  Days: {len(low_regime)}, Avg 20d fwd return: {low_regime['fwd_20d'].mean()*100:.2f}%")
print(f"  Daily vol: {low_regime['spy_r'].std()*np.sqrt(252)*100:.1f}%")

# Does sector momentum strategy work better in low-corr regime?
print("\n--- Momentum Strategy by Correlation Regime ---")
# Re-run 1m momentum L/S strategy, split by regime
mom_1m = momentum_signals['1m'].dropna(how='all')
strat_ret_series = pd.Series(dtype=float)

for i in range(len(mom_1m) - 1):
    row = mom_1m.iloc[i].dropna()
    if len(row) < 6:
        continue
    top3 = row.nlargest(3).index
    bot3 = row.nsmallest(3).index
    dt = mom_1m.index[i]

    if i+1 < len(daily_returns):
        next_day = daily_returns.iloc[i+1]
        long_ret = next_day[top3].mean() if all(t in next_day.index for t in top3) else 0
        short_ret = next_day[bot3].mean() if all(t in next_day.index for t in bot3) else 0
        strat_ret_series[dt] = long_ret - short_ret

# Split by regime
common_dates = strat_ret_series.index.intersection(corr_series.index)
if len(common_dates) > 50:
    strat_aligned = strat_ret_series.loc[common_dates]
    corr_aligned = corr_series.loc[common_dates]

    hi_mask = corr_aligned > corr_series.quantile(0.75)
    lo_mask = corr_aligned < corr_series.quantile(0.25)

    hi_sharpe = strat_aligned[hi_mask].mean() / strat_aligned[hi_mask].std() * np.sqrt(252) if strat_aligned[hi_mask].std() > 0 else 0
    lo_sharpe = strat_aligned[lo_mask].mean() / strat_aligned[lo_mask].std() * np.sqrt(252) if strat_aligned[lo_mask].std() > 0 else 0

    print(f"Momentum L/S Sharpe in HIGH correlation regime: {hi_sharpe:.3f}")
    print(f"Momentum L/S Sharpe in LOW correlation regime:  {lo_sharpe:.3f}")

# Current sector pair correlations
print("\n--- Current Top/Bottom Sector Correlations (60d) ---")
recent_returns = sector_returns.iloc[-60:]
corr_matrix = recent_returns.corr()

# Get all pairs
pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

pairs.sort(key=lambda x: x[2])
print("\nLeast correlated pairs (diversification):")
for a, b, c in pairs[:5]:
    print(f"  {SECTOR_ETFS.get(a,a)} / {SECTOR_ETFS.get(b,b)}: {c:.3f}")

print("\nMost correlated pairs (redundant):")
for a, b, c in pairs[-5:]:
    print(f"  {SECTOR_ETFS.get(a,a)} / {SECTOR_ETFS.get(b,b)}: {c:.3f}")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
all_results = {
    'time_of_day': results_tod,
    'vol_compression': vol_results,
    'wheel_scores': wheel_scores,
    'ma_crossover': ma_results,
    'fundamentals': fund_data,
    'correlation': {
        'current_avg_corr': round(current_corr, 3),
        'percentile': round(corr_percentile, 1),
    }
}

with open(f'{OUT}/analysis_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\nAll results saved to {OUT}/analysis_results.json")
print("ANALYSIS COMPLETE")
