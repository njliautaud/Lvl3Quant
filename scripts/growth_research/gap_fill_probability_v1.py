#!/usr/bin/env python3
"""
Gap Fill Probability Strategy v1
=================================
OBSERVATION:
  Stocks that gap up or down at the open tend to "fill the gap" (return to
  previous close) within a few days. But WHICH gaps fill and which don't?
  
  Key variables:
  - Gap size (small gaps fill more often than large ones)
  - Gap direction relative to trend (counter-trend gaps more likely to fill)
  - Volume on gap day (high volume = institutional = less likely to fill)
  - Prior volatility (compressed vol + gap = spring release, less likely to fill)

  This is DIFFERENT from same-day overnight gap reversal (which failed).
  This tests MULTI-DAY gap fill on a per-stock basis with conditional filtering.

STRATEGY:
  Gap DOWN → buy at open, hold until price returns to prev_close OR max hold expires
  Gap UP → short at open (expect weaker)

VARIANTS (~36):
  Gap size: 1%, 2%, 3%
  Direction: gap_down (long), gap_up (short), both
  Volume filter: none, low_vol (<1.5x avg), high_vol (>1.5x avg)
  Max hold: 3d, 5d, 10d
  Trend filter: none, with_trend, counter_trend

VALIDATION: Permutation test (200 shuffles, p<0.05) + regime test + per-year consistency
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

warnings.filterwarnings('ignore')

# ─── Configuration ───
START_DATE = '2014-01-01'
END_DATE = '2026-07-01'
OUTPUT_DIR = '/home/nick/Lvl3Quant/output/gap_fill_probability_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variant parameters
GAP_SIZES = [0.01, 0.02, 0.03]          # 1%, 2%, 3%
DIRECTIONS = ['gap_down', 'gap_up', 'both']
VOLUME_FILTERS = ['none', 'low_vol', 'high_vol']
MAX_HOLDS = [3, 5, 10]
TREND_FILTERS = ['none', 'with_trend', 'counter_trend']

# Constants
VOL_AVG_WINDOW = 20       # 20-day average volume for volume filter
TREND_WINDOW = 20         # 20-day SMA for trend filter
VOL_THRESHOLD = 1.5       # volume filter multiplier
N_PERMUTATIONS = 200
MIN_TRADES = 30           # minimum trades for statistical validity
REGIME_GAP_THRESHOLD = 0.50

print(f"{'='*70}")
print(f"GAP FILL PROBABILITY STRATEGY v1")
print(f"Observation-driven research")
print(f"Start: {START_DATE}  End: {END_DATE}")
print(f"{'='*70}")

# ─── Step 1: Universe ───
print("\n[1/6] Building stock universe (top ~200 S&P 500)...")

SP500_TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK-B','TSLA','UNH','XOM',
    'JNJ','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY',
    'PEP','KO','COST','AVGO','WMT','MCD','CSCO','TMO','ABT','CRM',
    'ACN','DHR','ADBE','NKE','TXN','LIN','NEE','PM','UNP','RTX',
    'BMY','LOW','AMGN','HON','UPS','QCOM','INTC','IBM','AMAT','BA',
    'CAT','GE','INTU','DE','ISRG','MDLZ','SBUX','GILD','ADP','SYK',
    'BLK','ADI','PLD','REGN','VRTX','MMC','BKNG','CI','CB','CME',
    'SCHW','ZTS','TMUS','TJX','SLB','LRCX','MO','DUK','SO','BDX',
    'CL','ICE','EOG','WM','PNC','AON','NOC','SHW','CSX','MCK',
    'FDX','ITW','NSC','EMR','GM','PXD','MPC','PSA','FCX','OXY',
    'GD','AIG','TRV','HUM','KMB','AEP','D','SRE','EXC','XEL',
    'APD','ECL','ORLY','AZO','ROP','CTAS','MNST','IDXX','ODFL','FAST',
    'CMG','DXCM','CPRT','MCHP','KLAC','CDNS','SNPS','FTNT','BIIB','MRNA',
    'KDP','EW','HCA','MSCI','MTD','IQV','DG','DLTR','ROST','TT',
    'A','PCAR','PAYX','VRSK','KEYS','ANSS','CTSH','WEC','ES','AEE',
    'CMS','LNT','EVRG','AWK','WBA','DVN','HAL','FANG','TRGP','KMI',
    'OKE','WMB','ET','LNG','PSX','VLO','MRO','HES','APA','CTRA',
    'COF','DFS','SYF','ALLY','KEY','RF','CFG','HBAN','ZION','FHN',
    'FITB','MTB','CMA','TFC','USB','PFG','LNC','GL','MET','PRU',
    'AFL','AJG','MMM','GWW','SWK','IEX','AME','ROK','NDSN','PH',
]

print(f"  Hardcoded universe: {len(SP500_TICKERS)} tickers")

# ─── Step 2: Download data ───
print("\n[2/6] Downloading daily OHLCV data from yfinance...")

import yfinance as yf

all_data = {}
failed = []
batch_size = 50

def flatten_cols(df):
    """Handle yfinance MultiIndex columns (newer versions)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) if df.columns.nlevels == 2 else df.columns
    return df

for i in range(0, len(SP500_TICKERS), batch_size):
    batch = SP500_TICKERS[i:i+batch_size]
    tickers_str = ' '.join(batch)
    print(f"  Downloading batch {i//batch_size + 1}/{(len(SP500_TICKERS)-1)//batch_size + 1} ({len(batch)} tickers)...")
    try:
        data = yf.download(tickers_str, start=START_DATE, end=END_DATE,
                          group_by='ticker', progress=False, threads=True)
        for t in batch:
            try:
                if len(batch) == 1:
                    df = flatten_cols(data[['Open','High','Low','Close','Volume']].copy()).dropna()
                else:
                    df = data[t][['Open','High','Low','Close','Volume']].copy()
                    df = flatten_cols(df).dropna()
                if len(df) > 500:  # need enough history
                    all_data[t] = df
            except:
                failed.append(t)
    except Exception as e:
        print(f"  Batch failed: {e}")
        failed.extend(batch)
    time.sleep(0.5)

print(f"  Successfully loaded: {len(all_data)} stocks")
if failed:
    print(f"  Failed: {len(failed)} tickers")

# ─── Step 3: Get SPY for regime classification ───
print("\n[3/6] Downloading SPY for regime classification...")

spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
spy = flatten_cols(spy)
spy_close = spy['Close']
spy_returns = spy_close.pct_change()

# Regime: green day = SPY close > prev close, red day = SPY close < prev close
spy_regime = pd.Series('flat', index=spy.index, dtype=str)
spy_regime.loc[spy_returns > 0] = 'green'
spy_regime.loc[spy_returns < 0] = 'red'
print(f"  SPY regime days - Green: {(spy_regime=='green').sum()}, Red: {(spy_regime=='red').sum()}")


# ─── Step 4: Run all variants ───
print(f"\n[4/6] Running strategy variants...")

def compute_gap_signals(df, gap_size, direction, volume_filter, trend_filter):
    """Identify gap events and return signal dates + directions."""
    opens = df['Open'].values
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    dates = df.index
    
    # Compute features
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan
    gap_pct = (opens - prev_close) / prev_close
    
    # Volume average
    vol_avg = pd.Series(volumes, index=dates).rolling(VOL_AVG_WINDOW).mean().values
    vol_ratio = volumes / np.where(vol_avg > 0, vol_avg, 1)
    
    # Trend: 20-day SMA of close
    sma20 = pd.Series(closes, index=dates).rolling(TREND_WINDOW).mean().values
    trend_up = closes > sma20  # price above SMA = uptrend
    
    signals = []
    
    for i in range(max(TREND_WINDOW, VOL_AVG_WINDOW) + 1, len(df)):
        if np.isnan(gap_pct[i]) or np.isnan(vol_avg[i]):
            continue
            
        abs_gap = abs(gap_pct[i])
        if abs_gap < gap_size:
            continue
        
        is_gap_down = gap_pct[i] < 0
        is_gap_up = gap_pct[i] > 0
        
        # Direction filter
        if direction == 'gap_down' and not is_gap_down:
            continue
        if direction == 'gap_up' and not is_gap_up:
            continue
        
        # Volume filter
        if volume_filter == 'low_vol' and vol_ratio[i] >= VOL_THRESHOLD:
            continue
        if volume_filter == 'high_vol' and vol_ratio[i] < VOL_THRESHOLD:
            continue
        
        # Trend filter
        if trend_filter == 'with_trend':
            # Gap in direction of trend (gap down in downtrend, gap up in uptrend)
            if is_gap_down and trend_up[i-1]:
                continue
            if is_gap_up and not trend_up[i-1]:
                continue
        elif trend_filter == 'counter_trend':
            # Gap against trend (gap down in uptrend, gap up in downtrend)
            if is_gap_down and not trend_up[i-1]:
                continue
            if is_gap_up and trend_up[i-1]:
                continue
        
        # Signal: trade direction (1 = long for gap down fill, -1 = short for gap up fill)
        trade_dir = 1 if is_gap_down else -1
        target_price = prev_close[i]  # gap fill target
        entry_price = opens[i]
        
        signals.append({
            'idx': i,
            'date': dates[i],
            'trade_dir': trade_dir,
            'entry_price': entry_price,
            'target_price': target_price,
            'gap_pct': gap_pct[i],
        })
    
    return signals


def simulate_trades(df, signals, max_hold):
    """Simulate gap fill trades with max hold period."""
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    n = len(df)
    
    trades = []
    
    for sig in signals:
        i = sig['idx']
        entry = sig['entry_price']
        target = sig['target_price']
        trade_dir = sig['trade_dir']
        
        if i + 1 >= n:
            continue
        
        # Check each day for gap fill
        filled = False
        exit_price = None
        exit_date = None
        hold_days = 0
        
        for j in range(i, min(i + max_hold, n)):
            hold_days = j - i + 1
            
            if trade_dir == 1:  # long, waiting for price to rise to target
                if highs[j] >= target:
                    exit_price = target
                    exit_date = dates[j]
                    filled = True
                    break
            else:  # short, waiting for price to drop to target
                if lows[j] <= target:
                    exit_price = target
                    exit_date = dates[j]
                    filled = True
                    break
        
        if not filled:
            # Exit at close on max hold day
            exit_idx = min(i + max_hold - 1, n - 1)
            exit_price = closes[exit_idx]
            exit_date = dates[exit_idx]
            hold_days = exit_idx - i + 1
        
        # P&L
        if trade_dir == 1:
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry
        
        trades.append({
            'entry_date': sig['date'],
            'exit_date': exit_date,
            'trade_dir': trade_dir,
            'pnl_pct': pnl_pct,
            'filled': filled,
            'hold_days': hold_days,
            'gap_pct': sig['gap_pct'],
        })
    
    return trades


def compute_metrics(trades_df):
    """Compute strategy metrics from trades DataFrame."""
    if len(trades_df) < MIN_TRADES:
        return None
    
    returns = trades_df['pnl_pct'].values
    n_trades = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return None
    
    # Annualize assuming ~252 trading days, average ~20 trades/year
    trades_per_year = max(1, n_trades / 10)  # rough: 10 years of data
    ann_return = mean_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year)
    
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else std_ret
    sortino = ann_return / (downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0
    
    # Win rate and profit factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    fill_rate = trades_df['filled'].mean()
    avg_hold = trades_df['hold_days'].mean()
    
    return {
        'n_trades': n_trades,
        'mean_return_pct': float(mean_ret * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'fill_rate': float(fill_rate),
        'avg_hold_days': float(avg_hold),
        'total_return_pct': float(np.sum(returns) * 100),
        'max_dd_pct': float(np.min(np.minimum.accumulate(np.cumsum(returns)) - np.cumsum(returns)) * 100) if len(returns) > 1 else 0,
    }


def permutation_test(returns, n_perms=N_PERMUTATIONS):
    """Shuffle returns, compute fraction of shuffled Sharpe >= actual."""
    if len(returns) < MIN_TRADES:
        return 1.0
    actual_mean = np.mean(returns)
    actual_std = np.std(returns)
    if actual_std == 0:
        return 1.0
    actual_sharpe = actual_mean / actual_std
    
    count_better = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(returns)
        s_mean = np.mean(shuffled)
        s_std = np.std(shuffled)
        if s_std > 0:
            s_sharpe = s_mean / s_std
            if s_sharpe >= actual_sharpe:
                count_better += 1
    
    return count_better / n_perms


def regime_test(trades_df, spy_regime):
    """Test if strategy works in both green and red regimes."""
    trades_with_regime = trades_df.copy()
    # Map entry date to regime
    regime_map = spy_regime.to_dict()
    trades_with_regime['regime'] = trades_with_regime['entry_date'].map(
        lambda d: regime_map.get(d, 'unknown')
    )
    
    green = trades_with_regime[trades_with_regime['regime'] == 'green']['pnl_pct']
    red = trades_with_regime[trades_with_regime['regime'] == 'red']['pnl_pct']
    
    if len(green) < 10 or len(red) < 10:
        return None, None, None
    
    green_std = np.std(green)
    red_std = np.std(red)
    tpy_g = max(1, len(green) / 10)
    tpy_r = max(1, len(red) / 10)
    
    sharpe_green = (np.mean(green) * tpy_g) / (green_std * np.sqrt(tpy_g)) if green_std > 0 else 0
    sharpe_red = (np.mean(red) * tpy_r) / (red_std * np.sqrt(tpy_r)) if red_std > 0 else 0
    
    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 999
    
    return float(sharpe_green), float(sharpe_red), float(regime_gap)


def year_consistency(trades_df):
    """Check if profitable in >60% of years."""
    trades_df = trades_df.copy()
    trades_df['year'] = trades_df['entry_date'].dt.year
    yearly = trades_df.groupby('year')['pnl_pct'].sum()
    if len(yearly) < 3:
        return 0.0, {}
    pct_profitable = (yearly > 0).sum() / len(yearly)
    return float(pct_profitable), {int(k): float(v*100) for k, v in yearly.items()}


# Generate all variant combinations
variants = list(product(GAP_SIZES, DIRECTIONS, VOLUME_FILTERS, MAX_HOLDS, TREND_FILTERS))
print(f"  Total variants to test: {len(variants)}")

results = []
best_sharpe = -999
best_variant = None

for vi, (gap_size, direction, vol_filter, max_hold, trend_filter) in enumerate(variants):
    variant_name = f"gap{int(gap_size*100)}pct_{direction}_vol{vol_filter}_hold{max_hold}d_trend{trend_filter}"
    
    if (vi + 1) % 20 == 0 or vi == 0:
        print(f"\n  Variant {vi+1}/{len(variants)}: {variant_name}")
    
    # Collect trades across all stocks
    all_trades = []
    
    for ticker, df in all_data.items():
        signals = compute_gap_signals(df, gap_size, direction, vol_filter, trend_filter)
        if not signals:
            continue
        trades = simulate_trades(df, signals, max_hold)
        for t in trades:
            t['ticker'] = ticker
        all_trades.extend(trades)
    
    if len(all_trades) < MIN_TRADES:
        continue
    
    trades_df = pd.DataFrame(all_trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    
    # Compute metrics
    metrics = compute_metrics(trades_df)
    if metrics is None:
        continue
    
    # Validation gate 1: Permutation test
    p_value = permutation_test(trades_df['pnl_pct'].values)
    
    # Validation gate 2: Regime test
    sharpe_green, sharpe_red, regime_gap = regime_test(trades_df, spy_regime)
    
    # Validation gate 3: Year consistency
    year_pct, yearly_pnl = year_consistency(trades_df)
    
    # Determine pass/fail
    pass_perm = p_value < 0.05
    pass_regime = regime_gap is not None and regime_gap < REGIME_GAP_THRESHOLD
    pass_year = year_pct > 0.60
    all_pass = pass_perm and pass_regime and pass_year
    
    result = {
        'variant': variant_name,
        'gap_size_pct': gap_size * 100,
        'direction': direction,
        'volume_filter': vol_filter,
        'max_hold_days': max_hold,
        'trend_filter': trend_filter,
        **metrics,
        'p_value': float(p_value),
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'regime_gap': regime_gap,
        'year_consistency_pct': year_pct,
        'yearly_pnl': yearly_pnl,
        'pass_permutation': pass_perm,
        'pass_regime': pass_regime,
        'pass_year_consistency': pass_year,
        'all_gates_pass': all_pass,
    }
    results.append(result)
    
    if metrics['sharpe'] > best_sharpe:
        best_sharpe = metrics['sharpe']
        best_variant = variant_name
    
    if all_pass:
        print(f"  ✓ PASS: {variant_name} | Sharpe={metrics['sharpe']:.2f} | WR={metrics['win_rate']:.1%} | "
              f"PF={metrics['profit_factor']:.2f} | N={metrics['n_trades']} | Fill={metrics['fill_rate']:.1%} | "
              f"p={p_value:.3f} | RegimeGap={regime_gap:.2f} | YearCons={year_pct:.0%}")

# ─── Step 5: Summary ───
print(f"\n{'='*70}")
print(f"[5/6] RESULTS SUMMARY")
print(f"{'='*70}")

passed = [r for r in results if r['all_gates_pass']]
print(f"\nTotal variants tested: {len(results)}")
print(f"Passed ALL gates: {len(passed)}")

if passed:
    print(f"\n{'─'*70}")
    print(f"PASSING VARIANTS (sorted by Sharpe):")
    print(f"{'─'*70}")
    passed_sorted = sorted(passed, key=lambda x: x['sharpe'], reverse=True)
    for r in passed_sorted[:20]:
        print(f"  {r['variant']}")
        print(f"    Sharpe={r['sharpe']:.2f} Sortino={r['sortino']:.2f} WR={r['win_rate']:.1%} "
              f"PF={r['profit_factor']:.2f} N={r['n_trades']} Fill={r['fill_rate']:.1%}")
        print(f"    p={r['p_value']:.3f} RegimeGap={r['regime_gap']:.2f} "
              f"Sharpe_G={r['sharpe_green']:.2f} Sharpe_R={r['sharpe_red']:.2f} "
              f"YearCons={r['year_consistency_pct']:.0%}")
else:
    print("\nNo variants passed all validation gates.")
    print("\nTop 10 by Sharpe (even though they failed validation):")
    top10 = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:10]
    for r in top10:
        flags = []
        if not r['pass_permutation']: flags.append(f"PERM(p={r['p_value']:.3f})")
        if not r['pass_regime']: flags.append(f"REGIME(gap={r.get('regime_gap','N/A')})")
        if not r['pass_year_consistency']: flags.append(f"YEAR({r['year_consistency_pct']:.0%})")
        print(f"  {r['variant']} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | "
              f"FAIL: {', '.join(flags)}")

# ─── Step 6: Save results ───
print(f"\n[6/6] Saving results...")

output = {
    'metadata': {
        'strategy': 'gap_fill_probability_v1',
        'run_date': datetime.now().isoformat(),
        'universe_size': len(all_data),
        'date_range': f"{START_DATE} to {END_DATE}",
        'total_variants': len(results),
        'passing_variants': len(passed),
    },
    'validation_gates': {
        'permutation_test': '200 shuffles, p < 0.05',
        'regime_test': '|Sharpe_green - Sharpe_red| / max < 0.50',
        'year_consistency': 'profitable in > 60% of years',
    },
    'results': sorted(results, key=lambda x: x['sharpe'], reverse=True),
    'passing_results': sorted(passed, key=lambda x: x['sharpe'], reverse=True) if passed else [],
}

results_path = os.path.join(OUTPUT_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved to {results_path}")

# Also save a CSV summary
if results:
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'variants_summary.csv'), index=False)
    print(f"  Saved CSV summary")

print(f"\n{'='*70}")
print(f"DONE. {len(passed)}/{len(results)} variants passed all validation gates.")
if best_variant:
    print(f"Best Sharpe: {best_sharpe:.2f} ({best_variant})")
print(f"{'='*70}")
