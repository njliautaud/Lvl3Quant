#!/usr/bin/env python3
"""
Vol Compression → Magnitude Explosion Strategy v1
==================================================
OBSERVATION (from market_observation_scanner, HC #735):
  When realized volatility compresses to its 10th percentile (for a stock's
  own history), subsequent absolute moves over the next 5-21 days are ~6% larger.

STRATEGY HYPOTHESIS:
  Buy ATM straddles (simulated via long stock + delta-hedged, or simple
  absolute-return capture) when vol is extremely compressed. The edge is in
  MAGNITUDE, not direction — so we need a strategy that profits from big moves
  regardless of direction.

IMPLEMENTATION:
  1. Identify vol compression events (realized vol < 10th percentile of trailing 252d)
  2. Enter long straddle equivalent: go long AND short in equal size (simulated)
     OR simpler: buy the stock and set symmetric TP/SL to capture magnitude
  3. Actually simplest: track stocks entering vol compression, measure subsequent
     returns, test a vol-breakout momentum strategy (enter on first big move after compression)

UNIVERSE: S&P 500 (large per HC #735 R3)
VALIDATION: Permutation test + regime test + sub-period stability (HC #428)
GPU: Uses PyTorch for accelerated computation on Neptune
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, '/home/nick/Lvl3Quant' if os.path.exists('/home/nick/Lvl3Quant') else '/home/nick/Lvl3Quant')

from datetime import datetime
import json

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    HAS_GPU = False

# ─── Configuration ───
UNIVERSE_SIZE = 300  # Large universe per HC #735
START_DATE = '2015-01-01'
END_DATE = '2026-07-01'
VOL_LOOKBACK = 21  # 1-month realized vol
VOL_HISTORY = 252  # 1-year history for percentile
COMPRESSION_PCT = 10  # Bottom 10th percentile = compressed
HOLD_PERIODS = [5, 10, 21]  # Test multiple holding periods
MIN_TRADES_PER_PERIOD = 20  # Minimum for statistical validity
OUTPUT_DIR = '/home/nick/Lvl3Quant/output/vol_compression_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"VOL COMPRESSION → MAGNITUDE EXPLOSION STRATEGY v1")
print(f"Observation-first research (HC #735)")
print(f"Universe: ~{UNIVERSE_SIZE} stocks, {START_DATE} to {END_DATE}")
print(f"{'='*70}")

# ─── Step 1: Get universe ───
print("\n[1/7] Building stock universe...")

try:
    import yfinance as yf

    # S&P 500 tickers - comprehensive list
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(sp500_url)
        sp500_tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"  Got {len(sp500_tickers)} S&P 500 tickers from Wikipedia")
    except:
        # Fallback to a large hardcoded list
        sp500_tickers = [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','BRK-B','UNH','JNJ',
            'V','XOM','JPM','WMT','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP',
            'KO','COST','AVGO','TMO','MCD','CSCO','ACN','ABT','DHR','WFC','NEE',
            'DIS','TXN','BMY','PM','UNP','UPS','RTX','AMGN','IBM','COP','LOW','GS',
            'SPGI','MS','BLK','AXP','INTC','CAT','BA','GE','AMD','ISRG','MDLZ',
            'ADI','GILD','BKNG','SYK','ADP','VRTX','TJX','MMC','PGR','CB','REGN',
            'SCHW','ZTS','CI','EOG','SLB','MO','BSX','CME','DUK','SO','LRCX',
            'BDX','CL','FI','ANET','ITW','SHW','APD','PLD','MCK','PNC','USB',
            'HUM','TGT','ICE','SNPS','CDNS','PSA','ORLY','AON','FCX','OXY',
            'AZO','KMB','CMG','MPC','VLO','EMR','SRE','AIG','ALL','D','KHC',
            'SPG','AFL','WMB','PSX','F','GM','CTSH','FDX','NEM','EL','CCI',
            'TFC','WELL','YUM','MNST','HSY','KEYS','NDAQ','WAB','ROP','GIS',
            'DG','DLTR','HCA','FAST','CTAS','PAYX','ON','WEC','AEP','XEL',
            'EXC','ED','AWK','ES','ETR','FE','PPL','CNP','CMS','EVRG',
            'ORCL','CRM','NOW','INTU','ADBE','PYPL','SQ','SHOP','SNOW','DDOG',
            'ZS','CRWD','PANW','FTNT','NET','OKTA','MDB','TEAM','WDAY','VEEV',
            'DOCU','ZM','ROKU','U','PATH','BILL','HUBS','TTD','PINS','SNAP',
            'NFLX','ABNB','UBER','LYFT','DASH','COIN','HOOD','SOFI','AFRM',
            'NKE','SBUX','TGT','LULU','ETSY','W','RH','WSM','TPR','RL',
            'GD','LMT','NOC','HII','LHX','TXT','HWM','CW','LDOS','BAH',
            'PFE','MRNA','BIIB','REGN','VRTX','ILMN','DXCM','ALGN','HOLX','MTD',
            'T','VZ','TMUS','CHTR','CMCSA','PARA','WBD','FOX','FOXA','NWSA',
            'GS','MS','SCHW','BLK','BX','KKR','APO','ARES','OWL','TROW',
            'WM','RSG','WCN','CLH','SRCL','US','GFL','CWST','MEG','ASTE',
            'DE','CAT','CNH','AGCO','TTC','VMC','MLM','SUM','EXP','USCR',
            'LIN','APD','ECL','SHW','PPG','RPM','AXTA','ASH','CC','HUN',
            'AMT','CCI','SBAC','EQIX','DLR','PSA','EXR','CUBE','MAA','UDR',
            'O','NNN','WPC','STOR','ADC','EPRT','VICI','GLPI','MGP','RYN'
        ]
        # Deduplicate
        sp500_tickers = list(dict.fromkeys(sp500_tickers))
        print(f"  Using {len(sp500_tickers)} tickers from hardcoded list")

    # Limit to target size
    tickers = sp500_tickers[:UNIVERSE_SIZE]

except ImportError:
    print("ERROR: yfinance not available")
    sys.exit(1)

# ─── Step 2: Download data ───
print(f"\n[2/7] Downloading price data for {len(tickers)} stocks...")

# Download in batches to avoid timeouts
batch_size = 50
all_close = {}
all_volume = {}

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[0]}..{batch[-1]}")
    try:
        data = yf.download(batch, start=START_DATE, end=END_DATE,
                          group_by='ticker', progress=False, threads=True)
        for t in batch:
            try:
                if len(batch) == 1:
                    close = data['Close'].dropna()
                    vol = data['Volume'].dropna()
                else:
                    close = data[t]['Close'].dropna()
                    vol = data[t]['Volume'].dropna()
                if len(close) > VOL_HISTORY + VOL_LOOKBACK + max(HOLD_PERIODS) + 50:
                    all_close[t] = close
                    all_volume[t] = vol
            except:
                pass
    except Exception as e:
        print(f"  Error in batch: {e}")

print(f"  Got data for {len(all_close)} stocks with sufficient history")

if len(all_close) < 50:
    print("ERROR: Too few stocks. Exiting.")
    sys.exit(1)

# ─── Step 3: Identify vol compression events ───
print(f"\n[3/7] Identifying vol compression events...")

events = []  # List of (ticker, date, realized_vol, vol_percentile, forward_returns)

for ticker, close in all_close.items():
    returns = close.pct_change().dropna()

    # Realized vol (annualized)
    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    # Rolling percentile of vol within its own history
    vol_pctile = realized_vol.rolling(VOL_HISTORY).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == VOL_HISTORY else np.nan,
        raw=False
    )

    # Find compression events
    for idx in vol_pctile.index:
        if pd.isna(vol_pctile[idx]):
            continue
        if vol_pctile[idx] <= COMPRESSION_PCT:
            # Calculate forward returns at various horizons
            idx_pos = close.index.get_loc(idx)
            fwd_rets = {}
            for hp in HOLD_PERIODS:
                if idx_pos + hp < len(close):
                    fwd_ret = (close.iloc[idx_pos + hp] / close.iloc[idx_pos]) - 1
                    fwd_abs_ret = abs(fwd_ret)
                    fwd_rets[f'ret_{hp}d'] = fwd_ret
                    fwd_rets[f'abs_ret_{hp}d'] = fwd_abs_ret

            if fwd_rets:
                events.append({
                    'ticker': ticker,
                    'date': idx,
                    'realized_vol': realized_vol[idx],
                    'vol_percentile': vol_pctile[idx],
                    **fwd_rets
                })

events_df = pd.DataFrame(events)
print(f"  Found {len(events_df)} vol compression events across {events_df['ticker'].nunique()} stocks")
print(f"  Date range: {events_df['date'].min()} to {events_df['date'].max()}")

# ─── Step 4: Analyze the observation ───
print(f"\n[4/7] Analyzing vol compression → magnitude relationship...")

# Compare compressed vs normal periods
# For each stock, compute average absolute returns in compressed vs non-compressed periods
comparison = []
for ticker, close in all_close.items():
    returns = close.pct_change().dropna()
    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    vol_pctile = realized_vol.rolling(VOL_HISTORY).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == VOL_HISTORY else np.nan,
        raw=False
    ).dropna()

    for hp in HOLD_PERIODS:
        fwd_abs = returns.rolling(hp).apply(lambda x: abs(x.sum()), raw=True).shift(-hp)

        merged = pd.DataFrame({
            'vol_pctile': vol_pctile,
            'fwd_abs': fwd_abs
        }).dropna()

        if len(merged) < 100:
            continue

        compressed = merged[merged['vol_pctile'] <= COMPRESSION_PCT]['fwd_abs']
        normal = merged[merged['vol_pctile'] > COMPRESSION_PCT]['fwd_abs']

        if len(compressed) > 10:
            comparison.append({
                'ticker': ticker,
                'hold_period': hp,
                'compressed_abs_ret': compressed.mean(),
                'normal_abs_ret': normal.mean(),
                'magnitude_ratio': compressed.mean() / normal.mean() if normal.mean() > 0 else np.nan,
                'n_compressed': len(compressed),
                'n_normal': len(normal)
            })

comp_df = pd.DataFrame(comparison)
print("\n  MAGNITUDE ANALYSIS (compressed vs normal periods):")
for hp in HOLD_PERIODS:
    sub = comp_df[comp_df['hold_period'] == hp]
    if len(sub) > 0:
        avg_ratio = sub['magnitude_ratio'].mean()
        median_ratio = sub['magnitude_ratio'].median()
        pct_bigger = (sub['magnitude_ratio'] > 1.0).mean() * 100
        print(f"  {hp}d hold: avg magnitude ratio = {avg_ratio:.3f}, "
              f"median = {median_ratio:.3f}, "
              f"{pct_bigger:.0f}% of stocks show larger moves after compression")

# ─── Step 5: Build tradeable strategy ───
print(f"\n[5/7] Building vol-breakout strategy...")

# Strategy: When vol is compressed, wait for a breakout (first day with |return| > 2x recent avg)
# Then follow the breakout direction for HOLD_PERIOD days
# This captures the magnitude edge while getting a directional signal from the breakout

BREAKOUT_MULT = 1.5  # Breakout = daily return > 1.5x average abs return
MAX_WAIT = 10  # Max days to wait for breakout after compression detected
BEST_HP = 10  # Will test all, but start with 10d

strategy_trades = []

for ticker, close in all_close.items():
    returns = close.pct_change().dropna()
    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    avg_abs_ret = returns.abs().rolling(VOL_LOOKBACK).mean()

    vol_pctile = realized_vol.rolling(VOL_HISTORY).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == VOL_HISTORY else np.nan,
        raw=False
    )

    # Find compression starts (first day entering compression)
    in_compression = (vol_pctile <= COMPRESSION_PCT).astype(int)
    compression_starts = in_compression.diff() == 1

    for start_date in compression_starts[compression_starts].index:
        start_pos = close.index.get_loc(start_date)

        # Wait for breakout within MAX_WAIT days
        for wait in range(1, MAX_WAIT + 1):
            check_pos = start_pos + wait
            if check_pos >= len(close) or check_pos >= len(returns):
                break

            try:
                daily_ret = returns.iloc[check_pos]
            except IndexError:
                break
            if pd.isna(daily_ret):
                continue
            threshold = avg_abs_ret.iloc[min(start_pos, len(avg_abs_ret)-1)] * BREAKOUT_MULT
            if pd.isna(threshold) or threshold == 0:
                continue

            if abs(daily_ret) > threshold:
                # Breakout detected! Enter in breakout direction
                entry_date = close.index[check_pos]
                direction = 1 if daily_ret > 0 else -1
                entry_price = close.iloc[check_pos]

                for hp in HOLD_PERIODS:
                    exit_pos = check_pos + hp
                    if exit_pos < len(close):
                        exit_price = close.iloc[exit_pos]
                        pnl_pct = direction * (exit_price / entry_price - 1)

                        # Get SPY return for same period (regime classification)
                        spy_close = all_close.get('SPY')
                        spy_ret = np.nan
                        if spy_close is not None and entry_date in spy_close.index:
                            spy_entry_pos = spy_close.index.get_loc(entry_date)
                            spy_exit_pos = spy_entry_pos + hp
                            if spy_exit_pos < len(spy_close):
                                spy_ret = spy_close.iloc[spy_exit_pos] / spy_close.iloc[spy_entry_pos] - 1

                        strategy_trades.append({
                            'ticker': ticker,
                            'compression_date': start_date,
                            'entry_date': entry_date,
                            'exit_date': close.index[exit_pos],
                            'direction': 'LONG' if direction > 0 else 'SHORT',
                            'wait_days': wait,
                            'breakout_magnitude': abs(daily_ret),
                            'hold_period': hp,
                            'pnl_pct': pnl_pct,
                            'spy_ret': spy_ret,
                            'year': entry_date.year,
                            'month': entry_date.month
                        })
                break  # Only take first breakout per compression event

trades_df = pd.DataFrame(strategy_trades)
print(f"  Generated {len(trades_df)} trades across {trades_df['ticker'].nunique()} stocks")

# ─── Step 6: Evaluate strategy ───
print(f"\n[6/7] Evaluating strategy performance...")

results = {}
for hp in HOLD_PERIODS:
    hp_trades = trades_df[trades_df['hold_period'] == hp].copy()
    if len(hp_trades) < MIN_TRADES_PER_PERIOD:
        print(f"\n  {hp}d hold: SKIPPED (only {len(hp_trades)} trades)")
        continue

    # Sort by entry date for walk-forward analysis
    hp_trades = hp_trades.sort_values('entry_date')

    # Basic metrics
    mean_ret = hp_trades['pnl_pct'].mean()
    median_ret = hp_trades['pnl_pct'].median()
    win_rate = (hp_trades['pnl_pct'] > 0).mean()
    n_trades = len(hp_trades)

    # Annualized Sharpe (assuming trades are semi-independent)
    # Group by entry_date to get daily strategy returns
    daily_rets = hp_trades.groupby('entry_date')['pnl_pct'].mean()
    if len(daily_rets) > 10:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252 / hp)
        sortino_denom = daily_rets[daily_rets < 0].std()
        sortino = daily_rets.mean() / sortino_denom * np.sqrt(252 / hp) if sortino_denom > 0 else np.nan
    else:
        sharpe = sortino = np.nan

    # Profit factor
    gross_profit = hp_trades[hp_trades['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(hp_trades[hp_trades['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max drawdown (cumulative)
    cum_pnl = hp_trades['pnl_pct'].cumsum()
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max
    max_dd = dd.min()

    # Regime analysis (HC #428 R1)
    hp_trades['regime'] = hp_trades['spy_ret'].apply(
        lambda x: 'GREEN' if x > 0.01 else ('RED' if x < -0.01 else 'FLAT')
    )

    regime_sharpes = {}
    for regime in ['GREEN', 'RED', 'FLAT']:
        regime_trades = hp_trades[hp_trades['regime'] == regime]
        if len(regime_trades) > 5:
            r_daily = regime_trades.groupby('entry_date')['pnl_pct'].mean()
            if len(r_daily) > 3 and r_daily.std() > 0:
                regime_sharpes[regime] = r_daily.mean() / r_daily.std() * np.sqrt(252 / hp)
            else:
                regime_sharpes[regime] = np.nan

    # Regime gap test
    if 'GREEN' in regime_sharpes and 'RED' in regime_sharpes:
        s_green = regime_sharpes.get('GREEN', 0)
        s_red = regime_sharpes.get('RED', 0)
        max_s = max(abs(s_green), abs(s_red))
        regime_gap = abs(s_green - s_red) / max_s if max_s > 0 else 0
    else:
        regime_gap = np.nan

    # Sub-period stability
    n_periods = 3
    period_size = len(hp_trades) // n_periods
    sub_sharpes = []
    for p in range(n_periods):
        sub = hp_trades.iloc[p*period_size:(p+1)*period_size]
        sub_daily = sub.groupby('entry_date')['pnl_pct'].mean()
        if len(sub_daily) > 5 and sub_daily.std() > 0:
            sub_sharpes.append(sub_daily.mean() / sub_daily.std() * np.sqrt(252 / hp))
        else:
            sub_sharpes.append(0)

    # Long vs Short breakdown
    long_trades = hp_trades[hp_trades['direction'] == 'LONG']
    short_trades = hp_trades[hp_trades['direction'] == 'SHORT']
    long_wr = (long_trades['pnl_pct'] > 0).mean() if len(long_trades) > 0 else 0
    short_wr = (short_trades['pnl_pct'] > 0).mean() if len(short_trades) > 0 else 0

    results[hp] = {
        'n_trades': n_trades,
        'mean_ret': mean_ret,
        'median_ret': median_ret,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': pf,
        'max_dd': max_dd,
        'regime_sharpes': regime_sharpes,
        'regime_gap': regime_gap,
        'sub_sharpes': sub_sharpes,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'years_active': hp_trades['year'].nunique()
    }

    print(f"\n  ── {hp}d HOLD PERIOD ──")
    print(f"  Trades: {n_trades} ({len(long_trades)}L / {len(short_trades)}S)")
    print(f"  Mean return: {mean_ret*100:.2f}%  |  Median: {median_ret*100:.2f}%")
    print(f"  Win rate: {win_rate:.1%} (Long: {long_wr:.1%}, Short: {short_wr:.1%})")
    print(f"  Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}  |  PF: {pf:.2f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Regime Sharpes: {', '.join(f'{k}={v:.2f}' for k,v in regime_sharpes.items())}")
    print(f"  Regime gap: {regime_gap:.2f} ({'PASS' if regime_gap < 0.50 else 'FAIL'} <0.50)")
    print(f"  Sub-period Sharpes: {', '.join(f'{s:.2f}' for s in sub_sharpes)}")

# ─── Step 7: Permutation test on best variant ───
print(f"\n[7/7] Running permutation tests...")

N_PERMS = 200  # Per HC #659

for hp in HOLD_PERIODS:
    hp_trades = trades_df[trades_df['hold_period'] == hp].copy()
    if len(hp_trades) < MIN_TRADES_PER_PERIOD:
        continue

    # Observed Sharpe
    daily_rets = hp_trades.groupby('entry_date')['pnl_pct'].mean()
    if daily_rets.std() == 0:
        continue
    observed_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252 / hp)

    # Permutation: shuffle direction labels
    perm_sharpes = []
    for _ in range(N_PERMS):
        shuffled = hp_trades.copy()
        # Randomly flip directions
        random_dirs = np.random.choice([1, -1], size=len(shuffled))
        original_dirs = shuffled['direction'].map({'LONG': 1, 'SHORT': -1}).values
        # Keep magnitude, randomize direction
        shuffled['pnl_pct'] = shuffled['pnl_pct'] * random_dirs / original_dirs

        perm_daily = shuffled.groupby('entry_date')['pnl_pct'].mean()
        if perm_daily.std() > 0:
            perm_sharpes.append(perm_daily.mean() / perm_daily.std() * np.sqrt(252 / hp))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).mean()

    print(f"\n  ── {hp}d PERMUTATION TEST ──")
    print(f"  Observed Sharpe: {observed_sharpe:.3f}")
    print(f"  Null mean Sharpe: {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}")
    print(f"  p-value: {p_value:.3f} ({'PASS' if p_value < 0.05 else 'FAIL'} <0.05)")

    if hp in results:
        results[hp]['perm_p_value'] = p_value
        results[hp]['perm_null_mean'] = perm_sharpes.mean()

# ─── Save results ───
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

best_hp = None
best_sharpe = -999
for hp, r in results.items():
    perm_pass = r.get('perm_p_value', 1.0) < 0.05
    regime_pass = r.get('regime_gap', 1.0) < 0.50
    print(f"\n{hp}d: Sharpe {r['sharpe']:.2f}, WR {r['win_rate']:.0%}, PF {r['profit_factor']:.2f}, "
          f"Perm {'PASS' if perm_pass else 'FAIL'} (p={r.get('perm_p_value', 'N/A')}), "
          f"Regime {'PASS' if regime_pass else 'FAIL'} (gap={r.get('regime_gap', 'N/A')})")

    if r['sharpe'] > best_sharpe and perm_pass:
        best_sharpe = r['sharpe']
        best_hp = hp

if best_hp:
    print(f"\nBEST VARIANT: {best_hp}d hold (Sharpe {best_sharpe:.2f})")
else:
    print(f"\nNO VARIANT PASSES PERMUTATION TEST — observation may not be tradeable")

# Save detailed results
trades_df.to_csv(f"{OUTPUT_DIR}/all_trades.csv", index=False)
comp_df.to_csv(f"{OUTPUT_DIR}/magnitude_comparison.csv", index=False)

summary = {
    'strategy': 'vol_compression_magnitude_v1',
    'observation': 'Vol compression to 10th percentile leads to larger subsequent moves',
    'universe_size': len(all_close),
    'date_range': f'{START_DATE} to {END_DATE}',
    'total_events': len(events_df),
    'results': {}
}
for hp, r in results.items():
    summary['results'][str(hp)] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                     for k, v in r.items() if k != 'regime_sharpes'}
    summary['results'][str(hp)]['regime_sharpes'] = {
        k: float(v) for k, v in r.get('regime_sharpes', {}).items()
    }

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
