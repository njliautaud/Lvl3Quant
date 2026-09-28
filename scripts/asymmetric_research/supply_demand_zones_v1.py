#!/usr/bin/env python3
"""
Supply/Demand Zone Memory v1 — Asymmetric Research
====================================================
Hypothesis: Stocks approaching price zones with 2+ prior bounces in the trailing
year produce asymmetric returns because institutional limit orders create a "floor."

Bounce = drop >=2% then recover >=2% within 10 days.
Zone = 2% tolerance around bounce low.
Entry = price enters a zone with N+ prior bounces in trailing 252 days.

Variants tested:
  zone2_long_21d        2+ bounces, hold 21d (base)
  zone3_long_21d        3+ bounces, hold 21d
  zone2_vc_21d          2+ bounces + vol compression (ATR < 20th pctile)
  zone2_oversold_21d    2+ bounces + RSI < 30
  zone2_mfi_21d         2+ bounces + MFI < 30
  zone2_volume_21d      2+ bounces + volume > 1.5x 20d avg
  zone2_long_10d        2+ bounces, hold 10d
  zone_fresh_21d        First revisit (2nd touch only)
  zone2_trend_21d       2+ bounces + above 50 SMA
  zone2_countertrend_21d 2+ bounces + below 50 SMA

Validation gates:
  - 200-shuffle permutation test (p < 0.05)
  - Regime gap < 0.50
  - Year consistency > 70%

Author: Claude (autonomous research)
Date: 2026-07-22
"""

import os
import sys
import time
import warnings
import hashlib
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR = '/home/nick/Lvl3Quant/data/supply_demand_cache'
RESULTS_DIR = '/home/nick/Lvl3Quant/data/supply_demand_results'
WORKERS = 8
LOOKBACK = 252          # trailing days for zone counting
BOUNCE_DROP_PCT = 0.02  # 2% drop to qualify as bounce
BOUNCE_RECOV_PCT = 0.02 # 2% recovery to confirm bounce
BOUNCE_WINDOW = 10      # days for recovery
ZONE_TOL = 0.02         # 2% zone tolerance
PERM_SHUFFLES = 200     # permutation test iterations
START_YEAR = 2012
END_YEAR = 2026
SPY_TICKER = 'SPY'

# S&P 500 tickers (comprehensive list)
SP500_TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK-B','UNH','XOM','JNJ',
    'JPM','V','PG','MA','AVGO','HD','CVX','MRK','ABBV','LLY',
    'PEP','KO','COST','TMO','MCD','WMT','CSCO','ABT','CRM','ACN',
    'DHR','ADBE','TXN','NEE','NKE','PM','BMY','UPS','LIN','RTX',
    'QCOM','MDT','HON','UNP','SCHW','LOW','MS','INTC','T','IBM',
    'GS','BLK','AMGN','AXP','BA','ISRG','CAT','ELV','GILD','SYK',
    'DE','MDLZ','ADI','PLD','ADP','BKNG','REGN','CI','VRTX','PYPL',
    'CB','MMC','SO','DUK','PGR','TJX','CME','MO','CL','SLB',
    'ICE','FIS','APD','WM','EMR','EQIX','SHW','NSC','ITW','GD',
    'AON','ECL','PH','ANET','MCK','MPC','PSA','KMB','CCI','D',
    'ORLY','AIG','FTNT','AEP','MNST','F','GM','TRGP','PSX','VLO',
    'HUM','SRE','TGT','KHC','CTVA','WELL','AZO','GIS','AFL','DOW',
    'CMG','ROP','WEC','O','HCA','TEL','ED','DTE','FE','EXC',
    'PPL','XEL','AWK','ETR','AES','LHX','ALL','STZ','YUM','DLTR',
    'DHI','LEN','NUE','FAST','CTAS','PAYX','CPRT','ODFL','PCAR','EW',
    'IDXX','ILMN','DXCM','BDX','ZTS','MTD','IQV','A','WST','RMD',
    'TECH','ALGN','TER','POOL','KEYS','MPWR','MKTX','TYL','ZBRA','CDW',
    'NDSN','BIO','EPAM','PTC','PAYC','GNRC','SEDG','ENPH','WRB','PKG',
    'AVY','IEX','FTV','TRMB','WAT','BR','LKQ','J','DGX','AKAM',
    'CE','CBOE','FMC','ALB','LNT','EVRG','ATO','NI','CMS','CNP',
    'PNW','ATMOS','WRK','IP','SEE','EMN','HRL','SJM','CPB','CAG',
    'MKC','CHD','CLX','K','HSY','TSN','HII','BWA','LVS','MGM',
    'WYNN','CZR','NCLH','RCL','CCL','DRI','SBUX','CMI','ROK','AME',
    'ETN','IR','DOV','SWK','GPC','TROW','BEN','IVZ','NDAQ','MKTX',
    'MSCI','SPGI','MCO','FDS','ICE','CME','CBOE','TFC','USB','PNC',
    'MTB','CFG','RF','FITB','KEY','HBAN','ZION','CMA','FHN','WBS',
    'WAL','SIVB','SBNY','FRC','PACW','NYCB','ALLY','DFS','COF','SYF',
    'AXP','BAC','C','WFC','BK','STT','NTRS','SCHW','ETFC','AMTD',
    'TMUS','VZ','T','LUMN','FYBR','DISH','CHTR','CMCSA','FOX','FOXA',
    'NWSA','NWS','OMC','IPG','PARA','WBD','DIS','NFLX','EA','TTWO',
    'ATVI','ZM','UBER','LYFT','ABNB','DASH','SNAP','PINS','TWTR','SQ',
    'COIN','HOOD','SOFI','AFRM','UPST','OPEN','RDFN','ZG','CSGP','FICO',
    'VRSK','TRU','EFX','GPN','FISV','FLT','WEX','JKHY','SQ','GDDY',
    'ANSS','CDNS','SNPS','KLAC','LRCX','AMAT','ASML','MU','WDC','STX',
    'SWKS','QRVO','MCHP','ON','NXPI','TXN','ADI','XLNX','MRVL','MTCH',
    'IAC','ETSY','EBAY','CPNG','SE','MELI','BABA','JD','PDD','BIDU',
    'AMD','TSM','INTC','NVDA','QCOM','AVGO','BRCM','MU','LRCX','KLAC',
]
# Deduplicate
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def download_data(ticker, start='2011-01-01', end='2026-07-22'):
    """Download OHLCV data via yfinance, cache to disk."""
    cache_file = os.path.join(CACHE_DIR, f'{ticker}.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or len(df) < 252:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except Exception as e:
        return None


def compute_rsi(close, period=14):
    """Standard RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(period, min_periods=period).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(period, min_periods=period).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    return mfi


def compute_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def identify_bounces(df):
    """
    Identify bounce events: price drops >= BOUNCE_DROP_PCT then recovers
    >= BOUNCE_RECOV_PCT within BOUNCE_WINDOW days.
    Returns list of (bounce_date_idx, bounce_low_price).
    No lookahead: bounce is confirmed only AFTER recovery is observed.
    """
    close = df['Close'].values
    low = df['Low'].values
    n = len(close)
    bounces = []

    for i in range(1, n - BOUNCE_WINDOW):
        # Check if there was a drop of >= 2% ending at day i
        # Look back up to 20 days for the prior high
        lookback_start = max(0, i - 20)
        prior_high = np.max(close[lookback_start:i])

        if prior_high <= 0:
            continue

        drop_pct = (prior_high - low[i]) / prior_high
        if drop_pct < BOUNCE_DROP_PCT:
            continue

        # Check recovery within BOUNCE_WINDOW days AFTER day i
        recovery_end = min(i + BOUNCE_WINDOW, n)
        future_highs = close[i+1:recovery_end]
        if len(future_highs) == 0:
            continue

        recovery_pct = (np.max(future_highs) - low[i]) / (low[i] + 1e-10)
        if recovery_pct >= BOUNCE_RECOV_PCT:
            # Bounce confirmed on the day recovery is observed
            recovery_day = i + 1 + np.argmax(future_highs >= low[i] * (1 + BOUNCE_RECOV_PCT))
            if recovery_day < recovery_end:
                bounces.append((recovery_day, low[i]))

    return bounces


def count_zone_bounces(bounce_history, price, current_idx, dates, lookback=LOOKBACK):
    """
    Count how many bounces occurred in the zone around `price` within
    the trailing `lookback` days. Zone = price +/- ZONE_TOL%.
    Only counts bounces BEFORE current_idx.
    """
    zone_low = price * (1 - ZONE_TOL)
    zone_high = price * (1 + ZONE_TOL)
    cutoff_idx = max(0, current_idx - lookback)

    count = 0
    for b_idx, b_price in bounce_history:
        if b_idx < cutoff_idx or b_idx >= current_idx:
            continue
        if zone_low <= b_price <= zone_high:
            count += 1
    return count


def process_ticker(ticker, spy_data):
    """Process a single ticker: find all entry signals across variants."""
    df = download_data(ticker)
    if df is None or len(df) < 500:
        return None

    try:
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        dates = df.index

        # Compute indicators
        rsi = compute_rsi(close)
        mfi = compute_mfi(high, low, close, volume)
        atr = compute_atr(high, low, close)
        sma50 = close.rolling(50, min_periods=50).mean()
        vol_avg20 = volume.rolling(20, min_periods=20).mean()

        # ATR percentile (rolling 252d)
        atr_pctile = atr.rolling(252, min_periods=100).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
        )

        # Identify all bounces (no lookahead — confirmed after recovery)
        bounces = identify_bounces(df)

        if len(bounces) < 2:
            return None

        # Generate signals
        signals = []
        close_vals = close.values
        low_vals = low.values

        for i in range(LOOKBACK + 50, len(df)):
            current_price = close_vals[i]
            current_low = low_vals[i]

            # Count zone bounces for current price level
            zone_count = count_zone_bounces(bounces, current_low, i, dates)

            if zone_count < 2:
                continue

            # Collect features for variant filtering
            date = dates[i]
            sig = {
                'ticker': ticker,
                'date': date,
                'idx': i,
                'price': current_price,
                'zone_count': zone_count,
                'rsi': rsi.iloc[i] if i < len(rsi) else np.nan,
                'mfi': mfi.iloc[i] if i < len(mfi) else np.nan,
                'atr_pctile': atr_pctile.iloc[i] if i < len(atr_pctile) else np.nan,
                'vol_ratio': (volume.iloc[i] / vol_avg20.iloc[i]) if vol_avg20.iloc[i] > 0 else np.nan,
                'above_sma50': current_price > sma50.iloc[i] if not np.isnan(sma50.iloc[i]) else np.nan,
            }

            # Forward returns (no lookahead: we compute these for evaluation only)
            if i + 21 < len(df):
                sig['ret_10d'] = (close_vals[min(i+10, len(df)-1)] / current_price) - 1
                sig['ret_21d'] = (close_vals[min(i+21, len(df)-1)] / current_price) - 1
            else:
                sig['ret_10d'] = np.nan
                sig['ret_21d'] = np.nan

            # SPY return for regime classification
            spy_date = date
            if spy_data is not None and spy_date in spy_data.index:
                spy_idx = spy_data.index.get_loc(spy_date)
                spy_open = spy_data['Open'].iloc[spy_idx]
                spy_close_val = spy_data['Close'].iloc[spy_idx]
                sig['spy_green'] = spy_close_val > spy_open
            else:
                sig['spy_green'] = np.nan

            sig['year'] = date.year

            signals.append(sig)

        if not signals:
            return None

        return signals

    except Exception as e:
        return None


def evaluate_variant(signals_df, variant_name, hold_col, min_bounces=2,
                     rsi_filter=None, mfi_filter=None, vc_filter=None,
                     volume_filter=None, trend_filter=None, fresh_only=False):
    """Evaluate a single variant. Returns metrics dict or None."""
    df = signals_df.copy()

    # Apply zone count filter
    if fresh_only:
        df = df[df['zone_count'] == 2]  # exactly 2nd touch
    else:
        df = df[df['zone_count'] >= min_bounces]

    # Apply indicator filters
    if rsi_filter is not None:
        df = df[df['rsi'] < rsi_filter]
    if mfi_filter is not None:
        df = df[df['mfi'] < mfi_filter]
    if vc_filter is not None:
        df = df[df['atr_pctile'] < vc_filter]
    if volume_filter is not None:
        df = df[df['vol_ratio'] > volume_filter]
    if trend_filter == 'up':
        df = df[df['above_sma50'] == True]
    elif trend_filter == 'down':
        df = df[df['above_sma50'] == False]

    # Drop NaN returns
    df = df.dropna(subset=[hold_col])

    if len(df) < 30:
        return None

    returns = df[hold_col].values
    spy_green = df['spy_green'].values
    years = df['year'].values

    metrics = compute_metrics(returns, spy_green, years, variant_name)
    metrics['n_trades'] = len(df)
    metrics['variant'] = variant_name

    # Permutation test
    perm_sharpes = []
    actual_sharpe = metrics['sharpe']
    n = len(returns)
    for _ in range(PERM_SHUFFLES):
        # Shuffle entry dates (permute returns)
        perm_ret = np.random.permutation(returns)
        perm_sharpe = compute_sharpe(perm_ret)
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    metrics['perm_p_value'] = p_value

    return metrics


def compute_sharpe(returns):
    """Annualized Sharpe from trade returns."""
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    # Assume ~12 trades per year (21d hold, ~250 trading days)
    trades_per_year = 252 / 21  # approximate
    return np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)


def compute_sortino(returns):
    """Annualized Sortino."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 10.0 if np.mean(returns) > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return 0.0
    trades_per_year = 252 / 21
    return np.mean(returns) / downside_std * np.sqrt(trades_per_year)


def compute_metrics(returns, spy_green, years, variant_name):
    """Compute all metrics for a variant."""
    n = len(returns)
    if n < 10:
        return {'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'regime_gap': 1.0,
                'year_consistency': 0, 'n_trades': n}

    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    wr = np.mean(returns > 0)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    pf = (np.sum(winners) / (-np.sum(losers) + 1e-10)) if len(losers) > 0 else 10.0

    # Regime analysis
    green_mask = spy_green == True
    red_mask = spy_green == False

    green_returns = returns[green_mask]
    red_returns = returns[red_mask]

    sharpe_green = compute_sharpe(green_returns) if len(green_returns) > 10 else 0
    sharpe_red = compute_sharpe(red_returns) if len(red_returns) > 10 else 0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / (max_sharpe + 1e-10) if max_sharpe > 0.01 else 0

    # Year consistency
    unique_years = np.unique(years)
    years_profitable = 0
    for y in unique_years:
        mask = years == y
        if np.sum(mask) >= 5:
            if np.mean(returns[mask]) > 0:
                years_profitable += 1
    year_consistency = years_profitable / len(unique_years) if len(unique_years) > 0 else 0

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'wr': wr,
        'pf': pf,
        'avg_ret': np.mean(returns),
        'med_ret': np.median(returns),
        'max_dd_trade': np.min(returns),
        'best_trade': np.max(returns),
        'regime_gap': regime_gap,
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'n_green': int(np.sum(green_mask)),
        'n_red': int(np.sum(red_mask)),
        'year_consistency': year_consistency,
        'years_profitable': years_profitable,
        'years_total': len(unique_years),
    }


def print_results(results):
    """Print formatted results table with PASS/FAIL."""
    print('\n' + '=' * 120)
    print(f'{"SUPPLY/DEMAND ZONE MEMORY v1 — RESULTS":^120}')
    print('=' * 120)

    header = (f'{"Variant":<28} {"Trades":>6} {"Sharpe":>7} {"Sortino":>8} '
              f'{"WR":>6} {"PF":>6} {"AvgRet":>8} {"RGap":>6} {"YrCon":>6} '
              f'{"Perm-p":>7} {"Gates":>12}')
    print(header)
    print('-' * 120)

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        # Gate checks
        perm_pass = r['perm_p_value'] < 0.05
        regime_pass = r['regime_gap'] < 0.50
        year_pass = r['year_consistency'] > 0.70

        gates = []
        if perm_pass:
            gates.append('P')
        else:
            gates.append('p')
        if regime_pass:
            gates.append('R')
        else:
            gates.append('r')
        if year_pass:
            gates.append('Y')
        else:
            gates.append('y')

        all_pass = perm_pass and regime_pass and year_pass
        gate_str = ''.join(gates) + (' PASS' if all_pass else ' FAIL')

        print(f'{r["variant"]:<28} {r["n_trades"]:>6} {r["sharpe"]:>7.3f} {r["sortino"]:>8.3f} '
              f'{r["wr"]:>5.1%} {r["pf"]:>6.2f} {r["avg_ret"]:>7.3%} {r["regime_gap"]:>6.3f} '
              f'{r["year_consistency"]:>5.1%} {r["perm_p_value"]:>7.3f} {gate_str:>12}')

    print('-' * 120)
    print('Gates: P=permutation(p<0.05) R=regime(gap<0.50) Y=year(>70% profitable)')
    print('       UPPERCASE=pass  lowercase=fail')
    print()

    # Detail per variant
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        perm_pass = r['perm_p_value'] < 0.05
        regime_pass = r['regime_gap'] < 0.50
        year_pass = r['year_consistency'] > 0.70
        all_pass = perm_pass and regime_pass and year_pass

        print(f'\n--- {r["variant"]} {"*** PASS ***" if all_pass else "FAIL"} ---')
        print(f'  Trades: {r["n_trades"]}  |  Sharpe: {r["sharpe"]:.3f}  |  Sortino: {r["sortino"]:.3f}')
        print(f'  WR: {r["wr"]:.1%}  |  PF: {r["pf"]:.2f}  |  Avg Ret: {r["avg_ret"]:.3%}  |  Med Ret: {r["med_ret"]:.3%}')
        print(f'  Best trade: {r["best_trade"]:.2%}  |  Worst trade: {r["max_dd_trade"]:.2%}')
        print(f'  Regime — Green Sharpe: {r["sharpe_green"]:.3f} ({r["n_green"]} trades) | '
              f'Red Sharpe: {r["sharpe_red"]:.3f} ({r["n_red"]} trades) | Gap: {r["regime_gap"]:.3f} '
              f'{"PASS" if regime_pass else "FAIL"}')
        print(f'  Year consistency: {r["years_profitable"]}/{r["years_total"]} = {r["year_consistency"]:.0%} '
              f'{"PASS" if year_pass else "FAIL"}')
        print(f'  Permutation p-value: {r["perm_p_value"]:.4f} {"PASS" if perm_pass else "FAIL"}')


def main():
    start_time = time.time()
    log('Supply/Demand Zone Memory v1 — Starting')
    log(f'Universe: S&P 500 ({len(SP500_TICKERS)} tickers)')
    log(f'Workers: {WORKERS}, Lookback: {LOOKBACK}d, Permutations: {PERM_SHUFFLES}')

    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Download SPY for regime classification
    log('Downloading SPY data for regime classification...')
    spy_data = download_data(SPY_TICKER, start='2011-01-01', end='2026-07-22')
    if spy_data is None:
        log('ERROR: Failed to download SPY data')
        sys.exit(1)
    log(f'SPY data: {len(spy_data)} days ({spy_data.index[0].date()} to {spy_data.index[-1].date()})')

    # Download all tickers in parallel
    log(f'Downloading {len(SP500_TICKERS)} tickers (cached where possible)...')
    download_start = time.time()

    # Process tickers in parallel
    with Pool(WORKERS) as pool:
        func = partial(process_ticker, spy_data=spy_data)
        results_raw = pool.map(func, SP500_TICKERS)

    # Collect all signals
    all_signals = []
    tickers_with_signals = 0
    for result in results_raw:
        if result is not None:
            all_signals.extend(result)
            tickers_with_signals += 1

    download_elapsed = time.time() - download_start
    log(f'Data processing done in {download_elapsed:.0f}s')
    log(f'Tickers with signals: {tickers_with_signals}/{len(SP500_TICKERS)}')
    log(f'Total raw signals: {len(all_signals)}')

    if len(all_signals) < 100:
        log('ERROR: Too few signals generated. Check data availability.')
        sys.exit(1)

    # Convert to DataFrame
    signals_df = pd.DataFrame(all_signals)
    log(f'Signals DataFrame: {len(signals_df)} rows, date range: '
        f'{signals_df["date"].min().date()} to {signals_df["date"].max().date()}')

    # Zone count distribution
    log(f'Zone count distribution:')
    for cnt in sorted(signals_df['zone_count'].unique()):
        n = (signals_df['zone_count'] == cnt).sum()
        log(f'  {cnt} bounces: {n} signals')

    # Define variants
    variants = [
        ('zone2_long_21d', {'hold_col': 'ret_21d', 'min_bounces': 2}),
        ('zone3_long_21d', {'hold_col': 'ret_21d', 'min_bounces': 3}),
        ('zone2_vc_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'vc_filter': 20}),
        ('zone2_oversold_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'rsi_filter': 30}),
        ('zone2_mfi_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'mfi_filter': 30}),
        ('zone2_volume_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'volume_filter': 1.5}),
        ('zone2_long_10d', {'hold_col': 'ret_10d', 'min_bounces': 2}),
        ('zone_fresh_21d', {'hold_col': 'ret_21d', 'fresh_only': True}),
        ('zone2_trend_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'trend_filter': 'up'}),
        ('zone2_countertrend_21d', {'hold_col': 'ret_21d', 'min_bounces': 2, 'trend_filter': 'down'}),
    ]

    log(f'\nEvaluating {len(variants)} variants with {PERM_SHUFFLES} permutation shuffles each...')

    results = []
    for i, (name, params) in enumerate(variants):
        log(f'  [{i+1}/{len(variants)}] {name}...')
        try:
            m = evaluate_variant(signals_df, name, **params)
            if m is not None:
                results.append(m)
                log(f'    -> Sharpe={m["sharpe"]:.3f} Sortino={m["sortino"]:.3f} WR={m["wr"]:.1%} '
                    f'PF={m["pf"]:.2f} Trades={m["n_trades"]} RGap={m["regime_gap"]:.3f} '
                    f'Perm-p={m["perm_p_value"]:.3f}')
            else:
                log(f'    -> SKIPPED (insufficient trades)')
        except Exception as e:
            log(f'    -> ERROR: {e}')

    if not results:
        log('ERROR: No variants produced results')
        sys.exit(1)

    # Print results
    print_results(results)

    # Save results
    results_file = os.path.join(RESULTS_DIR, 'supply_demand_zones_v1_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f'\nResults saved to {results_file}')

    elapsed = time.time() - start_time
    log(f'Total runtime: {elapsed/60:.1f} minutes')

    # Summary
    passing = [r for r in results if r['perm_p_value'] < 0.05
               and r['regime_gap'] < 0.50 and r['year_consistency'] > 0.70]
    log(f'\n{"="*60}')
    log(f'SUMMARY: {len(passing)}/{len(results)} variants PASS all gates')
    if passing:
        best = max(passing, key=lambda x: x['sharpe'])
        log(f'Best passing variant: {best["variant"]} (Sharpe={best["sharpe"]:.3f})')
    log(f'{"="*60}')


if __name__ == '__main__':
    main()
