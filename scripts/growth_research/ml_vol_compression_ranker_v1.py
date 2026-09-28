#!/usr/bin/env python3
"""
ML Vol Compression Breakout Ranker v1
=====================================
CONTEXT: Vol compression breakout strategy has Sharpe 6.06 at 5d hold, 81% WR.
But NOT all breakouts are equal. Can ML predict WHICH breakouts will be strongest?

APPROACH:
  Phase 1 — Feature engineering: pre-breakout features, market context, cross-sectional
  Phase 2 — LightGBM (GPU-accelerated): regression + classification, walk-forward
  Phase 3 — Portfolio improvement: equal-weight vs ML-ranked top-50% vs top-25%

VALIDATION (HC #428):
  R1: Per-regime Sharpe (bull/bear), regime gap < 0.50
  Permutation test on ML ranking improvement (p < 0.05)

GPU: LightGBM device='gpu' on Neptune RTX 3090
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import json
import time

# ─── Output ───
OUTPUT_DIR = '/home/nick/Lvl3Quant/output/ml_vol_compression_ranker_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Configuration ───
START_DATE = '2014-01-01'
END_DATE = '2026-07-15'
VOL_LOOKBACK = 21         # 1-month realized vol
VOL_HISTORY = 252          # 1-year history for percentile
COMPRESSION_PCT = 10       # Bottom 10th percentile = compressed
BREAKOUT_MULT = 1.5        # Breakout = daily return > 1.5x average abs return
MAX_WAIT = 10              # Max days to wait for breakout
HOLD_PERIOD = 5            # 5d hold (best from v1)
TRAIN_YEARS = 7            # First 7 years for training
SMA200_WINDOW = 200
N_PERMS = 500              # Permutation tests

print(f"{'='*70}")
print(f"ML VOL COMPRESSION BREAKOUT RANKER v1")
print(f"GPU-accelerated LightGBM on Neptune")
print(f"{'='*70}")

# ─── GPU Check ───
try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
except ImportError:
    print("ERROR: lightgbm not installed"); sys.exit(1)

try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    pass

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 0: DATA DOWNLOAD")
print(f"{'='*70}")

import yfinance as yf

# Get S&P 500 tickers
print("\n[0.1] Getting S&P 500 universe...")
try:
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(sp500_url)
    sp500_df = tables[0]
    sp500_tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
    # Also grab GICS Sector for sector features
    ticker_sectors = dict(zip(
        sp500_df['Symbol'].str.replace('.', '-', regex=False),
        sp500_df['GICS Sector']
    ))
    print(f"  Got {len(sp500_tickers)} S&P 500 tickers from Wikipedia")
except Exception as e:
    print(f"  Wikipedia fetch failed: {e}")
    print("  Using hardcoded large list")
    sp500_tickers = [
        # Mega/Large Cap Tech
        'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','META','TSLA','AVGO','ORCL',
        'CRM','ADBE','AMD','CSCO','ACN','INTU','TXN','IBM','QCOM','AMAT',
        'ADI','LRCX','MU','KLAC','SNPS','CDNS','MRVL','FTNT','PANW','NOW',
        'PLTR','CRWD','NFLX','ABNB','UBER','SHOP','SQ','PYPL','WDAY','TTD',
        # Financials
        'BRK-B','JPM','V','MA','BAC','WFC','GS','MS','SCHW','BLK',
        'BX','KKR','AXP','PNC','USB','TFC','AIG','MMC','AON','CB',
        'PGR','AJG','AFL','MET','PRU','ICE','CME','SPGI','MCO','MSCI',
        'TROW','IVZ','BEN','FITB','HBAN','RF','KEY','CFG','ZION','CMA',
        'MTB','NDAQ','CBOE','WTW','RJF','AMP','GL','BRO','L','LNC',
        # Healthcare
        'UNH','JNJ','LLY','ABBV','MRK','PFE','TMO','ABT','DHR','BMY',
        'AMGN','GILD','VRTX','REGN','ISRG','SYK','BSX','MDT','ZBH','EW',
        'DXCM','ALGN','HOLX','IQV','MTD','BIO','A','WAT','CI','HCA',
        'ELV','HUM','CNC','MOH','DVA','UHS','BIIB','MRNA','ILMN',
        # Consumer
        'WMT','PG','KO','PEP','COST','MCD','NKE','SBUX','TGT','HD',
        'LOW','TJX','ROST','DG','DLTR','CMG','YUM','DPZ','MNST','KDP',
        'STZ','CL','CLX','KMB','CHD','EL','SJM','GIS','CPB','CAG',
        'HRL','HSY','MDLZ','KHC','MKC','K','TSN','HLT','MAR','LVS',
        'MGM','WYNN','RCL','CCL','NCLH','LULU','TPR','RL','BBWI','ETSY',
        'W','RH','WSM','ORLY','AZO','AAP','BBY','POOL','TSCO','WHR',
        # Industrials
        'CAT','BA','GE','HON','UNP','UPS','RTX','LMT','GD','NOC',
        'DE','MMM','EMR','ITW','ROK','PH','ETN','AME','DOV','SWK',
        'FTV','NDSN','XYL','IEX','FDX','ODFL','JBHT','CHRW','CSX','NSC',
        'PCAR','CMI','FAST','GWW','URI','WM','RSG','PWR','CARR','TT',
        'JCI','LII','GNRC','AOS','MAS','WAB','TDY','GRMN','TER','ZBRA',
        'CTAS','PAYX','VRSK','LHX','HWM','TXT','LDOS','BAH','AXON','CPRT',
        # Energy
        'XOM','CVX','COP','EOG','SLB','MPC','VLO','PSX','OXY','WMB',
        'KMI','HAL','DVN','FANG','HES','BKR','TRGP','OKE','CTRA','MRO',
        # Materials
        'LIN','APD','SHW','ECL','FCX','NEM','NUE','CF','MOS','VMC',
        'MLM','PPG','ALB','DD','DOW','CE','LYB','EMN','RPM','FMC',
        'IFF','CTVA','IP','PKG','AVY','WRK','OLN','WLK','HUN','AXTA',
        # Utilities
        'NEE','DUK','SO','D','AEP','SRE','XEL','WEC','ED','EXC',
        'ETR','FE','PPL','CNP','CMS','EVRG','AES','AWK','PNW','NI',
        # REITs
        'PLD','AMT','CCI','EQIX','PSA','SPG','O','DLR','WELL','VICI',
        'ARE','MAA','UDR','CPT','EXR','AVB','EQR','BXP','KIM','REG',
        # Telecom/Media
        'T','VZ','TMUS','CHTR','CMCSA','DIS','WBD','PARA','FOX','NWSA',
        'PM','MO','ATVI','EA','TTWO','ZG','MTCH','IAC',
        # Additional S&P 500
        'BDX','FI','ANET','FIS','FISV','GPN','IT','ANSS','FICO','PAYC',
        'CTSH','WEX','EPAM','DXC','AKAM','CDW','JNPR','HPQ','HPE','DELL',
        'F','GM','RIVN','APTV','BWA','LEA','VC','ALV','MCHP','SWKS',
        'MPWR','ON','ENPH','SEDG','FSLR','GEV','GEHC','SOLV',
    ]
    ticker_sectors = {}
    # Deduplicate
    sp500_tickers = list(dict.fromkeys(sp500_tickers))

# Always include SPY and VIX proxy for market context
for extra in ['SPY', '^VIX']:
    if extra not in sp500_tickers:
        sp500_tickers.append(extra)

tickers = sp500_tickers

# Download in batches
print(f"\n[0.2] Downloading price data for {len(tickers)} stocks...")
batch_size = 50
all_data = {}

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[0]}..{batch[-1]}")
    try:
        data = yf.download(batch, start=START_DATE, end=END_DATE,
                          group_by='ticker', progress=False, threads=True)
        for t in batch:
            try:
                if len(batch) == 1:
                    df = data[['Open','High','Low','Close','Volume']].dropna()
                else:
                    df = data[t][['Open','High','Low','Close','Volume']].dropna()
                if len(df) > VOL_HISTORY + VOL_LOOKBACK + HOLD_PERIOD + SMA200_WINDOW + 50:
                    all_data[t] = df
            except:
                pass
    except Exception as e:
        print(f"  Error in batch: {e}")
    time.sleep(0.3)

print(f"  Got data for {len(all_data)} stocks with sufficient history")
assert len(all_data) >= 200, f"Only {len(all_data)} stocks — need 200+. Aborting."
print(f"  {'PASS' if len(all_data) >= 400 else 'NOTE'}: {len(all_data)} stocks ({'>=400 ideal' if len(all_data) >= 400 else 'below 400 target but sufficient for analysis'})")

# Extract SPY and VIX for market context
spy_data = all_data.get('SPY')
vix_data = all_data.get('^VIX')
assert spy_data is not None, "SPY data required"

spy_close = spy_data['Close']
spy_sma200 = spy_close.rolling(SMA200_WINDOW).mean()
spy_ret_21d = spy_close.pct_change(21)  # 1-month SPY return

if vix_data is not None:
    vix_close = vix_data['Close']
else:
    # Approximate VIX from SPY realized vol
    vix_close = spy_close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    print("  WARNING: Using SPY realized vol as VIX proxy")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 1: FEATURE ENGINEERING")
print(f"{'='*70}")

# Sector encoding
unique_sectors = sorted(set(s for s in ticker_sectors.values() if s))
sector_to_idx = {s: i for i, s in enumerate(unique_sectors)}

# Market cap bucket approximation: use average volume * price as proxy
# (actual market cap would need fundamentals data)

# Pre-compute per-stock metrics
print("\n[1.1] Computing per-stock features...")
stock_features = {}

for ticker, df in all_data.items():
    if ticker in ['SPY', '^VIX']:
        continue

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    returns = close.pct_change()

    # Realized vol (21d)
    realized_vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    # Vol percentile within own history
    vol_pctile = realized_vol.rolling(VOL_HISTORY).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == VOL_HISTORY else np.nan,
        raw=False
    )

    # Average absolute return
    avg_abs_ret = returns.abs().rolling(VOL_LOOKBACK).mean()

    # Additional features
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi14 = _compute_rsi(close, 14) if False else None  # Will compute inline

    # Volume features
    avg_vol_20 = volume.rolling(20).mean()
    avg_vol_50 = volume.rolling(50).mean()

    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    stock_features[ticker] = {
        'close': close,
        'returns': returns,
        'realized_vol': realized_vol,
        'vol_pctile': vol_pctile,
        'avg_abs_ret': avg_abs_ret,
        'sma50': sma50,
        'sma200': sma200,
        'avg_vol_20': avg_vol_20,
        'avg_vol_50': avg_vol_50,
        'atr14': atr14,
        'volume': volume,
        'high': high,
        'low': low,
    }

# Compute RSI inline
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Cross-sectional: count how many stocks are in compression on each date
print("\n[1.2] Computing cross-sectional compression count...")
all_dates = spy_close.index
compression_count = pd.Series(0.0, index=all_dates)

for ticker, feats in stock_features.items():
    vp = feats['vol_pctile'].reindex(all_dates)
    compressed = (vp <= COMPRESSION_PCT).astype(float)
    compression_count = compression_count.add(compressed, fill_value=0)

print(f"  Cross-sectional compression count: mean={compression_count.mean():.1f}, max={compression_count.max():.0f}")

# ─── Build breakout events with features ───
print("\n[1.3] Building breakout events with ML features...")

events = []
t0 = time.time()
n_stocks_done = 0

for ticker, feats in stock_features.items():
    close = feats['close']
    returns = feats['returns']
    vol_pctile = feats['vol_pctile']
    avg_abs_ret = feats['avg_abs_ret']
    realized_vol = feats['realized_vol']
    sma50 = feats['sma50']
    sma200 = feats['sma200']
    avg_vol_20 = feats['avg_vol_20']
    avg_vol_50 = feats['avg_vol_50']
    atr14 = feats['atr14']
    volume = feats['volume']
    high_s = feats['high']
    low_s = feats['low']

    # RSI
    rsi = compute_rsi(close, 14)

    # Compression starts
    in_compression = (vol_pctile <= COMPRESSION_PCT).astype(int)
    compression_starts = in_compression.diff() == 1

    for start_date in compression_starts[compression_starts].index:
        start_pos = close.index.get_loc(start_date)

        # How long has compression lasted? (count consecutive days at low vol before this)
        compression_duration = 1
        for lookback in range(1, 60):
            check = start_pos - lookback
            if check < 0:
                break
            vp_val = vol_pctile.iloc[check] if check < len(vol_pctile) else np.nan
            if pd.notna(vp_val) and vp_val <= COMPRESSION_PCT * 2:  # Within 20th pctile
                compression_duration += 1
            else:
                break

        # Wait for breakout
        for wait in range(1, MAX_WAIT + 1):
            check_pos = start_pos + wait
            if check_pos >= len(close) - HOLD_PERIOD - 1:
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
                entry_date = close.index[check_pos]
                direction = 1 if daily_ret > 0 else -1
                entry_price = close.iloc[check_pos]

                # Forward return (target)
                exit_pos = check_pos + HOLD_PERIOD
                if exit_pos >= len(close):
                    break
                exit_price = close.iloc[exit_pos]
                fwd_ret = direction * (exit_price / entry_price - 1)
                fwd_ret_abs = (exit_price / entry_price - 1)  # Unsigned

                # ──── FEATURES ────

                # F1: Vol compression depth (lower = more compressed)
                f_vol_pctile = vol_pctile.iloc[start_pos] if start_pos < len(vol_pctile) else np.nan

                # F2: Compression duration (days)
                f_compression_duration = compression_duration

                # F3: Wait days until breakout
                f_wait_days = wait

                # F4: Breakout magnitude (how strong the breakout day was)
                f_breakout_mag = abs(daily_ret)

                # F5: Breakout direction
                f_breakout_dir = direction

                # F6: Breakout volume surge (breakout day volume / avg)
                f_vol_surge = 1.0
                if check_pos < len(volume) and start_pos < len(avg_vol_20):
                    av = avg_vol_20.iloc[start_pos]
                    if pd.notna(av) and av > 0:
                        f_vol_surge = volume.iloc[check_pos] / av

                # F7: Pre-breakout trend (50d return before compression)
                f_pre_trend_50d = np.nan
                if start_pos >= 50 and start_pos < len(close):
                    f_pre_trend_50d = close.iloc[start_pos] / close.iloc[start_pos - 50] - 1

                # F8: Pre-breakout trend (200d)
                f_pre_trend_200d = np.nan
                if start_pos >= 200 and start_pos < len(close):
                    f_pre_trend_200d = close.iloc[start_pos] / close.iloc[start_pos - 200] - 1

                # F9: Price relative to SMA50
                f_price_vs_sma50 = np.nan
                if start_pos < len(sma50) and pd.notna(sma50.iloc[start_pos]) and sma50.iloc[start_pos] > 0:
                    f_price_vs_sma50 = close.iloc[start_pos] / sma50.iloc[start_pos] - 1

                # F10: Price relative to SMA200
                f_price_vs_sma200 = np.nan
                if start_pos < len(sma200) and pd.notna(sma200.iloc[start_pos]) and sma200.iloc[start_pos] > 0:
                    f_price_vs_sma200 = close.iloc[start_pos] / sma200.iloc[start_pos] - 1

                # F11: RSI at breakout
                f_rsi = rsi.iloc[check_pos] if check_pos < len(rsi) else np.nan

                # F12: Volume trend (20d avg vs 50d avg)
                f_vol_trend = np.nan
                if start_pos < len(avg_vol_20) and start_pos < len(avg_vol_50):
                    av20 = avg_vol_20.iloc[start_pos]
                    av50 = avg_vol_50.iloc[start_pos]
                    if pd.notna(av50) and av50 > 0:
                        f_vol_trend = av20 / av50

                # F13: ATR normalized (ATR / price = % volatility)
                f_atr_norm = np.nan
                if start_pos < len(atr14) and entry_price > 0:
                    a = atr14.iloc[start_pos]
                    if pd.notna(a):
                        f_atr_norm = a / entry_price

                # F14: Realized vol level at compression
                f_realized_vol = realized_vol.iloc[start_pos] if start_pos < len(realized_vol) else np.nan

                # F15: Recent range compression (high-low range shrinkage)
                f_range_compression = np.nan
                if start_pos >= 10:
                    recent_range = (high_s.iloc[start_pos-5:start_pos] - low_s.iloc[start_pos-5:start_pos]).mean()
                    prior_range = (high_s.iloc[start_pos-20:start_pos-10] - low_s.iloc[start_pos-20:start_pos-10]).mean()
                    if prior_range > 0:
                        f_range_compression = recent_range / prior_range

                # F16: Sector (encoded)
                f_sector = sector_to_idx.get(ticker_sectors.get(ticker, ''), -1)

                # F17: Market cap proxy (avg dollar volume over 50d)
                f_mktcap_proxy = np.nan
                if start_pos < len(avg_vol_50) and start_pos < len(close):
                    av50 = avg_vol_50.iloc[start_pos]
                    if pd.notna(av50):
                        f_mktcap_proxy = np.log10(av50 * close.iloc[start_pos] + 1)

                # ──── MARKET CONTEXT FEATURES ────

                # F18: SPY trend (above/below 200SMA)
                f_spy_above_sma200 = 0
                if entry_date in spy_close.index and entry_date in spy_sma200.index:
                    sc = spy_close.loc[entry_date]
                    ss = spy_sma200.loc[entry_date]
                    if pd.notna(sc) and pd.notna(ss):
                        f_spy_above_sma200 = 1 if sc > ss else 0

                # F19: VIX level
                f_vix = np.nan
                if entry_date in vix_close.index:
                    f_vix = vix_close.loc[entry_date]

                # F20: SPY 21d return (market momentum)
                f_spy_mom = np.nan
                if entry_date in spy_ret_21d.index:
                    f_spy_mom = spy_ret_21d.loc[entry_date]

                # F21: Cross-sectional compression count (crowding)
                f_compression_crowd = 0
                if entry_date in compression_count.index:
                    f_compression_crowd = compression_count.loc[entry_date]

                # F22: Day of week
                f_dow = entry_date.dayofweek

                # F23: Month
                f_month = entry_date.month

                # F24: Distance from 52w high
                f_dist_52w_high = np.nan
                if start_pos >= 252:
                    high_52w = close.iloc[start_pos-252:start_pos].max()
                    if high_52w > 0:
                        f_dist_52w_high = close.iloc[start_pos] / high_52w - 1

                # F25: Distance from 52w low
                f_dist_52w_low = np.nan
                if start_pos >= 252:
                    low_52w = close.iloc[start_pos-252:start_pos].min()
                    if low_52w > 0:
                        f_dist_52w_low = close.iloc[start_pos] / low_52w - 1

                # SPY return over same hold period (for regime classification)
                spy_ret = np.nan
                if entry_date in spy_close.index:
                    spy_entry_pos = spy_close.index.get_loc(entry_date)
                    spy_exit_pos = spy_entry_pos + HOLD_PERIOD
                    if spy_exit_pos < len(spy_close):
                        spy_ret = spy_close.iloc[spy_exit_pos] / spy_close.iloc[spy_entry_pos] - 1

                events.append({
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'direction': direction,
                    'fwd_ret': fwd_ret,          # Directional return (target for regression)
                    'fwd_ret_abs': fwd_ret_abs,  # Unsigned return
                    'spy_ret': spy_ret,
                    # Features
                    'f_vol_pctile': f_vol_pctile,
                    'f_compression_duration': f_compression_duration,
                    'f_wait_days': f_wait_days,
                    'f_breakout_mag': f_breakout_mag,
                    'f_breakout_dir': f_breakout_dir,
                    'f_vol_surge': f_vol_surge,
                    'f_pre_trend_50d': f_pre_trend_50d,
                    'f_pre_trend_200d': f_pre_trend_200d,
                    'f_price_vs_sma50': f_price_vs_sma50,
                    'f_price_vs_sma200': f_price_vs_sma200,
                    'f_rsi': f_rsi,
                    'f_vol_trend': f_vol_trend,
                    'f_atr_norm': f_atr_norm,
                    'f_realized_vol': f_realized_vol,
                    'f_range_compression': f_range_compression,
                    'f_sector': f_sector,
                    'f_mktcap_proxy': f_mktcap_proxy,
                    'f_spy_above_sma200': f_spy_above_sma200,
                    'f_vix': f_vix,
                    'f_spy_mom': f_spy_mom,
                    'f_compression_crowd': f_compression_crowd,
                    'f_dow': f_dow,
                    'f_month': f_month,
                    'f_dist_52w_high': f_dist_52w_high,
                    'f_dist_52w_low': f_dist_52w_low,
                })
                break  # Only first breakout per compression event

    n_stocks_done += 1
    if n_stocks_done % 50 == 0:
        elapsed = time.time() - t0
        rate = n_stocks_done / elapsed
        remaining = (len(stock_features) - n_stocks_done) / rate
        print(f"  Processed {n_stocks_done}/{len(stock_features)} stocks "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

events_df = pd.DataFrame(events)
events_df['entry_date'] = pd.to_datetime(events_df['entry_date'])
events_df = events_df.sort_values('entry_date').reset_index(drop=True)

print(f"\n  Total breakout events: {len(events_df)}")
print(f"  Unique stocks: {events_df['ticker'].nunique()}")
print(f"  Date range: {events_df['entry_date'].min().date()} to {events_df['entry_date'].max().date()}")
print(f"  Mean fwd return: {events_df['fwd_ret'].mean()*100:.2f}%")
print(f"  Win rate (directional): {(events_df['fwd_ret'] > 0).mean():.1%}")

# Feature columns
FEATURE_COLS = [c for c in events_df.columns if c.startswith('f_')]
print(f"  Features: {len(FEATURE_COLS)}")
for f in FEATURE_COLS:
    non_null = events_df[f].notna().mean()
    print(f"    {f}: {non_null:.1%} non-null, mean={events_df[f].mean():.4f}" if non_null > 0 else f"    {f}: ALL NULL")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: ML MODEL (LightGBM GPU)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 2: ML MODEL — LightGBM GPU-Accelerated")
print(f"{'='*70}")

# Walk-forward split: train on first TRAIN_YEARS years, validate on rest
split_date = events_df['entry_date'].min() + pd.DateOffset(years=TRAIN_YEARS)
print(f"\n  Train/val split date: {split_date.date()}")

train_df = events_df[events_df['entry_date'] < split_date].copy()
val_df = events_df[events_df['entry_date'] >= split_date].copy()

print(f"  Train: {len(train_df)} events ({train_df['entry_date'].min().date()} to {train_df['entry_date'].max().date()})")
print(f"  Val:   {len(val_df)} events ({val_df['entry_date'].min().date()} to {val_df['entry_date'].max().date()})")

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
X_val = val_df[FEATURE_COLS].values.astype(np.float32)

# ─── Model A: Regression (predict return magnitude) ───
print("\n[2.1] Training REGRESSION model (predict forward return)...")

y_train_reg = train_df['fwd_ret'].values.astype(np.float32)
y_val_reg = val_df['fwd_ret'].values.astype(np.float32)

lgb_params_reg = {
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu',
    'gpu_use_dp': False,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

dtrain_reg = lgb.Dataset(X_train, label=y_train_reg, feature_name=FEATURE_COLS)
dval_reg = lgb.Dataset(X_val, label=y_val_reg, feature_name=FEATURE_COLS, reference=dtrain_reg)

callbacks = [
    lgb.log_evaluation(period=100),
    lgb.early_stopping(stopping_rounds=50),
]

t_model = time.time()
model_reg = lgb.train(
    lgb_params_reg,
    dtrain_reg,
    num_boost_round=1000,
    valid_sets=[dval_reg],
    valid_names=['val'],
    callbacks=callbacks,
)
print(f"  Regression model trained in {time.time()-t_model:.1f}s, best iter={model_reg.best_iteration}")

# Predictions
val_df['pred_return'] = model_reg.predict(X_val)

# Regression performance
from scipy import stats
corr, p_corr = stats.spearmanr(val_df['pred_return'], val_df['fwd_ret'])
print(f"  Spearman rank correlation (pred vs actual): {corr:.4f} (p={p_corr:.4e})")

ic = np.corrcoef(val_df['pred_return'], val_df['fwd_ret'])[0, 1]
print(f"  IC (Pearson): {ic:.4f}")

# ─── Model B: Classification (predict above-median return) ───
print("\n[2.2] Training CLASSIFICATION model (predict top-half returns)...")

# Use expanding median from training set to avoid lookahead
train_median = train_df['fwd_ret'].median()
y_train_cls = (train_df['fwd_ret'] > train_median).astype(int).values
y_val_cls = (val_df['fwd_ret'] > train_median).astype(int).values

lgb_params_cls = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'gpu_use_dp': False,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

dtrain_cls = lgb.Dataset(X_train, label=y_train_cls, feature_name=FEATURE_COLS)
dval_cls = lgb.Dataset(X_val, label=y_val_cls, feature_name=FEATURE_COLS, reference=dtrain_cls)

t_model = time.time()
model_cls = lgb.train(
    lgb_params_cls,
    dtrain_cls,
    num_boost_round=1000,
    valid_sets=[dval_cls],
    valid_names=['val'],
    callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
)
print(f"  Classification model trained in {time.time()-t_model:.1f}s, best iter={model_cls.best_iteration}")

val_df['pred_prob'] = model_cls.predict(X_val)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_val_cls, val_df['pred_prob'])
print(f"  Validation AUC: {auc:.4f}")

# ─── Feature Importance ───
print("\n[2.3] Feature importance (top 15)...")
imp_reg = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_reg': model_reg.feature_importance(importance_type='gain'),
    'importance_cls': model_cls.feature_importance(importance_type='gain'),
}).sort_values('importance_reg', ascending=False)

print("\n  REGRESSION (gain):")
for _, row in imp_reg.head(15).iterrows():
    print(f"    {row['feature']:30s}  reg={row['importance_reg']:10.1f}  cls={row['importance_cls']:10.1f}")

imp_reg.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

# Save models
model_reg.save_model(f"{OUTPUT_DIR}/model_regression.txt")
model_cls.save_model(f"{OUTPUT_DIR}/model_classification.txt")
print(f"  Models saved to {OUTPUT_DIR}/")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: PORTFOLIO IMPROVEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 3: PORTFOLIO IMPROVEMENT — ML RANKING vs EQUAL WEIGHT")
print(f"{'='*70}")

def compute_portfolio_metrics(trades, label):
    """Compute risk-adjusted metrics for a set of trades."""
    if len(trades) == 0:
        return None

    # Group by entry date for daily returns
    daily = trades.groupby('entry_date')['fwd_ret'].mean()

    if len(daily) < 10:
        return None

    mean_ret = daily.mean()
    std_ret = daily.std()
    sharpe = mean_ret / std_ret * np.sqrt(252 / HOLD_PERIOD) if std_ret > 0 else np.nan

    downside = daily[daily < 0].std()
    sortino = mean_ret / downside * np.sqrt(252 / HOLD_PERIOD) if downside > 0 else np.nan

    win_rate = (daily > 0).mean()

    gross_profit = daily[daily > 0].sum()
    gross_loss = abs(daily[daily < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min()

    # Regime analysis
    trades_with_regime = trades.copy()
    trades_with_regime['regime'] = trades_with_regime['spy_ret'].apply(
        lambda x: 'BULL' if pd.notna(x) and x > 0.01 else ('BEAR' if pd.notna(x) and x < -0.01 else 'FLAT')
    )

    regime_sharpes = {}
    for regime in ['BULL', 'BEAR', 'FLAT']:
        rt = trades_with_regime[trades_with_regime['regime'] == regime]
        if len(rt) > 10:
            rd = rt.groupby('entry_date')['fwd_ret'].mean()
            if len(rd) > 5 and rd.std() > 0:
                regime_sharpes[regime] = rd.mean() / rd.std() * np.sqrt(252 / HOLD_PERIOD)

    # Regime gap
    s_bull = regime_sharpes.get('BULL', 0)
    s_bear = regime_sharpes.get('BEAR', 0)
    max_s = max(abs(s_bull), abs(s_bear)) if max(abs(s_bull), abs(s_bear)) > 0 else 1
    regime_gap = abs(s_bull - s_bear) / max_s

    return {
        'label': label,
        'n_trades': len(trades),
        'n_days': len(daily),
        'mean_daily_ret': mean_ret,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'profit_factor': pf,
        'max_dd': max_dd,
        'regime_sharpes': regime_sharpes,
        'regime_gap': regime_gap,
    }

# Strategy variants on validation set
print("\n[3.1] Computing portfolio variants on validation set...")

# Sort each day's breakouts by ML prediction, then take top-N
val_df_sorted = val_df.copy()

results = []

# A) Equal weight all breakouts
r = compute_portfolio_metrics(val_df_sorted, "ALL (equal weight)")
if r:
    results.append(r)
    print(f"\n  {r['label']}:")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")
    print(f"    Regime: {', '.join(f'{k}={v:.2f}' for k,v in r['regime_sharpes'].items())}, gap={r['regime_gap']:.2f}")

# B) ML-ranked top 50% (regression)
val_df_sorted['rank_pct'] = val_df_sorted.groupby('entry_date')['pred_return'].rank(pct=True)
top50 = val_df_sorted[val_df_sorted['rank_pct'] >= 0.50]
r = compute_portfolio_metrics(top50, "ML Top 50% (regression)")
if r:
    results.append(r)
    print(f"\n  {r['label']}:")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")
    print(f"    Regime: {', '.join(f'{k}={v:.2f}' for k,v in r['regime_sharpes'].items())}, gap={r['regime_gap']:.2f}")

# C) ML-ranked top 25% (regression)
top25 = val_df_sorted[val_df_sorted['rank_pct'] >= 0.75]
r = compute_portfolio_metrics(top25, "ML Top 25% (regression)")
if r:
    results.append(r)
    print(f"\n  {r['label']}:")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")
    print(f"    Regime: {', '.join(f'{k}={v:.2f}' for k,v in r['regime_sharpes'].items())}, gap={r['regime_gap']:.2f}")

# D) ML-ranked top 10% (regression — concentrated bets)
top10 = val_df_sorted[val_df_sorted['rank_pct'] >= 0.90]
r = compute_portfolio_metrics(top10, "ML Top 10% (regression)")
if r:
    results.append(r)
    print(f"\n  {r['label']}:")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")
    print(f"    Regime: {', '.join(f'{k}={v:.2f}' for k,v in r['regime_sharpes'].items())}, gap={r['regime_gap']:.2f}")

# E) Classification-based: top 50% by predicted probability
val_df_sorted['rank_pct_cls'] = val_df_sorted.groupby('entry_date')['pred_prob'].rank(pct=True)
top50_cls = val_df_sorted[val_df_sorted['rank_pct_cls'] >= 0.50]
r = compute_portfolio_metrics(top50_cls, "ML Top 50% (classification)")
if r:
    results.append(r)
    print(f"\n  {r['label']}:")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")
    print(f"    Regime: {', '.join(f'{k}={v:.2f}' for k,v in r['regime_sharpes'].items())}, gap={r['regime_gap']:.2f}")

# F) Bottom 25% (should be WORSE if ML works)
bot25 = val_df_sorted[val_df_sorted['rank_pct'] < 0.25]
r = compute_portfolio_metrics(bot25, "ML Bottom 25% (regression)")
if r:
    results.append(r)
    print(f"\n  {r['label']} (sanity check — should be worst):")
    print(f"    N={r['n_trades']}, Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
          f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, MaxDD={r['max_dd']*100:.1f}%")

# ─── Monotonicity check: quintile analysis ───
print("\n[3.2] Quintile analysis (does ML ranking correlate with returns?)...")

val_df_sorted['quintile'] = pd.qcut(val_df_sorted['pred_return'], 5, labels=['Q1(worst)','Q2','Q3','Q4','Q5(best)'])
quintile_stats = val_df_sorted.groupby('quintile')['fwd_ret'].agg(['mean','median','std','count'])
quintile_wr = val_df_sorted.groupby('quintile').apply(lambda x: (x['fwd_ret'] > 0).mean())
quintile_stats['win_rate'] = quintile_wr

print("\n  Quintile | Mean Ret | Median Ret | WR    | Count")
print("  " + "-"*55)
for q in ['Q1(worst)','Q2','Q3','Q4','Q5(best)']:
    if q in quintile_stats.index:
        row = quintile_stats.loc[q]
        print(f"  {q:10s} | {row['mean']*100:+7.2f}% | {row['median']*100:+8.2f}% | {row['win_rate']:.1%} | {int(row['count'])}")

# Check monotonicity
q_means = [quintile_stats.loc[q, 'mean'] for q in ['Q1(worst)','Q2','Q3','Q4','Q5(best)'] if q in quintile_stats.index]
is_monotonic = all(q_means[i] <= q_means[i+1] for i in range(len(q_means)-1))
spread = q_means[-1] - q_means[0] if len(q_means) >= 2 else 0
print(f"\n  Q5-Q1 spread: {spread*100:.2f}%")
print(f"  Monotonic: {'YES' if is_monotonic else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: PERMUTATION TEST ON ML RANKING IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 4: PERMUTATION TEST — Is ML ranking better than random?")
print(f"{'='*70}")

# Observed: Sharpe of top-25% ML-ranked
top25_trades = val_df_sorted[val_df_sorted['rank_pct'] >= 0.75]
observed_metrics = compute_portfolio_metrics(top25_trades, "observed")
if observed_metrics is None:
    print("  ERROR: Not enough top-25% trades for permutation test")
else:
    observed_sharpe = observed_metrics['sharpe']
    print(f"\n  Observed top-25% Sharpe: {observed_sharpe:.3f}")

    # Permutation: shuffle ML rankings, take random top-25%
    print(f"  Running {N_PERMS} permutations...")
    perm_sharpes = []

    for perm_i in range(N_PERMS):
        # Shuffle predictions within each day
        perm_df = val_df_sorted.copy()
        perm_df['pred_return_shuffled'] = perm_df.groupby('entry_date')['pred_return'].transform(
            lambda x: x.sample(frac=1.0).values
        )
        perm_df['rank_pct_perm'] = perm_df.groupby('entry_date')['pred_return_shuffled'].rank(pct=True)
        perm_top25 = perm_df[perm_df['rank_pct_perm'] >= 0.75]

        r = compute_portfolio_metrics(perm_top25, "perm")
        if r is not None and not np.isnan(r['sharpe']):
            perm_sharpes.append(r['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).mean()

    print(f"\n  Permutation test results:")
    print(f"    Observed top-25% Sharpe: {observed_sharpe:.3f}")
    print(f"    Null mean Sharpe:        {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
    print(f"    Null median Sharpe:      {np.median(perm_sharpes):.3f}")
    print(f"    p-value:                 {p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'} < 0.05)")
    print(f"    Improvement:             {((observed_sharpe / perm_sharpes.mean()) - 1)*100:.1f}% over random" if perm_sharpes.mean() != 0 else "")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: REGIME VALIDATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PHASE 5: REGIME VALIDATION (HC #428)")
print(f"{'='*70}")

# Per-year analysis
print("\n[5.1] Per-year performance of ML top-25% vs equal-weight...")
val_df_sorted['year'] = val_df_sorted['entry_date'].dt.year

print("\n  Year  | ALL Sharpe | Top25 Sharpe | Improvement | N_all | N_top25")
print("  " + "-"*70)

for year in sorted(val_df_sorted['year'].unique()):
    year_all = val_df_sorted[val_df_sorted['year'] == year]
    year_top25 = year_all[year_all['rank_pct'] >= 0.75]

    r_all = compute_portfolio_metrics(year_all, f"all_{year}")
    r_top = compute_portfolio_metrics(year_top25, f"top25_{year}")

    if r_all and r_top:
        impr = r_top['sharpe'] - r_all['sharpe']
        print(f"  {year}  | {r_all['sharpe']:+10.2f} | {r_top['sharpe']:+12.2f} | {impr:+11.2f} | {r_all['n_trades']:5d} | {r_top['n_trades']:7d}")

# Regime-specific validation
print("\n[5.2] Regime-specific validation (best ML variant)...")
best_variant = None
best_sharpe_val = -999
for r in results:
    if r['sharpe'] > best_sharpe_val and 'Bottom' not in r['label']:
        best_sharpe_val = r['sharpe']
        best_variant = r

if best_variant:
    print(f"\n  Best variant: {best_variant['label']}")
    print(f"  Overall Sharpe: {best_variant['sharpe']:.2f}")
    print(f"  Regime Sharpes: {best_variant['regime_sharpes']}")
    print(f"  Regime gap: {best_variant['regime_gap']:.2f} ({'PASS' if best_variant['regime_gap'] < 0.50 else 'FAIL'} < 0.50)")

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n  Universe: {len(all_data)} stocks, {events_df['entry_date'].min().date()} to {events_df['entry_date'].max().date()}")
print(f"  Total breakout events: {len(events_df)}")
print(f"  Train events: {len(train_df)}, Val events: {len(val_df)}")
print(f"  Hold period: {HOLD_PERIOD}d")

print(f"\n  REGRESSION MODEL:")
print(f"    Spearman correlation: {corr:.4f} (p={p_corr:.2e})")
print(f"    IC (Pearson): {ic:.4f}")

print(f"\n  CLASSIFICATION MODEL:")
print(f"    AUC: {auc:.4f}")

print(f"\n  PORTFOLIO COMPARISON (validation period):")
print(f"  {'Label':35s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'WR':>5s} | {'PF':>5s} | {'MaxDD':>6s} | {'Regime Gap':>10s}")
print(f"  {'-'*90}")
for r in results:
    rg = r.get('regime_gap', np.nan)
    print(f"  {r['label']:35s} | {r['sharpe']:7.2f} | {r['sortino']:7.2f} | {r['win_rate']:5.1%} | {r['profit_factor']:5.2f} | {r['max_dd']*100:5.1f}% | {rg:10.2f}")

if observed_metrics:
    print(f"\n  ML RANKING PERMUTATION TEST:")
    print(f"    p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})")
    print(f"    Top-25% Sharpe: {observed_sharpe:.3f} vs random {perm_sharpes.mean():.3f}")

# Validation gates
print(f"\n  VALIDATION GATES (HC #428):")
if best_variant:
    rg = best_variant['regime_gap']
    print(f"    R1 Regime gap: {rg:.2f} ({'PASS' if rg < 0.50 else 'FAIL'} < 0.50)")
if observed_metrics:
    print(f"    Permutation test: p={p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'} < 0.05)")

# Save everything
summary = {
    'strategy': 'ml_vol_compression_ranker_v1',
    'run_date': datetime.now().isoformat(),
    'universe_size': len(all_data),
    'total_events': len(events_df),
    'train_events': len(train_df),
    'val_events': len(val_df),
    'hold_period': HOLD_PERIOD,
    'split_date': str(split_date.date()),
    'regression_spearman': float(corr),
    'regression_ic': float(ic),
    'classification_auc': float(auc),
    'permutation_p_value': float(p_value) if observed_metrics else None,
    'portfolio_variants': [{k: v for k, v in r.items() if k != 'regime_sharpes'} for r in results],
    'quintile_means': [float(q) for q in q_means],
    'quintile_spread': float(spread),
    'is_monotonic': bool(is_monotonic),
    'feature_importance_top10': imp_reg.head(10)[['feature','importance_reg','importance_cls']].to_dict('records'),
}

# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=convert_numpy)

events_df.to_csv(f"{OUTPUT_DIR}/all_events_with_features.csv", index=False)
val_df_sorted.to_csv(f"{OUTPUT_DIR}/val_predictions.csv", index=False)

print(f"\n  Results saved to {OUTPUT_DIR}/")
print(f"  Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
